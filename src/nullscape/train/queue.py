"""Sequential experiment queue: preflight every recipe, evaluate the baseline, then train, evaluate on VAL and
compare each experiment in turn. Every stage runs as its own `nullscape` process so GPU memory is released between
stages. State is saved after every transition; rerunning the same command continues where the queue stopped.
The queue never selects a winner, promotes a checkpoint, or touches the TEST split."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from nullscape.inference.sampler import DEFAULT_SAMPLING, sampling_config
from nullscape.utils.config import dump_config, load_config
from nullscape.utils.paths import REPO_ROOT, resolve_checkpoint, unique_directory
from nullscape.utils.tracking import atomic_file, atomic_json

Runner = Callable[[list[str], Path, Callable[[str], None]], int]
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EVALUATION_DEFAULTS = {"split": "val", "n": 500, "seed": 7, "batch_size": 64, "modes": "conditional",
                       "gameplay_n": 128, "memorization": True, "device": None, "gpu_memory_fraction": 0.6,
                       "sampling": {}}
TARGETS = ("mountains", "ridges", "plains")


class QueueFailed(RuntimeError):
    pass


class QueuePaused(KeyboardInterrupt):
    pass


def _existing(path: str) -> Path:
    p = Path(path).expanduser()
    for candidate in (p, REPO_ROOT / p):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"file not found: {path}")


def load_plan(path: str | Path) -> dict:
    raw = load_config(path)
    unknown = set(raw) - {"name", "baseline", "evaluation", "experiments", "checkpoint"}
    if unknown:
        raise ValueError(f"unknown queue plan keys: {', '.join(sorted(unknown))}")
    name = str(raw.get("name") or Path(path).stem)
    baseline = raw.get("baseline") or {}
    if set(baseline) != {"name", "checkpoint"}:
        raise ValueError("baseline needs exactly name and checkpoint")
    evaluation = {**EVALUATION_DEFAULTS, **(raw.get("evaluation") or {})}
    unknown = set(evaluation) - set(EVALUATION_DEFAULTS) - {"dataset"}
    if unknown:
        raise ValueError(f"unknown evaluation keys: {', '.join(sorted(unknown))}")
    if evaluation.get("dataset") is None:  # default: each checkpoint's training dataset; kept out of older plans
        evaluation.pop("dataset", None)
    elif not isinstance(evaluation["dataset"], str) or not evaluation["dataset"]:
        raise ValueError("evaluation.dataset must be a dataset name or folder")
    if evaluation["split"] != "val":
        raise ValueError("the queue screens on VAL only; score TEST separately after choosing a checkpoint")
    modes = [m.strip() for m in str(evaluation["modes"]).split(",") if m.strip()]
    if "conditional" not in modes:
        raise ValueError("evaluation.modes must include conditional (the comparison page needs matched requests)")
    for key, minimum in (("n", 2), ("batch_size", 1), ("gameplay_n", 0), ("seed", 0)):
        if not isinstance(evaluation[key], int) or evaluation[key] < minimum:
            raise ValueError(f"evaluation.{key} must be an integer >= {minimum}")
    if evaluation["device"] not in (None, "cpu", "cuda"):
        raise ValueError("evaluation.device must be cpu, cuda or omitted")
    if not 0 < float(evaluation["gpu_memory_fraction"]) <= 1:
        raise ValueError("evaluation.gpu_memory_fraction must be in (0,1]")
    sampling = sampling_config(**{**DEFAULT_SAMPLING, **evaluation["sampling"]})
    evaluation.update(modes=",".join(modes), memorization=bool(evaluation["memorization"]),
                      gpu_memory_fraction=float(evaluation["gpu_memory_fraction"]),
                      sampling={**sampling, "guidance_interval": list(sampling["guidance_interval"])})
    checkpoint = raw.get("checkpoint", "last")
    if checkpoint not in ("last", "best"):
        raise ValueError("checkpoint must be last (final weights, default) or best (in-training VAL selection)")
    experiments = raw.get("experiments") or []
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("the plan needs at least one experiment")
    seen, cleaned = set(), []
    for item in experiments:
        if set(item) - {"name", "config", "set"} or not {"name", "config"} <= set(item):
            raise ValueError("each experiment needs name and config (optional: set)")
        ename = str(item["name"])
        if not NAME.match(ename) or ename in seen:
            raise ValueError(f"experiment names must be unique letters, digits, '.', '_' or '-': {ename!r}")
        seen.add(ename)
        overrides = [str(s) for s in item.get("set", [])]
        cleaned.append({"name": ename, "config": str(_existing(item["config"])), "set": overrides})
    return {"name": name, "baseline": {"name": str(baseline["name"]),
                                       "checkpoint": str(resolve_checkpoint(baseline["checkpoint"]))},
            "evaluation": evaluation, "checkpoint": checkpoint, "experiments": cleaned}


def stage_list(plan: dict) -> list[tuple[str, str, dict | None]]:
    stages = [(f"preflight-{e['name']}", "preflight", e) for e in plan["experiments"]]
    stages.append(("baseline-eval", "baseline", None))
    for e in plan["experiments"]:
        stages += [(f"train-{e['name']}", "train", e), (f"eval-{e['name']}", "eval", e),
                   (f"compare-{e['name']}", "compare", e)]
    return stages


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError(f"another queue is already using {path.parent}") from None
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


class _PauseRequest:
    def __init__(self):
        self.requested = False
        self._previous = {}

    def _handle(self, signum, frame):
        if not self.requested:
            print("\nPause requested. The current stage is stopping safely; rerun the same command to continue.",
                  flush=True)
        self.requested = True

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in {signal.SIGINT, getattr(signal, "SIGBREAK", signal.SIGINT)}:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def subprocess_runner(args: list[str], log: Path, on_line: Callable[[str], None]) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log, "a", encoding="utf-8") as file:
        file.write(f"$ nullscape {subprocess.list2cmdline(args)}\n")
        file.flush()
        proc = subprocess.Popen([sys.executable, "-m", "nullscape.cli", *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            file.write(line)
            file.flush()
            print(line, end="", flush=True)
            on_line(line.rstrip("\r\n"))
        return proc.wait()


def _checkpoint_step(path: Path) -> tuple[int, bool]:
    from nullscape.utils.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(path, mmap=True)
    return int(checkpoint["step"]), checkpoint.get("training_state") is not None


def _tail(path: Path, lines: int = 25) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) if path.is_file() else ""


class Queue:
    def __init__(self, plan_path: str | Path, out: str | Path | None, runner: Runner | None):
        self.plan = load_plan(plan_path)
        self.out = Path(out) if out else Path("reports") / "queues" / self.plan["name"]
        self.runner = runner or subprocess_runner
        self.state_path = self.out / "queue_state.json"

    def open(self) -> dict:
        normalized = json.loads(json.dumps(self.plan))
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") != 1 or state.get("plan") != normalized:
                raise ValueError(f"plan changed since {self.out} started; use a new --out folder for a new queue")
            return state
        if self.out.exists() and any(self.out.iterdir()):
            raise ValueError(f"{self.out} is not a queue folder; choose an empty or new --out")
        (self.out / "configs").mkdir(parents=True, exist_ok=True)
        for sub in ("logs", "evaluations", "comparisons"):
            (self.out / sub).mkdir(exist_ok=True)
        for e in self.plan["experiments"]:
            dump_config(load_config(e["config"], e["set"]), self.out / "configs" / f"{e['name']}.yaml")
        return {"version": 1, "plan": normalized, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": {}}

    def save(self, state: dict) -> None:
        atomic_json(self.state_path, state)

    def snapshot(self, experiment: dict) -> Path:
        path = self.out / "configs" / f"{experiment['name']}.yaml"
        live = load_config(experiment["config"], experiment["set"])
        if live != load_config(path):
            print(f"Note: {experiment['config']} changed after this queue started; using the snapshot {path}", flush=True)
        return path

    def eval_args(self, checkpoint: str, out: Path) -> list[str]:
        ev, s = self.plan["evaluation"], self.plan["evaluation"]["sampling"]
        args = ["evaluate", "--checkpoint", checkpoint, "--split", "val", "--n", str(ev["n"]), "--seed", str(ev["seed"]),
                "--batch-size", str(ev["batch_size"]), "--modes", ev["modes"], "--gameplay-n", str(ev["gameplay_n"]),
                "--gpu-memory-fraction", str(ev["gpu_memory_fraction"]), "--steps", str(s["steps"]),
                "--guidance", str(s["guidance"]), "--eta", str(s["eta"]), "--spacing", s["spacing"],
                "--guidance-interval", *(str(v) for v in s["guidance_interval"]), "--out", str(out)]
        if ev["device"]:
            args += ["--device", ev["device"]]
        if ev.get("dataset"):
            args += ["--dataset", ev["dataset"]]
        if not ev["memorization"]:
            args.append("--no-memorization")
        return args

    def run_stage(self, key: str, kind: str, experiment: dict | None, stage: dict, state: dict) -> str:
        attempt = stage["attempts"]
        log = self.out / "logs" / f"{key}-{attempt}.log"
        stage["log"] = str(log)
        result = self.out / "logs" / f"{key}-{attempt}.json"
        if kind == "preflight":
            rc = self.runner(["train", "--config", str(self.snapshot(experiment)), "--dry-run", "--result-json",
                              str(result)], log, lambda line: None)
            if rc == 0:
                stage["result"] = json.loads(result.read_text(encoding="utf-8"))["result"]
            return rc
        if kind in ("baseline", "eval"):
            checkpoint = (self.plan["baseline"]["checkpoint"] if kind == "baseline"
                          else state["stages"][f"train-{experiment['name']}"]["checkpoint"])
            folder = unique_directory(self.out / "evaluations", "baseline" if kind == "baseline" else experiment["name"])
            stage["dir"], stage["checkpoint"] = str(folder), checkpoint
            self.save(state)
            return self.runner(self.eval_args(checkpoint, folder), log, lambda line: None)
        if kind == "compare":
            folder = unique_directory(self.out / "comparisons", experiment["name"])
            stage["dir"] = str(folder)
            self.save(state)
            return self.runner(["comparison-report", "--baseline", state["stages"]["baseline-eval"]["dir"],
                                "--candidate", state["stages"][f"eval-{experiment['name']}"]["dir"],
                                "--baseline-name", self.plan["baseline"]["name"],
                                "--candidate-name", experiment["name"], "--out", str(folder)], log, lambda line: None)
        snapshot = self.snapshot(experiment)
        max_steps = int(load_config(snapshot).get("train", {}).get("max_steps", 60000))
        runs = stage.setdefault("runs", [])
        args = ["train", "--config", str(snapshot), "--result-json", str(result)]
        if runs:
            last = Path(runs[-1]) / "checkpoints" / "last.pt"
            if last.is_file():
                step, exact = _checkpoint_step(last)
                if step >= max_steps:
                    return self.finish_training(stage, Path(runs[-1]))
                if exact:
                    args += ["--resume", str(last)]
                    print(f"Resuming {experiment['name']} from step {step}: {last}", flush=True)

        def record(line: str) -> None:
            if line.startswith("run dir: "):
                runs.append(line[len("run dir: "):].strip())
                self.save(state)

        rc = self.runner(args, log, record)
        if rc == 0:
            return self.finish_training(stage, Path(json.loads(result.read_text(encoding="utf-8"))["result"]))
        return rc

    def finish_training(self, stage: dict, run: Path) -> int:
        checkpoint = run / "checkpoints" / f"{self.plan['checkpoint']}.pt"
        if not checkpoint.is_file():
            stage["error"] = f"{checkpoint} is missing"
            return 1
        stage["run"], stage["checkpoint"] = str(run), str(checkpoint)
        return 0

    def run(self, retry_failed: bool = False) -> dict:
        state = self.open()
        with exclusive_lock(self.out / "queue.lock"), _PauseRequest() as pause:
            self.save(state)
            for key, kind, experiment in stage_list(self.plan):
                stage = state["stages"].setdefault(key, {"state": "pending", "attempts": 0})
                if stage["state"] == "completed":
                    continue
                if stage["state"] == "failed" and not retry_failed:
                    raise QueueFailed(f"stage {key} failed earlier. Log: {stage.get('log')}. "
                                      "Fix the cause, then rerun with --retry-failed")
                if pause.requested:
                    break
                stage.update(state="running", attempts=stage["attempts"] + 1, error=None,
                             started=time.strftime("%Y-%m-%dT%H:%M:%S"))
                self.save(state)
                print(f"\n=== {key} (attempt {stage['attempts']}) ===", flush=True)
                t0 = time.time()
                rc = self.run_stage(key, kind, experiment, stage, state)
                stage["seconds"] = round(stage.get("seconds", 0) + time.time() - t0, 1)
                if rc == 0:
                    stage.update(state="completed", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    self.save(state)
                    self.write_summary(state)
                    continue
                if rc == 130 or pause.requested:
                    stage["state"] = "interrupted"
                    self.save(state)
                    break
                stage.update(state="failed", error=stage.get("error") or _tail(Path(stage["log"])))
                self.save(state)
                raise QueueFailed(f"stage {key} failed (exit {rc}). Log: {stage['log']}. "
                                  "Fix the cause, then rerun with --retry-failed")
            if pause.requested or any(s["state"] == "interrupted" for s in state["stages"].values()):
                raise QueuePaused(f"Queue paused. Rerun the same command to continue: {self.out}")
        self.write_summary(state)
        print(f"\nQueue complete. Review {self.out / 'summary.md'} and each comparison page.", flush=True)
        return state

    def write_summary(self, state: dict) -> None:
        from nullscape.eval.comparison import headline_metrics

        entries = [(self.plan["baseline"]["name"], "baseline-eval", None)]
        entries += [(e["name"], f"eval-{e['name']}", f"compare-{e['name']}") for e in self.plan["experiments"]]
        models = []
        for label, eval_key, compare_key in entries:
            stage = state["stages"].get(eval_key, {})
            row = {"model": label, "state": stage.get("state", "pending"), "checkpoint": stage.get("checkpoint"),
                   "evaluation": stage.get("dir"), "metrics": None, "targets": None, "comparison": None}
            report = Path(stage["dir"]) / "report.json" if stage.get("state") == "completed" else None
            if report and report.is_file():
                data = json.loads(report.read_text(encoding="utf-8"))
                row["metrics"] = headline_metrics(data)
                per = data["modes"]["conditional"].get("per_archetype", {})
                row["targets"] = {t: per[t]["ratio"] for t in TARGETS if t in per}
            compare = state["stages"].get(compare_key or "", {})
            if compare.get("state") == "completed":
                row["comparison"] = str(Path(compare["dir"]) / "index.html")
            models.append(row)
        summary = {"queue": self.plan["name"], "split": "val", "n_per_half": self.plan["evaluation"]["n"],
                   "sampling": self.plan["evaluation"]["sampling"], "checkpoint_rule": self.plan["checkpoint"],
                   "models": models,
                   "policy": "VAL screening only. The queue does not select, promote, or score TEST."}
        atomic_json(self.out / "summary.json", summary)

        def num(metrics, key):
            value = (metrics or {}).get(key)
            return "-" if value is None else f"{value:.3f}"

        lines = [f"# Queue {self.plan['name']}: VAL screening", "",
                 f"{self.plan['evaluation']['n']} maps per half, {self.plan['evaluation']['sampling']['steps']} "
                 f"{self.plan['evaluation']['sampling']['spacing']} steps, guidance "
                 f"{self.plan['evaluation']['sampling']['guidance']}. Experiments use their `{self.plan['checkpoint']}` "
                 "checkpoint.", "",
                 "The queue does not select a winner, promote a checkpoint, or score TEST. Review each comparison "
                 "page, choose on VAL, then score only the chosen checkpoint with `benchmarks/v2/score_test.py`.", "",
                 "| Model | State | Terrain W1/floor | Spectrum/floor | Slopes/floor | Control nMAE | "
                 "Mountains | Ridges | Plains | Comparison |", "|---" * 10 + "|"]
        for row in models:
            m, t = row["metrics"], row["targets"] or {}
            link = f"[open]({Path(row['comparison']).as_posix()})" if row["comparison"] else "-"
            lines.append(f"| {row['model']} | {row['state']} | {num(m, 'metric_w1_mean')} | {num(m, 'rapsd_distance')} | "
                         f"{num(m, 'slope_w1_deg')} | {num(m, 'control_nmae')} | {num(t, 'mountains')} | "
                         f"{num(t, 'ridges')} | {num(t, 'plains')} | {link} |")
        lines += ["", "Ratios are distance / real-vs-real noise floor (lower is better). Terrain columns are "
                      "per-type realism ratios."]
        with atomic_file(self.out / "summary.md") as file:
            file.write(("\n".join(lines) + "\n").encode("utf-8"))


def run_queue(plan: str | Path, out: str | Path | None = None, *, runner: Runner | None = None,
              retry_failed: bool = False) -> dict:
    return Queue(plan, out, runner).run(retry_failed=retry_failed)


def _cmd_queue(args) -> None:
    queue = Queue(args.plan, args.out, None)
    if args.check:
        print(f"Plan {queue.plan['name']} -> {queue.out} (nothing was run or written)")
        for key, kind, experiment in stage_list(queue.plan):
            detail = experiment["config"] if experiment else queue.plan["baseline"]["checkpoint"]
            print(f"  {key:32s} {detail}")
        return
    if args.status:
        if not queue.state_path.is_file():
            print(f"No queue state at {queue.out}; nothing has run yet.")
            return
        state = json.loads(queue.state_path.read_text(encoding="utf-8"))
        for key, _, _ in stage_list(queue.plan):
            stage = state["stages"].get(key, {"state": "pending", "attempts": 0})
            extra = stage.get("checkpoint") or stage.get("dir") or ""
            print(f"  {key:32s} {stage['state']:12s} attempts {stage['attempts']}  {extra}")
        return
    try:
        run_queue(args.plan, args.out, retry_failed=args.retry_failed)
    except QueueFailed as exc:
        raise SystemExit(f"nullscape: error: {exc}") from None


def add_queue_command(sub) -> None:
    parser = sub.add_parser("queue", help="run experiments one at a time: preflight, train, VAL evaluation, comparison")
    parser.add_argument("--plan", required=True, help="queue plan YAML, e.g. configs/queue/talus3.yaml")
    parser.add_argument("--out", default=None, help="queue folder (default reports/queues/<plan name>); rerun to resume")
    parser.add_argument("--retry-failed", action="store_true", help="retry a failed stage (training resumes from its last checkpoint)")
    parser.add_argument("--status", action="store_true", help="show stage progress without running anything")
    parser.add_argument("--check", action="store_true", help="validate the plan and list stages without running or writing")
    parser.set_defaults(func=_cmd_queue)
