import json
from pathlib import Path

import pytest
import torch
import yaml


def _write_plan(tmp_path, **changes):
    config = tmp_path / "experiment.yaml"
    if not config.exists():
        config.write_text(yaml.safe_dump({"name": "tiny", "dataset": "unused", "train": {"max_steps": 4}}))
    (tmp_path / "base.pt").touch()
    plan = {"name": "unit", "baseline": {"name": "Base", "checkpoint": str(tmp_path / "base.pt")},
            "evaluation": {"n": 4, "seed": 1},
            "experiments": [{"name": "a", "config": str(config)},
                            {"name": "b", "config": str(config), "set": ["train.lr=0.0001"]}]}
    plan.update(changes)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    return path


def _arg(args, name):
    return args[args.index(name) + 1]


class FakeRunner:
    """Stands in for `python -m nullscape.cli ...` and records every stage command."""

    def __init__(self, root, fail=(), interrupt=()):
        self.root, self.fail, self.interrupt, self.calls = Path(root), set(fail), set(interrupt), []

    def __call__(self, args, log, on_line):
        self.calls.append(list(args))
        command = args[0]
        if command == "train" and "--dry-run" in args:
            Path(_arg(args, "--result-json")).write_text(json.dumps({"state": "preflight", "result": {"peak_alloc_mb": 12.5}}))
            return 0
        if command == "train":
            name = yaml.safe_load(Path(_arg(args, "--config")).read_text())["name"]
            key = f"train-{Path(_arg(args, '--config')).stem}"
            run = self.root / "runs" / f"{Path(_arg(args, '--config')).stem}-{len(self.calls)}"
            (run / "checkpoints").mkdir(parents=True)
            on_line(f"run dir: {run}")
            if key in self.fail:
                return 1
            interrupted = key in self.interrupt and "--resume" not in args
            step = 2 if interrupted else 4
            for kind in ("last", "best"):
                torch.save({"step": step, "training_state": {"version": 1}}, run / "checkpoints" / f"{kind}.pt")
            if interrupted:
                return 130
            Path(_arg(args, "--result-json")).write_text(json.dumps({"state": "completed", "result": str(run), "name": name}))
            return 0
        out = Path(_arg(args, "--out"))
        if command == "evaluate":
            report = {"checkpoint": {"path": _arg(args, "--checkpoint")}, "split": "val",
                      "modes": {"conditional": {"ratio_to_floor": {"metric_w1_mean": 1.5, "rapsd_distance": 9.0,
                                                                   "height_w1": 1.0, "slope_w1_deg": 1.7},
                                                "model_vs_ref": {"trav_pass_rate_gen": 0.9, "diversity_gen": 1.0,
                                                                 "diversity_ref": 1.0},
                                                "per_archetype": {"mountains": {"ratio": 1.9}}}}}
            (out / "report.json").write_text(json.dumps(report))
            return 0
        if command == "comparison-report":
            (out / "index.html").write_text("<!doctype html>")
            return 0
        raise AssertionError(args)


def test_plan_validation_rejects_unsafe_or_ambiguous_plans(tmp_path):
    from nullscape.train.queue import load_plan

    plan = load_plan(_write_plan(tmp_path))
    assert [e["name"] for e in plan["experiments"]] == ["a", "b"]
    assert plan["evaluation"]["split"] == "val" and plan["checkpoint"] == "last"
    bad = [{"evaluation": {"split": "test"}}, {"experiments": []}, {"checkpoint": "newest"},
           {"experiments": [{"name": "a b", "config": "x"}]}, {"surprise": 1},
           {"experiments": [{"name": "a", "config": str(tmp_path / "experiment.yaml")}] * 2},
           {"experiments": [{"name": "a", "config": str(tmp_path / "missing.yaml")}]},
           {"baseline": {"name": "Base", "checkpoint": str(tmp_path / "missing.pt")}},
           {"evaluation": {"modes": "label_only"}}]
    for i, change in enumerate(bad):
        folder = tmp_path / f"bad{i}"
        folder.mkdir()
        (folder / "experiment.yaml").write_text((tmp_path / "experiment.yaml").read_text())
        with pytest.raises((ValueError, FileNotFoundError)):
            load_plan(_write_plan(folder, **change))


def test_queue_runs_stages_in_order_and_skips_completed_work(tmp_path):
    from nullscape.train.queue import run_queue

    runner = FakeRunner(tmp_path)
    out = tmp_path / "queue"
    state = run_queue(_write_plan(tmp_path), out, runner=runner)
    kinds = [c[0] + (" --dry-run" if "--dry-run" in c else "") for c in runner.calls]
    assert kinds == ["train --dry-run", "train --dry-run", "evaluate", "train", "evaluate", "comparison-report",
                     "train", "evaluate", "comparison-report"]
    assert all(s["state"] == "completed" for s in state["stages"].values())
    assert all(_arg(c, "--split") == "val" and _arg(c, "--steps") == "50" for c in runner.calls if c[0] == "evaluate")
    assert _arg(runner.calls[4], "--checkpoint").endswith("last.pt")
    summary = json.loads((out / "summary.json").read_text())
    assert [r["model"] for r in summary["models"]] == ["Base", "a", "b"]
    assert summary["models"][1]["metrics"]["metric_w1_mean"] == 1.5
    assert "does not select" in (out / "summary.md").read_text()
    snapshot = yaml.safe_load((out / "configs" / "b.yaml").read_text())
    assert snapshot["train"]["lr"] == pytest.approx(1e-4)
    assert all("--dataset" not in c for c in runner.calls if c[0] == "evaluate")
    runner.calls.clear()
    run_queue(_write_plan(tmp_path), out, runner=runner)
    assert runner.calls == []


def test_evaluation_dataset_is_optional_and_passed_to_every_evaluation(tmp_path):
    from nullscape.train.queue import load_plan, run_queue

    assert "dataset" not in load_plan(_write_plan(tmp_path))["evaluation"]  # older queue folders stay valid
    runner = FakeRunner(tmp_path)
    run_queue(_write_plan(tmp_path, evaluation={"n": 4, "seed": 1, "dataset": "base64"}), tmp_path / "q", runner=runner)
    evaluations = [c for c in runner.calls if c[0] == "evaluate"]
    assert len(evaluations) == 3 and all(_arg(c, "--dataset") == "base64" for c in evaluations)
    with pytest.raises(ValueError):
        load_plan(_write_plan(tmp_path, evaluation={"dataset": 5}))


def test_failure_stops_queue_and_retry_is_explicit(tmp_path):
    from nullscape.train.queue import QueueFailed, run_queue

    out, plan = tmp_path / "queue", _write_plan(tmp_path)
    runner = FakeRunner(tmp_path, fail={"train-a"})
    with pytest.raises(QueueFailed, match="retry-failed"):
        run_queue(plan, out, runner=runner)
    assert not any(c[0] == "train" and "configs" in c[2] and c[2].endswith("b.yaml") and "--dry-run" not in c
                   for c in runner.calls)
    state = json.loads((out / "queue_state.json").read_text())
    assert state["stages"]["train-a"]["state"] == "failed" and state["stages"]["train-a"]["log"]
    runner.calls.clear()
    with pytest.raises(QueueFailed):
        run_queue(plan, out, runner=runner)
    assert runner.calls == []
    runner.fail.clear()
    final = run_queue(plan, out, runner=runner, retry_failed=True)
    assert final["stages"]["train-a"]["attempts"] == 2
    assert all(s["state"] == "completed" for s in final["stages"].values())


def test_interrupted_training_resumes_from_its_recovery_checkpoint(tmp_path):
    from nullscape.train.queue import QueuePaused, run_queue

    out, plan = tmp_path / "queue", _write_plan(tmp_path)
    runner = FakeRunner(tmp_path, interrupt={"train-a"})
    with pytest.raises(QueuePaused):
        run_queue(plan, out, runner=runner)
    state = json.loads((out / "queue_state.json").read_text())
    stage = state["stages"]["train-a"]
    assert stage["state"] == "interrupted" and len(stage["runs"]) == 1
    run_queue(plan, out, runner=runner)
    resumed = [c for c in runner.calls if c[0] == "train" and "--resume" in c]
    assert len(resumed) == 1
    assert Path(_arg(resumed[0], "--resume")) == Path(stage["runs"][0]) / "checkpoints" / "last.pt"


def test_training_finished_before_queue_recorded_it_is_not_repeated(tmp_path):
    from nullscape.train.queue import run_queue

    out, plan = tmp_path / "queue", _write_plan(tmp_path)
    run_queue(plan, out, runner=FakeRunner(tmp_path))
    state = json.loads((out / "queue_state.json").read_text())
    for key in ("train-b", "eval-b", "compare-b"):
        state["stages"][key]["state"] = "running" if key == "train-b" else "pending"
    (out / "queue_state.json").write_text(json.dumps(state))
    runner = FakeRunner(tmp_path)
    run_queue(plan, out, runner=runner)
    assert [c[0] for c in runner.calls] == ["evaluate", "comparison-report"]


def test_changed_plan_and_foreign_folder_are_rejected(tmp_path):
    from nullscape.train.queue import run_queue

    out, plan = tmp_path / "queue", _write_plan(tmp_path)
    run_queue(plan, out, runner=FakeRunner(tmp_path))
    changed = _write_plan(tmp_path, evaluation={"n": 8, "seed": 1})
    with pytest.raises(ValueError, match="plan changed"):
        run_queue(changed, out, runner=FakeRunner(tmp_path))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not a queue folder"):
        run_queue(_write_plan(tmp_path), foreign, runner=FakeRunner(tmp_path))


def test_queue_lock_blocks_a_second_process(tmp_path):
    from nullscape.train.queue import exclusive_lock

    with exclusive_lock(tmp_path / "queue.lock"):
        with pytest.raises(ValueError, match="another queue"):
            with exclusive_lock(tmp_path / "queue.lock"):
                pass
    with exclusive_lock(tmp_path / "queue.lock"):
        pass


def test_status_reports_without_running(tmp_path, capsys):
    from nullscape.cli import main

    out, plan = tmp_path / "queue", _write_plan(tmp_path)
    main(["queue", "--plan", str(plan), "--out", str(out), "--check"])
    text = capsys.readouterr().out
    assert "preflight-a" in text and "compare-b" in text and not out.exists()


@pytest.mark.slow
def test_real_queue_trains_evaluates_and_compares_on_cpu(tmp_path, monkeypatch):
    from nullscape.data.build import build_dataset
    from nullscape.train.queue import run_queue
    from nullscape.train.trainer import train
    from test_train_smoke import DS_CFG, TRAIN_CFG

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("NULLSCAPE_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = build_dataset(DS_CFG, workers=1, out_root=tmp_path, progress=False)
    base_train = {**TRAIN_CFG["train"], "artifacts_every": 0, "eval_every": 4, "eval_steps": 3}
    base = train({**TRAIN_CFG, "dataset": str(dataset), "train": base_train}) / "checkpoints" / "last.pt"
    config = tmp_path / "candidate.yaml"
    config.write_text(yaml.safe_dump({**TRAIN_CFG, "name": "candidate", "dataset": str(dataset),
                                      "train": {**base_train, "init_from": str(base), "lr": 1e-4}}))
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump({
        "name": "cpu", "baseline": {"name": "Base", "checkpoint": str(base)},
        "evaluation": {"n": 4, "seed": 2, "batch_size": 4, "gameplay_n": 2, "memorization": False, "device": "cpu",
                       "sampling": {"steps": 3}},
        "experiments": [{"name": "candidate", "config": str(config)}]}))
    state = run_queue(plan, tmp_path / "queue")
    assert all(s["state"] == "completed" for s in state["stages"].values())
    page = Path(state["stages"]["compare-candidate"]["dir"]) / "index.html"
    assert page.exists() and "comparison-data" in page.read_text(encoding="utf-8")
    summary = json.loads((tmp_path / "queue" / "summary.json").read_text())
    assert summary["models"][1]["metrics"]["metric_w1_mean"] is not None
