"""NULLSCAPE v1 benchmark driver (frozen 64x64 model, frozen base64 dataset).

Stages are resumable: each writes benchmarks/v1/raw/<stage>.json (and figures) and is skipped
if its output exists, unless --force. Nothing here writes to data/ or runs/.

  provenance        git commit, checkpoint sha256, dataset manifest hash, env, specs
  checkpoints_test  full artifact suite per checkpoint on the TEST split (n=1000/half)
  checkpoints_val   same sampler on VAL (n=1000/half) -> used only for selection
  select            checkpoint selection rule applied to VAL results
  sampler_val       sampler grid on VAL with the selected checkpoint
  perf              isolated latency / throughput / VRAM / determinism (GPU otherwise idle)
  pareto            quality/cost frontier + recommended sampler (VAL quality, isolated cost)
  sampler_test      frontier configs re-measured on TEST
  stress            conditioning stress tests (selected checkpoint + recommended sampler)
  analysis          bootstrap CIs, archetype failure modes, memorization deep dive, adherence bins
  training          training throughput from run logs
  aggregate         benchmarks/v1/results.json

usage: python benchmarks/v1/run_v1_benchmark.py --stages checkpoints_test,checkpoints_val [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import os

import numpy as np

# NULLSCAPE_BENCH_DRY=1: CPU, tiny sizes, 3 sampler steps, outputs under benchmarks/v1/dryrun (code check only)
DRY = os.environ.get("NULLSCAPE_BENCH_DRY") == "1"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "v1" / ("dryrun" if DRY else "")
RAW = OUT / "raw"
FIG = OUT / "figures"
ART = OUT / "artifacts_test"
GEN = OUT / "generated"

MAIN = ROOT / "runs" / "20260925-164759_diffusion64" / "checkpoints"
CONT = ROOT / "runs" / "20260925-192102_diffusion64_cont" / "checkpoints"
CKPTS = {
    20000: MAIN / "step_0020000.pt",
    25000: MAIN / "step_0025000.pt",
    30000: MAIN / "step_0030000.pt",
    35000: CONT / "step_0035000.pt",
    40000: CONT / "step_0040000.pt",
}
DATASET = "base64"
N_HALF = 48 if DRY else 1000
DEFAULT_SAMPLER = {"steps": 200, "spacing": "uniform", "eta": 0.0, "guidance": 1.5}
BATCH = 8 if DRY else 128
DEVICE = "cpu" if DRY else "cuda"
if DRY:
    CKPTS = {30000: CKPTS[30000], 40000: CKPTS[40000]}


def _dry_cfg(cfg: dict) -> dict:
    return {**cfg, "steps": min(cfg["steps"], 3)} if DRY else cfg


def _sync() -> None:
    if DEVICE == "cuda":
        import torch

        torch.cuda.synchronize()

# Selection rules (fixed before any benchmark result was seen)
SELECTION_RULES = {
    "checkpoint": "lowest VAL ratio_to_floor.metric_w1_mean with the default sampler; if the runner-up is within "
                  "0.05, the lower mean of the four headline ratios wins",
    "sampler": "among configs whose VAL metric_w1_mean ratio is within 5% of the best config, the one with the "
               "lowest isolated seconds/map at batch 128; ties -> lower mean adherence nMAE",
    "final_numbers": "TEST split, n=1000 maps per half (A conditions / B reference), noise floor = A vs B",
}


# ----------------------------------------------------------------------------- helpers

def save_json(name: str, obj) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_json(name: str):
    return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def store():
    from nullscape.data.storage import TerrainStore

    return TerrainStore.open(DATASET)


def cfg_name(c: dict) -> str:
    return f"s{c['steps']}_{c['spacing'][0]}_g{c['guidance']:g}_e{c['eta']:g}"


def halves(st, split: str, n: int = N_HALF):
    from nullscape.eval.artifacts import ArtifactSpec, eval_halves

    return eval_halves(st, ArtifactSpec(split=split, n_eval=n))


def sampler_for(step: int):
    from nullscape.inference.sampler import TerrainSampler

    return TerrainSampler.from_checkpoint(CKPTS[step], device=DEVICE)


def conditional_gen(sampler, st, idx, cfg: dict, seed: int, batch: int = BATCH) -> tuple[np.ndarray, float]:
    k = len(st.condition_keys)
    _sync()
    t0 = time.time()
    g = sampler.sample(n=len(idx), seed=seed, cond_raw=st.conditions[idx], known=np.ones((len(idx), k), bool),
                       labels=st.labels[idx], batch_size=batch, **_dry_cfg(cfg))
    _sync()
    return g, time.time() - t0


def _empty_cache() -> None:
    if DEVICE == "cuda":
        import torch

        torch.cuda.empty_cache()


def compact(rep: dict) -> dict:
    """Headline subset of an evaluate_generated report."""
    mv = rep["model_vs_ref"]
    out = {
        "ratio": rep["ratio_to_floor"],
        "model": {k: mv[k] for k in ("metric_w1_mean", "rapsd_distance", "height_w1", "slope_w1_deg")},
        "floor": {k: rep["noise_floor"][k] for k in ("metric_w1_mean", "rapsd_distance", "height_w1", "slope_w1_deg")},
        "trav_pass_gen": mv["trav_pass_rate_gen"], "trav_pass_ref": mv["trav_pass_rate_ref"],
        "diversity_gen": mv["diversity_gen"], "diversity_ref": mv["diversity_ref"],
        "artifacts": rep["artifacts"],
        "peak_density": rep["metric_means"]["peak_density"],
        "sink_density": rep["metric_means"]["sink_density"],
    }
    if "condition_adherence" in rep:
        out["adherence_nmae"] = {k: v["nmae"] for k, v in rep["condition_adherence"].items()}
        out["adherence_r"] = {k: v["pearson_r"] for k, v in rep["condition_adherence"].items()}
        out["adherence_nmae_mean"] = float(np.mean(list(out["adherence_nmae"].values())))
    if "per_archetype" in rep:
        out["per_archetype_ratio"] = {k: v["ratio"] for k, v in rep["per_archetype"].items()}
    if "memorization" in rep:
        out["memorization"] = rep["memorization"]
    # n-independent view: distance in excess of the sampling-noise floor (same units as the distance)
    out["excess_over_floor"] = {k: out["model"][k] - out["floor"][k] for k in out["model"]}
    out["rapsd_bias_decades"] = float(np.sqrt(max(out["model"]["rapsd_distance"] ** 2 -
                                                  out["floor"]["rapsd_distance"] ** 2, 0.0)))
    return out


# ----------------------------------------------------------------------------- stages

def stage_provenance(args) -> None:
    from nullscape.eval.artifacts import ArtifactSpec
    from nullscape.utils.config import config_hash
    from nullscape.utils.tracking import environment_info

    st = store()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    import torch

    ckpts = {}
    for step, path in CKPTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ckpts[str(step)] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size,
                            "step_in_file": int(ck["step"]), "read_only": not bool(path.stat().st_mode & 0o200),
                            "dataset_config_sha256": ck["dataset"]["config_sha256"],
                            "train_run_git": ck.get("env", {}).get("git")}
    m = st.manifest
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        gpu = None
    save_json("provenance", {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": {"commit": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--porcelain").splitlines()},
        "checkpoints": ckpts,
        "dataset": {"name": m["name"], "root": str(st.root.relative_to(ROOT)), "n": m["n"], "splits": m["splits"],
                    "generator_version": m["generator_version"], "config_sha256": m["config_sha256"],
                    "config_sha256_recomputed": config_hash(m["config"]),
                    "manifest_sha256": sha256(st.root / "manifest.json"),
                    "heights_sha256": sha256(st.root / "heights.npy"), "world": m["world"],
                    "archetype_counts": m["archetype_counts"], "build_git": m.get("git")},
        "artifact_spec_test": asdict(ArtifactSpec(split="test", n_eval=N_HALF, batch_size=BATCH)),
        "default_sampler": DEFAULT_SAMPLER, "n_per_half": N_HALF, "selection_rules": SELECTION_RULES,
        "environment": {**environment_info(), "gpu_query": gpu, "cpu": platform.processor(),
                        "torch_cudnn": torch.backends.cudnn.version()},
    })


def test_spec():
    from nullscape.eval.artifacts import ArtifactSpec

    return ArtifactSpec(split="test", n_eval=N_HALF, batch_size=BATCH, **({"steps": 3, "n_3d": 1} if DRY else {}))


def stage_checkpoints_test(args) -> None:
    from nullscape.eval.artifacts import generate_checkpoint_artifacts, step_dir_name

    st = store()
    spec = test_spec()
    bank = st.heights(st.split("train"))
    timing = {}
    for step in CKPTS:
        d = ART / step_dir_name(step)
        if (d / "report.json").exists() and (d / "generated_eval.npy").exists() and not args.force:
            continue
        t0 = time.time()
        s = sampler_for(step)
        generate_checkpoint_artifacts(s, st, ART, step, spec, train_bank=bank, memorization_device=DEVICE,
                                      save_generated=True)
        timing[step] = time.time() - t0
        del s
        _empty_cache()
        print(f"[checkpoints_test] {step} done in {timing[step]:.0f}s", flush=True)
    reports = {str(s): json.loads((ART / step_dir_name(s) / "report.json").read_text())["modes"]["conditional"]
               for s in CKPTS}
    save_json("checkpoints_test", {"spec": asdict(spec), "wall_seconds": timing,
                                   "compact": {k: compact(v) for k, v in reports.items()}})


def stage_checkpoints_val(args) -> None:
    from nullscape.eval.core import evaluate_generated
    from nullscape.metrics.distribution import metric_table

    st = store()
    a, b = halves(st, "val")
    ref_a, ref_b = st.heights(a), st.heights(b)
    tables = {"a": metric_table(ref_a, st.world), "b": metric_table(ref_b, st.world)}
    out = {}
    GEN.mkdir(parents=True, exist_ok=True)
    for step in CKPTS:
        s = sampler_for(step)
        gen, secs = conditional_gen(s, st, a, DEFAULT_SAMPLER, seed=2027)
        np.save(GEN / f"val_{step}.npy", gen)
        rep = evaluate_generated(gen, ref_a, ref_b, st.world, requested_conds=st.conditions[a],
                                 gen_labels=st.labels[a], ref_a_labels=st.labels[a], ref_b_labels=st.labels[b],
                                 archetype_names=st.archetypes, tables={**tables, "gen": metric_table(gen, st.world)})
        out[str(step)] = {"compact": compact(rep), "sampling_seconds": secs, "report": rep}
        del s
        _empty_cache()
        print(f"[checkpoints_val] {step}: ratio {rep['ratio_to_floor']}", flush=True)
    save_json("checkpoints_val", {"n_per_half": len(a), "sampler": DEFAULT_SAMPLER, "results": out})


def stage_select(args) -> None:
    val = load_json("checkpoints_val")["results"]
    score = {int(k): v["compact"]["ratio"]["metric_w1_mean"] for k, v in val.items()}
    mean4 = {int(k): float(np.mean(list(v["compact"]["ratio"].values()))) for k, v in val.items()}
    order = sorted(score, key=score.get)
    best, runner = order[0], order[1]
    reason = f"lowest VAL metric_w1_mean ratio ({score[best]:.3f})"
    if score[runner] - score[best] < 0.05 and mean4[runner] < mean4[best]:
        best, reason = runner, (f"tie within 0.05 on metric_w1_mean ({score[order[0]]:.3f} vs {score[runner]:.3f}); "
                                f"lower mean headline ratio ({mean4[runner]:.3f} vs {mean4[order[0]]:.3f})")
    elif score[runner] - score[best] < 0.05:
        reason += f"; runner-up {runner} within 0.05 but higher mean headline ratio ({mean4[runner]:.3f} vs {mean4[best]:.3f})"
    save_json("selection", {"checkpoint": best, "reason": reason, "val_metric_w1_ratio": score,
                            "val_mean_headline_ratio": mean4, "rule": SELECTION_RULES["checkpoint"]})
    print(f"[select] checkpoint {best}: {reason}")


def sampler_grid() -> list[dict]:
    grid = [{"steps": s, "spacing": sp, "guidance": 1.5, "eta": 0.0}
            for s, sp in itertools.product((25, 50, 100, 200), ("uniform", "quadratic"))]
    grid += [{"steps": 100, "spacing": sp, "guidance": g, "eta": 0.0}
             for sp, g in itertools.product(("uniform", "quadratic"), (1.0, 2.0, 3.0))]
    grid += [{"steps": 100, "spacing": sp, "guidance": 1.5, "eta": e}
             for sp, e in itertools.product(("uniform", "quadratic"), (0.5, 1.0))]
    grid += [{"steps": 50, "spacing": "quadratic", "guidance": g, "eta": 0.0} for g in (1.0, 2.0)]
    return grid


def _sampler_eval(split: str, configs: list[dict], name: str, fixed_visual: bool, n: int = N_HALF) -> None:
    from nullscape.eval.artifacts import ArtifactSpec, fixed_indices
    from nullscape.eval.core import evaluate_generated
    from nullscape.metrics.distribution import metric_table

    st = store()
    step = load_json("selection")["checkpoint"]
    s = sampler_for(step)
    a, b = halves(st, split, n)
    ref_a, ref_b = st.heights(a), st.heights(b)
    tables = {"a": metric_table(ref_a, st.world), "b": metric_table(ref_b, st.world)}
    fx = fixed_indices(st, ArtifactSpec(split=split))
    rows, fixed = {}, {}
    for cfg in configs:
        gen, secs = conditional_gen(s, st, a, cfg, seed=2027)  # same noise seeds for every config
        rep = evaluate_generated(gen, ref_a, ref_b, st.world, requested_conds=st.conditions[a],
                                 gen_labels=st.labels[a], ref_a_labels=st.labels[a], ref_b_labels=st.labels[b],
                                 archetype_names=st.archetypes, tables={**tables, "gen": metric_table(gen, st.world)})
        rows[cfg_name(cfg)] = {"config": cfg, "compact": compact(rep), "inprocess_seconds_per_map": secs / len(a)}
        if fixed_visual:
            fixed[cfg_name(cfg)] = conditional_gen(s, st, fx, cfg, seed=2026)[0]
        print(f"[{name}] {cfg_name(cfg)}: ratio {rep['ratio_to_floor']['metric_w1_mean']:.3f} "
              f"rapsd {rep['ratio_to_floor']['rapsd_distance']:.3f} {secs / len(a) * 1000:.1f} ms/map", flush=True)
        _empty_cache()
    save_json(name, {"checkpoint": step, "split": split, "n_per_half": len(a), "rows": rows})
    if fixed_visual:
        GEN.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GEN / f"{name}_fixed.npz", indices=fx, **fixed)


N_SWEEP = 8 if DRY else 500  # VAL sampler grid only ranks configs against each other; TEST confirmation uses N_HALF
PERF_BATCHES = (1, 4) if DRY else (1, 32, 128, 256)
COST_BATCH = 4 if DRY else 128
PERF_VRAM_FRACTION = 0.6  # ~4.9 GB of 8 GB for PyTorch; the rest is CUDA context + desktop


def stage_sampler_val(args) -> None:
    _sampler_eval("val", sampler_grid(), "sampler_val", fixed_visual=True, n=N_SWEEP)


def _peak_rss_mb() -> float | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE  # without this the 64-bit pseudo-handle is truncated
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 2**20
    except (AttributeError, OSError):
        return None


def _nvsmi_used_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        return float(out.splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def stage_perf(args) -> None:
    import torch

    from nullscape.metrics.quality import compute_metrics
    from nullscape.metrics.traversability import analyze

    st = store()
    step = load_json("selection")["checkpoint"]
    idle_mb = _nvsmi_used_mb()
    gpu = DEVICE == "cuda"
    if gpu:
        # A first attempt without a cap spilled into system RAM at batch 256 + guidance (peak 5.0 GB allocated
        # plus desktop use) and slowed every later measurement ~20x. With a cap, over-budget configs raise OOM
        # and are recorded as such instead of silently spilling.
        torch.cuda.set_per_process_memory_fraction(PERF_VRAM_FRACTION)
    t0 = time.time()
    s = sampler_for(step)
    load_s = time.time() - t0
    fx = st.split("test")[:256]
    rows = []
    for steps, guidance in itertools.product((25, 50, 100, 200), (1.0, 1.5)):
        for bs in PERF_BATCHES:
            n = 4 if bs == 1 else bs
            cfg = {"steps": steps, "spacing": "uniform", "eta": 0.0, "guidance": guidance}
            row = {"steps": steps, "guidance": guidance, "cfg_doubles_batch": guidance != 1.0, "batch": bs, "maps": n}
            try:
                _empty_cache()
                conditional_gen(s, st, fx[:bs], {**cfg, "steps": 2}, seed=1, batch=bs)  # warm-up this exact shape
                if gpu:
                    torch.cuda.reset_peak_memory_stats()
                _, secs = conditional_gen(s, st, fx[:n], cfg, seed=7, batch=bs)
            except torch.OutOfMemoryError:
                _empty_cache()
                rows.append({**row, "oom": True})
                print(f"[perf] steps {steps} g {guidance} bs {bs}: OOM under the {PERF_VRAM_FRACTION:.2f} VRAM cap", flush=True)
                continue
            used = _nvsmi_used_mb()
            rows.append({**row, "oom": False, "seconds": secs, "seconds_per_map": secs / n, "maps_per_second": n / secs,
                         "latency_seconds_per_batch": secs / max(1, n // bs),
                         "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20 if gpu else 0.0,
                         "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if gpu else 0.0,
                         "nvidia_smi_used_mb": used,
                         "spill_suspect": bool(used is not None and used > 0.93 * 8151)})
            print(f"[perf] steps {steps} g {guidance} bs {bs}: {n / secs:.2f} maps/s "
                  f"peak {rows[-1]['peak_alloc_mb']:.0f} MB, smi {used} MB", flush=True)
    # GPU determinism: same seed twice, and batch-size invariance
    cfg = dict(DEFAULT_SAMPLER)
    x1, _ = conditional_gen(s, st, fx[:16], cfg, seed=11, batch=16)
    x2, _ = conditional_gen(s, st, fx[:16], cfg, seed=11, batch=16)
    x3, _ = conditional_gen(s, st, fx[:16], cfg, seed=11, batch=1)
    x4, _ = conditional_gen(s, st, fx[:16], cfg, seed=12, batch=16)
    det = {"same_seed_same_batch_max_abs_diff": float(np.abs(x1 - x2).max()),
           "batch16_vs_batch1_max_abs_diff": float(np.abs(x1 - x3).max()),
           "batch16_vs_batch1_max_abs_diff_m": float(np.abs(x1 - x3).max() * st.world.max_height_m),
           "different_seed_mean_abs_diff": float(np.abs(x1 - x4).mean())}
    # CPU cost of the evaluation metrics
    maps = st.heights(np.sort(fx[:200]))
    t0 = time.time()
    for h in maps:
        compute_metrics(h, st.world)
    t_metrics = (time.time() - t0) / len(maps)
    t0 = time.time()
    for h in maps:
        analyze(h, st.world)
    t_trav = (time.time() - t0) / len(maps)
    save_json("perf", {"checkpoint": step, "checkpoint_load_seconds": load_s, "gpu_idle_used_mb": idle_mb,
                       "vram_fraction_cap": PERF_VRAM_FRACTION if gpu else None,
                       "rows": rows, "determinism": det, "peak_process_rss_mb": _peak_rss_mb(),
                       "cpu_seconds_per_map": {"compute_metrics": t_metrics, "traversability": t_trav},
                       "note": "cost is independent of spacing and eta; measured with uniform spacing, eta 0"})


def iso_cost(perf: dict, cfg: dict, batch: int = COST_BATCH) -> float:
    g = 1.0 if cfg["guidance"] == 1.0 else 1.5
    for r in perf["rows"]:
        if r["steps"] == cfg["steps"] and r["guidance"] == g and r["batch"] == batch and not r.get("oom"):
            return r["seconds_per_map"]
    raise KeyError(cfg)


def pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
    """Names on the lower-left frontier of (cost, quality) where both are minimized."""
    front, best_q = [], np.inf
    for name, cost, q in sorted(points, key=lambda p: (p[1], p[2])):
        if q < best_q:
            front.append(name)
            best_q = q
    return front


def stage_pareto(args) -> None:
    val = load_json("sampler_val")["rows"]
    perf = load_json("perf")
    pts = [(n, iso_cost(perf, r["config"]), r["compact"]["ratio"]["metric_w1_mean"]) for n, r in val.items()]
    front = pareto_front(pts)
    best_q = min(q for _, _, q in pts)
    eligible = [(n, c, q) for n, c, q in pts if q <= 1.05 * best_q]
    eligible.sort(key=lambda p: (round(p[1], 6), val[p[0]]["compact"]["adherence_nmae_mean"]))
    rec = eligible[0][0]
    save_json("pareto", {"points": [{"name": n, "iso_seconds_per_map_b128": c, "val_metric_w1_ratio": q,
                                     "val_rapsd_ratio": val[n]["compact"]["ratio"]["rapsd_distance"],
                                     "val_adherence_nmae_mean": val[n]["compact"]["adherence_nmae_mean"],
                                     "config": val[n]["config"]} for n, c, q in pts],
                         "frontier": front, "best_quality": min(pts, key=lambda p: p[2])[0], "recommended": rec,
                         "recommended_config": val[rec]["config"], "rule": SELECTION_RULES["sampler"]})
    print(f"[pareto] frontier {front}; recommended {rec}")


def stage_sampler_test(args) -> None:
    p = load_json("pareto")
    val = load_json("sampler_val")["rows"]
    names = list(dict.fromkeys(p["frontier"] + [p["recommended"], p["best_quality"]]))
    configs = [val[n]["config"] for n in names if val[n]["config"] != DEFAULT_SAMPLER]
    _sampler_eval("test", configs, "sampler_test", fixed_visual=False)


def stage_stress(args) -> None:
    from nullscape.metrics.quality import CONDITION_KEYS, compute_metrics

    st = store()
    step = load_json("selection")["checkpoint"]
    cfg = load_json("pareto")["recommended_config"]
    s = sampler_for(step)
    keys = list(CONDITION_KEYS)
    K = len(keys)
    tr = st.split("train")
    ctr = st.conditions[tr].astype(np.float64)
    ltr = st.labels[tr]
    std = ctr.std(0)
    mean = ctr.mean(0)
    levels = {}
    for j, k in enumerate(keys):
        q05, q50, q95 = np.quantile(ctr[:, j], [0.05, 0.5, 0.95])
        if q50 - q05 < 1e-6:  # degenerate lower half (water_fraction is 0 for most maps)
            q50 = float(np.median(ctr[ctr[:, j] > q05 + 1e-6, j]))
        levels[k] = {"low": float(q05), "mid": float(q50), "high": float(q95)}

    def run(case_cond, case_known, labels, seed):
        n = len(case_cond)
        maps = s.sample(n=n, seed=seed, cond_raw=case_cond, known=case_known, labels=labels, batch_size=BATCH,
                        **_dry_cfg(cfg))
        return maps, np.stack([[compute_metrics(h, st.world)[k] for k in keys] for h in maps])

    def near(j, v, width=0.1):
        sel = np.abs(ctr[:, j] - v) < width * std[j]
        return sel if sel.sum() >= 50 else np.argsort(np.abs(ctr[:, j] - v))[:200]

    single, seed = {}, 50_000
    n1, n2, n3 = (4, 2, 2) if DRY else (64, 32, 16)
    for j, k in enumerate(keys):
        for lvl, v in levels[k].items():
            c = np.tile(mean, (n1, 1)); c[:, j] = v
            kn = np.zeros((n1, K), bool); kn[:, j] = True
            _, meas = run(c, kn, np.full(n1, -1), seed); seed += 1
            nat = ctr[near(j, v)]
            single[f"{k}/{lvl}"] = {
                "key": k, "level": lvl, "requested": v, "n": n1,
                "measured_mean": float(meas[:, j].mean()), "measured_std": float(meas[:, j].std()),
                "bias": float(meas[:, j].mean() - v), "mae": float(np.abs(meas[:, j] - v).mean()),
                "nmae": float(np.abs(meas[:, j] - v).mean() / std[j]),
                "others_vs_natural_z": {keys[i]: float((meas[:, i].mean() - nat[:, i].mean()) / std[i])
                                        for i in range(K) if i != j},
                "others_measured_mean": {keys[i]: float(meas[:, i].mean()) for i in range(K) if i != j},
                "others_natural_mean": {keys[i]: float(nat[:, i].mean()) for i in range(K) if i != j}}
            print(f"[stress] single {k}/{lvl}: req {v:.3f} got {meas[:, j].mean():.3f} nMAE {single[f'{k}/{lvl}']['nmae']:.3f}",
                  flush=True)

    pairs = {}
    for i, j in itertools.combinations(range(K), 2):
        for li, lj in itertools.product(("low", "high"), repeat=2):
            vi, vj = levels[keys[i]][li], levels[keys[j]][lj]
            c = np.tile(mean, (n2, 1)); c[:, i] = vi; c[:, j] = vj
            kn = np.zeros((n2, K), bool); kn[:, [i, j]] = True
            _, meas = run(c, kn, np.full(n2, -1), seed); seed += 1
            support = int(((np.abs(ctr[:, i] - vi) < 0.25 * std[i]) & (np.abs(ctr[:, j] - vj) < 0.25 * std[j])).sum())
            e_i = float(np.abs(meas[:, i] - vi).mean() / std[i])
            e_j = float(np.abs(meas[:, j] - vj).mean() / std[j])
            pairs[f"{keys[i]}={li}|{keys[j]}={lj}"] = {
                "a": keys[i], "b": keys[j], "a_level": li, "b_level": lj, "a_req": vi, "b_req": vj,
                "a_nmae": e_i, "b_nmae": e_j,
                "a_nmae_alone": single[f"{keys[i]}/{li}"]["nmae"], "b_nmae_alone": single[f"{keys[j]}/{lj}"]["nmae"],
                "train_support": support}

    # partial conditioning from real test maps: m of 5 properties known, labels unknown
    a, _ = halves(st, "test")
    rng = np.random.default_rng(123)
    m_known = np.repeat(np.arange(1, K + 1), len(a) // K + 1)[: len(a)]
    kn = np.zeros((len(a), K), bool)
    for r, m in enumerate(m_known):
        kn[r, rng.choice(K, m, replace=False)] = True
    _, meas = run(st.conditions[a], kn, np.full(len(a), -1), seed); seed += 1
    req = st.conditions[a].astype(np.float64)
    partial = {}
    for m in range(1, K + 1):
        rows = m_known == m
        partial[str(m)] = {k: float(np.abs(meas[rows & kn[:, j], j] - req[rows & kn[:, j], j]).mean() / std[j])
                           for j, k in enumerate(keys) if (rows & kn[:, j]).any()}
    unk = {k: float(np.abs(meas[~kn[:, j], j] - req[~kn[:, j], j]).mean() / std[j]) for j, k in enumerate(keys)}

    # within-archetype control gain: request the archetype's own p10 vs p90 with the label known
    gain = {}
    for c_id, arch in enumerate(st.archetypes):
        sel = ctr[ltr == c_id]
        m_arch = sel.mean(0)
        for j, k in enumerate(keys):
            lo, hi = np.quantile(sel[:, j], [0.1, 0.9])
            if hi - lo < 1e-6:
                gain[f"{arch}/{k}"] = {"archetype": arch, "key": k, "req_lo": float(lo), "req_hi": float(hi),
                                       "gain": None, "note": "no spread within archetype"}
                continue
            res = []
            for v in (lo, hi):
                c = np.tile(m_arch, (n3, 1)); c[:, j] = v
                knm = np.zeros((n3, K), bool); knm[:, j] = True
                _, mm = run(c, knm, np.full(n3, c_id), seed); seed += 1
                res.append(mm[:, j])
            gain[f"{arch}/{k}"] = {"archetype": arch, "key": k, "req_lo": float(lo), "req_hi": float(hi),
                                   "got_lo": float(res[0].mean()), "got_hi": float(res[1].mean()),
                                   "gain": float((res[1].mean() - res[0].mean()) / (hi - lo)),
                                   "nmae_lo": float(np.abs(res[0] - lo).mean() / std[j]),
                                   "nmae_hi": float(np.abs(res[1] - hi).mean() / std[j])}
        print(f"[stress] gain {arch}: " + " ".join(f"{k}={gain[f'{arch}/{k}']['gain']}" for k in keys), flush=True)
    del s
    _empty_cache()
    save_json("stress", {"checkpoint": step, "sampler": cfg, "levels": levels, "train_std": dict(zip(keys, std)),
                         "single": single, "pairs": pairs, "partial_nmae_by_known_count": partial,
                         "partial_unknown_prop_nmae_vs_source_map": unk, "archetype_gain": gain,
                         "notes": {"single": "only one property known, archetype unknown; others_vs_natural_z = "
                                             "(generated mean - train mean among maps near the requested value)/std",
                                   "pairs": "both properties known at the 5th/95th train percentile; train_support = "
                                            "train maps within 0.25 std of both values",
                                   "gain": "(mean achieved at p90 request - at p10 request)/(p90 - p10); 1 = perfect"}})


# ----------------------------------------------------------------------------- analysis (CPU + brief GPU NN)

def _log_rapsd(maps: np.ndarray) -> np.ndarray:
    from nullscape.metrics.quality import radial_power_spectrum

    return np.stack([np.log10(radial_power_spectrum(h)[1] + 1e-20) for h in maps])


def _w1_mean(gt: dict, rt: dict, gi: np.ndarray, ri: np.ndarray) -> float:
    from nullscape.metrics.distribution import normalized_w1

    return float(np.mean([normalized_w1(gt[k][gi], rt[k][ri]) for k in rt if k != "trav_passed"]))


def _best_dihedral(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    cands = []
    for flip in (False, True):
        y = m[:, ::-1] if flip else m
        cands += [np.rot90(y, k) for k in range(4)]
    return min(cands, key=lambda c: float(np.mean((c - q) ** 2)))


def stage_analysis(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nullscape.eval.artifacts import step_dir_name
    from nullscape.metrics.distribution import metric_table, nearest_neighbor_rmse, pooled_slope_w1
    from nullscape.metrics.quality import CONDITION_KEYS
    from nullscape.viz.render import save_grid

    st = store()
    world = st.world
    names = st.archetypes
    keys = list(CONDITION_KEYS)
    a, b = halves(st, "test")
    ref_a, ref_b = st.heights(a), st.heights(b)
    la, lb = st.labels[a], st.labels[b]
    req = st.conditions[a].astype(np.float64)
    ta, tb = metric_table(ref_a, world), metric_table(ref_b, world)
    La, Lb = _log_rapsd(ref_a), _log_rapsd(ref_b)
    gens = {s: np.load(ART / step_dir_name(s) / "generated_eval.npy") for s in CKPTS}
    tg = {s: metric_table(g, world) for s, g in gens.items()}
    Lg = {s: _log_rapsd(g) for s, g in gens.items()}
    bstd = np.array([tb[k].std() for k in keys])
    FIG.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    # 1) bootstrap CIs + paired comparison between checkpoints
    rng = np.random.default_rng(0)
    n = len(a)
    B = 300
    boot = {s: {"w1": [], "rapsd": []} for s in CKPTS}
    for _ in range(B):
        ia, ib, ig = rng.integers(0, n, n), rng.integers(0, n, n), rng.integers(0, n, n)
        fw = _w1_mean(ta, tb, ia, ib)
        fr = np.sqrt(np.mean((La[ia].mean(0) - Lb[ib].mean(0)) ** 2))
        for s in CKPTS:
            boot[s]["w1"].append(_w1_mean(tg[s], tb, ig, ib) / fw)
            boot[s]["rapsd"].append(np.sqrt(np.mean((Lg[s][ig].mean(0) - Lb[ib].mean(0)) ** 2)) / fr)
    ci = {str(s): {m: {"median": float(np.median(v)), "lo95": float(np.percentile(v, 2.5)),
                       "hi95": float(np.percentile(v, 97.5))} for m, v in d.items()} for s, d in boot.items()}
    w1 = np.array([boot[s]["w1"] for s in CKPTS])
    wins = np.bincount(w1.argmin(0), minlength=len(CKPTS)) / B
    result["bootstrap"] = {"B": B, "ci": ci, "prob_best_metric_w1": dict(zip(map(str, CKPTS), wins.tolist())),
                           "note": "A, B and generated sets resampled independently; the same B and generated "
                                   "indices are shared across checkpoints (paired, same conditions/seeds)"}

    # 2) per-archetype failure modes (every checkpoint) + leave-one-archetype-out contributions
    present = [(c, arch) for c, arch in enumerate(names) if (la == c).sum() >= 2 and (lb == c).sum() >= 2]
    arche = {}
    for s in CKPTS:
        g, t, L = gens[s], tg[s], Lg[s]
        per = {}
        for c, arch in present:
            ga, gb_ = la == c, lb == c
            ia_, ib_ = np.flatnonzero(ga), np.flatnonzero(gb_)
            fr = np.sqrt(np.mean((La[ia_].mean(0) - Lb[ib_].mean(0)) ** 2))
            mr = np.sqrt(np.mean((L[ia_].mean(0) - Lb[ib_].mean(0)) ** 2))
            sw_m = pooled_slope_w1(g[ga], ref_b[gb_], world)
            sw_f = pooled_slope_w1(ref_a[ga], ref_b[gb_], world)
            fw = _w1_mean(ta, tb, ia_, ib_)
            mw = _w1_mean(t, tb, ia_, ib_)
            per_metric = {k: float(np.nan_to_num(_w1_mean({k: t[k]}, {k: tb[k]}, ia_, ib_) /
                                                 max(_w1_mean({k: ta[k]}, {k: tb[k]}, ia_, ib_), 1e-9)))
                          for k in tb if k != "trav_passed"}
            adh = {k: float(np.abs(t[k][ga] - req[ga, j]).mean() / bstd[j]) for j, k in enumerate(keys)}
            per[arch] = {
                "n_gen": int(ga.sum()), "n_ref": int(gb_.sum()),
                "metric_w1_ratio": mw / fw, "rapsd_ratio": mr / fr, "rapsd_model": mr, "rapsd_floor": fr,
                "slope_w1_ratio": sw_m / sw_f, "slope_w1_model_deg": sw_m, "slope_w1_floor_deg": sw_f,
                "adherence_nmae": adh, "adherence_nmae_mean": float(np.mean(list(adh.values()))),
                "trav_pass_gen": float(t["trav_passed"][ga].mean()), "trav_pass_ref": float(tb["trav_passed"][gb_].mean()),
                "artifacts": {k: {"gen": float(t[k][ga].mean()), "ref": float(tb[k][gb_].mean()),
                                  "gen_over_ref": float(t[k][ga].mean() / max(tb[k][gb_].mean(), 1e-12))}
                              for k in ("checkerboard", "hf_energy", "peak_density", "sink_density")},
                "worst_metrics": sorted(per_metric, key=per_metric.get, reverse=True)[:5],
                "per_metric_ratio": per_metric}
        # leave-one-archetype-out: how much each archetype adds to the global error
        full_r = np.sqrt(np.mean((L.mean(0) - Lb.mean(0)) ** 2))
        full_s = pooled_slope_w1(g, ref_b, world)
        loo = {}
        for c, arch in present:
            ka, kb = la != c, lb != c
            loo[arch] = {
                "rapsd_delta": float(full_r - np.sqrt(np.mean((L[ka].mean(0) - Lb[kb].mean(0)) ** 2))),
                "slope_w1_delta_deg": float(full_s - pooled_slope_w1(g[ka], ref_b[kb], world)),
            }
        err = np.abs(np.stack([t[k] for k in keys], 1) - req) / bstd  # [n, K]
        fails_gen = 1 - t["trav_passed"]
        share = {}
        pnames = [arch for _, arch in present]
        excess = lambda k, cc: max(t[k][la == cc].sum() - tb[k][lb == cc].mean() * (la == cc).sum(), 0.0)  # noqa: E731
        for c, arch in present:
            ga = la == c
            share[arch] = {
                "adherence_error_share": float(err[ga].sum() / err.sum()),
                "trav_failure_share_gen": float(fails_gen[ga].sum() / max(fails_gen.sum(), 1)),
                "trav_excess_failure_rate": float(fails_gen[ga].mean() - (1 - tb["trav_passed"][lb == c]).mean()),
                **{f"{k}_excess_share": float(excess(k, c) / max(sum(excess(k, cc) for cc, _ in present), 1e-12))
                   for k in ("checkerboard", "hf_energy", "peak_density", "sink_density")}}
        rank = {
            "spectrum_error": sorted(pnames, key=lambda x: loo[x]["rapsd_delta"], reverse=True),
            "slope_error": sorted(pnames, key=lambda x: loo[x]["slope_w1_delta_deg"], reverse=True),
            "conditioning_error": sorted(pnames, key=lambda x: share[x]["adherence_error_share"], reverse=True),
            "traversability_failures": sorted(pnames, key=lambda x: share[x]["trav_excess_failure_rate"], reverse=True),
            "hf_energy_artifacts": sorted(pnames, key=lambda x: share[x]["hf_energy_excess_share"], reverse=True),
            "sink_artifacts": sorted(pnames, key=lambda x: share[x]["sink_density_excess_share"], reverse=True),
            "overall_metric_ratio": sorted(pnames, key=lambda x: per[x]["metric_w1_ratio"], reverse=True),
        }
        arche[str(s)] = {"per_archetype": per, "leave_one_out": loo, "shares": share, "rankings": rank}
    result["archetypes"] = arche

    # 3) adherence by requested-value tercile (bias shows regression toward the mean)
    adh_bins = {}
    for s in CKPTS:
        t = tg[s]
        out = {}
        for j, k in enumerate(keys):
            edges = np.quantile(req[:, j], [0, 1 / 3, 2 / 3, 1])
            if edges[1] - edges[0] < 1e-9:  # water_fraction: most requests are 0
                bins = [req[:, j] <= 1e-9, (req[:, j] > 1e-9) & (req[:, j] <= np.quantile(req[req[:, j] > 1e-9, j], 0.5)),
                        req[:, j] > np.quantile(req[req[:, j] > 1e-9, j], 0.5)]
                labels_ = ["zero", "low_nonzero", "high_nonzero"]
            else:
                bins = [(req[:, j] >= edges[0]) & (req[:, j] <= edges[1]), (req[:, j] > edges[1]) & (req[:, j] <= edges[2]),
                        req[:, j] > edges[2]]
                labels_ = ["lower", "middle", "upper"]
            out[k] = {lab: {"n": int(m.sum()), "nmae": float(np.abs(t[k][m] - req[m, j]).mean() / bstd[j]),
                            "bias_std": float((t[k][m] - req[m, j]).mean() / bstd[j]),
                            "req_range": [float(req[m, j].min()), float(req[m, j].max())]}
                      for lab, m in zip(labels_, bins) if m.any()}
        adh_bins[str(s)] = out
    result["adherence_bins"] = adh_bins

    # 4) memorization deep dive for the selected checkpoint (+ summary for all from the reports)
    sel = load_json("selection")["checkpoint"]
    bank = st.heights(st.split("train"))
    gnn, gidx = nearest_neighbor_rmse(gens[sel], bank, device=DEVICE)
    hnn, _ = nearest_neighbor_rmse(ref_a, bank, device=DEVICE)
    pct = lambda x: {f"p{p:g}": float(np.percentile(x, p)) for p in (0.1, 1, 5, 10, 50, 90)}  # noqa: E731
    per_arch_nn = {}
    rel = np.zeros(len(gnn))
    for c, arch in enumerate(names):
        m = la == c
        if not m.any():
            continue
        per_arch_nn[arch] = {"gen_median": float(np.median(gnn[m])), "heldout_median": float(np.median(hnn[m])),
                             "ratio": float(np.median(gnn[m]) / np.median(hnn[m])),
                             "frac_gen_below_heldout_p1": float((gnn[m] < np.percentile(hnn[m], 1)).mean())}
        rel[m] = np.searchsorted(np.sort(hnn[m]), gnn[m]) / m.sum()  # percentile within same-archetype held-out
    order = np.argsort(rel + gnn * 1e-6)[:12]
    hs, ts = [], []
    for i in order:
        match = _best_dihedral(gens[sel][i], bank[gidx[i]])
        hs += [gens[sel][i], match, np.clip(0.5 + (gens[sel][i] - match) * 5, 0, 1)]
        ts += [f"{names[la[i]]} gen rmse={gnn[i]:.4f}", f"train #{st.split('train')[gidx[i]]} (aligned)",
               f"diff x5 (held-out pct {rel[i] * 100:.1f})"]
    save_grid(hs, world, FIG / "memorization_most_suspicious.png", titles=ts, ncols=6,
              suptitle=f"step {sel}: 12 generated maps closest to training data relative to their archetype's "
                       f"held-out NN distribution")
    result["memorization"] = {
        "checkpoint": sel, "gen": pct(gnn), "heldout": pct(hnn), "nn_median_ratio": float(np.median(gnn) / np.median(hnn)),
        "frac_gen_below_heldout_p1": float((gnn < np.percentile(hnn, 1)).mean()),
        "frac_gen_below_heldout_min": float((gnn < hnn.min()).mean()), "gen_min": float(gnn.min()),
        "heldout_min": float(hnn.min()), "per_archetype": per_arch_nn,
        "most_suspicious": [{"gen_index": int(i), "archetype": names[la[i]], "nn_rmse": float(gnn[i]),
                             "nn_rmse_m": float(gnn[i] * world.max_height_m),
                             "train_index": int(st.split("train")[gidx[i]]), "heldout_percentile": float(rel[i])}
                            for i in order],
        "all_checkpoints": {str(s): json.loads((ART / step_dir_name(s) / "report.json").read_text())["modes"]
                            ["conditional"]["memorization"] for s in CKPTS}}
    fig, ax = plt.subplots(figsize=(6, 4))
    bins_ = np.linspace(0, np.percentile(np.concatenate([gnn, hnn]), 99.5), 60)
    ax.hist(hnn, bins=bins_, histtype="step", density=True, label="held-out test -> train")
    ax.hist(gnn, bins=bins_, histtype="step", density=True, label=f"generated ({sel // 1000}k) -> train")
    ax.set_xlabel("nearest-training-neighbour RMSE (normalized height, 8 rotations/flips)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "memorization_nn_hist.png", dpi=150)
    plt.close(fig)

    # 5) sensitivity: reference maps are uint16-quantized, generated maps are float. Re-score the selected
    #    checkpoint after quantizing its output (what an engine export would contain).
    from nullscape.data.storage import dequantize, quantize
    from nullscape.eval.core import evaluate_generated

    gq = dequantize(quantize(gens[sel]))
    rep_q = evaluate_generated(gq, ref_a, ref_b, world, requested_conds=req,
                               tables={"a": ta, "b": tb, "gen": metric_table(gq, world)})
    rep_f = json.loads((ART / step_dir_name(sel) / "report.json").read_text())["modes"]["conditional"]
    result["quantization_sensitivity"] = {
        "checkpoint": sel, "float": rep_f["ratio_to_floor"], "uint16": rep_q["ratio_to_floor"],
        "sink_density_gen_float": rep_f["artifacts"]["sink_density_mean_gen"],
        "sink_density_gen_uint16": rep_q["artifacts"]["sink_density_mean_gen"],
        "sink_density_ref": rep_f["artifacts"]["sink_density_mean_ref"],
        "peak_density_gen_float": rep_f["metric_means"]["peak_density"]["gen"],
        "peak_density_gen_uint16": rep_q["metric_means"]["peak_density"]["gen"],
        "peak_density_ref": rep_f["metric_means"]["peak_density"]["ref"]}

    # 6) where is the spectrum wrong? mean log10-power difference per radial frequency band
    kk = np.arange(1, La.shape[1] + 1)
    bands = {"k1-4 (map-scale shape, >1 km)": (1, 4), "k5-12 (valleys/ridges, 340-820 m)": (5, 12),
             "k13-24 (hillslopes, 170-315 m)": (13, 24), "k25-32 (finest, 128-165 m)": (25, 32)}
    spec = {}
    for s in CKPTS:
        d = Lg[s].mean(0) - Lb.mean(0)
        spec[str(s)] = {"per_k_diff_decades": d.tolist(),
                        "bands": {bn: {"mean_diff_decades": float(d[(kk >= lo) & (kk <= hi)].mean()),
                                       "rms_diff_decades": float(np.sqrt((d[(kk >= lo) & (kk <= hi)] ** 2).mean()))}
                                  for bn, (lo, hi) in bands.items()},
                        "per_archetype_bands": {arch: {bn: float((Lg[s][la == c].mean(0) - Lb[lb == c].mean(0))
                                                                  [(kk >= lo) & (kk <= hi)].mean())
                                                       for bn, (lo, hi) in bands.items()}
                                                for c, arch in present}}
    df = La.mean(0) - Lb.mean(0)
    spec["floor_A_minus_B"] = {"per_k_diff_decades": df.tolist(),
                               "bands": {bn: {"mean_diff_decades": float(df[(kk >= lo) & (kk <= hi)].mean())}
                                         for bn, (lo, hi) in bands.items()}}
    result["spectrum_bands"] = spec
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for s in CKPTS:
        axes[0].plot(kk, Lg[s].mean(0) - Lb.mean(0), label=f"{s // 1000}k")
    axes[0].plot(kk, df, "k--", lw=1, label="floor (A - B)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("radial frequency k (cycles / 4.1 km map)")
    axes[0].set_ylabel("mean log10 power: learned - procedural")
    axes[0].set_title("Spectrum error by frequency (TEST, all checkpoints)", fontsize=9)
    axes[0].legend(fontsize=7)
    for c, arch in present:
        axes[1].plot(kk, Lg[sel][la == c].mean(0) - Lb[lb == c].mean(0), label=arch)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("radial frequency k")
    axes[1].set_title(f"Spectrum error by archetype (step {sel})", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "spectrum_error_by_frequency.png", dpi=150)
    plt.close(fig)

    # 7) checkpoint trend figure with bootstrap CIs
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    xs = list(CKPTS)
    for ax, m, title in ((axes[0], "w1", "metric_w1_mean / floor"), (axes[1], "rapsd", "RAPSD distance / floor")):
        med = [ci[str(s)][m]["median"] for s in xs]
        lo = [ci[str(s)][m]["lo95"] for s in xs]
        hi = [ci[str(s)][m]["hi95"] for s in xs]
        ax.errorbar(xs, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], marker="o", capsize=3)
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_title(f"TEST {title} (95% bootstrap CI)", fontsize=9)
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(FIG / "checkpoints_test_ci.png", dpi=150)
    plt.close(fig)
    save_json("analysis", result)


def stage_figures(args) -> None:
    """Sampler Pareto plot, sampler fixed-seed grid, stress-test figures, archetype heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nullscape.metrics.quality import CONDITION_KEYS
    from nullscape.viz.render import _save, draw_heightmap

    st = store()
    FIG.mkdir(parents=True, exist_ok=True)
    keys = list(CONDITION_KEYS)
    if (RAW / "pareto.json").exists():
        p = load_json("pareto")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for pt in p["points"]:
            on = pt["name"] in p["frontier"]
            ax.scatter(pt["iso_seconds_per_map_b128"] * 1000, pt["val_metric_w1_ratio"], c="C3" if on else "C0",
                       s=30 if on else 15, zorder=3 if on else 2)
            ax.annotate(pt["name"], (pt["iso_seconds_per_map_b128"] * 1000, pt["val_metric_w1_ratio"]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
        fr = sorted((q for q in p["points"] if q["name"] in p["frontier"]), key=lambda q: q["iso_seconds_per_map_b128"])
        ax.plot([q["iso_seconds_per_map_b128"] * 1000 for q in fr], [q["val_metric_w1_ratio"] for q in fr], "C3-", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("isolated ms per map (batch 128, RTX 5060)")
        ax.set_ylabel("VAL metric_w1_mean / noise floor")
        ax.set_title(f"Sampler quality/speed (red = Pareto frontier; recommended: {p['recommended']})", fontsize=9)
        fig.tight_layout()
        _save(fig, FIG / "sampler_pareto.png")
    if (GEN / "sampler_val_fixed.npz").exists():
        z = np.load(GEN / "sampler_val_fixed.npz")
        fx = z["indices"]
        cfgs = [k for k in z.files if k != "indices" and k.endswith("_g1.5_e0")]
        rows = [("procedural", st.heights(fx))] + [(k, z[k]) for k in cfgs]
        fig, axes = plt.subplots(len(rows), len(fx), figsize=(1.3 * len(fx), 1.4 * len(rows)), squeeze=False)
        for r, (lab, maps) in enumerate(rows):
            for c in range(len(fx)):
                draw_heightmap(axes[r, c], maps[c], st.world, st.archetypes[st.labels[fx[c]]] if r == 0 else None)
            axes[r, 0].text(-0.1, 0.5, lab, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6)
        fig.suptitle("Sampler settings on the same fixed seeds/conditions (VAL fixed set, guidance 1.5, eta 0)", fontsize=9)
        _save(fig, FIG / "sampler_fixed_seed_grid.png")
    if (RAW / "stress.json").exists():
        sres = load_json("stress")
        fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), squeeze=False)
        for ax, k in zip(axes[0], keys):
            for lvl in ("low", "mid", "high"):
                c = sres["single"][f"{k}/{lvl}"]
                ax.errorbar([c["requested"]], [c["measured_mean"]], yerr=[[c["measured_std"]], [c["measured_std"]]],
                            fmt="o", capsize=3)
            lo = min(sres["single"][f"{k}/{l}"]["requested"] for l in ("low", "mid", "high"))
            hi = max(sres["single"][f"{k}/{l}"]["requested"] for l in ("low", "mid", "high"))
            pad = 0.15 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
            ax.set_title(f"{k} alone (p5 / p50 / p95)", fontsize=8)
            ax.set_xlabel("requested", fontsize=7)
            ax.set_ylabel("measured (mean ± std)", fontsize=7)
        fig.tight_layout()
        _save(fig, FIG / "stress_single_property.png")
        K = len(keys)
        mat = np.full((K, K), np.nan)
        for v in sres["pairs"].values():
            if v["train_support"] == 0:  # combination never occurs in training data: reported separately
                continue
            i, j = keys.index(v["a"]), keys.index(v["b"])
            mat[i, j] = np.nanmax([mat[i, j], v["a_nmae"] - v["a_nmae_alone"]])
            mat[j, i] = np.nanmax([mat[j, i], v["b_nmae"] - v["b_nmae_alone"]])
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        lim = max(0.5, float(np.nanmax(np.abs(mat)))) if np.isfinite(mat).any() else 0.5
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(K), keys, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(K), keys, fontsize=7)
        for i in range(K):
            for j in range(K):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="worst extra nMAE of row when paired with col (std units)")
        ax.set_title("Conditioning interference (p5/p95 pairs seen in training data)", fontsize=9)
        fig.tight_layout()
        _save(fig, FIG / "stress_interference.png")
        gains = np.array([[sres["archetype_gain"][f"{a}/{k}"]["gain"] if sres["archetype_gain"][f"{a}/{k}"]["gain"]
                           is not None else np.nan for k in keys] for a in st.archetypes])
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(gains, cmap="RdYlGn", vmin=0, vmax=1.2)
        ax.set_xticks(range(K), keys, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(st.archetypes)), st.archetypes, fontsize=7)
        for i in range(gains.shape[0]):
            for j in range(K):
                ax.text(j, i, "n/a" if np.isnan(gains[i, j]) else f"{gains[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="control gain (1 = perfect)")
        ax.set_title("Within-archetype control: p10 vs p90 request, label known", fontsize=9)
        fig.tight_layout()
        _save(fig, FIG / "stress_archetype_gain.png")
    if (RAW / "analysis.json").exists():
        an = load_json("analysis")
        sel = str(load_json("selection")["checkpoint"])
        per = an["archetypes"][sel]["per_archetype"]
        cols = ["metric_w1_ratio", "rapsd_ratio", "slope_w1_ratio", "adherence_nmae_mean"]
        arch_rows = [a for a in st.archetypes if a in per]
        mat = np.array([[per[a][c] for c in cols] for a in arch_rows])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        im = ax.imshow(mat, cmap="magma_r")
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(arch_rows)), arch_rows, fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Per-archetype failure metrics, TEST, step {sel}", fontsize=9)
        fig.tight_layout()
        _save(fig, FIG / "archetype_failure_heatmap.png")


OFFICIAL_ART = OUT / "artifacts_official"


def stage_official(args) -> None:
    """Full artifact suite on TEST for the official v1 configuration (selected checkpoint + recommended sampler)."""
    from nullscape.eval.artifacts import ArtifactSpec, generate_checkpoint_artifacts

    st = store()
    step = load_json("selection")["checkpoint"]
    cfg = load_json("pareto")["recommended_config"]
    spec = ArtifactSpec(split="test", n_eval=N_HALF, batch_size=BATCH, steps=3 if DRY else cfg["steps"],
                        spacing=cfg["spacing"], eta=cfg["eta"], guidance=cfg["guidance"])
    s = sampler_for(step)
    t0 = time.time()
    d = generate_checkpoint_artifacts(s, st, OFFICIAL_ART, step, spec, train_bank=st.heights(st.split("train")),
                                      memorization_device=DEVICE, save_generated=True)
    rep = json.loads((d / "report.json").read_text())["modes"]["conditional"]
    save_json("official", {"checkpoint": step, "sampler": cfg, "spec": asdict(spec), "wall_seconds": time.time() - t0,
                           "artifacts_dir": str(d.relative_to(ROOT)), "compact": compact(rep)})


def _spectral_noise(beta: float, mean: float, std: float, n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2) * n
    amp = np.where(f > 0, np.maximum(f, 1e-9) ** (-beta / 2), 0.0)
    field = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    field = (field - field.mean()) / (field.std() + 1e-12)
    return np.clip(mean + std * field, 0.0, 1.0)


def stage_baselines(args) -> None:
    """Non-learned reference generators scored with the same TEST protocol (CPU only).

    blur            each A map Gaussian-blurred (sigma 1.5 cells): right layout, missing detail
    spectral_noise  random-phase noise with A's spectral_beta, mean and std: right roughness, no structure
    retrieval       the training map (same archetype) whose conditions are closest to A's: a perfect
                    memorizer (real terrain, good control, zero originality)
    """
    from scipy.ndimage import gaussian_filter

    from nullscape.eval.core import evaluate_generated
    from nullscape.metrics.distribution import metric_table, nearest_neighbor_rmse

    st = store()
    a, b = halves(st, "test")
    ref_a, ref_b = st.heights(a), st.heights(b)
    tables = {"a": metric_table(ref_a, st.world), "b": metric_table(ref_b, st.world)}
    req = st.conditions[a]
    keys = st.condition_keys
    rng = np.random.default_rng(99)
    tr = st.split("train")
    z = (st.conditions - st.condition_mean) / st.condition_std
    gens = {"blur": np.stack([gaussian_filter(h, 1.5, mode="nearest") for h in ref_a])}
    j_beta = keys.index("spectral_beta")
    gens["spectral_noise"] = np.stack([_spectral_noise(float(req[i, j_beta]), float(h.mean()), float(h.std()),
                                                       st.world.resolution, rng) for i, h in enumerate(ref_a)])
    pick = []
    for i in a:
        cand = tr[st.labels[tr] == st.labels[i]]
        pick.append(cand[np.argmin(((z[cand] - z[i]) ** 2).sum(1))])
    gens["retrieval"] = st.heights(np.array(pick))
    bank = st.heights(tr)
    n_nn = 200  # NN search on CPU; a subsample is enough to separate copying from generating
    held_nn, _ = nearest_neighbor_rmse(ref_a[:n_nn], bank, device="cpu")
    out = {}
    for name, g in gens.items():
        rep = evaluate_generated(g, ref_a, ref_b, st.world, requested_conds=req, gen_labels=st.labels[a],
                                 ref_a_labels=st.labels[a], ref_b_labels=st.labels[b], archetype_names=st.archetypes,
                                 tables={**tables, "gen": metric_table(g, st.world)})
        gnn, _ = nearest_neighbor_rmse(g[:n_nn], bank, device="cpu")
        rep["memorization"] = {"gen_nn_rmse_median": float(np.median(gnn)),
                               "heldout_nn_rmse_median": float(np.median(held_nn)),
                               "nn_median_ratio": float(np.median(gnn) / np.median(held_nn)),
                               "frac_gen_below_heldout_p01": float((gnn < np.percentile(held_nn, 1)).mean()),
                               "n": n_nn}
        out[name] = {"compact": compact(rep)}
        print(f"[baselines] {name}: ratio {rep['ratio_to_floor']}", flush=True)
    save_json("baselines", {"n_per_half": len(a), "split": "test", "results": out,
                            "definitions": stage_baselines.__doc__})


def stage_training(args) -> None:
    out = {}
    for name, run in (("main_0_30k", MAIN.parent), ("cont_30k_40k", CONT.parent)):
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        its = np.array([r["it_per_s"] for r in rows if "it_per_s" in r])
        steady = its[its > 0.8 * np.median(its)]
        losses = [(r["step"], r["loss"]) for r in rows if "loss" in r]
        vals = [(r["step"], r["val_loss_ema"]) for r in rows if "val_loss_ema" in r]
        cfg = (run / "config.yaml").read_text()
        out[name] = {"intervals": int(len(its)), "it_per_s_median": float(np.median(its)),
                     "it_per_s_steady_median": float(np.median(steady)), "it_per_s_p10": float(np.percentile(its, 10)),
                     "it_per_s_p90": float(np.percentile(its, 90)), "batch_size": 48,
                     "maps_per_s_steady": float(np.median(steady) * 48),
                     "slow_intervals_fraction": float((its <= 0.8 * np.median(its)).mean()),
                     "wall_hours_first_to_last_log": float((rows[-1]["time"] - rows[0]["time"]) / 3600),
                     "final_loss": losses[-1], "val_loss_ema": vals, "config_excerpt_batch48": "batch_size: 48" in cfg}
    save_json("training", out)


def stage_aggregate(args) -> None:
    res = {"benchmark": "NULLSCAPE v1", "version": 1}
    for name in ("provenance", "checkpoints_test", "checkpoints_val", "selection", "sampler_val", "perf", "pareto",
                 "sampler_test", "stress", "analysis", "training", "engineering", "baselines", "quantization",
                 "official", "ram"):
        if (RAW / f"{name}.json").exists():
            d = load_json(name)
            if name == "checkpoints_val":  # rebuild with the current compact() from the stored full reports
                d = {"n_per_half": d["n_per_half"], "sampler": d["sampler"],
                     "compact": {k: compact(v["report"]) for k, v in d["results"].items()}}
            if name == "checkpoints_test":
                from nullscape.eval.artifacts import step_dir_name

                d["compact"] = {str(s): compact(json.loads((ART / step_dir_name(s) / "report.json").read_text())
                                                ["modes"]["conditional"]) for s in CKPTS}
            res[name] = d
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    print(f"[aggregate] -> {OUT / 'results.json'}")


STAGES = {
    "provenance": stage_provenance, "checkpoints_test": stage_checkpoints_test, "checkpoints_val": stage_checkpoints_val,
    "select": stage_select, "sampler_val": stage_sampler_val, "perf": stage_perf, "pareto": stage_pareto,
    "sampler_test": stage_sampler_test, "stress": stage_stress, "analysis": stage_analysis, "figures": stage_figures,
    "training": stage_training, "aggregate": stage_aggregate, "baselines": stage_baselines,
    "official": stage_official,
}
OUTPUTS = {"select": "selection"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True, help="comma-separated: " + ",".join(STAGES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name in args.stages.split(","):
        out = RAW / f"{OUTPUTS.get(name, name)}.json"
        if out.exists() and not args.force and name not in ("figures", "aggregate"):
            print(f"[skip] {name} (exists)")
            continue
        t0 = time.time()
        print(f"[stage] {name} ...", flush=True)
        STAGES[name](args)
        print(f"[stage] {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
