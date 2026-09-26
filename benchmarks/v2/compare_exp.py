"""Score v2 experiment checkpoints against frozen v1 on VAL with the official v1 sampler.

usage: python benchmarks/v2/compare_exp.py --name exp1 --ckpt runs/<run>/checkpoints/last.pt [--n 500]
Writes benchmarks/v2/<name>_val.json. Same A/B halves, conditions and noise seeds for every model, so
differences come from the weights only. Reports the headline ratios plus the hypothesis-specific
measures: spectrum error per frequency band and archetype, and micro-relief (sinks/peaks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "v1"))
import run_v1_benchmark as B  # noqa: E402

V1 = ROOT / "runs" / "20260925-164759_diffusion64" / "checkpoints" / "step_0030000.pt"
BANDS = {"k1-4": (1, 4), "k5-12": (5, 12), "k13-24": (13, 24), "k25-32": (25, 32)}
RUGGED = ("mountains", "ridges", "mesas")


def band_errors(Lg, Lb, lg, lb, names):
    kk = np.arange(1, Lg.shape[1] + 1)
    out = {}
    for c, arch in enumerate(names):
        d = Lg[lg == c].mean(0) - Lb[lb == c].mean(0)
        out[arch] = {b: float(d[(kk >= lo) & (kk <= hi)].mean()) for b, (lo, hi) in BANDS.items()}
    d = Lg.mean(0) - Lb.mean(0)
    out["all"] = {b: float(d[(kk >= lo) & (kk <= hi)].mean()) for b, (lo, hi) in BANDS.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="one or more checkpoints (label=path or path)")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    from nullscape.eval.core import evaluate_generated
    from nullscape.inference.sampler import TerrainSampler
    from nullscape.metrics.distribution import metric_table

    st = B.store()
    cfg = B.load_json("pareto")["recommended_config"]
    a, b = B.halves(st, "val", args.n)
    ref_a, ref_b = st.heights(a), st.heights(b)
    tables = {"a": metric_table(ref_a, st.world), "b": metric_table(ref_b, st.world)}
    Lb = B._log_rapsd(ref_b)
    la, lb = st.labels[a], st.labels[b]
    models = [("v1 30k", str(V1))] + [tuple(c.split("=", 1)) if "=" in c else (Path(c).stem, c) for c in args.ckpt]
    out = {"split": "val", "n_per_half": len(a), "sampler": cfg, "models": {}}
    for label, path in models:
        s = TerrainSampler.from_checkpoint(path, device=B.DEVICE)
        gen, secs = B.conditional_gen(s, st, a, cfg, seed=2027)
        gt = metric_table(gen, st.world)
        rep = evaluate_generated(gen, ref_a, ref_b, st.world, requested_conds=st.conditions[a], gen_labels=la,
                                 ref_a_labels=la, ref_b_labels=lb, archetype_names=st.archetypes,
                                 tables={**tables, "gen": gt})
        bands = band_errors(B._log_rapsd(gen), Lb, la, lb, st.archetypes)
        c = B.compact(rep)
        out["models"][label] = {"path": path, "compact": c, "bands": bands,
                                "rugged_k13_24": float(np.mean([bands[x]["k13-24"] for x in RUGGED])),
                                "rugged_k25_32": float(np.mean([bands[x]["k25-32"] for x in RUGGED])),
                                "plains_k25_32": bands["plains"]["k25-32"]}
        del s
        B._empty_cache()
        print(f"[{label}] done", flush=True)
    (ROOT / "benchmarks" / "v2").mkdir(parents=True, exist_ok=True)
    (ROOT / "benchmarks" / "v2" / f"{args.name}_val.json").write_text(json.dumps(out, indent=2, default=B._json_default))

    keys = [("metric W1 ratio", lambda m: m["compact"]["ratio"]["metric_w1_mean"]),
            ("RAPSD ratio", lambda m: m["compact"]["ratio"]["rapsd_distance"]),
            ("height W1 ratio", lambda m: m["compact"]["ratio"]["height_w1"]),
            ("slope W1 ratio", lambda m: m["compact"]["ratio"]["slope_w1_deg"]),
            ("adherence nMAE", lambda m: m["compact"]["adherence_nmae_mean"]),
            ("sink density (ref %.2f)" % out["models"]["v1 30k"]["compact"]["sink_density"]["ref"],
             lambda m: m["compact"]["sink_density"]["gen"]),
            ("peak density (ref %.2f)" % out["models"]["v1 30k"]["compact"]["peak_density"]["ref"],
             lambda m: m["compact"]["peak_density"]["gen"]),
            ("rugged k13-24 (0 = perfect)", lambda m: m["rugged_k13_24"]),
            ("rugged k25-32 (0 = perfect)", lambda m: m["rugged_k25_32"]),
            ("plains k25-32 (0 = perfect)", lambda m: m["plains_k25_32"])]
    labels = list(out["models"])
    print("| measure | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for name, fn in keys:
        print(f"| {name} | " + " | ".join(f"{fn(out['models'][l]):.3f}" for l in labels) + " |")


if __name__ == "__main__":
    main()
