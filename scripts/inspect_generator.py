"""Quick generator calibration: per-archetype metric means + preview grid.

usage: python scripts/inspect_generator.py [--n 12] [--seed 11] [--out reports/dev/archetypes.png]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from nullscape.metrics import analyze, compute_metrics
from nullscape.terrain import ARCHETYPES, GeneratorConfig, generate
from nullscape.viz.render import save_grid
from nullscape.world import WorldSpec

KEYS = ["mean_elevation", "max_elevation", "relief", "mean_slope_deg", "p99_slope_deg", "spectral_beta",
        "water_fraction", "sink_density", "hf_energy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--out", default="reports/dev/archetypes.png")
    args = ap.parse_args()
    world = WorldSpec(resolution=args.resolution)
    cfg = GeneratorConfig(blend_probability=0.0)
    print("archetype ".ljust(11) + " ".join(k[:10].rjust(10) for k in KEYS) + "  pass  clip  sec")
    grid, titles = [], []
    for a in ARCHETYPES:
        rows, passed, clipped, secs = [], [], [], []
        for i in range(args.n):
            t = time.time()
            h, rec = generate(i, args.seed, world, cfg, archetype=a)
            secs.append(time.time() - t)
            m = compute_metrics(h, world)
            rows.append([m[k] for k in KEYS])
            passed.append(analyze(h, world).passed)
            clipped.append(rec["clipped_fraction"])
            if i < 6:
                grid.append(h)
                titles.append(f"{a} r={m['relief'] * world.max_height_m:.0f}m b={m['spectral_beta']:.1f}")
        print(a.ljust(11) + " ".join(f"{v:10.3f}" for v in np.mean(rows, 0))
              + f"  {np.mean(passed):.2f}  {np.mean(clipped):.3f} {np.mean(secs):.2f}")
    save_grid(grid, world, args.out, titles=titles, ncols=6)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
