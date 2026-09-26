"""Generate one map per archetype from several checkpoints with identical seeds, side by side.

usage: python scripts/compare_samples.py --ckpt v1=PATH --ckpt exp1=PATH [--seed 7] [--out reports/samples]
Uses the official v1 sampler (50-step quadratic DDIM, guidance 2.0) unless overridden.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nullscape.inference.sampler import TerrainSampler
from nullscape.viz.render import save_grid, save_inspection, save_surface


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="label=path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--spacing", default="quadratic")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out", default="reports/samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, titles, world, archetypes = [], [], None, None
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        s = TerrainSampler.from_checkpoint(path)
        world, archetypes = s.world, s.archetypes
        for arch in archetypes:
            h = s.sample(n=1, seed=args.seed, archetype=arch, steps=args.steps, spacing=args.spacing,
                         guidance=args.guidance, eta=0.0)[0]
            rows.append(h)
            titles.append(f"{label}: {arch}")
            np.save(out / f"{label}_{arch}_seed{args.seed}.npy", h)
        last = label
    save_grid(rows, world, out / f"compare_seed{args.seed}.png", titles=titles, ncols=len(archetypes),
              suptitle=f"same seed ({args.seed}) and terrain type per column; {args.steps}-step {args.spacing}, "
                       f"guidance {args.guidance}")
    for arch in ("mountains", "islands"):
        h = np.load(out / f"{last}_{arch}_seed{args.seed}.npy")
        save_surface(h, world, out / f"{last}_{arch}_3d.png", title=f"{last}: {arch} (seed {args.seed})")
        save_inspection(h, world, out / f"{last}_{arch}_inspect.png", title=f"{last}: {arch} (seed {args.seed})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
