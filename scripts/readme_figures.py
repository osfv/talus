"""README figures from a released checkpoint: one row of samples per terrain type, plus three 3D views.

usage: python scripts/readme_figures.py --checkpoint talus-3 --out docs/figures
Seeds are fixed, so the figures are reproducible with the same checkpoint and sampler preset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nullscape.inference.sampler import TerrainSampler
from nullscape.viz.render import _save, draw_surface, save_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="talus-3")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--per-type", type=int, default=6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sampler = TerrainSampler.from_checkpoint(args.checkpoint, device=args.device)
    name = sampler.checkpoint_info.get("release", "checkpoint").split()[0].lower()
    maps, titles = [], []
    for k, archetype in enumerate(sampler.archetypes):
        maps.extend(sampler.sample(n=args.per_type, seed=100 + k, archetype=archetype))
        titles.extend([archetype] * args.per_type)
    out = Path(args.out)
    print(save_grid(maps, sampler.world, out / f"{name}_samples.png", titles=titles, ncols=args.per_type))

    picks = [sampler.archetypes.index(a) * args.per_type for a in ("mountains", "ridges", "islands")]
    world, exaggeration = sampler.world, 2.0
    fig = plt.figure(figsize=(15, 4.2))
    for i, j in enumerate(picks):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        draw_surface(ax, maps[j], world, vertical_exaggeration=exaggeration)
        ax.set_box_aspect((1, 1, exaggeration * world.max_height_m / world.extent_m), zoom=1.45)
        ax.set_title(titles[j], fontsize=11, y=0.95)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    print(_save(fig, out / f"{name}_3d.png"))


if __name__ == "__main__":
    main()
