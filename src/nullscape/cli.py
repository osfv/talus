"""Command-line entry point: ``nullscape <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cmd_gen_dataset(args: argparse.Namespace) -> None:
    from nullscape.data.build import build_dataset
    from nullscape.utils.config import load_config

    cfg = load_config(args.config, args.set)
    root = build_dataset(cfg, workers=args.workers, overwrite=args.overwrite)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({k: manifest[k] for k in ("name", "n", "splits", "archetype_counts", "summary", "generation_seconds")}, indent=2))
    print(f"dataset written to {root}")


def _cmd_viz(args: argparse.Namespace) -> None:
    from nullscape.data.storage import TerrainStore
    from nullscape.viz.render import save_grid, save_inspection, save_traversability_grid

    store = TerrainStore.open(args.dataset)
    out = Path(args.out or Path("reports") / "dataset" / store.root.name)
    idx = store.split(args.split)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(idx, size=min(args.n, len(idx)), replace=False))
    labels = store.labels
    names = store.archetypes
    heights = store.heights(pick)
    save_grid(heights, store.world, out / "mixed.png", titles=[f"#{i} {names[labels[i]]}" for i in pick],
              suptitle=f"{store.root.name}/{args.split}")
    for k, name in enumerate(names):
        sel = idx[labels[idx] == k][: args.per_archetype]
        if len(sel):
            save_grid(store.heights(sel), store.world, out / f"archetype_{name}.png", titles=[f"#{i}" for i in sel],
                      suptitle=name)
    save_traversability_grid(heights[:12], store.world, out / "traversability.png")
    for i in pick[: args.inspect]:
        save_inspection(store.heights(np.array([i]))[0], store.world, out / f"inspect_{i}.png",
                        title=f"#{i} {names[labels[i]]}")
    print(f"figures written to {out}")


def _cmd_export(args: argparse.Namespace) -> None:
    from nullscape.data.storage import TerrainStore
    from nullscape.export.engine import export_heightmap, unity_size

    store = TerrainStore.open(args.dataset)
    h = store.heights(np.array([args.index]))[0]
    size = unity_size(h.shape[0]) if args.unity else args.size
    meta = {"dataset": store.root.name, "index": args.index, "generator_version": store.manifest["generator_version"],
            "archetype": store.archetypes[int(store.labels[args.index])]}
    written = export_heightmap(h, store.world, args.out, f"{store.root.name}_{args.index}", args.formats.split(","),
                               meta, size)
    for k, v in written.items():
        print(f"{k:6s} {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nullscape", description="NULLSCAPE learned terrain toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-dataset", help="generate a procedural heightmap dataset")
    g.add_argument("--config", required=True)
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override config values")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--overwrite", action="store_true", help="delete and rebuild an existing dataset")
    g.set_defaults(func=_cmd_gen_dataset)

    v = sub.add_parser("viz", help="render dataset figures")
    v.add_argument("--dataset", required=True)
    v.add_argument("--split", default="val")
    v.add_argument("--n", type=int, default=36)
    v.add_argument("--per-archetype", type=int, default=12)
    v.add_argument("--inspect", type=int, default=4)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", default=None)
    v.set_defaults(func=_cmd_viz)

    e = sub.add_parser("export", help="export one dataset map for a game engine")
    e.add_argument("--dataset", required=True)
    e.add_argument("--index", type=int, required=True)
    e.add_argument("--formats", default="png16,r16,npy,obj")
    e.add_argument("--out", default="exports")
    e.add_argument("--size", type=int, default=None, help="resample to this size first")
    e.add_argument("--unity", action="store_true", help="resample to the next 2^n+1 size")
    e.set_defaults(func=_cmd_export)

    try:
        from nullscape.cli_model import add_model_commands

        add_model_commands(sub)
    except ImportError as exc:  # model stack not available
        if "nullscape.cli_model" not in str(exc):
            raise
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
