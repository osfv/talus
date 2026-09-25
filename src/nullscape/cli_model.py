"""Model-related CLI commands: train, sample, evaluate, compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_props(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        if not v:
            raise SystemExit(f"--prop expects key=value, got {item!r}")
        out[k.strip()] = float(v)
    return out


def _cmd_train(args: argparse.Namespace) -> None:
    from nullscape.train.trainer import train
    from nullscape.utils.config import load_config

    path = train(load_config(args.config, args.set), resume=args.resume)
    print(f"finished: {path}")


def _cmd_sample(args: argparse.Namespace) -> None:
    from nullscape.export.engine import export_heightmap, unity_size
    from nullscape.inference.sampler import TerrainSampler
    from nullscape.metrics.quality import compute_metrics
    from nullscape.viz.render import save_grid, save_inspection

    sampler = TerrainSampler.from_checkpoint(args.checkpoint, use_ema=not args.raw_weights)
    props = _parse_props(args.prop)
    maps = sampler.sample(n=args.n, seed=args.seed, properties=props, archetype=args.archetype,
                          guidance=args.guidance, steps=args.steps, eta=args.eta, spacing=args.spacing)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f for f in args.formats.split(",") if f]
    size = unity_size(sampler.world.resolution) if args.unity else None
    summary = []
    for i, h in enumerate(maps):
        m = compute_metrics(h, sampler.world)
        meta = {"checkpoint": sampler.checkpoint_info, "seed": args.seed, "sample_index": i,
                "archetype": args.archetype, "properties": props, "guidance": args.guidance, "steps": args.steps,
                "eta": args.eta, "spacing": args.spacing, "measured": {k: m[k] for k in sampler.condition_keys}}
        if formats:
            export_heightmap(h, sampler.world, out, f"sample_{args.seed}_{i:03d}", formats, meta, size)
        summary.append(meta)
    save_grid(maps, sampler.world, out / "grid.png", titles=[f"seed {args.seed} #{i}" for i in range(len(maps))])
    save_inspection(maps[0], sampler.world, out / "inspect_0.png", title="sample 0")
    (out / "samples.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(maps)} samples -> {out}")


def add_model_commands(sub) -> None:
    t = sub.add_parser("train", help="train the diffusion terrain model")
    t.add_argument("--config", required=True)
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    t.add_argument("--resume", default=None, help="checkpoint to resume from")
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("sample", help="generate heightmaps from a checkpoint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--archetype", default=None)
    s.add_argument("--prop", action="append", default=[], metavar="KEY=VALUE",
                   help="terrain property in raw units, e.g. relief=0.3 water_fraction=0.1")
    s.add_argument("--guidance", type=float, default=1.5)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--eta", type=float, default=0.0)
    s.add_argument("--spacing", default="uniform", choices=["uniform", "quadratic"])
    s.add_argument("--formats", default="png16,npy")
    s.add_argument("--unity", action="store_true")
    s.add_argument("--raw-weights", action="store_true", help="use raw instead of EMA weights")
    s.add_argument("--out", default="outputs/samples")
    s.set_defaults(func=_cmd_sample)

    try:
        from nullscape.eval.cli import add_eval_commands
    except ImportError as exc:
        if "nullscape.eval.cli" not in str(exc):
            raise
    else:
        add_eval_commands(sub)
