"""Model-related CLI commands: train, sample, evaluate, compare."""

from __future__ import annotations

import argparse
import json
from difflib import get_close_matches
from pathlib import Path

import numpy as np


def _parse_props(items: list[str]) -> dict[str, float]:
    from nullscape.metrics.quality import CONDITION_KEYS

    out = {}
    for item in items:
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if not v:
            raise ValueError(f"--prop expects key=value, got {item!r}")
        if k not in CONDITION_KEYS:
            similar = get_close_matches(k, CONDITION_KEYS, n=1)
            hint = f"; did you mean {similar[0]!r}?" if similar else f"; choose from {', '.join(CONDITION_KEYS)}"
            raise ValueError(f"unknown property {k!r}{hint}")
        if k in out:
            raise ValueError(f"property {k!r} was specified more than once")
        try:
            value = float(v)
        except ValueError:
            raise ValueError(f"{k} must be a number, got {v!r}") from None
        if not np.isfinite(value):
            raise ValueError(f"{k} must be finite")
        if k in ("mean_elevation", "relief", "water_fraction") and not 0 <= value <= 1:
            raise ValueError(f"{k} must be in [0,1]")
        if k == "mean_slope_deg" and not 0 <= value <= 90:
            raise ValueError("mean_slope_deg must be in [0,90]")
        out[k] = value
    return out


def _add_runtime_args(parser) -> None:
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="default: CUDA if available, otherwise CPU")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.6,
                        help="PyTorch VRAM cap in (0,1], default 0.6; ignored on CPU")


def _runtime_device(args) -> str:
    import torch

    fraction = getattr(args, "gpu_memory_fraction", 0.6)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("gpu-memory-fraction must be in (0,1]")
    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or check the CUDA-enabled PyTorch installation")
        torch.cuda.set_per_process_memory_fraction(fraction)
    return device


def _cmd_train(args: argparse.Namespace) -> None:
    from nullscape.train.recovery import TrainingInterrupted
    from nullscape.train.trainer import train
    from nullscape.utils.config import load_config
    from nullscape.utils.tracking import atomic_json

    if args.result_json and Path(args.result_json).exists():
        raise FileExistsError("result-json already exists; choose a new path")
    try:
        result = train(load_config(args.config, args.set), resume=args.resume, dry_run=args.dry_run,
                       allow_inexact_resume=args.allow_inexact_resume)
    except TrainingInterrupted as exc:
        if args.result_json:
            atomic_json(args.result_json, {"state": "interrupted", "run": str(exc.run_path),
                                           "checkpoint": str(exc.checkpoint)})
        raise
    if args.result_json:
        atomic_json(args.result_json, {"state": "preflight" if args.dry_run else "completed",
                                      "result": result if args.dry_run else str(result)})
    print(json.dumps(result, indent=2) if args.dry_run else f"finished: {result}")


def _cmd_sample(args: argparse.Namespace) -> None:
    from nullscape.export.engine import FORMATS, export_heightmap, unity_size
    from nullscape.inference.sampler import DEFAULT_SAMPLING, TerrainSampler, sampling_config
    from nullscape.metrics.quality import compute_metrics
    from nullscape.utils.paths import prepare_output_dir, resolve_checkpoint
    from nullscape.utils.seed import derive_seed
    from nullscape.viz.render import save_grid, save_inspection

    props = _parse_props(args.prop)
    overrides = {key: getattr(args, key) for key in DEFAULT_SAMPLING}
    sampling_config(**overrides)
    formats = list(dict.fromkeys(f.strip() for f in args.formats.split(",") if f.strip()))
    if set(formats) - set(FORMATS):
        raise ValueError(f"formats must be comma-separated choices from {', '.join(FORMATS)}")
    root_seed = 0 if args.seed is None else args.seed
    if args.seeds is not None:
        if args.seed is not None:
            raise ValueError("use either --seed or --seeds, not both")
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            raise ValueError("--seeds expects comma-separated nonnegative integers, e.g. 42,99") from None
        if any(s < 0 for s in seeds) or (args.n is not None and args.n != len(seeds)):
            raise ValueError("--seeds must be nonnegative and its length must match --n when provided")
        n = len(seeds)
    else:
        n = 16 if args.n is None else args.n
        seeds = [derive_seed(root_seed, i) for i in range(n)]
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = _runtime_device(args)
    out = prepare_output_dir(args.out)
    sampler = TerrainSampler.from_checkpoint(checkpoint, device=device, use_ema=not args.raw_weights)
    skw = sampler.resolve_sampling(**overrides)
    spatial = {}
    neighbors = {side: np.load(path, allow_pickle=False) for side in ("north", "south", "east", "west")
                 if (path := getattr(args, f"neighbor_{side}"))}
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8")) if args.layout else None
    if args.reference or args.preserve_mask or layout is not None or neighbors:
        from nullscape.inference.spatial import build_constraints

        ref, mask = build_constraints(
            sampler.world, layout=layout, neighbors=neighbors, overlap=args.overlap,
            reference=np.load(args.reference, allow_pickle=False) if args.reference else None,
            preserve_mask=np.load(args.preserve_mask, allow_pickle=False) if args.preserve_mask else None)
        spatial = {"reference": ref, "preserve_mask": mask}
    print(f"Sampling {n} maps on {device}: {checkpoint}\nOutput: {out}", flush=True)
    maps = sampler.sample(n=n, seeds=seeds, properties=props, archetype=args.archetype,
                          batch_size=args.batch_size, **skw, **spatial)
    size = unity_size(sampler.world.resolution) if args.unity else None
    summary = []
    for i, h in enumerate(maps):
        m = compute_metrics(h, sampler.world)
        meta = {"checkpoint": sampler.checkpoint_info, "seed": root_seed if args.seeds is None else None,
                "sample_index": i, "sample_seed": seeds[i],
                "archetype": args.archetype, "properties": props, **skw,
                "measured": {k: m[k] for k in sampler.condition_keys}}
        if spatial:
            meta["spatial"] = {"preserved_cells": int(mask.sum()), "layout": layout,
                               "reference": args.reference, "preserve_mask": args.preserve_mask,
                               "neighbors": {side: getattr(args, f"neighbor_{side}") for side in neighbors},
                               "overlap_cells": args.overlap if neighbors else 0,
                               "tile_stride_m": (sampler.world.resolution - args.overlap) * sampler.world.cell_size_m
                               if neighbors else None}
        if formats:
            file_seed = root_seed if args.seeds is None else seeds[i]
            export_heightmap(h, sampler.world, out, f"sample_{file_seed}_{i:03d}", formats, meta, size)
        summary.append(meta)
    if not args.no_previews:
        titles = [f"seed {seeds[i]}" if args.seeds is not None else f"seed {root_seed} #{i}" for i in range(n)]
        save_grid(maps, sampler.world, out / "grid.png", titles=titles)
        save_inspection(maps[0], sampler.world, out / "inspect_0.png", title="sample 0")
    (out / "samples.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(maps)} samples -> {out}")


def _cmd_checkpoints(args: argparse.Namespace) -> None:
    from nullscape.utils.paths import CHECKPOINT_ALIASES, REPO_ROOT, resolve_checkpoint, runs_root

    print("Official aliases:")
    for name, relative in CHECKPOINT_ALIASES.items():
        path = REPO_ROOT / relative
        print(f"  {name:12s} {'available' if path.is_file() else 'missing':9s} {path}")
    root = runs_root()
    if args.limit and root.is_dir():
        print("Recent runs (directory arguments prefer best.pt, then last.pt):")
        found = 0
        for folder in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
            try:
                checkpoint = resolve_checkpoint(folder)
            except FileNotFoundError:
                continue
            print(f"  {folder.name}: {checkpoint.name}")
            found += 1
            if found == args.limit:
                break


def _cmd_checkpoint_info(args: argparse.Namespace) -> None:
    from nullscape.inference.sampler import TerrainSampler
    from nullscape.models.unet import count_parameters

    sampler = TerrainSampler.from_checkpoint(args.checkpoint, device="cpu", use_ema=not args.raw_weights)
    info = {"checkpoint": sampler.checkpoint_info, "parameters": count_parameters(sampler.model),
            "world": sampler.world.to_dict(), "height_parameterization": sampler.height_param,
            "sampling": sampler.resolve_sampling(), "archetypes": sampler.archetypes,
            "properties": {key: {"training_mean": float(sampler.cond_mean[i]),
                                  "training_std": float(sampler.cond_std[i])}
                           for i, key in enumerate(sampler.condition_keys)},
            "prior_bank_entries": len(sampler._bank[0]) if sampler._bank is not None else 0}
    print(json.dumps(info, indent=2))


def _cmd_release_checkpoint(args: argparse.Namespace) -> None:
    from nullscape.utils.checkpoint import export_release
    from nullscape.utils.paths import resolve_checkpoint

    info = export_release(resolve_checkpoint(args.checkpoint), args.out, name=args.name, version=args.version,
                          license=args.license)
    print(json.dumps(info, indent=2))


def add_model_commands(sub) -> None:
    listing = sub.add_parser("checkpoints", help="list official model aliases and recent local runs without loading weights")
    listing.add_argument("--limit", type=int, default=10, help="maximum recent runs to list; 0 shows aliases only")
    listing.set_defaults(func=_cmd_checkpoints)
    info = sub.add_parser("checkpoint-info", help="inspect a trusted local checkpoint on CPU, without sampling")
    info.add_argument("--checkpoint", required=True, help=".pt file, run directory, or official alias")
    info.add_argument("--raw-weights", action="store_true")
    info.set_defaults(func=_cmd_checkpoint_info)
    release = sub.add_parser("release-checkpoint",
                             help="write a shareable checkpoint: EMA weights and metadata, without optimizer, "
                                  "training state or local paths")
    release.add_argument("--checkpoint", required=True, help=".pt file, run directory, or official alias")
    release.add_argument("--out", required=True, help="new .pt file, e.g. checkpoints/talus-3.pt")
    release.add_argument("--name", required=True, help="model name, e.g. Talus-3")
    release.add_argument("--version", required=True, help="release version, e.g. 3.0.0")
    release.add_argument("--license", default="Apache-2.0")
    release.set_defaults(func=_cmd_release_checkpoint)

    t = sub.add_parser("train", help="train the diffusion terrain model")
    t.add_argument("--config", required=True)
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    t.add_argument("--resume", default=None, help="checkpoint to resume from")
    t.add_argument("--dry-run", action="store_true", help="check three training steps without writing a run or checkpoint")
    t.add_argument("--allow-inexact-resume", action="store_true", help="acknowledge missing legacy state or device changes")
    t.add_argument("--result-json", help="new machine-readable result file for experiment automation")
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("sample", help="generate heightmaps from a checkpoint")
    s.add_argument("--checkpoint", required=True, help=".pt file, run directory, or alias such as talus-2")
    s.add_argument("--n", type=int, default=None, help="default: 16, or the number of explicit --seeds")
    s.add_argument("--seed", type=int, default=None, help="root seed (default 0)")
    s.add_argument("--seeds", help="explicit per-map seeds, e.g. 42,99; replay sample_seed from samples.json")
    _add_runtime_args(s)
    s.add_argument("--archetype", default=None)
    s.add_argument("--prop", action="append", default=[], metavar="KEY=VALUE",
                   help="terrain property in raw units, e.g. relief=0.3 water_fraction=0.1")
    s.add_argument("--guidance", type=float, default=None, help="default: checkpoint sampling preset")
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--eta", type=float, default=None)
    s.add_argument("--spacing", default=None, choices=["uniform", "quadratic"])
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--guidance-interval", type=float, nargs=2, metavar=("MIN_T", "MAX_T"),
                   help="guided normalized noise timesteps: 0 is clean, 1 is noisy; outside uses guidance 1")
    s.add_argument("--reference", help="normalized float [R,R] .npy heightmap to edit")
    s.add_argument("--preserve-mask", help="[R,R] bool/0-1 .npy; true cells stay unchanged")
    s.add_argument("--layout", help="JSON routes (points, height, width_cells) / flat_areas (center, height, radius_cells)")
    for side in ("north", "south", "east", "west"):
        s.add_argument(f"--neighbor-{side}", help="neighbor .npy heightmap; preserve its overlapping strip")
    s.add_argument("--overlap", type=int, default=4, help="tile overlap in cells, not extra border padding")
    s.add_argument("--formats", default="png16,npy")
    s.add_argument("--unity", action="store_true")
    s.add_argument("--raw-weights", action="store_true", help="use raw instead of EMA weights")
    s.add_argument("--out", default=None, help="new/empty output folder; default: unique folder under outputs/samples")
    s.add_argument("--no-previews", action="store_true", help="skip grid/inspection rendering; keep exports and metadata")
    s.set_defaults(func=_cmd_sample)

    from nullscape.train.queue import add_queue_command

    add_queue_command(sub)

    try:
        from nullscape.eval.cli import add_eval_commands
    except ImportError as exc:
        if "nullscape.eval.cli" not in str(exc):
            raise
    else:
        add_eval_commands(sub)
