from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from nullscape.inference.sampler import TerrainSampler


def benchmark_sampler(sampler: TerrainSampler, *, batches=(1, 32, 64), repeats: int = 3, warmup: int = 1,
                      sampling: dict | None = None, inputs: Callable[[int], dict] | None = None) -> dict:
    if not batches or any(b < 1 for b in batches) or repeats < 1 or warmup < 1:
        raise ValueError("batches, repeats, and warmup must be positive")
    cfg = sampler.resolve_sampling(**(sampling or {}))
    gpu = sampler.device.type == "cuda"

    def sync():
        if gpu:
            torch.cuda.synchronize(sampler.device)

    rows = []
    for batch in batches:
        row = {**cfg, "batch": int(batch), "oom": False}
        kwargs = inputs(batch) if inputs is not None else {}
        try:
            for i in range(warmup):
                sampler.sample(n=batch, batch_size=batch, seed=1700 + i, **cfg, **kwargs)
            sync()
            if gpu:
                torch.cuda.reset_peak_memory_stats(sampler.device)
            elapsed = []
            for i in range(repeats):
                sync()
                t0 = time.perf_counter()
                sampler.sample(n=batch, batch_size=batch, seed=2700 + i, **cfg, **kwargs)
                sync()
                elapsed.append(time.perf_counter() - t0)
            median = float(np.median(elapsed))
            row.update(batch_seconds=elapsed, latency_seconds_per_batch=median,
                       batch_latency_p95_seconds=float(np.percentile(elapsed, 95)),
                       seconds_per_map=median / batch, maps_per_second=batch / median,
                       peak_alloc_mb=torch.cuda.max_memory_allocated(sampler.device) / 2**20 if gpu else 0.0,
                       peak_reserved_mb=torch.cuda.max_memory_reserved(sampler.device) / 2**20 if gpu else 0.0)
        except torch.OutOfMemoryError:
            row["oom"] = True
        rows.append(row)
        if gpu:
            torch.cuda.empty_cache()
    return {"checkpoint": getattr(sampler, "checkpoint_info", {}), "device": str(sampler.device),
            "device_name": torch.cuda.get_device_name(sampler.device) if gpu else "CPU",
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "repeats": repeats, "warmup_batches": warmup, "rows": rows,
            "timing_scope": "warm end-to-end sampling, including prior lookup, noise creation, transfers and CPU output; excludes checkpoint load",
            "isolation": "caller must ensure no competing GPU work; no isolation guarantee is inferred from timings"}


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from nullscape.eval.cli import _load, _mode_inputs, _sampler_kwargs

    batches = [int(b) for b in args.batches.split(",")]
    if any(b < 1 for b in batches):
        raise ValueError("batch sizes must be positive")
    args.n = max(2, max(batches))
    sampler, store, a, _, _ = _load(args)
    _mode_inputs(args.mode, store, a)
    inputs = lambda n: _mode_inputs(args.mode, store, a[np.arange(n) % len(a)])
    cfg = _sampler_kwargs(args, sampler)
    configs = [cfg] if not args.steps_list else [{**cfg, "steps": int(s)} for s in args.steps_list.split(",")]
    report = None
    rows = []
    for config in configs:
        report = benchmark_sampler(sampler, batches=batches, repeats=args.repeats, warmup=args.warmup,
                                   sampling=config, inputs=inputs)
        rows.extend(report["rows"])
        for row in report["rows"]:
            print(f"steps={row['steps']} batch={row['batch']} " +
                  ("OOM under memory cap" if row["oom"] else
                   f"latency={row['latency_seconds_per_batch']:.3f}s, {row['maps_per_second']:.3f} maps/s"), flush=True)
    with open(sampler.checkpoint_info["path"], "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    report.update(rows=rows, mode=args.mode, gpu_memory_fraction=args.gpu_memory_fraction,
                  checkpoint={**report["checkpoint"], "sha256": digest},
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def add_performance_command(sub) -> None:
    from nullscape.cli_model import _add_runtime_args
    from nullscape.eval.cli import _add_sampler_args

    p = sub.add_parser("benchmark-sampling", help="measure this checkpoint's latency, throughput and peak VRAM")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="conditional")
    p.add_argument("--batches", default="1,32,64")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps-list", default="", help="optional comma-separated step counts to time")
    _add_runtime_args(p)
    _add_sampler_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_benchmark)
