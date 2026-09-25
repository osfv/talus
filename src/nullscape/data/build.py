"""Parallel, reproducible dataset construction."""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from nullscape.data.splits import make_splits
from nullscape.data.storage import SPLITS, quantize
from nullscape.metrics.quality import CONDITION_KEYS, compute_metrics, condition_vector
from nullscape.metrics.traversability import AgentSpec, analyze
from nullscape.terrain.archetypes import ARCHETYPES
from nullscape.terrain.generator import GENERATOR_VERSION, GeneratorConfig, generate
from nullscape.utils.config import config_hash
from nullscape.utils.paths import data_root
from nullscape.utils.tracking import git_info
from nullscape.world import WorldSpec

_W: dict[str, Any] = {}


def _init_worker(seed: int, world: dict, gen: dict, agent: dict) -> None:
    _W.update(seed=seed, world=WorldSpec.from_dict(world), gen=GeneratorConfig.from_dict(gen), agent=AgentSpec(**agent))


def make_sample(index: int, seed: int, world: WorldSpec, gen: GeneratorConfig, agent: AgentSpec):
    h, rec = generate(index, seed, world, gen)
    metrics = compute_metrics(h, world)
    rec = {"index": index, "seed": [seed, index], **rec, "metrics": metrics,
           "traversability": analyze(h, world, agent).to_dict()}
    return index, quantize(h), condition_vector(metrics), rec["archetype_id"], rec


def _work(index: int):
    return make_sample(index, _W["seed"], _W["world"], _W["gen"], _W["agent"])


def build_dataset(cfg: dict[str, Any], workers: int | None = None, out_root: Path | None = None,
                  overwrite: bool = False, progress: bool = True) -> Path:
    name = cfg["name"]
    seed = int(cfg["seed"])
    n = int(cfg["n"])
    world = WorldSpec.from_dict(cfg.get("world", {}))
    gen = GeneratorConfig.from_dict(cfg.get("generator"))
    agent = AgentSpec(**cfg.get("agent", {}))
    fractions = cfg.get("splits", {"train": 0.9, "val": 0.05, "test": 0.05})
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    root = Path(out_root or data_root()) / name
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass overwrite=True / --overwrite to rebuild")
        shutil.rmtree(root)
    tmp = root.with_name(root.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    r = world.resolution
    heights = np.lib.format.open_memmap(tmp / "heights.npy", mode="w+", dtype=np.uint16, shape=(n, r, r))
    conds = np.zeros((n, len(CONDITION_KEYS)), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    clipped = np.zeros(n)
    passed = np.zeros(n, dtype=bool)

    t0 = time.time()
    init_args = (seed, world.to_dict(), gen.__dict__, agent.__dict__)
    with open(tmp / "meta.jsonl", "w", encoding="utf-8") as meta:
        if workers == 1:
            _init_worker(*init_args)
            results = map(_work, range(n))
            pool = None
        else:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")  # inherited by spawned workers; avoids BLAS oversubscription
            pool = get_context("spawn").Pool(workers, initializer=_init_worker, initargs=init_args)
            results = pool.imap(_work, range(n), chunksize=max(1, min(64, n // (workers * 8) or 1)))
        try:
            for idx, q, c, lab, rec in tqdm(results, total=n, disable=not progress, desc=f"gen {name}"):
                heights[idx] = q
                conds[idx] = c
                labels[idx] = lab
                clipped[idx] = rec["clipped_fraction"]
                passed[idx] = rec["traversability"]["passed"]
                meta.write(json.dumps(rec) + "\n")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    heights.flush()
    del heights
    wall = time.time() - t0

    np.save(tmp / "conditions.npy", conds)
    np.save(tmp / "labels.npy", labels)
    splits = make_splits(n, fractions, seed)
    for s in SPLITS:
        np.save(tmp / f"split_{s}.npy", splits.get(s, np.zeros(0, dtype=np.int64)))
    train_c = conds[splits["train"]]
    counts = Counter(labels.tolist())
    manifest = {
        "name": name,
        "format_version": 1,
        "generator_version": GENERATOR_VERSION,
        "config": cfg,
        "config_sha256": config_hash(cfg),
        "world": world.to_dict(),
        "agent": agent.__dict__,
        "n": n,
        "splits": {s: int(len(splits.get(s, []))) for s in SPLITS},
        "condition_keys": list(CONDITION_KEYS),
        "condition_stats": {"mean": train_c.mean(0).tolist(), "std": (train_c.std(0) + 1e-8).tolist()},
        "archetypes": list(ARCHETYPES),
        "archetype_counts": {ARCHETYPES[k]: int(v) for k, v in sorted(counts.items())},
        "summary": {
            "clipped_fraction_mean": float(clipped.mean()),
            "clipped_fraction_max": float(clipped.max()),
            "maps_with_clipping": int((clipped > 0).sum()),
            "traversability_pass_rate": float(passed.mean()),
            "traversability_pass_rate_by_archetype": {
                ARCHETYPES[k]: float(passed[labels == k].mean()) for k in sorted(counts)
            },
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generation_seconds": wall,
        "workers": workers,
        "git": git_info(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.rename(root)
    return root
