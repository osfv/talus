"""On-disk dataset format.

<root>/
  heights.npy       uint16 [N, R, R]; h = value / 65535 (global normalization)
  conditions.npy    float32 [N, K] measured properties, ordered as manifest["condition_keys"]
  labels.npy        int64 [N] archetype ids
  meta.jsonl        one JSON record per map (seed, params, metrics, traversability)
  split_{train,val,test}.npy   int64 index arrays
  manifest.json     provenance, world spec, split sizes, train-split condition stats
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from nullscape.utils.paths import dataset_dir
from nullscape.world import WorldSpec

SPLITS = ("train", "val", "test")
UINT16_MAX = 65535


def quantize(h: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(h, 0.0, 1.0) * UINT16_MAX).astype(np.uint16)


def dequantize(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float32) / UINT16_MAX


@dataclass
class TerrainStore:
    """Read access to a built dataset (heights are memory-mapped)."""

    root: Path

    @classmethod
    def open(cls, name_or_path: str | Path) -> "TerrainStore":
        root = dataset_dir(name_or_path)
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"no dataset manifest at {root}")
        return cls(root)

    @cached_property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    @cached_property
    def world(self) -> WorldSpec:
        return WorldSpec.from_dict(self.manifest["world"])

    @cached_property
    def heights_u16(self) -> np.ndarray:
        return np.load(self.root / "heights.npy", mmap_mode="r")

    @cached_property
    def conditions(self) -> np.ndarray:
        return np.load(self.root / "conditions.npy")

    @cached_property
    def labels(self) -> np.ndarray:
        return np.load(self.root / "labels.npy")

    @property
    def condition_keys(self) -> list[str]:
        return list(self.manifest["condition_keys"])

    @property
    def archetypes(self) -> list[str]:
        return list(self.manifest["archetypes"])

    @property
    def condition_mean(self) -> np.ndarray:
        return np.asarray(self.manifest["condition_stats"]["mean"], dtype=np.float32)

    @property
    def condition_std(self) -> np.ndarray:
        return np.asarray(self.manifest["condition_stats"]["std"], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.manifest["n"])

    def split(self, name: str) -> np.ndarray:
        if name not in SPLITS:
            raise KeyError(f"split must be one of {SPLITS}")
        return np.load(self.root / f"split_{name}.npy")

    def heights(self, indices: np.ndarray | slice | None = None) -> np.ndarray:
        """Float32 heights in [0, 1] for the given indices (sorted access is fastest)."""
        q = self.heights_u16 if indices is None else self.heights_u16[indices]
        return dequantize(q)

    def split_heights(self, name: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        idx = self.split(name)
        if limit is not None:
            idx = idx[:limit]
        return self.heights(idx), idx

    def iter_meta(self) -> Iterator[dict[str, Any]]:
        with open(self.root / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def meta(self, index: int) -> dict[str, Any]:
        for rec in self.iter_meta():
            if rec["index"] == index:
                return rec
        raise IndexError(index)
