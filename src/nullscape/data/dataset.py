from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nullscape.data.storage import TerrainStore, dequantize


def dihedral(x: np.ndarray, k: int) -> np.ndarray:
    """Apply one of the 8 rotations/flips (k in 0..7) to the last two axes."""
    if k >= 4:
        x = x[..., ::-1]
    return np.rot90(x, k % 4, axes=(-2, -1))


class TerrainDataset(Dataset):
    """Model-ready samples: x in [-1, 1] as [1, R, R], z-scored conditions, archetype label."""

    def __init__(self, root: str | Path | TerrainStore, split: str, augment: bool = False, limit: int | None = None,
                 in_memory: bool = True):
        self.store = root if isinstance(root, TerrainStore) else TerrainStore.open(root)
        self.indices = self.store.split(split)
        if limit is not None:
            self.indices = self.indices[:limit]
        self.augment = augment
        cond = (self.store.conditions - self.store.condition_mean) / self.store.condition_std
        self.cond = cond[self.indices].astype(np.float32)
        self.labels = self.store.labels[self.indices].astype(np.int64)
        # 50k x 64x64 uint16 is ~400 MB; loading the split into RAM avoids memmap
        # page faults in the training loop. Set in_memory=False for huge datasets.
        self.heights = np.asarray(self.store.heights_u16[self.indices]) if in_memory else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        q = self.heights[i] if self.heights is not None else self.store.heights_u16[int(self.indices[i])]
        h = dequantize(q)
        if self.augment:
            h = dihedral(h, int(np.random.randint(8)))
        x = np.ascontiguousarray(h, dtype=np.float32) * 2.0 - 1.0
        return {
            "x": torch.from_numpy(x)[None],
            "cond": torch.from_numpy(self.cond[i]),
            "label": torch.tensor(self.labels[i]),
            "index": torch.tensor(int(self.indices[i])),
        }
