from __future__ import annotations

import numpy as np


def make_splits(n: int, fractions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Deterministic disjoint, covering splits. Fractions are normalized; train gets the remainder."""
    names = list(fractions)
    if "train" not in names:
        raise ValueError("fractions must include 'train'")
    total = sum(fractions.values())
    perm = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5B117])).permutation(n)
    out: dict[str, np.ndarray] = {}
    start = 0
    for name in [k for k in names if k != "train"]:
        size = int(round(n * fractions[name] / total))
        out[name] = np.sort(perm[start : start + size])
        start += size
    out["train"] = np.sort(perm[start:])
    return out
