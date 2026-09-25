from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def derive_seed(*parts: int) -> int:
    """Stable 63-bit seed from integer parts (independent of process / ordering)."""
    hi, lo = np.random.SeedSequence([int(p) for p in parts]).generate_state(2, np.uint32)
    return ((int(hi) << 32) | int(lo)) & ((1 << 63) - 1)
