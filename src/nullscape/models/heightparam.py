"""How heights are presented to the diffusion model.

absolute  x = 2h - 1 (v1 / Talus-1). Flat maps use a few percent of the range, so small residual
          denoising errors become visible grain on plains.
relative  x = 2 (h - mean) / max(relief, floor) (Talus-2). Every map's shape uses the full range; the
          sampler places the shape at the requested mean elevation and relief (p98 - p2) when decoding.
Works on numpy arrays and torch tensors; ``mean``/``relief`` broadcast against the trailing map dims.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

REL_SCALE = 2.0
RELIEF_FLOOR = 0.01
RELIEF_FLOOR_M = 12.0  # physical floor in metres (0.01 of the 1200 m procedural world)
X0_RANGE = {"absolute": 1.0, "relative": 8.0}  # clamp for the predicted clean sample during DDIM


def spec(kind: str = "absolute", max_height_m: float = 1200.0) -> dict[str, Any]:
    if kind not in X0_RANGE:
        raise ValueError(f"unknown height parameterization {kind!r}")
    if not math.isfinite(max_height_m) or max_height_m <= 0:
        raise ValueError("max_height_m must be finite and positive")
    return {"kind": kind, "scale": REL_SCALE, "relief_floor": RELIEF_FLOOR_M / float(max_height_m),
            "relief_floor_m": RELIEF_FLOOR_M, "mean_key": "mean_elevation", "relief_key": "relief"}


def _resolve(kind: str | Mapping[str, Any], floor: float | None) -> tuple[str, float]:
    if isinstance(kind, Mapping):
        k = str(kind["kind"])
        f = float(kind.get("relief_floor", RELIEF_FLOOR) if floor is None else floor)
    else:
        k = str(kind)
        f = float(RELIEF_FLOOR if floor is None else floor)
    if k not in X0_RANGE:
        raise ValueError(f"unknown height parameterization {k!r}")
    return k, f


def encode(h, mean=None, relief=None, kind: str | Mapping[str, Any] = "absolute", *, floor: float | None = None):
    k, f = _resolve(kind, floor)
    if k == "absolute":
        return h * 2 - 1
    return (h - mean) * REL_SCALE / _floor(relief, f)


def decode(x, mean=None, relief=None, kind: str | Mapping[str, Any] = "absolute", *, floor: float | None = None):
    k, f = _resolve(kind, floor)
    if k == "absolute":
        return (x + 1) / 2
    return mean + x * _floor(relief, f) / REL_SCALE


def _floor(r, floor: float = RELIEF_FLOOR):
    return r.clip(min=floor) if hasattr(r, "clip") else max(r, floor)
