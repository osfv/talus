"""How heights are presented to the diffusion model.

absolute  x = 2h - 1 (v1 / Talus-1). Flat maps use a few percent of the range, so small residual
          denoising errors become visible grain on plains.
relative  x = 2 (h - mean) / max(relief, floor) (Talus-2). Every map's shape uses the full range; the
          sampler places the shape at the requested mean elevation and relief (p98 - p2) when decoding.
Works on numpy arrays and torch tensors; ``mean``/``relief`` broadcast against the trailing map dims.
"""

from __future__ import annotations

from typing import Any

REL_SCALE = 2.0
RELIEF_FLOOR = 0.01
X0_RANGE = {"absolute": 1.0, "relative": 8.0}  # clamp for the predicted clean sample during DDIM


def spec(kind: str = "absolute") -> dict[str, Any]:
    if kind not in X0_RANGE:
        raise ValueError(f"unknown height parameterization {kind!r}")
    return {"kind": kind, "scale": REL_SCALE, "relief_floor": RELIEF_FLOOR,
            "mean_key": "mean_elevation", "relief_key": "relief"}


def encode(h, mean=None, relief=None, kind: str = "absolute"):
    if kind == "absolute":
        return h * 2 - 1
    return (h - mean) * REL_SCALE / _floor(relief)


def decode(x, mean=None, relief=None, kind: str = "absolute"):
    if kind == "absolute":
        return (x + 1) / 2
    return mean + x * _floor(relief) / REL_SCALE


def _floor(r):
    return r.clip(min=RELIEF_FLOOR) if hasattr(r, "clip") else max(r, RELIEF_FLOOR)
