"""Export heightmaps for game engines.

Formats:
  png16  16-bit grayscale PNG (Unreal, Godot, Unity via texture import)
  r16    headerless little-endian uint16 RAW (Unity terrain "Import Raw")
  npy    float32 array in [0, 1]
  obj    triangle mesh in meters with vertex normals (quick preview in any DCC tool)
Every export writes a JSON sidecar with the physical scale needed to import correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nullscape.world import WorldSpec

FORMATS = ("png16", "r16", "npy", "obj")
_SUFFIX = {"png16": "png", "r16": "r16", "npy": "npy", "obj": "obj"}


def resample(h: np.ndarray, size: int) -> np.ndarray:
    """Bicubic resample to size x size (corner-aligned, so the terrain footprint is preserved)."""
    t = torch.from_numpy(np.asarray(h, dtype=np.float32))[None, None]
    out = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=True)
    return out[0, 0].clamp(0.0, 1.0).numpy()


def unity_size(resolution: int) -> int:
    """Smallest 2^n + 1 >= resolution + 1 (Unity terrain heightmap sizes)."""
    n = 1
    while n + 1 < resolution + 1:
        n *= 2
    return n + 1


def to_uint16(h: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(h, 0.0, 1.0) * 65535).astype(np.uint16)


def write_png16(h: np.ndarray, path: Path) -> None:
    img = Image.fromarray(to_uint16(h))
    assert img.mode == "I;16", img.mode
    img.save(path)


def read_png16(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float64).astype(np.float32) / 65535.0


def write_r16(h: np.ndarray, path: Path) -> None:
    to_uint16(h).astype("<u2").tofile(path)


def read_r16(path: Path, size: int) -> np.ndarray:
    return np.fromfile(path, dtype="<u2").reshape(size, size).astype(np.float32) / 65535.0


def write_obj(h: np.ndarray, world: WorldSpec, path: Path) -> None:
    """Grid mesh: x east, y up, z south; vertices at cell centers, in meters."""
    n = h.shape[0]
    cell = world.extent_m / n
    z = np.asarray(h, dtype=np.float64) * world.max_height_m
    c = (np.arange(n) + 0.5) * cell
    xx, zz = np.meshgrid(c, c)
    gy, gx = np.gradient(z, cell)
    normals = np.stack([-gx, np.ones_like(z), -gy], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    ids = np.arange(n * n).reshape(n, n) + 1
    a, b, cc, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, cc, b], 1), np.stack([b, cc, d], 1)])
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# NULLSCAPE heightmap {n}x{n}, cell {cell:.3f} m\n")
        np.savetxt(f, np.stack([xx.ravel(), z.ravel(), zz.ravel()], 1), fmt="v %.3f %.3f %.3f")
        np.savetxt(f, normals.reshape(-1, 3), fmt="vn %.5f %.5f %.5f")
        np.savetxt(f, np.repeat(faces, 2, axis=1).reshape(-1, 6), fmt="f %d//%d %d//%d %d//%d")


def export_heightmap(
    h: np.ndarray,
    world: WorldSpec,
    out_dir: str | Path,
    stem: str,
    formats: Iterable[str] = ("png16", "r16", "npy"),
    metadata: dict[str, Any] | None = None,
    size: int | None = None,
) -> dict[str, Path]:
    """Write ``h`` in the requested formats plus ``<stem>.json``. ``size`` resamples first (e.g. unity_size)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = np.clip(np.asarray(h, dtype=np.float32), 0.0, 1.0)
    if size is not None and size != h.shape[0]:
        h = resample(h, size)
    res = h.shape[0]
    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
        path = out_dir / f"{stem}.{_SUFFIX[fmt]}"
        if fmt == "png16":
            write_png16(h, path)
        elif fmt == "r16":
            write_r16(h, path)
        elif fmt == "npy":
            np.save(path, h)
        else:
            write_obj(h, world, path)
        written[fmt] = path
    sidecar = {
        "resolution": res,
        "extent_m": world.extent_m,
        "cell_size_m": world.extent_m / res,
        "max_height_m": world.max_height_m,
        "min_height_m": 0.0,
        "sea_level": world.sea_level,
        "sea_level_m": world.sea_level * world.max_height_m,
        "encoding": "uint16 = round(h * 65535); height_m = h * max_height_m",
        "byte_order_r16": "little-endian",
        "files": {k: v.name for k, v in written.items()},
        **({"source": metadata} if metadata else {}),
    }
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    written["json"] = path
    return written
