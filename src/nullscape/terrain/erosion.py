"""Vectorized landscape evolution operators on heights in meters.

* ``stream_power_erosion``: fluvial incision dz = -K * A^m * S (D8 flow routing,
  drainage area from a sparse linear solve). Carves dendritic valley networks.
* ``hillslope_diffusion``: linear diffusion that rounds crests and fills small pits.
* ``thermal_erosion``: moves material downhill where slope exceeds the talus angle.

All operators are deterministic and conserve their inputs' shape.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

_OFFS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
_DIST = np.hypot(_OFFS[:, 0], _OFFS[:, 1])


def _neighbors(z: np.ndarray, pad_value: float | None = None) -> np.ndarray:
    p = np.pad(z, 1, mode="edge") if pad_value is None else np.pad(z, 1, constant_values=pad_value)
    rows, cols = z.shape
    return np.stack([p[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols] for dy, dx in _OFFS])


def d8_receivers(z: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent receiver index per cell (-1 for sinks/outlets) and its slope (tan)."""
    rows, cols = z.shape
    nb = _neighbors(z, pad_value=np.inf)
    slopes = (z[None] - nb) / (_DIST[:, None, None] * cell)
    k = slopes.argmax(axis=0)
    s = np.take_along_axis(slopes, k[None], axis=0)[0]
    yy, xx = np.mgrid[0:rows, 0:cols]
    ry, rx = yy + _OFFS[k, 0], xx + _OFFS[k, 1]
    recv = np.where(s > 0, ry * cols + rx, -1).ravel()
    return recv, np.maximum(s, 0.0).ravel()


def drainage_area(recv: np.ndarray, cell_area: float) -> np.ndarray:
    """Upstream contributing area per cell: solves a = cell_area + sum(a of donors)."""
    n = recv.size
    donors = np.flatnonzero(recv >= 0)
    m = csr_matrix((np.ones(donors.size), (recv[donors], donors)), shape=(n, n))
    return spsolve((identity(n, format="csr") - m).tocsc(), np.full(n, cell_area))


def stream_power_erosion(
    z: np.ndarray,
    cell: float,
    iterations: int,
    k: float,
    m: float = 0.5,
    base_level: float = -np.inf,
    max_drop_fraction: float = 0.9,
) -> np.ndarray:
    """Explicit stream-power incision. ``k`` is meters removed per iteration at A = 1 km^2, S = 1.

    Each step never lowers a cell below max_drop_fraction of the way to its receiver,
    which keeps the explicit scheme stable. Cells at or below ``base_level`` (sea) do not erode.
    """
    z = z.astype(np.float64).copy()
    cell_area_km2 = (cell / 1000.0) ** 2
    for _ in range(iterations):
        recv, s = d8_receivers(z, cell)
        area = drainage_area(recv, cell_area_km2)
        flat = z.ravel()
        drop = k * area**m * s
        has = recv >= 0
        limit = np.zeros_like(flat)
        limit[has] = max_drop_fraction * (flat[has] - flat[recv[has]])
        drop = np.minimum(drop, limit)
        drop[flat <= base_level] = 0.0
        z = (flat - drop).reshape(z.shape)
    return z


def hillslope_diffusion(z: np.ndarray, iterations: int, rate: float = 0.2) -> np.ndarray:
    """Explicit linear diffusion (rate <= 0.25 for stability), reflecting boundaries."""
    z = z.astype(np.float64).copy()
    for _ in range(iterations):
        p = np.pad(z, 1, mode="edge")
        lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * z
        z += rate * lap
    return z


def thermal_erosion(z: np.ndarray, cell: float, talus_deg: float, iterations: int, rate: float = 0.5) -> np.ndarray:
    """Mass-conserving thermal weathering toward the talus angle."""
    z = z.astype(np.float64).copy()
    rows, cols = z.shape
    thr = np.tan(np.radians(talus_deg)) * cell * _DIST[:, None, None]
    for _ in range(iterations):
        excess = np.maximum(z[None] - _neighbors(z) - thr, 0.0)
        total = excess.sum(axis=0)
        moved = rate * 0.5 * excess.max(axis=0)
        frac = np.divide(excess, total[None], out=np.zeros_like(excess), where=total[None] > 0)
        z -= moved
        recv = np.zeros((rows + 2, cols + 2))
        for i, (dy, dx) in enumerate(_OFFS):
            recv[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols] += moved * frac[i]
        z += recv[1:-1, 1:-1]
    return z
