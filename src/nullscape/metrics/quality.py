"""Per-map terrain descriptors.

All metrics take a heightmap ``h`` (float array, [H, W], values in [0, 1]) and a
``WorldSpec`` that gives it physical scale. Metrics are rotation/flip invariant
(up to discretization) so they are valid under dihedral augmentation.

A subset (``CONDITION_KEYS``) is used as the conditioning vector for the
generative model: the model is conditioned on properties *measured* from the
heightmap, so controllability can be checked by measuring generated samples.
"""

from __future__ import annotations

import numpy as np

from nullscape.world import WorldSpec

CONDITION_KEYS: tuple[str, ...] = (
    "mean_elevation",
    "relief",
    "mean_slope_deg",
    "water_fraction",
    "spectral_beta",
)

_EPS = 1e-12


def _as_height(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"expected a 2D heightmap, got shape {h.shape}")
    return h


def detrend_plane(h: np.ndarray) -> np.ndarray:
    """Remove the least-squares best-fit plane."""
    rows, cols = h.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    a = np.stack([xx.ravel(), yy.ravel(), np.ones(h.size)], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(a, h.ravel(), rcond=None)
    return h - (a @ coef).reshape(h.shape)


def _power_spectrum(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windowed 2D power spectrum of the detrended map and integer radial frequency."""
    rows, cols = h.shape
    d = detrend_plane(_as_height(h))
    win = np.outer(np.hanning(rows), np.hanning(cols))
    power = np.abs(np.fft.fft2(d * win)) ** 2
    fy = np.fft.fftfreq(rows) * rows
    fx = np.fft.fftfreq(cols) * cols
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return power, radius


def radial_power_spectrum(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum (RAPSD).

    Returns (k, P) for integer radial frequencies k = 1 .. min(H, W) // 2
    (cycles per map), P = mean power in the annulus round(|f|) == k.
    """
    power, radius = _power_spectrum(h)
    kmax = min(h.shape) // 2
    bins = np.rint(radius).astype(np.int64).ravel()
    sums = np.bincount(bins, weights=power.ravel(), minlength=kmax + 1)
    counts = np.bincount(bins, minlength=kmax + 1)
    k = np.arange(1, kmax + 1)
    return k, sums[1 : kmax + 1] / np.maximum(counts[1 : kmax + 1], 1)


def spectral_beta(h: np.ndarray) -> float:
    """Spectral exponent beta of P(k) ~ k^-beta, fitted on k in [2, 0.75 * kmax].

    Larger beta = smoother terrain dominated by large-scale features; smaller
    beta = more small-scale roughness. 2D fBm with Hurst H has beta = 2H + 2.
    """
    k, p = radial_power_spectrum(h)
    kmax = k[-1]
    sel = (k >= 2) & (k <= max(3, int(0.75 * kmax)))
    if p[sel].max() <= _EPS:
        return 0.0
    slope, _ = np.polyfit(np.log(k[sel]), np.log(p[sel] + _EPS), 1)
    return float(-slope)


def high_frequency_energy(h: np.ndarray) -> float:
    """Fraction of non-DC spectral power at radial frequency > kmax / 2."""
    power, radius = _power_spectrum(h)
    kmax = min(h.shape) // 2
    total = power[radius >= 1].sum()
    if total <= _EPS:
        return 0.0
    return float(power[radius > kmax / 2].sum() / total)


def checkerboard_score(h: np.ndarray) -> float:
    """Amplitude of the (-1)^(i+j) Nyquist pattern relative to detrended std.

    Near 0 for natural terrain; large values flag transposed-conv style artifacts.
    """
    d = detrend_plane(_as_height(h))
    std = d.std()
    if std <= _EPS:
        return 0.0
    rows, cols = d.shape
    sign = np.where((np.add.outer(np.arange(rows), np.arange(cols)) % 2) == 0, 1.0, -1.0)
    return float(abs((d * sign).mean()) / std)


def slope_degrees(h: np.ndarray, world: WorldSpec) -> np.ndarray:
    """Per-cell slope in degrees from central differences in world units."""
    z = _as_height(h) * world.max_height_m
    gy, gx = np.gradient(z, world.cell_size_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _neighbor_stack(a: np.ndarray, pad_value: float | None) -> np.ndarray:
    """[8, H, W] stack of the 8-neighborhood (edge-replicated or constant padding)."""
    if pad_value is None:
        p = np.pad(a, 1, mode="edge")
    else:
        p = np.pad(a, 1, mode="constant", constant_values=pad_value)
    rows, cols = a.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return np.stack([p[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols] for dy, dx in offs])


def terrain_ruggedness_index(h: np.ndarray, world: WorldSpec) -> np.ndarray:
    """Riley et al. (1999) TRI per cell, in meters."""
    z = _as_height(h) * world.max_height_m
    nb = _neighbor_stack(z, None)
    return np.sqrt(((nb - z[None]) ** 2).sum(axis=0))


def local_extrema(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strict local maxima / minima over the 8-neighborhood, interior cells only."""
    h = _as_height(h)
    maxima = (h[None] > _neighbor_stack(h, -np.inf)).all(axis=0)
    minima = (h[None] < _neighbor_stack(h, np.inf)).all(axis=0)
    interior = np.zeros_like(maxima)
    interior[1:-1, 1:-1] = True
    return maxima & interior, minima & interior


def compute_metrics(h: np.ndarray, world: WorldSpec) -> dict[str, float]:
    h = _as_height(h)
    if h.shape != (world.resolution, world.resolution):
        raise ValueError(f"heightmap shape {h.shape} does not match world resolution {world.resolution}")

    mean = h.mean()
    std = h.std()
    lo, hi = h.min(), h.max()
    p2, p98 = np.percentile(h, [2, 98])
    centered = h - mean
    skew = (centered**3).mean() / std**3 if std > 1e-9 else 0.0
    kurt = (centered**4).mean() / std**4 - 3.0 if std > 1e-9 else 0.0
    hyps = (mean - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5

    land = h >= world.sea_level
    slope = slope_degrees(h, world)
    z = h * world.max_height_m
    lap = _neighbor_stack(z, None)[[1, 3, 4, 6]].sum(axis=0) - 4.0 * z
    maxima, minima = local_extrema(h)
    interior_cells = max((h.shape[0] - 2) * (h.shape[1] - 2), 1)
    interior_land = land.copy()
    interior_land[[0, -1], :] = False
    interior_land[:, [0, -1]] = False
    n_interior_land = int(interior_land.sum())

    return {
        "mean_elevation": float(mean),
        "std_elevation": float(std),
        "min_elevation": float(lo),
        "max_elevation": float(hi),
        "relief": float(p98 - p2),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "hypsometric_integral": float(hyps),
        "water_fraction": float(1.0 - land.mean()),
        "mean_slope_deg": float(slope.mean()),
        "p90_slope_deg": float(np.percentile(slope, 90)),
        "p99_slope_deg": float(np.percentile(slope, 99)),
        "ruggedness_m": float(terrain_ruggedness_index(h, world).mean()),
        "curvature_std": float((lap / world.cell_size_m).std()),
        "spectral_beta": spectral_beta(h),
        "hf_energy": high_frequency_energy(h),
        "checkerboard": checkerboard_score(h),
        "peak_density": float(maxima.sum() * 1000.0 / interior_cells),
        "sink_density": float((minima & land).sum() * 1000.0 / n_interior_land) if n_interior_land else 0.0,
    }


def condition_vector(metrics: dict[str, float]) -> np.ndarray:
    return np.array([metrics[k] for k in CONDITION_KEYS], dtype=np.float32)
