"""Vectorized, seeded 2D gradient noise and fractal combinations.

Coordinates are in kilometers and frequencies in cycles per kilometer, so the
same parameters describe the same world regardless of grid resolution. Octaves
above the grid's Nyquist frequency are skipped (anti-aliasing).
"""

from __future__ import annotations

import numpy as np

_SQRT2 = np.sqrt(2.0)


class Perlin:
    """Classic gradient noise with random gradient angles and a random domain offset."""

    def __init__(self, rng: np.random.Generator):
        p = rng.permutation(256)
        self.perm = np.concatenate([p, p])
        ang = rng.uniform(0.0, 2.0 * np.pi, 256)
        self.grad = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.offset = rng.uniform(0.0, 256.0, 2)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = x + self.offset[0]
        y = y + self.offset[1]
        x0 = np.floor(x)
        y0 = np.floor(y)
        xf, yf = x - x0, y - y0
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        xj, yj = (xi + 1) & 255, (yi + 1) & 255

        def dot(ix: np.ndarray, iy: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
            g = self.grad[self.perm[self.perm[ix] + iy]]
            return g[..., 0] * dx + g[..., 1] * dy

        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
        n0 = dot(xi, yi, xf, yf) + u * (dot(xj, yi, xf - 1, yf) - dot(xi, yi, xf, yf))
        n1 = dot(xi, yj, xf, yf - 1) + u * (dot(xj, yj, xf - 1, yf - 1) - dot(xi, yj, xf, yf - 1))
        return (n0 + v * (n1 - n0)) * _SQRT2


def grid_km(resolution: int, extent_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center coordinates (x, y) in km for a square grid."""
    c = (np.arange(resolution) + 0.5) * (extent_m / resolution) / 1000.0
    return np.meshgrid(c, c, indexing="xy")


def _octaves(base_freq: float, octaves: int, lacunarity: float, nyquist: float) -> list[float]:
    freqs = [base_freq * lacunarity**i for i in range(octaves)]
    kept = [f for f in freqs if f <= nyquist]
    return kept or freqs[:1]


def fbm(noise: Perlin, x, y, base_freq, octaves, lacunarity=2.0, gain=0.5, nyquist=np.inf) -> np.ndarray:
    """Fractional Brownian motion, roughly in [-1, 1]."""
    total = np.zeros_like(x)
    norm = 0.0
    amp = 1.0
    for i, f in enumerate(_octaves(base_freq, octaves, lacunarity, nyquist)):
        total += amp * noise(x * f + 31.7 * i, y * f - 17.3 * i)
        norm += amp
        amp *= gain
    return total / norm


def billow(noise: Perlin, x, y, base_freq, octaves, lacunarity=2.0, gain=0.5, nyquist=np.inf) -> np.ndarray:
    """Absolute-value fBm in [0, 1] (rounded hills)."""
    total = np.zeros_like(x)
    norm = 0.0
    amp = 1.0
    for i, f in enumerate(_octaves(base_freq, octaves, lacunarity, nyquist)):
        total += amp * np.abs(noise(x * f + 11.1 * i, y * f + 5.3 * i))
        norm += amp
        amp *= gain
    return np.clip(total / norm, 0.0, 1.0)


def ridged(noise: Perlin, x, y, base_freq, octaves, lacunarity=2.0, gain=0.5, nyquist=np.inf, sharpness=2.0) -> np.ndarray:
    """Musgrave ridged multifractal in [0, 1]; sharp crests, detail concentrated on ridges."""
    total = np.zeros_like(x)
    weight = np.ones_like(x)
    norm = 0.0
    amp = 1.0
    for i, f in enumerate(_octaves(base_freq, octaves, lacunarity, nyquist)):
        s = (1.0 - np.abs(noise(x * f - 7.9 * i, y * f + 23.1 * i))) ** sharpness
        s *= weight
        weight = np.clip(s * 2.0, 0.0, 1.0)
        total += amp * s
        norm += amp
        amp *= gain
    return np.clip(total / norm, 0.0, 1.0)


def warp(noise_x: Perlin, noise_y: Perlin, x, y, strength_km: float, base_freq: float, octaves: int = 3, nyquist=np.inf):
    """Domain warping: displace coordinates by fBm fields (strength in km)."""
    if strength_km <= 0:
        return x, y
    dx = fbm(noise_x, x, y, base_freq, octaves, nyquist=nyquist)
    dy = fbm(noise_y, x, y, base_freq, octaves, nyquist=nyquist)
    return x + strength_km * dx, y + strength_km * dy


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
