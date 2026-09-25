"""Terrain archetypes: parameter distributions and base-field synthesis.

The order of ``ARCHETYPES`` defines label ids and is a public contract (stored
datasets and trained models depend on it). Append new archetypes; never reorder.

Heights are in normalized units (1.0 = WorldSpec.max_height_m). Sea level is
taken from the WorldSpec, so islands/lakes are defined relative to it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from nullscape.terrain.noise import Perlin, billow, fbm, ridged, smoothstep, warp

ARCHETYPES: tuple[str, ...] = ("plains", "hills", "mountains", "ridges", "islands", "mesas")

Params = dict[str, Any]


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _i(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def sample_params(name: str, rng: np.random.Generator) -> Params:
    """Sample generator parameters for one map of archetype ``name``."""
    if name == "plains":
        p = dict(base=_u(rng, 0.21, 0.34), amp=_u(rng, 0.04, 0.12), freq=_u(rng, 0.4, 1.0), gain=_u(rng, 0.50, 0.60),
                 warp_km=_u(rng, 0.0, 0.3), erosion_k=_u(rng, 2.0, 5.0), erosion_iters=_i(rng, 10, 25),
                 diffusion_iters=_i(rng, 1, 4), talus_deg=_u(rng, 30, 40))
    elif name == "hills":
        p = dict(base=_u(rng, 0.28, 0.42), amp=_u(rng, 0.20, 0.40), freq=_u(rng, 0.6, 1.4), gain=_u(rng, 0.52, 0.62),
                 billow_mix=_u(rng, 0.0, 0.6), warp_km=_u(rng, 0.1, 0.4), erosion_k=_u(rng, 6.0, 14.0),
                 erosion_iters=_i(rng, 20, 35), diffusion_iters=_i(rng, 0, 2), talus_deg=_u(rng, 34, 42))
    elif name == "mountains":
        p = dict(base=_u(rng, 0.20, 0.34), amp=_u(rng, 0.70, 1.00), freq=_u(rng, 0.4, 0.9), gain=_u(rng, 0.52, 0.60),
                 ridge_mix=_u(rng, 0.3, 0.7), envelope_freq=_u(rng, 0.15, 0.35), warp_km=_u(rng, 0.1, 0.5),
                 erosion_k=_u(rng, 10.0, 25.0), erosion_iters=_i(rng, 30, 50), diffusion_iters=_i(rng, 0, 1),
                 talus_deg=_u(rng, 38, 45))
    elif name == "ridges":
        p = dict(base=_u(rng, 0.18, 0.30), amp=_u(rng, 0.50, 0.80), freq=_u(rng, 0.4, 0.9), gain=_u(rng, 0.50, 0.60),
                 sharpness=_u(rng, 1.5, 2.5), stretch=_u(rng, 1.5, 3.0), angle=_u(rng, 0.0, np.pi),
                 warp_km=_u(rng, 0.05, 0.3), erosion_k=_u(rng, 8.0, 20.0), erosion_iters=_i(rng, 25, 45),
                 diffusion_iters=_i(rng, 0, 1), talus_deg=_u(rng, 38, 45))
    elif name == "islands":
        n = int(rng.choice([1, 1, 2, 3, 4]))
        p = dict(n_centers=n, centers=rng.uniform(0.2, 0.8, (n, 2)).tolist(),
                 radii=(rng.uniform(0.8, 1.6, n) / np.sqrt(n) * 1.3).tolist(), floor=_u(rng, 0.04, 0.12),
                 peak=_u(rng, 0.20, 0.45), shape_power=_u(rng, 1.0, 2.0), freq=_u(rng, 0.8, 1.6),
                 gain=_u(rng, 0.50, 0.60), warp_km=_u(rng, 0.4, 0.9), coast_noise=_u(rng, 0.2, 0.5),
                 ridge_mix=_u(rng, 0.0, 0.6), erosion_k=_u(rng, 6.0, 15.0),
                 erosion_iters=_i(rng, 20, 40), diffusion_iters=_i(rng, 0, 2), talus_deg=_u(rng, 35, 42))
    elif name == "mesas":
        p = dict(base=_u(rng, 0.22, 0.36), amp=_u(rng, 0.35, 0.60), freq=_u(rng, 0.4, 0.9), gain=_u(rng, 0.45, 0.55),
                 levels=_i(rng, 2, 5), terrace_sharpness=_u(rng, 3.0, 8.0), warp_km=_u(rng, 0.1, 0.4),
                 erosion_k=_u(rng, 2.0, 6.0), erosion_iters=_i(rng, 10, 20), diffusion_iters=0,
                 talus_deg=_u(rng, 50, 60))
    else:
        raise KeyError(f"unknown archetype {name!r}; expected one of {ARCHETYPES}")
    p["octaves"] = 8
    return p


def _plains(p, nz, x, y, nyq, extent_km, sea):
    return p["base"] + p["amp"] * fbm(nz[0], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)


def _hills(p, nz, x, y, nyq, extent_km, sea):
    f = fbm(nz[0], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    b = 2.0 * billow(nz[1], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq) - 1.0
    return p["base"] + p["amp"] * ((1 - p["billow_mix"]) * f + p["billow_mix"] * b)


def _mountains(p, nz, x, y, nyq, extent_km, sea):
    r = ridged(nz[0], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    f = 0.5 + 0.5 * fbm(nz[1], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    env = smoothstep(-0.4, 0.6, fbm(nz[2], x, y, p["envelope_freq"], 3, nyquist=nyq))
    n01 = p["ridge_mix"] * r + (1 - p["ridge_mix"]) * f
    return p["base"] + p["amp"] * (n01 * (0.35 + 0.65 * env) - 0.2)


def _ridges(p, nz, x, y, nyq, extent_km, sea):
    c, s = np.cos(p["angle"]), np.sin(p["angle"])
    xr = (x * c + y * s) / p["stretch"]
    yr = -x * s + y * c
    r = ridged(nz[0], xr, yr, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq, sharpness=p["sharpness"])
    f = 0.5 + 0.5 * fbm(nz[1], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    return p["base"] + p["amp"] * (0.85 * r + 0.15 * f - 0.15)


def _islands(p, nz, x, y, nyq, extent_km, sea):
    coast = fbm(nz[1], x, y, 1.5 * p["freq"], 5, gain=0.55, nyquist=nyq)
    mask = np.zeros_like(x)
    for (cx, cy), rad in zip(p["centers"], p["radii"]):
        d = np.hypot(x - cx * extent_km, y - cy * extent_km) * (1.0 + p["coast_noise"] * coast)
        mask = np.maximum(mask, 1.0 - smoothstep(0.2 * rad, rad, d))
    land = mask ** p["shape_power"]
    f = 0.5 + 0.5 * fbm(nz[0], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    r = ridged(nz[2], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    relief = (1 - p["ridge_mix"]) * f + p["ridge_mix"] * r
    return p["floor"] + p["peak"] * land * (0.4 + 1.2 * relief) + 0.06 * (f - 0.5)


def _mesas(p, nz, x, y, nyq, extent_km, sea):
    n01 = 0.5 + 0.5 * fbm(nz[0], x, y, p["freq"], p["octaves"], gain=p["gain"], nyquist=nyq)
    t = np.clip(n01, 0.0, 1.0) * p["levels"]
    terr = (np.floor(t) + (t - np.floor(t)) ** p["terrace_sharpness"]) / p["levels"]
    return p["base"] + p["amp"] * (0.85 * terr + 0.15 * n01 - 0.3)


_SYNTH: dict[str, Callable[..., np.ndarray]] = {
    "plains": _plains, "hills": _hills, "mountains": _mountains,
    "ridges": _ridges, "islands": _islands, "mesas": _mesas,
}


def synthesize(name: str, p: Params, rng: np.random.Generator, x: np.ndarray, y: np.ndarray,
               nyquist: float, extent_km: float, sea_level: float) -> np.ndarray:
    """Base (pre-erosion) height field in normalized units on km coordinates ``x, y``."""
    noises = [Perlin(rng) for _ in range(5)]
    wx, wy = warp(noises[3], noises[4], x, y, p["warp_km"], base_freq=0.4, nyquist=nyquist)
    return _SYNTH[name](p, noises, wx, wy, nyquist, extent_km, sea_level)
