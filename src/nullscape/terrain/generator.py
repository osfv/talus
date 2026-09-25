"""Procedural heightmap generator: archetype synthesis -> optional blend -> erosion.

Every map is a pure function of (global_seed, index, world, config): the RNG is
``SeedSequence([global_seed, index])``, so datasets are reproducible and
independent of worker count or ordering.

Output heights use GLOBAL normalization: meters / world.max_height_m, clipped
to [0, 1]. Maps are never min-max normalized individually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nullscape.terrain.archetypes import ARCHETYPES, sample_params, synthesize
from nullscape.terrain.erosion import hillslope_diffusion, stream_power_erosion, thermal_erosion
from nullscape.terrain.noise import Perlin, fbm, grid_km, smoothstep
from nullscape.world import WorldSpec

GENERATOR_VERSION = "1.0.0"


@dataclass
class GeneratorConfig:
    supersample: int = 2
    blend_probability: float = 0.25
    archetype_weights: dict[str, float] = field(default_factory=lambda: {a: 1.0 for a in ARCHETYPES})
    thermal_iters: int = 5
    erosion: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GeneratorConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown generator config keys: {sorted(unknown)}")
        return cls(**d)

    def probabilities(self) -> np.ndarray:
        w = np.array([float(self.archetype_weights.get(a, 0.0)) for a in ARCHETYPES])
        if w.sum() <= 0:
            raise ValueError("archetype_weights must have positive total weight")
        return w / w.sum()


def _downsample(a: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return a
    r = a.shape[0] // factor
    return a.reshape(r, factor, r, factor).mean(axis=(1, 3))


def generate(
    index: int,
    global_seed: int,
    world: WorldSpec,
    cfg: GeneratorConfig | None = None,
    archetype: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one heightmap. Returns (h float32 [R, R] in [0, 1], record)."""
    cfg = cfg or GeneratorConfig()
    ss = np.random.SeedSequence([int(global_seed), int(index)])
    rng = np.random.default_rng(ss)

    primary = archetype or str(rng.choice(ARCHETYPES, p=cfg.probabilities()))
    params = sample_params(primary, rng)

    res_hi = world.resolution * cfg.supersample
    x, y = grid_km(res_hi, world.extent_m)
    extent_km = world.extent_m / 1000.0
    nyquist = 0.5 / (extent_km / res_hi)
    h = synthesize(primary, params, rng, x, y, nyquist, extent_km, world.sea_level)

    record: dict[str, Any] = {"archetype": primary, "archetype_id": ARCHETYPES.index(primary), "params": params}
    if rng.random() < cfg.blend_probability:
        secondary = str(rng.choice([a for a in ARCHETYPES if a != primary]))
        p2 = sample_params(secondary, rng)
        h2 = synthesize(secondary, p2, rng, x, y, nyquist, extent_km, world.sea_level)
        weight = float(rng.uniform(0.3, 0.7))
        m = weight * smoothstep(-0.3, 0.3, fbm(Perlin(rng), x, y, float(rng.uniform(0.15, 0.4)), 3, nyquist=nyquist))
        h = (1.0 - m) * h + m * h2
        record.update(secondary=secondary, secondary_params=p2, blend_weight=float(m.mean()))
    else:
        record.update(secondary=None, secondary_params=None, blend_weight=0.0)

    z = _downsample(h, cfg.supersample) * world.max_height_m
    if cfg.erosion:
        z = stream_power_erosion(z, world.cell_size_m, params["erosion_iters"], params["erosion_k"],
                                 base_level=world.sea_level * world.max_height_m)
        z = hillslope_diffusion(z, params["diffusion_iters"])
        z = thermal_erosion(z, world.cell_size_m, params["talus_deg"], cfg.thermal_iters)

    h = z / world.max_height_m
    record["clipped_fraction"] = float(((h < 0.0) | (h > 1.0)).mean())
    return np.clip(h, 0.0, 1.0).astype(np.float32), record
