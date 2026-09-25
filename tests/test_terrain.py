import numpy as np
import pytest

from nullscape.terrain.archetypes import ARCHETYPES
from nullscape.terrain.erosion import (
    d8_receivers,
    drainage_area,
    hillslope_diffusion,
    stream_power_erosion,
    thermal_erosion,
)
from nullscape.terrain.generator import GeneratorConfig, generate
from nullscape.terrain.noise import Perlin, fbm, grid_km
from nullscape.world import WorldSpec
from terrain_helpers import spectral_fbm

W = WorldSpec(resolution=32, extent_m=4096.0, max_height_m=1200.0, sea_level=0.2)


def test_archetype_order_is_public_contract():
    assert ARCHETYPES == ("plains", "hills", "mountains", "ridges", "islands", "mesas")


def test_generate_bit_identical_for_same_seed_and_index():
    a, ra = generate(7, 12345, W)
    b, rb = generate(7, 12345, W)
    assert np.array_equal(a, b)
    assert ra["params"] == rb["params"]


def test_generate_differs_for_different_index():
    a, _ = generate(7, 12345, W)
    b, _ = generate(8, 12345, W)
    assert not np.array_equal(a, b)


def test_generate_output_contract():
    h, rec = generate(0, 12345, W64 := WorldSpec(resolution=64))
    assert h.dtype == np.float32 and h.shape == (W64.resolution, W64.resolution)
    assert 0.0 <= float(h.min()) and float(h.max()) <= 1.0
    for key in ("archetype", "archetype_id", "params", "clipped_fraction"):
        assert key in rec
    assert ARCHETYPES[rec["archetype_id"]] == rec["archetype"]


def test_generate_forced_archetype():
    for i, name in enumerate(ARCHETYPES):
        _, rec = generate(0, 12345, W, archetype=name)
        assert rec["archetype"] == name and rec["archetype_id"] == i


def test_generator_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        GeneratorConfig.from_dict({"supersample": 2, "bogus_key": 1})
    cfg = GeneratorConfig.from_dict({"supersample": 1, "erosion": False})
    assert cfg.supersample == 1 and cfg.erosion is False


def test_no_per_map_normalization():
    """Plains must stay low-relief; mountains must have clearly more relief.

    Per-map min-max normalization would stretch plains to the full [0, 1] range
    and erase the difference, so this also guards the global-normalization rule.
    """
    plains = [generate(i, 2024, W, archetype="plains")[0] for i in range(10)]
    mountains = [generate(i, 2024, W, archetype="mountains")[0] for i in range(10)]

    def relief(h: np.ndarray) -> float:
        p2, p98 = np.percentile(h, [2, 98])
        return float(p98 - p2)

    pr = np.mean([relief(h) for h in plains])
    mr = np.mean([relief(h) for h in mountains])
    assert pr < 0.5 * mr, f"plains relief {pr:.3f} vs mountains {mr:.3f}"
    assert any(float(h.max()) < 0.6 for h in plains)


def test_perlin_deterministic_for_same_seed():
    x, y = grid_km(32, W.extent_m)
    a = Perlin(np.random.default_rng(11))(x, y)
    b = Perlin(np.random.default_rng(11))(x, y)
    assert np.array_equal(a, b)


def test_fbm_range():
    x, y = grid_km(32, W.extent_m)
    v = fbm(Perlin(np.random.default_rng(3)), x, y, base_freq=0.6, octaves=8)
    assert float(v.max()) <= 1.5 and float(v.min()) >= -1.5


def _field_m(seed: int = 0, n: int = 32) -> np.ndarray:
    return spectral_fbm(3.0, seed=seed, n=n) * 500.0


def test_thermal_erosion_conserves_mass():
    z = _field_m()
    out = thermal_erosion(z, cell=W.cell_size_m, talus_deg=35.0, iterations=5)
    assert out.shape == z.shape
    assert float(out.sum()) == pytest.approx(float(z.sum()), rel=1e-9)


def test_hillslope_diffusion_conserves_mass():
    z = _field_m()
    out = hillslope_diffusion(z, iterations=3, rate=0.2)
    assert float(out.sum()) == pytest.approx(float(z.sum()), rel=1e-9)


def test_stream_power_only_lowers_and_respects_base_level():
    z = _field_m(seed=1)
    base = 240.0  # cuts through the field, so both sides are exercised
    out = stream_power_erosion(z, cell=W.cell_size_m, iterations=3, k=10.0, base_level=base)
    assert (out <= z + 1e-9).all()
    assert np.array_equal(out[z <= base], z[z <= base])


def test_drainage_area_on_tilted_plane():
    n, cell = 16, 64.0
    z = (np.arange(n, dtype=np.float64)[None, :] * 10.0).repeat(n, axis=0)
    recv, _ = d8_receivers(z, cell)
    area = drainage_area(recv, cell * cell)
    assert (area >= cell * cell - 1e-9).all()
    outlets = recv < 0
    assert outlets.sum() > 0
    assert float(area[outlets].sum()) == pytest.approx(n * n * cell * cell, rel=1e-9)
