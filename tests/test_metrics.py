import math

import numpy as np
import pytest

from nullscape.metrics.distribution import compare_sets, nearest_neighbor_rmse, rapsd_distance
from nullscape.metrics.quality import CONDITION_KEYS, compute_metrics, condition_vector, spectral_beta
from nullscape.metrics.traversability import AgentSpec, analyze, longest_route, shortest_path
from nullscape.world import WorldSpec
from terrain_helpers import spectral_fbm

W = WorldSpec(resolution=64, extent_m=4096.0, max_height_m=1200.0, sea_level=0.2)


def tilted_plane(deg: float, base: float = 0.3) -> np.ndarray:
    step = math.tan(math.radians(deg)) * W.cell_size_m / W.max_height_m
    return base + step * np.arange(W.resolution)[None, :].repeat(W.resolution, 0)


def test_worldspec_roundtrip():
    assert W.cell_size_m == 64.0
    assert WorldSpec.from_dict(W.to_dict()) == W
    with pytest.raises(ValueError):
        WorldSpec(sea_level=1.5)


def test_flat_map_metrics():
    m = compute_metrics(np.full((64, 64), 0.5), W)
    assert m["relief"] == 0.0 and m["mean_slope_deg"] == 0.0 and m["water_fraction"] == 0.0
    assert m["spectral_beta"] == 0.0 and m["checkerboard"] == 0.0
    assert compute_metrics(np.full((64, 64), 0.1), W)["water_fraction"] == 1.0


@pytest.mark.parametrize("deg", [5.0, 20.0, 40.0])
def test_plane_slope_is_exact(deg):
    m = compute_metrics(tilted_plane(deg, base=0.0), W)
    assert m["mean_slope_deg"] == pytest.approx(deg, abs=1e-6)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((32, 32)), W)


@pytest.mark.parametrize("beta", [2.0, 3.0, 4.0])
def test_spectral_beta_recovers_synthetic_exponent(beta):
    est = np.mean([spectral_beta(spectral_fbm(beta, seed=s)) for s in range(8)])
    assert est == pytest.approx(beta, abs=0.5)


def test_checkerboard_detector():
    board = 0.5 + 0.01 * np.where(np.add.outer(np.arange(64), np.arange(64)) % 2 == 0, 1.0, -1.0)
    assert compute_metrics(board, W)["checkerboard"] > 0.99
    assert compute_metrics(spectral_fbm(3.0), W)["checkerboard"] < 0.05
    assert compute_metrics(board, W)["hf_energy"] > 0.9
    assert compute_metrics(spectral_fbm(3.0), W)["hf_energy"] < 0.05


def test_metrics_dihedral_invariant():
    h = spectral_fbm(3.0, seed=3)
    a = compute_metrics(h, W)
    for t in (np.rot90(h), np.fliplr(h), np.rot90(h, 3).T):
        b = compute_metrics(np.ascontiguousarray(t), W)
        for k in a:
            assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-9), k


def test_condition_vector_order():
    m = compute_metrics(spectral_fbm(3.0), W)
    v = condition_vector(m)
    assert v.dtype == np.float32 and v.shape == (len(CONDITION_KEYS),)
    assert v[CONDITION_KEYS.index("relief")] == pytest.approx(m["relief"])


def test_traversability_flat_land_passes():
    r = analyze(np.full((64, 64), 0.5), W)
    assert r.passed and r.n_components == 1
    assert r.largest_component_fraction == 1.0 and r.connectivity_probability == 1.0 and r.spans_map


def test_traversability_all_water_fails():
    r = analyze(np.full((64, 64), 0.1), W)
    assert not r.passed and r.land_fraction == 0.0 and r.n_components == 0


def test_steep_plane_unwalkable_gentle_walkable():
    assert analyze(tilted_plane(10.0), W).walkable_land_fraction == 1.0
    assert analyze(tilted_plane(50.0), W).walkable_land_fraction == 0.0


@pytest.mark.parametrize("connectivity", [4, 8])
def test_cliff_splits_map_in_two(connectivity):
    h = np.full((64, 64), 0.3)
    h[:, 32:] = 0.8  # 600 m one-cell cliff
    r = analyze(h, W, AgentSpec(connectivity=connectivity))
    assert r.n_components == 2
    assert r.connectivity_probability == pytest.approx(0.5, abs=0.02)
    assert shortest_path(r, (10, 5), (10, 60)) is None
    path = shortest_path(r, (0, 0), (63, 20))
    assert path is not None and path[0] == (0, 0) and path[-1] == (63, 20)
    assert len(path) == (64 if connectivity == 8 else 84)


def test_water_channel_blocks_unless_passable():
    h = np.full((64, 64), 0.21)
    h[:, 30:34] = 0.195  # shallow 18 m channel: gentle banks, only the water blocks
    assert analyze(h, W).n_components == 2
    assert analyze(h, W, AgentSpec(water_passable=True)).n_components == 1


def test_longest_route_on_flat_map_is_corner_to_corner():
    route = longest_route(analyze(np.full((16, 16), 0.5), W.with_resolution(16)))
    assert route is not None and len(route) == 16


def test_nearest_neighbor_detects_rotated_copies():
    bank = np.stack([spectral_fbm(3.0, seed=s) for s in range(10)])
    queries = np.stack([np.rot90(bank[4]), np.fliplr(bank[7]), spectral_fbm(3.0, seed=99)])
    rmse, idx = nearest_neighbor_rmse(queries, bank, chunk=3)
    assert rmse[0] < 1e-3 and idx[0] == 4
    assert rmse[1] < 1e-3 and idx[1] == 7
    assert rmse[2] > 1e-2
    rmse_nd, _ = nearest_neighbor_rmse(queries[:1], bank, dihedral=False)
    assert rmse_nd[0] > 1e-2


def test_set_distances_zero_on_identical_sets_and_grow_with_shift():
    a = np.stack([spectral_fbm(3.0, seed=s) for s in range(12)])
    b = np.stack([spectral_fbm(2.0, seed=100 + s) for s in range(12)])
    assert rapsd_distance(a, a) == 0.0
    same = compare_sets(a, a, W)
    diff = compare_sets(b, a, W)
    assert same["metric_w1_mean"] == 0.0 and same["height_w1"] < 1e-3
    assert diff["metric_w1"]["spectral_beta"] > 1.0
    assert diff["rapsd_distance"] > 0.2
