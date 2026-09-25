import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from nullscape.eval.core import condition_adherence, evaluate_generated, memorization, split_halves
from nullscape.metrics.distribution import metric_table
from nullscape.metrics.quality import CONDITION_KEYS, compute_metrics, condition_vector
from nullscape.world import WorldSpec
from terrain_helpers import spectral_fbm

W = WorldSpec(resolution=64)


@pytest.fixture(scope="module")
def ref():
    maps = np.stack([spectral_fbm(2.5 + 0.1 * (s % 10), seed=s) for s in range(40)])
    labels = np.arange(40) % 2
    a, b = split_halves(40, seed=0)
    return maps, labels, a, b


def test_split_halves_disjoint_equal():
    a, b = split_halves(11, seed=3)
    assert len(a) == len(b) == 5 and not set(a) & set(b)


def test_real_data_scores_at_floor_and_blur_scores_worse(ref):
    maps, labels, a, b = ref
    real = evaluate_generated(maps[a], maps[a], maps[b], W, gen_labels=labels[a], ref_a_labels=labels[a],
                              ref_b_labels=labels[b], min_per_archetype=5)
    assert real["ratio_to_floor"]["metric_w1_mean"] == pytest.approx(1.0)
    assert all(v["ratio"] == pytest.approx(1.0) for v in real["per_archetype"].values())
    blurred = np.stack([gaussian_filter(m, 2.0) for m in maps[a]])
    bad = evaluate_generated(blurred, maps[a], maps[b], W)
    assert bad["ratio_to_floor"]["metric_w1_mean"] > 2.0
    assert bad["ratio_to_floor"]["rapsd_distance"] > 2.0
    with pytest.raises(ValueError):
        evaluate_generated(maps[:3], maps[a], maps[b], W)


def test_condition_adherence_perfect_vs_shuffled(ref):
    maps, *_ = ref
    table = metric_table(maps, W)
    req = np.stack([condition_vector(compute_metrics(m, W)) for m in maps])
    perfect = condition_adherence(table, req)
    assert all(v["mae"] < 1e-4 for v in perfect.values())  # float32 conditions
    assert np.isnan(perfect["water_fraction"]["pearson_r"])  # fixture has no water: correlation undefined
    assert all(v["pearson_r"] > 0.999 for k, v in perfect.items() if k != "water_fraction")
    known = np.ones_like(req, dtype=bool)
    known[:, CONDITION_KEYS.index("relief")] = False
    partial = condition_adherence(table, req[::-1], known)
    assert "relief" not in partial and partial["spectral_beta"]["mae"] > 0.05


def test_memorization_flags_copies(ref):
    maps, *_ = ref
    train, held = maps[:30], maps[30:]
    copies = np.stack([np.rot90(m) for m in train[:10]])
    m = memorization(copies, held, train)
    assert m["gen_nn_rmse_median"] < 1e-3 and m["frac_gen_below_heldout_p01"] == 1.0
    m2 = memorization(held, held, train)
    assert m2["nn_median_ratio"] == pytest.approx(1.0)
