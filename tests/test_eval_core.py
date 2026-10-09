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


@pytest.mark.parametrize("mode,label_known", [("property:water_fraction", False),
                                              ("label+property:relief", True)])
def test_partial_mode_masks_only_requested_property(mode, label_known):
    from types import SimpleNamespace

    from nullscape.eval.cli import _mode_inputs

    st = SimpleNamespace(condition_keys=list(CONDITION_KEYS), conditions=np.arange(30).reshape(6, 5),
                         labels=np.arange(6))
    inputs = _mode_inputs(mode, st, np.array([1, 3]))
    j = list(CONDITION_KEYS).index(mode.split(":")[1])
    assert inputs["known"].sum() == 2 and inputs["known"][:, j].all()
    assert np.array_equal(inputs["cond_raw"][:, j], st.conditions[[1, 3], j])
    assert np.all(inputs["cond_raw"][~inputs["known"]] == 0)
    assert np.array_equal(inputs["labels"], [1, 3] if label_known else [-1, -1])


def test_repeat_summary_reports_seed_variation():
    from nullscape.eval.core import summarize_repeats

    summary = summarize_repeats([{"x": 1.0}, {"x": 3.0}, {"x": 5.0}])
    assert summary["x"] == {"n": 3, "mean": 3.0, "std": 2.0, "min": 1.0, "max": 5.0}


def test_spectral_bands_identical_are_zero(ref):
    from nullscape.eval.core import spectral_band_errors

    maps, *_ = ref
    assert all(v == 0 for v in spectral_band_errors(maps, maps).values())


def test_scorecard_speed_matches_guidance_interval():
    from benchmarks.scorecard import speed

    cfg = {"steps": 50, "guidance": 2.0, "spacing": "quadratic", "eta": 0.0, "guidance_interval": [0.05, 0.8]}
    row = {**cfg, "batch": 128, "maps_per_second": 7.0, "oom": False}
    assert speed({"rows": [row]}, cfg) == 7.0
    assert speed({"rows": [row]}, {**cfg, "guidance_interval": [0.0, 1.0]}) is None
    assert speed(None, cfg) is None


def test_artifact_interval_spec_roundtrip_and_legacy(tmp_path):
    import json
    from dataclasses import asdict

    from nullscape.eval.artifacts import ArtifactSpec, _check_spec

    limited = ArtifactSpec(guidance_interval=(0.05, 0.8))
    _check_spec(tmp_path / "limited", limited)
    _check_spec(tmp_path / "limited", limited)
    with pytest.raises(ValueError):
        _check_spec(tmp_path / "limited", ArtifactSpec())
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    old = asdict(ArtifactSpec())
    old.pop("guidance_interval")
    (legacy / "spec.json").write_text(json.dumps(old))
    _check_spec(legacy, ArtifactSpec())
    assert "guidance_interval" not in json.loads((legacy / "spec.json").read_text())
