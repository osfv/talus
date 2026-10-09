import json

import numpy as np
import pytest

from nullscape.eval.core import evaluate_generated
from nullscape.metrics.distribution import metric_table
from nullscape.metrics.quality import CONDITION_KEYS
from nullscape.world import WorldSpec
from terrain_helpers import spectral_fbm


@pytest.fixture
def paired_reports(tmp_path):
    world = WorldSpec(resolution=16)
    reference = np.stack([spectral_fbm(3, seed=i, n=16) for i in range(8)]).astype(np.float32)
    a, b = reference[:4], reference[4:]
    table = metric_table(a, world)
    req = np.stack([table[k] for k in CONDITION_KEYS], axis=1)
    inputs = {"config_sha256": "same-data", "condition_keys": list(CONDITION_KEYS), "conditions": req.tolist(),
              "labels": [0, 1, 0, 1], "archetypes": ["plains", "hills"], "seeds": [10, 11, 12, 13],
              "reference_std": [1.0] * 5}
    paths = []
    for name in ("baseline", "candidate"):
        path = tmp_path / name
        path.mkdir()
        maps = a.copy()
        if name == "candidate":
            maps[0] = np.clip(maps[0] + 0.15, 0, 1)
        report = {"checkpoint": {"path": f"{name}.pt", "step": 4}, "dataset": "example", "split": "val",
                  "n_per_half": 4, "reference_indices_a": list(range(4)), "reference_indices_b": list(range(4, 8)),
                  "world": world.to_dict(), "comparison_inputs": inputs, "steps": 3, "guidance": 2.0,
                  "spacing": "quadratic", "eta": 0.0,
                  "modes": {"conditional": evaluate_generated(maps, a, b, world, requested_conds=req)}}
        (path / "report.json").write_text(json.dumps(report))
        np.save(path / "generated_conditional.npy", maps)
        paths.append(path)
    return paths


def test_comparison_contains_measured_cases_and_safe_html(paired_reports, tmp_path):
    from nullscape.eval.comparison import build_comparison

    out = tmp_path / "view"
    report = build_comparison(*paired_reports, out, baseline_name="Talus-2", candidate_name="</script><script>alert(1)</script>")
    assert report["split"] == "val" and report["n"] == 4
    assert report["worst_adherence"][0] == 0 and report["worst_regressions"][0] == 0
    assert report["cases"][0]["candidate_error"] > report["cases"][0]["baseline_error"]
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script" in html and "comparison-data" in html
    assert (out / "comparison.json").exists()
    assert all(k in report for k in ("headline", "spectrum", "slopes"))


@pytest.mark.parametrize("field", ["seeds", "conditions", "reference_std"])
def test_comparison_rejects_unmatched_inputs(paired_reports, tmp_path, field):
    from nullscape.eval.comparison import build_comparison

    path = paired_reports[1] / "report.json"
    report = json.loads(path.read_text())
    report["comparison_inputs"][field][0] = 999
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="matched"):
        build_comparison(*paired_reports, tmp_path / "bad")


def test_comparison_rejects_different_samplers(paired_reports, tmp_path):
    from nullscape.eval.comparison import build_comparison

    path = paired_reports[1] / "report.json"
    report = json.loads(path.read_text())
    report["guidance"] = 1.0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="matched"):
        build_comparison(*paired_reports, tmp_path / "bad")
