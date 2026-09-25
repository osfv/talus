import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from nullscape.cli import main
from nullscape.utils.config import apply_overrides, load_config
from nullscape.utils.seed import derive_seed
from nullscape.utils.tracking import RunDir
from nullscape.viz.render import (
    save_grid,
    save_inspection,
    save_metric_histograms,
    save_rapsd,
)
from nullscape.world import WorldSpec
from terrain_helpers import spectral_fbm

W = WorldSpec(resolution=32, extent_m=4096.0, max_height_m=1200.0, sea_level=0.2)


def test_apply_overrides_nested_and_typed():
    cfg = apply_overrides(
        {"train": {"lr": 1e-4}, "name": "a"},
        ["train.lr=0.002", "train.opt.name=adamw", "new.flag=true", "vals=[1, 2, 3]", "ratio=0.5"],
    )
    assert cfg["train"]["lr"] == pytest.approx(0.002)
    assert cfg["train"]["opt"]["name"] == "adamw"
    assert cfg["new"]["flag"] is True
    assert cfg["vals"] == [1, 2, 3]
    assert cfg["ratio"] == pytest.approx(0.5)
    assert apply_overrides({}, ["lr=2e-3"])["lr"] == pytest.approx(2e-3)  # YAML 1.1 would give a string


def test_apply_overrides_rejects_malformed():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no-equals-sign"])
    with pytest.raises(ValueError):
        apply_overrides({"a": 5}, ["a.b=1"])


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"a": {"b": 1}}), encoding="utf-8")
    assert load_config(p, ["a.b=2"]) == {"a": {"b": 2}}


def test_derive_seed_stable_and_distinct():
    assert derive_seed(1, 2) == derive_seed(1, 2)
    assert derive_seed(1, 2) != derive_seed(2, 1)
    assert derive_seed(7) != derive_seed(8)
    assert 0 <= derive_seed(1, 2) < 2**63


def test_rundir_writes_artifacts(tmp_path):
    rd = RunDir("t", {"a": 1, "b": {"c": 2}}, root=tmp_path, tensorboard=False)
    rd.log(0, loss=1.0, acc=0.5)
    rd.log(1, loss=0.8)
    rd.close()
    run = rd.path
    assert (run / "config.yaml").exists() and (run / "env.json").exists()
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert cfg == {"a": 1, "b": {"c": 2}}
    lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["step"] == 0 and rec0["loss"] == pytest.approx(1.0)
    env = json.loads((run / "env.json").read_text(encoding="utf-8"))
    assert "python" in env and "git" in env


@pytest.fixture(scope="module")
def maps() -> list[np.ndarray]:
    return [spectral_fbm(3.0, seed=s, n=32) for s in range(4)]


def _assert_png(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_grid(maps, tmp_path):
    _assert_png(save_grid(maps, W, tmp_path / "grid.png", titles=["a", "b", "c", "d"]))


def test_save_inspection(maps, tmp_path):
    _assert_png(save_inspection(maps[0], W, tmp_path / "insp.png"))


def test_save_metric_histograms(tmp_path):
    rng = np.random.default_rng(0)
    keys = ["relief", "mean_slope_deg", "water_fraction"]
    tables = {
        "procedural": {k: rng.random(40) for k in keys},
        "learned": {k: rng.random(40) * 0.9 + 0.05 for k in keys},
    }
    _assert_png(save_metric_histograms(tables, tmp_path / "hists.png", keys=keys))


def test_save_rapsd(maps, tmp_path):
    _assert_png(save_rapsd({"a": maps[:2], "b": maps[2:]}, tmp_path / "rapsd.png"))


def test_cli_export_smoke(tmp_path):
    smoke = Path(__file__).resolve().parents[1] / "data" / "smoke"
    if not (smoke / "manifest.json").exists():
        pytest.skip("data/smoke dataset not built")
    out = tmp_path / "ex"
    main(["export", "--dataset", str(smoke), "--index", "0", "--out", str(out)])
    for ext in ("png", "r16", "npy", "obj", "json"):
        assert (out / f"smoke_0.{ext}").exists(), ext
