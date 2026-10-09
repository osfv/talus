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
    save_condition_scatter,
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


def test_save_condition_scatter(tmp_path):
    rng = np.random.default_rng(0)
    keys = ["relief", "mean_slope_deg", "water_fraction"]
    requested = rng.random((16, 3))
    measured = requested + rng.normal(0, 0.05, (16, 3))
    adherence = {k: {"pearson_r": 0.9, "nmae": 0.1} for k in keys}
    baseline = {k: {"pearson_r": 0.0, "nmae": 0.5} for k in keys}
    _assert_png(save_condition_scatter(requested, measured, keys, tmp_path / "scatter.png",
                                       adherence, baseline, title="t"))
    _assert_png(save_condition_scatter(requested, measured, keys, tmp_path / "scatter_plain.png"))


def test_cli_export_smoke(tmp_path):
    smoke = Path(__file__).resolve().parents[1] / "data" / "smoke"
    if not (smoke / "manifest.json").exists():
        pytest.skip("data/smoke dataset not built")
    out = tmp_path / "ex"
    main(["export", "--dataset", str(smoke), "--index", "0", "--out", str(out)])
    for ext in ("png", "r16", "npy", "obj", "json"):
        assert (out / f"smoke_0.{ext}").exists(), ext


def test_checkpoint_alias_and_run_directory_resolution(tmp_path, monkeypatch):
    from nullscape.utils import paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    alias = tmp_path / paths.CHECKPOINT_ALIASES["talus-2"]
    alias.parent.mkdir(parents=True)
    alias.touch()
    assert paths.resolve_checkpoint("Talus-2") == alias
    run = tmp_path / "run"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "best.pt").touch()
    (ck / "last.pt").touch()
    assert paths.resolve_checkpoint(run) == ck / "best.pt"
    assert paths.resolve_checkpoint(run, prefer="last") == ck / "last.pt"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for name in ("step_2.pt", "step_10.pt", "step_backup.pt"):
        (snapshots / name).touch()
    assert paths.resolve_checkpoint(snapshots) == snapshots / "step_10.pt"
    with pytest.raises(FileNotFoundError, match="checkpoints"):
        paths.resolve_checkpoint("talus-does-not-exist")


def test_output_folders_are_unique_and_protect_existing_files(tmp_path):
    from nullscape.utils.paths import prepare_output_dir

    a = prepare_output_dir(None, default_root=tmp_path)
    b = prepare_output_dir(None, default_root=tmp_path)
    assert a != b and a.is_dir() and b.is_dir()
    (a / "keep.txt").write_text("original")
    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_output_dir(a)
    assert (a / "keep.txt").read_text() == "original"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prepare_output_dir(empty) == empty


def test_rundir_handles_timestamp_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr("nullscape.utils.tracking.time.strftime", lambda *_: "20000101-000000")
    a = RunDir("same", {}, root=tmp_path, tensorboard=False)
    b = RunDir("same", {}, root=tmp_path, tensorboard=False)
    try:
        assert a.path != b.path and a.path.is_dir() and b.path.is_dir()
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("options", [
    ["--n", "0"], ["--batch-size", "0"], ["--steps", "1"], ["--guidance", "nan"],
    ["--prop", "relief=oops"], ["--prop", "water_fraction=2"], ["--prop", "water_faction=0.2"],
    ["--prop", "relief=0.1", "--prop", "relief=0.2"], ["--formats", "wrong"],
    ["--seeds", "1,nope"], ["--seed", "1", "--seeds", "2"], ["--n", "3", "--seeds", "1,2"],
])
def test_sample_invalid_input_fails_before_checkpoint_load(options, tmp_path, monkeypatch, capsys):
    from nullscape.inference.sampler import TerrainSampler

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input must fail before loading weights")

    monkeypatch.setattr(TerrainSampler, "from_checkpoint", forbidden)
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--checkpoint", "unused.pt", "--out", str(tmp_path / "out"), *options])
    assert exc.value.code == 2
    error = capsys.readouterr().err.lower()
    assert "error:" in error and "not found" not in error


def test_cli_errors_are_short_and_debug_preserves_tracebacks(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--checkpoint", str(tmp_path / "missing.pt")])
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):
        main(["sample", "--checkpoint", str(tmp_path / "missing.pt"), "--debug"])


def test_training_option_typo_rejected_before_loading_dataset():
    from nullscape.train.trainer import train

    with pytest.raises(ValueError, match="batch_szie"):
        train({"dataset": "missing_dataset", "train": {"batch_szie": 4}}, dry_run=True)


@pytest.mark.parametrize("command", ["evaluate", "compare", "sampler-sweep", "benchmark-sampling"])
def test_cli_report_commands_refuse_nonempty_output(command, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "report.json").write_text("keep this")
    with pytest.raises(SystemExit) as exc:
        main([command, "--checkpoint", "unused.pt", "--out", str(out)])
    assert exc.value.code == 2 and (out / "report.json").read_text() == "keep this"


def test_cli_cuda_unavailable_has_actionable_error(tmp_path, monkeypatch, capsys):
    import torch

    checkpoint = tmp_path / "fake.pt"
    checkpoint.touch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--checkpoint", str(checkpoint), "--device", "cuda", "--out", str(tmp_path / "out")])
    assert exc.value.code == 2 and "--device cpu" in capsys.readouterr().err


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"max_steps": -1}, {"eval_every": 0},
                                     {"lr_schedule": "cosnie"}, {"lr": float("nan")}])
def test_invalid_training_settings_fail_before_dataset_loading(options):
    from nullscape.train.trainer import train

    with pytest.raises(ValueError):
        train({"dataset": "missing_dataset", "train": options}, dry_run=True)


def test_yaml_root_must_be_a_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- this\n- is a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
