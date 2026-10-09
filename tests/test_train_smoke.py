import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nullscape.cli import main
from nullscape.data.build import build_dataset

DS_CFG = {
    "name": "tsmoke_ds",
    "seed": 11,
    "n": 48,
    "world": {"resolution": 16, "extent_m": 4096.0, "max_height_m": 1200.0, "sea_level": 0.2},
    "generator": {"supersample": 2, "blend_probability": 0.25},
    "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
}

TRAIN_CFG = {
    "name": "tsmoke",
    "seed": 0,
    "model": {"base_channels": 32, "channel_mults": [1, 2], "num_res_blocks": 1,
              "attention_resolutions": [8], "dropout": 0.0},
    "train": {"batch_size": 8, "lr": 1e-3, "warmup_steps": 1, "max_steps": 4,
              "log_every": 2, "val_every": 2, "val_size": 8, "sample_every": 4,
              "eval_every": 4, "eval_samples": 4, "eval_steps": 3,
              "checkpoint_every": 4, "bf16": False, "data_on_gpu": False,
              "keep_checkpoints": True, "artifacts_every": 2,
              "artifacts_spec": {"per_archetype": 1, "n_eval": 4, "n_3d": 1,
                                 "steps": 3, "batch_size": 8}},
}

# files generate_checkpoint_artifacts writes into each step dir
STEP_DIR_FILES = [
    "procedural_vs_learned.png", "compare_3d.png", "rapsd.png", "metric_histograms.png",
    "condition_adherence.png", "nearest_neighbors.png", "traversability.png",
    "report.json", "summary.md", "fixed_seed.npy", "fixed_seed_meta.json", "timing.json",
]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """One tiny end-to-end training run shared by the smoke tests, forced to CPU."""
    tmp = tmp_path_factory.mktemp("tsmoke")
    prev_env = os.environ.get("NULLSCAPE_RUNS_ROOT")
    prev_cuda = torch.cuda.is_available
    os.environ["NULLSCAPE_RUNS_ROOT"] = str(tmp / "runs")
    torch.cuda.is_available = lambda: False  # hardware-independent: force CPU
    try:
        ds_root = build_dataset(DS_CFG, workers=1, out_root=tmp, progress=False)

        from nullscape.train.trainer import train

        cfg = {**TRAIN_CFG, "dataset": str(ds_root),
               "train": {**TRAIN_CFG["train"], "artifacts_dir": str(tmp / "art")}}
        run = train(cfg)
        yield run, run / "checkpoints" / "last.pt", ds_root, tmp
    finally:
        torch.cuda.is_available = prev_cuda
        if prev_env is None:
            os.environ.pop("NULLSCAPE_RUNS_ROOT", None)
        else:
            os.environ["NULLSCAPE_RUNS_ROOT"] = prev_env


@pytest.mark.slow
def test_end_to_end_train_sample_evaluate_compare(trained_run):
    run, last, ds_root, tmp = trained_run

    # run directory artifacts
    assert (run / "config.yaml").exists() and (run / "env.json").exists()
    assert (run / "model.json").exists()
    model_info = json.loads((run / "model.json").read_text(encoding="utf-8"))
    assert model_info["parameters"] > 0 and model_info["unet"]["image_size"] == 16
    saved_cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert saved_cfg["dataset"] == str(ds_root) and saved_cfg["train"]["max_steps"] == 4

    metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {k for m in metrics for k in m}
    assert "loss" in keys and "val_loss" in keys

    assert last.exists() and (run / "checkpoints" / "best.pt").exists()
    assert (run / "samples" / "step_0000004.png").exists()
    rep = json.loads((run / "eval" / "step_0000004.json").read_text(encoding="utf-8"))
    assert "ratio_to_floor" in rep

    # CLI: sample
    s_out = tmp / "samples_out"
    main(["sample", "--checkpoint", str(last), "--n", "2", "--steps", "3",
          "--out", str(s_out), "--formats", "png16,r16"])
    for ext in ("png", "r16", "json"):
        assert (s_out / f"sample_0_000.{ext}").exists(), ext
    assert (s_out / "grid.png").exists() and (s_out / "samples.json").exists()

    # CLI: evaluate
    e_out = tmp / "eval_out"
    main(["evaluate", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "4",
          "--steps", "3", "--modes", "conditional,unconditional",
          "--spacing", "quadratic", "--eta", "0.5", "--out", str(e_out)])
    assert (e_out / "report.json").exists() and (e_out / "summary.md").exists()
    report = json.loads((e_out / "report.json").read_text(encoding="utf-8"))
    assert set(report["modes"]) == {"conditional", "unconditional"}
    assert report["spacing"] == "quadratic" and report["eta"] == 0.5
    assert (e_out / "generated_conditional.npy").exists()

    # CLI: compare
    c_out = tmp / "compare_out"
    main(["compare", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "2",
          "--seeds", "2", "--steps", "3", "--n-3d", "1", "--out", str(c_out)])
    assert (c_out / "compare_2d.png").exists() and (c_out / "compare_3d.png").exists()


@pytest.mark.slow
def test_sampler_sweep_cli(trained_run):
    _, last, ds_root, tmp = trained_run
    w_out = tmp / "sweep_out"
    main(["sampler-sweep", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "4",
          "--steps-list", "3", "--spacings", "uniform,quadratic",
          "--etas", "0", "--guidances", "1.5", "--out", str(w_out)])
    assert (w_out / "sampler_sweep.md").exists()
    sweep = json.loads((w_out / "sampler_sweep.json").read_text(encoding="utf-8"))
    assert len(sweep["rows"]) == 2 and "best" in sweep
    assert {r["spacing"] for r in sweep["rows"]} == {"uniform", "quadratic"}
    assert all("passes_gates" in row for row in sweep["rows"])


@pytest.mark.slow
def test_partial_repeated_evaluation_and_blind_comparison(trained_run):
    _, last, ds_root, tmp = trained_run
    out = tmp / "partial_repeats"
    main(["evaluate", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "4",
          "--steps", "3", "--modes", "property:water_fraction,label_only", "--repeat-seeds", "0,1",
          "--no-memorization", "--device", "cpu", "--out", str(out)])
    repeats = json.loads((out / "repeats.json").read_text())
    assert repeats["seeds"] == [0, 1] and "property:water_fraction" in repeats["modes"]
    report = json.loads((out / "seed_0" / "report.json").read_text())
    partial = report["modes"]["property:water_fraction"]
    assert set(partial["condition_adherence"]) == {"water_fraction"}
    assert "per_archetype" not in partial and "gameplay" in partial
    assert (out / "seed_0" / "generated_property_water_fraction.npy").exists()
    blind = tmp / "blind"
    main(["compare", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "2",
          "--steps", "3", "--seeds", "1", "--blind", "--out", str(blind)])
    key = json.loads((blind / "blind_key.json").read_text())
    assert len(key) == 4 and (blind / "blind_2d.png").exists()
    assert sorted(v["source"] for v in key.values()) == ["learned", "learned", "procedural", "procedural"]


@pytest.mark.slow
def test_spatial_sample_cli(trained_run):
    _, last, _, tmp = trained_run
    out = tmp / "spatial_samples"
    ref = np.full((16, 16), 0.4, np.float32)
    mask = np.zeros((16, 16), bool)
    mask[:, :4] = True
    np.save(tmp / "reference.npy", ref)
    np.save(tmp / "mask.npy", mask)
    main(["sample", "--checkpoint", str(last), "--n", "1", "--steps", "3", "--formats", "npy",
          "--reference", str(tmp / "reference.npy"), "--preserve-mask", str(tmp / "mask.npy"), "--out", str(out)])
    generated = np.load(out / "sample_0_000.npy")
    assert np.array_equal(generated[mask], ref[mask])
    metadata = json.loads((out / "samples.json").read_text())[0]
    assert metadata["spacing"] == "quadratic" and metadata["spatial"]["preserved_cells"] == 64


@pytest.mark.slow
def test_relative_weighted_finetune_preflight_writes_no_run(trained_run, monkeypatch):
    from nullscape.train.trainer import train

    _, last, ds_root, tmp = trained_run
    root = tmp / "preflight_runs"
    monkeypatch.setenv("NULLSCAPE_RUNS_ROOT", str(root))
    cfg = {**TRAIN_CFG, "dataset": str(ds_root),
           "train": {**TRAIN_CFG["train"], "init_from": str(last), "height_param": "relative",
                     "archetype_loss_weights": {"mountains": 2.0, "ridges": 2.0}}}
    result = train(cfg, dry_run=True)
    assert np.isfinite(result["loss"]) and not result["checkpoint_written"]
    assert len(result["losses"]) == 3 and np.isfinite(result["losses"]).all()
    assert result["height_param"] == "relative" and not root.exists()


@pytest.mark.slow
def test_single_map_benchmark_cli(trained_run):
    _, last, _, tmp = trained_run
    out = tmp / "single_map_perf"
    main(["benchmark-sampling", "--checkpoint", str(last), "--batches", "1", "--repeats", "2",
          "--warmup", "1", "--steps", "3", "--device", "cpu", "--out", str(out)])
    report = json.loads((out / "performance.json").read_text())
    assert report["rows"][0]["batch"] == 1 and len(report["rows"]) == 1
    assert len(report["checkpoint"]["sha256"]) == 64


@pytest.mark.slow
def test_explicit_seed_replay_without_previews(trained_run):
    run, _, _, tmp = trained_run
    first, replay = tmp / "explicit_seeds", tmp / "replay_seed"
    main(["sample", "--checkpoint", str(run), "--seeds", "42,99", "--steps", "3", "--device", "cpu",
          "--no-previews", "--formats", "npy", "--out", str(first)])
    metadata = json.loads((first / "samples.json").read_text())
    assert [m["sample_seed"] for m in metadata] == [42, 99]
    assert all(m["seed"] is None for m in metadata)
    assert metadata[0]["checkpoint"]["path"].endswith("best.pt")
    assert not (first / "grid.png").exists() and not (first / "inspect_0.png").exists()
    main(["sample", "--checkpoint", str(run), "--seeds", "99", "--steps", "3", "--device", "cpu",
          "--batch-size", "1", "--no-previews", "--formats", "npy", "--out", str(replay)])
    assert np.allclose(np.load(first / "sample_99_001.npy"), np.load(replay / "sample_99_000.npy"), atol=1e-5)
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--checkpoint", str(run), "--out", str(first)])
    assert exc.value.code == 2


@pytest.mark.slow
def test_checkpoint_info_is_cpu_only(trained_run, monkeypatch, capsys):
    _, last, _, _ = trained_run

    def no_gpu():
        raise AssertionError("checkpoint inspection must not query or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", no_gpu)
    capsys.readouterr()
    main(["checkpoint-info", "--checkpoint", str(last)])
    info = json.loads(capsys.readouterr().out)
    assert info["parameters"] > 0 and info["world"]["resolution"] == 16
    assert info["sampling"]["spacing"] == "quadratic" and len(info["properties"]) == 5


@pytest.mark.slow
def test_trainer_checkpoint_artifacts(trained_run):
    """artifacts_every renders the per-checkpoint suite during training."""
    run, last, ds_root, tmp = trained_run
    art = tmp / "art"

    assert (run / "checkpoints" / "step_0000004.pt").exists()  # keep_checkpoints
    for step in (2, 4):  # artifacts_every=2, max_steps=4
        d = art / f"step_{step:07d}"
        for name in STEP_DIR_FILES:
            assert (d / name).exists(), f"{d.name}/{name}"

    index = (art / "index.md").read_text(encoding="utf-8")
    for s in (2, 4):
        assert f"| {s} |" in index, f"index.md missing row for step {s}"
    assert (art / "progression.png").exists() and (art / "progression_metrics.png").exists()
    assert (art / "spec.json").exists()

    a = np.load(art / "step_0000002" / "fixed_seed.npy")
    b = np.load(art / "step_0000004" / "fixed_seed.npy")
    assert a.shape == b.shape  # same fixed-seed set at every checkpoint


@pytest.mark.slow
def test_artifacts_cli(trained_run):
    """`nullscape artifacts` renders step_*.pt snapshots and enforces the frozen spec."""
    run, last, ds_root, tmp = trained_run
    art2 = tmp / "art2"
    main(["artifacts", "--run", str(run), "--out", str(art2), "--dataset", str(ds_root),
          "--device", "cpu", "--n", "4", "--steps", "3", "--batch-size", "8"])
    assert (art2 / "index.md").exists()
    d = art2 / "step_0000004"
    for name in STEP_DIR_FILES:
        assert (d / name).exists(), f"art2/{d.name}/{name}"

    # a different spec is rejected up front, even when every snapshot is already rendered
    with pytest.raises(SystemExit) as exc:
        main(["artifacts", "--run", str(run), "--out", str(art2), "--dataset", str(ds_root),
              "--device", "cpu", "--n", "4", "--steps", "5", "--batch-size", "8"])
    assert exc.value.code == 2
