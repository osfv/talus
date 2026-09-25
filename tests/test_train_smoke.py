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
    with pytest.raises(ValueError):
        main(["artifacts", "--run", str(run), "--out", str(art2), "--dataset", str(ds_root),
              "--device", "cpu", "--n", "4", "--steps", "5", "--batch-size", "8"])
