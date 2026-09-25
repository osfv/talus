import json
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


@pytest.mark.slow
def test_end_to_end_train_sample_evaluate_compare(tmp_path, monkeypatch):
    monkeypatch.setenv("NULLSCAPE_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # hardware-independent: force CPU

    ds_root = build_dataset(DS_CFG, workers=1, out_root=tmp_path, progress=False)

    from nullscape.train.trainer import train

    cfg = {
        "name": "tsmoke",
        "seed": 0,
        "dataset": str(ds_root),
        "model": {"base_channels": 32, "channel_mults": [1, 2], "num_res_blocks": 1,
                  "attention_resolutions": [8], "dropout": 0.0},
        "train": {"batch_size": 8, "lr": 1e-3, "warmup_steps": 1, "max_steps": 4,
                  "log_every": 2, "val_every": 2, "val_size": 8, "sample_every": 4,
                  "eval_every": 4, "eval_samples": 4, "eval_steps": 3,
                  "checkpoint_every": 4, "bf16": False, "data_on_gpu": False},
    }
    run = train(cfg)

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

    last = run / "checkpoints" / "last.pt"
    assert last.exists() and (run / "checkpoints" / "best.pt").exists()
    assert (run / "samples" / "step_0000004.png").exists()
    rep = json.loads((run / "eval" / "step_0000004.json").read_text(encoding="utf-8"))
    assert "ratio_to_floor" in rep

    # CLI: sample
    s_out = tmp_path / "samples_out"
    main(["sample", "--checkpoint", str(last), "--n", "2", "--steps", "3",
          "--out", str(s_out), "--formats", "png16,r16"])
    for ext in ("png", "r16", "json"):
        assert (s_out / f"sample_0_000.{ext}").exists(), ext
    assert (s_out / "grid.png").exists() and (s_out / "samples.json").exists()

    # CLI: evaluate
    e_out = tmp_path / "eval_out"
    main(["evaluate", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "4",
          "--steps", "3", "--modes", "conditional,unconditional", "--out", str(e_out)])
    assert (e_out / "report.json").exists() and (e_out / "summary.md").exists()
    report = json.loads((e_out / "report.json").read_text(encoding="utf-8"))
    assert set(report["modes"]) == {"conditional", "unconditional"}
    assert (e_out / "generated_conditional.npy").exists()

    # CLI: compare
    c_out = tmp_path / "compare_out"
    main(["compare", "--checkpoint", str(last), "--dataset", str(ds_root), "--n", "2",
          "--seeds", "2", "--steps", "3", "--n-3d", "1", "--out", str(c_out)])
    assert (c_out / "compare_2d.png").exists() and (c_out / "compare_3d.png").exists()
