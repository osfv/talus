import copy
import random
import signal

import numpy as np
import pytest
import torch

from nullscape.models.ema import EMA
from nullscape.models.unet import UNet, UNetConfig
from nullscape.train.trainer import TensorBatcher, save_checkpoint


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    model = UNet(UNetConfig(image_size=8, base_channels=32, channel_mults=(1,), num_res_blocks=1))
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous complete checkpoint")

    def broken_save(payload, file):
        if hasattr(file, "write"):
            file.write(b"incomplete")
        else:
            file.write_bytes(b"incomplete")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk failure"):
        save_checkpoint(path, model, EMA(model), torch.optim.AdamW(model.parameters()), 0,
                        {"_diffusion_config": {}}, {})
    assert path.read_bytes() == b"previous complete checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_rng_and_batcher_roundtrip():
    from nullscape.train.recovery import capture_rng, restore_rng

    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    h = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    b = TensorBatcher(h, np.zeros((4, 5)), np.arange(4), torch.device("cpu"), False, seed=8)
    state = capture_rng()
    batch_state = b.state_dict()
    expected = (random.random(), np.random.rand(), torch.rand(3), b.sample(4))
    restore_rng(state)
    b.load_state_dict(batch_state)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert all(torch.equal(a, z) for a, z in zip(b.sample(4), expected[3]))


def test_interrupt_is_deferred_and_handler_restored():
    from nullscape.train.recovery import StopRequest

    old = signal.getsignal(signal.SIGINT)
    with StopRequest() as stop:
        signal.raise_signal(signal.SIGINT)
        assert stop.requested
    assert signal.getsignal(signal.SIGINT) == old


def test_resume_compatibility_rejects_changed_dataset_and_recipe():
    from nullscape.train.recovery import validate_resume

    meta = {"world": {"resolution": 16}, "condition_keys": ["relief"], "condition_stats": {},
            "archetypes": ["plains"], "config_sha256": "data-a", "generator_version": "1",
            "fingerprint": "content-a", "height_param": {"kind": "relative"}}
    cfg = {"seed": 3, "train": {"batch_size": 4, "max_steps": 8, "height_param": "relative"}}
    ck = {"dataset": meta, "train_config": cfg, "unet_config": {"x": 1}, "diffusion_config": {"y": 2},
          "step": 4, "training_state": {"version": 1}}
    validate_resume(ck, cfg, meta, {"x": 1}, {"y": 2})
    for field, changed in [("fingerprint", "other"), ("condition_keys", ["mean_elevation"]),
                           ("world", {"resolution": 64})]:
        with pytest.raises(ValueError, match="dataset"):
            validate_resume(ck, cfg, {**meta, field: changed}, {"x": 1}, {"y": 2})
    changed = copy.deepcopy(cfg)
    changed["train"]["max_steps"] = 9
    with pytest.raises(ValueError, match="max_steps"):
        validate_resume(ck, changed, meta, {"x": 1}, {"y": 2})
    with pytest.raises(ValueError, match="legacy"):
        validate_resume({**ck, "training_state": None}, cfg, meta, {"x": 1}, {"y": 2})


@pytest.mark.slow
def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    from nullscape.data.build import build_dataset
    from nullscape.train.recovery import TrainingInterrupted
    from nullscape.train.trainer import train
    from test_train_smoke import DS_CFG, TRAIN_CFG

    monkeypatch.setenv("NULLSCAPE_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = build_dataset(DS_CFG, workers=1, out_root=tmp_path, progress=False)
    cfg = {**TRAIN_CFG, "dataset": str(dataset), "name": "recovery",
           "model": {**TRAIN_CFG["model"], "dropout": 0.1},
           "train": {**TRAIN_CFG["train"], "artifacts_every": 0, "lr_schedule": "cosine",
                     "eval_every": 2, "checkpoint_every": 2}}
    complete = train(cfg)
    original = TensorBatcher.sample
    calls = 0

    def interrupted_batch(self, *args, **kwargs):
        nonlocal calls
        batch = original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            signal.raise_signal(signal.SIGINT)
        return batch

    with monkeypatch.context() as patch:
        patch.setattr(TensorBatcher, "sample", interrupted_batch)
        with pytest.raises(TrainingInterrupted) as exc:
            train(cfg)
    checkpoint = exc.value.checkpoint
    original_bytes = checkpoint.read_bytes()
    resumed = train(cfg, resume=str(checkpoint))
    assert resumed != checkpoint.parent.parent and checkpoint.read_bytes() == original_bytes
    a = torch.load(complete / "checkpoints" / "last.pt", weights_only=False, map_location="cpu")
    b = torch.load(resumed / "checkpoints" / "last.pt", weights_only=False, map_location="cpu")
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
        assert torch.equal(a["ema"]["shadow"][key], b["ema"]["shadow"][key]), key
    assert a["training_state"]["scheduler"] == b["training_state"]["scheduler"]
    assert a["training_state"]["best_score"] == b["training_state"]["best_score"]
    assert torch.equal(a["training_state"]["batcher"]["rng"], b["training_state"]["batcher"]["rng"])
    assert (resumed / "checkpoints" / "best.pt").exists()


@pytest.mark.slow
def test_crash_during_eval_keeps_the_step_and_resumes_exactly(tmp_path, monkeypatch):
    """Talus-3 rugged lost steps 6000-9000 to an OOM in the step-9000 eval; that step must now survive."""
    import json

    import nullscape.train.trainer as trainer
    from nullscape.data.build import build_dataset
    from test_train_smoke import DS_CFG, TRAIN_CFG

    monkeypatch.setenv("NULLSCAPE_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = build_dataset(DS_CFG, workers=1, out_root=tmp_path, progress=False)
    cfg = {**TRAIN_CFG, "dataset": str(dataset), "name": "evalcrash",
           "model": {**TRAIN_CFG["model"], "dropout": 0.1},
           "train": {**TRAIN_CFG["train"], "artifacts_every": 0, "max_steps": 6, "sample_every": 100,
                     "eval_every": 3, "checkpoint_every": 6, "recovery_every": 2}}
    complete = trainer.train(cfg)
    original = trainer.evaluate_generated

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory (simulated)")

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "evaluate_generated", oom)
        with pytest.raises(RuntimeError, match="out of memory"):
            trainer.train(cfg)
    crashed = max((tmp_path / "runs").iterdir(), key=lambda p: p.stat().st_mtime)
    status = json.loads((crashed / "status.json").read_text())
    last = crashed / "checkpoints" / "last.pt"
    assert status["state"] == "failed" and status["step"] == 3
    assert torch.load(last, weights_only=False, map_location="cpu")["step"] == 3
    assert trainer.evaluate_generated is original
    resumed = trainer.train(cfg, resume=str(last))
    a = torch.load(complete / "checkpoints" / "last.pt", weights_only=False, map_location="cpu")
    b = torch.load(resumed / "checkpoints" / "last.pt", weights_only=False, map_location="cpu")
    assert a["step"] == b["step"] == 6
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
        assert torch.equal(a["ema"]["shadow"][key], b["ema"]["shadow"][key]), key
    assert torch.equal(a["training_state"]["batcher"]["rng"], b["training_state"]["batcher"]["rng"])


@pytest.mark.slow
def test_crash_inside_the_update_does_not_write_a_recovery_checkpoint(tmp_path, monkeypatch):
    import nullscape.train.trainer as trainer
    from nullscape.data.build import build_dataset
    from test_train_smoke import DS_CFG, TRAIN_CFG

    monkeypatch.setenv("NULLSCAPE_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = build_dataset(DS_CFG, workers=1, out_root=tmp_path, progress=False)
    cfg = {**TRAIN_CFG, "dataset": str(dataset), "name": "updatecrash",
           "train": {**TRAIN_CFG["train"], "artifacts_every": 0, "max_steps": 6, "sample_every": 100,
                     "eval_every": 100, "checkpoint_every": 6, "recovery_every": 2}}
    original = trainer.GaussianDiffusion.loss
    calls = 0

    def failing_loss(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("CUDA out of memory in backward (simulated)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(trainer.GaussianDiffusion, "loss", failing_loss)
    with pytest.raises(RuntimeError, match="backward"):
        trainer.train(cfg)
    run = next((tmp_path / "runs").iterdir())
    assert torch.load(run / "checkpoints" / "last.pt", weights_only=False, map_location="cpu")["step"] == 2
