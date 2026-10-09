from __future__ import annotations

import hashlib
import json
import random
import signal
import threading
import warnings
from pathlib import Path

import numpy as np
import torch

RECIPE_KEYS = ("batch_size", "lr", "weight_decay", "betas", "warmup_steps", "max_steps", "grad_clip",
               "ema_decay", "bf16", "lr_schedule", "min_lr_ratio", "height_param", "archetype_loss_weights",
               "data_on_gpu", "prior_bank_per_class")


def dataset_fingerprint(store) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(store.manifest, sort_keys=True).encode())
    for name in ("heights.npy", "conditions.npy", "labels.npy", "split_train.npy", "split_val.npy", "split_test.npy"):
        digest.update(name.encode())
        with (store.root / name).open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG restoration is unavailable")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def validate_resume(checkpoint: dict, cfg: dict, meta: dict, unet: dict, diffusion: dict,
                    *, allow_inexact: bool = False) -> None:
    if checkpoint["unet_config"] != unet or checkpoint["diffusion_config"] != diffusion:
        raise ValueError("resume model/diffusion configuration differs; use train.init_from for a new experiment")
    old = checkpoint["dataset"]
    for key in ("world", "condition_keys", "condition_stats", "archetypes", "config_sha256", "generator_version"):
        if old.get(key) != meta.get(key):
            raise ValueError(f"resume dataset {key} differs from the checkpoint")
    old_kind = (old.get("height_param") or {"kind": "absolute"})["kind"]
    if old_kind != meta["height_param"]["kind"]:
        raise ValueError("resume dataset height parameterization differs; use train.init_from")
    state = checkpoint.get("training_state")
    if state is None:
        if not allow_inexact:
            raise ValueError("legacy checkpoint lacks complete resume state; use --allow-inexact-resume or train.init_from")
        warnings.warn("Legacy resume: RNG/batcher/scheduler state cannot be recovered exactly", RuntimeWarning)
    elif state.get("version") != 1 or old.get("fingerprint") != meta.get("fingerprint"):
        raise ValueError("resume dataset fingerprint or training-state version differs")
    from nullscape.train.trainer import DEFAULT_TRAIN

    previous = checkpoint["train_config"]
    before = {**DEFAULT_TRAIN, **previous.get("train", {})}
    after = {**DEFAULT_TRAIN, **cfg["train"]}
    changed = [f"train.{k}" for k in RECIPE_KEYS if before[k] != after[k]]
    if previous.get("seed", 0) != cfg.get("seed", 0):
        changed.append("seed")
    if changed:
        raise ValueError(f"resume recipe changed ({', '.join(changed)}); use train.init_from for intentional changes")
    if not 0 <= checkpoint["step"] < cfg["train"]["max_steps"]:
        raise ValueError("resume checkpoint is already at/after max_steps; use train.init_from for further training")


def state_dict(scheduler, batcher, *, best_score: float, best_step: int | None, losses, device) -> dict:
    return {"version": 1, "scheduler": scheduler.state_dict(), "batcher": batcher.state_dict(), "rng": capture_rng(),
            "best_score": best_score, "best_step": best_step, "losses": [float(x.detach()) for x in losses],
            "device": str(device)}


class TrainingInterrupted(KeyboardInterrupt):
    def __init__(self, run_path: Path, step: int):
        self.run_path = run_path
        self.checkpoint = run_path / "checkpoints" / "last.pt"
        super().__init__(f'Training stopped after step {step}. Resume with --resume "{self.checkpoint}"')


class StopRequest:
    def __init__(self):
        self.requested = False
        self.handlers = {}

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt("Forced stop; use the last complete checkpoint")
        self.requested = True
        print("Stop requested. Finishing the current training/evaluation cycle before saving recovery state.", flush=True)

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", signal.SIGINT)):
                if sig not in self.handlers:
                    self.handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
