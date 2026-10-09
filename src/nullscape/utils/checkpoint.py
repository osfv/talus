"""Checkpoint loading and release export.

Every checkpoint loads with ``torch.load(weights_only=True)``, which rebuilds tensors and plain Python
containers but cannot import or call anything else from the file. Training checkpoints also hold numpy RNG
state and torch's version string, so those types are allowlisted. Release checkpoints need no allowlist:
they keep the EMA weights, configs and dataset statistics, and drop the optimizer, raw weights, training
state and local paths (the dataset folder becomes its name, e.g. ``base64``).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.torch_version import TorchVersion

from nullscape.utils.paths import REPO_ROOT
from nullscape.utils.tracking import atomic_file

_NUMPY_CORE = getattr(np, "_core", None) or np.core
SAFE_GLOBALS = [_NUMPY_CORE.multiarray._reconstruct, np.ndarray, np.dtype, TorchVersion,
                *(type(np.dtype(t)) for t in (np.bool_, np.uint8, np.uint32, np.int64, np.float32, np.float64))]
RELEASE_DROP = ("model", "optimizer", "training_state", "best_score")


def load_checkpoint(path: str | Path, *, mmap: bool = False) -> dict[str, Any]:
    with torch.serialization.safe_globals(SAFE_GLOBALS):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def file_sha256(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _portable_path(value: str | None) -> str | None:
    """Repo-relative form of a local path, so a published file does not carry the author's folders."""
    if not value:
        return value
    path = Path(value)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def release_payload(ck: dict[str, Any], *, name: str, version: str, license: str,
                    source: str | Path) -> dict[str, Any]:
    payload = {k: v for k, v in ck.items() if k not in RELEASE_DROP}
    payload["dataset"] = dict(ck["dataset"])
    if payload["dataset"].get("root"):  # a dataset name resolves under NULLSCAPE_DATA_ROOT on any machine
        payload["dataset"]["root"] = Path(payload["dataset"]["root"]).name
    train_config = dict(ck.get("train_config", {}))
    if "train" in train_config:
        train = dict(train_config["train"])
        for key in ("init_from", "artifacts_dir"):
            if key in train:
                train[key] = _portable_path(train[key])
        train_config["train"] = train
    payload["train_config"] = train_config
    if "env" in ck:
        payload["env"] = {**ck["env"], **({"torch": str(ck["env"]["torch"])} if "torch" in ck["env"] else {})}
    payload["release"] = {"name": name, "version": version, "license": license,
                          "source": {"path": _portable_path(str(source)), "sha256": file_sha256(source),
                                     "step": int(ck["step"])},
                          "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return payload


def export_release(source: str | Path, out: str | Path, *, name: str, version: str, license: str) -> dict[str, Any]:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new path")
    payload = release_payload(load_checkpoint(source), name=name, version=version, license=license, source=source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_file(out) as file:
        torch.save(payload, file)
    torch.load(out, map_location="cpu", weights_only=True)  # no allowlist needed: tensors and plain values only
    return {"path": str(out), "sha256": file_sha256(out), "bytes": out.stat().st_size, **payload["release"]}
