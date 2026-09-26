"""Local-first experiment tracking: one directory per run.

<runs_root>/<YYYYmmdd-HHMMSS>_<name>/
    config.yaml     resolved config
    env.json        git commit / dirty flag, library versions, GPU
    metrics.jsonl   one JSON object per logged step
    tb/             TensorBoard event files
    checkpoints/, samples/, eval/
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from nullscape.utils.config import dump_config
from nullscape.utils.paths import REPO_ROOT, runs_root


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)

    try:
        head = run("rev-parse", "--verify", "HEAD")
        # a repository without commits prints the literal "HEAD" and fails: record no commit
        commit = head.stdout.strip() if head.returncode == 0 else ""
        dirty = bool(run("status", "--porcelain", "--untracked-files=no").stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    return {"commit": commit or None, "dirty": dirty}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git": git_info(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


class RunDir:
    def __init__(self, name: str, config: dict[str, Any], root: Path | None = None, tensorboard: bool = True):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(root or runs_root()) / f"{stamp}_{name}"
        self.path.mkdir(parents=True, exist_ok=False)
        for sub in ("checkpoints", "samples", "eval"):
            (self.path / sub).mkdir()
        dump_config(config, self.path / "config.yaml")
        (self.path / "env.json").write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
        self._metrics = open(self.path / "metrics.jsonl", "a", encoding="utf-8")
        self._tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.path / "tb"))

    def log(self, step: int, **values: float) -> None:
        self._metrics.write(json.dumps({"step": step, "time": time.time(), **values}) + "\n")
        self._metrics.flush()
        if self._tb is not None:
            for k, v in values.items():
                self._tb.add_scalar(k, v, step)

    def log_image(self, tag: str, image_hwc: np.ndarray, step: int) -> None:
        if self._tb is not None:
            self._tb.add_image(tag, image_hwc, step, dataformats="HWC")

    def close(self) -> None:
        self._metrics.close()
        if self._tb is not None:
            self._tb.close()
