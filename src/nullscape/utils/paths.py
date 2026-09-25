from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return Path(os.environ.get("NULLSCAPE_DATA_ROOT", REPO_ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("NULLSCAPE_RUNS_ROOT", REPO_ROOT / "runs"))


def dataset_dir(name_or_path: str | Path) -> Path:
    """Resolve a dataset name (under data_root) or an explicit directory path."""
    p = Path(name_or_path)
    return p if p.is_dir() or p.is_absolute() or len(p.parts) > 1 else data_root() / p
