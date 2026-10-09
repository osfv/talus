from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ALIASES = {
    "talus-1": "runs/20260925-164759_diffusion64/checkpoints/step_0030000.pt",
    "talus-1.1": "runs/20260926-181917_v2_exp1_nominsnr_cosine/checkpoints/step_0008000.pt",
    "talus-2": "runs/20260926-192753_talus2_relative/checkpoints/step_0010000.pt",
    "talus-3": "checkpoints/talus-3.pt",
    "talus-3.1": "checkpoints/talus-3.1.pt",
}
RELEASES_URL = "https://github.com/osfv/talus/releases"


def data_root() -> Path:
    return Path(os.environ.get("NULLSCAPE_DATA_ROOT", REPO_ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("NULLSCAPE_RUNS_ROOT", REPO_ROOT / "runs"))


def dataset_dir(name_or_path: str | Path) -> Path:
    """Resolve a dataset name (under data_root) or an explicit directory path."""
    p = Path(name_or_path)
    return p if p.is_dir() or p.is_absolute() or len(p.parts) > 1 else data_root() / p


def resolve_checkpoint(source: str | Path, *, prefer: str = "best") -> Path:
    if prefer not in ("best", "last"):
        raise ValueError("checkpoint preference must be best or last")
    value = os.path.expandvars(str(source))
    alias = CHECKPOINT_ALIASES.get(value.casefold())
    path = REPO_ROOT / alias if alias else Path(value).expanduser()
    if not path.exists() and not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        folder = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        for name in (prefer, "last" if prefer == "best" else "best"):
            candidate = folder / f"{name}.pt"
            if candidate.is_file():
                return candidate.resolve()
        snapshots = [p for p in folder.glob("step_*.pt") if p.is_file() and p.stem[5:].isdecimal()]
        if snapshots:
            return max(snapshots, key=lambda p: int(p.stem[5:])).resolve()
    if alias and alias.startswith("checkpoints/"):
        raise FileNotFoundError(f"{source} is not downloaded yet: save {Path(alias).name} from {RELEASES_URL} "
                                f"as {REPO_ROOT / alias}")
    raise FileNotFoundError(f"checkpoint {source!s} not found. Use a .pt file, run directory, or "
                            f"{', '.join(CHECKPOINT_ALIASES)}; run 'nullscape checkpoints' to list them")


def unique_directory(root: str | Path, stem: str) -> Path:
    if not stem or any(c in stem for c in '/\\<>:"|?*') or stem in (".", ".."):
        raise ValueError("directory name contains invalid filename characters")
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for i in itertools.count():
        candidate = root / (stem if i == 0 else f"{stem}-{i:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue


def prepare_output_dir(path: str | Path | None, *, default_root: str | Path = "outputs/samples") -> Path:
    if path is None:
        out = unique_directory(default_root, time.strftime("%Y%m%d-%H%M%S"))
    else:
        out = Path(path).expanduser()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError(f"output {out} is a file or non-empty directory; choose a new --out folder")
        out.mkdir(parents=True, exist_ok=True)
    try:
        with (out / ".nullscape-output").open("x", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    except FileExistsError:
        raise FileExistsError(f"output {out} is already reserved; choose a new --out folder") from None
    return out.resolve()
