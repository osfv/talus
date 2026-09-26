import json

import numpy as np
import pytest
from PIL import Image

from nullscape.export.engine import (
    export_heightmap,
    read_png16,
    read_r16,
    resample,
    to_uint16,
    unity_size,
    write_obj,
    write_png16,
    write_r16,
)
from nullscape.world import WorldSpec

W = WorldSpec(resolution=32, extent_m=4096.0, max_height_m=1200.0, sea_level=0.2)


@pytest.fixture(scope="module")
def h() -> np.ndarray:
    return np.random.default_rng(5).random((32, 32), dtype=np.float32)


def test_png16_roundtrip(h, tmp_path):
    p = tmp_path / "a.png"
    write_png16(h, p)
    back = np.asarray(Image.open(p))
    assert back.shape == h.shape
    assert np.array_equal(back.astype(np.int64), to_uint16(h).astype(np.int64))
    assert np.abs(read_png16(p) - h).max() <= 1.0 / 65535


def test_r16_roundtrip(h, tmp_path):
    p = tmp_path / "a.r16"
    write_r16(h, p)
    back = np.fromfile(p, dtype="<u2")
    assert np.array_equal(back, to_uint16(h).ravel())
    assert np.abs(read_r16(p, 32) - h).max() <= 1.0 / 65535


@pytest.mark.parametrize("res,size", [(16, 33), (33, 33), (64, 65), (65, 65), (128, 129), (129, 129),
                                      (257, 257), (513, 513), (1025, 1025)])
def test_unity_size(res, size):
    # valid Unity sizes (2^n + 1) are kept; others round up to the next valid size (bug B1 in v1)
    assert unity_size(res) == size


def test_resample_preserves_corners(h):
    out = resample(h, 65)
    assert out.shape == (65, 65)
    for ij in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert float(out[ij]) == pytest.approx(float(h[ij]), abs=1e-5)


def test_obj_counts(h, tmp_path):
    p = tmp_path / "a.obj"
    write_obj(h, W, p)
    lines = p.read_text(encoding="utf-8").splitlines()
    n = h.shape[0]
    assert sum(1 for l in lines if l.startswith("v ")) == n * n
    assert sum(1 for l in lines if l.startswith("vn ")) == n * n
    assert sum(1 for l in lines if l.startswith("f ")) == 2 * (n - 1) ** 2


def test_export_heightmap_writes_sidecar(h, tmp_path):
    written = export_heightmap(h, W, tmp_path, "stem", formats=("png16", "r16", "npy"),
                               metadata={"dataset": "t", "index": 0})
    for fmt in ("png16", "r16", "npy", "json"):
        assert written[fmt].exists()
    sidecar = json.loads(written["json"].read_text(encoding="utf-8"))
    assert sidecar["cell_size_m"] == pytest.approx(W.extent_m / 32)
    assert sidecar["footprint_m"] == pytest.approx(W.extent_m / 32 * 31)  # first to last sample (cell centers)
    assert sidecar["max_height_m"] == W.max_height_m
    assert sidecar["sea_level"] == W.sea_level
    npy = np.load(written["npy"])
    assert np.array_equal(npy, np.clip(h.astype(np.float32), 0.0, 1.0))


def test_resampled_sidecar_keeps_footprint(h, tmp_path):
    # bug B4 in v1: after corner-aligned resampling the sidecar claimed the full extent
    written = export_heightmap(h, W, tmp_path, "u", formats=("r16",), size=unity_size(32))
    sidecar = json.loads(written["json"].read_text(encoding="utf-8"))
    assert sidecar["resolution"] == 33
    assert sidecar["footprint_m"] == pytest.approx(W.extent_m / 32 * 31)
    assert sidecar["cell_size_m"] == pytest.approx(W.extent_m / 32 * 31 / 32)


def test_git_info_without_commits(tmp_path):
    # bug B2 in v1: an empty repository recorded the literal "HEAD" as the commit
    import subprocess

    from nullscape.utils.tracking import git_info

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert git_info(tmp_path)["commit"] is None
