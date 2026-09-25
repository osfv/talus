import json

import numpy as np
import pytest

from nullscape.data.build import build_dataset
from nullscape.data.dataset import TerrainDataset
from nullscape.data.splits import make_splits
from nullscape.data.storage import SPLITS, TerrainStore, dequantize, quantize
from nullscape.metrics.quality import CONDITION_KEYS

RES = 16

BUILD_CFG = {
    "name": "t_worker",
    "seed": 4242,
    "n": 24,
    "world": {"resolution": RES, "extent_m": 4096.0, "max_height_m": 1200.0, "sea_level": 0.2},
    "generator": {"supersample": 2, "blend_probability": 0.25},
    "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("datasets")
    a = build_dataset({**BUILD_CFG, "name": "tw1"}, workers=1, out_root=out, progress=False)
    b = build_dataset({**BUILD_CFG, "name": "tw2"}, workers=2, out_root=out, progress=False)
    return a, b


def test_build_independent_of_worker_count(built):
    a, b = built
    for fname in ("heights.npy", "conditions.npy", "labels.npy"):
        va = np.load(a / fname)
        vb = np.load(b / fname)
        assert va.shape == vb.shape and np.array_equal(va, vb), fname


def test_build_artifacts_and_manifest(built):
    root, _ = built
    for fname in ("heights.npy", "conditions.npy", "labels.npy", "meta.jsonl", "manifest.json"):
        assert (root / fname).exists(), fname
    for s in SPLITS:
        assert (root / f"split_{s}.npy").exists()

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for key in ("name", "n", "world", "splits", "condition_keys", "condition_stats",
                "archetypes", "generator_version", "config_sha256", "created_at"):
        assert key in manifest, key
    assert manifest["n"] == 24
    assert len(manifest["condition_keys"]) == len(CONDITION_KEYS)
    assert len(manifest["condition_stats"]["mean"]) == len(CONDITION_KEYS)
    assert len(manifest["condition_stats"]["std"]) == len(CONDITION_KEYS)

    heights = np.load(root / "heights.npy")
    conds = np.load(root / "conditions.npy")
    labels = np.load(root / "labels.npy")
    assert heights.shape == (24, RES, RES) and heights.dtype == np.uint16
    assert conds.shape == (24, len(CONDITION_KEYS)) and conds.dtype == np.float32
    assert labels.dtype == np.int64 and set(np.unique(labels)) <= set(range(6))

    records = [json.loads(line) for line in (root / "meta.jsonl").read_text().splitlines()]
    assert len(records) == 24
    for key in ("index", "archetype", "archetype_id", "params", "metrics", "traversability",
                "clipped_fraction"):
        assert key in records[0], key


def test_splits_disjoint_and_covering(built):
    root, _ = built
    parts = {s: np.load(root / f"split_{s}.npy") for s in SPLITS}
    union = np.sort(np.concatenate(list(parts.values())))
    assert np.array_equal(union, np.arange(24))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"] == {s: int(len(parts[s])) for s in SPLITS}


def test_build_refuses_overwrite(built):
    root, _ = built
    with pytest.raises(FileExistsError):
        build_dataset({**BUILD_CFG, "name": "tw1"}, workers=1, out_root=root.parent, progress=False)


def test_make_splits_deterministic_and_sized():
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = make_splits(24, fractions, seed=99)
    b = make_splits(24, fractions, seed=99)
    for s in a:
        assert np.array_equal(a[s], b[s])
    assert len(a["val"]) == len(a["test"]) == 2 and len(a["train"]) == 20
    assert np.array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(24))
    c = make_splits(24, fractions, seed=100)
    assert not np.array_equal(c["train"], a["train"])


def test_quantize_dequantize_roundtrip(built):
    root, _ = built
    h = dequantize(np.load(root / "heights.npy")[:4])
    err = np.abs(dequantize(quantize(h)) - h)
    assert float(err.max()) <= 0.5 / 65535 + 1e-7


def test_terrain_dataset_contents(built):
    root, _ = built
    store = TerrainStore.open(root)
    ds = TerrainDataset(store, split="train")
    assert len(ds) == len(store.split("train"))
    item = ds[0]
    x, cond = item["x"].numpy(), item["cond"].numpy()
    assert x.shape == (1, RES, RES)
    assert float(x.min()) >= -1.0 - 1e-6 and float(x.max()) <= 1.0 + 1e-6
    assert cond.shape == (len(CONDITION_KEYS),) and cond.dtype == np.float32
    assert int(item["label"]) == int(store.labels[int(item["index"])])


def test_dihedral_augmentation_preserves_values(built):
    root, _ = built
    store = TerrainStore.open(root)
    ds = TerrainDataset(store, split="train", augment=True)
    for i in range(8):
        item = ds[i]
        idx = int(item["index"])
        orig = dequantize(store.heights_u16[idx]) * 2.0 - 1.0
        assert np.array_equal(
            np.sort(item["x"].numpy().ravel()), np.sort(orig.ravel())
        )
