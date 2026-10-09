import json
import math
import zlib

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from nullscape.data import earth  # noqa: E402
from nullscape.world import WorldSpec  # noqa: E402


def _encode_float_predictor(rows: np.ndarray) -> bytes:
    """Reference encoder for TIFF PREDICTOR=3 (big-endian byte planes, then horizontal byte differences)."""
    tl, tw = rows.shape
    planes = rows.astype(">f4").view(np.uint8).reshape(tl, tw, 4).transpose(0, 2, 1).reshape(tl, 4 * tw)
    diff = planes.copy()
    diff[:, 1:] = (planes[:, 1:].astype(np.int16) - planes[:, :-1]).astype(np.uint8)
    return zlib.compress(diff.tobytes())


def test_float_predictor_matches_libtiff_byte_layout():
    # [1.0, 2.0] big-endian = 3F800000 40000000 -> planes 3F 40 | 80 00 | 00 00 | 00 00 -> differenced below
    encoded = zlib.compress(bytes([0x3F, 0x01, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00]))
    decoded = earth.decode_float_predictor_tile(encoded, width=2, length=1)
    assert decoded.dtype == np.float32 and decoded.tolist() == [[1.0, 2.0]]
    values = np.random.default_rng(0).normal(300, 400, (7, 5)).astype(np.float32)
    assert np.array_equal(earth.decode_float_predictor_tile(_encode_float_predictor(values), 5, 7), values)


def _write_geotiff(path, z, lat_top, lon_left, dlat, dlon):
    tags = [(33550, "d", 3, (dlon, dlat, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, lon_left, lat_top, 0.0)),
            (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 2, 2048, 0, 1, 4326))]
    tifffile.imwrite(path, z.astype(np.float32), tile=(64, 64), extratags=tags)


def test_geotiff_reader_returns_pixel_centre_georeference(tmp_path):
    z = np.arange(128 * 96, dtype=np.float32).reshape(128, 96)
    path = tmp_path / "t.tif"
    _write_geotiff(path, z, 48.0, 19.0, 1 / 3600, 1.5 / 3600)
    tile = earth.read_tile(path)
    assert np.array_equal(tile.z, z)
    assert tile.lat_top == 48.0 and tile.lon_left == 19.0
    assert tile.dlat == pytest.approx(1 / 3600) and tile.dlon == pytest.approx(1.5 / 3600)


def test_tile_names_parse_both_hemispheres():
    assert earth.parse_tile_name("Copernicus_DSM_COG_10_N47_00_E019_00_DEM") == (47, 19)
    assert earth.parse_tile_name("Copernicus_DSM_COG_10_S34_00_W071_00_DEM") == (-34, -71)
    with pytest.raises(ValueError):
        earth.parse_tile_name("not_a_tile")


def _plane_tile(lat0, slope_east, slope_north, size=600, offset=3500.0):
    """Synthetic 1-arcsec tile whose surface is an exact plane in metres."""
    dlat, dlon = 1 / 3600, 1 / 3600
    rows, cols = np.mgrid[:size, :size]
    lat = lat0 + 0.25 - rows * dlat
    y_m = (lat - lat.mean()) * earth.metres_per_degree_lat(lat0 + 0.125)
    x_m = cols * dlon * earth.metres_per_degree_lon(lat0 + 0.125)
    z = offset + slope_east * x_m + slope_north * y_m
    return earth.Tile(name="synthetic", z=z.astype(np.float32), lat_top=lat0 + 0.25, lon_left=10.0, dlat=dlat, dlon=dlon)


@pytest.mark.parametrize("lat0,east,north", [(45.0, 0.2, 0.0), (45.0, 0.0, 0.3), (62.0, 0.25, 0.1)])
def test_windows_are_square_in_metres(lat0, east, north):
    from nullscape.metrics.quality import slope_degrees

    world = WorldSpec(resolution=32, extent_m=2048.0, max_height_m=9500.0, sea_level=500.5 / 9500)
    tile = _plane_tile(lat0, east, north)
    windows = earth.extract_windows(tile, world, datum_offset_m=500.0, count=4, seed=1)
    assert len(windows) == 4
    expected = math.degrees(math.atan(math.hypot(east, north)))
    for w in windows:
        interior = slope_degrees(w.h, world)[2:-2, 2:-2]
        assert interior.mean() == pytest.approx(expected, abs=0.6)
        assert abs(w.lat - lat0) < 0.3 and w.clipped_fraction == 0.0


def test_window_selection_is_deterministic_and_skips_invalid_data():
    world = WorldSpec(resolution=32, extent_m=2048.0, max_height_m=9500.0, sea_level=500.5 / 9500)
    tile = _plane_tile(45.0, 0.1, 0.0)
    a = earth.extract_windows(tile, world, 500.0, count=3, seed=7)
    b = earth.extract_windows(tile, world, 500.0, count=3, seed=7)
    assert [(w.row, w.col) for w in a] == [(w.row, w.col) for w in b]
    tile.z[:, :] = np.nan
    assert earth.extract_windows(tile, world, 500.0, count=3, seed=7) == []


def test_geographic_split_is_stable_and_block_pure():
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = earth.split_for_block((9, 3), fractions, seed=5)
    assert a == earth.split_for_block((9, 3), fractions, seed=5)
    assert earth.block_of(47.9, 19.2, 5) == earth.block_of(45.1, 15.0, 5) == (9, 3)
    assert earth.block_of(-0.5, -0.5, 5) == (-1, -1)
    counts = {s: 0 for s in fractions}
    for i in range(-40, 40):
        for j in range(-40, 40):
            counts[earth.split_for_block((i, j), fractions, seed=5)] += 1
    total = sum(counts.values())
    assert abs(counts["train"] / total - 0.8) < 0.02 and abs(counts["test"] / total - 0.1) < 0.02


def test_label_rules_use_first_match():
    rules = [{"name": "coastal", "when": {"water_fraction": [0.05, None]}},
             {"name": "mountains", "when": {"relief_m": [600, None]}},
             {"name": "plains"}]
    assert earth.label_for({"water_fraction": 0.2, "relief_m": 900}, rules) == 0
    assert earth.label_for({"water_fraction": 0.0, "relief_m": 900}, rules) == 1
    assert earth.label_for({"water_fraction": 0.0, "relief_m": 10}, rules) == 2
    with pytest.raises(ValueError):
        earth.validate_rules([{"name": "a", "when": {"relief_m": [0, 1]}}])


def test_download_is_resumable_and_capped(tmp_path):
    calls = []

    def fake_fetch(url, target):
        calls.append(url)
        target.write_bytes(b"tiff-bytes")

    names = ["Copernicus_DSM_COG_10_N47_00_E019_00_DEM", "Copernicus_DSM_COG_10_N46_00_E010_00_DEM"]
    with pytest.raises(ValueError, match="max-gb"):
        earth.download_tiles(names, tmp_path, max_gb=0.01, fetch=fake_fetch)
    assert calls == []
    paths = earth.download_tiles(names, tmp_path, max_gb=1, fetch=fake_fetch)
    assert all(p.exists() for p in paths) and len(calls) == 2
    assert not list(tmp_path.glob("*.part"))
    earth.download_tiles(names, tmp_path, max_gb=1, fetch=fake_fetch)
    assert len(calls) == 2


def _sample_config(tmp_path, tiles):
    return {"name": "earth_test", "seed": 3, "source": {"tiles": tiles, "cache": str(tmp_path / "cache")},
            "world": {"resolution": 32, "extent_m": 2048.0, "max_height_m": 9500.0},
            "datum_offset_m": 500.0, "sea_threshold_m": 0.5, "windows_per_tile": 6,
            "filters": {"max_water_fraction": 0.95, "min_relief_m": 0.5},
            "splits": {"fractions": {"train": 0.5, "val": 0.25, "test": 0.25}, "block_deg": 1},
            "labels": [{"name": "coastal", "when": {"water_fraction": [0.05, None]}},
                       {"name": "hills", "when": {"relief_m": [50, None]}}, {"name": "plains"}]}


def test_end_to_end_build_writes_a_terrain_store(tmp_path):
    from nullscape.data.storage import TerrainStore
    from nullscape.metrics.quality import compute_metrics, condition_vector

    tiles = []
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    for k, (lat, lon) in enumerate([(45, 7), (46, 8), (47, 9), (48, 10), (44, 11), (43, 12)]):
        name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
        yy, xx = np.mgrid[:720, :720] / 720.0
        z = 600 + 400 * np.sin(6 * xx + k) * np.cos(5 * yy) + rng.normal(0, 2, (720, 720))
        if k == 0:
            z[:, :360] = 0.0  # sea along the western half
        _write_geotiff(cache / f"{name}.tif", z, lat + 0.2, lon, 1 / 3600, 1 / 3600)
        tiles.append(name)
    cfg = _sample_config(tmp_path, tiles)
    root = earth.build_earth_dataset(cfg, workers=1, out_root=tmp_path / "data", download=False)
    store = TerrainStore.open(root)
    manifest = store.manifest
    assert "Copernicus WorldDEM-30" in manifest["source"]["attribution"]
    assert manifest["world"]["sea_level"] == pytest.approx(500.5 / 9500)
    assert store.archetypes == ["coastal", "hills", "plains"]
    n = len(store)
    assert n > 0 and store.heights_u16.shape == (n, 32, 32)
    splits = {s: set(store.split(s).tolist()) for s in ("train", "val", "test")}
    assert set().union(*splits.values()) == set(range(n)) and sum(map(len, splits.values())) == n
    meta = list(store.iter_meta())
    by_block = {}
    for rec in meta:
        by_block.setdefault(tuple(rec["block"]), set()).add(rec["split"])
        assert rec["index"] in splits[rec["split"]]
    assert all(len(s) == 1 for s in by_block.values())
    h = store.heights(np.array([0]))[0]
    assert np.allclose(store.conditions[0], condition_vector(compute_metrics(h, store.world)), atol=1e-3)
    assert any(rec["metrics"]["water_fraction"] > 0 for rec in meta)
    with pytest.raises(FileExistsError):
        earth.build_earth_dataset(cfg, workers=1, out_root=tmp_path / "data", download=False)


def test_plan_samples_tiles_reproducibly_within_latitude_limits():
    names = [f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_E{lon:03d}_00_DEM"
             for lat in range(-80, 80, 5) for lon in range(0, 50, 7)]
    source = {"sample_tiles": 20, "lat_range": [-60, 70]}
    a, b = earth.select_tiles(source, names, seed=1), earth.select_tiles(source, names, seed=1)
    assert a == b and len(a) == 20
    assert all(-60 <= earth.parse_tile_name(t)[0] < 70 for t in a)
    explicit = earth.select_tiles({"tiles": names[:3]}, names, seed=1)
    assert explicit == names[:3]
    with pytest.raises(ValueError, match="not in"):
        earth.select_tiles({"tiles": ["Copernicus_DSM_COG_10_N89_00_E000_00_DEM"]}, names, seed=1)


def test_config_validation_rejects_unknown_keys(tmp_path):
    cfg = _sample_config(tmp_path, ["Copernicus_DSM_COG_10_N47_00_E019_00_DEM"])
    cfg["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        earth.validate_config(cfg)
    normalized = earth.validate_config(_sample_config(tmp_path, ["x"]))
    json.dumps(normalized["world"])
    assert earth.validate_config(normalized) == normalized
    bad = {**normalized, "world": {**normalized["world"], "sea_level": 0.3}}
    with pytest.raises(ValueError, match="derived"):
        earth.validate_config(bad)
