"""Real-terrain datasets from the Copernicus GLO-30 Public DEM, in the same on-disk format as procedural
datasets (see storage.py), so training and evaluation code can read them unchanged.

Source: 1x1 degree Cloud-Optimized GeoTIFFs on AWS Open Data (no account needed), float32 metres above
the EGM2008 geoid, 1 arc-second latitude spacing, DEFLATE + floating-point predictor. The data is a surface
model (buildings and vegetation included). Ocean has no tiles and is 0 m inside coastal tiles.

Pipeline, all deterministic from the config:
  select tiles -> download (resumable, size-capped) -> decode -> cut square windows in metres at the target
  cell size (anti-aliased resampling) -> filter (invalid data, clipping, open water, flat) -> measure with the
  standard metrics -> label with ordered rules -> split by geographic block -> write the store.

Heights: h = (z + datum_offset_m) / max_height_m, clipped to [0, 1]. Water is z < sea_threshold_m, encoded as
world.sea_level = (datum_offset_m + sea_threshold_m) / max_height_m. With the defaults (offset 500 m, range
9500 m) the uint16 step is 0.145 m.

Splits: every 1-degree tile belongs to a block of block_deg x block_deg degrees; a block's split is a pure
function of (split seed, block), so neighbouring (strongly correlated) terrain never straddles train and test,
and growing the dataset later never moves a region between splits.

Licence: Copernicus WorldDEM-30 free licence (reproduction, distribution, modification allowed). Anything
built from it must carry ATTRIBUTION, which is stored in every manifest.

usage:
  python -m nullscape.data.earth tiles --config configs/dataset/earth64_sample.yaml
  python -m nullscape.data.earth download --config ... [--max-gb 5]
  python -m nullscape.data.earth build --config ... [--workers 2] [--max-gb 5] [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from nullscape.data.storage import SPLITS, dequantize, quantize
from nullscape.metrics.quality import CONDITION_KEYS, compute_metrics, condition_vector
from nullscape.metrics.traversability import AgentSpec, analyze
from nullscape.utils.config import config_hash, load_config
from nullscape.utils.paths import data_root
from nullscape.utils.tracking import git_info
from nullscape.world import WorldSpec

EARTH_VERSION = "earth-1.0.0"
BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
ATTRIBUTION = ("produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010-2014 and \u00a9 Airbus Defence and Space "
               "GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved")
LICENCE_URL = ("https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/"
               "collections-description/COP-DEM")
AVG_TILE_MB = 40.0
_TILE = re.compile(r"_(N|S)(\d{2})_00_(E|W)(\d{3})_00_")
PHYSICAL_KEYS = ("relief_m", "mean_elevation_m", "water_fraction", "mean_slope_deg", "p90_slope_deg")
DEFAULT_LABELS = [{"name": "coastal", "when": {"water_fraction": [0.05, None]}},
                  {"name": "mountains", "when": {"relief_m": [600, None]}},
                  {"name": "hills", "when": {"relief_m": [120, None]}},
                  {"name": "plateau", "when": {"mean_elevation_m": [1000, None]}},
                  {"name": "plains"}]


@dataclass
class Tile:
    name: str
    z: np.ndarray
    lat_top: float  # latitude of the centre of row 0
    lon_left: float  # longitude of the centre of column 0
    dlat: float
    dlon: float


@dataclass
class Window:
    h: np.ndarray
    row: int
    col: int
    lat: float
    lon: float
    clipped_fraction: float


def parse_tile_name(name: str) -> tuple[int, int]:
    """(latitude, longitude) of a tile's south-west corner."""
    m = _TILE.search(name)
    if not m:
        raise ValueError(f"not a Copernicus tile name: {name!r}")
    lat, lon = int(m.group(2)), int(m.group(4))
    return (-lat if m.group(1) == "S" else lat), (-lon if m.group(3) == "W" else lon)


def metres_per_degree_lat(lat: float) -> float:
    p = math.radians(lat)
    return 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p) - 0.0023 * math.cos(6 * p)


def metres_per_degree_lon(lat: float) -> float:
    p = math.radians(lat)
    return 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p) + 0.118 * math.cos(5 * p)


def decode_float_predictor_tile(data: bytes, width: int, length: int) -> np.ndarray:
    """Inflate one DEFLATE segment and undo TIFF PREDICTOR=3 (libtiff fpAcc) for 32-bit floats."""
    raw = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
    if raw.size != 4 * width * length:
        raise ValueError(f"segment holds {raw.size} bytes, expected {4 * width * length}")
    acc = np.cumsum(raw.reshape(length, 4 * width), axis=1, dtype=np.uint8)
    planes = acc.reshape(length, 4, width).transpose(0, 2, 1).copy()
    return planes.view(">f4").reshape(length, width).astype(np.float32)


def read_tile(path: str | Path, name: str | None = None) -> Tile:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if page.samplesperpixel != 1 or page.bitspersample != 32 or page.sampleformat != 3:
            raise ValueError(f"{path}: expected single-band float32 elevations")
        height, width = page.imagelength, page.imagewidth
        if page.predictor == 3 and int(page.compression) in (8, 32946):
            tw, tl = (page.tilewidth, page.tilelength) if page.is_tiled else (width, page.rowsperstrip)
            across = -(-width // tw)
            z = np.empty((height, width), np.float32)
            for i, (offset, count) in enumerate(zip(page.dataoffsets, page.databytecounts)):
                r0, c0 = (i // across) * tl, (i % across) * tw
                rows = tl if page.is_tiled else min(tl, height - r0)
                tif.filehandle.seek(offset)
                block = decode_float_predictor_tile(tif.filehandle.read(count), tw, rows)
                z[r0:r0 + tl, c0:c0 + tw] = block[: height - r0, : width - c0]
        else:
            z = page.asarray().astype(np.float32)
        nodata = page.tags.get("GDAL_NODATA")
        if nodata is not None:
            z[z == float(str(nodata.value).strip("\x00 "))] = np.nan
        dlon, dlat = page.tags["ModelPixelScaleTag"].value[:2]
        i, j, _, lon0, lat0, _ = page.tags["ModelTiepointTag"].value[:6]
        raster = (tif.geotiff_metadata or {}).get("GTRasterTypeGeoKey", 2)
        area = int(raster) == 1
        lon0, lat0 = lon0 - i * dlon + (dlon / 2 if area else 0), lat0 + j * dlat - (dlat / 2 if area else 0)
    return Tile(name or Path(path).stem, z, float(lat0), float(lon0), float(dlat), float(dlon))


def iter_windows(tile: Tile, world: WorldSpec, datum_offset_m: float, seed: int,
                 stats: Counter | None = None) -> Iterator[Window]:
    """Square windows of world.extent_m metres, in a reproducible random order, resampled to world.resolution."""
    import torch
    import torch.nn.functional as F

    height, width = tile.z.shape
    ext, res = world.extent_m, world.resolution
    h_px = ext / (metres_per_degree_lat(tile.lat_top - height / 2 * tile.dlat) * tile.dlat)
    candidates = []
    top = 0.0
    while top + h_px <= height:
        lat = tile.lat_top - (top + h_px / 2 - 0.5) * tile.dlat
        w_px = ext / (metres_per_degree_lon(lat) * tile.dlon)
        left = 0.0
        while left + w_px <= width:
            candidates.append((top, left, h_px, w_px))
            left += w_px
        top += h_px
    rng = np.random.default_rng([int(seed), zlib.crc32(tile.name.encode())])
    for k in rng.permutation(len(candidates)):
        top, left, hp, wp = candidates[k]
        r0, c0 = int(round(top)), int(round(left))
        r1, c1 = min(height, r0 + max(2, int(round(hp)))), min(width, c0 + max(2, int(round(wp))))
        crop = tile.z[r0:r1, c0:c1]
        if not np.isfinite(crop).all():
            if stats is not None:
                stats["invalid"] += 1
            continue
        t = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))[None, None]
        z = F.interpolate(t, size=(res, res), mode="bilinear", antialias=True, align_corners=False)[0, 0].numpy()
        h = (z.astype(np.float64) + datum_offset_m) / world.max_height_m
        clipped = float(((h < 0) | (h > 1)).mean())
        yield Window(np.clip(h, 0, 1).astype(np.float32), r0, c0,
                     tile.lat_top - ((r0 + r1) / 2 - 0.5) * tile.dlat,
                     tile.lon_left + ((c0 + c1) / 2 - 0.5) * tile.dlon, clipped)


def extract_windows(tile: Tile, world: WorldSpec, datum_offset_m: float, count: int, seed: int,
                    accept: Callable[[Window], bool] | None = None) -> list[Window]:
    out = []
    for window in iter_windows(tile, world, datum_offset_m, seed):
        if len(out) == count:
            break
        if accept is None or accept(window):
            out.append(window)
    return out


def block_of(lat: float, lon: float, block_deg: int) -> tuple[int, int]:
    return math.floor(lat / block_deg), math.floor(lon / block_deg)


def split_for_block(block: tuple[int, int], fractions: dict[str, float], seed: int) -> str:
    digest = hashlib.sha256(f"nullscape-earth-split:{seed}:{block[0]}:{block[1]}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    total = sum(fractions.values())
    edge = 0.0
    for name in sorted(k for k in fractions if k != "train"):
        edge += fractions[name] / total
        if u < edge:
            return name
    return "train"


def validate_rules(rules: list[dict]) -> list[dict]:
    if not rules or not isinstance(rules, list):
        raise ValueError("labels must be a non-empty list of rules")
    names = [r.get("name") for r in rules]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("label rules need unique names")
    for rule in rules:
        if set(rule) - {"name", "when"}:
            raise ValueError(f"label rule keys must be name and when: {rule}")
        for key, bounds in (rule.get("when") or {}).items():
            if key not in PHYSICAL_KEYS or len(bounds) != 2:
                raise ValueError(f"label condition {key!r} must be one of {PHYSICAL_KEYS} with [min, max]")
    if rules[-1].get("when"):
        raise ValueError("the last label rule must have no conditions (it catches everything else)")
    return rules


def label_for(values: dict[str, float], rules: list[dict]) -> int:
    for index, rule in enumerate(rules):
        if all((lo is None or values[key] >= lo) and (hi is None or values[key] < hi)
               for key, (lo, hi) in (rule.get("when") or {}).items()):
            return index
    raise ValueError("no label rule matched")


def default_cache() -> Path:
    if os.environ.get("NULLSCAPE_DEM_CACHE"):
        return Path(os.environ["NULLSCAPE_DEM_CACHE"])
    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".cache"
    return Path(base) / "nullscape" / "dem_cache" / "glo30"


def validate_config(cfg: dict) -> dict:
    allowed = {"name", "seed", "source", "world", "datum_offset_m", "sea_threshold_m", "windows_per_tile", "filters",
               "splits", "labels", "agent"}
    if set(cfg) - allowed:
        raise ValueError(f"unknown earth dataset keys: {', '.join(sorted(set(cfg) - allowed))}")
    source = dict(cfg.get("source") or {})
    if set(source) - {"tiles", "sample_tiles", "lat_range", "lon_range", "cache"} or \
            ("tiles" in source) == ("sample_tiles" in source):
        raise ValueError("source needs exactly one of tiles or sample_tiles (optional lat_range, lon_range, cache)")
    source["cache"] = str(Path(source.get("cache") or default_cache()).expanduser())
    world = dict(cfg.get("world") or {})
    given_sea = world.pop("sea_level", None)
    if set(world) != {"resolution", "extent_m", "max_height_m"}:
        raise ValueError("world needs resolution, extent_m and max_height_m (sea level is derived)")
    offset, threshold = float(cfg.get("datum_offset_m", 500.0)), float(cfg.get("sea_threshold_m", 0.5))
    world["sea_level"] = (offset + threshold) / float(world["max_height_m"])
    if given_sea is not None and not math.isclose(float(given_sea), world["sea_level"], rel_tol=1e-9):
        raise ValueError("world.sea_level is derived from datum_offset_m and sea_threshold_m; remove it")
    WorldSpec.from_dict(world)
    filters = {"max_water_fraction": 0.9, "min_relief_m": 1.0, "max_clipped_fraction": 0.001, **(cfg.get("filters") or {})}
    if set(filters) != {"max_water_fraction", "min_relief_m", "max_clipped_fraction"}:
        raise ValueError("filters are max_water_fraction, min_relief_m and max_clipped_fraction")
    splits = {"fractions": {"train": 0.9, "val": 0.05, "test": 0.05}, "block_deg": 5, **(cfg.get("splits") or {})}
    if set(splits) != {"fractions", "block_deg"} or set(splits["fractions"]) != set(SPLITS) or \
            not isinstance(splits["block_deg"], int) or splits["block_deg"] < 1:
        raise ValueError("splits needs fractions for train/val/test and an integer block_deg >= 1")
    windows = cfg.get("windows_per_tile", 32)
    if not isinstance(windows, int) or windows < 1:
        raise ValueError("windows_per_tile must be a positive integer")
    return {"name": str(cfg["name"]), "seed": int(cfg.get("seed", 0)), "source": source, "world": world,
            "datum_offset_m": offset, "sea_threshold_m": threshold, "windows_per_tile": windows,
            "filters": {k: float(v) for k, v in filters.items()}, "splits": splits,
            "labels": validate_rules(cfg.get("labels") or DEFAULT_LABELS),
            "agent": AgentSpec(**(cfg.get("agent") or {})).__dict__}


def load_tile_list(cache: Path, refresh: bool = False) -> list[str]:
    path = cache / "tileList.txt"
    if refresh or not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        _fetch(f"{BUCKET}/tileList.txt", path.with_suffix(".part"))
        path.with_suffix(".part").replace(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_tiles(source: dict, available: list[str] | None, seed: int) -> list[str]:
    if "tiles" in source:
        names = [str(t) for t in source["tiles"]]
        missing = [t for t in names if available is not None and t not in set(available)]
        if missing:
            raise ValueError(f"tiles not in the GLO-30 public tile list: {missing[:5]}")
        return names
    lo, hi = source.get("lat_range", [-60, 72])
    lon_lo, lon_hi = source.get("lon_range", [-180, 180])
    pool = sorted(t for t in available if lo <= parse_tile_name(t)[0] < hi and lon_lo <= parse_tile_name(t)[1] < lon_hi)
    count = int(source["sample_tiles"])
    if count > len(pool):
        raise ValueError(f"requested {count} tiles but only {len(pool)} match the latitude/longitude limits")
    rng = np.random.default_rng([int(seed), 0xEA27])
    return sorted(pool[i] for i in rng.choice(len(pool), size=count, replace=False))


def _fetch(url: str, target: Path, attempts: int = 4) -> None:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nullscape-earth/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as file:
                shutil.copyfileobj(response, file, length=1 << 20)
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                raise FileNotFoundError(f"{url} is not available (HTTP {exc.code})") from None
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def download_tiles(names: list[str], cache: str | Path, *, max_gb: float,
                   fetch: Callable[[str, Path], None] = _fetch, workers: int = 4) -> list[Path]:
    cache = Path(cache)
    paths = [cache / f"{name}.tif" for name in names]
    missing = [(n, p) for n, p in zip(names, paths) if not p.is_file()]
    estimate = len(missing) * AVG_TILE_MB / 1024
    if estimate > max_gb:
        raise ValueError(f"about {estimate:.1f} GB to download for {len(missing)} tiles exceeds --max-gb {max_gb}; "
                         "raise the limit deliberately")
    if missing:
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(missing)} tiles (about {estimate:.1f} GB) to {cache}", flush=True)

        def one(item):
            name, path = item
            part = path.with_name(path.name + ".part")
            fetch(f"{BUCKET}/{name}/{name}.tif", part)
            part.replace(path)

        with ThreadPoolExecutor(max(1, workers)) as pool:
            for i, _ in enumerate(pool.map(one, missing), 1):
                if i % 10 == 0 or i == len(missing):
                    print(f"  {i}/{len(missing)} tiles", flush=True)
    return paths


def _physical(metrics: dict, world: WorldSpec, offset: float) -> dict[str, float]:
    return {"relief_m": metrics["relief"] * world.max_height_m,
            "mean_elevation_m": metrics["mean_elevation"] * world.max_height_m - offset,
            "water_fraction": metrics["water_fraction"], "mean_slope_deg": metrics["mean_slope_deg"],
            "p90_slope_deg": metrics["p90_slope_deg"]}


def _process_tile(job: tuple[str, str, dict]):
    import torch

    torch.set_num_threads(1)
    path, name, cfg = job
    world, agent, filters = WorldSpec.from_dict(cfg["world"]), AgentSpec(**cfg["agent"]), cfg["filters"]
    tile = read_tile(path, name)
    stats: Counter = Counter()
    samples = []
    for window in iter_windows(tile, world, cfg["datum_offset_m"], cfg["seed"], stats):
        if len(samples) == cfg["windows_per_tile"]:
            break
        if window.clipped_fraction > filters["max_clipped_fraction"]:
            stats["clipped"] += 1
            continue
        q = quantize(window.h)
        h = dequantize(q)
        metrics = compute_metrics(h, world)
        phys = _physical(metrics, world, cfg["datum_offset_m"])
        if phys["water_fraction"] > filters["max_water_fraction"]:
            stats["water"] += 1
            continue
        if phys["relief_m"] < filters["min_relief_m"]:
            stats["flat"] += 1
            continue
        label = label_for(phys, cfg["labels"])
        rec = {"tile": name, "lat": round(window.lat, 6), "lon": round(window.lon, 6), "row": window.row,
               "col": window.col, "archetype": cfg["labels"][label]["name"], "archetype_id": label,
               "physical": phys, "clipped_fraction": window.clipped_fraction, "metrics": metrics,
               "traversability": analyze(h, world, agent).to_dict()}
        samples.append((q, condition_vector(metrics), label, rec))
    stats["accepted"] = len(samples)
    return name, samples, dict(stats), float(np.nanmin(tile.z)), float(np.nanmax(tile.z))


def build_earth_dataset(cfg: dict, workers: int = 2, out_root: str | Path | None = None, overwrite: bool = False,
                        download: bool = True, max_gb: float = 5.0, progress: bool = True) -> Path:
    cfg = validate_config(cfg)
    cache = Path(cfg["source"]["cache"])
    explicit = "tiles" in cfg["source"]
    available = None if explicit and not download else load_tile_list(cache)
    names = select_tiles(cfg["source"], available, cfg["seed"])
    paths = download_tiles(names, cache, max_gb=max_gb) if download else [cache / f"{n}.tif" for n in names]
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} tiles are not in the cache, e.g. {absent[0]}; run download first")
    root = Path(out_root or data_root()) / cfg["name"]
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite to rebuild")
        shutil.rmtree(root)
    tmp = root.with_name(root.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    world = WorldSpec.from_dict(cfg["world"])
    fractions, block_deg = cfg["splits"]["fractions"], cfg["splits"]["block_deg"]
    conds, labels, splits, records = [], [], [], []
    totals: Counter = Counter()
    elevation = [math.inf, -math.inf]
    jobs = [(str(p), n, cfg) for p, n in zip(paths, names)]
    t0 = time.time()
    raw = tmp / "heights.u16.part"
    with open(raw, "wb") as heights_file, open(tmp / "meta.jsonl", "w", encoding="utf-8") as meta_file:
        if workers <= 1:
            results = map(_process_tile, jobs)
            pool = None
        else:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            pool = get_context("spawn").Pool(workers)
            results = pool.imap(_process_tile, jobs)
        try:
            for done, (name, samples, stats, zmin, zmax) in enumerate(results, 1):
                lat, lon = parse_tile_name(name)
                block = block_of(lat, lon, block_deg)
                split = split_for_block(block, fractions, cfg["seed"])
                totals.update(stats)
                elevation = [min(elevation[0], zmin), max(elevation[1], zmax)]
                for q, cond, label, rec in samples:
                    index = len(labels)
                    heights_file.write(np.ascontiguousarray(q, dtype="<u2").tobytes())
                    conds.append(cond)
                    labels.append(label)
                    splits.append(split)
                    meta_file.write(json.dumps({"index": index, **rec, "block": list(block), "split": split}) + "\n")
                if progress:
                    print(f"  [{done}/{len(jobs)}] {name}: {stats.get('accepted', 0)} maps, split {split}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    n = len(labels)
    if n == 0:
        raise ValueError("no windows passed the filters")
    r = world.resolution
    heights = np.lib.format.open_memmap(tmp / "heights.npy", mode="w+", dtype=np.uint16, shape=(n, r, r))
    with open(raw, "rb") as file:
        for start in range(0, n, 1024):
            count = min(1024, n - start)
            heights[start:start + count] = np.fromfile(file, dtype="<u2", count=count * r * r).reshape(count, r, r)
    heights.flush()
    del heights
    raw.unlink()
    conds_arr, labels_arr, split_arr = np.stack(conds).astype(np.float32), np.asarray(labels, np.int64), np.asarray(splits)
    for s in SPLITS:
        idx = np.flatnonzero(split_arr == s).astype(np.int64)
        if not len(idx):
            raise ValueError(f"split {s} received no maps; add tiles from more regions or use a smaller block_deg")
        np.save(tmp / f"split_{s}.npy", idx)
    np.save(tmp / "conditions.npy", conds_arr)
    np.save(tmp / "labels.npy", labels_arr)
    train_c = conds_arr[split_arr == "train"]
    names_out = [rule["name"] for rule in cfg["labels"]]
    counts = Counter(labels_arr.tolist())
    passed = np.array([json.loads(line)["traversability"]["passed"]
                       for line in (tmp / "meta.jsonl").read_text(encoding="utf-8").splitlines()])
    blocks: dict[str, set] = {s: set() for s in SPLITS}
    for rec_block, s in zip((block_of(*parse_tile_name(nm), block_deg) for nm in names),
                            (split_for_block(block_of(*parse_tile_name(nm), block_deg), fractions, cfg["seed"]) for nm in names)):
        blocks[s].add(rec_block)
    manifest = {
        "name": cfg["name"], "format_version": 1, "generator_version": EARTH_VERSION, "config": cfg,
        "config_sha256": config_hash(cfg), "world": world.to_dict(), "agent": cfg["agent"], "n": n,
        "splits": {s: int((split_arr == s).sum()) for s in SPLITS},
        "condition_keys": list(CONDITION_KEYS),
        "condition_stats": {"mean": train_c.mean(0).tolist(), "std": (train_c.std(0) + 1e-8).tolist()},
        "archetypes": names_out,
        "archetype_counts": {names_out[k]: int(v) for k, v in sorted(counts.items())},
        "source": {"dataset": "Copernicus DEM GLO-30 Public (2021 release) via AWS Open Data", "bucket": BUCKET,
                   "attribution": ATTRIBUTION, "licence": LICENCE_URL, "tiles": names,
                   "tile_bytes": {nm: p.stat().st_size for nm, p in zip(names, paths)},
                   "accessed": time.strftime("%Y-%m-%d"), "surface": "DSM (includes buildings and vegetation)",
                   "vertical_datum": "EGM2008 geoid, metres"},
        "height_encoding": {"formula": "h = (z_m + datum_offset_m) / max_height_m, clipped to [0, 1]",
                            "datum_offset_m": cfg["datum_offset_m"], "sea_threshold_m": cfg["sea_threshold_m"],
                            "uint16_step_m": world.max_height_m / 65535},
        "split_method": {"type": "geographic blocks", "block_deg": block_deg, "seed": cfg["seed"],
                         "blocks": {s: sorted(map(list, b)) for s, b in blocks.items()}},
        "summary": {"windows": dict(totals), "source_elevation_range_m": elevation,
                    "traversability_pass_rate": float(passed.mean()),
                    "traversability_pass_rate_by_archetype": {names_out[k]: float(passed[labels_arr == k].mean())
                                                              for k in sorted(counts)}},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generation_seconds": time.time() - t0,
        "workers": workers, "git": git_info(), "python": platform.python_version(), "numpy": np.__version__,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.rename(root)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m nullscape.data.earth", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tiles", "download", "build"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--max-gb", type=float, default=5.0, help="refuse downloads estimated above this size")
        if name == "build":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--overwrite", action="store_true")
            p.add_argument("--no-download", action="store_true", help="use only tiles already in the cache")
            p.add_argument("--out-root", default=None)
    args = parser.parse_args(argv)
    cfg = validate_config(load_config(args.config, args.set))
    cache = Path(cfg["source"]["cache"])
    if args.command == "build":
        root = build_earth_dataset(cfg, workers=args.workers, out_root=args.out_root, overwrite=args.overwrite,
                                   download=not args.no_download, max_gb=args.max_gb)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps({k: manifest[k] for k in ("name", "n", "splits", "archetype_counts")}, indent=2))
        print(f"dataset written to {root}\nAttribution (required when sharing anything built from it; exact text "
              f"in manifest.json): {ATTRIBUTION.replace(chr(0xA9), '(c)')}")
        return
    names = select_tiles(cfg["source"], load_tile_list(cache), cfg["seed"])
    missing = [n for n in names if not (cache / f"{n}.tif").is_file()]
    if args.command == "tiles":
        print(f"{len(names)} tiles selected, {len(missing)} not cached "
              f"(about {len(missing) * AVG_TILE_MB / 1024:.1f} GB to download) -> {cache}")
        for name in names[:20]:
            print(f"  {name}{'' if name in missing else '  (cached)'}")
        if len(names) > 20:
            print(f"  ... {len(names) - 20} more")
        return
    download_tiles(names, cache, max_gb=args.max_gb)
    print(f"{len(names)} tiles in {cache}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"error: {exc}") from None
