# Real-terrain data for Talus-4

`nullscape.data.earth` turns the Copernicus GLO-30 elevation model into datasets in the same format as the
procedural ones (`docs/DATASET.md`). The trainer, sampler, evaluation and comparison tools read them
unchanged: a CPU training preflight on `earth64_sample` runs without code changes.

## Source and licence

- **Copernicus DEM GLO-30 Public**, 2021 release, from AWS Open Data (`s3://copernicus-dem-30m`, no account).
  26,450 one-degree tiles, float32 metres above the EGM2008 geoid, 1 arc-second latitude spacing.
  Longitude spacing widens above 50 degrees latitude; the pipeline handles this.
- It is a **surface** model: buildings and trees are included. Ocean has no tiles and is 0 m in coastal tiles.
- **Licence:** Copernicus WorldDEM-30 free licence. It allows reproduction, distribution, modification and
  combination with other data. Anything built from it, including datasets and model releases, must carry:

  > produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018
  > provided under COPERNICUS by the European Union and ESA; all rights reserved

  Every manifest stores this text. Check the licence yourself before a commercial release; this note is not
  legal advice.

## Pipeline

```powershell
$py = "$env:USERPROFILE\.venvs\nullscape\Scripts\python.exe"
& $py -m nullscape.data.earth tiles    --config configs/dataset/earth64_sample.yaml   # list, estimate download
& $py -m nullscape.data.earth download --config configs/dataset/earth64_sample.yaml   # resumable
& $py -m nullscape.data.earth build    --config configs/dataset/earth64_sample.yaml --workers 2
```

1. **Select tiles:** an explicit list, or a seeded random sample within latitude/longitude limits.
2. **Download** to `%LOCALAPPDATA%\nullscape\dem_cache\glo30` (outside OneDrive; override with `source.cache`
   or `NULLSCAPE_DEM_CACHE`). Files arrive as `.part` and are renamed when complete, so an interrupted run
   resumes. Downloads estimated above `--max-gb` (default 5) are refused.
3. **Decode.** The tiles use DEFLATE with the floating-point predictor, which `tifffile` cannot decode
   without the large `imagecodecs` package. `decode_float_predictor_tile` implements it in numpy.
   Validation on N47 E019: averaging the decoded 3600x3600 grid 2x2 reproduces GDAL's own overview
   exactly (mean difference 0.0000 m). Decoding takes about 0.4 s per tile.
4. **Cut windows** that are square in metres at the target cell size. The window size in pixels uses the
   WGS84 metres-per-degree at each window's latitude; resampling is anti-aliased bilinear. Windows stay
   inside one tile. Up to `windows_per_tile` are taken in a seeded random order. A tilted-plane test
   recovers slopes to within 0.6 degrees at 45 and 62 degrees latitude.
5. **Filter:** non-finite data, clipping, more than `max_water_fraction` water (default 0.9), and relief
   below `min_relief_m` (default 1 m).
6. **Measure** each window with the standard metrics on the stored uint16 heights, so conditions match the
   data exactly.
7. **Label** with ordered rules on physical values (`relief_m`, `mean_elevation_m`, `water_fraction`,
   `mean_slope_deg`, `p90_slope_deg`). Defaults: coastal, mountains, hills, plateau, plains (catch-all).
8. **Split by geography.** Each tile belongs to a `block_deg` x `block_deg` block, and a block's split is a
   hash of (seed, block). Neighbouring terrain never straddles train and test, and adding tiles later
   never moves a region to another split.

### Height encoding

`h = (z + datum_offset_m) / max_height_m`, clipped to [0, 1]. Defaults: offset 500 m (Dead Sea -430 m),
range 9500 m (Everest fits), so the uint16 step is 0.145 m. Water is `z < sea_threshold_m` (0.5 m); the
world's `sea_level` is derived from these values. The manifest records the formula, the vertical datum,
and the source elevation range.

## Datasets

| config | windows | cells | tiles | split blocks | download | build time (measured) |
|---|---|---|---|---|---|---|
| `earth64_sample` (built) | 1,024 | 64x64 at 64 m | 16 hand-picked | 1 degree | 0.6 GB | 9.5 s, 2 workers |
| `earth64` | about 50k | 64x64 at 64 m | 1500 random, 60S-72N | 5 degrees | about 59 GB | about 8 min expected |
| `earth256` | about 48k | 256x256 at 30 m | same 1500 | 5 degrees | shared with earth64 | 0.11 s/map with 2 workers (about 90 min) |

`earth64` and `earth256` select the same tiles and blocks, so one download serves both and they share
held-out regions. Neither has been downloaded yet (it needs `--max-gb 70`).

`earth64_sample` puts Hawaii and Tenerife in test, and Norway, Uluru and Colorado in validation.
Figures: `reports/earth64_sample/` (`mixed.png`, per-type grids, `earth_examples.png` next to
`procedural_examples.png`).

## Findings

**Real terrain spans a much wider range than our generator** (same 4.1 km scale; the sample leans mountainous):

| property | earth p10 / p50 / p90 | procedural base64 p10 / p50 / p90 |
|---|---|---|
| relief (m) | 14 / 399 / 1181 | 81 / 269 / 455 |
| mean slope (deg) | 0.8 / 9.1 / 28.5 | 3.8 / 12.4 / 23.4 |
| spectral beta | 2.7 / 4.2 / 5.0 | 3.5 / 4.5 / 6.2 |

Real terrain has flatter plains, taller mountains, and rougher fine-scale detail (lower beta). A model that
matches the procedural data perfectly would still miss much of this range.

**Flat real terrain is bumpy at 64 m cells.** Hungarian plain windows (relief about 15 m) have 30-40 local
minima/maxima per 1000 cells; procedural data averages 8.4/11.9. Windows with relief of 60 m or more match
procedural levels (about 8 each). On near-flat ground, 1-2 m of real micro-relief and DEM noise dominate a
metric that ignores amplitude.

**A bare-earth model does not fix it.** GEDTM30 v1.2 (OpenGeoHub, CC-BY 4.0, trees and buildings removed)
was tested on the same 200 windows. It reads fine by HTTP range requests (one 432 GB global COG; one tile
costs about 34 MB and 17 s). Flat windows got bumpier (minima 33 -> 40, high-frequency energy 0.064 -> 0.075),
and rugged windows stayed the same. FABDEM is also bare-earth, but CC BY-NC-SA, which would block commercial
use of the weights. Decision: stay with Copernicus GLO-30.

## What Talus-4 must change

- **Relief floor in metres.** `models/heightparam.py` floors relief at 0.01 normalized units, which is 12 m at
  the procedural scale but 95 m at `max_height_m` 9500. 25% of real windows fall below it, so their shapes would
  be squashed, recreating the Talus-1 plains problem. Make the floor a physical value (a few metres).
- **Amplitude-aware micro-relief metric.** "Clean detail" counts extrema regardless of size. On real data it
  needs a minimum prominence (for example 0.5 m) or it will penalize real plains.
- **Terrain types are rules, not generator archetypes.** Five classes instead of six, so Talus-2/3 checkpoints
  cannot be fine-tuned on Earth data without resizing the label embedding. Talus-4 is a new model anyway.
- **Register the CLI** under `nullscape earth ...` once the Talus-3 queue finishes. It is standalone for now so
  the running queue never loads edited code.
