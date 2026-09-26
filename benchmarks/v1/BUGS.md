# Bugs and findings found during the v1 benchmark

None of these were fixed during the benchmark (behaviour frozen). Evidence lives in
`benchmarks/v1/raw/engineering.json` unless noted.

**Status (after the benchmark):** B1-B4 are fixed, with regression tests in
`tests/test_export.py`. The v1 benchmark numbers are unaffected, because these bugs only
touch export metadata, provenance and figure rendering.
- B1: `unity_size` keeps valid sizes.
- B2: `git_info` records no commit for repositories without commits.
- B3: artifact figures tolerate missing adherence keys.
- B4: sidecars report `footprint_m`, and `cell_size_m` is the true sample spacing.

## B1: `unity_size` upsamples inputs that are already valid Unity sizes (medium)

`src/nullscape/export/engine.py:35` computes the smallest `2^n + 1 >= resolution + 1`, i.e. the
next size strictly larger than the input:

| input | `unity_size` | already `2^n+1`? | R16 bytes | bytes if kept |
|---|---|---|---|---|
| 64 | 65 | no | 8,450 | 8,192 |
| 65 | **129** | yes | 33,282 | 8,450 |
| 128 | 129 | no | 33,282 | 32,768 |
| 129 | **257** | yes | 132,098 | 33,282 |
| 257 | **513** | yes | 526,338 | 132,098 |
| 513 | **1025** | yes | 2,101,250 | 526,338 |
| 1025 | **2049** | yes | 8,396,802 | 2,101,250 |

`513 -> 1025` is not intended. 513 is already `2^9 + 1`, so the export bicubically
upsamples to 4x the file size and adds no information. The current 64x64 and 128x128 outputs
are unaffected (64 -> 65, 128 -> 129 are correct). Any future 2^n+1 source (e.g. a 513 DEM
tile) would be silently doubled.

`tests/test_export.py::test_unity_size` asserts `(65, 129)` and `(513, 1025)`, and
`docs/ENGINE_EXPORT.md` repeats `513 -> 1025`, so the tests and docs enshrine the bug.
Suggested fix: smallest `2^n + 1 >= resolution`. Also correct the test cases and docs.

## B2: dataset manifest records `git.commit = "HEAD"` (low)

`utils/tracking.py:git_info` stores the stdout of `git rev-parse HEAD`. For a repository without
commits (the state when `data/base64` was built), git prints the literal `HEAD` to stdout, so
`data/base64/manifest.json` records `{"commit": "HEAD", "dirty": false}`. The dataset
provenance is therefore not tied to a commit. Reproducibility is still demonstrated directly:
100/100 regenerated maps are bit-identical to the stored ones.

## B3: artifact suite crashes when an eval half has < 2 maps (low)

`eval/core.py:condition_adherence` skips properties with fewer than 2 known samples, and
`eval/artifacts.py:refresh_run_figures` then indexes `condition_adherence[key]` and raises
`KeyError: 'mean_elevation'`. Only reachable with toy datasets (seen with the 24-map CLI smoke
dataset, whose val split has 2 maps). Suggested fix: require n >= 2 per half up front, or skip
missing keys in the figure.

## B4: Unity/engine sidecar footprint is off by one cell (low)

Samples are cell centers, so a 64-sample map spans `extent - cell = 4032 m`, not `4096 m`.
The `--unity` export resamples corner-to-corner to 65 samples (true spacing 63.0 m) but the
sidecar reports `extent_m = 4096` and `cell_size_m = 4096/65 = 63.015`. Setting a Unity terrain
to 4096 m therefore stretches the map horizontally by 1.6%. The native sidecar has the same
ambiguity for engines that place samples at grid vertices (4096/63 = 65.0 m spacing vs 64 m).
Suggested fix: report the sample-span footprint (`(n-1) * spacing`) and document which
convention each field uses.

## N1: `CUDA_VISIBLE_DEVICES=""` breaks CPU forcing on this setup (environment note)

With an empty `CUDA_VISIBLE_DEVICES`, torch 2.11 on Windows reports `is_available() == True`
with `device_count() == 0`. Code that picks `"cuda"` when available (`TerrainSampler`,
`environment_info`) then fails with `Invalid device id`. `CUDA_VISIBLE_DEVICES=-1` works. The
first engineering run (kept as `engineering/engineering_run1_empty_cvd.json`) hit this. Not a
NULLSCAPE logic bug, but the device choice could check `device_count() > 0`.

## N2: uint16 storage slightly changes a few measured properties (measurement note)

`benchmarks/v1/raw/quantization.json`: stored conditioning values equal the float-heightmap
measurements exactly (max diff 0). Measuring the same maps after uint16 round-trip changes:
- `spectral_beta` by up to 0.19 on individual very smooth maps (mean shift -0.002, i.e. -0.002 std).
- `sink_density` by -0.25 per 1000 land cells on average (-0.045 std), because quantization turns
  strict minima into plateaus.
- `peak_density` by -0.13 (-0.02 std).

Reference maps are evaluated after dequantization while generated maps are float, so
sink/peak comparisons carry this small bias. `analysis.json:quantization_sensitivity` re-scores
the selected checkpoint with quantized outputs.
