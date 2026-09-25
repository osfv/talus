# Evaluation

The question we care about: **is learned terrain indistinguishable from held-out
procedural terrain, controllable, and not copied from the training set?** No single
number answers that, so the harness reports several and shows them next to visual
comparisons.

Code: per-map metrics in `src/nullscape/metrics/`, set-level protocol in
`src/nullscape/eval/core.py`.

## Per-map metrics (`metrics/quality.py`)

All metrics are computed in physical units using `WorldSpec` (cell size, max height)
and are invariant to rotations/flips.

| Metric | Meaning |
|---|---|
| `mean_elevation`, `std_elevation`, `min/max_elevation` | Normalized height statistics |
| `relief` | 98th minus 2nd percentile height |
| `skewness`, `kurtosis`, `hypsometric_integral` | Shape of the height distribution (eroded landscapes skew toward low values) |
| `water_fraction` | Fraction of cells below `sea_level` |
| `mean_slope_deg`, `p90_slope_deg`, `p99_slope_deg` | Central-difference slope in degrees |
| `ruggedness_m` | Terrain Ruggedness Index (Riley et al. 1999), meters |
| `curvature_std` | Std of the discrete Laplacian (change in slope per cell) |
| `spectral_beta` | Exponent of radially averaged power spectrum `P(k) ~ k^-beta` |
| `hf_energy` | Share of spectral power above half the Nyquist radius (noise / artifacts) |
| `checkerboard` | Amplitude of the `(-1)^(i+j)` pattern relative to std (upsampling artifacts) |
| `peak_density`, `sink_density` | Strict local maxima per 1000 cells; local minima per 1000 land cells (eroded terrain has few sinks) |

The model's conditioning vector is `CONDITION_KEYS = (mean_elevation, relief,
mean_slope_deg, water_fraction, spectral_beta)`. These are measured from the map, so
we can check whether a generated map actually has the requested properties.

## Traversability (`metrics/traversability.py`)

The agent is ground-bound with `max_slope_deg` (default 35). A cell is walkable if it
is land and its slope is within the limit. Neighboring walkable cells are connected
only if the height step between them is climbable (`|dz| <= tan(max_slope) * distance`),
so a one-cell cliff separates regions. Reported: land fraction, walkable fraction of
land, number of components, largest component share, the probability that two random
walkable cells can reach each other (`sum s_i^2 / (sum s_i)^2`), and whether the
largest component spans the map. A map passes if land >= 5%, walkable land >= 50%,
and largest component >= 70% of walkable cells (all configurable via `AgentSpec`).
`longest_route` returns an approximate diameter path for visualization.

## Set-level protocol (`eval/core.py`)

1. Split reference data (validation, or test for final numbers) into disjoint equal
   halves **A** and **B**.
2. Generate one map per map in A, conditioned on A's properties and labels (or with
   conditioning dropped for unconditional evaluation).
3. Compute every distance for **generated vs B** and for **A vs B** (the noise
   floor). Both use the same B and the same set sizes, so estimator bias cancels.
   `ratio_to_floor ~ 1` means the model is as close to B as real held-out data is.

Distances (lower is better):

- `metric_w1_mean`: mean over all per-map metrics (and traversability fractions) of
  the 1D Wasserstein-1 distance, normalized by the pooled std of that metric.
- `rapsd_distance`: RMS gap, in decades, between mean log power spectra. This is
  the most direct check of multi-scale realism.
- `height_w1`, `slope_w1_deg`: W1 between pooled per-cell height / slope distributions.
- Traversability pass rates and diversity (mean pairwise RMSE) are reported for both sets.

Also reported:

- **Condition adherence:** per property MAE, MAE / reference std, and Pearson r between
  requested and measured values, next to a random-pairing baseline.
- **Per-archetype distances** with their own noise floors.
- **Memorization:** nearest-training-neighbor RMSE (all 8 rotations/flips) of generated
  maps vs held-out real maps. `nn_median_ratio` well below 1, or a large
  `frac_gen_below_heldout_p01`, means the model is copying training maps.
- **Artifacts:** mean checkerboard / high-frequency energy / sink density, generated vs reference.

## Caveats

- Metrics are computed on procedural data, so they measure how well we learn the
  generator's distribution. They do not measure realism against Earth. Swapping in real
  DEMs is a planned follow-up.
- W1 on small sets is noisy. Use at least 500 maps per half for reported numbers and
  quote the noise floor alongside.
