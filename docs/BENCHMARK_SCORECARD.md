# NULLSCAPE-Bench scorecard

Held-out TEST split, 1000 maps per half, noise-floor protocol (`docs/EVALUATION.md`). Scores are 0-100, higher is better; **100 = indistinguishable from real procedural terrain** at this sample size (fidelity scores are `100 / (distance / noise floor)`). Definitions: `benchmarks/scorecard.py`.

## Headline: v1 official

Configuration: checkpoint 30000, 50-step DDIM (quadratic spacing), guidance 2.0, eta 0.0.

| Benchmark | What it measures | Real procedural | v1 official | v1 30k | Δ vs v1 30k (default sampler) | baseline: blur | baseline: spectral_noise | baseline: retrieval |
|---|---|---|---|---|---|---|---|---|
| **Realism** | all 25 terrain metrics vs real (normalized W1) | 100 | **34.4** | 33.3 | +1.2 (+3%) | 9.3 | 10.7 | 100.0 |
| **Spectrum** | multi-scale detail: power spectrum vs real | 100 | **3.6** | 4.2 | -0.6 (-14%) | 0.3 | 1.2 | 63.7 |
| **Heights** | elevation distribution vs real | 100 | **88.5** | 92.0 | -3.5 (-4%) | 97.9 | 71.0 | 97.8 |
| **Slopes** | slope distribution vs real | 100 | **41.2** | 34.9 | +6.3 (+18%) | 7.3 | 6.1 | 93.8 |
| **Control** | requested vs measured properties (1 - mean nMAE) | 100 | **81.8** | 81.1 | +0.7 (+1%) | 7.5 | 77.6 | 95.6 |
| **Playability** | traversability pass rate vs real | 100 | **98.4** | 98.9 | -0.5 (-1%) | 100.0 | 100.0 | 98.7 |
| **Diversity** | mean pairwise difference vs real | 100 | **96.9** | 96.6 | +0.3 (+0%) | 92.6 | 94.0 | 95.0 |
| **Originality** | distance to nearest training map vs held-out maps | 100 | **100.0** | 100.0 | +0.0 (+0%) | 90.1 | 98.3 | 0.0 |
| **Clean detail** | no excess micro-bumps (sink + peak density vs real) | 100 | **58.4** | 65.8 | -7.4 (-11%) | 100.0 | 92.0 | 100.0 |
| **Average** | mean of the 9 scores (only if Originality >= 90) | 100 | **67.0** | 67.4 | -0.4 (-1%) | 56.1 | 61.2 | gated (copies) |
| **Speed** | maps/s, RTX 5060, batch 128 | - | **5.48** | 1.38 | 4.0x | - | - | - |

Baselines are non-learned reference generators scored with the same protocol: **blur** = real maps blurred (missing detail), **spectral_noise** = noise with each map's roughness and height statistics (no landforms), **retrieval** = the training map with the closest conditions (a perfect memorizer: real terrain, zero originality).

How to read it: fidelity scores (Realism, Spectrum, Heights, Slopes) are strict, because at 1000 maps per half the real-vs-real floor is very small. Clean detail only penalizes *excess* micro-bumps, so over-smoothed output (blur) scores 100 there; always read it together with Spectrum. The Average is a tracking number, not a ranking. It is withheld for anything that copies training data.

## Training progression (same sampler for all checkpoints: 200-step DDIM, uniform, guidance 1.5)

| Benchmark | v1 20k | v1 25k | v1 30k | v1 35k | v1 40k | Δ v1 40k vs v1 20k |
|---|---|---|---|---|---|---|
| **Realism** | 31.4 | 29.3 | 33.3 | 31.1 | 33.2 | +1.8 (+6%) |
| **Spectrum** | 4.3 | 3.7 | 4.2 | 4.5 | 4.1 | -0.3 (-7%) |
| **Heights** | 88.6 | 99.5 | 92.0 | 92.6 | 100.0 | +11.4 (+13%) |
| **Slopes** | 34.0 | 24.1 | 34.9 | 33.1 | 28.8 | -5.2 (-15%) |
| **Control** | 76.0 | 77.4 | 81.1 | 80.5 | 80.6 | +4.6 (+6%) |
| **Playability** | 99.3 | 99.3 | 98.9 | 99.5 | 98.7 | -0.7 (-1%) |
| **Diversity** | 98.2 | 96.0 | 96.6 | 96.3 | 95.9 | -2.2 (-2%) |
| **Originality** | 100.0 | 100.0 | 100.0 | 100.0 | 97.8 | -2.2 (-2%) |
| **Clean detail** | 74.5 | 71.7 | 65.8 | 68.7 | 65.7 | -8.7 (-12%) |
| **Average** | 67.4 | 66.8 | 67.4 | 67.4 | 67.2 | -0.2 (-0%) |

## Raw values behind the scores (v1 official)

| distance | learned vs real | real vs real (floor) | ratio |
|---|---|---|---|
| metric W1 mean | 0.1437 | 0.0495 | 2.90x |
| RAPSD distance (decades) | 0.2880 | 0.0103 | 28.0x |
| height W1 (normalized) | 0.0095 | 0.0084 | 1.13x |
| slope W1 (deg) | 0.608 | 0.250 | 2.43x |

| quantity | value |
|---|---|
| adherence nMAE (mean of 5) | 0.182 |
|   mean_elevation nMAE | 0.017 |
|   relief nMAE | 0.074 |
|   mean_slope_deg nMAE | 0.111 |
|   water_fraction nMAE | 0.052 |
|   spectral_beta nMAE | 0.657 |
| traversability pass (learned / real) | 0.902 / 0.917 |
| diversity (learned / real) | 0.1611 / 0.1662 |
| sink density per 1000 land cells (learned / real) | 14.64 / 8.40 |
| peak density per 1000 cells (learned / real) | 20.08 / 11.93 |
| NN-to-train median ratio | 1.013 |
