# NULLSCAPE-Bench scorecard

Held-out TEST split, 1000 maps per half, noise-floor protocol (`docs/EVALUATION.md`). Scores are 0-100, higher is better; **100 = indistinguishable from real procedural terrain** at this sample size (fidelity scores are `100 / (distance / noise floor)`). Definitions: `benchmarks/scorecard.py`.

## Headline: Talus-3.1 official

Configuration: checkpoint 12000, 50-step DDIM (quadratic spacing), guidance 2.0, eta 0.0.

| Benchmark | What it measures | Real procedural | Talus-3.1 official | Talus-3 official | Δ vs Talus-3 official | baseline: blur | baseline: spectral_noise | baseline: retrieval |
|---|---|---|---|---|---|---|---|---|
| **Realism** | all 25 terrain metrics vs real (normalized W1) | 100 | **64.5** | 66.2 | -1.8 (-3%) | 9.3 | 10.7 | 100.0 |
| **Spectrum** | multi-scale detail: power spectrum vs real | 100 | **11.3** | 11.0 | +0.3 (+3%) | 0.3 | 1.2 | 63.7 |
| **Heights** | elevation distribution vs real | 100 | **98.3** | 99.1 | -0.8 (-1%) | 97.9 | 71.0 | 97.8 |
| **Slopes** | slope distribution vs real | 100 | **59.0** | 60.5 | -1.5 (-3%) | 7.3 | 6.1 | 93.8 |
| **Control** | requested vs measured properties (1 - mean nMAE) | 100 | **92.4** | 92.1 | +0.3 (+0%) | 7.5 | 77.6 | 95.6 |
| **Playability** | traversability pass rate vs real | 100 | **99.3** | 99.9 | -0.5 (-1%) | 100.0 | 100.0 | 98.7 |
| **Diversity** | mean pairwise difference vs real | 100 | **94.6** | 95.0 | -0.3 (-0%) | 92.6 | 94.0 | 95.0 |
| **Originality** | distance to nearest training map vs held-out maps | 100 | **98.4** | 99.1 | -0.8 (-1%) | 90.1 | 98.3 | 0.0 |
| **Clean detail** | no excess micro-bumps (sink + peak density vs real) | 100 | **99.5** | 99.4 | +0.2 (+0%) | 100.0 | 92.0 | 100.0 |
| **Average** | mean of the 9 scores (only if Originality >= 90) | 100 | **79.7** | 80.3 | -0.6 (-1%) | 56.1 | 61.2 | gated (copies) |
| **Speed** | maps/s, RTX 5060, batch 128 | - | **5.60** | 5.84 | 1.0x | - | - | - |

Baselines are non-learned reference generators scored with the same protocol: **blur** = real maps blurred (missing detail), **spectral_noise** = noise with each map's roughness and height statistics (no landforms), **retrieval** = the training map with the closest conditions (a perfect memorizer: real terrain, zero originality).

How to read it: fidelity scores (Realism, Spectrum, Heights, Slopes) are strict, because at 1000 maps per half the real-vs-real floor is very small. Clean detail only penalizes *excess* micro-bumps, so over-smoothed output (blur) scores 100 there; always read it together with Spectrum. The Average is a tracking number, not a ranking. It is withheld for anything that copies training data.

## Training progression (same sampler for all checkpoints: 200-step DDIM, uniform, guidance 1.5)

| Benchmark | Talus-1 20k | Talus-1 25k | Talus-1 30k | Talus-1 35k | Talus-1 40k | Δ Talus-1 40k vs Talus-1 20k |
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

## Raw values behind the scores (Talus-3.1 official)

| distance | learned vs real | real vs real (floor) | ratio |
|---|---|---|---|
| metric W1 mean | 0.0768 | 0.0495 | 1.55x |
| RAPSD distance (decades) | 0.0913 | 0.0103 | 8.9x |
| height W1 (normalized) | 0.0085 | 0.0084 | 1.02x |
| slope W1 (deg) | 0.424 | 0.250 | 1.70x |

| quantity | value |
|---|---|
| adherence nMAE (mean of 5) | 0.076 |
|   mean_elevation nMAE | 0.002 |
|   relief nMAE | 0.044 |
|   mean_slope_deg nMAE | 0.046 |
|   water_fraction nMAE | 0.042 |
|   spectral_beta nMAE | 0.248 |
| traversability pass (learned / real) | 0.911 / 0.917 |
| diversity (learned / real) | 0.1573 / 0.1662 |
| sink density per 1000 land cells (learned / real) | 8.48 / 8.40 |
| peak density per 1000 cells (learned / real) | 11.15 / 11.93 |
| NN-to-train median ratio | 0.984 |
