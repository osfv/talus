# v2 experiment 1: no Min-SNR cap, cosine learning-rate decay

Hypothesis 2 from [`V1_BENCHMARK.md`](V1_BENCHMARK.md): v1's Min-SNR-5 loss weighting down-weights
the low-noise steps that form fine detail, and the constant learning rate never let the model
settle. Removing the cap and decaying the learning rate should restore fine-scale detail on
rugged terrain. It should *not* fix the plains grain, which hypothesis 1 attributes to global
height normalization.

## Setup

- **Start:** the frozen official v1 checkpoint (30k). It stays untouched; the fine-tune writes a
  new run.
- **Config:** `configs/train/v2_exp1_finetune.yaml`
  - `min_snr_gamma: null`
  - lr 1e-4 with 200 warmup steps, cosine decay to 1e-5
  - 8,000 steps, batch 48, same data and architecture
- **Run:** `runs/20260926-181917_v2_exp1_nominsnr_cosine`, 31 minutes at 4.3 it/s on the RTX 5060.
- **Scoring:** the v1 protocol with the official v1 sampler (50-step quadratic DDIM, guidance 2.0)
  and the same halves, conditions and seeds, so every difference comes from the weights.
  - Validation: `benchmarks/v2/compare_exp.py`
  - Test: `benchmarks/v2/score_test.py`

## Result on TEST (1000 maps per half)

Full scorecard: [`V2_EXP1_SCORECARD.md`](V2_EXP1_SCORECARD.md).

| Benchmark | v1 official | exp1 | Δ |
|---|---|---|---|
| Realism | 34.4 | **35.8** | +1.4 |
| Spectrum | **3.6** | 2.9 | -0.7 |
| Heights | 88.5 | **96.7** | +8.2 |
| Slopes | 41.2 | **60.0** | +18.8 (+46%) |
| Control | 81.8 | **84.5** | +2.8 |
| Playability | **98.4** | 97.6 | -0.8 |
| Diversity | **96.9** | 95.2 | -1.7 |
| Originality | 100 | 100 | 0 |
| Clean detail | 58.4 | **60.9** | +2.5 |
| **Average** | 67.0 | **70.4** | **+3.4** |

Raw TEST values are in `benchmarks/v2/exp1/results.json`. The full visual suite is in
`benchmarks/v2/exp1/artifacts_official/`.

## Hypothesis check (VAL, 500 maps per half)

Spectrum error is the mean log10 power, learned minus procedural; 0 is perfect.

| measure | v1 30k | exp1 4k | exp1 8k |
|---|---|---|---|
| metric W1 ratio | 1.951 | 2.079 | 1.934 |
| slope W1 ratio | 0.761 | 0.975 | 0.486 |
| adherence nMAE | 0.169 | 0.150 | 0.144 |
| sinks / peaks per 1000 cells (ref 8.40 / 12.05) | 15.4 / 19.6 | 15.4 / 19.2 | 14.5 / 18.3 |
| rugged archetypes, finest band k25-32 | -0.167 | -0.093 | **-0.053** |
| rugged archetypes, k13-24 | -0.058 | -0.108 | -0.070 |
| plains, finest band k25-32 | 3.504 | 3.524 | 3.369 |

- **Confirmed for rugged terrain.** The finest-scale deficit of mountains, ridges and mesas shrank
  by about 70%. The slope distribution came much closer to real, and conditioning improved.
- **Plains are unchanged,** as predicted: +3.4 decades of excess fine-scale power remains. That is
  the target of hypothesis 1 (relative-height parameterization).
- **Why the global Spectrum score dropped:** in v1, the rugged archetypes' missing detail partly
  cancelled the plains' excess in the averaged spectrum. Fixing the rugged side removes that
  cancellation, so the plains error shows more clearly in the global number. The per-archetype
  spectra above are the reliable view.
- **The 4k checkpoint was worse than 8k.** The fine-tune only pays off once the learning rate has
  decayed.

## Verdict

exp1 is a better baseline than v1 for everything except the global spectrum number and small
drops in diversity and playability. The next experiment is hypothesis 1: relative heights for the
plains, started from exp1.
