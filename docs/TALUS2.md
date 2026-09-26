# Talus-2: relative heights

Hypothesis 1 from [`V1_BENCHMARK.md`](V1_BENCHMARK.md): plains were grainy because the model worked in
absolute heights, where a flat map uses only a few percent of the value range. Small denoising errors
therefore looked large next to the plains' tiny relief.

**Talus-2** feeds the model each map's *shape* instead: `2 (h - mean) / relief`, with relief =
p98 - p2. The sampler then places the generated shape at the requested mean elevation and relief.
When a request leaves those open, it borrows them from a random one of the 16 most similar real
training maps, which are stored in the checkpoint.

Code:
- `src/nullscape/models/heightparam.py`
- `TerrainSampler._fill_placement`
- `train.height_param: relative`

## Setup

- **Start:** fine-tuned from Talus-1.1 (`configs/train/talus2_relative.yaml`).
  - no Min-SNR cap
  - lr 1e-4, cosine decay to 1e-5
  - 10,000 steps, batch 48
  - about 40 minutes on the RTX 5060
- **Run:** `runs/20260926-192753_talus2_relative`, checkpoint `step_0010000.pt`.
- **Sampler:** chosen on VAL among guidance 1.5 / 2.0 / 2.5 at 50 quadratic steps, by best VAL
  metric W1: 1.276 / **1.193** / 1.199. That keeps the Talus-1 configuration (guidance 2.0).
- **Scoring:** the v1 protocol on TEST with 1,000 maps per half, and the same halves, conditions and
  seeds as Talus-1.

## Result on TEST

Full scorecard, with baselines, and Δ against Talus-1.1: [`TALUS2_SCORECARD.md`](TALUS2_SCORECARD.md).

| Benchmark | Talus-1 | Talus-1.1 | **Talus-2** |
|---|---|---|---|
| Realism | 34.4 | 35.8 | **56.5** |
| Spectrum | 3.6 | 2.9 | **7.6** |
| Heights | 88.5 | 96.7 | **98.3** |
| Slopes | 41.2 | **60.0** | 56.4 |
| Control | 81.8 | 84.5 | **90.4** |
| Playability | **98.4** | 97.6 | 97.6 |
| Diversity | **96.9** | 95.2 | 94.4 |
| Originality | **100** | **100** | 98.1 |
| Clean detail | 58.4 | 60.9 | **97.7** |
| **Average** | 67.0 | 70.4 | **77.5** |

Raw TEST values (ratio = distance / real-vs-real noise floor; 1.0 = indistinguishable):

| | Talus-1 | Talus-1.1 | Talus-2 |
|---|---|---|---|
| metric W1 ratio (25 metrics) | 2.90 | 2.79 | **1.77** |
| RAPSD ratio | 28.0 | 34.5 | **13.2** |
| slope W1 ratio | 2.43 | **1.67** | 1.77 |
| sinks / peaks per 1000 cells (real: 8.40 / 11.93) | 14.6 / 20.1 | 14.1 / 19.2 | **8.8 / 11.6** |
| plains metric W1 ratio | 3.98 | 3.64 | **1.23** |
| nMAE mean elevation / relief / slope / water / spectral beta | .017 / .074 / .111 / .052 / .657 | .009 / .050 / .088 / .044 / .582 | **.003 / .045 / .066 / .042 / .322** |
| NN-to-train median ratio, share below held-out p1 | 1.013, 2.5% | 1.006, 2.1% | 0.981, **1.4%** |

## What changed, and what didn't

- **Plains are fixed.** The plains ratio went from 3.98 to 1.23. On VAL, their finest-band spectrum
  error dropped from +3.4 to +0.9 decades.
- **Micro-bumps are gone.** Sink and peak densities now match real terrain (8.8 vs 8.4 and 11.6 vs
  11.9), so Clean detail went from 58 to 98.
- **The spectrum error halved** (RAPSD ratio 28 to 13).
  - The in-training check had suggested a bigger drop (10.7x to 1.1x). That check uses 256 maps and
    50 uniform steps, where the noise floor is much larger.
  - The TEST number with 1,000 maps per half is the one to quote.
- **The roughness control started working.** Spectral beta nMAE halved (0.66 to 0.32).
- **Control is partly by design.** Placing each shape at the requested mean elevation makes that
  property nearly exact (nMAE 0.003). Relief, slope, water and roughness are still produced by the
  model, and they improved too.
- **Mountains and ridges are now the weakest types** (ratios 1.93 and 2.16). They lost part of
  Talus-1.1's fine-detail gain: on VAL the rugged finest band is back to -0.18 decades. This is the
  next target.
- **No memorization.**
  - The NN ratio dipped to 0.98, but fewer generated maps fall below the held-out 1st percentile
    than for Talus-1 (1.4% vs 2.5%).
  - Originality stays above the 90 gate (98.1).
- **Speed is unchanged:** same architecture and sampler, 5.5 maps/s.

## Next

- **Rugged fine detail:** a longer relative-height fine-tune, or per-archetype loss balancing.
- **The remaining spectrum gap** (RAPSD 13x the noise floor): limited-interval guidance, since
  guidance affects fine scales.
- **Distillation for speed.**
