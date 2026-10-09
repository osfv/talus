# Talus-3: a longer, gentler fine-tune

Talus-3 continues Talus-2 for 12,000 more steps at half its learning rate, with the same data, loss
and sampler. It raises the TEST scorecard average from 77.5 to 80.3. All nine scores go up, most of
all Realism (+9.7), Slopes (+4.1) and Spectrum (+3.4). It is the first Talus release with public
weights.

## Setup

- **Start:** Talus-2 (`runs/20260926-192753_talus2_relative`, step 10,000), relative heights.
- **Recipe** (`configs/train/talus3_control.yaml`): learning rate 5e-5 with 300 warmup steps and
  cosine decay to 5e-6, batch 40, 12,000 steps, EMA 0.9995, bf16, seed 3, equal loss weights for all
  terrain types. It took 47 minutes on an RTX 5060.
- **Run:** `runs/20260927-184009_talus3_control`, step 12,000.
- **Release file:** `talus-3.pt` (93 MB, EMA weights only), made with `nullscape release-checkpoint`.
  It samples bit-for-bit like the training checkpoint.
- **Sampler:** unchanged: 50-step DDIM, quadratic spacing, guidance 2.0, eta 0. A VAL sweep on Talus-2
  found nothing better among 25 steps or guidance limited to the [0.05, 0.8] noise interval.
- **Scoring:** the v1 protocol on TEST, 1,000 maps per half, with the same halves, conditions and
  seeds as every earlier version.

## Choosing the checkpoint on VAL

Two matched recipes ran through the experiment queue (`configs/queue/talus3.yaml`): this one, and a
"rugged" variant that doubled the loss weight of mountains and ridges. VAL, 500 maps per half, seed 7,
ratios to the noise floor (lower is better):

| | Talus-2 | **Talus-3 (control)** | rugged |
|---|---|---|---|
| metric W1 | 1.252 | **1.182** | 1.212 |
| spectrum (RAPSD) | 2.133 | **1.828** | 1.924 |
| slopes | 1.427 | 1.341 | **1.316** |
| control nMAE | 0.092 | **0.077** | 0.081 |
| mountains | 1.279 | **1.163** | 1.173 |
| ridges | 1.628 | 1.417 | **1.405** |
| plains | 1.440 | **1.359** | 1.412 |

Doubling the rugged loss weights did not help mountains and cost plains, so the control recipe won.
Its in-training check preferred step 6,000, so we compared step 6,000 and step 12,000 with Talus-2 on
three more VAL seeds (0, 1, 2). These are raw distances, paired by seed, with the number of seeds
that beat Talus-2 in brackets:

| | Talus-2 | step 6,000 | **step 12,000** |
|---|---|---|---|
| metric W1 | 0.0732 | 0.0736 [1/3] | 0.0744 [1/3] |
| RAPSD | 0.1014 | 0.1005 [2/3] | **0.0816 [3/3]** |
| slope W1 (deg) | 0.408 | **0.375 [2/3]** | 0.411 [1/3] |
| control nMAE | 0.0874 | 0.0812 [3/3] | **0.0745 [3/3]** |
| mountains W1 | 0.2003 | 0.2238 [0/3] | **0.1969 [3/3]** |
| ridges W1 | 0.2351 | 0.2326 [2/3] | **0.2163 [3/3]** |
| plains W1 | 0.1968 | 0.1975 [1/3] | **0.1940 [2/3]** |

Step 12,000 won on spectrum, control, mountains and ridges on every seed. The overall metric W1 tied
Talus-2 on these seeds, so the TEST gain in Realism below is larger than VAL predicted.

## Result on TEST

Full scorecard with baselines and the delta against Talus-2: [`TALUS3_SCORECARD.md`](TALUS3_SCORECARD.md).

| Benchmark | Talus-1 | Talus-1.1 | Talus-2 | **Talus-3** |
|---|---|---|---|---|
| Realism | 34.4 | 35.8 | 56.5 | **66.2** |
| Spectrum | 3.6 | 2.9 | 7.6 | **11.0** |
| Heights | 88.5 | 96.7 | 98.3 | **99.1** |
| Slopes | 41.2 | 60.0 | 56.4 | **60.5** |
| Control | 81.8 | 84.5 | 90.4 | **92.1** |
| Playability | 98.4 | 97.6 | 97.6 | **99.9** |
| Diversity | **96.9** | 95.2 | 94.4 | 95.0 |
| Originality | **100** | **100** | 98.1 | 99.1 |
| Clean detail | 58.4 | 60.9 | 97.7 | **99.4** |
| **Average** | 67.0 | 70.4 | 77.5 | **80.3** |

Raw TEST values (ratio = distance / real-vs-real noise floor; 1.0 = indistinguishable):

| | Talus-2 | Talus-3 |
|---|---|---|
| metric W1 ratio (25 metrics) | 1.77 | **1.51** |
| RAPSD ratio | 13.2 | **9.1** |
| slope W1 ratio | 1.77 | **1.65** |
| per-type ratio: plains / hills / mountains | 1.23 / 1.40 / 1.93 | **1.17 / 1.16 / 1.71** |
| per-type ratio: ridges / islands / mesas | 2.16 / 1.44 / **1.15** | **1.80 / 1.21** / 1.43 |
| sinks / peaks per 1000 cells (real: 8.40 / 11.93) | 8.81 / 11.63 | **8.51** / 11.23 |
| nMAE mean elevation / relief / slope / water / spectral beta | .003 / .045 / .066 / .042 / .322 | **.002 / .042 / .051 / .042 / .256** |
| traversability pass rate (real: 0.917) | 0.895 | **0.916** |
| NN-to-train median ratio, share below held-out p1 | 0.981, 1.4% | **0.991, 1.1%** |
| maps/s at batch 128, RTX 5060 | 5.48 (Talus-1 timing) | 5.84 (measured) |

## What changed, and what didn't

- **Realism rose from 56.5 to 66.2.** The 25-metric distance dropped from 1.77x to 1.51x the noise
  floor.
- **Rugged terrain improved most.** Hills went from 1.40 to 1.16, ridges from 2.16 to 1.80, mountains
  from 1.93 to 1.71. Equal loss weights and more steps did this; the doubled rugged weights did not.
- **The spectrum error fell by a third** (RAPSD 13.2x to 9.1x), but 9x is still the largest gap in the
  scorecard. Mountains lack fine detail (-0.44 decades in the finest band), and plains carry too much
  (+0.84 decades).
- **Mesas regressed** from 1.15 to 1.43. VAL showed mesas slightly better on two of three seeds, so
  this needs a closer look in the next round.
- **Control tightened.** The roughness control (spectral beta) improved from 0.32 to 0.26 nMAE and
  slope from 0.066 to 0.051.
- **No memorization.** Generated maps sit slightly farther from the training set than Talus-2's
  (NN ratio 0.991), and Originality rose to 99.1.
- **Speed is unchanged** in principle (same architecture and sampler). The new measurement of the
  release file gives 5.84 maps/s at batch 128, 6.12 at batch 64 and 1.32 s for a single map; earlier
  speed numbers were Talus-1 timings.

## Talus-3.1 so far

Two short continuations of Talus-3 (`configs/queue/talus31.yaml`, 5,000 steps at learning rate 2e-5)
both scored worse than Talus-3 on VAL (seed 7):

| | Talus-3 | from Talus-3 rugged, equal weights | from Talus-3, same recipe |
|---|---|---|---|
| metric W1 ratio | **1.182** | 1.261 | 1.270 |
| spectrum ratio | **1.828** | 2.089 | 2.121 |
| control nMAE | **0.077** | 0.083 | 0.083 |
| NN-to-train median ratio | 0.977 | 0.956 | 0.955 |

Both moved closer to their training maps. Across Talus-1 to Talus-3, the model has seen each of
base64's 45,000 training maps about 60 times. So a third run (`configs/queue/talus31_fresh.yaml`) repeated
the second recipe on 96,000 freshly generated maps from the same generator
(`configs/dataset/base64_fresh.yaml`), scored on the same VAL maps:

| | Talus-3 | same recipe, base64 | **same recipe, fresh maps** |
|---|---|---|---|
| metric W1 ratio | **1.182** | 1.270 | 1.242 |
| spectrum ratio | **1.828** | 2.121 | 1.980 |
| slopes ratio | 1.341 | 1.409 | **1.288** |
| control nMAE | **0.077** | 0.083 | 0.079 |
| mountains / ridges / plains | 1.163 / 1.417 / **1.359** | 1.217 / 1.460 / 1.446 | **1.120 / 1.412** / 1.392 |

Fresh maps beat the same recipe on old maps in every column, so repeated epochs explain part of the
regression. They still trail Talus-3 on overall realism and spectrum. The remaining suspect is the restart
itself: each continuation begins with a fresh optimizer and warms the learning rate back up to 2e-5, four
times the rate Talus-3 ended on. These are single-seed VAL screens; nothing here was scored on TEST, and
Talus-3 stays the release.

## Next

- **Talus-3.1:** continue on fresh maps without the restart: a low constant learning rate near where
  Talus-3 ended, for longer than 5,000 steps.
- **Spectrum:** the finest-band errors in mountains and plains.
- **Talus-4:** real terrain from Copernicus GLO-30 ([`EARTH_DATA.md`](EARTH_DATA.md)).
