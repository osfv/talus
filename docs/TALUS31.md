# Talus-3.1: a variant for flatter maps

Talus-3.1 continues Talus-3 for 12,000 steps on 96,000 procedural maps the model had never seen. On the
TEST scorecard it trades a little overall realism (average 79.7 against Talus-3's 80.3) for better plains,
islands and mesas. Ridges, hills and mountains get worse. **Talus-3 stays the default.** Pick Talus-3.1
when your maps are mostly flat ground, coastlines or plateaus.

## Which one to use

| You are making | Use |
|---|---|
| A mix of terrain, or ridges, hills and mountains | **Talus-3** |
| Plains, islands and coasts, or mesas and plateaus | **Talus-3.1** |

```bash
nullscape sample --checkpoint talus-3.1 --archetype mesas --n 16 --formats png16,r16,obj --unity
```

## Setup

- **Start:** Talus-3 (`runs/20260927-184009_talus3_control`, step 12,000).
- **Data:** `base64_fresh` (`configs/dataset/base64_fresh.yaml`): 100,000 maps from the same generator,
  world and settings as `base64`, with a new seed; 96,000 for training. By Talus-3 the model had seen each
  of base64's 45,000 training maps about 60 times.
- **Recipe** (`configs/train/talus31_fresh_12k.yaml`): learning rate 2e-5 with 200 warmup steps and cosine
  decay to 2e-6, batch 40, 12,000 steps, EMA 0.9995, bf16, seed 4. It took 45 minutes on an RTX 5060.
- **Run:** `runs/20261009-193828_talus31_fresh_12k`, step 12,000. Release file `talus-3.1.pt`
  (93 MB, EMA weights only).
- **Sampler:** the same as every version: 50-step DDIM, quadratic spacing, guidance 2.0, eta 0.

## How it was chosen

All screening used VAL maps from `base64`, so every row below is comparable (500 maps per half, seed 7,
ratio to the noise floor, lower is better).

| | Talus-3 | 5k, base64 | 5k, fresh | **12k, fresh** |
|---|---|---|---|---|
| metric W1 | **1.182** | 1.270 | 1.242 | 1.200 |
| spectrum | 1.828 | 2.121 | 1.980 | **1.756** |
| slopes | 1.341 | 1.409 | **1.288** | 1.372 |
| control nMAE | 0.077 | 0.083 | 0.079 | **0.073** |
| plains / hills | 1.359 / 1.073 | 1.446 / 1.073 | 1.392 / 1.056 | **1.282 / 1.030** |
| ridges | 1.417 | 1.460 | **1.412** | 1.433 |

- More passes over base64 made Talus-3 worse and moved it closer to its training maps.
- Fresh maps beat the same recipe on base64 in every column.
- More steps on fresh maps helped: from step 2,500 to 5,000 to 12,000, the spectrum went 2.096, 1.980, 1.756.

On three more VAL seeds (0, 1, 2), paired with Talus-3, the 12k model was better on every seed for control,
plains, mountain fine detail and narrow route bottlenecks. It was worse on every seed for ridges. Overall
metric W1 was 3% worse and the spectrum tied. That is a trade-off, not an upgrade, so it ships as a variant.
That decision came before TEST, and TEST could not change which model is the default.

## Result on TEST

Full scorecard with baselines: [`TALUS31_SCORECARD.md`](TALUS31_SCORECARD.md). Same protocol, halves,
conditions and seeds as every earlier version.

| Benchmark | Talus-2 | Talus-3 | **Talus-3.1** |
|---|---|---|---|
| Realism | 56.5 | **66.2** | 64.5 |
| Spectrum | 7.6 | 11.0 | **11.3** |
| Heights | 98.3 | **99.1** | 98.3 |
| Slopes | 56.4 | **60.5** | 59.0 |
| Control | 90.4 | 92.1 | **92.4** |
| Playability | 97.6 | **99.9** | 99.3 |
| Diversity | 94.4 | **95.0** | 94.6 |
| Originality | 98.1 | **99.1** | 98.4 |
| Clean detail | 97.7 | 99.4 | **99.5** |
| **Average** | 77.5 | **80.3** | 79.7 |

Per-type realism ratio on TEST (lower is better):

| | plains | islands | mesas | hills | mountains | ridges |
|---|---|---|---|---|---|---|
| Talus-3 | 1.17 | 1.21 | 1.43 | **1.16** | **1.71** | **1.80** |
| Talus-3.1 | **1.13** | **1.16** | **1.33** | 1.20 | 1.75 | 1.93 |

- **Flat terrain improved.** Plains, islands and mesas all moved closer to real maps. Mesas recovered most
  of the regression they had in Talus-3 (1.43 to 1.33; Talus-2 had 1.15).
- **Rugged terrain got worse,** ridges most of all (1.80 to 1.93).
- **VAL overstated the mountains.** VAL showed mountains slightly better; TEST shows them slightly worse.
  Their finest-scale detail did improve a little on both (-0.44 to -0.42 decades on TEST).
- **Control is level.** Slope and roughness got a little more accurate (nMAE 0.051 to 0.046 and 0.256 to
  0.248), and relief a little less (0.042 to 0.044).
- **No memorization.** Against its own fresh training maps, Talus-3.1's nearest-neighbour ratio is 0.972 with
  1.4% of maps below the held-out 1st percentile. Talus-3, which never saw those maps, gets 0.977 and 1.4%.
  Against base64's training maps the TEST ratio is 0.984.
- **Speed is unchanged:** 5.6 maps/s at batch 128 on the release file.

## Next

The remaining gap is rugged terrain and the spectrum. A Talus-3.2 that beats Talus-3 on every type
probably needs more than longer training: more or different data for ridges and mountains, or changes to
the model.
