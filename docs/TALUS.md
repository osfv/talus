# Talus: the NULLSCAPE terrain model

Talus is NULLSCAPE's terrain generator. The name comes from the rock debris that collects at the
foot of a cliff: terrain shaped by erosion. It is a conditional pixel-space diffusion U-Net
(23.2M parameters) that we trained from scratch on our own procedural heightmaps. There are no
pretrained weights and no external API.

## Versions

| version | checkpoint | sampler | TEST scorecard average | notes |
|---|---|---|---|---|
| **Talus-1** | `runs/20260925-164759_diffusion64/checkpoints/step_0030000.pt` (sha256 `1717b9f7…cb30`) | 50-step DDIM, quadratic, guidance 2.0 | 67.0 | frozen v1 baseline, [benchmark](V1_BENCHMARK.md) |
| **Talus-1.1** | `runs/20260926-181917_v2_exp1_nominsnr_cosine/checkpoints/step_0008000.pt` | same | 70.4 | 8k-step fine-tune of Talus-1 without the Min-SNR cap, cosine LR, [report](V2_EXP1.md) |
| **Talus-2** | `runs/20260926-192753_talus2_relative/checkpoints/step_0010000.pt` | same | 77.5 | relative heights, 10k-step fine-tune of Talus-1.1, [report](TALUS2.md) |
| **Talus-3** | release file `talus-3.pt` (sha256 `13b4297b…a6d9`), from `runs/20260927-184009_talus3_control/checkpoints/step_0012000.pt` | same | **80.3** | 12k-step fine-tune of Talus-2 at half the learning rate, [report](TALUS3.md) |
| **Talus-3.1** | release file `talus-3.1.pt` (sha256 `7d7d7e41…98e0`), from `runs/20261009-193828_talus31_fresh_12k/checkpoints/last.pt` | same | 79.7 | variant for flatter maps: 12k steps on 96k fresh maps from Talus-3, [report](TALUS31.md) |

Scorecards:
- [`BENCHMARK_SCORECARD.md`](BENCHMARK_SCORECARD.md): Talus-1
- [`V2_EXP1_SCORECARD.md`](V2_EXP1_SCORECARD.md): Talus-1.1 vs Talus-1
- [`TALUS2_SCORECARD.md`](TALUS2_SCORECARD.md): Talus-2 vs Talus-1.1
- [`TALUS3_SCORECARD.md`](TALUS3_SCORECARD.md): Talus-3 vs Talus-2
- [`TALUS31_SCORECARD.md`](TALUS31_SCORECARD.md): Talus-3.1 vs Talus-3

**Talus-3 is the default and the best all-round version** (GitHub release `v3.0.0`, Apache-2.0).
**Talus-3.1** (release `v3.1.0`) is better on plains, islands and mesas and worse on hills, mountains and
ridges. Only Talus-3 and Talus-3.1 weights are published; the `talus-1`, `talus-1.1` and `talus-2` aliases
work on the machine that trained them.

## What it makes

- **Output:** 64x64 normalized heightmaps covering 4 km, with heights up to 1,200 m and sea
  level at 0.2.
- **Terrain types:** plains, hills, mountains, ridges, islands, mesas.
- **Optional controls:** mean elevation, relief, mean slope, water fraction, spectral beta. Give
  any subset.
- **Speed:** 5.8 maps/s at batch 128 on an RTX 5060, about 1.3 s for a single map.
- **Determinism:** the same seed and settings give the same map.

```powershell
nullscape sample --checkpoint talus-3 `
  --archetype islands --prop water_fraction=0.6 --steps 50 --spacing quadratic --guidance 2.0 `
  --n 16 --formats png16,r16,obj --unity --out exports/talus_islands
```

## Known limits (measured, Talus-3)

- **Mountains and ridges are still the weakest types** (metric ratios 1.71 and 1.80). Mountains lack
  the finest-scale detail.
- **Mesas regressed** against Talus-2 (1.43 vs 1.15).
- **Overall spectrum is 9x the real-vs-real noise floor.** Realism is 1.51x.
- **The spectral beta control is the loosest** (nMAE 0.26).
- **The model only knows our procedural generator's world.** It has no real-Earth terrain yet.
