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
| **Talus-2** | `runs/20260926-192753_talus2_relative/checkpoints/step_0010000.pt` | same | **77.5** | relative heights, 10k-step fine-tune of Talus-1.1, [report](TALUS2.md) |

Scorecards:
- [`BENCHMARK_SCORECARD.md`](BENCHMARK_SCORECARD.md): Talus-1
- [`V2_EXP1_SCORECARD.md`](V2_EXP1_SCORECARD.md): Talus-1.1 vs Talus-1
- [`TALUS2_SCORECARD.md`](TALUS2_SCORECARD.md): Talus-2 vs Talus-1.1

**Talus-2 is the current best version.**

## What it makes

- **Output:** 64x64 normalized heightmaps covering 4 km, with heights up to 1,200 m and sea
  level at 0.2.
- **Terrain types:** plains, hills, mountains, ridges, islands, mesas.
- **Optional controls:** mean elevation, relief, mean slope, water fraction, spectral beta. Give
  any subset.
- **Speed:** 5.5 maps/s at batch 128 on an RTX 5060, about 1.8 s for a single map.
- **Determinism:** the same seed and settings give the same map.

```powershell
nullscape sample --checkpoint runs/20260926-192753_talus2_relative/checkpoints/step_0010000.pt `
  --archetype islands --prop water_fraction=0.6 --steps 50 --spacing quadratic --guidance 2.0 `
  --n 16 --formats png16,r16,obj --unity --out exports/talus_islands
```

## Known limits (measured, Talus-2)

- **Mountains and ridges are the weakest types** (metric ratios 1.9 and 2.2). They lack some
  finest-scale detail.
- **Overall spectrum is 13x the real-vs-real noise floor.** Realism is 1.77x.
- **The spectral beta control is usable but loose** (nMAE 0.32).
- **The model only knows our procedural generator's world.** It has no real-Earth terrain yet.
