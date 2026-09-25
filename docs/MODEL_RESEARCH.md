# Model research: choosing a terrain generator architecture

## Problem statement

Learn a distribution over normalized heightmaps (64x64 to start, 128x128 next, larger
later) and sample from it given a seed and optional terrain properties. Requirements
that follow from the game use case:

1. **Visual quality at every scale.** Terrain has structure at all frequencies
   (power spectrum roughly `k^-beta`). Blurry output shows up right away as
   unnatural smooth slopes. High-frequency artifacts show up right away in hillshade.
2. **Coverage.** A world generator that produces a handful of near-identical
   maps is useless. Mode collapse is a hard failure.
3. **Controllability.** Designers must steer outputs (elevation, relief,
   ruggedness, water, terrain type), including partial specifications.
4. **Spatial editing later.** Seamless tiling for large worlds, inpainting
   (fix a region, regenerate the rest), sketch-guided generation.
5. **Trainable on one consumer GPU** (dev machine: RTX 5060, 8 GB) in hours, not weeks.
6. **Deterministic inference from a seed**, fast enough for offline world generation.

## Candidates

| Family | Representative work | Quality | Coverage | Conditioning / editing | Training cost & stability |
|---|---|---|---|---|---|
| VAE | Kingma & Welling 2014 | Blurry (Gaussian likelihood averages high frequencies) | Good | Easy conditioning; weak editing | Cheap, stable |
| GAN (DCGAN / ProGAN / StyleGAN2) | Beckham & Pal 2017 (terrain DCGAN); Spick et al. 2019 (spatial GAN on DEMs); Karras et al. 2018, 2020 | Sharp | Mode collapse risk, weak on small datasets | Class/continuous conditioning possible; inpainting requires separate training | Cheap per step, unstable; heavy tuning (R1, ADA, lazy reg) |
| Conditional GAN / pix2pix | Guérin et al. 2017 (sketch-to-terrain cGAN) | Sharp | Low diversity per input | Excellent for sketch->terrain, needs paired data | Moderate |
| VQ-VAE/VQGAN + autoregressive transformer | van den Oord 2017; Esser et al. 2021 | Good | Good | Good (token-level inpainting, unbounded outpainting) | Two stages, codebook collapse issues, slow sequential sampling |
| Denoising diffusion (pixel space) | Ho et al. 2020; Nichol & Dhariwal 2021; Lochner et al. 2023 (terrain authoring with diffusion); Hu et al. 2024 (Terrain Diffusion Network) | State of the art | Excellent (likelihood-based, no collapse) | Classifier-free guidance, training-free inpainting (RePaint), sketch channels by concatenation | Stable, simple MSE loss; slower sampling |
| Latent diffusion | Rombach et al. 2022 | State of the art at high res | Excellent | Same as diffusion | Extra autoencoder stage; not needed at 64-256 px |
| Flow matching / rectified flow | Lipman et al. 2023; Liu et al. 2023 | On par with diffusion | Excellent | Same as diffusion | Stable; fewer sampling steps |
| Foundation-scale text-to-image | SD/Imagen class | n/a | n/a | n/a | Out of scope: far beyond budget, not heightmap-native |

## Decision: small conditional pixel-space diffusion model (DDPM, v-prediction, U-Net)

**Why diffusion over a GAN.** GANs sample in one forward pass, but the costs land
exactly on our requirements. Coverage (req. 2) is the main GAN failure mode. Editing
(req. 4) needs extra networks or training. Stabilizing training needs careful tuning,
which slows experimentation. Diffusion trains with a plain regression loss, does not
mode-collapse, and gives guided sampling, inpainting and outpainting from the same
network. Lochner et al. 2023 and Hu et al. 2024 show this works for terrain.
At 64-128 px, sampling cost is small: 50 DDIM steps of a ~15M-parameter U-Net takes
milliseconds per map in a batch on a consumer GPU. For a game that generates
worlds offline or at load time, this does not matter.

**Why not VAE.** Blur kills realism in hillshade renders; fixing it needs adversarial
or perceptual losses, which brings GAN problems back.

**Why pixel space, not latent.** A 64x64x1 heightmap is 4096 values, smaller than the
latents of most latent-diffusion models. An autoencoder would add a training stage
and a reconstruction ceiling for no compute gain. Revisit at 512 px and above.

**Why DDPM/v-prediction rather than flow matching.** Both are viable. Flow matching
with linear interpolation is close to v-prediction diffusion under a particular
noise schedule. We pick DDPM + DDIM because its guidance and inpainting methods
(classifier-free guidance, RePaint) are well documented, which lowers risk.
Flow matching is a cheap follow-up ablation: same U-Net, different loss/sampler.

## Model specification (v1)

- **Denoiser:** 2D U-Net, 1 input channel, base width 64, channel multipliers
  (1, 2, 3, 4) over resolutions 64/32/16/8, 2 residual blocks per level, GroupNorm +
  SiLU, self-attention at 16x16 and 8x8, dropout 0.1, FiLM (scale-shift) conditioning in
  every residual block. 23.2M parameters. Downsampling
  with strided conv; upsampling with nearest-neighbor + conv (avoids the checkerboard
  artifacts of transposed convolutions, which we also measure).
- **Diffusion:** T = 1000, cosine noise schedule (Nichol & Dhariwal 2021),
  v-prediction target (Salimans & Ho 2022), MSE loss with Min-SNR-5 weighting (Hang et al.
  2023). Sampling with DDIM (50 steps default, eta configurable; eta = 1 approximates
  ancestral DDPM).
- **Conditioning:** a vector of terrain properties *measured from each heightmap*
  (`mean_elevation`, `relief`, `mean_slope_deg`, `water_fraction`, `spectral_beta`,
  z-scored on the train split) plus the archetype label. Each property has its own
  learned "unknown" embedding. During training each property is dropped independently
  with p = 0.15 and all conditioning with p = 0.1. At inference any subset can
  be specified, and classifier-free guidance (Ho & Salimans 2022) trades adherence
  against diversity. Because conditions are measurable, adherence is itself an
  evaluation metric.
- **Training:** AdamW (lr 2e-4, wd 0), linear warmup, bf16 autocast, grad clip 1.0,
  EMA of weights (decay 0.9995) used for sampling, random dihedral augmentation
  (terrain statistics are isotropic in our generator, and all conditioning properties
  are rotation invariant).
- **Data range:** heights in [0, 1] with global normalization, mapped to [-1, 1] for
  the model.

## Compute estimate

Measured on the dev GPU (RTX 5060 8 GB, bf16, `scripts/bench_train.py`): 3.6 it/s at
batch 64 (6.3 GB peak) and 4.4 it/s at batch 48, about 210 maps/s either way. The step
cost is dominated by the full-resolution 64x64 level, so shrinking the deeper levels (10M
parameter variant) only gains about 10%. The default run is 30k steps at batch 48 (about 32
epochs of the 45k-map train split, about 2 hours). 128x128 costs about 4x per step. The same
code runs unchanged on a single cloud GPU (L4/A10/A100) for larger sweeps.

On 8 GB Windows cards, keep total VRAM (activations + GPU-resident dataset + CUDA context +
desktop) under about 6.5 GB. Past that, WDDM silently spills to system memory and training
slows by more than 10x instead of failing with an out-of-memory error.

## How we will judge success

See `docs/EVALUATION.md`. In short: generated sets are compared with held-out
procedural maps using per-metric Wasserstein distances, radially averaged power
spectrum distance, pooled height/slope distributions, traversability pass rates,
diversity, a nearest-neighbor memorization check against the training set, and
condition adherence. Every distance is read against a noise floor computed between
two disjoint halves of real data. Visual side-by-side comparisons complement the
numbers.

## Follow-ups this design enables

- Seamless large worlds: tile-wise generation with overlap inpainting / outpainting.
- Sketch or mask guidance: extra input channels (river masks, ridge sketches).
- Flow-matching ablation; distillation for fewer steps if latency ever matters.
- Replace procedural training data with real DEMs (e.g. SRTM/Copernicus) once the
  pipeline is proven. That is the step that makes learned terrain beat its source.

## References

- Beckham, Pal. *A step towards procedural terrain generation with GANs.* arXiv 2017.
- Esser, Rombach, Ommer. *Taming Transformers for High-Resolution Image Synthesis.* CVPR 2021.
- Guérin et al. *Interactive Example-Based Terrain Authoring with Conditional GANs.* ACM TOG (SIGGRAPH Asia) 2017.
- Hang et al. *Efficient Diffusion Training via Min-SNR Weighting Strategy.* ICCV 2023.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Ho, Salimans. *Classifier-Free Diffusion Guidance.* 2022.
- Hu et al. *Terrain Diffusion Network: Climatic-Aware Terrain Generation with Geological Sketch Guidance.* AAAI 2024.
- Karras et al. *Progressive Growing of GANs.* ICLR 2018; *Analyzing and Improving the Image Quality of StyleGAN.* CVPR 2020.
- Kingma, Welling. *Auto-Encoding Variational Bayes.* ICLR 2014.
- Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023.
- Liu, Gong, Liu. *Flow Straight and Fast: Rectified Flow.* ICLR 2023.
- Lochner et al. *Interactive Authoring of Terrain using Diffusion Models.* Computer Graphics Forum 2023.
- Lugmayr et al. *RePaint: Inpainting using Denoising Diffusion Probabilistic Models.* CVPR 2022.
- Nichol, Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
- Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
- Spick, Cowling, Walker. *Procedural Generation using Spatial GANs for Region-Specific Learning of Elevation Data.* IEEE CoG 2019.
- van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017.
