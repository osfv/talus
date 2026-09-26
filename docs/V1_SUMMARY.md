# NULLSCAPE v1: summary

Plain-English findings from the frozen v1 benchmark. Details and every number:
[`V1_BENCHMARK.md`](V1_BENCHMARK.md). Scorecard: [`BENCHMARK_SCORECARD.md`](BENCHMARK_SCORECARD.md).

## The official v1

**Checkpoint 30k + 50-step sampler (quadratic spacing, guidance 2.0).**

- **The checkpoint was chosen on validation data.** The held-out test split agrees: 30k and 40k
  are tied, and 30k is better on slopes and conditioning. Training more than about 20-30k steps
  did not help.
- **The sampler is 4x faster than the old 200-step default** (5.5 vs 1.4 maps per second on the
  RTX 5060) and slightly better on realism, slopes and control. 200 steps is never worth it.

## What v1 does well

- **Big-picture terrain.** Height distributions are nearly indistinguishable from the procedural
  terrain (1.1x the real-vs-real noise floor). Every terrain type has the right large-scale shape.
- **Control.** Mean elevation, relief, slope and water can be dialled in:
  - errors are 2%, 7%, 11% and 5% of the natural spread
  - they hold across the whole range, from low to high values
  - these controls barely interfere with each other
- **Playable maps.** 90% pass the traversability check, the same as procedural terrain (92%).
- **Original, not copied.** Learned maps are no closer to the 45,000 training maps than brand-new
  procedural maps are, even allowing all rotations and flips. The closest cases were inspected by
  eye and none are copies.
- **Clean rendering.** No checkerboard or upsampling artifacts at any checkpoint.
- **Reproducible.** The dataset regenerates bit-for-bit, all 90 tests pass, and every export format
  round-trips exactly.

## Where v1 falls short

1. **Flat terrain is noisy.** This is the biggest problem: plains carry fine grain that real plains
   don't have (about 3,000x too much finest-scale detail, 4.6x too many tiny pits). Asking for "flat"
   gives flat-but-rough ground.
2. **Rugged terrain is slightly too smooth** at hillside scale. Mountains and ridges lose some fine
   detail, and mesa cliff edges are a bit soft.
3. **Too many tiny bumps overall.** About 50-70% more micro-peaks and pits than real terrain.
4. **The roughness control (spectral beta) doesn't work.** It barely moves the output and pulls
   everything toward average roughness.
5. **Water needs context.** It works when the other properties are given, but requesting water on
   its own produces much less water than asked for.
6. **It is slow for real-time use.** 5.5 maps per second in batches, about 2 s for a single map.
7. **It is statistically distinguishable.** Across all 25 metrics it is 2.9x the noise floor. That
   is far better than simple baselines (blurred maps 10.7x, shaped noise 9.3x) but not
   indistinguishable.

**On mesas:** the data does not support the suspicion that mesas are the weakest terrain type. They
are second best overall, with the best spectrum and the best conditioning. **Plains are the weakest**
and cause most of the spectrum, detail and conditioning error.

## Why (best current explanation)

The model works on a global height scale where plains use only about 5% of the range. Small leftover
errors from the denoising process are therefore large relative to plains' tiny relief, and show up as
grain. Rugged terrain hides the same errors but is under-trained on fine detail. The loss weighting
used in training down-weights exactly the low-noise steps that form fine detail.

## What v2 should try (in order)

1. **Fix the export bugs.** Unity sizing upsamples already-valid sizes (e.g. 513 -> 1025), and the
   sidecar footprint is off by one cell. See `benchmarks/v1/BUGS.md`.
2. **Cheap experiment:** change the loss weighting and add learning-rate decay, as a short fine-tune.
   It should restore fine detail on rugged terrain.
3. **Main v2 change:** generate each map's shape at full scale and rescale it with the requested
   elevation and relief. Flat maps then get full precision. It should remove the plains grain and the
   extra micro-bumps.
4. **Replace the roughness control** with a better-defined roughness measure.
5. **Speed:** distil to fewer steps and lighten the full-resolution layers.

Each change gets scored with this same benchmark. The scorecard will show a "Δ vs v1" column, and a
change only counts if it improves its target without hurting originality or control.
