# NULLSCAPE

NULLSCAPE is a game-terrain research codebase: worlds are normalized heightmaps
(64x64 now, 128x128 next) with physical scale (extent, max height, sea level),
generated either procedurally (noise + archetypes + erosion) or by a learned
terrain model trained on the procedural data. The current phase trains and
evaluates a **conditional pixel-space diffusion model** — a U-Net denoiser with
cosine schedule, v-prediction, classifier-free guidance, and per-property +
archetype conditioning — to prove learned terrain can match the procedural
distribution and be steered by measured properties. See
[docs/MODEL_RESEARCH.md](docs/MODEL_RESEARCH.md) for the design rationale.

## Setup

Windows + RTX 50-series (Blackwell, sm_120): needs a CUDA 12.8 PyTorch build —
stock cu118 wheels do not support it. Keep the venv outside OneDrive.

```powershell
py -3.13 -m venv "$env:USERPROFILE\.venvs\nullscape"
& "$env:USERPROFILE\.venvs\nullscape\Scripts\Activate.ps1"
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

Data lands in `data/` and runs in `runs/`; override with `NULLSCAPE_DATA_ROOT`
and `NULLSCAPE_RUNS_ROOT`.

## Workflow

```bash
# 1. Procedural datasets (deterministic, parallel)
nullscape gen-dataset --config configs/dataset/smoke.yaml              # 512 maps, sanity
nullscape gen-dataset --config configs/dataset/base64.yaml --workers 8 # 50k, 64x64
nullscape gen-dataset --config configs/dataset/base128.yaml            # 50k, 128x128

# 2. Inspect a dataset
nullscape viz --dataset base64 --split val --n 36 --out reports/base64

# 3. Train
nullscape train --config configs/train/smoke.yaml                      # tiny sanity run
nullscape train --config configs/train/diffusion64.yaml                # the real model
nullscape train --config configs/train/diffusion64.yaml --set train.lr=1e-4
nullscape train --config configs/train/diffusion64.yaml \
    --resume runs/<run>/checkpoints/last.pt
tensorboard --logdir runs

# 4. Sample from a checkpoint (self-contained: weights + world + normalization)
nullscape sample --checkpoint runs/<run>/checkpoints/best.pt --n 16 \
    --steps 50 --guidance 1.5 --out outputs/samples
nullscape sample --checkpoint ... --archetype islands \
    --prop relief=0.3 --prop water_fraction=0.4   # any subset of properties

# 5. Evaluate against held-out procedural data (noise-floor protocol)
nullscape evaluate --checkpoint ... --n 1000 \
    --modes conditional,label_only,unconditional \
    --guidance-sweep 1,1.5,2,3 --out reports/eval

# 6. Side-by-side figures and engine exports
nullscape compare --checkpoint ... --n 6 --seeds 3 --out reports/compare
nullscape export --dataset base64 --index 123 --unity --out exports/
```

`--prop` keys are `manifest["condition_keys"]`: `mean_elevation`,
`relief` (both in normalized height units), `mean_slope_deg` (degrees),
`water_fraction` (fraction of cells below sea level), `spectral_beta`
(power-spectrum exponent). Unspecified properties and `--archetype` are left
to the model.

## Layout

```
configs/dataset/    dataset build configs (smoke, base64, base128)
configs/train/      training configs (smoke sanity, diffusion64)
docs/               MODEL_RESEARCH, DATASET, EVALUATION, ENGINE_EXPORT
src/nullscape/
  world.py          WorldSpec: physical meaning of normalized heightmaps
  terrain/          noise, erosion, archetypes, generate()
  metrics/          per-map quality metrics, traversability, distribution distances
  data/             dataset build (multiprocessing), storage, splits, TerrainDataset
  models/           UNet, GaussianDiffusion, EMA
  train/            training loop
  inference/        TerrainSampler (checkpoint -> heightmaps)
  eval/             evaluation protocol + CLI
  viz/              hillshade/color/traversability/3D rendering
  export/           png16 / r16 / npy / OBJ engine exports
  utils/            config, seeds, run tracking, paths
tests/              pytest suite (CPU)
```

## Reproducibility

- Every procedural map is a pure function of `(seed, index, world, generator
  config)` via `np.random.SeedSequence([seed, index])` — identical regardless
  of worker count or order.
- Each dataset manifest records the resolved config and its sha256, the
  generator version, git commit + dirty flag, and library versions.
- Each training run gets `runs/<timestamp>_<name>/` with `config.yaml`,
  `env.json`, `model.json`, `metrics.jsonl`, TensorBoard events, samples,
  eval reports, and checkpoints.
- Checkpoints are self-contained: UNet/diffusion configs, raw + EMA weights,
  optimizer state, dataset manifest subset (world spec, condition keys and
  train-split normalization stats, archetypes), so sampling/eval need no data.

## Docs

- [docs/MODEL_RESEARCH.md](docs/MODEL_RESEARCH.md) — model design rationale
- [docs/DATASET.md](docs/DATASET.md) — dataset format and generator pipeline
- [docs/EVALUATION.md](docs/EVALUATION.md) — evaluation protocol and metrics
- [docs/ENGINE_EXPORT.md](docs/ENGINE_EXPORT.md) — Unity/Unreal/Godot import

## Testing

```bash
python -m pytest              # full suite (CPU)
python -m pytest -m "not slow"   # skip the end-to-end train/eval smoke test
```

## Results

_Filled in after the first full training run._
