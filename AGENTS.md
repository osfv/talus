# NULLSCAPE agent notes

## Environment
- Python venv lives outside OneDrive: `%USERPROFILE%\.venvs\nullscape` (torch 2.11+cu128; RTX 5060 sm_120 needs cu128 builds).
- Run tools via `"$env:USERPROFILE\.venvs\nullscape\Scripts\python.exe"` / `nullscape.exe`.
- Set `PYTHONUNBUFFERED=1` when redirecting training output to a log.

## Verification
- `python -m pytest` (full suite ~40-60 s, CPU). `-m "not slow"` skips end-to-end train/eval tests.
- When a GPU job is running, run tests with `OMP_NUM_THREADS=2..4` to avoid slowing it.

## GPU caveats (8 GB, Windows)
- Exceeding physical VRAM silently spills to system RAM (WDDM) and slows training 3-15x instead of OOM.
  Training caps itself with `train.cuda_memory_fraction` (default 0.72). Don't run a second GPU job next to training.
- Throughput reference: diffusion64 config, batch 48, ~4.3-4.6 it/s.

## State (2026-09-25)
- Dataset: `data/base64` (50k, 64x64). Main run: `runs/20260925-164759_diffusion64` (0-30k) +
  continuation `runs/20260925-192102_diffusion64_cont` (30k-40k). Checkpoint snapshots: `checkpoints/step_*.pt`.
- Artifacts: `artifacts/20260925-164759_diffusion64/index.md` (20k-40k). Model plateaued after ~30k
  (overall ratio ~1.7x noise floor; heights ~0.97x; no memorization).
- Next planned: `nullscape sampler-sweep` on the 40k checkpoint (val), then final `nullscape evaluate --split test`;
  then choose between real DEM data vs game features (tiling/outpainting, engine import).
