# NULLSCAPE agent notes

## Environment
- Python venv lives outside OneDrive: `%USERPROFILE%\.venvs\nullscape` (torch 2.11+cu128; RTX 5060 sm_120 needs cu128 builds).
- Run tools via `"$env:USERPROFILE\.venvs\nullscape\Scripts\python.exe"` / `nullscape.exe`.
- Set `PYTHONUNBUFFERED=1` when redirecting training output to a log.
- To force CPU use `CUDA_VISIBLE_DEVICES=-1`, not `""` (empty makes torch report a GPU with 0 devices).

## Verification
- `python -m pytest` (full suite ~40-60 s, CPU). `-m "not slow"` skips end-to-end train/eval tests.
- When a GPU job is running, run tests with `OMP_NUM_THREADS=2..4` and `CUDA_VISIBLE_DEVICES=-1`.
- Benchmark code paths: `NULLSCAPE_BENCH_DRY=1 python benchmarks/v1/run_v1_benchmark.py --stages ...` (CPU, tiny).

## GPU caveats (8 GB, Windows)
- Exceeding physical VRAM silently spills to system RAM (WDDM) and slows work 3-20x instead of OOM.
  Training caps itself with `train.cuda_memory_fraction` (default 0.72); the benchmark perf stage caps at 0.6.
  Don't run a second GPU job next to training or timing measurements.
- Throughput: training diffusion64 batch 48 ~4.3 it/s. Sampling at batch 128: 5.5 maps/s (50 steps + guidance),
  1.4 maps/s (200 steps + guidance). Guidance at batch 256 does not fit.

## State (2026-09-26)
- v1 is FROZEN: checkpoints under `runs/*/checkpoints/*.pt` and `data/base64/*` are read-only. Don't modify them.
- Official v1 = checkpoint 30k (`runs/20260925-164759_diffusion64/checkpoints/step_0030000.pt`) +
  50-step DDIM, quadratic spacing, guidance 2.0, eta 0.
- Benchmark: `docs/V1_BENCHMARK.md`, `docs/V1_SUMMARY.md`, `docs/BENCHMARK_SCORECARD.md`,
  `benchmarks/v1/results.json`, bugs in `benchmarks/v1/BUGS.md` (B1 unity_size upsamples valid sizes, B2 manifest
  git "HEAD", B3 artifacts crash on tiny splits, B4 sidecar footprint off by one cell). Not fixed yet.
- B1-B4 fixed (commit a813876). Trainer supports `train.init_from` (fine-tune) and `train.lr_schedule: cosine`.
- v2 exp1 (no Min-SNR, cosine LR, 8k steps from v1 30k): `runs/20260926-181917_v2_exp1_nominsnr_cosine`,
  scored in `docs/V2_EXP1.md` (TEST average 67.0 -> 70.4; slopes +46%; plains grain unchanged).
- Score an experiment: `python benchmarks/v2/compare_exp.py --name X --ckpt label=path` (VAL), then
  `python benchmarks/v2/score_test.py --name X --ckpt path` and
  `python benchmarks/scorecard.py v1=benchmarks/v1/results.json X=benchmarks/v2/X/results.json --out=docs/X_SCORECARD.md --json=benchmarks/v2/X/scorecard.json --fig=docs/figures/X_scorecard.png`
  (always pass --out/--json/--fig so the v1 scorecard is not overwritten).
- Playable demo: `demo/nullscape-demo.html` (build: `python demo/build_demo.py`). Blog: `docs/blog/nullscape-v1.html`.
- Next planned: v2 exp2 = relative-height parameterization (fixes plains grain), started from exp1;
  robust roughness condition; faster sampling. Game direction: Tamashika-like low-res neon FPS.
