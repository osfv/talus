"""Does sampling precision (bf16 vs fp32) or step count change small-scale artifacts?

usage: python scripts/precision_check.py <checkpoint> [--n 32] [--device cpu]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nullscape.data.storage import TerrainStore
from nullscape.inference.sampler import TerrainSampler
from nullscape.metrics.quality import compute_metrics
from nullscape.models.diffusion import per_sample_noise
from nullscape.utils.seed import derive_seed

KEYS = ("peak_density", "sink_density", "hf_energy", "curvature_std", "spectral_beta")


def summarize(name: str, maps: np.ndarray, world) -> None:
    ms = [compute_metrics(h, world) for h in maps]
    print(f"{name:28s} " + " ".join(f"{k}={np.mean([m[k] for m in ms]):.4f}" for k in KEYS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--steps", default="50,100")
    ap.add_argument("--dtypes", default="fp32,bf16", help="bf16 autocast on CPU is very slow without AMX")
    ap.add_argument("--spacing", default="uniform")
    args = ap.parse_args()
    torch.set_num_threads(4)
    s = TerrainSampler.from_checkpoint(args.checkpoint, device=args.device)
    store = TerrainStore.open(s.meta["root"])
    idx = store.split("val")[: args.n]
    summarize("procedural (val)", store.heights(idx), s.world)
    z, kn = s._conditions(args.n, None, store.conditions[idx], None)
    lab = store.labels[idx]
    r = s.world.resolution
    dev = torch.device(args.device)
    dtypes = {"fp32": None, "bf16": torch.bfloat16}
    for steps in (int(x) for x in args.steps.split(",")):
        for dtype in (dtypes[d] for d in args.dtypes.split(",")):
            x = s.diffusion.ddim_sample(
                s.model, torch.from_numpy(z).to(dev), torch.from_numpy(kn).to(dev), torch.from_numpy(lab).to(dev),
                per_sample_noise([derive_seed(0, i) for i in range(args.n)], (1, r, r)),
                steps=steps, guidance=args.guidance, autocast_dtype=dtype, spacing=args.spacing)
            maps = ((x[:, 0].float().clamp(-1, 1) + 1) / 2).cpu().numpy()
            summarize(f"learned {args.spacing} steps={steps} {'bf16' if dtype else 'fp32'}", maps, s.world)


if __name__ == "__main__":
    main()
