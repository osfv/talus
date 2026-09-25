"""Measure training-step throughput and peak VRAM for a model config (synthetic data).

usage: python scripts/bench_train.py [--batch 64] [--base 64] [--mults 1,2,3,4] [--attn 16,8] [--steps 30]
"""

from __future__ import annotations

import argparse
import time

import torch

from nullscape.models.diffusion import DiffusionConfig, GaussianDiffusion
from nullscape.models.unet import UNet, UNetConfig, count_parameters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--mults", default="1,2,3,4")
    ap.add_argument("--attn", default="16,8")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    cfg = UNetConfig(image_size=args.res, base_channels=args.base, channel_mults=tuple(int(m) for m in args.mults.split(",")),
                     attention_resolutions=tuple(int(a) for a in args.attn.split(",") if a), num_res_blocks=args.blocks)
    model = UNet(cfg).to(dev)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    diff = GaussianDiffusion(DiffusionConfig(), cfg.num_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(args.batch, 1, args.res, args.res, device=dev)
    c = torch.randn(args.batch, cfg.num_conditions, device=dev)
    y = torch.randint(0, cfg.num_classes, (args.batch,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_bf16):
            loss = diff.loss(model, x, c, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    print(f"params {count_parameters(model) / 1e6:.2f}M batch {args.batch}: {1 / dt:.2f} it/s "
          f"({dt * 1000:.0f} ms/step), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
