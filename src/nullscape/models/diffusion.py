"""Gaussian diffusion with a cosine schedule, v-prediction, Min-SNR loss weighting,
classifier-free guidance and DDIM sampling.

References: Ho et al. 2020 (DDPM), Nichol & Dhariwal 2021 (cosine schedule),
Salimans & Ho 2022 (v-prediction), Song et al. 2021 (DDIM), Ho & Salimans 2022
(classifier-free guidance), Hang et al. 2023 (Min-SNR weighting).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    min_snr_gamma: float | None = 5.0
    p_drop_property: float = 0.15
    p_drop_label: float = 0.15
    p_drop_all: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiffusionConfig":
        return cls(**d)


def cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(cfg.timesteps), persistent=False)

    def _ab(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas_cumprod[t].view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self._ab(t)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def drop_conditions(self, known: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-time condition dropout that teaches the model partial and null conditioning."""
        b = known.shape[0]
        c = self.cfg
        keep = torch.rand_like(known, dtype=torch.float32) >= c.p_drop_property
        drop_all = torch.rand(b, device=known.device) < c.p_drop_all
        known = known & keep & ~drop_all[:, None]
        drop_label = (torch.rand(b, device=label.device) < c.p_drop_label) | drop_all
        label = torch.where(drop_label, torch.full_like(label, self.num_classes), label)
        return known, label

    def loss(self, model: nn.Module, x0: torch.Tensor, cond: torch.Tensor, label: torch.Tensor,
             known: torch.Tensor | None = None, t: torch.Tensor | None = None, noise: torch.Tensor | None = None,
             drop: bool = True) -> torch.Tensor:
        """Weighted v-prediction MSE. Pass fixed ``t``/``noise`` and ``drop=False`` for a low-variance validation loss."""
        b = x0.shape[0]
        known = torch.ones_like(cond, dtype=torch.bool) if known is None else known
        if drop:
            known, label = self.drop_conditions(known, label)
        t = torch.randint(0, self.cfg.timesteps, (b,), device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        ab = self._ab(t)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        v_target = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        v_pred = model(x_t, t, cond, known, label)
        mse = (v_pred.float() - v_target).pow(2).mean(dim=(1, 2, 3))
        if self.cfg.min_snr_gamma is not None:
            snr = (ab / (1 - ab)).view(-1)
            mse = mse * snr.clamp(max=self.cfg.min_snr_gamma) / (snr + 1)
        return mse.mean()

    def _guided_v(self, model, x, t, cond, known, label, guidance: float) -> torch.Tensor:
        if guidance == 1.0:
            return model(x, t, cond, known, label).float()
        null_known = torch.zeros_like(known)
        null_label = torch.full_like(label, self.num_classes)
        v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cond, cond]),
                  torch.cat([known, null_known]), torch.cat([label, null_label])).float()
        v_c, v_u = v.chunk(2)
        return v_u + guidance * (v_c - v_u)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        cond: torch.Tensor,
        known: torch.Tensor,
        label: torch.Tensor,
        noise_fn: Callable[[int], torch.Tensor],
        steps: int = 50,
        guidance: float = 1.0,
        eta: float = 0.0,
        clip_x0: bool = True,
        clip_range: float = 1.0,
        autocast_dtype: torch.dtype | None = None,
        spacing: str = "uniform",
    ) -> torch.Tensor:
        """DDIM sampling. ``noise_fn(k)`` returns the k-th standard-normal tensor [B, C, H, W]
        (k = 0 is the initial noise; k >= 1 are per-step noises used when eta > 0)."""
        device = cond.device
        ts = timestep_schedule(self.cfg.timesteps, steps, spacing).to(device)
        x = noise_fn(0).to(device)
        for i, t_cur in enumerate(ts):
            t = t_cur.repeat(x.shape[0])
            ab = self.alphas_cumprod[t_cur]
            ab_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                v = self._guided_v(model, x, t, cond, known, label, guidance)
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            if clip_x0:
                x0 = x0.clamp(-clip_range, clip_range)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * noise_fn(i + 1).to(device)
        return x


SPACINGS = ("uniform", "quadratic")


def timestep_schedule(timesteps: int, steps: int, spacing: str = "uniform") -> torch.Tensor:
    """Descending, de-duplicated sampling timesteps ending at 0.

    "quadratic" (Song et al. 2021) places more steps at low noise, where the fine
    detail of smooth terrain is resolved; with uniform spacing the final DDIM jump
    starts from a noise level larger than that detail.
    """
    if spacing == "uniform":
        ts = torch.linspace(timesteps - 1, 0, steps, dtype=torch.float64)
    elif spacing == "quadratic":
        ts = torch.linspace(math.sqrt(timesteps - 1), 0, steps, dtype=torch.float64) ** 2
    else:
        raise ValueError(f"spacing must be one of {SPACINGS}, got {spacing!r}")
    return torch.unique_consecutive(ts.round().long())


def per_sample_noise(seeds: Sequence[int], shape: tuple[int, ...]) -> Callable[[int], torch.Tensor]:
    """Noise source where sample j depends only on seeds[j] and the step index,
    so a sample is reproducible regardless of batch size or batch position."""

    def fn(k: int) -> torch.Tensor:
        out = []
        for s in seeds:
            g = torch.Generator().manual_seed((int(s) * 1_000_003 + k) & ((1 << 63) - 1))
            out.append(torch.randn(shape, generator=g))
        return torch.stack(out)

    return fn
