"""Conditional U-Net denoiser for heightmap diffusion (see docs/MODEL_RESEARCH.md).

Conditioning = diffusion time + per-property terrain conditions + archetype label.
Every property has a learned "unknown" embedding, so any subset of properties
can be specified at inference; classifier-free guidance uses the all-unknown
embedding as the unconditional branch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    image_size: int = 64
    in_channels: int = 1
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    num_conditions: int = 5
    num_classes: int = 6
    groups: int = 32
    head_channels: int = 64
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UNetConfig":
        d = dict(d)
        for k in ("channel_mults", "attention_resolutions"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous or integer timesteps, [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)  # scale-shift (FiLM) conditioning
        self.norm2 = nn.GroupNorm(groups, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, channels: int, head_channels: int, groups: int):
        super().__init__()
        self.heads = max(1, channels // head_channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampling + conv (no transposed conv -> no checkerboard artifacts)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ConditionEmbedding(nn.Module):
    """Embeds z-scored properties with per-property 'unknown' tokens, plus a label with an 'unknown' class."""

    def __init__(self, num_conditions: int, num_classes: int, dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.value = nn.Parameter(torch.randn(num_conditions, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_conditions, dim))
        self.unknown = nn.Parameter(torch.randn(num_conditions, dim) * 0.02)
        self.fourier = nn.Linear(16, dim)  # nonlinear features of each scalar
        self.register_buffer("freqs", torch.randn(16))
        self.label = nn.Embedding(num_classes + 1, dim)  # index num_classes = unknown
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, cond: torch.Tensor, known: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cond [B, K] z-scores, known [B, K] bool, label [B] long in [0, C] (C = unknown)
        x = cond.float()[..., None]  # [B, K, 1]
        feats = torch.sin(x * self.freqs + (torch.pi / 4))  # [B, K, 16]
        known_emb = x * self.value + self.bias + self.fourier(feats)  # [B, K, D]
        e = torch.where(known[..., None], known_emb, self.unknown.expand_as(known_emb)).sum(1)
        return self.mlp(e + self.label(label))


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        emb_dim = 4 * ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cond_emb = ConditionEmbedding(cfg.num_conditions, cfg.num_classes, emb_dim)
        self.inp = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        self.down = nn.ModuleList()
        skips = [ch]
        cur, res = ch, cfg.image_size
        for level, mult in enumerate(cfg.channel_mults):
            for _ in range(cfg.num_res_blocks):
                out = ch * mult
                blocks = nn.ModuleList([ResBlock(cur, out, emb_dim, cfg.dropout, cfg.groups)])
                if res in cfg.attention_resolutions:
                    blocks.append(Attention(out, cfg.head_channels, cfg.groups))
                self.down.append(blocks)
                cur = out
                skips.append(cur)
            if level != len(cfg.channel_mults) - 1:
                self.down.append(nn.ModuleList([Downsample(cur)]))
                skips.append(cur)
                res //= 2

        self.mid = nn.ModuleList([
            ResBlock(cur, cur, emb_dim, cfg.dropout, cfg.groups),
            Attention(cur, cfg.head_channels, cfg.groups),
            ResBlock(cur, cur, emb_dim, cfg.dropout, cfg.groups),
        ])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.channel_mults))):
            for i in range(cfg.num_res_blocks + 1):
                out = ch * mult
                blocks = nn.ModuleList([ResBlock(cur + skips.pop(), out, emb_dim, cfg.dropout, cfg.groups)])
                if res in cfg.attention_resolutions:
                    blocks.append(Attention(out, cfg.head_channels, cfg.groups))
                if level != 0 and i == cfg.num_res_blocks:
                    blocks.append(Upsample(out))
                    res *= 2
                self.up.append(blocks)
                cur = out

        self.out = nn.Sequential(nn.GroupNorm(cfg.groups, cur), nn.SiLU(), nn.Conv2d(cur, cfg.in_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, known: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.cfg.base_channels)) + self.cond_emb(cond, known, label)
        h = self.inp(x)
        hs = [h]
        for blocks in self.down:
            for layer in blocks:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in blocks:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
