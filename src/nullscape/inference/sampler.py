"""Inference: heightmaps from seeds and optional terrain properties.

A checkpoint is self-contained: it stores the network configs, weights (raw and
EMA; release checkpoints keep EMA only), the world spec, conditioning keys and
their normalization statistics, and the archetype names, so sampling needs no
dataset on disk.

Reproducibility: sample i of a call depends only on (seed, i, conditions,
guidance, steps, eta, weights), not on batch size. GPU kernels are not bitwise
deterministic, so repeated runs agree to within floating-point noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nullscape.models import heightparam
from nullscape.models.diffusion import DiffusionConfig, GaussianDiffusion, per_sample_noise
from nullscape.models.unet import UNet, UNetConfig
from nullscape.utils.checkpoint import load_checkpoint
from nullscape.utils.paths import resolve_checkpoint
from nullscape.utils.seed import derive_seed
from nullscape.world import WorldSpec


DEFAULT_SAMPLING = {"steps": 50, "guidance": 2.0, "eta": 0.0, "spacing": "quadratic",
                    "guidance_interval": (0.0, 1.0)}


def sampling_config(defaults: Mapping[str, Any] | None = None, **overrides) -> dict[str, Any]:
    cfg = {**DEFAULT_SAMPLING, **(defaults or {}), **{k: v for k, v in overrides.items() if v is not None}}
    if set(cfg) != set(DEFAULT_SAMPLING):
        raise ValueError(f"sampling keys must be {tuple(DEFAULT_SAMPLING)}")
    if not isinstance(cfg["steps"], (int, np.integer)) or cfg["steps"] < 2:
        raise ValueError("steps must be an integer >= 2")
    if not np.isfinite(cfg["guidance"]) or cfg["guidance"] < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if not np.isfinite(cfg["eta"]) or not 0 <= cfg["eta"] <= 1:
        raise ValueError("eta must be between 0 and 1")
    if cfg["spacing"] not in ("uniform", "quadratic"):
        raise ValueError("spacing must be uniform or quadratic")
    interval = cfg["guidance_interval"]
    if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
        raise ValueError("guidance_interval must satisfy 0 <= min <= max <= 1")
    cfg["guidance_interval"] = tuple(interval)
    return cfg


class TerrainSampler:
    def __init__(self, model: UNet, diffusion: GaussianDiffusion, dataset_meta: Mapping[str, Any],
                 device: torch.device | str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.diffusion = diffusion.to(self.device)
        self.meta = dict(dataset_meta)
        self.world = WorldSpec.from_dict(self.meta["world"])
        self.condition_keys: list[str] = list(self.meta["condition_keys"])
        self.archetypes: list[str] = list(self.meta["archetypes"])
        self.cond_mean = np.asarray(self.meta["condition_stats"]["mean"], dtype=np.float32)
        self.cond_std = np.asarray(self.meta["condition_stats"]["std"], dtype=np.float32)
        self.height_param = dict(self.meta.get("height_param") or heightparam.spec("absolute"))
        bank = self.meta.get("prior_bank")
        self._bank = None if bank is None else (np.asarray(bank["conditions"], dtype=np.float32),
                                                np.asarray(bank["labels"], dtype=np.int64))
        self.sampling_defaults = {**DEFAULT_SAMPLING, **self.meta.get("sampling", {})}
        self.resolve_sampling()

    def resolve_sampling(self, **overrides) -> dict[str, Any]:
        return sampling_config(self.sampling_defaults, **overrides)

    def _fill_placement(self, raw: np.ndarray, kn: np.ndarray, lab: np.ndarray,
                        seeds: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Relative heights need a mean elevation and relief for every sample. Where the caller left them
        open, sample the nearest 16 bank entries including ties (same archetype if known).
        With no known properties, sample the whole eligible bank instead of a fixed subset."""
        jm = self.condition_keys.index(self.height_param["mean_key"])
        jr = self.condition_keys.index(self.height_param["relief_key"])
        need = ~(kn[:, jm] & kn[:, jr])
        if not need.any():
            return raw, kn
        if self._bank is None:
            raise ValueError("relative-height checkpoint without prior_bank: give mean_elevation and relief")
        bc, bl = self._bank
        bz = (bc - self.cond_mean) / self.cond_std
        raw, kn = raw.copy(), kn.copy()
        for i in np.flatnonzero(need):
            cand = np.flatnonzero(bl == lab[i]) if lab[i] < len(self.archetypes) else np.arange(len(bl))
            if not len(cand):
                raise ValueError("prior_bank has no entries for the requested archetype")
            if kn[i].any():
                z = (raw[i, kn[i]] - self.cond_mean[kn[i]]) / self.cond_std[kn[i]]
                d = ((bz[cand][:, kn[i]] - z) ** 2).sum(1)
                cutoff = np.partition(d, min(15, len(d) - 1))[min(15, len(d) - 1)]
                cand = cand[d <= cutoff]
            pick = np.random.default_rng(seeds[i]).choice(cand)
            for j in (jm, jr):
                if not kn[i, j]:
                    raw[i, j] = bc[pick, j]
                    kn[i, j] = True
        return raw, kn

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | str | None = None,
                        use_ema: bool = True) -> "TerrainSampler":
        path = resolve_checkpoint(path)
        ck = load_checkpoint(path)
        if not use_ema and "model" not in ck:
            raise ValueError("this release checkpoint contains EMA weights only; drop --raw-weights")
        ucfg = UNetConfig.from_dict(ck["unet_config"])
        model = UNet(ucfg)
        model.load_state_dict(ck["ema"]["shadow"] if use_ema else ck["model"])
        diffusion = GaussianDiffusion(DiffusionConfig.from_dict(ck["diffusion_config"]), ucfg.num_classes)
        meta = dict(ck["dataset"])
        if "sampling" in ck:
            meta["sampling"] = ck["sampling"]
        sampler = cls(model, diffusion, meta, device)
        sampler.checkpoint_info = {"path": str(path), "step": ck["step"], "use_ema": use_ema}
        if "release" in ck:
            sampler.checkpoint_info["release"] = f"{ck['release']['name']} {ck['release']['version']}"
        return sampler

    def _label_ids(self, n: int, archetype: str | int | None, labels: Sequence[int] | None) -> np.ndarray:
        unknown = len(self.archetypes)
        if labels is not None:
            out = np.asarray(labels, dtype=np.int64).copy()
            if out.shape != (n,):
                raise ValueError(f"labels must have shape ({n},)")
            out[out < 0] = unknown
            if (out > unknown).any():
                raise ValueError("label id out of range")
            return out
        if archetype is None:
            return np.full(n, unknown, dtype=np.int64)
        if isinstance(archetype, str) and archetype not in self.archetypes:
            raise ValueError(f"unknown archetype {archetype!r}; choose from {self.archetypes}")
        idx = self.archetypes.index(archetype) if isinstance(archetype, str) else int(archetype)
        if not 0 <= idx < unknown:
            raise ValueError(f"unknown archetype {archetype!r}; choose from {self.archetypes}")
        return np.full(n, idx, dtype=np.int64)

    def _raw_conditions(self, n: int, properties: Mapping[str, float] | None, cond_raw: np.ndarray | None,
                        known: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        k = len(self.condition_keys)
        if cond_raw is not None:
            raw = np.asarray(cond_raw, dtype=np.float32).reshape(n, k)
            kn = np.ones((n, k), dtype=bool) if known is None else np.asarray(known, dtype=bool).reshape(n, k)
        else:
            raw = np.tile(self.cond_mean, (n, 1))
            kn = np.zeros((n, k), dtype=bool)
            for key, value in (properties or {}).items():
                if key not in self.condition_keys:
                    raise KeyError(f"unknown property {key!r}; choose from {self.condition_keys}")
                j = self.condition_keys.index(key)
                raw[:, j] = value
                kn[:, j] = True
        return raw, kn

    @torch.no_grad()
    def sample(
        self,
        n: int = 1,
        seed: int = 0,
        properties: Mapping[str, float] | None = None,
        archetype: str | int | None = None,
        *,
        cond_raw: np.ndarray | None = None,
        known: np.ndarray | None = None,
        labels: Sequence[int] | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        eta: float | None = None,
        batch_size: int = 64,
        seeds: Sequence[int] | None = None,
        spacing: str | None = None,
        guidance_interval: tuple[float, float] | None = None,
        reference: np.ndarray | None = None,
        preserve_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return float32 heightmaps [n, R, R] in [0, 1].

        ``properties`` maps condition keys to raw values (e.g. {"relief": 0.3}) for all samples;
        ``cond_raw``/``known`` give per-sample values and masks instead. Unspecified properties
        and ``archetype=None`` are left to the model (sampled from the learned distribution).
        """
        skw = self.resolve_sampling(guidance=guidance, steps=steps, eta=eta, spacing=spacing,
                                    guidance_interval=guidance_interval)
        if n < 1 or batch_size < 1:
            raise ValueError("n and batch_size must be positive")
        raw, kn = self._raw_conditions(n, properties, cond_raw, known)
        if not np.isfinite(raw[kn]).all():
            raise ValueError("specified properties must be finite")
        lab = self._label_ids(n, archetype, labels)
        seeds = list(seeds) if seeds is not None else [derive_seed(seed, i) for i in range(n)]
        if len(seeds) != n:
            raise ValueError("seeds must contain exactly n entries")
        kind = self.height_param["kind"]
        if kind == "relative":
            raw, kn = self._fill_placement(raw, kn, lab, seeds)
        z = np.where(kn, (raw - self.cond_mean) / self.cond_std, 0.0).astype(np.float32)
        placement = None
        if kind == "relative":
            jm = self.condition_keys.index(self.height_param["mean_key"])
            jr = self.condition_keys.index(self.height_param["relief_key"])
            placement = (raw[:, jm, None, None], raw[:, jr, None, None])
        r = self.world.resolution
        if (reference is None) != (preserve_mask is None):
            raise ValueError("reference and preserve_mask must be provided together")
        ref_x = None
        if reference is not None:
            reference, preserve_mask = np.asarray(reference, dtype=np.float32), np.asarray(preserve_mask)
            if any(a.shape not in ((r, r), (n, r, r)) for a in (reference, preserve_mask)):
                raise ValueError("reference and preserve_mask must have shape [R,R] or [n,R,R]")
            if not np.isfinite(reference).all() or ((reference < 0) | (reference > 1)).any():
                raise ValueError("reference heights must be finite and in [0,1]")
            if not np.isin(preserve_mask, [0, 1]).all():
                raise ValueError("preserve_mask must contain only booleans or 0/1")
            reference = np.broadcast_to(reference, (n, r, r))
            preserve_mask = np.broadcast_to(preserve_mask.astype(bool), (n, r, r))
            m, rel = placement if placement is not None else (None, None)
            ref_x = heightparam.encode(reference, m, rel, kind).astype(np.float32)
        use_bf16 = self.device.type == "cuda"
        was_training = self.model.training
        self.model.eval()
        out = []
        for s in range(0, n, batch_size):
            sl = slice(s, min(n, s + batch_size))
            x = self.diffusion.ddim_sample(
                self.model,
                torch.from_numpy(z[sl]).to(self.device),
                torch.from_numpy(kn[sl]).to(self.device),
                torch.from_numpy(lab[sl]).to(self.device),
                per_sample_noise(seeds[sl], (1, r, r)),
                **skw,
                clip_range=heightparam.X0_RANGE[kind],
                autocast_dtype=torch.bfloat16 if use_bf16 else None,
                reference=torch.from_numpy(ref_x[sl].copy())[:, None] if ref_x is not None else None,
                preserve_mask=torch.from_numpy(preserve_mask[sl].copy())[:, None] if ref_x is not None else None,
            )
            x = x[:, 0].float().cpu().numpy()
            m, rel = (placement[0][sl], placement[1][sl]) if placement else (None, None)
            decoded = np.clip(heightparam.decode(x, m, rel, kind), 0.0, 1.0)
            out.append(np.where(preserve_mask[sl], reference[sl], decoded) if ref_x is not None else decoded)
        self.model.train(was_training)
        return np.concatenate(out).astype(np.float32)
