"""Inference: heightmaps from seeds and optional terrain properties.

A checkpoint is self-contained: it stores the network configs, weights (raw and
EMA), the world spec, conditioning keys and their normalization statistics, and
the archetype names, so sampling needs no dataset on disk.

Reproducibility: sample i of a call depends only on (seed, i, conditions,
guidance, steps, eta, weights), not on batch size. GPU kernels are not bitwise
deterministic, so repeated runs agree to within floating-point noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nullscape.models.diffusion import DiffusionConfig, GaussianDiffusion, per_sample_noise
from nullscape.models.unet import UNet, UNetConfig
from nullscape.utils.seed import derive_seed
from nullscape.world import WorldSpec


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

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | str | None = None,
                        use_ema: bool = True) -> "TerrainSampler":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ucfg = UNetConfig.from_dict(ck["unet_config"])
        model = UNet(ucfg)
        model.load_state_dict(ck["ema"]["shadow"] if use_ema else ck["model"])
        diffusion = GaussianDiffusion(DiffusionConfig.from_dict(ck["diffusion_config"]), ucfg.num_classes)
        sampler = cls(model, diffusion, ck["dataset"], device)
        sampler.checkpoint_info = {"path": str(path), "step": ck["step"], "use_ema": use_ema}
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
        idx = self.archetypes.index(archetype) if isinstance(archetype, str) else int(archetype)
        if not 0 <= idx < unknown:
            raise ValueError(f"unknown archetype {archetype!r}; choose from {self.archetypes}")
        return np.full(n, idx, dtype=np.int64)

    def _conditions(self, n: int, properties: Mapping[str, float] | None, cond_raw: np.ndarray | None,
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
        z = np.where(kn, (raw - self.cond_mean) / self.cond_std, 0.0).astype(np.float32)
        return z, kn

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
        guidance: float = 1.5,
        steps: int = 50,
        eta: float = 0.0,
        batch_size: int = 256,
        seeds: Sequence[int] | None = None,
        spacing: str = "uniform",
    ) -> np.ndarray:
        """Return float32 heightmaps [n, R, R] in [0, 1].

        ``properties`` maps condition keys to raw values (e.g. {"relief": 0.3}) for all samples;
        ``cond_raw``/``known`` give per-sample values and masks instead. Unspecified properties
        and ``archetype=None`` are left to the model (sampled from the learned distribution).
        """
        z, kn = self._conditions(n, properties, cond_raw, known)
        lab = self._label_ids(n, archetype, labels)
        seeds = list(seeds) if seeds is not None else [derive_seed(seed, i) for i in range(n)]
        r = self.world.resolution
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
                steps=steps, guidance=guidance, eta=eta, spacing=spacing,
                autocast_dtype=torch.bfloat16 if use_bf16 else None,
            )
            out.append(((x[:, 0].float().clamp(-1, 1) + 1) / 2).cpu().numpy())
        self.model.train(was_training)
        return np.concatenate(out).astype(np.float32)
