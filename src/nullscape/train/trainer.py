"""Training loop for the conditional heightmap diffusion model."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nullscape.data.storage import TerrainStore
from nullscape.eval.artifacts import ARTIFACTS_ROOT, ArtifactSpec, generate_checkpoint_artifacts
from nullscape.eval.core import HEADLINE_KEYS, evaluate_generated, split_halves
from nullscape.inference.sampler import TerrainSampler
from nullscape.models.diffusion import DiffusionConfig, GaussianDiffusion
from nullscape.models.ema import EMA
from nullscape.models.unet import UNet, UNetConfig, count_parameters
from nullscape.utils.seed import seed_everything
from nullscape.utils.tracking import RunDir, environment_info
from nullscape.viz.render import save_grid, to_uint8_image

DEFAULT_TRAIN: dict[str, Any] = {
    "batch_size": 64, "lr": 2e-4, "weight_decay": 0.0, "betas": [0.9, 0.999], "warmup_steps": 1000,
    "max_steps": 60000, "grad_clip": 1.0, "ema_decay": 0.9995, "bf16": True, "log_every": 100,
    "val_every": 2000, "val_size": 1024, "sample_every": 5000, "eval_every": 10000, "eval_samples": 256,
    "eval_steps": 50, "eval_guidance": 1.5, "checkpoint_every": 5000, "data_on_gpu": True,
    "sample_batch_size": 64,
    # Cap PyTorch's share of VRAM so the caching allocator frees cached blocks instead of
    # growing past physical memory (on Windows/WDDM that spills to system RAM, ~10x slower).
    # 0.72 of 8 GB leaves room for the desktop/browser; 0.8 spilled once other apps grew.
    "cuda_memory_fraction": 0.72,
    "keep_checkpoints": True,     # also save checkpoints/step_XXXXXXX.pt at every checkpoint_every
    "artifacts_every": 0,         # >0: render eval/artifacts.py suite at these steps
    "artifacts_dir": None,        # default artifacts/<run dir name>; set to continue a lineage when resuming
    "artifacts_spec": {},         # ArtifactSpec overrides (must stay fixed within a lineage)
    "lr_schedule": "constant",    # "constant" or "cosine" (decays to min_lr_ratio * lr after warmup)
    "min_lr_ratio": 0.1,
    "init_from": None,            # checkpoint to start from (weights + EMA only; fresh optimizer, step 0)
}


def lr_lambda(tcfg: dict[str, Any]):
    warm, total, floor = max(1, tcfg["warmup_steps"]), max(1, tcfg["max_steps"]), tcfg["min_lr_ratio"]

    def f(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        if tcfg["lr_schedule"] == "cosine":
            p = min(1.0, (s - warm) / max(1, total - warm))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
        return 1.0

    return f


class TensorBatcher:
    """Random minibatches from an in-memory uint16 split with random dihedral augmentation."""

    def __init__(self, heights_u16: np.ndarray, cond: np.ndarray, labels: np.ndarray, device: torch.device,
                 on_device: bool, seed: int):
        store_dev = device if on_device else torch.device("cpu")
        # exact int16 storage (u16 - 32768): half the VRAM of int32, which matters on 8 GB cards where
        # exceeding physical VRAM silently spills to system memory and slows training ~15x on Windows
        self.h = torch.from_numpy((heights_u16.astype(np.int32) - 32768).astype(np.int16)).to(store_dev)
        self.c = torch.from_numpy(cond.astype(np.float32)).to(store_dev)
        self.y = torch.from_numpy(labels.astype(np.int64)).to(store_dev)
        self.device = device
        self.g = torch.Generator(device=store_dev).manual_seed(seed)

    def __len__(self) -> int:
        return self.h.shape[0]

    def sample(self, batch_size: int, augment: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self), (batch_size,), device=self.h.device, generator=self.g)
        x = self.h[idx].float().add_(32768.0).div_(65535.0).mul_(2).sub_(1).unsqueeze(1)
        if augment:
            k = torch.randint(0, 8, (batch_size,), device=self.h.device, generator=self.g)
            out = torch.empty_like(x)
            for j in range(8):
                m = k == j
                if m.any():
                    xj = torch.flip(x[m], dims=[-1]) if j >= 4 else x[m]
                    out[m] = torch.rot90(xj, j % 4, dims=[-2, -1])
            x = out
        return x.to(self.device, non_blocking=True), self.c[idx].to(self.device), self.y[idx].to(self.device)


def build_model(train_cfg: dict[str, Any], store: TerrainStore) -> tuple[UNet, GaussianDiffusion]:
    ucfg = UNetConfig.from_dict({**train_cfg.get("model", {}), "image_size": store.world.resolution,
                                 "num_conditions": len(store.condition_keys), "num_classes": len(store.archetypes)})
    dcfg = DiffusionConfig.from_dict(train_cfg.get("diffusion", {}))
    return UNet(ucfg), GaussianDiffusion(dcfg, ucfg.num_classes)


def dataset_meta(store: TerrainStore) -> dict[str, Any]:
    m = store.manifest
    return {"name": m["name"], "root": str(store.root), "world": m["world"], "condition_keys": m["condition_keys"],
            "condition_stats": m["condition_stats"], "archetypes": m["archetypes"],
            "generator_version": m["generator_version"], "config_sha256": m["config_sha256"]}


def save_checkpoint(path: Path, model, ema, opt, step: int, cfg: dict, store_meta: dict, extra: dict | None = None):
    torch.save({
        "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(),
        "unet_config": model.cfg.to_dict(), "diffusion_config": cfg["_diffusion_config"], "train_config": cfg,
        "dataset": store_meta, "env": environment_info(), **(extra or {}),
    }, path)


def train(cfg: dict[str, Any], resume: str | None = None) -> Path:
    tcfg = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda" and tcfg["cuda_memory_fraction"]:
        torch.cuda.set_per_process_memory_fraction(float(tcfg["cuda_memory_fraction"]))

    store = TerrainStore.open(cfg["dataset"])
    model, diffusion = build_model(cfg, store)
    model.to(device)
    diffusion.to(device)
    cfg = {**cfg, "train": tcfg, "_diffusion_config": diffusion.cfg.to_dict()}
    ema = EMA(model, decay=tcfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=tuple(tcfg["betas"]),
                            weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg))
    step = 0
    if tcfg["init_from"] and not resume:
        ck = torch.load(tcfg["init_from"], map_location="cpu", weights_only=False)
        if ck["unet_config"] != model.cfg.to_dict():
            raise ValueError("init_from checkpoint has a different UNet config")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        print(f"initialized from {tcfg['init_from']} (step {ck['step']})")
        del ck
    if resume:
        # load on CPU and drop the dict afterwards: a device-mapped checkpoint would pin ~370 MB of
        # duplicate tensors on the GPU for the whole run
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])  # moves optimizer state to the params' device
        step = ck["step"]
        sched.last_epoch = step
        del ck

    run = RunDir(cfg.get("name", "run"), cfg)
    meta = dataset_meta(store)
    n_params = count_parameters(model)
    (run.path / "model.json").write_text(json.dumps({"parameters": n_params, "unet": model.cfg.to_dict(),
                                                     "diffusion": diffusion.cfg.to_dict()}, indent=2))
    print(f"run dir: {run.path}\nparameters: {n_params / 1e6:.2f}M on {device}")

    zc = lambda c: (c - store.condition_mean) / store.condition_std  # noqa: E731
    train_idx = store.split("train")
    batcher = TensorBatcher(np.asarray(store.heights_u16[train_idx]), zc(store.conditions[train_idx]),
                            store.labels[train_idx], device, tcfg["data_on_gpu"] and device.type == "cuda", seed)

    # fixed validation batch (fixed t and noise) for a low-variance loss curve
    val_idx = store.split("val")
    vi = val_idx[: tcfg["val_size"]]
    vgen = torch.Generator().manual_seed(seed + 1)
    val_x = torch.from_numpy(store.heights(vi) * 2 - 1)[:, None].to(device)
    val_c = torch.from_numpy(zc(store.conditions[vi]).astype(np.float32)).to(device)
    val_y = torch.from_numpy(store.labels[vi]).to(device)
    val_t = torch.randint(0, diffusion.cfg.timesteps, (len(vi),), generator=vgen).to(device)
    val_n = torch.randn(val_x.shape, generator=vgen).to(device)

    # in-training evaluation set: val halves A (conditions) / B (reference)
    ea, eb = split_halves(len(val_idx), seed=seed)
    ne = min(tcfg["eval_samples"], len(ea))
    ea, eb = val_idx[ea[:ne]], val_idx[eb[:ne]]
    eval_a, eval_b = store.heights(ea), store.heights(eb)

    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=tcfg["bf16"] and device.type == "cuda")
    artifacts_root = Path(tcfg["artifacts_dir"]) if tcfg["artifacts_dir"] else ARTIFACTS_ROOT / run.path.name
    train_bank = None
    best = math.inf
    t0 = time.time()
    losses: list[torch.Tensor] = []
    model.train()
    while step < tcfg["max_steps"]:
        x, c, y = batcher.sample(tcfg["batch_size"])
        with torch.autocast(**amp):
            loss = diffusion.loss(model, x, c, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        sched.step()
        ema.update(model)
        step += 1
        losses.append(loss.detach())  # no .item() here: avoids a GPU sync every step

        if step % tcfg["log_every"] == 0:
            dt = time.time() - t0
            mean_loss = torch.stack(losses).mean().item()
            if not math.isfinite(mean_loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            run.log(step, loss=mean_loss, grad_norm=float(gnorm), lr=sched.get_last_lr()[0],
                    it_per_s=tcfg["log_every"] / dt)
            print(f"step {step:7d} loss {mean_loss:.4f} gnorm {gnorm:.3f} {tcfg['log_every'] / dt:.1f} it/s")
            losses, t0 = [], time.time()

        if step % tcfg["val_every"] == 0 or step == tcfg["max_steps"]:
            vl = _val_loss(diffusion, model, ema.shadow, val_x, val_c, val_y, val_t, val_n, amp)
            run.log(step, **vl)
            print(f"step {step:7d} " + " ".join(f"{k} {v:.4f}" for k, v in vl.items()))

        if step % tcfg["sample_every"] == 0 or step == tcfg["max_steps"]:
            sampler = TerrainSampler(ema.shadow, diffusion, meta, device)
            imgs = sampler.sample(n=36, seed=1234, guidance=tcfg["eval_guidance"], steps=tcfg["eval_steps"],
                                  labels=np.arange(36) % len(store.archetypes), batch_size=tcfg["sample_batch_size"])
            path = save_grid(imgs, store.world, run.path / "samples" / f"step_{step:07d}.png",
                             titles=[store.archetypes[i % len(store.archetypes)] for i in range(36)])
            run.log_image("samples/first", to_uint8_image(imgs[0], store.world), step)
            print(f"samples -> {path}")
            model.train()
            _release_cache(device)

        if step % tcfg["eval_every"] == 0 or step == tcfg["max_steps"]:
            sampler = TerrainSampler(ema.shadow, diffusion, meta, device)
            known = np.ones((ne, len(store.condition_keys)), dtype=bool)
            gen = sampler.sample(n=ne, seed=99, cond_raw=store.conditions[ea], known=known, labels=store.labels[ea],
                                 guidance=tcfg["eval_guidance"], steps=tcfg["eval_steps"],
                                 batch_size=tcfg["sample_batch_size"])
            rep = evaluate_generated(gen, eval_a, eval_b, store.world, requested_conds=store.conditions[ea],
                                     cond_known=known, gen_labels=store.labels[ea], ref_a_labels=store.labels[ea],
                                     ref_b_labels=store.labels[eb], archetype_names=store.archetypes)
            (run.path / "eval" / f"step_{step:07d}.json").write_text(json.dumps(rep, indent=2))
            scalars = {f"eval/{k}": rep["model_vs_ref"][k] for k in HEADLINE_KEYS}
            scalars.update({f"eval/ratio_{k}": v for k, v in rep["ratio_to_floor"].items()})
            scalars.update({f"eval/nmae_{k}": v["nmae"] for k, v in rep["condition_adherence"].items()})
            run.log(step, **scalars)
            score = rep["ratio_to_floor"]["metric_w1_mean"]
            print(f"eval step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in rep["ratio_to_floor"].items()))
            if score < best:
                best = score
                save_checkpoint(run.path / "checkpoints" / "best.pt", model, ema, opt, step, cfg, meta,
                                {"best_score": score})
            model.train()
            _release_cache(device)

        if step % tcfg["checkpoint_every"] == 0 or step == tcfg["max_steps"]:
            last = run.path / "checkpoints" / "last.pt"
            save_checkpoint(last, model, ema, opt, step, cfg, meta)
            if tcfg["keep_checkpoints"]:
                shutil.copy2(last, last.with_name(f"step_{step:07d}.pt"))

        if tcfg["artifacts_every"] and (step % tcfg["artifacts_every"] == 0 or step == tcfg["max_steps"]):
            if train_bank is None:
                train_bank = store.heights(train_idx)
            sampler = TerrainSampler(ema.shadow, diffusion, meta, device)
            sampler.checkpoint_info = {"path": str(run.path / "checkpoints" / f"step_{step:07d}.pt"), "step": step,
                                       "use_ema": True}
            d = generate_checkpoint_artifacts(sampler, store, artifacts_root, step,
                                              ArtifactSpec(**tcfg["artifacts_spec"]), train_bank=train_bank)
            print(f"artifacts -> {d}")
            model.train()
            _release_cache(device)

    run.close()
    return run.path


def _release_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def _val_loss(diffusion, model, ema_model, x, c, y, t, n, amp, chunk: int = 256) -> dict[str, float]:
    out = {}
    for name, m in (("val_loss", model), ("val_loss_ema", ema_model)):
        was = m.training
        m.eval()
        tot = 0.0
        for s in range(0, x.shape[0], chunk):
            sl = slice(s, s + chunk)
            with torch.autocast(**amp):
                tot += diffusion.loss(m, x[sl], c[sl], y[sl], t=t[sl], noise=n[sl], drop=False).item() * x[sl].shape[0]
        out[name] = tot / x.shape[0]
        m.train(was)
    return out
