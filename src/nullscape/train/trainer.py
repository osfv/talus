"""Training loop for the conditional heightmap diffusion model."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nullscape.data.storage import TerrainStore
from nullscape.eval.artifacts import ARTIFACTS_ROOT, ArtifactSpec, generate_checkpoint_artifacts
from nullscape.eval.core import HEADLINE_KEYS, evaluate_generated, split_halves
from nullscape.inference.sampler import DEFAULT_SAMPLING, TerrainSampler
from nullscape.models import heightparam
from nullscape.models.diffusion import DiffusionConfig, GaussianDiffusion
from nullscape.models.ema import EMA
from nullscape.models.unet import UNet, UNetConfig, count_parameters
from nullscape.utils.checkpoint import load_checkpoint
from nullscape.utils.paths import resolve_checkpoint
from nullscape.utils.seed import seed_everything
from nullscape.train.recovery import StopRequest, TrainingInterrupted, dataset_fingerprint, restore_rng, state_dict, validate_resume
from nullscape.utils.tracking import RunDir, atomic_file, atomic_json, environment_info
from nullscape.viz.render import save_grid, to_uint8_image

DEFAULT_TRAIN: dict[str, Any] = {
    "batch_size": 64, "lr": 2e-4, "weight_decay": 0.0, "betas": [0.9, 0.999], "warmup_steps": 1000,
    "max_steps": 60000, "grad_clip": 1.0, "ema_decay": 0.9995, "bf16": True, "log_every": 100,
    "val_every": 2000, "val_size": 1024, "sample_every": 5000, "eval_every": 10000, "eval_samples": 256,
    "eval_steps": 50, "eval_guidance": 1.5, "checkpoint_every": 5000, "data_on_gpu": True,
    "sample_batch_size": 32,
    "eval_spacing": "quadratic", "eval_guidance_interval": [0.0, 1.0],
    "archetype_loss_weights": {},
    # Cap PyTorch's share of VRAM so the caching allocator frees cached blocks instead of
    # growing past physical memory (on Windows/WDDM that spills to system RAM, ~10x slower).
    # 0.72 of 8 GB leaves room for the desktop/browser; 0.8 spilled once other apps grew.
    "cuda_memory_fraction": 0.72,
    "keep_checkpoints": True,     # also save checkpoints/step_XXXXXXX.pt at every checkpoint_every
    "recovery_every": 1000,       # refresh checkpoints/last.pt (no numbered copy) this often
    "artifacts_every": 0,         # >0: render eval/artifacts.py suite at these steps
    "artifacts_dir": None,        # default artifacts/<run dir name>; set to continue a lineage when resuming
    "artifacts_spec": {},         # ArtifactSpec overrides (must stay fixed within a lineage)
    "lr_schedule": "constant",    # "constant" or "cosine" (decays to min_lr_ratio * lr after warmup)
    "min_lr_ratio": 0.1,
    "init_from": None,            # checkpoint to start from (weights + EMA only; fresh optimizer, step 0)
    "height_param": "absolute",   # "relative": model sees 2(h - mean)/relief (see models/heightparam.py)
    "prior_bank_per_class": 2000, # training conditions stored per archetype for relative-height placement
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
                 on_device: bool, seed: int, placement: np.ndarray | None = None):
        store_dev = device if on_device else torch.device("cpu")
        # placement: raw (mean elevation, relief) per map for relative heights; None = absolute heights
        self.pl = None if placement is None else torch.from_numpy(placement.astype(np.float32)).to(store_dev)
        # exact int16 storage (u16 - 32768): half the VRAM of int32, which matters on 8 GB cards where
        # exceeding physical VRAM silently spills to system memory and slows training ~15x on Windows
        self.h = torch.from_numpy((heights_u16.astype(np.int32) - 32768).astype(np.int16)).to(store_dev)
        self.c = torch.from_numpy(cond.astype(np.float32)).to(store_dev)
        self.y = torch.from_numpy(labels.astype(np.int64)).to(store_dev)
        self.device = device
        self.g = torch.Generator(device=store_dev).manual_seed(seed)

    def __len__(self) -> int:
        return self.h.shape[0]

    def state_dict(self) -> dict:
        return {"rng": self.g.get_state(), "device": str(self.g.device), "size": len(self)}

    def load_state_dict(self, state: dict) -> None:
        if state["size"] != len(self) or state["device"] != str(self.g.device):
            raise ValueError("batcher size/device differs from the saved training state")
        self.g.set_state(state["rng"].cpu())

    def sample(self, batch_size: int, augment: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self), (batch_size,), device=self.h.device, generator=self.g)
        x = self.h[idx].float().add_(32768.0).div_(65535.0).mul_(2).sub_(1).unsqueeze(1)
        if self.pl is not None:
            p = self.pl[idx].view(-1, 2, 1, 1)
            x = heightparam.encode((x + 1) / 2, p[:, 0:1], p[:, 1:2], "relative")
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
            "generator_version": m["generator_version"], "config_sha256": m["config_sha256"],
            "fingerprint": dataset_fingerprint(store)}


def save_checkpoint(path: Path, model, ema, opt, step: int, cfg: dict, store_meta: dict, extra: dict | None = None):
    payload = {
        "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(),
        "unet_config": model.cfg.to_dict(), "diffusion_config": cfg["_diffusion_config"], "train_config": cfg,
        "dataset": store_meta, "sampling": {**DEFAULT_SAMPLING, **cfg.get("sampling", {})},
        "env": environment_info(), **(extra or {}),
    }
    with atomic_file(path) as file:
        torch.save(payload, file)


def train(cfg: dict[str, Any], resume: str | None = None, *, dry_run: bool = False,
          allow_inexact_resume: bool = False) -> Path | dict[str, Any]:
    options = cfg.get("train", {})
    if not isinstance(options, dict):
        raise ValueError("train settings must be a mapping")
    unknown = set(options) - set(DEFAULT_TRAIN)
    if unknown:
        raise ValueError(f"unknown training options: {', '.join(sorted(unknown))}")
    tcfg = {**DEFAULT_TRAIN, **options}
    for key in ("batch_size", "max_steps", "log_every", "val_every", "sample_every", "eval_every",
                "checkpoint_every", "recovery_every", "val_size", "eval_samples", "sample_batch_size",
                "prior_bank_per_class"):
        if not isinstance(tcfg[key], int) or tcfg[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    if tcfg["lr_schedule"] not in ("constant", "cosine"):
        raise ValueError("train.lr_schedule must be constant or cosine")
    try:
        tcfg["lr"] = float(tcfg["lr"])
    except (TypeError, ValueError):
        raise ValueError("train.lr must be a number") from None
    if not np.isfinite(tcfg["lr"]) or tcfg["lr"] <= 0:
        raise ValueError("train.lr must be finite and positive")
    if tcfg["init_from"] and not resume:
        tcfg["init_from"] = str(resolve_checkpoint(tcfg["init_from"]))
    if resume:
        resume = str(resolve_checkpoint(resume, prefer="last"))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda" and tcfg["cuda_memory_fraction"]:
        torch.cuda.set_per_process_memory_fraction(float(tcfg["cuda_memory_fraction"]))

    store = TerrainStore.open(cfg["dataset"])
    meta = dataset_meta(store)
    meta["height_param"] = heightparam.spec(tcfg["height_param"])
    meta["sampling"] = {**DEFAULT_SAMPLING, **cfg.get("sampling", {})}
    loss_weights = tcfg["archetype_loss_weights"]
    if set(loss_weights) - set(store.archetypes):
        raise ValueError("archetype_loss_weights contains unknown terrain types")
    weights = np.array([loss_weights.get(name, 1.0) for name in store.archetypes], dtype=np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("archetype_loss_weights must be finite and positive")
    class_weights = torch.from_numpy(weights).to(device) if loss_weights else None
    model, diffusion = build_model(cfg, store)
    cfg = {**cfg, "train": tcfg, "_diffusion_config": diffusion.cfg.to_dict()}
    resume_checkpoint = load_checkpoint(resume) if resume else None
    if resume_checkpoint is not None:
        validate_resume(resume_checkpoint, cfg, meta, model.cfg.to_dict(), diffusion.cfg.to_dict(),
                        allow_inexact=allow_inexact_resume)
    model.to(device)
    diffusion.to(device)
    ema = EMA(model, decay=tcfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=tuple(tcfg["betas"]),
                            weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg))
    step = 0
    if tcfg["init_from"] and not resume:
        ck = load_checkpoint(tcfg["init_from"])
        if ck["unet_config"] != model.cfg.to_dict():
            raise ValueError("init_from checkpoint has a different UNet config")
        model.load_state_dict(ck["model"] if "model" in ck else ck["ema"]["shadow"])  # releases carry EMA only
        ema.load_state_dict(ck["ema"])
        print(f"initialized from {tcfg['init_from']} (step {ck['step']})")
        del ck
    resume_state = None
    inherited_best = None
    if resume:
        # load on CPU and drop the dict afterwards: a device-mapped checkpoint would pin ~370 MB of
        # duplicate tensors on the GPU for the whole run
        ck = resume_checkpoint
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])  # moves optimizer state to the params' device
        step = ck["step"]
        resume_state = ck.get("training_state")
        if resume_state is not None:
            sched.load_state_dict(resume_state["scheduler"])
            best_step = resume_state["best_step"]
            if best_step is not None:
                candidates = [Path(resume), Path(resume).parent / f"step_{best_step:07d}.pt",
                              Path(resume).parent / "best.pt"]
                for candidate in dict.fromkeys(candidates):
                    if candidate.is_file():
                        saved = load_checkpoint(candidate)
                        matches = saved["step"] == best_step and saved["dataset"].get("fingerprint") == meta["fingerprint"]
                        del saved
                        if matches:
                            inherited_best = candidate
                            break
                if inherited_best is None:
                    raise ValueError(f"best checkpoint at step {best_step} is missing; keep it beside the resume checkpoint")
        else:
            sched.last_epoch = step
            sched._last_lr = [group["lr"] for group in opt.param_groups]
        del ck, resume_checkpoint

    if dry_run:
        idx = np.resize(store.split("train"), tcfg["batch_size"])
        cond = store.conditions[idx]
        jm, jr = [list(store.condition_keys).index(k) for k in ("mean_elevation", "relief")]
        x = torch.from_numpy(heightparam.encode(store.heights(idx), cond[:, jm, None, None],
                                                cond[:, jr, None, None], tcfg["height_param"]).astype(np.float32))[:, None].to(device)
        c = torch.from_numpy(((cond - store.condition_mean) / store.condition_std).astype(np.float32)).to(device)
        y = torch.as_tensor(store.labels[idx], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=tcfg["bf16"] and device.type == "cuda"):
                loss = diffusion.loss(model, x, c, y, sample_weight=class_weights[y] if class_weights is not None else None)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            if not torch.isfinite(loss) or not torch.isfinite(gnorm):
                raise FloatingPointError("non-finite loss or gradients in training preflight")
            opt.step()
            ema.update(model)
            losses.append(float(loss.detach()))
        report = {"name": cfg.get("name"), "device": str(device), "batch_size": tcfg["batch_size"],
                  "loss": losses[-1], "losses": losses, "grad_norm": float(gnorm), "height_param": tcfg["height_param"],
                  "archetype_loss_weights": loss_weights, "checkpoint_written": False,
                  "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                  "scope": "three forward/backward/optimizer/EMA steps, including allocated optimizer state; no run directory; dataset kept on CPU"}
        return report

    run = RunDir(cfg.get("name", "run"), cfg)
    if inherited_best is not None:
        with inherited_best.open("rb") as source, atomic_file(run.path / "checkpoints" / "best.pt") as target:
            shutil.copyfileobj(source, target)
    kind = tcfg["height_param"]
    meta["height_param"] = heightparam.spec(kind)
    keys = store.condition_keys
    place_cols = [keys.index(meta["height_param"]["mean_key"]), keys.index(meta["height_param"]["relief_key"])]
    if kind == "relative":
        tr = store.split("train")
        rng = np.random.default_rng(seed)
        pick = np.concatenate([rng.permutation(tr[store.labels[tr] == c])[: tcfg["prior_bank_per_class"]]
                               for c in range(len(store.archetypes))])
        meta["prior_bank"] = {"conditions": store.conditions[pick].astype(float).round(6).tolist(),
                              "labels": store.labels[pick].astype(int).tolist()}
    n_params = count_parameters(model)
    (run.path / "model.json").write_text(json.dumps({"parameters": n_params, "unet": model.cfg.to_dict(),
                                                     "diffusion": diffusion.cfg.to_dict()}, indent=2))
    print(f"run dir: {run.path}\nparameters: {n_params / 1e6:.2f}M on {device}")

    zc = lambda c: (c - store.condition_mean) / store.condition_std  # noqa: E731
    train_idx = store.split("train")
    batcher = TensorBatcher(np.asarray(store.heights_u16[train_idx]), zc(store.conditions[train_idx]),
                            store.labels[train_idx], device, tcfg["data_on_gpu"] and device.type == "cuda", seed,
                            placement=store.conditions[train_idx][:, place_cols] if kind == "relative" else None)

    # fixed validation batch (fixed t and noise) for a low-variance loss curve
    val_idx = store.split("val")
    vi = val_idx[: tcfg["val_size"]]
    vgen = torch.Generator().manual_seed(seed + 1)
    vp = store.conditions[vi][:, place_cols].astype(np.float32)
    val_x = torch.from_numpy(heightparam.encode(store.heights(vi), vp[:, 0, None, None], vp[:, 1, None, None],
                                                kind).astype(np.float32))[:, None].to(device)
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
    best, best_step = math.inf, None
    t0 = time.time()
    losses: list[torch.Tensor] = []

    def checkpoint_extra():
        return {"best_score": best, "training_state": state_dict(sched, batcher, best_score=best,
                best_step=best_step, losses=losses, device=device)}

    def status(state, error=None):
        last = run.path / "checkpoints" / "last.pt"
        atomic_json(run.path / "status.json", {"state": state, "step": step, "max_steps": tcfg["max_steps"],
                    "run": str(run.path), "resume_from": resume, "best_step": best_step,
                    "best_score": best if math.isfinite(best) else None,
                    "last_checkpoint": str(last) if last.is_file() else None, "error": error})

    last = run.path / "checkpoints" / "last.pt"
    start_step, consistent = step, True  # consistent: model/optimizer/batcher/RNG all describe completed `step`
    try:
        if resume_state is not None:
            if resume_state["device"] != str(device) and not allow_inexact_resume:
                raise ValueError("resume device changed; use --allow-inexact-resume to acknowledge numerical/RNG differences")
            if resume_state["device"] != str(device):
                warnings.warn("Resuming on a different device is not an exact continuation", RuntimeWarning)
            batcher.load_state_dict(resume_state["batcher"])
            best, best_step = resume_state["best_score"], resume_state["best_step"]
            losses = [torch.tensor(v, device=device) for v in resume_state["losses"]]
            restore_rng(resume_state["rng"])
        status("running")
        model.train()
        with StopRequest() as stop:
            while step < tcfg["max_steps"] and not stop.requested:
                save_best = False
                consistent = False
                x, c, y = batcher.sample(tcfg["batch_size"])
                opt.zero_grad(set_to_none=True)
                with torch.autocast(**amp):
                    loss = diffusion.loss(model, x, c, y, sample_weight=class_weights[y] if class_weights is not None else None)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"], error_if_nonfinite=True)
                opt.step()
                sched.step()
                ema.update(model)
                step += 1
                losses.append(loss.detach())  # no .item() here: avoids a GPU sync every step
                consistent = True
                snapshot_due = step % tcfg["checkpoint_every"] == 0 or step == tcfg["max_steps"]
                recovery_due = snapshot_due or step % tcfg["recovery_every"] == 0
                if recovery_due:  # before previews/eval, so a failure there cannot lose this step
                    save_checkpoint(last, model, ema, opt, step, cfg, meta, checkpoint_extra())

                if step % tcfg["log_every"] == 0:
                    dt = time.time() - t0
                    mean_loss = torch.stack(losses).mean().item()
                    if not math.isfinite(mean_loss):
                        raise FloatingPointError(f"non-finite loss at step {step}")
                    run.log(step, loss=mean_loss, grad_norm=float(gnorm), lr=sched.get_last_lr()[0],
                            it_per_s=tcfg["log_every"] / dt)
                    print(f"step {step:7d} loss {mean_loss:.4f} gnorm {gnorm:.3f} {tcfg['log_every'] / dt:.1f} it/s")
                    losses, t0 = [], time.time()
                    status("running")

                if step % tcfg["val_every"] == 0 or step == tcfg["max_steps"]:
                    vl = _val_loss(diffusion, model, ema.shadow, val_x, val_c, val_y, val_t, val_n, amp)
                    run.log(step, **vl)
                    print(f"step {step:7d} " + " ".join(f"{k} {v:.4f}" for k, v in vl.items()))

                if step % tcfg["sample_every"] == 0 or step == tcfg["max_steps"]:
                    sampler = TerrainSampler(ema.shadow, diffusion, meta, device)
                    imgs = sampler.sample(n=36, seed=1234, guidance=tcfg["eval_guidance"], steps=tcfg["eval_steps"],
                                          spacing=tcfg["eval_spacing"], guidance_interval=tcfg["eval_guidance_interval"],
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
                                         spacing=tcfg["eval_spacing"], guidance_interval=tcfg["eval_guidance_interval"],
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
                        best, best_step, save_best = score, step, True
                    model.train()
                    _release_cache(device)

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

                if save_best:
                    save_checkpoint(run.path / "checkpoints" / "best.pt", model, ema, opt, step, cfg, meta, checkpoint_extra())
                    if recovery_due:  # keep last.pt's best-score state current
                        save_checkpoint(last, model, ema, opt, step, cfg, meta, checkpoint_extra())
                if snapshot_due and tcfg["keep_checkpoints"]:
                    with last.open("rb") as source, atomic_file(last.with_name(f"step_{step:07d}.pt")) as target:
                        shutil.copyfileobj(source, target)
            if stop.requested and step < tcfg["max_steps"]:
                save_checkpoint(last, model, ema, opt, step, cfg, meta, checkpoint_extra())
                raise TrainingInterrupted(run.path, step)
        status("completed")
    except BaseException as exc:
        if isinstance(exc, Exception) and not isinstance(exc, FloatingPointError) and consistent and step > start_step:
            _save_recovery(last, model, ema, opt, step, cfg, meta, checkpoint_extra, device)
        with suppress(OSError):
            status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise
    finally:
        run.close()
    return run.path


def _save_recovery(last: Path, model, ema, opt, step, cfg, meta, extra, device) -> None:
    """After a failure outside the parameter update (e.g. OOM while previewing), keep the completed step."""
    try:
        _release_cache(device)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print("not saving a recovery checkpoint: model weights are non-finite", file=sys.stderr)
            return
        save_checkpoint(last, model, ema, opt, step, cfg, meta, extra())
        print(f"saved recovery checkpoint at step {step}: {last}\nresume with: nullscape train --config <same config> "
              f"--resume {last}", file=sys.stderr)
    except Exception as err:  # never hide the original failure
        print(f"could not save a recovery checkpoint: {err}", file=sys.stderr)


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
