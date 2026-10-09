"""Export a Talus checkpoint for the browser: ONNX denoiser (fp32 and fp16) plus sampler metadata.

usage: python scripts/export_web.py --checkpoint talus-3 --out site/public/model
Writes into --out:
  <name>.onnx        fp32 denoiser: v = f(x, t, cond, known, label), dynamic batch
  <name>.fp16.onnx   the same with weights stored as fp16 (half the download; compute stays fp32)
  <name>.json        world, conditioning statistics, schedules, sampler preset, slider ranges, file hashes
  <name>.prior.bin   placement bank: float32 [N, K] raw conditions, then uint8 [N] labels
  fixtures/          reference samples for the JavaScript sampler test (site/test/sampler.test.mjs)
Needs `pip install -e .[web]` (onnx, onnxruntime). ONNX inputs are float32 except
label (int32), so the browser needs no BigInt or bool tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nullscape.inference.sampler import TerrainSampler
from nullscape.models import heightparam
from nullscape.models.diffusion import timestep_schedule
from nullscape.utils.checkpoint import load_checkpoint
from nullscape.utils.paths import resolve_checkpoint

INPUTS = ("x", "t", "cond", "known", "label")


class WebDenoiser(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, cond, known, label):
        return self.unet(x, t, cond, known > 0.5, label.long())


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _example(sampler: TerrainSampler, b: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    k, r = len(sampler.condition_keys), sampler.world.resolution
    return (torch.randn(b, 1, r, r, generator=g), torch.randint(0, 1000, (b,), generator=g).float(),
            torch.randn(b, k, generator=g), (torch.rand(b, k, generator=g) > 0.5).float(),
            torch.randint(0, len(sampler.archetypes) + 1, (b,), generator=g).int())


def half_weights(model, min_size: int = 1024):
    """Store large fp32 weights as fp16 with a Cast back to fp32: half the download, fp32 compute.
    ONNX Runtime folds the casts when the session is created, so inference cost is unchanged."""
    from onnx import TensorProto, helper, numpy_helper

    graph, casts = model.graph, []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_size:
            continue
        stored = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
        graph.initializer.remove(init)
        graph.initializer.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [init.name], to=TensorProto.FLOAT))
    for node in reversed(casts):
        graph.node.insert(0, node)
    return model


def export(sampler: TerrainSampler, out: Path, name: str) -> dict[str, Path]:
    import onnx

    wrapper = WebDenoiser(sampler.model.cpu().eval()).eval()
    fp32 = out / f"{name}.onnx"
    torch.onnx.export(wrapper, _example(sampler, 2), str(fp32), input_names=list(INPUTS), output_names=["v"],
                      dynamic_axes={n: {0: "batch"} for n in (*INPUTS, "v")}, opset_version=17,
                      do_constant_folding=True, dynamo=False)
    model = onnx.load(str(fp32))
    onnx.checker.check_model(model)
    fp16 = out / f"{name}.fp16.onnx"
    model = half_weights(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(fp16))
    return {"fp32": fp32, "fp16": fp16}


def check_forward(sampler: TerrainSampler, files: dict[str, Path]) -> dict[str, float]:
    import onnxruntime as ort

    args = _example(sampler, 4, seed=1)
    with torch.no_grad():
        ref = WebDenoiser(sampler.model).eval()(*args).numpy()
    feeds = {n: a.numpy() for n, a in zip(INPUTS, args)}
    out = {}
    for kind, path in files.items():
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            out[kind] = float(np.abs(sess.run(["v"], feeds)[0] - ref).max())
        except Exception as exc:  # fp16 kernels missing on the CPU provider are not fatal for the browser
            out[kind] = f"not runnable on ORT CPU: {type(exc).__name__}"
    return out


def reference_fixtures(sampler: TerrainSampler, out: Path) -> list[dict]:
    """Full samples from the PyTorch sampler with injected noise, for the JavaScript parity test."""
    r, k = sampler.world.resolution, len(sampler.condition_keys)
    cases = [("islands", {"water_fraction": 0.6}, 50, 2.0), ("ridges", {}, 25, 2.0), (None, {}, 25, 1.0)]
    rng = np.random.default_rng(7)
    sampler.model.cpu().eval()
    written = []
    for i, (archetype, props, steps, guidance) in enumerate(cases):
        raw, kn = sampler._raw_conditions(1, props, None, None)
        lab = sampler._label_ids(1, archetype, None)
        raw, kn = sampler._fill_placement(raw, kn, lab, [1000 + i])
        z = np.where(kn, (raw - sampler.cond_mean) / sampler.cond_std, 0.0).astype(np.float32)
        noise = rng.standard_normal((1, 1, r, r)).astype(np.float32)
        x = sampler.diffusion.cpu().ddim_sample(
            sampler.model, torch.from_numpy(z), torch.from_numpy(kn), torch.from_numpy(lab),
            lambda _: torch.from_numpy(noise), steps=steps, guidance=guidance, eta=0.0, spacing="quadratic",
            clip_range=heightparam.X0_RANGE["relative"])[:, 0].numpy()
        jm, jr = (sampler.condition_keys.index(sampler.height_param[key]) for key in ("mean_key", "relief_key"))
        heights = np.clip(heightparam.decode(x, raw[:, jm, None, None], raw[:, jr, None, None], "relative"), 0, 1)
        stem = f"case{i}"
        noise.tofile(out / f"{stem}.noise.bin")
        heights.astype(np.float32).tofile(out / f"{stem}.heights.bin")
        written.append({"stem": stem, "steps": steps, "guidance": guidance, "label": int(lab[0]),
                        "z": z[0].tolist(), "known": kn[0].astype(int).tolist(),
                        "mean": float(raw[0, jm]), "relief": float(raw[0, jr])})
    (out / "fixtures.json").write_text(json.dumps(written, indent=1), encoding="utf-8")
    assert k == len(written[0]["z"])
    return written


def metadata(sampler: TerrainSampler, ck: dict, source: Path, files: dict[str, Path], name: str) -> dict:
    bank_c, bank_l = sampler._bank
    keys, archetypes = sampler.condition_keys, sampler.archetypes
    ranges = {}
    for a, label in [("any", None), *[(n, i) for i, n in enumerate(archetypes)]]:
        sel = bank_c if label is None else bank_c[bank_l == label]
        ranges[a] = {key: [float(np.percentile(sel[:, j], 2)), float(np.median(sel[:, j])),
                           float(np.percentile(sel[:, j], 98))] for j, key in enumerate(keys)}
    release = ck.get("release", {})
    return {
        "name": release.get("name", name), "version": release.get("version"), "license": release.get("license"),
        "checkpoint_sha256": _sha256(source), "step": int(ck["step"]),
        "world": sampler.world.to_dict(), "condition_keys": keys, "archetypes": archetypes,
        "cond_mean": sampler.cond_mean.tolist(), "cond_std": sampler.cond_std.tolist(),
        "height_param": sampler.height_param, "x0_clip": heightparam.X0_RANGE[sampler.height_param["kind"]],
        "sampling": {**sampler.resolve_sampling(), "guidance_interval": list(sampler.resolve_sampling()["guidance_interval"])},
        "alphas_cumprod": sampler.diffusion.alphas_cumprod.cpu().tolist(),
        "schedules": {str(s): timestep_schedule(sampler.diffusion.cfg.timesteps, s, "quadratic").tolist()
                      for s in (12, 25, 50)},
        "prior": {"file": f"{name}.prior.bin", "count": int(len(bank_l)), "keys": len(keys)},
        "ranges": ranges,
        "models": {kind: {"file": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for kind, p in files.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="talus-3")
    ap.add_argument("--out", default="site/public/model")
    ap.add_argument("--name", default=None, help="file stem; default from the release name, e.g. talus-3")
    args = ap.parse_args()

    source = resolve_checkpoint(args.checkpoint)
    ck = load_checkpoint(source)
    name = args.name or ck.get("release", {}).get("name", source.stem).lower()
    out = Path(args.out)
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    sampler = TerrainSampler.from_checkpoint(source, device="cpu")
    files = export(sampler, out, name)
    print("forward max |ONNX - PyTorch|:", check_forward(sampler, files))
    bank_c, bank_l = sampler._bank
    with open(out / f"{name}.prior.bin", "wb") as f:
        f.write(np.ascontiguousarray(bank_c, dtype=np.float32).tobytes())
        f.write(np.asarray(bank_l, dtype=np.uint8).tobytes())
    meta = metadata(sampler, ck, source, files, name)
    (out / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    fixtures = reference_fixtures(sampler, out / "fixtures")
    for kind, info in meta["models"].items():
        print(f"{kind}: {info['file']} {info['bytes'] / 1e6:.1f} MB")
    print(f"wrote {out / (name + '.json')} and {len(fixtures)} reference samples")


if __name__ == "__main__":
    main()
