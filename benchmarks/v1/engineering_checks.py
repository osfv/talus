"""Engineering benchmark for NULLSCAPE v1 (CPU only; never writes to data/ or runs/).

Writes benchmarks/v1/raw/engineering.json and scratch outputs under benchmarks/v1/engineering/.
Run with CUDA_VISIBLE_DEVICES="" so nothing touches the GPU while GPU benchmarks run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "v1"
ENG = OUT / "engineering"
PY = sys.executable
NULLSCAPE = str(Path(PY).with_name("nullscape.exe"))
CKPT = ROOT / "runs" / "20260925-164759_diffusion64" / "checkpoints" / "step_0030000.pt"


def cpu_env(**extra) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", **extra}
    return env


def run(cmd: list[str], env: dict | None = None, timeout: int = 1800) -> dict:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env or cpu_env(), capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd)), "exit_code": p.returncode,
            "seconds": round(time.time() - t0, 2), "stderr_tail": p.stderr.strip().splitlines()[-3:] if p.returncode else []}


def check_pytest() -> dict:
    xml = ENG / "pytest_junit.xml"
    r = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", "--durations=10"])
    suite = ET.parse(xml).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    slow = sorted(((float(tc.get("time", 0)), f"{tc.get('classname')}::{tc.get('name')}") for tc in suite.iter("testcase")),
                  reverse=True)[:8]
    return {**r, "tests": int(suite.get("tests")), "failures": int(suite.get("failures")),
            "errors": int(suite.get("errors")), "skipped": int(suite.get("skipped")), "time": float(suite.get("time")),
            "slowest": [{"test": n, "seconds": t} for t, n in slow]}


def check_dataset() -> dict:
    from nullscape.data.dataset import TerrainDataset
    from nullscape.data.storage import SPLITS, TerrainStore, quantize
    from nullscape.metrics.quality import compute_metrics, condition_vector
    from nullscape.terrain.generator import GeneratorConfig, generate
    from nullscape.utils.config import config_hash
    from nullscape.world import WorldSpec

    st = TerrainStore.open("base64")
    m = st.manifest
    splits = {s: st.split(s) for s in SPLITS}
    union = np.sort(np.concatenate(list(splits.values())))
    ds = TerrainDataset(st, "test", augment=False, in_memory=False)
    item = ds[0]
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(len(st), 200, replace=False))
    stored = st.conditions[sample]
    recomputed = np.stack([condition_vector(compute_metrics(h, st.world)) for h in st.heights(sample)])
    world = WorldSpec.from_dict(m["world"])
    gen_cfg = GeneratorConfig.from_dict(m["config"].get("generator"))
    regen_idx = sample[:100]
    exact, max_diff = 0, 0
    labels_ok = 0
    for i in regen_idx:
        h, rec = generate(int(i), int(m["config"]["seed"]), world, gen_cfg)
        q = quantize(h)
        d = int(np.abs(q.astype(np.int64) - st.heights_u16[i].astype(np.int64)).max())
        exact += d == 0
        max_diff = max(max_diff, d)
        labels_ok += rec["archetype_id"] == int(st.labels[i])
    return {
        "manifest_config_sha256_matches": config_hash(m["config"]) == m["config_sha256"],
        "n": len(st), "heights_shape": list(st.heights_u16.shape), "heights_dtype": str(st.heights_u16.dtype),
        "splits": {s: int(len(v)) for s, v in splits.items()},
        "splits_disjoint_and_covering": bool(np.array_equal(union, np.arange(len(st)))),
        "dataset_item": {"x_shape": list(item["x"].shape), "x_min": float(item["x"].min()), "x_max": float(item["x"].max()),
                         "cond_shape": list(item["cond"].shape)},
        "conditions_recomputed_max_abs_diff": float(np.abs(stored - recomputed).max()),
        "conditions_recomputed_n": len(sample),
        "regeneration": {"n": len(regen_idx), "bit_exact": int(exact), "max_uint16_diff": max_diff,
                         "labels_match": int(labels_ok)},
        "build_git": m.get("git"), "generator_version": m["generator_version"],
    }


def check_exports() -> dict:
    from nullscape.data.storage import TerrainStore
    from nullscape.export.engine import (
        export_heightmap, read_png16, read_r16, resample, to_uint16, unity_size,
    )

    st = TerrainStore.open("base64")
    idx = int(st.split("test")[0])
    h = st.heights(np.array([idx]))[0]
    out = ENG / "export"
    if out.exists():
        shutil.rmtree(out)
    w = export_heightmap(h, st.world, out, "native", ["png16", "r16", "npy", "obj"], {"index": idx})
    obj = (out / "native.obj").read_text().splitlines()
    verts = np.array([[float(x) for x in l.split()[1:]] for l in obj if l.startswith("v ")])
    side = json.loads(w["json"].read_text())
    native = {
        "files": sorted(p.name for p in out.iterdir()),
        "png16_roundtrip_exact": bool(np.array_equal(to_uint16(read_png16(w["png16"])), to_uint16(h))),
        "r16_roundtrip_exact": bool(np.array_equal(to_uint16(read_r16(w["r16"], 64)), to_uint16(h))),
        "r16_bytes": w["r16"].stat().st_size, "r16_expected_bytes": 64 * 64 * 2,
        "npy_exact": bool(np.array_equal(np.load(w["npy"]), h.astype(np.float32))),
        "obj_counts": {"v": int(sum(l.startswith("v ") for l in obj)), "vn": int(sum(l.startswith("vn ") for l in obj)),
                       "f": int(sum(l.startswith("f ") for l in obj)), "expected_v": 64 * 64, "expected_f": 2 * 63 * 63},
        "obj_height_max_abs_err_m": float(np.abs(verts[:, 1] - (h * st.world.max_height_m).ravel()).max()),
        "obj_xz_extent_m": [float(verts[:, 0].min()), float(verts[:, 0].max())],
        "sidecar": side,
    }
    wu = export_heightmap(h, st.world, out, "unity", ["png16", "r16"], {"index": idx}, size=unity_size(64))
    hu = read_r16(wu["r16"], unity_size(64))
    side_u = json.loads(wu["json"].read_text())
    unity_export = {
        "size": int(hu.shape[0]), "corners_preserved_max_abs": float(max(abs(hu[0, 0] - h[0, 0]), abs(hu[-1, -1] - h[-1, -1]),
                                                                          abs(hu[0, -1] - h[0, -1]), abs(hu[-1, 0] - h[-1, 0]))),
        "sidecar_cell_size_m": side_u["cell_size_m"], "sidecar_extent_m": side_u["extent_m"],
        "actual_sample_spacing_m_after_resample": (st.world.extent_m - st.world.cell_size_m) / (hu.shape[0] - 1),
        "actual_footprint_m": st.world.extent_m - st.world.cell_size_m,
    }
    sizes = {}
    valid = {2 ** k + 1 for k in range(3, 14)}
    for r in (64, 65, 128, 129, 257, 513, 1025):
        u = unity_size(r)
        sizes[str(r)] = {"unity_size": u, "input_already_valid_unity_size": r in valid,
                         "upsampled_although_valid": r in valid and u != r,
                         "file_bytes_r16": u * u * 2, "file_bytes_if_kept": r * r * 2}
    t = np.linspace(0, 1, 513)[None, :] * np.ones((513, 1))
    identity_513 = float(np.abs(resample(t.astype(np.float32), 513) - t).max())
    return {"native": native, "unity_export_64": unity_export, "unity_size_table": sizes,
            "resample_identity_513_max_abs": identity_513}


def check_cli() -> dict:
    tmp = ENG / "cli"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    data_root, runs_root = tmp / "data", tmp / "runs"
    env = cpu_env(NULLSCAPE_DATA_ROOT=str(data_root), NULLSCAPE_RUNS_ROOT=str(runs_root))
    base64 = str(ROOT / "data" / "base64")
    cmds = []
    for sub in ("", "gen-dataset", "viz", "export", "train", "sample", "evaluate", "compare", "sampler-sweep", "artifacts"):
        cmds.append(run([NULLSCAPE, *([sub] if sub else []), "--help"], env))
    cmds.append(run([NULLSCAPE, "gen-dataset", "--config", "configs/dataset/smoke.yaml", "--set", "name=eng_tiny",
                     "--set", "n=24", "--workers", "2"], env))
    tiny = data_root / "eng_tiny"
    cmds.append(run([NULLSCAPE, "viz", "--dataset", str(tiny), "--n", "6", "--per-archetype", "2", "--inspect", "1",
                     "--out", str(tmp / "viz")], env))
    cmds.append(run([NULLSCAPE, "export", "--dataset", base64, "--index", "0", "--formats", "png16,r16,npy,obj",
                     "--unity", "--out", str(tmp / "export")], env))
    cmds.append(run([NULLSCAPE, "sample", "--checkpoint", str(CKPT), "--n", "2", "--steps", "5", "--archetype", "islands",
                     "--prop", "water_fraction=0.6", "--formats", "png16,r16,obj", "--unity", "--out", str(tmp / "sample")], env))
    cmds.append(run([NULLSCAPE, "evaluate", "--checkpoint", str(CKPT), "--dataset", base64, "--split", "test", "--n", "8",
                     "--steps", "5", "--modes", "conditional,unconditional", "--no-memorization", "--out", str(tmp / "evaluate")], env))
    cmds.append(run([NULLSCAPE, "compare", "--checkpoint", str(CKPT), "--dataset", base64, "--split", "test", "--n", "2",
                     "--seeds", "1", "--steps", "5", "--n-3d", "1", "--out", str(tmp / "compare")], env))
    cmds.append(run([NULLSCAPE, "sampler-sweep", "--checkpoint", str(CKPT), "--dataset", base64, "--n", "4",
                     "--steps-list", "3", "--spacings", "uniform,quadratic", "--etas", "0", "--out", str(tmp / "sweep")], env))
    cmds.append(run([NULLSCAPE, "train", "--config", "configs/train/smoke.yaml", "--set", f"dataset={tiny}",
                     "--set", "train.max_steps=4", "--set", "train.eval_every=4", "--set", "train.sample_every=4",
                     "--set", "train.val_every=2", "--set", "train.val_size=4", "--set", "train.eval_samples=2",
                     "--set", "train.eval_steps=3", "--set", "train.checkpoint_every=4", "--set", "train.bf16=false"], env))
    runs = sorted(runs_root.glob("*_smoke")) if runs_root.exists() else []
    if runs:
        cmds.append(run([NULLSCAPE, "artifacts", "--run", str(runs[-1]), "--out", str(tmp / "artifacts"),
                         "--device", "cpu", "--n", "2", "--steps", "3", "--batch-size", "4"], env))
    expected = {"viz/mixed.png": tmp / "viz" / "mixed.png", "export/*.r16": next((tmp / "export").glob("*.r16"), None),
                "sample/grid.png": tmp / "sample" / "grid.png", "evaluate/report.json": tmp / "evaluate" / "report.json",
                "compare/compare_2d.png": tmp / "compare" / "compare_2d.png",
                "sweep/sampler_sweep.json": tmp / "sweep" / "sampler_sweep.json",
                "artifacts/index.md": tmp / "artifacts" / "index.md"}
    sample_json = json.loads((tmp / "sample" / "sample_0_000.json").read_text()) if (tmp / "sample" / "sample_0_000.json").exists() else None
    return {"commands": cmds, "all_exit_zero": all(c["exit_code"] == 0 for c in cmds),
            "expected_outputs": {k: bool(v and Path(v).exists()) for k, v in expected.items()},
            "sample_unity_sidecar_resolution": sample_json["resolution"] if sample_json else None}


def main() -> None:
    ENG.mkdir(parents=True, exist_ok=True)
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for name, fn in (("dataset", check_dataset), ("exports", check_exports), ("cli", check_cli), ("pytest", check_pytest)):
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as exc:  # record and continue: this is a report, not a gate
            res[name] = {"error": f"{type(exc).__name__}: {exc}"}
        res[name + "_seconds"] = round(time.time() - t0, 1)
        print(f"[engineering] {name} done in {res[name + '_seconds']}s", flush=True)
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        (OUT / "raw" / "engineering.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
