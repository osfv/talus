"""TEST-split scoring of an experiment checkpoint with the official v1 sampler, in scorecard format.

usage: python benchmarks/v2/score_test.py --name exp1 --ckpt runs/<run>/checkpoints/last.pt
Writes benchmarks/v2/<name>/results.json (+ artifacts_official/), then:
  python benchmarks/scorecard.py v1=benchmarks/v1/results.json exp1=benchmarks/v2/exp1/results.json \
      --out=docs/V2_EXP1_SCORECARD.md
Uses the same ArtifactSpec as v1's official suite (same halves, conditions, seeds), so the Δ is weights-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "v1"))
import run_v1_benchmark as B  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    import torch

    from nullscape.eval.artifacts import ArtifactSpec, generate_checkpoint_artifacts
    from nullscape.inference.sampler import TerrainSampler

    out = ROOT / "benchmarks" / "v2" / args.name
    out.mkdir(parents=True, exist_ok=True)
    v1 = json.loads((ROOT / "benchmarks" / "v1" / "results.json").read_text())
    spec = ArtifactSpec(**json.loads((ROOT / "benchmarks" / "v1" / "artifacts_official" / "spec.json").read_text()))
    st = B.store()
    s = TerrainSampler.from_checkpoint(args.ckpt, device="cuda")
    t0 = time.time()
    d = generate_checkpoint_artifacts(s, st, out / "artifacts_official", s.checkpoint_info["step"], spec,
                                      train_bank=st.heights(st.split("train")), memorization_device="cuda",
                                      save_generated=True)
    rep = json.loads((d / "report.json").read_text())["modes"]["conditional"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sha = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()
    res = {
        "benchmark": "NULLSCAPE v1 protocol", "experiment": args.name,
        "provenance": {"default_sampler": v1["provenance"]["default_sampler"],
                       "checkpoint": {"path": args.ckpt, "sha256": sha, "step": int(ck["step"]),
                                      "train_config": ck.get("train_config", {}).get("name"),
                                      "diffusion_config": ck["diffusion_config"]},
                       "spec": asdict(spec), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "checkpoints_test": {"compact": {}},
        "official": {"checkpoint": int(ck["step"]), "sampler": v1["official"]["sampler"],
                     "compact": B.compact(rep), "wall_seconds": time.time() - t0,
                     "artifacts_dir": str(d.relative_to(ROOT))},
        "perf": v1["perf"],  # same architecture and sampler, so v1's isolated timings apply
        "baselines": v1["baselines"],
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, default=B._json_default))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
