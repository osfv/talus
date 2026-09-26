"""Peak process RAM for typical v1 workloads (run with the GPU otherwise idle).

Writes benchmarks/v1/raw/ram.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v1_benchmark as B  # noqa: E402


def main() -> None:
    out = {}
    st = B.store()
    out["after_open_dataset_mb"] = B._peak_rss_mb()
    s = B.sampler_for(B.load_json("selection")["checkpoint"])
    out["after_load_checkpoint_mb"] = B._peak_rss_mb()
    cfg = B.load_json("pareto")["recommended_config"]
    idx = st.split("test")[:128]
    t0 = time.time()
    maps, secs = B.conditional_gen(s, st, idx, cfg, seed=1)
    out["sample_128_official_seconds"] = secs
    out["after_sampling_128_mb"] = B._peak_rss_mb()
    from nullscape.metrics.distribution import metric_table

    metric_table(maps, st.world)
    bank = st.heights(st.split("train"))
    out["train_bank_mb"] = bank.nbytes / 2**20
    out["after_train_bank_mb"] = B._peak_rss_mb()
    out["note"] = "peak working set of this process (Windows PeakWorkingSetSize), MB"
    (B.RAW / "ram.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
