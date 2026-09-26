"""Does uint16 storage change the measured conditioning properties / metrics?

Stored conditions were measured on the float heightmap before quantization; evaluation measures
reference maps after dequantization. Regenerating maps (bit-exact generator) lets us compare
float vs quantized measurements on identical terrain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nullscape.data.storage import TerrainStore, dequantize, quantize
from nullscape.metrics.quality import CONDITION_KEYS, compute_metrics
from nullscape.terrain.generator import GeneratorConfig, generate
from nullscape.world import WorldSpec

OUT = Path(__file__).resolve().parent / "raw" / "quantization.json"


def main() -> None:
    st = TerrainStore.open("base64")
    m = st.manifest
    world = WorldSpec.from_dict(m["world"])
    cfg = GeneratorConfig.from_dict(m["config"]["generator"])
    idx = np.sort(np.random.default_rng(3).choice(len(st), 300, replace=False))
    rows = []
    for i in idx:
        h, rec = generate(int(i), int(m["config"]["seed"]), world, cfg)
        mf = compute_metrics(h, world)
        mq = compute_metrics(dequantize(quantize(h)), world)
        rows.append((rec["archetype"], mf, mq, st.conditions[i]))
    keys = list(rows[0][1])
    out = {"n": len(rows), "stored_vs_float_max_abs": {k: float(max(abs(r[1][k] - r[3][j]) for r in rows))
                                                       for j, k in enumerate(CONDITION_KEYS)},
           "float_vs_quantized": {}}
    for k in keys:
        d = np.array([r[2][k] - r[1][k] for r in rows])
        sd = np.std([r[1][k] for r in rows]) + 1e-12
        per = {a: float(np.mean([r[2][k] - r[1][k] for r in rows if r[0] == a])) for a in sorted({r[0] for r in rows})}
        out["float_vs_quantized"][k] = {"mean_diff": float(d.mean()), "max_abs_diff": float(np.abs(d).max()),
                                        "mean_diff_in_std": float(d.mean() / sd), "per_archetype_mean_diff": per}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out["float_vs_quantized"].items():
        if abs(v["mean_diff_in_std"]) > 0.01 or v["max_abs_diff"] > 1e-3:
            print(f"{k:22s} mean {v['mean_diff']:+.5f} ({v['mean_diff_in_std']:+.3f} std) max {v['max_abs_diff']:.5f}")
    print("stored vs float:", out["stored_vs_float_max_abs"])


if __name__ == "__main__":
    main()
