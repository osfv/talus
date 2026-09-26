"""Build demo/nullscape-demo.html: a self-contained, playable first-person run through v1-generated maps.

usage: python demo/build_demo.py
Maps come from the official v1 configuration's benchmark output
(benchmarks/v1/artifacts_official/step_0030000/generated_eval.npy, 50-step quadratic DDIM, guidance 2.0).
Routes come from nullscape.metrics.traversability (longest walkable route in the largest component).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from nullscape.data.storage import TerrainStore
from nullscape.eval.artifacts import ArtifactSpec, eval_halves
from nullscape.metrics.traversability import analyze, longest_route

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OFFICIAL = ROOT / "benchmarks" / "v1" / "artifacts_official" / "step_0030000"
PER_ARCHETYPE = 5
MIN_ROUTE_CELLS = 24


def main() -> None:
    st = TerrainStore.open("base64")
    spec = ArtifactSpec(**json.loads((ROOT / "benchmarks" / "v1" / "artifacts_official" / "spec.json").read_text()))
    a, _ = eval_halves(st, spec)
    gens = np.load(OFFICIAL / "generated_eval.npy")
    labels = st.labels[a]
    world = st.world
    maps = []
    for c, arch in enumerate(st.archetypes):
        taken = 0
        for i in np.flatnonzero(labels == c):
            if taken == PER_ARCHETYPE:
                break
            h = gens[i]
            route = longest_route(analyze(h, world))
            if not route or len(route) < MIN_ROUTE_CELLS:
                continue
            step = max(1, len(route) // 60)
            pts = route[::step] + ([route[-1]] if (len(route) - 1) % step else [])
            q = np.rint(np.clip(h, 0, 1) * 65535).astype("<u2")
            p2, p98 = np.percentile(h, [2, 98])
            maps.append({"archetype": arch, "relief_m": float((p98 - p2) * world.max_height_m),
                         "b64": base64.b64encode(q.tobytes()).decode(), "route": [[int(r), int(cc)] for r, cc in pts],
                         "source_index": int(i)})
            taken += 1
    order = np.random.default_rng(26).permutation(len(maps))  # mix archetypes for next/prev browsing
    data = {"size": world.resolution, "cell_size_m": world.cell_size_m, "max_height_m": world.max_height_m,
            "sea_level": world.sea_level, "source": "NULLSCAPE v1 official (checkpoint 30k, 50-step quadratic, g2.0)",
            "maps": [maps[k] for k in order]}
    html = (HERE / "template.html").read_text(encoding="utf-8").replace("{{DATA}}", json.dumps(data, separators=(",", ":")))
    if "\u2014" in html or "\u2013" in html:
        raise SystemExit("em/en dash in output")
    out = HERE / "nullscape-demo.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, {len(maps)} maps)")


if __name__ == "__main__":
    main()
