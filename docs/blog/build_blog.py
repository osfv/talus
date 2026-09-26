"""Build docs/blog/nullscape-v1.html: a single self-contained file (images and data inlined).

usage: python docs/blog/build_blog.py
Inputs: docs/blog/template.html, benchmark figures, benchmarks/scorecard.json and real v1 samples
(benchmarks/v1/artifacts_official/step_0030000/fixed_seed.npy).
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OFFICIAL = ROOT / "benchmarks" / "v1" / "artifacts_official" / "step_0030000"
HERO_ARCHETYPES = ["mountains", "islands", "ridges", "mesas", "hills"]
MODEL = "Talus-1"  # scorecard tag used for the v1 results (benchmarks/scorecard.json)

SCORE_ROWS = [
    ("Realism", "25 terrain metrics vs real"),
    ("Spectrum", "detail at every scale"),
    ("Heights", "elevation distribution"),
    ("Slopes", "slope distribution"),
    ("Control", "requested vs measured properties"),
    ("Playability", "walkable maps vs real"),
    ("Diversity", "variety between maps"),
    ("Originality", "distance from training maps"),
    ("Clean detail", "no excess micro-bumps"),
]


def image_data_uri(rel: str, max_w: int) -> str:
    img = Image.open(ROOT / rel).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=84, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def terrain_data() -> str:
    maps = np.load(OFFICIAL / "fixed_seed.npy")
    labels = json.loads((OFFICIAL / "fixed_seed_meta.json").read_text())["labels"]
    out = []
    for name in HERO_ARCHETYPES:
        i = labels.index(name)
        q = np.clip(np.rint(maps[i] * 255), 0, 255).astype(np.uint8)
        out.append({"name": name, "b64": base64.b64encode(q.tobytes()).decode()})
    return json.dumps({"size": int(maps.shape[-1]), "sea_level": 0.2, "maps": out}, separators=(",", ":"))


def score_rows() -> str:
    sc = json.loads((ROOT / "benchmarks" / "scorecard.json").read_text())
    s = sc["scores"]
    cols = [f"{MODEL} 30k", "baseline: blur", "baseline: spectral_noise", "baseline: retrieval"]
    rows = []
    for name, desc in SCORE_ROWS:
        v = s[f"{MODEL} official"][name]
        cells = "".join(f'<span class="score-cell{" dim" if c.startswith("baseline") else ""}" role="cell">'
                        f'{s[c][name]:.0f}</span>' for c in cols)
        rows.append(f'<div class="score-row" role="row"><span class="score-name" role="rowheader"><b>{name}</b>'
                    f'<span>{desc}</span></span><span class="meter" role="cell"><i style="--v:{v:.1f}" '
                    f'class="{"zero" if v < 0.5 else ""}"></i><em>{v:.0f}</em></span>{cells}</div>')
    sp = sc["speed_maps_per_s"]
    rows.append(f'<div class="score-row" role="row"><span class="score-name" role="rowheader"><b>Speed</b>'
                f'<span>maps per second, batch 128</span></span><span class="meter" role="cell"><em>'
                f'{sp[f"{MODEL} official"]:.1f}</em></span><span class="score-cell" role="cell">{sp[f"{MODEL} 30k"]:.1f}</span>'
                + '<span class="score-cell dim" role="cell">-</span>' * 3 + "</div>")
    return "\n        ".join(rows)


def main() -> None:
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{IMG:([^|}]+)\|(\d+)\}\}", lambda m: image_data_uri(m.group(1), int(m.group(2))), html)
    html = html.replace("{{TERRAIN_DATA}}", terrain_data()).replace("{{SCORE_ROWS}}", score_rows())
    left = re.findall(r"\{\{[^}]+\}\}", html)
    if left:
        raise SystemExit(f"unreplaced placeholders: {left}")
    for bad in ("\u2014", "\u2013"):
        if bad in html:
            raise SystemExit("em/en dash found in output")
    out = HERE / "nullscape-v1.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
