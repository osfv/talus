"""Build docs/blog/talus-2.html: the Talus-2 release post, one self-contained file (pixel theme).

usage: python docs/blog/build_talus2.py
Inputs (all real model output, nothing hand-made):
  benchmarks/v2/talus2/artifacts_official/step_0010000/fixed_seed.npy (+ meta)  hero flyover, "spot the real" pairs
  docs/blog/assets/talus_compare_seed7.npz  same-seed maps from Talus-1 / 1.1 / 2
      (created from scripts/compare_samples.py output in reports/samples/talus_all if missing)
  benchmarks/v2/talus2/scorecard.json  scorecard rows
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

from nullscape.data.storage import TerrainStore

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
T2 = ROOT / "benchmarks" / "v2" / "talus2" / "artifacts_official" / "step_0010000"
NPZ = HERE / "assets" / "talus_compare_seed7.npz"
MODELS = ["Talus-1", "Talus-1.1", "Talus-2"]
ARCH = ["plains", "hills", "mountains", "ridges", "islands", "mesas"]
HERO = ["mountains", "islands", "ridges", "mesas", "hills"]
ROWS = [("Realism", "25 terrain metrics"), ("Spectrum", "detail at every scale"), ("Heights", "elevation distribution"),
        ("Slopes", "slope distribution"), ("Control", "follows requested properties"), ("Playability", "walkable maps"),
        ("Diversity", "variety between maps"), ("Originality", "distance from training maps"),
        ("Clean detail", "no excess micro-bumps"), ("Average", "mean of the nine")]


def b64(h: np.ndarray) -> str:
    return base64.b64encode(np.rint(np.clip(h, 0, 1) * 65535).astype("<u2").tobytes()).decode()


def void_and_cluster(n: int = 64, sigma: float = 1.5, seed: int = 7) -> np.ndarray:
    """Blue-noise dither ranks 0..n*n-1 (Ulichney 1993), toroidal Gaussian void/cluster filter."""
    rng = np.random.default_rng(seed)
    d = np.minimum(np.arange(n), n - np.arange(n))
    kern = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma**2))

    def add(E, p, s):
        E += s * np.roll(np.roll(kern, p[0], 0), p[1], 1)

    def energy(mask):
        E = np.zeros((n, n))
        for p in np.argwhere(mask):
            add(E, p, 1)
        return E

    m = n * n
    pat = np.zeros((n, n), bool)
    pat.flat[rng.choice(m, m // 10, replace=False)] = True
    E = energy(pat)
    while True:                                   # relax: move tightest cluster into largest void
        c = np.unravel_index(np.argmax(np.where(pat, E, -np.inf)), pat.shape)
        pat[c] = False
        add(E, c, -1)
        v = np.unravel_index(np.argmin(np.where(~pat, E, np.inf)), pat.shape)
        pat[v] = True
        add(E, v, 1)
        if v == c:
            break
    rank = np.zeros((n, n), np.int32)
    ones = int(pat.sum())
    p1, e1 = pat.copy(), E.copy()
    for r in range(ones - 1, -1, -1):             # phase 1: rank the initial points
        c = np.unravel_index(np.argmax(np.where(p1, e1, -np.inf)), pat.shape)
        p1[c] = False
        add(e1, c, -1)
        rank[c] = r
    p2, e2 = pat.copy(), E.copy()
    for r in range(ones, m // 2):                 # phase 2: fill largest voids up to half
        v = np.unravel_index(np.argmin(np.where(~p2, e2, np.inf)), pat.shape)
        p2[v] = True
        add(e2, v, 1)
        rank[v] = r
    e3 = energy(~p2)
    for r in range(m // 2, m):                    # phase 3: minority is now the zeros
        c = np.unravel_index(np.argmax(np.where(~p2, e3, -np.inf)), pat.shape)
        p2[c] = True
        add(e3, c, -1)
        rank[c] = r
    return rank


def blue_noise_b64() -> str:
    cache = HERE / "assets" / "bluenoise64.npy"
    if not cache.exists():
        np.save(cache, void_and_cluster())
    rank = np.load(cache)
    assert sorted(rank.ravel().tolist()) == list(range(rank.size)), "blue-noise ranks must be a permutation"
    return base64.b64encode(rank.astype("<u2").tobytes()).decode()


def compare_maps() -> dict[str, dict[str, np.ndarray]]:
    if not NPZ.exists():
        src = ROOT / "reports" / "samples" / "talus_all"
        np.savez_compressed(NPZ, **{f"{m}|{a}": np.load(src / f"{m}_{a}_seed7.npy") for m in MODELS for a in ARCH})
    z = np.load(NPZ)
    return {m: {a: z[f"{m}|{a}"] for a in ARCH} for m in MODELS}


def main() -> None:
    fixed = np.load(T2 / "fixed_seed.npy")
    meta = json.loads((T2 / "fixed_seed_meta.json").read_text())
    labels = meta["labels"]
    store = TerrainStore.open("base64")
    cmp = compare_maps()
    rng = np.random.default_rng(11)
    pairs = []
    for a in ARCH:
        i = labels.index(a)
        pairs.append({"archetype": a, "real": b64(store.heights(np.array([meta["indices"][i]]))[0]),
                      "model": b64(fixed[i]), "real_first": bool(rng.random() < 0.5)})
    sc = json.loads((ROOT / "benchmarks" / "v2" / "talus2" / "scorecard.json").read_text())["scores"]
    rows = [{"name": n, "desc": d, "t2": sc["Talus-2 official"][n], "t11": sc["Talus-1.1 official"][n],
             "t1": sc["Talus-1 official"][n]} for n, d in ROWS]
    data = {
        "bn": blue_noise_b64(),
        "sea_level": store.world.sea_level,
        "hero": [{"name": a, "b64": b64(fixed[labels.index(a)])} for a in HERO],
        "plains": {"t1": b64(cmp["Talus-1"]["plains"]), "t2": b64(cmp["Talus-2"]["plains"])},
        "compare": {"models": MODELS, "archetypes": ARCH, "maps": {m: {a: b64(cmp[m][a]) for a in ARCH} for m in MODELS}},
        "pairs": pairs,
        "score": {"rows": rows},
        # three real Talus-2 maps per terrain type for the "out of the static" tiles
        "forms": [{"archetype": a, "maps": [b64(fixed[i]) for i, lab in enumerate(labels) if lab == a] + [b64(cmp["Talus-2"][a])]}
                  for a in ARCH],
    }
    html = (HERE / "talus2_template.html").read_text(encoding="utf-8").replace("{{DATA}}", json.dumps(data, separators=(",", ":")))
    if re.findall(r"\{\{[^}]+\}\}", html):
        raise SystemExit("unreplaced placeholder")
    if "\u2014" in html or "\u2013" in html:
        raise SystemExit("em/en dash in output")
    out = HERE / "talus-2.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
