"""NULLSCAPE-Bench scorecard: one table of named benchmarks, 0-100 scores, and deltas between versions.

usage:
  python benchmarks/scorecard.py                      # v1 only (benchmarks/v1/results.json)
  python benchmarks/scorecard.py v1=benchmarks/v1/results.json v2=benchmarks/v2/results.json

Every score is computed on the held-out TEST split with the noise-floor protocol (1000 maps per half) and
is defined so that 100 = indistinguishable from real procedural terrain (or a perfect result) and higher is
better. Definitions are fixed here and must not change between versions; add new benchmarks instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS = [
    # name, what it measures, function(compact) -> score
    ("Realism", "all 25 terrain metrics vs real (normalized W1)", lambda c: _inv(c["ratio"]["metric_w1_mean"])),
    ("Spectrum", "multi-scale detail: power spectrum vs real", lambda c: _inv(c["ratio"]["rapsd_distance"])),
    ("Heights", "elevation distribution vs real", lambda c: _inv(c["ratio"]["height_w1"])),
    ("Slopes", "slope distribution vs real", lambda c: _inv(c["ratio"]["slope_w1_deg"])),
    ("Control", "requested vs measured properties (1 - mean nMAE)",
     lambda c: None if "adherence_nmae_mean" not in c else 100 * max(0.0, 1 - c["adherence_nmae_mean"])),
    ("Playability", "traversability pass rate vs real", lambda c: min(100.0, 100 * c["trav_pass_gen"] / c["trav_pass_ref"])),
    ("Diversity", "mean pairwise difference vs real", lambda c: min(100.0, 100 * c["diversity_gen"] / c["diversity_ref"])),
    ("Originality", "distance to nearest training map vs held-out maps",
     lambda c: None if "memorization" not in c else min(100.0, 100 * c["memorization"]["nn_median_ratio"])),
    ("Clean detail", "no excess micro-bumps (sink + peak density vs real)",
     lambda c: 50 * (min(1.0, c["sink_density"]["ref"] / c["sink_density"]["gen"]) +
                     min(1.0, c["peak_density"]["ref"] / c["peak_density"]["gen"]))),
]


def _inv(ratio: float) -> float:
    return min(100.0, 100.0 / ratio) if ratio > 0 else 100.0


ORIGINALITY_GATE = 90.0  # a generator that copies training maps gets no Average, whatever its other scores


def scores(compact: dict) -> dict:
    out = {name: fn(compact) for name, _, fn in BENCHMARKS}
    vals = [v for v in out.values() if v is not None]
    gated = out["Originality"] is not None and out["Originality"] < ORIGINALITY_GATE
    out["Average"] = float(np.mean(vals)) if len(vals) == len(BENCHMARKS) and not gated else None
    out["gated"] = gated
    return out


def speed(perf: dict | None, cfg: dict, batch: int = 128) -> float | None:
    if not perf:
        return None
    g = 1.0 if cfg["guidance"] == 1.0 else 1.5  # any guidance != 1 costs the same (batch doubled)
    for r in perf["rows"]:
        if r["steps"] == cfg["steps"] and r["guidance"] == g and r["batch"] == batch and not r.get("oom"):
            return r["maps_per_second"]
    return None


def entries(tag: str, r: dict) -> dict[str, dict]:
    """Columns contributed by one results.json."""
    out: dict[str, dict] = {}
    perf = r.get("perf")
    default = r["provenance"]["default_sampler"]
    for step, c in r["checkpoints_test"]["compact"].items():
        out[f"{tag} {int(step) // 1000}k"] = {"compact": c, "speed": speed(perf, default), "kind": "checkpoint",
                                              "sampler": default}
    if "official" in r:
        o = r["official"]
        out[f"{tag} official"] = {"compact": o["compact"], "speed": speed(perf, o["sampler"]), "kind": "official",
                                  "sampler": o["sampler"], "checkpoint": o["checkpoint"]}
    return out


def fmt(v, d=1):
    return "n/a" if v is None else f"{v:.{d}f}"


def fmt_avg(s: dict) -> str:
    return "gated (copies)" if s.get("gated") else fmt(s["Average"])


def delta(new, old):
    if new is None or old is None:
        return "n/a"
    d = new - old
    rel = f" ({d / old * 100:+.0f}%)" if old else ""
    return f"{d:+.1f}{rel}"


def main() -> None:
    specs = sys.argv[1:] or ["v1=benchmarks/v1/results.json"]
    results = {s.split("=")[0]: json.loads((ROOT / s.split("=")[1]).read_text(encoding="utf-8")) for s in specs}
    cols: dict[str, dict] = {}
    for tag, r in results.items():
        cols.update(entries(tag, r))
    base = next(iter(results.values())).get("baselines", {}).get("results", {})
    base_cols = {f"baseline: {k}": {"compact": v["compact"], "speed": None, "kind": "baseline"} for k, v in base.items()}
    tags = list(results)
    officials = [f"{t} official" for t in tags if f"{t} official" in cols]
    cur = officials[-1] if officials else list(cols)[-1]
    sc = {k: scores(v["compact"]) for k, v in {**cols, **base_cols}.items()}
    names = [b[0] for b in BENCHMARKS] + ["Average"]

    lines = ["# NULLSCAPE-Bench scorecard", "",
             "Held-out TEST split, 1000 maps per half, noise-floor protocol (`docs/EVALUATION.md`). Scores are 0-100, "
             "higher is better; **100 = indistinguishable from real procedural terrain** at this sample size "
             "(fidelity scores are `100 / (distance / noise floor)`). Definitions: `benchmarks/scorecard.py`.", ""]

    # headline: current official vs previous official (or vs its own default-sampler checkpoint for v1)
    cur_c = cols[cur]
    if len(officials) >= 2:
        prev = officials[-2]
        cmp_label = f"Δ vs {prev}"
    else:
        ck = cur_c.get("checkpoint")
        prev = f"{tags[-1]} {ck // 1000}k" if ck else None
        cmp_label = f"Δ vs {prev} (default sampler)" if prev else ""
    header = ["Benchmark", "What it measures", "Real procedural", cur] + ([prev, cmp_label] if prev else []) + \
             [c for c in base_cols]
    rows = []
    for name, what, _ in BENCHMARKS + [("Average", f"mean of the 9 scores (only if Originality >= {ORIGINALITY_GATE:.0f})",
                                        None)]:
        cell = (lambda k: fmt_avg(sc[k])) if name == "Average" else (lambda k: fmt(sc[k][name]))
        row = [f"**{name}**", what, "100", f"**{cell(cur)}**"]
        if prev:
            row += [cell(prev), delta(sc[cur][name], sc[prev][name])]
        row += [cell(c) for c in base_cols]
        rows.append(row)
    sp = [cur_c["speed"]] + ([cols[prev]["speed"]] if prev else [])
    rows.append(["**Speed**", "maps/s, RTX 5060, batch 128", "-", f"**{fmt(sp[0], 2)}**"] +
                ([fmt(sp[1], 2), f"{sp[0] / sp[1]:.1f}x" if sp[0] and sp[1] else "n/a"] if prev else []) +
                ["-"] * len(base_cols))
    lines += [f"## Headline: {cur}", ""]
    if cur_c.get("sampler"):
        s = cur_c["sampler"]
        lines.append(f"Configuration: checkpoint {cur_c.get('checkpoint', '')}, {s['steps']}-step DDIM "
                     f"({s['spacing']} spacing), guidance {s['guidance']}, eta {s['eta']}.")
        lines.append("")
    lines += _table(header, rows)
    lines += ["", "Baselines are non-learned reference generators scored with the same protocol: "
                  "**blur** = real maps blurred (missing detail), **spectral_noise** = noise with each map's "
                  "roughness and height statistics (no landforms), **retrieval** = the training map with the closest "
                  "conditions (a perfect memorizer: real terrain, zero originality).", "",
              "How to read it: fidelity scores (Realism, Spectrum, Heights, Slopes) are strict, because at 1000 maps "
              "per half the real-vs-real floor is very small. Clean detail only penalizes *excess* micro-bumps, so "
              "over-smoothed output (blur) scores 100 there; always read it together with Spectrum. The Average is "
              "a tracking number, not a ranking. It is withheld for anything that copies training data.", ""]

    # progression across checkpoints
    ck_cols = [k for k, v in cols.items() if v["kind"] == "checkpoint"]
    lines += ["## Training progression (same sampler for all checkpoints: 200-step DDIM, uniform, guidance 1.5)", ""]
    first, last = ck_cols[0], ck_cols[-1]
    lines += _table(["Benchmark"] + ck_cols + [f"Δ {last} vs {first}"],
                    [[f"**{n}**"] + [fmt(sc[c][n]) for c in ck_cols] + [delta(sc[last][n], sc[first][n])] for n in names])

    # raw numbers behind the headline
    lines += ["", f"## Raw values behind the scores ({cur})", ""]
    c = cur_c["compact"]
    raw = [
        ["metric W1 mean", f"{c['model']['metric_w1_mean']:.4f}", f"{c['floor']['metric_w1_mean']:.4f}", f"{c['ratio']['metric_w1_mean']:.2f}x"],
        ["RAPSD distance (decades)", f"{c['model']['rapsd_distance']:.4f}", f"{c['floor']['rapsd_distance']:.4f}", f"{c['ratio']['rapsd_distance']:.1f}x"],
        ["height W1 (normalized)", f"{c['model']['height_w1']:.4f}", f"{c['floor']['height_w1']:.4f}", f"{c['ratio']['height_w1']:.2f}x"],
        ["slope W1 (deg)", f"{c['model']['slope_w1_deg']:.3f}", f"{c['floor']['slope_w1_deg']:.3f}", f"{c['ratio']['slope_w1_deg']:.2f}x"],
    ]
    lines += _table(["distance", "learned vs real", "real vs real (floor)", "ratio"], raw)
    extra = [
        ["adherence nMAE (mean of 5)", f"{c['adherence_nmae_mean']:.3f}"] if "adherence_nmae_mean" in c else None,
        *[[f"  {k} nMAE", f"{v:.3f}"] for k, v in c.get("adherence_nmae", {}).items()],
        ["traversability pass (learned / real)", f"{c['trav_pass_gen']:.3f} / {c['trav_pass_ref']:.3f}"],
        ["diversity (learned / real)", f"{c['diversity_gen']:.4f} / {c['diversity_ref']:.4f}"],
        ["sink density per 1000 land cells (learned / real)", f"{c['sink_density']['gen']:.2f} / {c['sink_density']['ref']:.2f}"],
        ["peak density per 1000 cells (learned / real)", f"{c['peak_density']['gen']:.2f} / {c['peak_density']['ref']:.2f}"],
        ["NN-to-train median ratio", f"{c['memorization']['nn_median_ratio']:.3f}"] if "memorization" in c else None,
    ]
    lines += [""] + _table(["quantity", "value"], [e for e in extra if e])
    out_md = ROOT / "docs" / "BENCHMARK_SCORECARD.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = ROOT / "benchmarks" / "scorecard.json"
    out_json.write_text(json.dumps({"headline": cur, "compared_to": prev, "scores": sc,
                                    "speed_maps_per_s": {k: v["speed"] for k, v in cols.items()},
                                    "definitions": {n: w for n, w, _ in BENCHMARKS}}, indent=2), encoding="utf-8")
    _figure(sc, cur, prev, list(base_cols), names[:-1], ROOT / "docs" / "figures" / "benchmark_scorecard.png")
    print(f"wrote {out_md} and {out_json}")


def _table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def _figure(sc, cur, prev, baselines, names, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [cur] + ([prev] if prev else []) + baselines
    x = np.arange(len(names))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, s in enumerate(series):
        vals = [sc[s][n] if sc[s][n] is not None else 0 for n in names]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=s,
                      color=None if i else "#d6336c", alpha=1.0 if i == 0 else 0.75)
        if i == 0:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.text(len(names) - 0.5, 101, "real procedural = 100", ha="right", va="bottom", fontsize=7)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("NULLSCAPE-Bench (TEST split)", fontsize=10)
    ax.legend(fontsize=7, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
