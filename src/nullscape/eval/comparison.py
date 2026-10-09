from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from nullscape.inference.sampler import DEFAULT_SAMPLING, sampling_config
from nullscape.metrics.distribution import mean_log_rapsd, metric_table
from nullscape.metrics.quality import CONDITION_KEYS, slope_degrees
from nullscape.utils.tracking import atomic_file, atomic_json
from nullscape.world import WorldSpec


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_evaluation(folder: str | Path) -> dict:
    folder = Path(folder)
    if not (folder / "report.json").is_file():
        matches = list(folder.glob("step_*/report.json"))
        if len(matches) != 1:
            raise ValueError(f"choose one evaluation directory containing report.json: {folder}")
        folder = matches[0].parent
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if "conditional" not in report.get("modes", {}):
        raise ValueError("comparison needs fully conditioned evaluations with matched requests")
    if "comparison_inputs" in report:
        inputs = report["comparison_inputs"]
        world = WorldSpec.from_dict(report["world"])
        maps = np.load(folder / "generated_conditional.npy", allow_pickle=False)
        a, b = report["reference_indices_a"], report["reference_indices_b"]
        provenance = "recorded evaluation requests"
    elif "spec" in report and (folder / "generated_eval.npy").is_file():
        from nullscape.data.storage import TerrainStore
        from nullscape.eval.artifacts import ArtifactSpec, eval_halves
        from nullscape.utils.seed import derive_seed

        store = TerrainStore.open(report["dataset"])
        spec = ArtifactSpec(**report["spec"])
        ai, bi = eval_halves(store, spec)
        table = metric_table(store.heights(bi), store.world)
        inputs = {"config_sha256": store.manifest["config_sha256"], "condition_keys": store.condition_keys,
                  "conditions": store.conditions[ai].tolist(), "labels": store.labels[ai].tolist(),
                  "archetypes": store.archetypes, "seeds": [derive_seed(spec.seed + 1, i) for i in range(len(ai))],
                  "reference_std": [float(table[k].std()) for k in store.condition_keys]}
        world, a, b = store.world, ai.tolist(), bi.tolist()
        maps = np.load(folder / "generated_eval.npy", allow_pickle=False)
        provenance = "legacy ArtifactSpec reconstructed from the frozen dataset"
    else:
        raise ValueError("evaluation lacks recorded requests; rerun evaluate or use official generated_eval artifacts")
    cfg = sampling_config(**{k: report[k] for k in DEFAULT_SAMPLING if k in report})
    n, r = len(inputs["seeds"]), world.resolution
    if n < 2 or maps.shape != (n, r, r) or not np.isfinite(maps).all() or ((maps < 0) | (maps > 1)).any():
        raise ValueError("evaluation maps must be finite [N,R,R] arrays in [0,1] matching the requests")
    if inputs["condition_keys"] != list(CONDITION_KEYS):
        raise ValueError("comparison condition keys do not match the metric protocol")
    return {"folder": str(folder.resolve()), "report": report, "maps": maps, "world": world,
            "inputs": inputs, "identity": {"world": world.to_dict(), "inputs": inputs, "a": a, "b": b,
                                            "split": report["split"], "sampling": cfg}, "provenance": provenance}


def headline_metrics(report: dict) -> dict:
    rep = report["modes"]["conditional"]
    mv = rep["model_vs_ref"]
    values = {key: rep["ratio_to_floor"][key] for key in ("metric_w1_mean", "rapsd_distance", "height_w1", "slope_w1_deg")}
    adherence = rep.get("condition_adherence", {})
    values["control_nmae"] = float(np.mean([v["nmae"] for v in adherence.values()])) if adherence else None
    values["traversability"] = 100 * mv["trav_pass_rate_gen"] if "trav_pass_rate_gen" in mv else None
    values["diversity_ratio"] = mv["diversity_gen"] / max(mv["diversity_ref"], 1e-12) if "diversity_gen" in mv else None
    values["nn_ratio"] = rep.get("memorization", {}).get("nn_median_ratio")
    return values


def _slope_histogram(maps, world):
    edges = np.linspace(0, 90, 31)
    counts = sum((np.histogram(slope_degrees(h, world), edges)[0] for h in maps), np.zeros(30))
    return (counts / max(counts.sum(), 1)).tolist()


def build_comparison(baseline: str | Path, candidate: str | Path, out: str | Path, *,
                     baseline_name: str = "Baseline", candidate_name: str = "Candidate", worst: int = 20) -> dict:
    if not 1 <= worst <= 100:
        raise ValueError("worst must be between 1 and 100")
    left, right = load_evaluation(baseline), load_evaluation(candidate)
    if json.dumps(left["identity"], sort_keys=True) != json.dumps(right["identity"], sort_keys=True):
        raise ValueError("comparison requires matched dataset, split, reference halves, properties, seeds and sampler settings")
    world, inputs = left["world"], left["inputs"]
    lm, rm = left["maps"], right["maps"]
    lt, rt = metric_table(lm, world), metric_table(rm, world)
    req = np.asarray(inputs["conditions"], dtype=float)
    std = np.maximum(np.asarray(inputs["reference_std"], dtype=float), 1e-12)
    measured = [np.stack([t[k] for k in CONDITION_KEYS], axis=1) for t in (lt, rt)]
    if req.shape != (len(lm), len(CONDITION_KEYS)) or not np.isfinite(req).all():
        raise ValueError("matched condition requests have an invalid shape or non-finite values")
    le, re = [np.abs(m - req) / std for m in measured]
    le, re = le.mean(1), re.mean(1)
    worst_ids = np.argsort(-re, kind="stable")[:worst].tolist()
    regressed_ids = [int(i) for i in np.argsort(-(re - le), kind="stable") if re[i] > le[i]][:worst]
    labels = np.asarray(inputs["labels"])
    representative = [int(i) for c in np.unique(labels) for i in np.flatnonzero(labels == c)[:2]]
    selected = sorted(set(representative + worst_ids + regressed_ids))

    def encode(h):
        return base64.b64encode(np.rint(h * 65535).astype("<u2").tobytes()).decode("ascii")

    cases = [{"id": i, "source_index": left["identity"]["a"][i], "seed": str(inputs["seeds"][i]),
              "archetype": inputs["archetypes"][labels[i]], "requested": req[i].tolist(),
              "baseline_measured": measured[0][i].tolist(), "candidate_measured": measured[1][i].tolist(),
              "baseline_error": float(le[i]), "candidate_error": float(re[i]), "regression": float(re[i] - le[i]),
              "baseline": encode(lm[i]), "candidate": encode(rm[i])} for i in selected]
    lh, rh = headline_metrics(left["report"]), headline_metrics(right["report"])
    headline = {key: {"baseline": lh[key], "candidate": rh[key],
                      "delta": rh[key] - lh[key] if rh[key] is not None and lh[key] is not None else None} for key in lh}
    k, ls = mean_log_rapsd(lm)
    _, rs = mean_log_rapsd(rm)
    result = _clean({"schema_version": 1, "baseline_name": baseline_name, "candidate_name": candidate_name,
                    "split": left["identity"]["split"], "n": len(lm), "world": world.to_dict(),
                    "sampling": left["identity"]["sampling"], "condition_keys": list(CONDITION_KEYS),
                    "sources": [{"directory": item["folder"], "checkpoint": item["report"]["checkpoint"],
                                 "provenance": item["provenance"]} for item in (left, right)],
                    "headline": headline, "spectrum": {"k": k, "baseline": ls, "candidate": rs},
                    "slopes": {"centers": np.linspace(1.5, 88.5, 30), "baseline": _slope_histogram(lm, world),
                               "candidate": _slope_histogram(rm, world)},
                    "per_archetype": {name: item["report"]["modes"]["conditional"].get("per_archetype", {})
                                      for name, item in (("baseline", left), ("candidate", right))},
                    "representative": representative, "worst_adherence": worst_ids, "worst_regressions": regressed_ids,
                    "cases": cases, "selection_policy": "Review only. No checkpoint selection or promotion occurs here.",
                    "ranking": "Mean absolute requested-property error divided by reference standard deviation; not visual realism."})
    payload = json.dumps(result, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("comparison.html").read_text(encoding="utf-8")
    html = template.replace("{{COMPARISON_DATA}}", payload)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "comparison.json", result)
    with atomic_file(out / "index.html") as file:
        file.write(html.encode("utf-8"))
    return result


def _cmd_comparison(args):
    report = build_comparison(args.baseline, args.candidate, args.out, baseline_name=args.baseline_name,
                              candidate_name=args.candidate_name, worst=args.worst)
    print(f"Matched {report['n']} {report['split']} maps. Open {Path(args.out) / 'index.html'}")


def add_comparison_command(sub):
    parser = sub.add_parser("comparison-report", help="offline paired-model report from cached evaluations; no model sampling")
    parser.add_argument("--baseline", required=True, help="evaluate output or official artifact step directory")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-name", default="Baseline")
    parser.add_argument("--candidate-name", default="Candidate")
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.set_defaults(func=_cmd_comparison)
