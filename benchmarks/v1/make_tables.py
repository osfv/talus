"""Render markdown tables from benchmarks/v1/results.json -> benchmarks/v1/tables.md.

The report (docs/V1_BENCHMARK.md) quotes these tables, so numbers are never hand-copied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
KEYS = ["mean_elevation", "relief", "mean_slope_deg", "water_fraction", "spectral_beta"]
ALL_ARCH = ["plains", "hills", "mountains", "ridges", "islands", "mesas"]


def f(x, d=3):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def main() -> None:
    parts = []
    ct = R["checkpoints_test"]["compact"]
    first = next(iter(ct.values()))
    ARCH = [a for a in ALL_ARCH if a in first.get("per_archetype_ratio", {})]
    cv = R.get("checkpoints_val", {}).get("compact", {})
    steps = list(ct)

    parts.append("### Checkpoints (TEST, 1000 maps/half, 200-step DDIM uniform, guidance 1.5)\n")
    parts.append(table(["step", "metric W1 ratio", "RAPSD ratio", "height W1 ratio", "slope W1 ratio",
                        "RAPSD bias (dec)", "slope W1 excess (deg)", "adherence nMAE", "trav pass gen/ref",
                        "diversity gen/ref", "NN ratio"],
                       [[s, f(c["ratio"]["metric_w1_mean"], 2), f(c["ratio"]["rapsd_distance"], 1),
                         f(c["ratio"]["height_w1"], 2), f(c["ratio"]["slope_w1_deg"], 2), f(c["rapsd_bias_decades"]),
                         f(c["excess_over_floor"]["slope_w1_deg"], 2), f(c["adherence_nmae_mean"]),
                         f"{c['trav_pass_gen']:.3f}/{c['trav_pass_ref']:.3f}",
                         f"{c['diversity_gen']:.4f}/{c['diversity_ref']:.4f}",
                         f(c["memorization"]["nn_median_ratio"], 3)] for s, c in ct.items()]))
    fl = ct[steps[0]]["floor"]
    parts.append(f"Noise floor (A vs B, TEST): metric W1 {fl['metric_w1_mean']:.4f}, RAPSD {fl['rapsd_distance']:.4f} "
                 f"decades, height W1 {fl['height_w1']:.4f}, slope W1 {fl['slope_w1_deg']:.3f} deg.\n")

    if "analysis" in R:
        ci = R["analysis"]["bootstrap"]["ci"]
        pb = R["analysis"]["bootstrap"]["prob_best_metric_w1"]
        parts.append("\n### Bootstrap 95% CIs (TEST)\n")
        parts.append(table(["step", "metric W1 ratio [95% CI]", "RAPSD ratio [95% CI]", "P(best metric W1)"],
                           [[s, f"{ci[s]['w1']['median']:.2f} [{ci[s]['w1']['lo95']:.2f}, {ci[s]['w1']['hi95']:.2f}]",
                             f"{ci[s]['rapsd']['median']:.1f} [{ci[s]['rapsd']['lo95']:.1f}, {ci[s]['rapsd']['hi95']:.1f}]",
                             f(pb[s], 2)] for s in steps]))

    parts.append("\n### Artifacts and extrema (TEST means)\n")
    parts.append(table(["step", "checkerboard gen/ref", "hf_energy gen/ref", "peak density gen/ref", "sink density gen/ref"],
                       [[s, f"{c['artifacts']['checkerboard_mean_gen']:.4f}/{c['artifacts']['checkerboard_mean_ref']:.4f}",
                         f"{c['artifacts']['hf_energy_mean_gen']:.4f}/{c['artifacts']['hf_energy_mean_ref']:.4f}",
                         f"{c['peak_density']['gen']:.2f}/{c['peak_density']['ref']:.2f}",
                         f"{c['sink_density']['gen']:.2f}/{c['sink_density']['ref']:.2f}"] for s, c in ct.items()]))

    parts.append("\n### Conditioning adherence per property (TEST, nMAE = MAE / reference std; r = Pearson)\n")
    parts.append(table(["step"] + [f"{k} nMAE / r" for k in KEYS],
                       [[s] + [f"{c['adherence_nmae'][k]:.3f} / {c['adherence_r'][k]:.3f}" for k in KEYS]
                        for s, c in ct.items()]))

    parts.append("\n### Per-archetype metric W1 ratio (TEST)\n")
    parts.append(table(["step"] + ARCH, [[s] + [f(c["per_archetype_ratio"][a], 2) for a in ARCH] for s, c in ct.items()]))

    if cv:
        parts.append("\n### Checkpoints on VAL (selection only, 1000 maps/half)\n")
        parts.append(table(["step", "metric W1 ratio", "RAPSD ratio", "height W1 ratio", "slope W1 ratio", "adherence nMAE"],
                           [[s, f(c["ratio"]["metric_w1_mean"], 2), f(c["ratio"]["rapsd_distance"], 1),
                             f(c["ratio"]["height_w1"], 2), f(c["ratio"]["slope_w1_deg"], 2), f(c["adherence_nmae_mean"])]
                            for s, c in cv.items()]))
    if "selection" in R:
        parts.append(f"\nSelected checkpoint: **{R['selection']['checkpoint']}** ({R['selection']['reason']}).\n")

    if "pareto" in R:
        p = R["pareto"]
        parts.append("\n### Sampler grid (VAL, 500 maps/half, selected checkpoint; cost = isolated ms/map at batch 128)\n")
        rows = sorted(p["points"], key=lambda q: q["iso_seconds_per_map_b128"])
        val = R["sampler_val"]["rows"]
        parts.append(table(["config", "ms/map", "maps/s", "metric W1 ratio", "RAPSD ratio", "slope W1 ratio",
                            "adherence nMAE", "hf_energy gen", "sink density gen", "frontier"],
                           [[q["name"], f(q["iso_seconds_per_map_b128"] * 1000, 1), f(1 / q["iso_seconds_per_map_b128"], 1),
                             f(q["val_metric_w1_ratio"], 2), f(q["val_rapsd_ratio"], 1),
                             f(val[q["name"]]["compact"]["ratio"]["slope_w1_deg"], 2), f(q["val_adherence_nmae_mean"]),
                             f(val[q["name"]]["compact"]["artifacts"]["hf_energy_mean_gen"], 4),
                             f(val[q["name"]]["compact"]["sink_density"]["gen"], 2),
                             "yes" if q["name"] in p["frontier"] else ""] for q in rows]))
        ref = val[rows[0]["name"]]["compact"]
        parts.append(f"VAL reference: hf_energy {ref['artifacts']['hf_energy_mean_ref']:.4f}, sink density "
                     f"{ref['sink_density']['ref']:.2f}. Recommended: **{p['recommended']}**; best quality: "
                     f"{p['best_quality']}.\n")
    if "sampler_test" in R:
        st = R["sampler_test"]["rows"]
        base = ct[str(R["selection"]["checkpoint"])]
        parts.append("\n### Sampler confirmation on TEST (1000 maps/half, selected checkpoint)\n")
        rows = [["s200_u_g1.5_e0 (default)", base]] + [[n, r["compact"]] for n, r in st.items()]
        parts.append(table(["config", "metric W1 ratio", "RAPSD ratio", "height W1 ratio", "slope W1 ratio",
                            "adherence nMAE", "trav pass gen", "hf_energy gen", "sink density gen"],
                           [[n, f(c["ratio"]["metric_w1_mean"], 2), f(c["ratio"]["rapsd_distance"], 1),
                             f(c["ratio"]["height_w1"], 2), f(c["ratio"]["slope_w1_deg"], 2), f(c["adherence_nmae_mean"]),
                             f(c["trav_pass_gen"]), f(c["artifacts"]["hf_energy_mean_gen"], 4),
                             f(c["sink_density"]["gen"], 2)] for n, c in rows]))

    if "perf" in R:
        pf = R["perf"]
        parts.append("\n### Inference performance (isolated, RTX 5060, bf16)\n")
        parts.append(table(["steps", "guidance", "batch", "maps/s", "s/map", "latency/batch (s)", "peak alloc MB",
                            "nvidia-smi MB"],
                           [[r["steps"], r["guidance"], r["batch"], "OOM under cap", "", "", "", ""] if r.get("oom") else
                            [r["steps"], r["guidance"], r["batch"], f(r["maps_per_second"], 2), f(r["seconds_per_map"], 4),
                             f(r["latency_seconds_per_batch"], 2), f(r["peak_alloc_mb"], 0), f(r["nvidia_smi_used_mb"], 0)]
                            for r in pf["rows"]]))
        d = pf["determinism"]
        parts.append(f"GPU determinism: same seed + batch max |diff| {d['same_seed_same_batch_max_abs_diff']:.2e}; "
                     f"batch 16 vs 1 max |diff| {d['batch16_vs_batch1_max_abs_diff']:.2e} "
                     f"({d['batch16_vs_batch1_max_abs_diff_m']:.3f} m). Checkpoint load {pf['checkpoint_load_seconds']:.1f} s. "
                     f"CPU metrics: compute_metrics {pf['cpu_seconds_per_map']['compute_metrics'] * 1000:.1f} ms/map, "
                     f"traversability {pf['cpu_seconds_per_map']['traversability'] * 1000:.1f} ms/map. Peak process RSS "
                     f"{f(pf['peak_process_rss_mb'], 0)} MB.\n")
    if "training" in R:
        parts.append("\n### Training throughput (from run logs, batch 48)\n")
        parts.append(table(["run", "median it/s", "steady median it/s", "p10-p90 it/s", "maps/s", "slow intervals",
                            "wall h"],
                           [[k, f(v["it_per_s_median"], 2), f(v["it_per_s_steady_median"], 2),
                             f"{v['it_per_s_p10']:.2f}-{v['it_per_s_p90']:.2f}", f(v["maps_per_s_steady"], 0),
                             f(v["slow_intervals_fraction"], 3), f(v["wall_hours_first_to_last_log"], 2)]
                            for k, v in R["training"].items()]))

    if "analysis" in R:
        an = R["analysis"]
        sel = str(R["selection"]["checkpoint"])
        per = an["archetypes"][sel]["per_archetype"]
        parts.append(f"\n### Archetype failure modes (TEST, step {sel})\n")
        parts.append(table(["archetype", "metric W1 ratio", "RAPSD ratio", "slope W1 ratio", "adherence nMAE",
                            "trav pass gen/ref", "hf_energy gen/ref", "sink gen/ref", "worst metrics"],
                           [[a, f(per[a]["metric_w1_ratio"], 2), f(per[a]["rapsd_ratio"], 1), f(per[a]["slope_w1_ratio"], 2),
                             f(per[a]["adherence_nmae_mean"]), f"{per[a]['trav_pass_gen']:.2f}/{per[a]['trav_pass_ref']:.2f}",
                             f(per[a]["artifacts"]["hf_energy"]["gen_over_ref"], 2),
                             f(per[a]["artifacts"]["sink_density"]["gen_over_ref"], 2),
                             ", ".join(per[a]["worst_metrics"][:3])] for a in ARCH]))
        rk = an["archetypes"][sel]["rankings"]
        loo = an["archetypes"][sel]["leave_one_out"]
        sh = an["archetypes"][sel]["shares"]
        parts.append("\nLargest contributors (leave-one-archetype-out deltas / error shares):\n")
        parts.append(table(["failure", "rank 1", "rank 2", "rank 3"],
                           [[k, *v[:3]] for k, v in rk.items()]))
        parts.append(table(["archetype", "RAPSD delta (dec)", "slope W1 delta (deg)", "adherence error share",
                            "trav excess failure rate", "hf_energy excess share", "sink excess share"],
                           [[a, f(loo[a]["rapsd_delta"], 4), f(loo[a]["slope_w1_delta_deg"], 3),
                             f(sh[a]["adherence_error_share"], 3), f(sh[a]["trav_excess_failure_rate"], 3),
                             f(sh[a]["hf_energy_excess_share"], 3), f(sh[a]["sink_density_excess_share"], 3)] for a in ARCH]))
        sb = an["spectrum_bands"]
        bands = list(sb[sel]["bands"])
        parts.append("\n### Spectrum error by frequency band (mean log10 power learned - procedural, TEST)\n")
        parts.append(table(["step"] + bands, [[s] + [f(sb[s]["bands"][b]["mean_diff_decades"]) for b in bands]
                                              for s in steps] +
                           [["floor A-B"] + [f(sb["floor_A_minus_B"]["bands"][b]["mean_diff_decades"]) for b in bands]]))
        parts.append(table(["archetype"] + bands, [[a] + [f(sb[sel]["per_archetype_bands"][a][b]) for b in bands]
                                                   for a in ARCH]))
        ab = an["adherence_bins"][sel]
        parts.append(f"\n### Adherence by requested-value bin (TEST, step {sel}; bias in std units)\n")
        rows = []
        for k in KEYS:
            for lab, v in ab[k].items():
                rows.append([k, lab, v["n"], f"{v['req_range'][0]:.3f}-{v['req_range'][1]:.3f}", f(v["nmae"]),
                             f"{v['bias_std']:+.3f}"])
        parts.append(table(["property", "bin", "n", "requested range", "nMAE", "bias"], rows))
        mm = an["memorization"]
        parts.append(f"\n### Memorization (TEST, step {mm['checkpoint']}, 8 rotations/flips vs 45,000 training maps)\n")
        parts.append(table(["set"] + list(mm["gen"]), [["generated"] + [f(v, 4) for v in mm["gen"].values()],
                                                       ["held-out test"] + [f(v, 4) for v in mm["heldout"].values()]]))
        parts.append(f"nn_median_ratio {mm['nn_median_ratio']:.3f}; generated below held-out p1: "
                     f"{mm['frac_gen_below_heldout_p1']:.3f}; below held-out minimum: {mm['frac_gen_below_heldout_min']:.3f} "
                     f"(generated min {mm['gen_min']:.4f} vs held-out min {mm['heldout_min']:.4f}).\n")
        parts.append(table(["archetype", "gen median", "held-out median", "ratio", "gen below held-out p1"],
                           [[a, f(v["gen_median"], 4), f(v["heldout_median"], 4), f(v["ratio"], 3),
                             f(v["frac_gen_below_heldout_p1"], 3)] for a, v in mm["per_archetype"].items()]))
        q = an.get("quantization_sensitivity")
        if q:
            parts.append(f"\nQuantization sensitivity (step {q['checkpoint']}): float ratios "
                         f"{ {k: round(v, 2) for k, v in q['float'].items()} } vs uint16 "
                         f"{ {k: round(v, 2) for k, v in q['uint16'].items()} }; sink density gen float "
                         f"{q['sink_density_gen_float']:.2f} / uint16 {q['sink_density_gen_uint16']:.2f} / ref "
                         f"{q['sink_density_ref']:.2f}.\n")

    if "stress" in R:
        s = R["stress"]
        parts.append(f"\n### Conditioning stress tests (step {s['checkpoint']}, sampler {s['sampler']})\n")
        parts.append("Single property (only that property given, archetype unknown; 64 maps per case):\n\n")
        parts.append(table(["property", "level", "requested", "measured mean ± std", "bias", "nMAE",
                            "largest other-property deviation from natural (std)"],
                           [[v["key"], v["level"], f(v["requested"]), f"{v['measured_mean']:.3f} ± {v['measured_std']:.3f}",
                             f"{v['bias']:+.3f}", f(v["nmae"]),
                             max(v["others_vs_natural_z"].items(), key=lambda kv: abs(kv[1]))[0] + " " +
                             f"{max(v['others_vs_natural_z'].values(), key=abs):+.2f}"] for v in s["single"].values()]))
        parts.append("\nPairs at extreme percentiles (32 maps per case):\n\n")
        parts.append(table(["case", "train support", "nMAE a (alone)", "nMAE b (alone)"],
                           [[k, v["train_support"], f"{v['a_nmae']:.3f} ({v['a_nmae_alone']:.3f})",
                             f"{v['b_nmae']:.3f} ({v['b_nmae_alone']:.3f})"] for k, v in s["pairs"].items()]))
        parts.append("\nPartial conditioning from real TEST maps (nMAE of known properties by number known):\n\n")
        pk = s["partial_nmae_by_known_count"]
        parts.append(table(["# known"] + KEYS, [[m] + [f(v.get(k)) for k in KEYS] for m, v in pk.items()]))
        parts.append("\nWithin-archetype control gain (request p10 vs p90 of that archetype, label known; 1 = perfect):\n\n")
        g = s["archetype_gain"]
        parts.append(table(["archetype"] + KEYS, [[a] + [f(g[f"{a}/{k}"]["gain"], 2) for k in KEYS] for a in ARCH]))

    if "engineering" in R:
        e = R["engineering"]
        parts.append("\n### Engineering\n")
        pt = e["pytest"]
        parts.append(f"pytest: {pt['tests']} tests, {pt['failures']} failures, {pt['errors']} errors, {pt['time']:.1f} s.\n\n")
        parts.append(table(["command", "exit", "seconds"],
                           [[c["cmd"][:90].replace("|", "/"), c["exit_code"], c["seconds"]] for c in e["cli"]["commands"]
                            if "--help" not in c["cmd"]]))
        d = e["dataset"]
        parts.append(f"\nDataset: manifest hash matches {d['manifest_config_sha256_matches']}; splits disjoint/covering "
                     f"{d['splits_disjoint_and_covering']}; regeneration bit-exact {d['regeneration']['bit_exact']}/"
                     f"{d['regeneration']['n']} (labels {d['regeneration']['labels_match']}/{d['regeneration']['n']}).\n")
        ex = e["exports"]["native"]
        parts.append(f"Exports: PNG16 exact {ex['png16_roundtrip_exact']}, R16 exact {ex['r16_roundtrip_exact']} "
                     f"({ex['r16_bytes']} bytes), NPY exact {ex['npy_exact']}, OBJ v/vn/f {ex['obj_counts']['v']}/"
                     f"{ex['obj_counts']['vn']}/{ex['obj_counts']['f']} (expected {ex['obj_counts']['expected_v']}/"
                     f"{ex['obj_counts']['expected_v']}/{ex['obj_counts']['expected_f']}), OBJ height error "
                     f"{ex['obj_height_max_abs_err_m']:.4f} m.\n\n")
        parts.append(table(["input", "unity_size", "already 2^n+1", "upsampled although valid"],
                           [[k, v["unity_size"], v["input_already_valid_unity_size"], v["upsampled_although_valid"]]
                            for k, v in e["exports"]["unity_size_table"].items()]))
    (OUT / "tables.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
