"""Print an eval JSON (in-training eval/step_*.json or a mode of evaluate's report.json) as a table.

usage: python scripts/show_eval.py <path.json> [--mode conditional]
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mode", default="conditional")
    args = ap.parse_args()
    r = json.load(open(args.path, encoding="utf-8"))
    if "modes" in r:
        r = r["modes"][args.mode]
    w = r["model_vs_ref"]["metric_w1"]
    fw = r["noise_floor"]["metric_w1"]
    mm = r["metric_means"]
    print(f"{'metric':30s} {'W1n':>7s} {'floor':>7s} {'gen mean':>10s} {'ref mean':>10s}")
    for k in sorted(w, key=lambda k: -w[k]):
        print(f"{k:30s} {w[k]:7.3f} {fw[k]:7.3f} {mm[k]['gen']:10.4f} {mm[k]['ref']:10.4f}")
    print("\nratio_to_floor:", {k: round(v, 3) for k, v in r["ratio_to_floor"].items()})
    if "condition_adherence" in r:
        print("adherence nmae:", {k: round(v["nmae"], 3) for k, v in r["condition_adherence"].items()})
        print("random   nmae:", {k: round(v["nmae"], 3) for k, v in r["condition_adherence_random_baseline"].items()})
    if "per_archetype" in r:
        print("per archetype ratio:", {k: round(v["ratio"], 2) for k, v in r["per_archetype"].items()})
    if "memorization" in r:
        print("memorization:", {k: round(v, 4) for k, v in r["memorization"].items()})


if __name__ == "__main__":
    main()
