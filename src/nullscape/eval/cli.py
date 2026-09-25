"""``nullscape evaluate`` and ``nullscape compare``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nullscape.eval.core import HEADLINE_KEYS, evaluate_generated, split_halves


def _load(args, limit: int | None = None):
    import torch

    from nullscape.data.storage import TerrainStore
    from nullscape.inference.sampler import TerrainSampler

    sampler = TerrainSampler.from_checkpoint(args.checkpoint, use_ema=not getattr(args, "raw_weights", False))
    store = TerrainStore.open(args.dataset or sampler.meta["root"])
    if store.manifest["config_sha256"] != sampler.meta["config_sha256"]:
        print("WARNING: dataset config differs from the one the checkpoint was trained on")
    idx = store.split(args.split)
    a, b = split_halves(len(idx), seed=args.seed)
    n = min(args.n if limit is None else limit, len(a))
    a, b = idx[a[:n]], idx[b[:n]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return sampler, store, a, b, device


def _mode_inputs(mode: str, store, a: np.ndarray):
    k = len(store.condition_keys)
    if mode == "conditional":
        return dict(cond_raw=store.conditions[a], known=np.ones((len(a), k), bool), labels=store.labels[a])
    if mode == "label_only":
        return dict(cond_raw=store.conditions[a], known=np.zeros((len(a), k), bool), labels=store.labels[a])
    if mode == "unconditional":
        return dict(cond_raw=store.conditions[a], known=np.zeros((len(a), k), bool), labels=np.full(len(a), -1))
    raise ValueError(mode)


def _headline(rep: dict) -> dict:
    out = {k: rep["model_vs_ref"][k] for k in HEADLINE_KEYS}
    out.update({f"ratio_{k}": v for k, v in rep["ratio_to_floor"].items()})
    if "condition_adherence" in rep:
        out.update({f"nmae_{k}": v["nmae"] for k, v in rep["condition_adherence"].items()})
    if "memorization" in rep:
        out["nn_median_ratio"] = rep["memorization"]["nn_median_ratio"]
    out["trav_pass_gen"] = rep["model_vs_ref"]["trav_pass_rate_gen"]
    out["trav_pass_ref"] = rep["model_vs_ref"]["trav_pass_rate_ref"]
    return out


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from nullscape.metrics.distribution import metric_table, nearest_neighbor_rmse
    from nullscape.viz.render import save_grid, save_metric_histograms, save_rapsd, save_traversability_grid

    sampler, store, a, b, device = _load(args)
    world = store.world
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_a, ref_b = store.heights(a), store.heights(b)
    tables = {"a": metric_table(ref_a, world), "b": metric_table(ref_b, world)}
    train_bank = store.heights(store.split("train")) if not args.no_memorization else None
    names = store.archetypes

    report: dict = {"checkpoint": sampler.checkpoint_info, "dataset": store.root.name, "split": args.split,
                    "n_per_half": len(a), "guidance": args.guidance, "steps": args.steps, "modes": {}}
    gens: dict[str, np.ndarray] = {}
    for mode in args.modes.split(","):
        t0 = time.time()
        gen = sampler.sample(n=len(a), seed=args.seed + 1000, guidance=args.guidance, steps=args.steps,
                             **_mode_inputs(mode, store, a))
        secs = time.time() - t0
        gt = metric_table(gen, world)
        rep = evaluate_generated(
            gen, ref_a, ref_b, world,
            requested_conds=store.conditions[a] if mode == "conditional" else None,
            gen_labels=store.labels[a] if mode != "unconditional" else None,
            ref_a_labels=store.labels[a], ref_b_labels=store.labels[b], archetype_names=names,
            train_bank=train_bank if mode in ("conditional", "unconditional") else None, device=device,
            tables={**tables, "gen": gt},
        )
        rep["sampling_seconds"] = secs
        rep["sampling_ms_per_map"] = 1000 * secs / len(a)
        report["modes"][mode] = rep
        gens[mode] = gen
        tables[mode] = gt
        np.save(out / f"generated_{mode}.npy", gen)
        print(f"[{mode}] " + " ".join(f"{k}={v:.3f}" for k, v in _headline(rep).items()))

    if args.guidance_sweep:
        sweep = {}
        for g in [float(x) for x in args.guidance_sweep.split(",")]:
            gen = sampler.sample(n=len(a), seed=args.seed + 1000, guidance=g, steps=args.steps,
                                 **_mode_inputs("conditional", store, a))
            rep = evaluate_generated(gen, ref_a, ref_b, world, requested_conds=store.conditions[a],
                                     tables={k: tables[k] for k in ("a", "b")})
            sweep[str(g)] = _headline(rep)
            print(f"[guidance {g}] " + " ".join(f"{k}={v:.3f}" for k, v in sweep[str(g)].items()))
        report["guidance_sweep"] = sweep

    (out / "report.json").write_text(json.dumps(report, indent=2))
    _write_summary(report, out / "summary.md")

    # figures
    main_mode = next(iter(gens))
    save_metric_histograms({"procedural (B)": tables["b"], **{f"learned {m}": tables[m] for m in gens}},
                           out / "metric_histograms.png")
    save_rapsd({"procedural (B)": ref_b[:300], **{f"learned {m}": g[:300] for m, g in gens.items()}}, out / "rapsd.png")
    save_traversability_grid(gens[main_mode][:18], world, out / "traversability_learned.png")
    for m, g in gens.items():
        save_grid(g[:36], world, out / f"grid_{m}.png", suptitle=f"learned ({m})",
                  titles=[names[store.labels[a[i]]] if m != "unconditional" else "" for i in range(min(36, len(g)))])
    if train_bank is not None:
        q = gens[main_mode][:12]
        rmse, nn_idx = nearest_neighbor_rmse(q, train_bank, device=device)
        pairs, titles = [], []
        for i in range(len(q)):
            pairs += [q[i], train_bank[nn_idx[i]]]
            titles += [f"learned #{i}", f"nearest train rmse={rmse[i]:.3f}"]
        save_grid(pairs, world, out / "nearest_neighbors.png", titles=titles, ncols=6,
                  suptitle="memorization check: learned vs nearest training map (8 rotations/flips)")
    print(f"report -> {out / 'report.json'}")


def _write_summary(report: dict, path: Path) -> None:
    lines = [f"# Evaluation: {report['checkpoint']['path']}", "",
             f"step {report['checkpoint']['step']}, dataset `{report['dataset']}`/{report['split']}, "
             f"{report['n_per_half']} maps per half, guidance {report['guidance']}, {report['steps']} DDIM steps", ""]
    for mode, rep in report["modes"].items():
        lines += [f"## {mode}", "", "| metric | learned vs B | noise floor (A vs B) | ratio |", "|---|---|---|---|"]
        for k in HEADLINE_KEYS:
            lines.append(f"| {k} | {rep['model_vs_ref'][k]:.4f} | {rep['noise_floor'][k]:.4f} | "
                         f"{rep['ratio_to_floor'][k]:.2f} |")
        mv = rep["model_vs_ref"]
        lines += ["", f"traversability pass rate: learned {mv['trav_pass_rate_gen']:.3f}, "
                      f"procedural {mv['trav_pass_rate_ref']:.3f}; diversity (mean pairwise RMSE): learned "
                      f"{mv['diversity_gen']:.4f}, procedural {mv['diversity_ref']:.4f}", ""]
        if "condition_adherence" in rep:
            lines += ["| property | MAE | MAE / std | Pearson r | random-pair MAE / std |", "|---|---|---|---|---|"]
            base = rep["condition_adherence_random_baseline"]
            for k, v in rep["condition_adherence"].items():
                lines.append(f"| {k} | {v['mae']:.4f} | {v['nmae']:.3f} | {v['pearson_r']:.3f} | {base[k]['nmae']:.3f} |")
            lines.append("")
        if "per_archetype" in rep:
            lines += ["| archetype | metric W1 mean | floor | ratio |", "|---|---|---|---|"]
            for k, v in rep["per_archetype"].items():
                lines.append(f"| {k} | {v['metric_w1_mean']:.3f} | {v['noise_floor']:.3f} | {v['ratio']:.2f} |")
            lines.append("")
        if "memorization" in rep:
            mm = rep["memorization"]
            lines += [f"memorization: median NN-RMSE learned {mm['gen_nn_rmse_median']:.4f} vs held-out "
                      f"{mm['heldout_nn_rmse_median']:.4f} (ratio {mm['nn_median_ratio']:.2f}); "
                      f"fraction of learned maps closer than held-out 1st percentile: "
                      f"{mm['frac_gen_below_heldout_p01']:.3f}", ""]
        lines += [f"sampling: {rep['sampling_ms_per_map']:.1f} ms/map", ""]
    if "guidance_sweep" in report:
        keys = ["ratio_metric_w1_mean", "ratio_rapsd_distance"] + [k for k in next(iter(report["guidance_sweep"].values())) if k.startswith("nmae_")]
        lines += ["## guidance sweep (conditional)", "", "| guidance | " + " | ".join(keys) + " |",
                  "|---" * (len(keys) + 1) + "|"]
        for g, h in report["guidance_sweep"].items():
            lines.append(f"| {g} | " + " | ".join(f"{h[k]:.3f}" for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmd_compare(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    from nullscape.metrics.quality import compute_metrics
    from nullscape.viz.render import _save, draw_heightmap, draw_surface

    sampler, store, a, _, _ = _load(args, limit=10**9)
    world = store.world
    names = store.archetypes
    out = Path(args.out)
    # one reference map per archetype (cycling), each regenerated from its measured properties + label
    picks = []
    for k in range(args.n):
        cand = a[store.labels[a] == k % len(names)]
        picks.append(cand[(k // len(names)) % len(cand)] if len(cand) else a[k])
    picks = np.array(picks)
    ref = store.heights(picks)
    s = args.seeds
    gen = sampler.sample(n=len(picks) * s, seed=args.seed,
                         cond_raw=np.repeat(store.conditions[picks], s, 0),
                         known=np.ones((len(picks) * s, len(store.condition_keys)), bool),
                         labels=np.repeat(store.labels[picks], s), guidance=args.guidance, steps=args.steps)
    gen = gen.reshape(len(picks), s, *gen.shape[1:])

    def label(h):
        m = compute_metrics(h, world)
        return f"relief {m['relief'] * world.max_height_m:.0f}m slope {m['mean_slope_deg']:.0f}\u00b0 water {m['water_fraction']:.2f}"

    fig, axes = plt.subplots(len(picks), s + 1, figsize=(2.3 * (s + 1), 2.4 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        draw_heightmap(axes[r, 0], ref[r], world, f"PROCEDURAL {names[store.labels[i]]}\n{label(ref[r])}")
        for j in range(s):
            draw_heightmap(axes[r, j + 1], gen[r, j], world, f"LEARNED seed {j}\n{label(gen[r, j])}")
    fig.suptitle("Procedural (left) vs learned terrain conditioned on the procedural map's measured properties",
                 fontsize=10)
    _save(fig, out / "compare_2d.png")

    rows = min(len(picks), args.n_3d)
    fig = plt.figure(figsize=(9, 3.6 * rows))
    for r in range(rows):
        ax = fig.add_subplot(rows, 2, 2 * r + 1, projection="3d")
        draw_surface(ax, ref[r], world)
        ax.set_title(f"procedural {names[store.labels[picks[r]]]}", fontsize=8)
        ax = fig.add_subplot(rows, 2, 2 * r + 2, projection="3d")
        draw_surface(ax, gen[r, 0], world)
        ax.set_title("learned", fontsize=8)
    _save(fig, out / "compare_3d.png")
    print(f"figures -> {out}")


def add_eval_commands(sub) -> None:
    e = sub.add_parser("evaluate", help="quantitative evaluation of a checkpoint against held-out data")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset", default=None, help="defaults to the checkpoint's training dataset")
    e.add_argument("--split", default="val", choices=["val", "test"])
    e.add_argument("--n", type=int, default=1000, help="maps per half (A and B)")
    e.add_argument("--modes", default="conditional,label_only,unconditional")
    e.add_argument("--guidance", type=float, default=1.5)
    e.add_argument("--guidance-sweep", default="", help="comma-separated guidance scales, e.g. 1,1.5,2,3")
    e.add_argument("--steps", type=int, default=50)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--no-memorization", action="store_true")
    e.add_argument("--out", required=True)
    e.set_defaults(func=_cmd_evaluate)

    c = sub.add_parser("compare", help="side-by-side procedural vs learned figures")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--dataset", default=None)
    c.add_argument("--split", default="val", choices=["val", "test"])
    c.add_argument("--n", type=int, default=6, help="reference maps (cycled over archetypes)")
    c.add_argument("--n-3d", type=int, default=3)
    c.add_argument("--seeds", type=int, default=3, help="learned samples per reference map")
    c.add_argument("--guidance", type=float, default=1.5)
    c.add_argument("--steps", type=int, default=50)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_compare)
