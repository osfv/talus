"""``nullscape evaluate`` and ``nullscape compare``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nullscape.eval.core import HEADLINE_KEYS, evaluate_generated, gameplay_summary, split_halves, summarize_repeats
from nullscape.metrics.quality import CONDITION_KEYS


def _load(args, limit: int | None = None):
    from nullscape.cli_model import _runtime_device
    from nullscape.data.storage import TerrainStore
    from nullscape.inference.sampler import TerrainSampler, sampling_config

    if hasattr(args, "steps"):
        sampling_config(**_sampler_kwargs(args))
    device = _runtime_device(args)
    sampler = TerrainSampler.from_checkpoint(args.checkpoint, device=device,
                                             use_ema=not getattr(args, "raw_weights", False))
    store = TerrainStore.open(args.dataset or sampler.meta["root"])
    if store.manifest["config_sha256"] != sampler.meta["config_sha256"]:
        print("WARNING: dataset config differs from the one the checkpoint was trained on")
    idx = store.split(args.split)
    a, b = split_halves(len(idx), seed=args.seed)
    n = min(args.n if limit is None else limit, len(a))
    a, b = idx[a[:n]], idx[b[:n]]
    if len(a) < 2:
        raise ValueError("evaluation needs at least two maps per half")
    return sampler, store, a, b, device


def _mode_inputs(mode: str, store, a: np.ndarray):
    known = np.zeros((len(a), len(store.condition_keys)), bool)
    label_known = mode in ("conditional", "label_only") or mode.startswith("label+property:")
    if mode == "conditional":
        known[:] = True
    elif mode.startswith(("property:", "label+property:")):
        key = mode.split(":", 1)[1]
        if key not in store.condition_keys:
            raise ValueError(f"unknown conditioning property {key!r}")
        known[:, list(store.condition_keys).index(key)] = True
    elif mode not in ("label_only", "unconditional"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    return dict(cond_raw=np.where(known, store.conditions[a], 0).astype(np.float32), known=known,
                labels=store.labels[a] if label_known else np.full(len(a), -1))


def _mode_slug(mode: str) -> str:
    return mode.replace(":", "_").replace("+", "_")


def _sampler_kwargs(args: argparse.Namespace, sampler=None) -> dict:
    cfg = {"guidance": args.guidance, "steps": args.steps, "eta": args.eta, "spacing": args.spacing,
           "guidance_interval": args.guidance_interval}
    return sampler.resolve_sampling(**cfg) if sampler is not None else cfg


def _add_sampler_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--guidance", type=float, default=None, help="default: checkpoint sampling preset")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--spacing", default=None, choices=["uniform", "quadratic"])
    p.add_argument("--guidance-interval", type=float, nargs=2, metavar=("MIN_T", "MAX_T"),
                   help="guided normalized noise timestep range [0,1]; outside uses guidance 1")


def _cmd_sampler_sweep(args: argparse.Namespace) -> None:
    """Grid over sampler settings (conditional mode) to choose inference defaults on validation data."""
    import itertools

    from nullscape.metrics.distribution import metric_table

    sampler, store, a, b, _ = _load(args)
    world = store.world
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_a, ref_b = store.heights(a), store.heights(b)
    tables = {"a": metric_table(ref_a, world), "b": metric_table(ref_b, world)}
    if args.split != "val":
        raise ValueError("sampler selection must use validation data, not TEST")
    if not 0 <= args.max_regression <= 1:
        raise ValueError("max-regression must be between 0 and 1")
    intervals = [tuple(float(v) for v in part.split(":")) for part in args.guidance_intervals.split(",")]
    grid = list(itertools.product(args.steps_list.split(","), args.spacings.split(","), args.etas.split(","),
                                  args.guidances.split(","), intervals))
    configs = [sampler.resolve_sampling(steps=int(st), spacing=sp, eta=float(et), guidance=float(g),
                                        guidance_interval=interval) for st, sp, et, g, interval in grid]
    rows = []
    for skw in configs:
        t0 = time.time()
        gen = sampler.sample(n=len(a), seed=args.seed + 1000, batch_size=args.batch_size,
                             **skw, **_mode_inputs("conditional", store, a))
        secs = time.time() - t0
        rep = evaluate_generated(gen, ref_a, ref_b, world, requested_conds=store.conditions[a], tables=tables,
                                 gen_labels=store.labels[a], ref_a_labels=store.labels[a], ref_b_labels=store.labels[b],
                                 archetype_names=store.archetypes)
        mv = rep["model_vs_ref"]
        row = {**skw, **_headline(rep), "ms_per_map": 1000 * secs / len(a),
               "per_archetype": rep.get("per_archetype", {}),
               "diversity_ratio": mv["diversity_gen"] / max(mv["diversity_ref"], 1e-12),
               **{k: v for k, v in rep["artifacts"].items() if k.endswith("_gen")}}
        rows.append(row)
        print(f"{skw}: realism={row['ratio_metric_w1_mean']:.3f} spectrum={row['ratio_rapsd_distance']:.3f}", flush=True)
    baseline = rows[0]
    for row in rows:
        keys = ["ratio_metric_w1_mean", "ratio_rapsd_distance", "ratio_slope_w1_deg"]
        keys += [key for key in baseline if key.startswith("nmae_")]
        failed = [key for key in keys if row[key] > baseline[key] * (1 + args.max_regression) + 1e-9]
        for key in ("diversity_ratio", "trav_pass_gen"):
            if row[key] < baseline[key] * (1 - args.max_regression):
                failed.append(key)
        for arch, base in baseline["per_archetype"].items():
            got = row["per_archetype"][arch]
            for key in ("ratio", "slope_w1_deg", "sink_density_gen", "peak_density_gen"):
                if got[key] > base[key] * (1 + args.max_regression) + 1e-9:
                    failed.append(f"{arch}.{key}")
            for band, value in base["spectrum_band_error_decades"].items():
                if abs(got["spectrum_band_error_decades"][band]) > abs(value) * (1 + args.max_regression) + 0.02:
                    failed.append(f"{arch}.spectrum.{band}")
        row["regressions"], row["passes_gates"] = failed, not failed
    best = min((r for r in rows if r["passes_gates"]), key=lambda r: r["ratio_metric_w1_mean"])
    ref_art = {k: v for k, v in rep["artifacts"].items() if k.endswith("_ref")}
    (out / "sampler_sweep.json").write_text(json.dumps({"checkpoint": sampler.checkpoint_info, "split": args.split,
                                                        "seed": args.seed, "n_per_half": len(a), "rows": rows, "best": best,
                                                        "baseline": configs[0], "max_regression": args.max_regression,
                                                        "screening_only": True,
                                                        "promotion_requires": "repeat on VAL >=500/half; verify originality; TEST only after selection",
                                                        "reference_artifacts": ref_art}, indent=2))
    keys = ["steps", "spacing", "eta", "guidance", "guidance_interval", "passes_gates", "ratio_metric_w1_mean",
            "ratio_rapsd_distance", "ratio_slope_w1_deg", "hf_energy_mean_gen", "sink_density_mean_gen", "ms_per_map"]
    lines = [f"# Sampler sweep ({args.split}, {len(a)} maps per half)", "",
             f"reference: hf_energy {ref_art['hf_energy_mean_ref']:.4f}, sink_density "
             f"{ref_art['sink_density_mean_ref']:.2f}", "", "| " + " | ".join(keys) + " |", "|---" * len(keys) + "|"]
    for r in sorted(rows, key=lambda r: r["ratio_metric_w1_mean"]):
        lines.append("| " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    (out / "sampler_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("best:", best)


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


def _cmd_evaluate(args: argparse.Namespace) -> dict:
    from nullscape.metrics.distribution import metric_table, nearest_neighbor_rmse
    from nullscape.viz.render import save_grid, save_metric_histograms, save_rapsd, save_traversability_grid

    if args.repeat_seeds:
        seeds = [int(s) for s in args.repeat_seeds.split(",")]
        if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
            raise ValueError("repeat-seeds must contain distinct nonnegative integers")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        reports = [_cmd_evaluate(argparse.Namespace(**{**vars(args), "repeat_seeds": "", "seed": seed,
                                                      "out": str(out / f"seed_{seed}")})) for seed in seeds]
        modes = {mode: summarize_repeats([_headline(r["modes"][mode]) for r in reports])
                 for mode in reports[0]["modes"]}
        result = {"checkpoint": reports[0]["checkpoint"], "split": args.split, "seeds": seeds,
                  "n_per_half": reports[0]["n_per_half"], "modes": modes,
                  "interpretation": "mean/std/range across split and sampling seeds; not training-run confidence intervals"}
        (out / "repeats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    sampler, store, a, b, device = _load(args)
    world = store.world
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_a, ref_b = store.heights(a), store.heights(b)
    tables = {"a": metric_table(ref_a, world), "b": metric_table(ref_b, world)}
    train_bank = store.heights(store.split("train")) if not args.no_memorization else None
    names = store.archetypes

    skw = _sampler_kwargs(args, sampler)
    gameplay_kw = dict(limit=args.gameplay_n, start=args.start, goal=args.goal, combat_radius_m=args.combat_radius_m)
    reference_gameplay = gameplay_summary(ref_b, world, **gameplay_kw) if args.gameplay_n else None
    from nullscape.utils.seed import derive_seed

    report: dict = {"checkpoint": sampler.checkpoint_info, "dataset": store.root.name, "split": args.split,
                    "n_per_half": len(a), "seed": args.seed, "reference_indices_a": a.tolist(),
                    "reference_indices_b": b.tolist(), "world": world.to_dict(),
                    "comparison_inputs": {"config_sha256": store.manifest["config_sha256"],
                        "condition_keys": store.condition_keys, "conditions": store.conditions[a].tolist(),
                        "labels": store.labels[a].tolist(), "archetypes": list(names),
                        "seeds": [derive_seed(args.seed + 1000, i) for i in range(len(a))],
                        "reference_std": [float(tables["b"][key].std()) for key in store.condition_keys]},
                    **skw, "modes": {}}
    gens: dict[str, np.ndarray] = {}
    for mode in args.modes.split(","):
        inputs = _mode_inputs(mode, store, a)
        t0 = time.time()
        gen = sampler.sample(n=len(a), seed=args.seed + 1000, batch_size=args.batch_size, **skw, **inputs)
        secs = time.time() - t0
        gt = metric_table(gen, world)
        rep = evaluate_generated(
            gen, ref_a, ref_b, world,
            requested_conds=inputs["cond_raw"] if inputs["known"].any() else None, cond_known=inputs["known"],
            gen_labels=inputs["labels"] if (inputs["labels"] >= 0).all() else None,
            ref_a_labels=store.labels[a], ref_b_labels=store.labels[b], archetype_names=names,
            train_bank=train_bank if mode in ("conditional", "unconditional") else None, device=device,
            tables={**tables, "gen": gt},
        )
        rep["sampling_seconds"] = secs
        rep["sampling_ms_per_map"] = 1000 * secs / len(a)
        if reference_gameplay is not None:
            rep["gameplay"] = {"generated": gameplay_summary(gen, world, **gameplay_kw),
                               "reference": reference_gameplay}
        report["modes"][mode] = rep
        gens[mode] = gen
        tables[mode] = gt
        np.save(out / f"generated_{_mode_slug(mode)}.npy", gen)
        print(f"[{mode}] " + " ".join(f"{k}={v:.3f}" for k, v in _headline(rep).items()))

    if args.guidance_sweep:
        sweep = {}
        for g in [float(x) for x in args.guidance_sweep.split(",")]:
            gen = sampler.sample(n=len(a), seed=args.seed + 1000, **{**skw, "guidance": g},
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
        label_known = m in ("conditional", "label_only") or m.startswith("label+property:")
        save_grid(g[:36], world, out / f"grid_{_mode_slug(m)}.png", suptitle=f"learned ({m})",
                  titles=[names[store.labels[a[i]]] if label_known else "" for i in range(min(36, len(g)))])
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
    return report


def _write_summary(report: dict, path: Path) -> None:
    lines = [f"# Evaluation: {report['checkpoint']['path']}", "",
             f"step {report['checkpoint']['step']}, dataset `{report['dataset']}`/{report['split']}, "
             f"{report['n_per_half']} maps per half, guidance {report['guidance']}, {report['steps']} DDIM steps "
             f"({report['spacing']} spacing, eta {report['eta']})", ""]
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
        if "gameplay" in rep:
            gp, rp = rep["gameplay"]["generated"], rep["gameplay"]["reference"]
            lines += [f"gameplay ({gp['n']} maps, endpoints {gp['start']} -> {gp['goal']}):", "",
                      "| metric | generated | procedural |", "|---|---|---|"]
            for key in gp["means"]:
                values = ["n/a" if group["means"][key] is None else f"{group['means'][key]:.3f}" for group in (gp, rp)]
                lines.append(f"| {key} | {values[0]} | {values[1]} |")
            lines += ["", "Route statistics exclude unreachable pairs. These are geometry checks, not playtest scores.", ""]
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
                         labels=np.repeat(store.labels[picks], s), batch_size=args.batch_size,
                         **_sampler_kwargs(args, sampler))
    gen = gen.reshape(len(picks), s, *gen.shape[1:])
    if args.blind:
        from nullscape.viz.render import save_grid

        out.mkdir(parents=True, exist_ok=True)
        maps = np.concatenate([ref, gen.reshape(-1, *gen.shape[-2:])])
        order = np.random.default_rng(args.seed + 17).permutation(len(maps))
        ids = [f"sample {i + 1:03d}" for i in range(len(maps))]
        save_grid(maps[order], world, out / "blind_2d.png", titles=ids, suptitle="Blind terrain comparison")
        key = {name: {"source": "procedural" if j < len(ref) else "learned",
                      "reference_index": int(picks[j if j < len(ref) else (j - len(ref)) // s])}
               for name, j in zip(ids, order)}
        (out / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"blind comparison -> {out}; keep blind_key.json hidden from reviewers")
        return

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


def _cmd_artifacts(args: argparse.Namespace) -> None:
    """Render per-checkpoint artifacts for a run; with --watch, follow a run that is still training."""
    import shutil

    import torch
    import yaml

    from nullscape.data.storage import TerrainStore
    from nullscape.eval.artifacts import (
        ARTIFACTS_ROOT, ArtifactSpec, _check_spec, generate_checkpoint_artifacts, step_dir_name,
    )
    from nullscape.inference.sampler import TerrainSampler
    from nullscape.utils.checkpoint import load_checkpoint

    run = Path(args.run)
    ck_dir = run / "checkpoints"
    out = Path(args.out) if args.out else ARTIFACTS_ROOT / run.name
    spec = ArtifactSpec(steps=args.steps, spacing=args.spacing, eta=args.eta, guidance=args.guidance,
                        guidance_interval=tuple(args.guidance_interval), n_eval=args.n, batch_size=args.batch_size)
    _check_spec(out, spec)  # fail fast even if every checkpoint is already rendered
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device == "cuda" and args.gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    max_steps = int((yaml.safe_load((run / "config.yaml").read_text()) or {}).get("train", {}).get("max_steps", 0))
    state: dict = {}

    def load_shared(ckpt: Path):
        if "store" not in state:
            meta = load_checkpoint(ckpt)["dataset"]
            store = TerrainStore.open(args.dataset or meta["root"])
            state["store"] = store
            state["bank"] = None if args.no_memorization else store.heights(store.split("train"))
        return state["store"], state["bank"]

    def render(ckpt: Path, step: int) -> None:
        if (out / step_dir_name(step) / "report.json").exists():
            return
        store, bank = load_shared(ckpt)
        t0 = time.time()
        sampler = TerrainSampler.from_checkpoint(ckpt, device=device)
        d = generate_checkpoint_artifacts(sampler, store, out, step, spec, train_bank=bank)
        del sampler
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[artifacts] step {step} -> {d} ({time.time() - t0:.0f}s)", flush=True)

    def snapshots() -> list[tuple[int, Path]]:
        return sorted((int(p.stem.split("_")[1]), p) for p in ck_dir.glob("step_*.pt"))

    def snapshot_last() -> int | None:
        last = ck_dir / "last.pt"
        if not last.exists():
            return None
        mtime = last.stat().st_mtime
        if state.get("last_mtime") == mtime or time.time() - mtime < 20:  # skip files still being written
            return None
        state["last_mtime"] = mtime
        step = int(load_checkpoint(last)["step"])
        dst = ck_dir / f"step_{step:07d}.pt"
        if not dst.exists():
            shutil.copy2(last, dst)
        return step

    while True:
        latest = snapshot_last()
        for step, path in snapshots():
            if step >= args.min_step:
                render(path, step)
        if not args.watch or (latest is not None and max_steps and latest >= max_steps):
            break
        time.sleep(args.poll)
    print(f"[artifacts] index -> {out / 'index.md'}")


def add_eval_commands(sub) -> None:
    from nullscape.cli_model import _add_runtime_args
    from nullscape.eval.performance import add_performance_command
    from nullscape.eval.comparison import add_comparison_command

    add_performance_command(sub)
    add_comparison_command(sub)
    a = sub.add_parser("artifacts", help="per-checkpoint visual/quantitative artifacts (optionally watch a live run)")
    a.add_argument("--run", required=True, help="run directory (runs/<timestamp>_<name>)")
    a.add_argument("--out", default=None, help="default: artifacts/<run dir name>")
    a.add_argument("--dataset", default=None)
    a.add_argument("--watch", action="store_true", help="snapshot last.pt as it changes until max_steps is reached")
    a.add_argument("--poll", type=float, default=60.0)
    a.add_argument("--min-step", type=int, default=0)
    a.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a.add_argument("--gpu-memory-fraction", type=float, default=0.12,
                   help="cap when sharing the GPU with a training job")
    a.add_argument("--n", type=int, default=256, help="maps per half for distribution metrics")
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--no-memorization", action="store_true")
    a.add_argument("--guidance", type=float, default=1.5)
    a.add_argument("--guidance-interval", type=float, nargs=2, default=[0.0, 1.0])
    a.add_argument("--steps", type=int, default=200)
    a.add_argument("--eta", type=float, default=0.0)
    a.add_argument("--spacing", default="uniform", choices=["uniform", "quadratic"])
    a.set_defaults(func=_cmd_artifacts)

    e = sub.add_parser("evaluate", help="quantitative evaluation of a checkpoint against held-out data")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset", default=None, help="defaults to the checkpoint's training dataset")
    e.add_argument("--split", default="val", choices=["val", "test"])
    e.add_argument("--n", type=int, default=1000, help="maps per half (A and B)")
    e.add_argument("--modes", default="conditional,label_only,unconditional," +
                   ",".join(f"property:{key}" for key in CONDITION_KEYS),
                   help="comma-separated conditional, label_only, unconditional, property:KEY, label+property:KEY")
    e.add_argument("--repeat-seeds", default="", help="e.g. 0,1,2; repeats both split and sampling seeds")
    e.add_argument("--batch-size", type=int, default=64)
    _add_runtime_args(e)
    e.add_argument("--gameplay-n", type=int, default=128, help="gameplay maps per mode; 0 disables")
    e.add_argument("--start", type=int, nargs=2, metavar=("ROW", "COL"))
    e.add_argument("--goal", type=int, nargs=2, metavar=("ROW", "COL"))
    e.add_argument("--combat-radius-m", type=float, default=32.0)
    _add_sampler_args(e)
    e.add_argument("--guidance-sweep", default="", help="comma-separated guidance scales, e.g. 1,1.5,2,3")
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
    c.add_argument("--blind", action="store_true", help="shuffle anonymized samples; save a separate answer key")
    _add_runtime_args(c)
    c.add_argument("--batch-size", type=int, default=64)
    _add_sampler_args(c)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_compare)

    s = sub.add_parser("sampler-sweep", help="grid over sampler settings to choose inference defaults")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--dataset", default=None)
    s.add_argument("--split", default="val", choices=["val", "test"])
    s.add_argument("--n", type=int, default=500, help="maps per half (A and B)")
    s.add_argument("--steps-list", default="50,100")
    s.add_argument("--spacings", default="uniform,quadratic")
    s.add_argument("--etas", default="0,0.5,1")
    s.add_argument("--guidances", default="2.0")
    s.add_argument("--guidance-intervals", default="0:1", help="comma-separated normalized ranges, e.g. 0:1,0.05:0.8")
    s.add_argument("--max-regression", type=float, default=0.05, help="allowed relative regression vs first grid entry")
    s.add_argument("--batch-size", type=int, default=64)
    _add_runtime_args(s)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_sampler_sweep)
