"""Per-checkpoint visual + quantitative evaluation artifacts.

Layout (one directory per training lineage, one subdirectory per checkpoint):

artifacts/<run>/
  spec.json                    frozen ArtifactSpec (seeds, conditions, sampler) shared by all checkpoints
  index.md                     table of headline metrics per checkpoint + links
  progression.png              fixed-seed samples: rows = checkpoints, first row = procedural reference
  progression_metrics.png      headline ratios / adherence vs training step
  step_0025000/
    procedural_vs_learned.png  same conditions + seeds: procedural row vs learned row
    compare_3d.png             3D surfaces, procedural vs learned
    rapsd.png                  radially averaged power spectra
    metric_histograms.png      per-map metric distributions
    condition_adherence.png    requested vs measured property scatter
    nearest_neighbors.png      memorization check against the training set
    traversability.png         walkability overlays of learned maps
    report.json / summary.md   full numeric report (eval/core.py protocol)
    fixed_seed.npy             the fixed-seed learned maps (frozen input for progression.png)

Every checkpoint uses the same spec, so differences between step directories are
due to the weights only. Figures that span checkpoints are rebuilt from the saved
per-step arrays and reports, never by re-sampling old checkpoints.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nullscape.data.storage import TerrainStore
from nullscape.eval.core import HEADLINE_KEYS, evaluate_generated, split_halves
from nullscape.inference.sampler import TerrainSampler
from nullscape.metrics.distribution import metric_table, nearest_neighbor_rmse
from nullscape.metrics.quality import CONDITION_KEYS
from nullscape.utils.paths import REPO_ROOT

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
_STEP_DIR = re.compile(r"^step_(\d{7})$")


@dataclass(frozen=True)
class ArtifactSpec:
    per_archetype: int = 2      # fixed-seed maps per archetype (progression + side-by-side)
    n_eval: int = 256           # maps per half (A conditions / B reference) for distributions
    n_3d: int = 3
    seed: int = 2026
    steps: int = 200
    spacing: str = "uniform"
    eta: float = 0.0
    guidance: float = 1.5
    batch_size: int = 32        # small: artifacts may run next to a training job on the same GPU
    split: str = "val"

    def sampler_kwargs(self) -> dict[str, Any]:
        return {"steps": self.steps, "spacing": self.spacing, "eta": self.eta, "guidance": self.guidance,
                "batch_size": self.batch_size}


def step_dir_name(step: int) -> str:
    return f"step_{int(step):07d}"


def fixed_indices(store: TerrainStore, spec: ArtifactSpec) -> np.ndarray:
    """First ``per_archetype`` maps of each archetype in the split (deterministic)."""
    idx = store.split(spec.split)
    labels = store.labels[idx]
    return np.concatenate([idx[labels == k][: spec.per_archetype] for k in range(len(store.archetypes))])


def eval_halves(store: TerrainStore, spec: ArtifactSpec) -> tuple[np.ndarray, np.ndarray]:
    idx = store.split(spec.split)
    a, b = split_halves(len(idx), seed=spec.seed)
    n = min(spec.n_eval, len(a))
    return idx[a[:n]], idx[b[:n]]


def _check_spec(root: Path, spec: ArtifactSpec) -> None:
    path = root / "spec.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != asdict(spec):
            raise ValueError(f"{path} has a different ArtifactSpec; checkpoints would not be comparable. "
                             f"saved={saved} requested={asdict(spec)}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")


def generate_checkpoint_artifacts(
    sampler: TerrainSampler,
    store: TerrainStore,
    root: str | Path,
    step: int,
    spec: ArtifactSpec = ArtifactSpec(),
    train_bank: np.ndarray | None = None,
    memorization_device: str = "cpu",
    save_generated: bool = False,
) -> Path:
    """Render all artifacts for one checkpoint into ``root/step_XXXXXXX`` and refresh run-level figures.

    ``save_generated`` also stores the distribution-set maps as ``generated_eval.npy`` (row i was
    generated from the conditions of ``eval_halves(store, spec)[0][i]``) for offline analysis.
    """
    import matplotlib.pyplot as plt

    from nullscape.eval.cli import _write_summary
    from nullscape.viz.render import (
        _save, draw_heightmap, draw_surface, save_condition_scatter, save_grid, save_metric_histograms,
        save_rapsd, save_traversability_grid,
    )

    root = Path(root)
    _check_spec(root, spec)
    out = root / step_dir_name(step)
    out.mkdir(parents=True, exist_ok=True)
    world = store.world
    names = store.archetypes
    k = len(store.condition_keys)
    t_start = time.time()

    # fixed-seed set: identical conditions, labels and seeds at every checkpoint
    fx = fixed_indices(store, spec)
    fx_ref = store.heights(fx)
    fx_gen = sampler.sample(n=len(fx), seed=spec.seed, cond_raw=store.conditions[fx], known=np.ones((len(fx), k), bool),
                            labels=store.labels[fx], **spec.sampler_kwargs())
    np.save(out / "fixed_seed.npy", fx_gen)
    (out / "fixed_seed_meta.json").write_text(json.dumps({
        "indices": fx.tolist(), "labels": [names[i] for i in store.labels[fx]], "seed": spec.seed,
        "conditions": store.conditions[fx].tolist(), "condition_keys": store.condition_keys}, indent=2))

    # distribution set: generate one map per map in A (its conditions), compare against B
    a, b = eval_halves(store, spec)
    ref_a, ref_b = store.heights(a), store.heights(b)
    t0 = time.time()
    gen = sampler.sample(n=len(a), seed=spec.seed + 1, cond_raw=store.conditions[a], known=np.ones((len(a), k), bool),
                         labels=store.labels[a], **spec.sampler_kwargs())
    sample_secs = time.time() - t0
    if save_generated:
        np.save(out / "generated_eval.npy", gen)
    tables = {"a": metric_table(ref_a, world), "b": metric_table(ref_b, world), "gen": metric_table(gen, world)}
    rep = evaluate_generated(gen, ref_a, ref_b, world, requested_conds=store.conditions[a],
                             gen_labels=store.labels[a], ref_a_labels=store.labels[a], ref_b_labels=store.labels[b],
                             archetype_names=names, train_bank=train_bank, device=memorization_device, tables=tables)
    rep["sampling_ms_per_map"] = 1000 * sample_secs / len(a)
    report = {"checkpoint": {**getattr(sampler, "checkpoint_info", {}), "step": int(step)}, "dataset": store.root.name,
              "split": spec.split, "n_per_half": len(a), "spec": asdict(spec),
              **{kk: v for kk, v in spec.sampler_kwargs().items() if kk != "batch_size"},
              "modes": {"conditional": rep}}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _write_summary(report, out / "summary.md")

    # 1. procedural vs learned, same conditions + seeds (pairs of rows)
    cols = 6
    pairs_h, pairs_t = [], []
    for r0 in range(0, len(fx), cols):
        chunk = range(r0, min(r0 + cols, len(fx)))
        pairs_h += [fx_ref[i] for i in chunk] + [fx_gen[i] for i in chunk]
        pairs_t += [f"PROCEDURAL {names[store.labels[fx[i]]]}" for i in chunk] + [f"LEARNED {step // 1000}k" for i in chunk]
    save_grid(pairs_h, world, out / "procedural_vs_learned.png", titles=pairs_t, ncols=cols,
              suptitle=f"step {step}: procedural (upper row) vs learned (lower row), same conditions and seeds")

    # 2. 3D comparisons (one per archetype, first n_3d)
    _, first = np.unique(store.labels[fx], return_index=True)  # first fixed map of each present archetype
    pick = sorted(first.tolist())[: spec.n_3d]
    fig = plt.figure(figsize=(9, 3.6 * len(pick)))
    for r, i in enumerate(pick):
        ax = fig.add_subplot(len(pick), 2, 2 * r + 1, projection="3d")
        draw_surface(ax, fx_ref[i], world)
        ax.set_title(f"procedural {names[store.labels[fx[i]]]}", fontsize=8)
        ax = fig.add_subplot(len(pick), 2, 2 * r + 2, projection="3d")
        draw_surface(ax, fx_gen[i], world)
        ax.set_title(f"learned (step {step})", fontsize=8)
    _save(fig, out / "compare_3d.png")

    # 3-4. spectra + metric distributions (A shown as the real-vs-real reference)
    save_rapsd({"procedural B": ref_b, "procedural A (noise floor)": ref_a, f"learned {step // 1000}k": gen},
               out / "rapsd.png")
    save_metric_histograms({"procedural B": tables["b"], f"learned {step // 1000}k": tables["gen"]},
                           out / "metric_histograms.png")

    # 5. conditioning adherence
    measured = np.stack([tables["gen"][key] for key in CONDITION_KEYS], axis=1)
    save_condition_scatter(store.conditions[a], measured, list(CONDITION_KEYS), out / "condition_adherence.png",
                           rep["condition_adherence"], rep["condition_adherence_random_baseline"],
                           title=f"step {step}: requested vs measured properties")

    # 6. memorization: fixed-seed maps next to their nearest training maps (8 rotations/flips)
    if train_bank is not None:
        rmse, nn = nearest_neighbor_rmse(fx_gen, train_bank, device=memorization_device)
        mm = rep["memorization"]
        hs, ts = [], []
        for i in range(len(fx_gen)):
            hs += [fx_gen[i], train_bank[nn[i]]]
            ts += [f"learned #{i}", f"nearest train rmse={rmse[i]:.3f}"]
        save_grid(hs, world, out / "nearest_neighbors.png", titles=ts, ncols=6,
                  suptitle=f"step {step}: median NN-RMSE learned {mm['gen_nn_rmse_median']:.4f} vs held-out "
                           f"{mm['heldout_nn_rmse_median']:.4f} (ratio {mm['nn_median_ratio']:.2f})")

    # 7. traversability of learned maps
    save_traversability_grid(list(fx_gen), world, out / "traversability.png")

    (out / "timing.json").write_text(json.dumps({"seconds": time.time() - t_start}))
    refresh_run_figures(root, store)
    return out


def _step_dirs(root: Path) -> list[tuple[int, Path]]:
    found = [(int(m.group(1)), p) for p in root.iterdir() if p.is_dir() and (m := _STEP_DIR.match(p.name))]
    return sorted(found)


def refresh_run_figures(root: str | Path, store: TerrainStore) -> None:
    """Rebuild progression.png, progression_metrics.png and index.md from saved per-step outputs."""
    import matplotlib.pyplot as plt

    from nullscape.viz.render import _save, draw_heightmap

    root = Path(root)
    spec = ArtifactSpec(**json.loads((root / "spec.json").read_text(encoding="utf-8")))
    world = store.world
    names = store.archetypes
    steps = [(s, p) for s, p in _step_dirs(root) if (p / "fixed_seed.npy").exists() and (p / "report.json").exists()]
    if not steps:
        return
    fx = fixed_indices(store, spec)
    ref = store.heights(fx)

    rows = [("procedural", ref)] + [(f"step {s / 1000:g}k", np.load(p / "fixed_seed.npy")) for s, p in steps]
    fig, axes = plt.subplots(len(rows), len(fx), figsize=(1.35 * len(fx), 1.45 * len(rows)), squeeze=False)
    for r, (label, maps) in enumerate(rows):
        for c in range(len(fx)):
            draw_heightmap(axes[r, c], maps[c], world, names[store.labels[fx[c]]] if r == 0 else None)
        axes[r, 0].text(-0.1, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=8)
    fig.suptitle("Fixed-seed progression: same conditions and noise at every checkpoint", fontsize=10)
    _save(fig, root / "progression.png")

    reports = [(s, json.loads((p / "report.json").read_text(encoding="utf-8"))["modes"]["conditional"]) for s, p in steps]
    xs = [s for s, _ in reports]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for key in HEADLINE_KEYS:
        ax1.plot(xs, [r["ratio_to_floor"][key] for _, r in reports], marker="o", label=key)
    ax1.axhline(1.0, color="k", lw=0.8, ls="--")
    ax1.set_title("distance / noise floor (1.0 = indistinguishable)", fontsize=9)
    ax1.set_xlabel("step")
    ax1.legend(fontsize=7)
    for key in CONDITION_KEYS:
        ax2.plot(xs, [r["condition_adherence"][key]["nmae"] for _, r in reports], marker="o", label=key)
    ax2.set_title("condition adherence: MAE / reference std (lower is better)", fontsize=9)
    ax2.set_xlabel("step")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, root / "progression_metrics.png")

    lines = [f"# Checkpoint artifacts: {root.name}", "",
             f"Sampler: {spec.steps} DDIM steps ({spec.spacing}), eta {spec.eta}, guidance {spec.guidance}; "
             f"{spec.n_eval} maps per half from `{spec.split}`; fixed-seed set = {spec.per_archetype} per archetype.", "",
             "![progression](progression.png)", "", "![metrics](progression_metrics.png)", "",
             "| step | " + " | ".join(f"ratio {k}" for k in HEADLINE_KEYS) + " | mean nMAE | NN ratio | trav pass (gen/ref) | dir |",
             "|---" * (len(HEADLINE_KEYS) + 5) + "|"]
    for s, r in reports:
        nmae = np.mean([v["nmae"] for v in r["condition_adherence"].values()])
        nn = r.get("memorization", {}).get("nn_median_ratio", float("nan"))
        mv = r["model_vs_ref"]
        lines.append(f"| {s} | " + " | ".join(f"{r['ratio_to_floor'][k]:.2f}" for k in HEADLINE_KEYS)
                     + f" | {nmae:.3f} | {nn:.2f} | {mv['trav_pass_rate_gen']:.2f}/{mv['trav_pass_rate_ref']:.2f} "
                     f"| [{step_dir_name(s)}]({step_dir_name(s)}/summary.md) |")
    (root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoint_step(path: str | Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["step"])
