"""Model-agnostic evaluation of a set of generated heightmaps.

Protocol (see docs/EVALUATION.md):
  * Reference data (validation or test split) is split into two disjoint halves
    A and B of equal size.
  * The model generates one map per map in A, conditioned on A's measured
    properties and archetype labels (or unconditionally).
  * Distances are computed gen-vs-B and, as a noise floor, A-vs-B. Both
    comparisons use the same B and the same set sizes, so a ratio of ~1 means
    the generated set is as close to B as real held-out data is.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nullscape.metrics.distribution import (
    compare_sets,
    metric_table,
    mean_log_rapsd,
    pooled_slope_w1,
    nearest_neighbor_rmse,
    normalized_w1,
)
from nullscape.metrics.quality import CONDITION_KEYS
from nullscape.metrics.traversability import AgentSpec
from nullscape.world import WorldSpec

HEADLINE_KEYS: tuple[str, ...] = ("metric_w1_mean", "rapsd_distance", "height_w1", "slope_w1_deg")


def split_halves(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic disjoint index halves (A, B) of equal size (drops one if n is odd)."""
    perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return np.sort(perm[:half]), np.sort(perm[half : 2 * half])


def _table_w1_mean(gt: dict[str, np.ndarray], rt: dict[str, np.ndarray]) -> float:
    return float(np.mean([normalized_w1(gt[k], rt[k]) for k in rt if k != "trav_passed"]))


def condition_adherence(
    gen_table: dict[str, np.ndarray],
    requested: np.ndarray,
    known: np.ndarray | None = None,
    ref_std: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """How well measured properties of generated maps match the requested ones.

    ``requested`` is [N, K] in raw units ordered as CONDITION_KEYS; ``known`` masks
    which entries were actually given to the model. ``nmae`` is MAE divided by the
    reference std of that property (``ref_std``), so 1.0 means "no better than a
    random real map" in scale.
    """
    requested = np.asarray(requested, dtype=np.float64)
    known = np.ones_like(requested, dtype=bool) if known is None else np.asarray(known, dtype=bool)
    out: dict[str, dict[str, float]] = {}
    for j, key in enumerate(CONDITION_KEYS):
        sel = known[:, j]
        if sel.sum() < 2:
            continue
        got, want = gen_table[key][sel], requested[sel, j]
        mae = float(np.abs(got - want).mean())
        r = float(np.corrcoef(got, want)[0, 1]) if got.std() > 1e-12 and want.std() > 1e-12 else float("nan")
        entry = {"mae": mae, "pearson_r": r, "n": int(sel.sum())}
        if ref_std is not None:
            entry["nmae"] = mae / float(ref_std[j] + 1e-12)
        out[key] = entry
    return out


def memorization(
    gen: np.ndarray,
    held_out: np.ndarray,
    train_bank: np.ndarray,
    device: str = "cpu",
) -> dict[str, float]:
    """Nearest-training-neighbor RMSE of generated maps vs. of held-out real maps.

    Held-out maps were never trained on, so their NN distances describe how close
    an *independent* sample from the data distribution typically lands to the
    training set. Generated maps landing much closer than that indicates copying.
    """
    g, _ = nearest_neighbor_rmse(gen, train_bank, device=device)
    h, _ = nearest_neighbor_rmse(held_out, train_bank, device=device)
    p1 = float(np.percentile(h, 1))
    return {
        "gen_nn_rmse_median": float(np.median(g)),
        "gen_nn_rmse_p05": float(np.percentile(g, 5)),
        "heldout_nn_rmse_median": float(np.median(h)),
        "heldout_nn_rmse_p05": float(np.percentile(h, 5)),
        "nn_median_ratio": float(np.median(g) / (np.median(h) + 1e-12)),
        "frac_gen_below_heldout_p01": float((g < p1).mean()),
    }


def spectral_band_errors(gen: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    k, gl = mean_log_rapsd(gen)
    _, rl = mean_log_rapsd(ref)
    f = k / (gen.shape[-1] / 2)
    bands = {"low": (0, 0.125), "mid": (0.125, 0.375), "fine": (0.375, 0.75), "finest": (0.75, 1.0)}
    return {name: float((gl - rl)[(f > lo) & (f <= hi)].mean())
            for name, (lo, hi) in bands.items() if ((f > lo) & (f <= hi)).any()}


def summarize_repeats(rows: list[dict[str, float]]) -> dict:
    if not rows:
        raise ValueError("at least one repeat is required")
    result = {}
    for key in set.intersection(*(set(r) for r in rows)):
        values = np.asarray([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            result[key] = {"n": len(values), "mean": float(values.mean()),
                           "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                           "min": float(values.min()), "max": float(values.max())}
    return result


def gameplay_summary(maps: np.ndarray, world: WorldSpec, *, limit: int = 128, start=None, goal=None,
                     combat_radius_m: float = 32.0) -> dict:
    from nullscape.metrics.traversability import gameplay_metrics

    if limit < 1 or not len(maps):
        raise ValueError("gameplay limit and map count must be positive")
    rows = [gameplay_metrics(h, world, start=start, goal=goal, combat_radius_m=combat_radius_m) for h in maps[:limit]]
    reachable = [row for row in rows if row["spawn_goal_reachable"]]
    route_keys = [k for k in rows[0] if k.startswith("route_")]
    means = {k: float(np.mean([row[k] for row in rows])) for k in rows[0] if k not in route_keys}
    means.update({k: float(np.mean([row[k] for row in reachable])) if reachable else None for k in route_keys})
    return {"n": len(rows), "reachable_n": len(reachable), "means": means,
            "start": list(start) if start is not None else [world.resolution // 2, 0],
            "goal": list(goal) if goal is not None else [world.resolution // 2, world.resolution - 1],
            "combat_radius_m": combat_radius_m, "combat_max_slope_deg": 10.0,
            "route_statistics": "conditional on reachable endpoints; not a measure of player enjoyment"}


def evaluate_generated(
    gen: np.ndarray,
    ref_a: np.ndarray,
    ref_b: np.ndarray,
    world: WorldSpec,
    *,
    agent: AgentSpec = AgentSpec(),
    requested_conds: np.ndarray | None = None,
    cond_known: np.ndarray | None = None,
    gen_labels: np.ndarray | None = None,
    ref_a_labels: np.ndarray | None = None,
    ref_b_labels: np.ndarray | None = None,
    archetype_names: Sequence[str] | None = None,
    train_bank: np.ndarray | None = None,
    device: str = "cpu",
    min_per_archetype: int = 10,
    tables: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict:
    """Full report for ``gen`` (generated with ref_a's conditions) against ref_b.

    All arrays are [N, H, W] heightmaps in [0, 1]; ``len(gen)`` must equal
    ``len(ref_a)`` and ``len(ref_b)``. ``tables`` may carry precomputed
    ``metric_table`` results under keys "gen", "a", "b".
    """
    if not (len(gen) == len(ref_a) == len(ref_b)):
        raise ValueError(f"set sizes must match: gen={len(gen)} ref_a={len(ref_a)} ref_b={len(ref_b)}")
    gen = np.clip(np.asarray(gen, dtype=np.float64), 0.0, 1.0)
    tables = tables or {}
    gt = tables["gen"] if "gen" in tables else metric_table(gen, world, agent)
    at = tables["a"] if "a" in tables else metric_table(ref_a, world, agent)
    bt = tables["b"] if "b" in tables else metric_table(ref_b, world, agent)

    model = compare_sets(gen, ref_b, world, agent, gen_table=gt, ref_table=bt)
    floor = compare_sets(ref_a, ref_b, world, agent, gen_table=at, ref_table=bt)
    report: dict = {
        "n": len(gen),
        "model_vs_ref": model,
        "noise_floor": floor,
        "ratio_to_floor": {k: float(model[k] / (floor[k] + 1e-12)) for k in HEADLINE_KEYS},
        "artifacts": {
            "checkerboard_mean_gen": float(gt["checkerboard"].mean()),
            "checkerboard_mean_ref": float(bt["checkerboard"].mean()),
            "hf_energy_mean_gen": float(gt["hf_energy"].mean()),
            "hf_energy_mean_ref": float(bt["hf_energy"].mean()),
            "sink_density_mean_gen": float(gt["sink_density"].mean()),
            "sink_density_mean_ref": float(bt["sink_density"].mean()),
        },
        "metric_means": {
            k: {"gen": float(gt[k].mean()), "ref": float(bt[k].mean())} for k in bt
        },
    }

    if requested_conds is not None:
        ref_std = np.array([bt[k].std() for k in CONDITION_KEYS])
        report["condition_adherence"] = condition_adherence(gt, requested_conds, cond_known, ref_std)
        # Floor for adherence: how well a *different* real map's properties match
        # (A's own properties re-measured are exact, so compare A's requests to B).
        report["condition_adherence_random_baseline"] = condition_adherence(
            bt, requested_conds, cond_known, ref_std
        )

    if gen_labels is not None and ref_a_labels is not None and ref_b_labels is not None:
        per: dict[str, dict[str, float]] = {}
        for lab in np.unique(ref_b_labels):
            g_sel, a_sel, b_sel = gen_labels == lab, ref_a_labels == lab, ref_b_labels == lab
            if min(g_sel.sum(), a_sel.sum(), b_sel.sum()) < min_per_archetype:
                continue
            sub = lambda t, s: {k: v[s] for k, v in t.items()}  # noqa: E731
            m = _table_w1_mean(sub(gt, g_sel), sub(bt, b_sel))
            f = _table_w1_mean(sub(at, a_sel), sub(bt, b_sel))
            name = archetype_names[int(lab)] if archetype_names is not None else str(int(lab))
            per[name] = {
                "metric_w1_mean": m,
                "noise_floor": f,
                "ratio": m / (f + 1e-12),
                "n_gen": int(g_sel.sum()),
                "n_ref": int(b_sel.sum()),
                "spectrum_band_error_decades": spectral_band_errors(gen[g_sel], ref_b[b_sel]),
                "slope_w1_deg": pooled_slope_w1(gen[g_sel], ref_b[b_sel], world),
                "sink_density_gen": float(gt["sink_density"][g_sel].mean()),
                "sink_density_ref": float(bt["sink_density"][b_sel].mean()),
                "peak_density_gen": float(gt["peak_density"][g_sel].mean()),
                "peak_density_ref": float(bt["peak_density"][b_sel].mean()),
            }
        report["per_archetype"] = per

    if train_bank is not None:
        report["memorization"] = memorization(gen, ref_a, train_bank, device=device)
    return report
