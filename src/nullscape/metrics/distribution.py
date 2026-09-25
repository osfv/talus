"""Set-level comparisons between two collections of heightmaps.

Used to answer "does the learned terrain look like it came from the same
distribution as held-out procedural terrain?" Every distance here should be
read against a noise floor: the same distance computed between two disjoint
halves of the reference data (see ``nullscape.eval``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from nullscape.metrics.quality import compute_metrics, radial_power_spectrum, slope_degrees
from nullscape.metrics.traversability import AgentSpec, analyze
from nullscape.world import WorldSpec

TRAVERSABILITY_KEYS: tuple[str, ...] = (
    "walkable_land_fraction",
    "largest_component_fraction",
    "connectivity_probability",
)


def metric_table(heights: Iterable[np.ndarray], world: WorldSpec, agent: AgentSpec = AgentSpec()) -> dict[str, np.ndarray]:
    """Per-map quality + traversability metrics as {name: array[N]}."""
    rows: list[dict[str, float]] = []
    for h in heights:
        m = compute_metrics(h, world)
        t = analyze(h, world, agent).to_dict()
        m.update({f"trav_{k}": float(t[k]) for k in TRAVERSABILITY_KEYS})
        m["trav_passed"] = float(t["passed"])
        rows.append(m)
    if not rows:
        raise ValueError("empty heightmap set")
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}


def normalized_w1(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein-1 distance divided by the std of the pooled samples.

    Pooling keeps the value bounded when the reference is (near-)constant,
    e.g. traversability fractions that are 1.0 for every reference map.
    """
    scale = float(np.std(np.concatenate([a, b])))
    return float(wasserstein_distance(a, b) / scale) if scale > 1e-12 else 0.0


def mean_log_rapsd(heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over maps of log10 radially averaged power spectrum."""
    logs = []
    k = None
    for h in heights:
        k, p = radial_power_spectrum(h)
        logs.append(np.log10(p + 1e-20))
    return k, np.mean(logs, axis=0)


def rapsd_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference (in decades) between the mean log-RAPSD curves of two sets."""
    _, la = mean_log_rapsd(a)
    _, lb = mean_log_rapsd(b)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def _pooled_subsample(values: np.ndarray, n: int, seed: int) -> np.ndarray:
    values = values.ravel()
    if values.size <= n:
        return values
    return np.random.default_rng(seed).choice(values, size=n, replace=False)


def pooled_height_w1(a: np.ndarray, b: np.ndarray, n: int = 200_000, seed: int = 0) -> float:
    """W1 between pooled per-cell height distributions (in normalized height units)."""
    return float(wasserstein_distance(_pooled_subsample(a, n, seed), _pooled_subsample(b, n, seed + 1)))


def pooled_slope_w1(a: np.ndarray, b: np.ndarray, world: WorldSpec, n: int = 200_000, seed: int = 0) -> float:
    """W1 between pooled per-cell slope distributions, in degrees."""
    sa = np.stack([slope_degrees(h, world) for h in a])
    sb = np.stack([slope_degrees(h, world) for h in b])
    return float(wasserstein_distance(_pooled_subsample(sa, n, seed), _pooled_subsample(sb, n, seed + 1)))


def _dihedral(x: torch.Tensor) -> torch.Tensor:
    """[N, H, W] -> [8N, H, W] with all rotations/flips (requires square maps)."""
    out = []
    for flip in (False, True):
        y = torch.flip(x, dims=[-1]) if flip else x
        out.extend(torch.rot90(y, k, dims=[-2, -1]) for k in range(4))
    return torch.cat(out)


@torch.no_grad()
def nearest_neighbor_rmse(
    queries: np.ndarray,
    bank: np.ndarray,
    dihedral: bool = True,
    device: str | torch.device = "cpu",
    chunk: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query map, RMSE to its nearest map in ``bank`` and that map's index.

    With ``dihedral=True`` every bank map is also compared under all 8
    rotations/flips (matching the training augmentation), so a sample that is a
    rotated copy of a training map is still detected as memorized.
    """
    q = torch.as_tensor(np.asarray(queries, dtype=np.float32), device=device)
    n_q, dim = q.shape[0], q[0].numel()
    q = q.reshape(n_q, dim)
    q_sq = (q * q).sum(1, keepdim=True)
    best = torch.full((n_q,), float("inf"), device=device)
    best_idx = torch.zeros(n_q, dtype=torch.long, device=device)
    for start in range(0, len(bank), chunk):
        b = torch.as_tensor(np.asarray(bank[start : start + chunk], dtype=np.float32), device=device)
        n_b = b.shape[0]
        if dihedral:
            b = _dihedral(b)
        b = b.reshape(b.shape[0], dim)
        d2 = (q_sq + (b * b).sum(1)[None, :] - 2.0 * q @ b.T).clamp_min_(0.0)
        val, arg = d2.min(dim=1)
        better = val < best
        best = torch.where(better, val, best)
        best_idx = torch.where(better, start + arg % n_b, best_idx)
    return (best / dim).sqrt().cpu().numpy(), best_idx.cpu().numpy()


def mean_pairwise_rmse(heights: np.ndarray, n_pairs: int = 2000, seed: int = 0) -> float:
    """Diversity: mean RMSE between random distinct pairs within a set."""
    n = len(heights)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = (i + rng.integers(1, n, n_pairs)) % n
    flat = np.asarray(heights, dtype=np.float64).reshape(n, -1)
    return float(np.sqrt(((flat[i] - flat[j]) ** 2).mean(axis=1)).mean())


def compare_sets(
    gen: np.ndarray,
    ref: np.ndarray,
    world: WorldSpec,
    agent: AgentSpec = AgentSpec(),
    gen_table: dict[str, np.ndarray] | None = None,
    ref_table: dict[str, np.ndarray] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Distribution distances of ``gen`` relative to ``ref`` (both [N, H, W] in [0, 1]).

    Lower is better for every distance. ``*_table`` can be passed to reuse
    precomputed ``metric_table`` results.
    """
    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    gt = gen_table if gen_table is not None else metric_table(gen, world, agent)
    rt = ref_table if ref_table is not None else metric_table(ref, world, agent)
    per_metric = {k: normalized_w1(gt[k], rt[k]) for k in rt if k != "trav_passed"}
    return {
        "metric_w1": per_metric,
        "metric_w1_mean": float(np.mean(list(per_metric.values()))),
        "rapsd_distance": rapsd_distance(gen, ref),
        "height_w1": pooled_height_w1(gen, ref),
        "slope_w1_deg": pooled_slope_w1(gen, ref, world),
        "trav_pass_rate_gen": float(gt["trav_passed"].mean()),
        "trav_pass_rate_ref": float(rt["trav_passed"].mean()),
        "diversity_gen": mean_pairwise_rmse(gen),
        "diversity_ref": mean_pairwise_rmse(ref),
    }
