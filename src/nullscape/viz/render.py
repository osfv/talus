"""Rendering heightmaps for inspection: hillshade composites, 3D surfaces, overlays, plots.

``draw_*`` functions draw on a matplotlib Axes; ``save_*`` functions write PNGs.
Everything uses the Agg backend, so it works headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from nullscape.metrics.quality import radial_power_spectrum, slope_degrees  # noqa: E402
from nullscape.metrics.traversability import AgentSpec, analyze, longest_route  # noqa: E402
from nullscape.world import WorldSpec  # noqa: E402

_LAND = LinearSegmentedColormap.from_list(
    "nullscape_land",
    [(0.0, "#c9c28f"), (0.06, "#6f9b4f"), (0.3, "#3f6e34"), (0.55, "#8a7452"), (0.8, "#8c8680"), (1.0, "#f4f4f2")],
)
_WATER = LinearSegmentedColormap.from_list("nullscape_water", [(0.0, "#0c2d57"), (1.0, "#4f93c8")])
_TRAV = ListedColormap(["#1f5fa8", "#b23a3a", "#e3c26b", "#3f9b4f"])


def hillshade(h: np.ndarray, world: WorldSpec, azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
              z_factor: float = 2.0) -> np.ndarray:
    """Horn (1981) hillshade in [0, 1] with physical scaling and vertical exaggeration ``z_factor``."""
    z = np.pad(np.asarray(h, dtype=np.float64) * world.max_height_m * z_factor, 1, mode="edge")
    c = world.cell_size_m
    dzdx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * c)
    dzdy = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8 * c)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    zen = np.radians(90.0 - altitude_deg)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    shade = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)


def terrain_rgb(h: np.ndarray, world: WorldSpec) -> np.ndarray:
    """Hypsometric tint: water gradient below sea level, land colormap above."""
    h = np.asarray(h, dtype=np.float64)
    sea = world.sea_level
    land_t = np.clip((h - sea) / max(1.0 - sea, 1e-6), 0.0, 1.0)
    water_t = np.clip(h / max(sea, 1e-6), 0.0, 1.0)
    rgb = _LAND(land_t)[..., :3]
    water = h < sea
    rgb[water] = _WATER(water_t[water])[..., :3]
    return rgb


def shaded_rgb(h: np.ndarray, world: WorldSpec, shade_strength: float = 0.75) -> np.ndarray:
    shade = hillshade(h, world)
    water = np.asarray(h) < world.sea_level
    factor = np.where(water, 1.0, (1 - shade_strength) + shade_strength * shade)
    return np.clip(terrain_rgb(h, world) * factor[..., None], 0.0, 1.0)


def draw_heightmap(ax, h: np.ndarray, world: WorldSpec, title: str | None = None, contours: bool = False) -> None:
    ax.imshow(shaded_rgb(h, world), interpolation="nearest")
    if contours:
        levels = np.linspace(0.0, 1.0, 21)
        ax.contour(h, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
        ax.contour(h, levels=[world.sea_level], colors="#0c2d57", linewidths=0.8)
    if title:
        ax.set_title(title, fontsize=7)
    ax.set_axis_off()


def draw_slope(ax, h: np.ndarray, world: WorldSpec, title: str | None = None) -> None:
    im = ax.imshow(slope_degrees(h, world), cmap="magma", vmin=0, vmax=60, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, label="slope (deg)")
    ax.set_title(title or "slope", fontsize=8)
    ax.set_axis_off()


def draw_traversability(ax, h: np.ndarray, world: WorldSpec, agent: AgentSpec = AgentSpec(), route: bool = True,
                        title: str | None = None) -> None:
    r = analyze(h, world, agent)
    cls = np.full(r.walkable.shape, 1)
    cls[r.water] = 0
    cls[r.walkable] = 2
    cls[r.largest_component] = 3
    ax.imshow(hillshade(h, world), cmap="gray", interpolation="nearest")
    ax.imshow(cls, cmap=_TRAV, vmin=-0.5, vmax=3.5, alpha=0.55, interpolation="nearest")
    if route:
        path = longest_route(r)
        if path:
            py, px = zip(*path)
            ax.plot(px, py, color="white", linewidth=1.0)
    status = "PASS" if r.passed else "FAIL"
    ax.set_title(title or f"{status} walk={r.walkable_land_fraction:.2f} lcc={r.largest_component_fraction:.2f}", fontsize=7)
    ax.set_axis_off()


def draw_surface(ax, h: np.ndarray, world: WorldSpec, vertical_exaggeration: float = 1.5) -> None:
    n = h.shape[0]
    c = np.arange(n) * world.cell_size_m
    xx, yy = np.meshgrid(c, c)
    zz = np.asarray(h) * world.max_height_m
    ax.plot_surface(xx, yy, zz, facecolors=shaded_rgb(h, world), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, vertical_exaggeration * world.max_height_m / world.extent_m))
    ax.set_zlim(0, world.max_height_m)
    ax.view_init(elev=35, azim=-60)
    ax.set_axis_off()


def _save(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_grid(heights: Sequence[np.ndarray], world: WorldSpec, path: str | Path, titles: Sequence[str] | None = None,
              ncols: int = 6, suptitle: str | None = None, contours: bool = False) -> Path:
    n = len(heights)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 1.9 * nrows + (0.3 if suptitle else 0)), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            draw_heightmap(ax, heights[i], world, titles[i] if titles else None, contours=contours)
        else:
            ax.set_axis_off()
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    return _save(fig, path)


def save_surface(h: np.ndarray, world: WorldSpec, path: str | Path, title: str | None = None) -> Path:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    draw_surface(ax, h, world)
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, path)


def save_inspection(h: np.ndarray, world: WorldSpec, path: str | Path, title: str | None = None,
                    agent: AgentSpec = AgentSpec()) -> Path:
    """One map in four views: shaded relief + contours, slope, traversability, 3D surface."""
    fig = plt.figure(figsize=(14, 3.6))
    draw_heightmap(fig.add_subplot(1, 4, 1), h, world, "relief", contours=True)
    draw_slope(fig.add_subplot(1, 4, 2), h, world)
    draw_traversability(fig.add_subplot(1, 4, 3), h, world, agent)
    draw_surface(fig.add_subplot(1, 4, 4, projection="3d"), h, world)
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def save_traversability_grid(heights: Sequence[np.ndarray], world: WorldSpec, path: str | Path, ncols: int = 6,
                             agent: AgentSpec = AgentSpec()) -> Path:
    n = len(heights)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.1 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            draw_traversability(ax, heights[i], world, agent)
        else:
            ax.set_axis_off()
    return _save(fig, path)


def save_metric_histograms(tables: Mapping[str, Mapping[str, np.ndarray]], path: str | Path,
                           keys: Sequence[str] | None = None, ncols: int = 5) -> Path:
    """Overlaid histograms of per-map metrics for several named sets."""
    names = list(tables)
    keys = list(keys or tables[names[0]].keys())
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        allv = np.concatenate([np.asarray(tables[n][key]) for n in names])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1e-6, 40)
        for n in names:
            ax.hist(np.asarray(tables[n][key]), bins=bins, density=True, histtype="step", linewidth=1.3, label=n)
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_axis_off()
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def save_rapsd(sets: Mapping[str, np.ndarray], path: str | Path) -> Path:
    """Mean radially averaged power spectrum (log-log) of several sets of maps."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, maps in sets.items():
        curves = [radial_power_spectrum(h) for h in maps]
        k = curves[0][0]
        logp = np.log10(np.stack([c[1] for c in curves]) + 1e-20)
        mean, std = logp.mean(0), logp.std(0)
        ax.plot(k, mean, label=name)
        ax.fill_between(k, mean - std, mean + std, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("radial frequency (cycles / map)")
    ax.set_ylabel("log10 power")
    ax.legend(fontsize=8)
    ax.set_title("Radially averaged power spectrum", fontsize=9)
    return _save(fig, path)


def save_condition_scatter(requested: np.ndarray, measured: np.ndarray, keys: Sequence[str], path: str | Path,
                           adherence: Mapping[str, Mapping[str, float]] | None = None,
                           baseline: Mapping[str, Mapping[str, float]] | None = None,
                           title: str | None = None) -> Path:
    """Requested vs measured value per conditioning property (y = x is perfect control)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.2), squeeze=False)
    for j, (ax, key) in enumerate(zip(axes[0], keys)):
        x, y = np.asarray(requested)[:, j], np.asarray(measured)[:, j]
        lo, hi = np.percentile(np.concatenate([x, y]), [0.5, 99.5])
        pad = 0.05 * (hi - lo + 1e-9)
        ax.scatter(x, y, s=5, alpha=0.5)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        sub = key
        if adherence and key in adherence:
            a = adherence[key]
            sub += f"\nr={a['pearson_r']:.3f} nMAE={a.get('nmae', float('nan')):.3f}"
            if baseline and key in baseline:
                sub += f" (random {baseline[key].get('nmae', float('nan')):.2f})"
        ax.set_title(sub, fontsize=8)
        ax.set_xlabel("requested", fontsize=7)
        ax.set_ylabel("measured on learned map", fontsize=7)
        ax.tick_params(labelsize=6)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def to_uint8_image(h: np.ndarray, world: WorldSpec) -> np.ndarray:
    return (shaded_rgb(h, world) * 255).astype(np.uint8)
