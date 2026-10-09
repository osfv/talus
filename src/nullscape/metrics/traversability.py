"""Traversability analysis for a ground-bound game agent.

Model: a cell is *walkable* if it is land (unless water is passable) and its
slope is at most ``max_slope_deg``. Two neighboring walkable cells are connected
if the height step between them is climbable, i.e. |dz| <= tan(max_slope) * d,
where d is the horizontal distance between cell centers. Connectivity is
computed on that edge graph, so one-cell cliffs split regions even when the
central-difference slope of each cell looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from nullscape.metrics.quality import slope_degrees
from nullscape.world import WorldSpec


@dataclass(frozen=True)
class AgentSpec:
    max_slope_deg: float = 35.0
    water_passable: bool = False
    connectivity: int = 8
    min_land_fraction: float = 0.05
    min_walkable_land_fraction: float = 0.5
    min_largest_component_fraction: float = 0.7

    def __post_init__(self) -> None:
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


@dataclass
class TraversabilityResult:
    land_fraction: float
    walkable_fraction: float
    walkable_land_fraction: float
    n_components: int
    largest_component_fraction: float
    connectivity_probability: float
    spans_map: bool
    passed: bool
    walkable: np.ndarray = field(repr=False)
    water: np.ndarray = field(repr=False)
    largest_component: np.ndarray = field(repr=False)
    component_labels: np.ndarray = field(repr=False)
    graph: csr_matrix = field(repr=False)

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "land_fraction": self.land_fraction,
            "walkable_fraction": self.walkable_fraction,
            "walkable_land_fraction": self.walkable_land_fraction,
            "n_components": self.n_components,
            "largest_component_fraction": self.largest_component_fraction,
            "connectivity_probability": self.connectivity_probability,
            "spans_map": self.spans_map,
            "passed": self.passed,
        }


def _edge_graph(z: np.ndarray, walkable: np.ndarray, cell: float, max_slope_deg: float, connectivity: int) -> csr_matrix:
    rows, cols = z.shape
    ids = np.arange(rows * cols).reshape(rows, cols)
    tan = np.tan(np.radians(max_slope_deg))
    offsets = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if connectivity == 8 else [])
    src, dst = [], []
    for dy, dx in offsets:
        ys = slice(0, rows - dy)
        yd = slice(dy, rows)
        xs = slice(max(0, -dx), cols - max(0, dx))
        xd = slice(max(0, dx), cols - max(0, -dx))
        dist = cell * np.hypot(dy, dx)
        ok = walkable[ys, xs] & walkable[yd, xd] & (np.abs(z[ys, xs] - z[yd, xd]) <= tan * dist)
        src.append(ids[ys, xs][ok])
        dst.append(ids[yd, xd][ok])
    s = np.concatenate(src)
    d = np.concatenate(dst)
    n = rows * cols
    g = coo_matrix((np.ones(2 * s.size, dtype=np.float64), (np.concatenate([s, d]), np.concatenate([d, s]))), shape=(n, n))
    return g.tocsr()


def analyze(h: np.ndarray, world: WorldSpec, agent: AgentSpec = AgentSpec()) -> TraversabilityResult:
    h = np.asarray(h, dtype=np.float64)
    z = h * world.max_height_m
    water = h < world.sea_level
    walkable = slope_degrees(h, world) <= agent.max_slope_deg
    if not agent.water_passable:
        walkable &= ~water

    graph = _edge_graph(z, walkable, world.cell_size_m, agent.max_slope_deg, agent.connectivity)
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(h.shape)

    n_walk = int(walkable.sum())
    land_cells = int((~water).sum()) if not agent.water_passable else h.size
    if n_walk:
        counts = np.bincount(labels[walkable])
        largest = walkable & (labels == counts.argmax())
        sizes = counts[counts > 0]
        n_components = int(sizes.size)
        largest_frac = float(sizes.max() / n_walk)
        conn_prob = float((sizes.astype(np.float64) ** 2).sum() / float(n_walk) ** 2)
        spans = bool((largest[:, 0].any() and largest[:, -1].any()) or (largest[0, :].any() and largest[-1, :].any()))
    else:
        largest = np.zeros_like(walkable)
        n_components, largest_frac, conn_prob, spans = 0, 0.0, 0.0, False

    land_fraction = float(1.0 - water.mean())
    walk_land = float(n_walk / land_cells) if land_cells else 0.0
    passed = (
        land_fraction >= agent.min_land_fraction
        and walk_land >= agent.min_walkable_land_fraction
        and largest_frac >= agent.min_largest_component_fraction
    )
    return TraversabilityResult(
        land_fraction=land_fraction,
        walkable_fraction=float(walkable.mean()),
        walkable_land_fraction=walk_land,
        n_components=n_components,
        largest_component_fraction=largest_frac,
        connectivity_probability=conn_prob,
        spans_map=spans,
        passed=bool(passed),
        walkable=walkable,
        water=water,
        largest_component=largest,
        component_labels=labels,
        graph=graph,
    )


def shortest_path(result: TraversabilityResult, start: tuple[int, int], goal: tuple[int, int],
                  *, distance_weighted: bool = False) -> list[tuple[int, int]] | None:
    """Fewest-steps walkable path between two cells, or None if unreachable."""
    rows, cols = result.walkable.shape
    if any(not (0 <= p[0] < rows and 0 <= p[1] < cols) for p in (start, goal)):
        raise ValueError("path endpoints must be inside the map")
    if not result.walkable[start] or not result.walkable[goal]:
        return None
    s, g = start[0] * cols + start[1], goal[0] * cols + goal[1]
    graph = result.graph
    if distance_weighted:
        edges = graph.tocoo(copy=True)
        edges.data = np.hypot(edges.row // cols - edges.col // cols, edges.row % cols - edges.col % cols)
        graph = edges.tocsr()
    _, pred = dijkstra(graph, indices=s, unweighted=not distance_weighted, return_predecessors=True)
    if s != g and pred[g] < 0:
        return None
    path = [g]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    return [(p // cols, p % cols) for p in reversed(path)]


def longest_route(result: TraversabilityResult) -> list[tuple[int, int]] | None:
    """Approximate diameter path of the largest walkable component (double sweep)."""
    cells = np.flatnonzero(result.largest_component.ravel())
    if cells.size < 2:
        return None
    cols = result.walkable.shape[1]
    dist = dijkstra(result.graph, indices=int(cells[0]), unweighted=True)
    a = int(cells[np.argmax(dist[cells])])
    dist = dijkstra(result.graph, indices=a, unweighted=True)
    b = int(cells[np.argmax(dist[cells])])
    return shortest_path(result, (a // cols, a % cols), (b // cols, b % cols))


def gameplay_metrics(h: np.ndarray, world: WorldSpec, agent: AgentSpec = AgentSpec(), *,
                     start: tuple[int, int] | None = None, goal: tuple[int, int] | None = None,
                     combat_radius_m: float = 32.0, combat_slope_deg: float = 10.0) -> dict[str, float]:
    if not np.isfinite(combat_radius_m) or combat_radius_m <= 0:
        raise ValueError("combat_radius_m must be finite and positive")
    if not np.isfinite(combat_slope_deg) or not 0 <= combat_slope_deg <= 90:
        raise ValueError("combat_slope_deg must be between 0 and 90")
    result = analyze(h, world, agent)
    r, cell = world.resolution, world.cell_size_m
    start = (r // 2, 0) if start is None else tuple(start)
    goal = (r // 2, r - 1) if goal is None else tuple(goal)
    path = shortest_path(result, start, goal, distance_weighted=True)
    clearance = distance_transform_edt(np.pad(result.walkable, 1))[1:-1, 1:-1] * cell
    flat = result.largest_component & (slope_degrees(h, world) <= combat_slope_deg)
    flat_clearance = distance_transform_edt(np.pad(flat, 1))[1:-1, 1:-1] * cell
    combat = flat & (flat_clearance >= combat_radius_m)
    length, stretch, width, bottleneck = 0.0, 0.0, 0.0, 0.0
    if path:
        p = np.asarray(path)
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * cell)
        direct = float(np.linalg.norm(np.asarray(goal) - start) * cell)
        stretch = length / direct if direct else 1.0
        widths = 2 * clearance[p[:, 0], p[:, 1]]
        width = float(widths.min())
        interior = widths[1:-1] if len(widths) > 2 else widths
        bottleneck = float((interior < 4 * cell).mean())
    return {"spawn_goal_reachable": float(path is not None), "route_length_m": length,
            "route_stretch": stretch, "route_min_width_m": width, "route_bottleneck_fraction": bottleneck,
            "combat_space_fraction": float(combat.mean()),
            "combat_space_m2": float(combat.sum() * cell ** 2)}
