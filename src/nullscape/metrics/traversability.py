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


def shortest_path(result: TraversabilityResult, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Fewest-steps walkable path between two cells, or None if unreachable."""
    rows, cols = result.walkable.shape
    s, g = start[0] * cols + start[1], goal[0] * cols + goal[1]
    _, pred = dijkstra(result.graph, indices=s, unweighted=True, return_predecessors=True)
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
