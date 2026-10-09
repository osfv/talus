from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

from nullscape.world import WorldSpec


def build_constraints(world: WorldSpec, *, layout: Mapping | None = None,
                      reference: np.ndarray | None = None, preserve_mask: np.ndarray | None = None,
                      neighbors: Mapping[str, np.ndarray] | None = None, overlap: int = 4
                      ) -> tuple[np.ndarray, np.ndarray]:
    r = world.resolution
    shape = (r, r)
    heights = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    def merge(values, selected):
        values = np.broadcast_to(np.asarray(values, dtype=np.float32), shape)
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != shape or not np.isfinite(values[selected]).all():
            raise ValueError("constraint arrays must match the world and contain finite heights")
        if ((values[selected] < 0) | (values[selected] > 1)).any():
            raise ValueError("constraint heights must be in [0,1]")
        both = mask & selected
        if not np.array_equal(heights[both], values[both]):
            raise ValueError("spatial constraints conflict on overlapping cells")
        heights[selected] = values[selected]
        mask[selected] = True

    if (reference is None) != (preserve_mask is None):
        raise ValueError("reference and preserve_mask must be provided together")
    if reference is not None:
        if np.shape(reference) != shape or np.shape(preserve_mask) != shape:
            raise ValueError("reference and preserve_mask must have shape [R,R]")
        if not np.isin(preserve_mask, [0, 1]).all():
            raise ValueError("preserve_mask must contain booleans or 0/1")
        merge(reference, preserve_mask)
    layout = dict(layout or {})
    if set(layout) - {"routes", "flat_areas"}:
        raise ValueError("layout keys must be routes and/or flat_areas")

    def points(value, minimum):
        p = np.asarray(value, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < minimum or not np.isfinite(p).all():
            raise ValueError("points must be a list of [row,column] cell coordinates")
        if ((p < 0) | (p > r - 1)).any():
            raise ValueError("points must lie inside the map")
        return p

    yy, xx = np.mgrid[:r, :r]
    for route in layout.get("routes", []):
        if set(route) - {"points", "height", "width_cells"}:
            raise ValueError("route keys must be points, height, width_cells")
        p = points(route["points"], 2)
        width = route.get("width_cells", 3)
        if not isinstance(width, int) or width < 3 or width > r or width % 2 != 1:
            raise ValueError("route width_cells must be an odd integer between 3 and resolution")
        height = float(route["height"])
        if not np.isfinite(height) or not world.sea_level < height <= 1:
            raise ValueError("route height must be above sea level and at most 1")
        stroke = np.zeros(shape, dtype=bool)
        for a, b in zip(p[:-1], p[1:]):
            line = np.rint(np.linspace(a, b, max(2, int(np.max(np.abs(b - a)) * 2) + 1))).astype(int)
            stroke[line[:, 0], line[:, 1]] = True
        merge(height, distance_transform_edt(~stroke) <= width // 2)
    for area in layout.get("flat_areas", []):
        if set(area) - {"center", "height", "radius_cells"}:
            raise ValueError("flat area keys must be center, height, radius_cells")
        cy, cx = points([area["center"]], 1)[0]
        radius = float(area.get("radius_cells", 3))
        if not np.isfinite(radius) or not 1 <= radius <= r:
            raise ValueError("radius_cells must be between 1 and resolution")
        merge(area["height"], np.hypot(yy - cy, xx - cx) <= radius)
    if neighbors:
        if not isinstance(overlap, int) or not 1 <= overlap < r:
            raise ValueError("overlap must be an integer between 1 and resolution - 1")
        slices = {"west": ((slice(None), slice(0, overlap)), (slice(None), slice(-overlap, None))),
                  "east": ((slice(None), slice(-overlap, None)), (slice(None), slice(0, overlap))),
                  "north": ((slice(0, overlap), slice(None)), (slice(-overlap, None), slice(None))),
                  "south": ((slice(-overlap, None), slice(None)), (slice(0, overlap), slice(None)))}
        for side, neighbor in neighbors.items():
            if side not in slices or np.shape(neighbor) != shape:
                raise ValueError("neighbors must be north/south/east/west heightmaps matching the world")
            dst, src = slices[side]
            values, selected = np.zeros(shape, np.float32), np.zeros(shape, bool)
            values[dst], selected[dst] = np.asarray(neighbor)[src], True
            merge(values, selected)
    return heights, mask
