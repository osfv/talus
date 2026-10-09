import numpy as np
import pytest

from nullscape.inference.spatial import build_constraints
from nullscape.metrics.traversability import analyze, shortest_path
from nullscape.world import WorldSpec

W = WorldSpec(resolution=16, extent_m=128, max_height_m=40)


def test_route_and_flat_area_constraints():
    layout = {"routes": [{"points": [[3, 2], [3, 10], [12, 10]], "height": 0.4, "width_cells": 3}],
              "flat_areas": [{"center": [10, 4], "radius_cells": 2, "height": 0.5}]}
    h, mask = build_constraints(W, layout=layout)
    assert mask[3, 2:11].all() and mask[3:13, 10].all()
    assert (h[3, 2:11] == np.float32(0.4)).all()
    assert h[10, 4] == np.float32(0.5) and mask[10, 4]
    terrain = np.where(mask, h, 0.0)
    assert shortest_path(analyze(terrain, W), (3, 2), (12, 10)) is not None


@pytest.mark.parametrize("side", ["west", "east", "north", "south"])
def test_neighbor_overlap(side):
    neighbor = np.linspace(0.1, 0.7, 256, dtype=np.float32).reshape(16, 16)
    h, mask = build_constraints(W, neighbors={side: neighbor}, overlap=3)
    slices = {"west": ((slice(None), slice(0, 3)), (slice(None), slice(-3, None))),
              "east": ((slice(None), slice(-3, None)), (slice(None), slice(0, 3))),
              "north": ((slice(0, 3), slice(None)), (slice(-3, None), slice(None))),
              "south": ((slice(-3, None), slice(None)), (slice(0, 3), slice(None)))}
    dst, src = slices[side]
    assert np.array_equal(h[dst], neighbor[src]) and mask[dst].all()
    assert mask.sum() == 48


def test_conflicting_constraints_rejected():
    reference = np.full((16, 16), 0.6, np.float32)
    with pytest.raises(ValueError, match="conflict"):
        build_constraints(W, reference=reference, preserve_mask=np.ones_like(reference, bool),
                          layout={"flat_areas": [{"center": [8, 8], "radius_cells": 2, "height": 0.4}]})
    with pytest.raises(ValueError):
        build_constraints(W, layout={"routes": [{"points": [[0, 0], [20, 1]], "height": 0.4}]})
    with pytest.raises(ValueError):
        build_constraints(W, layout={"routes": [{"points": [[0, 0], [10, 1]], "height": 0.1}]})
    with pytest.raises(ValueError):
        build_constraints(W, neighbors={"west": reference}, overlap=0)
    with pytest.raises(ValueError):
        build_constraints(W, layout={"unknown": []})
