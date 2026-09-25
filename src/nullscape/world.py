"""Physical interpretation of a normalized heightmap.

A heightmap is a float array in [0, 1]. ``WorldSpec`` maps it to world units:
height_m = h * max_height_m, and the map spans ``extent_m`` meters on each side.
Normalization is global (shared by every map of a dataset), never per map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class WorldSpec:
    resolution: int = 64
    extent_m: float = 4096.0
    max_height_m: float = 1200.0
    sea_level: float = 0.2

    def __post_init__(self) -> None:
        if self.resolution < 4:
            raise ValueError(f"resolution must be >= 4, got {self.resolution}")
        if self.extent_m <= 0 or self.max_height_m <= 0:
            raise ValueError("extent_m and max_height_m must be positive")
        if not 0.0 <= self.sea_level <= 1.0:
            raise ValueError(f"sea_level must be in [0, 1], got {self.sea_level}")

    @property
    def cell_size_m(self) -> float:
        return self.extent_m / self.resolution

    def with_resolution(self, resolution: int) -> "WorldSpec":
        return WorldSpec(resolution, self.extent_m, self.max_height_m, self.sea_level)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cell_size_m"] = self.cell_size_m
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorldSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
