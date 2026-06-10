"""SimGrid: the simulation's projected-coordinate axis-aligned grid.

The grid is the canonical discretization for FMM, masks, cost field, and
visualization. It is defined in EPSG:32650 (UTM 50N) meters. World <-> grid
conversion is vectorized and pure numpy so no GIS dependency is needed at
runtime.

Convention (matches aura/README.md):
- arrays shape (nx, ny[, nz]); axes ("x", "y", "z").
- index (ix, iy, iz) maps to world center
    (origin_x + (ix + 0.5) * dx, origin_y + (iy + 0.5) * dy,
     origin_z + (iz + 0.5) * dz).
- z=0 is the projected ground plane; z grows upward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SimGrid:
    origin_xy: Tuple[float, float]
    size_xy: Tuple[float, float]
    dx: float
    dy: float
    z_min: float
    z_max: float
    dz: float
    crs: str = "EPSG:32650"

    @property
    def nx(self) -> int:
        return int(np.ceil(self.size_xy[0] / self.dx))

    @property
    def ny(self) -> int:
        return int(np.ceil(self.size_xy[1] / self.dy))

    @property
    def nz(self) -> int:
        return int(np.ceil((self.z_max - self.z_min) / self.dz))

    @property
    def shape3(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def shape2(self) -> Tuple[int, int]:
        return (self.nx, self.ny)

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """xyz: (..., 3) in projected meters; returns (..., 3) integer indices.

        Out-of-bounds indices are clipped to the valid range.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        ix = np.floor((xyz[..., 0] - self.origin_xy[0]) / self.dx).astype(np.int64)
        iy = np.floor((xyz[..., 1] - self.origin_xy[1]) / self.dy).astype(np.int64)
        iz = np.floor((xyz[..., 2] - self.z_min) / self.dz).astype(np.int64)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)
        return np.stack([ix, iy, iz], axis=-1)

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=np.float64)
        x = self.origin_xy[0] + (ijk[..., 0] + 0.5) * self.dx
        y = self.origin_xy[1] + (ijk[..., 1] + 0.5) * self.dy
        z = self.z_min + (ijk[..., 2] + 0.5) * self.dz
        return np.stack([x, y, z], axis=-1)

    def cell_centers_xy(self) -> np.ndarray:
        """Return (nx, ny, 2) world-coordinate centers of every horizontal cell."""
        ix = np.arange(self.nx)
        iy = np.arange(self.ny)
        gx, gy = np.meshgrid(ix, iy, indexing="ij")
        x = self.origin_xy[0] + (gx + 0.5) * self.dx
        y = self.origin_xy[1] + (gy + 0.5) * self.dy
        return np.stack([x, y], axis=-1)

    @classmethod
    def from_bbox_meters(cls, min_x: float, min_y: float, max_x: float,
                        max_y: float, dx: float, dy: float,
                        z_min: float, z_max: float, dz: float,
                        crs: str = "EPSG:32650") -> "SimGrid":
        return cls(
            origin_xy=(min_x, min_y),
            size_xy=(max_x - min_x, max_y - min_y),
            dx=dx, dy=dy,
            z_min=z_min, z_max=z_max, dz=dz,
            crs=crs,
        )
