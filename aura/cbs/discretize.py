"""Discretize a NominalTrajectory into a (cell, time-bin) reservation table.

Per Stage 1 plan (ε-tube swept volume, 1 s time bins): for each sample on the
trajectory we enumerate the lattice of horizontal cells inside ε around the
sample, tagged by the integer time-bin index since launch. The reservation
table keys are `(ix, iy, t_bin)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from aura.schemas import NominalTrajectory


@dataclass
class ReservationTable:
    """time-bin granularity reservation across the discretized trunk space."""
    bin_s: float = 1.0
    table: Dict[Tuple[int, int, int], List[int]] = field(default_factory=dict)

    def add(self, key: Tuple[int, int, int], mission_id: int) -> None:
        self.table.setdefault(key, []).append(mission_id)

    def conflicts(self) -> List[Tuple[Tuple[int, int, int], List[int]]]:
        return [(k, v) for k, v in self.table.items() if len(v) > 1]


def cells_along_trajectory(nominal: NominalTrajectory, grid,
                           eps_m: float, bin_s: float = 1.0,
                           t_launch_s: float = 0.0) -> List[Tuple[int, int, int]]:
    """Return the list of `(ix, iy, t_bin)` reservation keys for one mission.

    Trajectory samples are inflated to ε in horizontal plane; vertical extent
    is collapsed since Stage 1 has a single cruise altitude.
    """
    if not nominal.xyz:
        return []
    pts = np.asarray(nominal.xyz, dtype=np.float64)
    out: List[Tuple[int, int, int]] = []
    radius_cells = int(np.ceil(eps_m / max(grid.dx, grid.dy)))
    for k, (x, y, z) in enumerate(pts):
        t_bin = int((t_launch_s + k * nominal.dt_s) / bin_s)
        center_ijk = grid.world_to_index(np.array([[x, y, z]]))[0]
        cx, cy = int(center_ijk[0]), int(center_ijk[1])
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                if di * di + dj * dj > radius_cells * radius_cells:
                    continue
                ix = cx + di
                iy = cy + dj
                if 0 <= ix < grid.nx and 0 <= iy < grid.ny:
                    out.append((ix, iy, t_bin))
    return out


def reserve_mission(table: ReservationTable, mission_id: int,
                    keys: List[Tuple[int, int, int]]) -> None:
    for k in keys:
        table.add(k, mission_id)
