"""High-level station-to-station path planner.

`plan_path(mask, dx,dy,dz, src_xyz, dst_xyz)` is the one-shot entrypoint:

    1. Build static wave fields (or reuse a cached `StaticFields`).
    2. Snap source / destination to integer cell indices, nudging out
       of any building cell horizontally then vertically.
    3. Solve the convected wave PDE from the source (`routine.wave`).
    4. Trace the optimal path back from destination via gradient descent
       on T (`routine.path`).
    5. Read off the travel time at the destination cell.

Returns a `PathResult` with the path, travel time, status string, and
(optionally) the full T-field for downstream visualisation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np

from .wave import (
    WaveConfig,
    StaticFields,
    build_static_fields,
    solve_forward,
)
from .path import (
    trace_path_3d,
    build_three_phase_path,
    smooth_path,
)

__all__ = ["PathResult", "snap_to_free_cell", "plan_path"]


@dataclass
class PathResult:
    path: np.ndarray             # (M, 3) trajectory in (x, y, z) metres
    travel_time: float           # seconds from src to dst (T at dst cell)
    status: str                  # "reached" / "max_steps" / "no_finite" / ...
    T: Optional[np.ndarray]      # full (NX,NY,NZ) arrival-time field, if kept
    src_idx: Tuple[int, int, int]
    dst_idx: Tuple[int, int, int]


def snap_to_free_cell(mask: np.ndarray, xyz_m, dx, dy, dz,
                      max_radius: int = 5) -> Tuple[int, int, int]:
    """Round metres → cell idx, nudging out of buildings.

    First search horizontally in expanding square rings (last-mile pickup
    must stay on the ground); if no free cell within `max_radius`, climb
    vertically until a free cell is found.
    """
    nx, ny, nz = mask.shape
    ix = int(np.clip(round(xyz_m[0] / dx), 1, nx - 2))
    iy = int(np.clip(round(xyz_m[1] / dy), 1, ny - 2))
    iz = int(np.clip(round(xyz_m[2] / dz), 1, nz - 2))
    if not mask[ix, iy, iz]:
        return ix, iy, iz
    # Horizontal nudge: walk expanding square shells.
    for r in range(1, max_radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                nix = ix + di; niy = iy + dj
                if not (0 < nix < nx - 1 and 0 < niy < ny - 1):
                    continue
                if not mask[nix, niy, iz]:
                    return nix, niy, iz
    # Fall back to vertical climb.
    iz2 = iz
    while iz2 < nz - 1 and mask[ix, iy, iz2]:
        iz2 += 1
    return ix, iy, iz2


def plan_path(mask: np.ndarray,
              dx: float, dy: float, dz: float,
              src_xyz: Tuple[float, float, float],
              dst_xyz: Tuple[float, float, float],
              *,
              cfg: Optional[WaveConfig] = None,
              static: Optional[StaticFields] = None,
              mode: Literal["3d", "three_phase"] = "3d",
              cruise_z_m: float = 40.0,
              smooth_sigma_m: Optional[float] = None,
              return_T: bool = False,
              ) -> PathResult:
    """Plan a flight path from `src_xyz` to `dst_xyz` over a 3D building mask.

    Parameters
    ----------
    mask    : (NX,NY,NZ) bool — True = building / no-fly cell.
    dx,dy,dz: cell sizes in metres.
    src_xyz, dst_xyz : station positions in metres (x, y, z).
    cfg     : `WaveConfig`. Defaults to Meituan delivery-UAV regime.
    static  : pre-built `StaticFields`. Pass this when planning many
              paths over the same domain to amortise the build cost
              (and the JAX compile cost).
    mode    :
        "3d"          → straight gradient descent on the full 3D T-field.
        "three_phase" → vertical climb + 2D cruise on T[:,:,cruise_iz]
                        + vertical descent. Matches low-altitude Meituan
                        ops with a regulatory cruise altitude.
    cruise_z_m : cruise altitude for "three_phase" mode (m above ground).
    smooth_sigma_m : if set, apply `smooth_path` with this Gaussian sigma.
    return_T : keep the full (NX,NY,NZ) T-field in the result (useful
               for visualisation / debugging; otherwise discarded).

    Returns
    -------
    `PathResult` with:
      - `path`         (M, 3) numpy trajectory in metres, src→dst order
      - `travel_time`  seconds from source to destination
      - `status`       trace exit reason
      - `T`            full arrival-time field if `return_T=True`, else None
      - `src_idx`, `dst_idx`  snapped integer cell indices
    """
    if cfg is None:
        cfg = WaveConfig()

    src_idx = snap_to_free_cell(mask, src_xyz, dx, dy, dz)
    dst_idx = snap_to_free_cell(mask, dst_xyz, dx, dy, dz)

    if static is None:
        static = build_static_fields(mask, dx, dy, dz, cfg=cfg)

    T = solve_forward(mask, dx, dy, dz, src_idx, cfg=cfg, static=static)

    # Trace
    if mode == "three_phase":
        path = build_three_phase_path(T, src_xyz, dst_xyz, dx, dy, dz,
                                       cruise_z_m=cruise_z_m)
        status = "three_phase" if path is not None else "no_finite"
        if path is None:
            path = np.array([src_xyz, dst_xyz], dtype=np.float64)
    else:
        path, status = trace_path_3d(T, src_xyz, dst_xyz, dx, dy, dz)

    if smooth_sigma_m is not None and path.shape[0] >= 4:
        path = smooth_path(path, sigma_m=smooth_sigma_m)

    travel_time = float(T[dst_idx[0], dst_idx[1], dst_idx[2]])

    return PathResult(
        path=path,
        travel_time=travel_time,
        status=status,
        T=T if return_T else None,
        src_idx=src_idx,
        dst_idx=dst_idx,
    )
