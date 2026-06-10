"""Incremental FMM updates triggered by environmental change.

Stage 1 only handles hazard-driven cylindrical no-fly insertions. The strategy
is "affected-zone resolve": find the source set whose nominal trajectory
passes within `buffer_m` of the new no-fly cylinder, patch rho, and re-solve
only those sources.

This is intentionally simpler than D* Lite. R2 regression in `aura.tests`
verifies it agrees with a full re-solve to <1 m Hausdorff.
"""

from __future__ import annotations

from typing import Iterable, List, Set

import numpy as np

from aura.cost.metric import RanderField, patch_rho_for_zone
from aura.fmm.agd_solver import (
    AllPairsDB,
    extract_geodesic,
    solve_single_source,
)
from aura.geom.grid import SimGrid
from aura.geom.noflyzone import CylinderZone, NoFlyRegistry


def _trajectory_intersects_zone(traj_xyz: List, zone: CylinderZone,
                                buffer_m: float) -> bool:
    if not traj_xyz:
        return False
    xy = np.asarray([(p[0], p[1]) for p in traj_xyz], dtype=np.float64)
    d2 = (xy[:, 0] - zone.cx) ** 2 + (xy[:, 1] - zone.cy) ** 2
    r2 = (zone.radius_m + buffer_m) ** 2
    return bool((d2 <= r2).any())


def affected_sources(db: AllPairsDB, zone: CylinderZone,
                     buffer_m: float = 50.0) -> Set[int]:
    out: Set[int] = set()
    for (src, dst), traj in db.pairs.items():
        if _trajectory_intersects_zone(traj.xyz, zone, buffer_m):
            out.add(src)
    return out


def affected_pairs(db: AllPairsDB, zone: CylinderZone,
                   buffer_m: float = 50.0) -> List[tuple]:
    return [k for k, traj in db.pairs.items()
            if _trajectory_intersects_zone(traj.xyz, zone, buffer_m)]


def resolve_affected(field: RanderField, grid: SimGrid,
                     db: AllPairsDB,
                     no_fly: NoFlyRegistry,
                     new_zone: CylinderZone,
                     station_xys: np.ndarray,
                     station_z_m: float,
                     buffer_m: float = 50.0,
                     trajectory_dt_s: float = 0.5,
                     backend: str = "cpu",
                     now_t_s: float = 0.0) -> AllPairsDB:
    """In-place re-solve: keep db, replace affected entries.

    Returns the same db object with `metric_version` bumped.
    """
    sources = affected_sources(db, new_zone, buffer_m)
    if not sources:
        return db
    # Apply rho patch (does not mutate field — returns a new one).
    patched = patch_rho_for_zone(field, grid, no_fly, now_t_s,
                                 alpha_obstacle=1e6)
    n = len(station_xys)
    for src in sources:
        sx, sy = station_xys[src]
        src_xyz = (float(sx), float(sy), float(station_z_m))
        arrival = solve_single_source(patched, grid, src_xyz, backend=backend)
        db.arrivals[src] = arrival
        for did in range(n):
            if did == src:
                continue
            dx, dy = station_xys[did]
            traj = extract_geodesic(arrival, grid,
                                    (float(dx), float(dy), float(station_z_m)),
                                    trajectory_dt_s=trajectory_dt_s,
                                    v_cruise=patched.v_cruise)
            traj.src = src
            traj.dst = did
            db.pairs[(src, did)] = traj
    db.metric_version = patched.metric_version
    return db


def hausdorff(a_xyz: List, b_xyz: List) -> float:
    """Symmetric Hausdorff distance between two polylines, used by R2."""
    A = np.asarray(a_xyz, dtype=np.float64)
    B = np.asarray(b_xyz, dtype=np.float64)
    if A.size == 0 or B.size == 0:
        return float("inf")
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))
