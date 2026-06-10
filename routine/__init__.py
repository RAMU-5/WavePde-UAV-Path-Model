"""routine — neural-physics station-to-station path planner.

Pipeline:

    mask + (src, dst)
        ── routine.wave ──>  T(x,y,z) arrival-time field
        ── routine.path ──>  streamline trace on -∇T
        ── routine.plan ──>  one-call wrapper

Reference: `neural_physics_uav_flight_path_optimisation/main_article.tex`.

Quick start:

    from routine import plan_path, WaveConfig
    res = plan_path(mask, dx, dy, dz, src_xyz, dst_xyz)
    print(res.travel_time, res.path.shape, res.status)
"""
from __future__ import annotations

__version__ = "0.1.0"

from .wave import (
    WaveConfig,
    StaticFields,
    build_static_fields,
    gaussian_source,
    solve_one,
    solve_forward,
    solve_backward,
)
from .path import (
    trace_path_3d,
    trace_path_2d_at_z,
    build_three_phase_path,
    smooth_path,
)
from .plan import (
    PathResult,
    snap_to_free_cell,
    plan_path,
)

__all__ = [
    "__version__",
    # wave PDE
    "WaveConfig", "StaticFields",
    "build_static_fields", "gaussian_source",
    "solve_one", "solve_forward", "solve_backward",
    # path tracing
    "trace_path_3d", "trace_path_2d_at_z",
    "build_three_phase_path", "smooth_path",
    # high-level
    "PathResult", "snap_to_free_cell", "plan_path",
]
