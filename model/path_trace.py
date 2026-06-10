"""Path tracing on a wave-equation arrival-time field T(x,y,z).

Given a precomputed T from `wave_api.solve_forward`, the optimal
path from source to destination follows the integral curve of `-∇T`.
We trace numerically by streamline (Euler) gradient descent starting at
the destination, until we reach the source cell.

Three trace modes are provided:

  - trace_path_3d           : full 3D descent, simplest case.
  - trace_path_2d_at_z      : descent on a fixed cruise altitude slice.
  - build_three_phase_path  : vertical-climb + 2D cruise + vertical-descent
                              (matches Meituan low-altitude takeoff/landing).

`smooth_path` applies a Gaussian arc-length smoother to remove cell-scale
wiggles before passing the trajectory downstream.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

__all__ = [
    "trace_path_3d",
    "trace_path_2d_at_z",
    "build_three_phase_path",
    "smooth_path",
]


def _prepare_T_grad_3d(T: np.ndarray, dx: float, dy: float, dz: float
                       ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Replace +inf cells with a large finite value, then compute ∇T.

    Returns None if T is all non-finite (no wavefront ever arrived).
    """
    finite = np.isfinite(T)
    if not finite.any():
        return None
    T_max = T[finite].max()
    # Soft fill outside the reached region with a distance ramp so the
    # gradient field still points back toward the finite region.
    from scipy.ndimage import distance_transform_edt
    inf_mask = ~finite
    if inf_mask.any():
        dist_to_finite = distance_transform_edt(inf_mask)
        fill_alpha = T_max / 30.0
        T_safe = np.where(finite, T, T_max + fill_alpha * dist_to_finite)
    else:
        T_safe = T
    gx, gy, gz = np.gradient(T_safe, dx, dy, dz)
    return gx, gy, gz


def trace_path_3d(T, source_xyz, dest_xyz, dx, dy, dz,
                  gamma: float = 0.3,
                  max_steps: int = 10000,
                  max_stale: int = 200
                  ) -> Tuple[np.ndarray, str]:
    """Full 3D gradient descent on T from `dest_xyz` back to `source_xyz`.

    Parameters
    ----------
    T            : (NX, NY, NZ) arrival-time field.
    source_xyz   : (3,) source coordinates in metres.
    dest_xyz     : (3,) destination coordinates in metres.
    dx,dy,dz     : cell sizes in metres.
    gamma        : step size as a fraction of `min(dx,dy,dz)`.
    max_steps    : hard cap on the trace iterations.
    max_stale    : abort if best-so-far doesn't improve for this many steps.

    Returns
    -------
    path  : (M, 3) trajectory in src->dst order (metres).
    status: one of "reached", "max_steps", "truncated_<dist>m", "no_finite".
    """
    grads = _prepare_T_grad_3d(T, dx, dy, dz)
    if grads is None:
        return np.array([source_xyz, dest_xyz], dtype=np.float64), "no_finite"
    gx, gy, gz = grads

    step = gamma * min(dx, dy, dz)
    pos = np.array(dest_xyz, dtype=np.float64)
    src = np.array(source_xyz, dtype=np.float64)
    path = [pos.copy()]
    best_d = np.linalg.norm(pos - src)
    best_idx = 0
    stale = 0
    for _ in range(max_steps):
        coords = np.array([[pos[0]/dx], [pos[1]/dy], [pos[2]/dz]])
        gv = np.array([
            float(map_coordinates(gx, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gy, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gz, coords, order=1, mode="nearest")[0]),
        ])
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            break
        pos = pos - step * gv / gn
        path.append(pos.copy())
        d = np.linalg.norm(pos - src)
        if d < best_d - 0.5 * step:
            best_d = d; best_idx = len(path) - 1; stale = 0
        else:
            stale += 1
        if d < 2.0 * max(dx, dy, dz):
            path.append(src.copy())
            return np.array(path[::-1]), "reached"
        if stale > max_stale:
            path = path[: best_idx + 1]
            path.append(src.copy())
            return np.array(path[::-1]), f"truncated_{best_d:.0f}m"
    return np.array(path[::-1]), "max_steps"


def trace_path_2d_at_z(T, iz, src_xy_m, dst_xy_m, dx, dy,
                       gamma: float = 0.3,
                       max_steps: int = 20000,
                       max_stale: int = 400,
                       smooth_sigma_cells: float = 3.0
                       ) -> Optional[np.ndarray]:
    """Pure 2D gradient descent on T[:, :, iz] (a fixed-altitude slice).

    Used as the "cruise" phase of a three-phase trajectory. Returns
    (M, 2) coordinates in metres, src->dst order, or None if the slice
    has no finite cells.
    """
    T2 = np.asarray(T[:, :, iz])
    finite = np.isfinite(T2)
    if not finite.any():
        return None
    T_max = T2[finite].max()
    T_safe = np.where(finite, T2, T_max + 1e6)
    # Smooth out reflection-induced noise so descent has a clean field.
    T_safe = gaussian_filter(T_safe, sigma=smooth_sigma_cells)
    gx, gy = np.gradient(T_safe, dx, dy)
    step = gamma * min(dx, dy)
    pos = np.asarray(dst_xy_m, dtype=np.float64)
    src = np.asarray(src_xy_m, dtype=np.float64)
    path = [pos.copy()]
    best_d = np.linalg.norm(pos - src)
    best_idx = 0
    stale = 0
    for _ in range(max_steps):
        coords = np.array([[pos[0] / dx], [pos[1] / dy]])
        gv = np.array([
            float(map_coordinates(gx, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gy, coords, order=1, mode="nearest")[0]),
        ])
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            break
        pos = pos - step * gv / gn
        path.append(pos.copy())
        d = np.linalg.norm(pos - src)
        if d < best_d - 0.5 * step:
            best_d = d; best_idx = len(path) - 1; stale = 0
        else:
            stale += 1
        if d < 2.0 * max(dx, dy):
            path.append(src.copy())
            return np.array(path[::-1])
        if stale > max_stale:
            path = path[: best_idx + 1]
            path.append(src.copy())
            return np.array(path[::-1])
    return np.array(path[::-1])


def build_three_phase_path(T, src_xyz_m, dst_xyz_m, dx, dy, dz,
                           cruise_z_m: float = 40.0
                           ) -> Optional[np.ndarray]:
    """Vertical-climb + 2D cruise + vertical-descent.

    Matches the Meituan delivery-UAV operation pattern: takeoff vertically
    to cruise altitude, cruise horizontally on a 2D T-slice, descend
    vertically over the destination.

    Returns (N, 3) path in (x,y,z) metres, or None if the 2D cruise
    slice has no finite cells.
    """
    nz = T.shape[2]
    cruise_iz = int(np.clip(round(cruise_z_m / dz), 1, nz - 1))
    cruise_z_actual = cruise_iz * dz

    xy_path = trace_path_2d_at_z(T, cruise_iz,
                                  src_xyz_m[:2], dst_xyz_m[:2], dx, dy)
    if xy_path is None or len(xy_path) < 2:
        return None
    # Phase 1: vertical climb
    climb_N = max(4, int(round((cruise_z_actual - src_xyz_m[2]) / dz)) + 1)
    climb = np.linspace(src_xyz_m[2], cruise_z_actual, climb_N)
    climb_xyz = np.stack([np.full(climb_N, src_xyz_m[0]),
                          np.full(climb_N, src_xyz_m[1]),
                          climb], axis=1)
    # Phase 2: cruise (constant z)
    cruise_xyz = np.column_stack([xy_path,
                                   np.full(len(xy_path), cruise_z_actual)])
    # Phase 3: vertical descent
    desc_N = max(4, int(round((cruise_z_actual - dst_xyz_m[2]) / dz)) + 1)
    desc = np.linspace(cruise_z_actual, dst_xyz_m[2], desc_N)
    desc_xyz = np.stack([np.full(desc_N, dst_xyz_m[0]),
                         np.full(desc_N, dst_xyz_m[1]),
                         desc], axis=1)
    return np.vstack([climb_xyz, cruise_xyz[1:-1], desc_xyz])


def smooth_path(path_xyz, sigma_m: float = 60.0,
                step_size: Optional[float] = None) -> np.ndarray:
    """Resample to constant arc-length then Gaussian-smooth each coord.

    Removes per-cell wiggles introduced by the discrete gradient field;
    output is a smooth (Dubins-lite) trajectory suitable for handing off
    to a controller.
    """
    if path_xyz.shape[0] < 4:
        return path_xyz
    seg = np.diff(path_xyz, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    if step_size is None:
        step_size = max(50.0, sigma_m / 12.0)
    s_new = np.arange(0.0, arc[-1] + 1e-6, step_size)
    if s_new.size < 4:
        return path_xyz
    rs = np.column_stack([
        np.interp(s_new, arc, path_xyz[:, 0]),
        np.interp(s_new, arc, path_xyz[:, 1]),
        np.interp(s_new, arc, path_xyz[:, 2]),
    ])
    sigma_steps = max(1.5, sigma_m / step_size)
    half = int(np.ceil(3.0 * sigma_steps))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_steps) ** 2)
    k = k / k.sum()
    out = np.empty_like(rs)
    for axis in range(3):
        v = rs[:, axis]
        if half > 0:
            head = v[half:0:-1]
            tail = v[-2:-half - 2:-1]
            padded = np.concatenate([head, v, tail])
            if padded.size != v.size + 2 * half:
                padded = np.pad(v, half, mode="edge")
        else:
            padded = v
        out[:, axis] = np.convolve(padded, k, mode="valid")
    # Pin endpoints so smoothing doesn't drift off src or dst.
    out[0] = path_xyz[0]
    out[-1] = path_xyz[-1]
    return out
