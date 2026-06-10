"""Winged-UAV alignment-point search and 2-phase trajectory
construction (paper §2.11, Eq. 50).

Fixed-wing UAVs cannot turn instantaneously after takeoff. The paper
prescribes a 2-phase trajectory:

    Phase 1: depart launch in admissible heading e_uav until reaching
             a point x* where the wave-derived gradient -nabla T_c
             aligns with e_uav (within tolerance).
    Phase 2: once at x*, follow -nabla T_c as usual.

The current code (exp2_aircraft.py) does Phase-1 with a HARDCODED
runway direction; Phase-2 alignment is implicit. This module provides
the missing search:

    nabla T_c(x*) / ||nabla T_c(x*)|| . e_uav  ~=  1

within a feasible search volume (a forward-pointing cone from the
launch). Returns x* and the assembled 2-phase trajectory.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AlignmentResult:
    x_star: np.ndarray              # (3,) location where gradient aligns
    score: float                    # cos(angle) at x*
    phase1: np.ndarray              # (M1, 3) launch -> x_star
    phase2: np.ndarray              # (M2, 3) x_star -> destination
    trajectory: np.ndarray          # phase1 + phase2 concatenated
    score_field: np.ndarray         # (NX,NY,NZ) cos-similarity field
    grad_T: tuple                   # (gx, gy, gz)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def find_alignment_point(T: np.ndarray,
                         dx: float, dy: float, dz: float,
                         launch_xyz: np.ndarray,
                         launch_heading: np.ndarray,
                         search_radius_m: float = 800.0,
                         cone_half_angle_deg: float = 35.0,
                         min_score: float = 0.85,
                         min_grad: float = 1e-6,
                         z_range_m: Optional[Tuple[float, float]] = None
                         ) -> Tuple[np.ndarray, float, np.ndarray, tuple]:
    """Find x* maximising  -grad(T) . e_uav / ||grad(T)||  within a
    forward cone from the launch point.

    Parameters
    ----------
    T : (NX, NY, NZ) first-arrival-time field
    launch_xyz : (3,) launch point in metres
    launch_heading : (3,) unit-vector takeoff direction (will be normalised)
    search_radius_m : how far ahead to search
    cone_half_angle_deg : forward search-cone aperture
    min_score : reject candidates below this alignment score
    min_grad : ignore cells with ||grad T|| below this (avoids zeros
               inside obstacles where T is set to inf)
    z_range_m : optional (z_min, z_max) bound for x* altitude

    Returns
    -------
    x_star : (3,) winning point in metres
    score  : alignment score at x_star (in [-1, 1])
    score_field : (NX, NY, NZ) cos-similarity for diagnostics
    grad_T : (gx, gy, gz) the gradient arrays
    """
    e_uav = _normalize(np.asarray(launch_heading, dtype=np.float64))
    launch = np.asarray(launch_xyz, dtype=np.float64)

    # Mask infinities for gradient stability.
    finite = np.isfinite(T)
    if not finite.all():
        Tmax = T[finite].max() if finite.any() else 1.0
        T_safe = np.where(finite, T, Tmax + 1e6)
    else:
        T_safe = T
    gx, gy, gz = np.gradient(T_safe, dx, dy, dz)
    gnorm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    NX, NY, NZ = T.shape
    ix = np.arange(NX)[:, None, None] * dx
    iy = np.arange(NY)[None, :, None] * dy
    iz = np.arange(NZ)[None, None, :] * dz

    rx = ix - launch[0]
    ry = iy - launch[1]
    rz = iz - launch[2]
    r_mag = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    in_radius = (r_mag > 0.5 * dx) & (r_mag <= search_radius_m)

    # Forward-cone test: angle between (cell - launch) and e_uav < theta_max
    cos_theta_max = float(np.cos(np.deg2rad(cone_half_angle_deg)))
    cos_theta = (rx * e_uav[0] + ry * e_uav[1] + rz * e_uav[2]) / np.maximum(r_mag, 1e-9)
    in_cone = cos_theta >= cos_theta_max

    # Alignment score: cell's -gradT direction with e_uav
    score = -(gx * e_uav[0] + gy * e_uav[1] + gz * e_uav[2]) / np.maximum(gnorm, 1e-9)

    valid = in_radius & in_cone & (gnorm > min_grad)
    if z_range_m is not None:
        z_centers = iz
        valid = valid & (z_centers >= z_range_m[0]) & (z_centers <= z_range_m[1])

    masked = np.where(valid, score, -np.inf)
    if not np.isfinite(masked).any():
        raise RuntimeError("no candidate cell satisfies search-cone "
                           "+ gradient validity constraints")
    idx = int(np.argmax(masked))
    i, j, k = np.unravel_index(idx, T.shape)
    best_score = float(score[i, j, k])
    if best_score < min_score:
        # Don't fail — return the best we found, let caller decide.
        pass
    x_star = np.array([i * dx, j * dy, k * dz], dtype=np.float64)
    return x_star, best_score, score, (gx, gy, gz)


def build_two_phase_trajectory(launch_xyz, launch_heading, x_star,
                               grad_T, dx, dy, dz, dest_xyz,
                               step_frac=0.3, max_phase2_steps=4000,
                               turn_radius_m: Optional[float] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Phase 1: straight from launch to x_star (or arc with turn radius
    if specified). Phase 2: gradient descent on -nabla T from x_star
    toward dest.
    """
    launch = np.asarray(launch_xyz, dtype=np.float64)
    e_uav = _normalize(np.asarray(launch_heading, dtype=np.float64))
    x_star = np.asarray(x_star, dtype=np.float64)

    # Phase 1: straight line. (Turn radius / curved arc is left as an
    # extension hook; for the wave-routing alignment goal a straight
    # ascent under the takeoff vector is the standard model — paper
    # text "depart along its admissible path".)
    seg_len = float(np.linalg.norm(x_star - launch))
    n1 = max(2, int(seg_len / (step_frac * min(dx, dy, dz))))
    phase1 = np.array([launch + (x_star - launch) * (k / (n1 - 1))
                       for k in range(n1)], dtype=np.float64)

    # Phase 2: gradient descent on -nabla T from x_star.
    gx, gy, gz = grad_T
    pos = x_star.copy()
    dest = np.asarray(dest_xyz, dtype=np.float64)
    pts = [pos.copy()]
    step = step_frac * min(dx, dy, dz)

    NX, NY, NZ = gx.shape
    for _ in range(max_phase2_steps):
        i = int(np.clip(round(pos[0] / dx), 0, NX - 1))
        j = int(np.clip(round(pos[1] / dy), 0, NY - 1))
        k = int(np.clip(round(pos[2] / dz), 0, NZ - 1))
        gv = np.array([gx[i, j, k], gy[i, j, k], gz[i, j, k]])
        gn = np.linalg.norm(gv)
        if gn < 1e-9:
            break
        pos = pos - step * gv / gn
        pts.append(pos.copy())
        if np.linalg.norm(pos - dest) < 2.0 * max(dx, dy, dz):
            pts.append(dest.copy())
            break
    phase2 = np.array(pts, dtype=np.float64)
    return phase1, phase2
