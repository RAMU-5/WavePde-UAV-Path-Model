"""Build the Rander metric tensor field for the AGD anisotropic FMM.

Eq. (4) of the Aura plan: ||grad T(x)||_M(x,t) = 1.
The metric M(x,t) encodes:
  - UAV cruise speed bound v_cruise (-> Riemann part).
  - Wind w(x,t) advecting the airframe (-> Rander offset).
  - Building / no-fly mask (-> rho multiplier; near-infinite inside walls).
  - Population noise penalty (-> mild rho bump).

In Stage 1 wind is constant, so the Rander offset is uniform and the metric is
"piecewise homogeneous" — only rho varies across cells. We still emit the full
field so the consumer is identical when SCALEDWind replaces ConstantWind in
Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from aura.wind.base import IWindField
from aura.geom.noflyzone import NoFlyRegistry


@dataclass
class RanderField:
    """AGD-compatible Rander metric field.

    For Riemannian part we use the isotropic case M = (1/v_cruise) I, scaled
    by rho. For the Rander offset omega we encode the wind: omega = -w/v_cruise^2.
    AGD's Rander solver minimizes integral of (sqrt(g_ij dx^i dx^j) + omega_i dx^i).
    With w << v_cruise this is the standard Zermelo metric.
    """
    riemann_diag: np.ndarray       # (nx, ny, nz, 3) — isotropic = same value across last dim
    rander_omega: np.ndarray       # (nx, ny, nz, 3)
    rho: np.ndarray                # (nx, ny, nz) scalar density multiplier
    walls: Optional[np.ndarray]    # (nx, ny, nz) bool — passed to AGD walls flag
    v_cruise: float
    metric_version: int = 0


def build_rho(grid, density: np.ndarray, height_mask: np.ndarray,
              alpha_noise: float, alpha_obstacle: float,
              no_fly: NoFlyRegistry | None = None,
              now_t_s: float = 0.0) -> np.ndarray:
    """Compose the scalar rho(x) factor on the grid.

    rho = 1 + alpha_noise * pop_density + alpha_obstacle * obstacle_mask
    """
    nx, ny, nz = grid.shape3
    rho = np.ones((nx, ny, nz), dtype=np.float32)

    # population noise penalty: broadcast (nx, ny) -> (nx, ny, nz).
    if alpha_noise > 0.0 and density is not None:
        rho += (alpha_noise * density.astype(np.float32))[:, :, None]

    # building obstacles: cell i,j blocked at all altitudes z < height_mask[i,j].
    if alpha_obstacle > 0.0 and height_mask is not None:
        z_centers = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
        block = (z_centers[None, None, :] < height_mask[:, :, None]).astype(np.float32)
        rho += alpha_obstacle * block

    # no-fly cylinders: stamp.
    if no_fly is not None and len(no_fly.zones) > 0:
        centers_xy = grid.cell_centers_xy()  # (nx, ny, 2)
        z_centers = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
        x = np.broadcast_to(centers_xy[..., 0:1], (nx, ny, nz))
        y = np.broadcast_to(centers_xy[..., 1:2], (nx, ny, nz))
        z = np.broadcast_to(z_centers[None, None, :], (nx, ny, nz))
        xyz = np.stack([x, y, z], axis=-1)
        mask = no_fly.mask(xyz, now_t_s)
        rho += alpha_obstacle * mask.astype(np.float32)

    return rho


def build_rander(grid, wind: IWindField, density: np.ndarray,
                 height_mask: np.ndarray,
                 v_cruise: float = 15.0,
                 alpha_noise: float = 5e-4,
                 alpha_obstacle: float = 1e6,
                 use_walls_flag: bool = True,
                 no_fly: NoFlyRegistry | None = None,
                 now_t_s: float = 0.0,
                 metric_version: int = 0) -> RanderField:
    """End-to-end metric assembly.

    Returns a RanderField that AGD can consume via aura.fmm.agd_solver.
    """
    nx, ny, nz = grid.shape3
    rho = build_rho(grid, density, height_mask, alpha_noise, alpha_obstacle,
                    no_fly=no_fly, now_t_s=now_t_s)

    # Riemannian diag = (1 / v_cruise)^2 * I, scaled by rho.
    inv_v2 = (1.0 / v_cruise) ** 2
    riemann_diag = (rho * inv_v2)[..., None] * np.ones(3, dtype=np.float32)

    # Rander offset omega = -wind / v_cruise^2 (Zermelo -> Rander linearization).
    if wind.is_constant():
        u, v, w = wind.constant_vector()
        offset = np.asarray([u, v, w], dtype=np.float32) * (-1.0 / v_cruise ** 2)
        rander_omega = np.broadcast_to(offset, (nx, ny, nz, 3)).copy()
    else:
        # Sample at every cell center. (nx, ny, nz, 3)
        z_centers = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
        cx, cy = grid.cell_centers_xy()[..., 0], grid.cell_centers_xy()[..., 1]
        x = np.broadcast_to(cx[..., None], (nx, ny, nz))
        y = np.broadcast_to(cy[..., None], (nx, ny, nz))
        z = np.broadcast_to(z_centers[None, None, :], (nx, ny, nz))
        xyz = np.stack([x, y, z], axis=-1)
        w_field = wind.sample(xyz, now_t_s).astype(np.float32)
        rander_omega = -w_field / v_cruise ** 2

    walls = None
    if use_walls_flag and height_mask is not None:
        z_centers = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
        walls = (z_centers[None, None, :] < height_mask[:, :, None]).astype(bool)

    return RanderField(
        riemann_diag=riemann_diag.astype(np.float32),
        rander_omega=rander_omega.astype(np.float32),
        rho=rho.astype(np.float32),
        walls=walls,
        v_cruise=float(v_cruise),
        metric_version=int(metric_version),
    )


def patch_rho_for_zone(field: RanderField, grid,
                       new_no_fly: NoFlyRegistry, now_t_s: float,
                       alpha_obstacle: float = 1e6) -> RanderField:
    """Return a new RanderField with rho updated to reflect new_no_fly only.

    Used by aura.fmm.incremental to avoid re-running the whole pipeline.
    The Riemann/Rander offset are unchanged in Stage 1 (wind constant).
    """
    nx, ny, nz = grid.shape3
    centers_xy = grid.cell_centers_xy()
    z_centers = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
    x = np.broadcast_to(centers_xy[..., 0:1], (nx, ny, nz))
    y = np.broadcast_to(centers_xy[..., 1:2], (nx, ny, nz))
    z = np.broadcast_to(z_centers[None, None, :], (nx, ny, nz))
    xyz = np.stack([x, y, z], axis=-1)
    extra_mask = new_no_fly.mask(xyz, now_t_s).astype(np.float32)
    new_rho = field.rho + alpha_obstacle * extra_mask
    new_riemann = (new_rho * (1.0 / field.v_cruise) ** 2)[..., None] * np.ones(3, dtype=np.float32)
    walls = field.walls.copy() if field.walls is not None else None
    if walls is not None:
        walls |= extra_mask.astype(bool)
    return RanderField(
        riemann_diag=new_riemann.astype(np.float32),
        rander_omega=field.rander_omega.copy(),
        rho=new_rho.astype(np.float32),
        walls=walls,
        v_cruise=field.v_cruise,
        metric_version=field.metric_version + 1,
    )
