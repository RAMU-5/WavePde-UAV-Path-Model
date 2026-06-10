"""Shared 3D rendering helpers for the UAV / aircraft experiments.

Buildings are rendered as a heightmap mesh (one tall block per (ix,iy)
that has any True cell). Trajectories are 3D polylines. Optional
Gaussian "pollution clouds" are rendered as semi-transparent voxel
clouds at low resolution.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def building_heightmap(mask, dz):
    """For each (ix,iy), return the highest True z-cell + 1 (i.e. building height in cells).
    Returns an (nx,ny) int array; 0 means no building."""
    if not mask.any():
        return np.zeros(mask.shape[:2], dtype=np.int32)
    nz = mask.shape[2]
    iz = np.arange(nz)[None, None, :]
    h_cells = np.where(mask, iz + 1, 0).max(axis=2)
    return h_cells.astype(np.int32)


def add_buildings(ax, mask, dx, dy, dz, alpha=0.45, colour="#777777"):
    """Render building footprints as 3D blocks. Uses bar3d for speed."""
    h_cells = building_heightmap(mask, dz)
    iy, ix = np.meshgrid(np.arange(mask.shape[1]),
                         np.arange(mask.shape[0]),
                         indexing="xy")
    flat_h = h_cells.ravel()
    sel = flat_h > 0
    if not sel.any():
        return
    xs = (ix.ravel()[sel]) * dx
    ys = (iy.ravel()[sel]) * dy
    zs = np.zeros_like(xs, dtype=np.float64)
    hs = flat_h[sel] * dz
    ax.bar3d(xs, ys, zs, dx, dy, hs,
             color=colour, alpha=alpha, shade=True,
             edgecolor=(0, 0, 0, 0.05), linewidth=0.0,
             zsort="average")


def add_path(ax, path_xyz, colour="#1aa04a", lw=2.2, label=None):
    """Plot a 3D path. Returns the Line3D for legend handling."""
    p = np.asarray(path_xyz)
    return ax.plot(p[:, 0], p[:, 1], p[:, 2],
                   color=colour, lw=lw, alpha=0.95, label=label)[0]


def add_gaussian_cloud(ax, centre_m, sigma_m, dx, dy, dz, mask_shape,
                       n_voxels=80, alpha_max=0.18, colour="#a45ad9"):
    """Render a Gaussian cloud as voxel-style scatter at radii 1.0σ and 0.5σ."""
    nx, ny, nz = mask_shape
    # Sample points within ±2σ box
    cx, cy, cz = centre_m
    r_max = 2.5 * sigma_m
    n = n_voxels
    xs = np.linspace(cx - r_max, cx + r_max, n)
    ys = np.linspace(cy - r_max, cy + r_max, n)
    zs = np.linspace(max(0, cz - r_max), cz + r_max, n // 3)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    g = np.exp(-0.5 * r2 / sigma_m ** 2)
    keep = g > 0.30
    if not keep.any():
        return
    sample_idx = np.random.default_rng(7).choice(
        np.where(keep.ravel())[0],
        size=min(2000, keep.sum()),
        replace=False,
    )
    Xs = X.ravel()[sample_idx]
    Ys = Y.ravel()[sample_idx]
    Zs = Z.ravel()[sample_idx]
    gs = g.ravel()[sample_idx]
    ax.scatter(Xs, Ys, Zs, c=colour, s=12,
               alpha=alpha_max, marker="o",
               edgecolors="none", depthshade=True)


def setup_axes(ax, mask, dx, dy, dz, title, elev=25, azim=-55):
    nx, ny, nz = mask.shape
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    # Equalise xy aspect; z is naturally compressed (210m vs 7km)
    ax.set_box_aspect((Lx, Ly, max(Lz, 0.05 * max(Lx, Ly))))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)


def collision_check(path_xyz, mask, dx, dy, dz, slack_cells=0):
    """For each point on the path, look up mask[ix,iy,iz] (rounded to
    cell). Returns (n_violations, list_of_violation_xyz).
    `slack_cells` lets the path be slack_cells cells "into" a building
    before counting (default 0 = strict)."""
    p = np.asarray(path_xyz)
    if p.ndim != 2 or p.shape[1] != 3:
        return 0, np.zeros((0, 3))
    ix = np.clip(np.round(p[:, 0] / dx).astype(int), 0, mask.shape[0] - 1)
    iy = np.clip(np.round(p[:, 1] / dy).astype(int), 0, mask.shape[1] - 1)
    iz = np.clip(np.round(p[:, 2] / dz).astype(int), 0, mask.shape[2] - 1)
    inside = mask[ix, iy, iz]
    if slack_cells > 0:
        # Erode the mask by slack_cells before checking
        from scipy.ndimage import binary_erosion
        m2 = binary_erosion(mask, iterations=slack_cells)
        inside = m2[ix, iy, iz]
    return int(inside.sum()), p[inside]
