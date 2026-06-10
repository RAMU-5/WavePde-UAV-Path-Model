"""L0 station siting: weighted k-medoids on street-centroid candidates.

We avoid sklearn-extra (extra dep) and write a small weighted k-medoids
ourselves — Voronoi reassignment + medoid re-pick is enough at K~50.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from aura.schemas import Station


def population_at_points(points_xy: np.ndarray,
                         density: np.ndarray, grid) -> np.ndarray:
    """Sample population density at each (N, 2) point via nearest cell.

    Returns a (N,) array. Out-of-bounds points return 0.
    """
    points_xyz = np.concatenate(
        [points_xy, np.full((len(points_xy), 1), grid.z_min)], axis=1
    )
    ijk = grid.world_to_index(points_xyz)
    ix = ijk[..., 0]
    iy = ijk[..., 1]
    return density[ix, iy]


def weighted_kmedoids(points: np.ndarray, weights: np.ndarray, k: int,
                      max_iter: int = 30, seed: int = 0,
                      verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted k-medoids by Voronoi-iteration (PAM-lite).

    points:  (N, 2) candidate coords.
    weights: (N,) non-negative population weights.
    Returns (medoid_indices (k,), assignment (N,)).
    """
    rng = np.random.default_rng(seed)
    N = len(points)
    if N <= k:
        return np.arange(N), np.arange(N)

    # Init: weighted random sample without replacement (k-medoids++ flavor).
    p_init = (weights + 1e-9)
    p_init = p_init / p_init.sum()
    medoids = rng.choice(N, size=k, replace=False, p=p_init)

    for it in range(max_iter):
        d2 = ((points[:, None, :] - points[medoids][None, :, :]) ** 2).sum(axis=-1)
        assign = d2.argmin(axis=1)  # (N,)
        new_medoids = np.empty(k, dtype=np.int64)
        for c in range(k):
            members = np.where(assign == c)[0]
            if len(members) == 0:
                new_medoids[c] = medoids[c]
                continue
            w = weights[members]
            if w.sum() < 1e-9:
                new_medoids[c] = members[len(members) // 2]
                continue
            local = points[members]
            costs = ((local[:, None, :] - local[None, :, :]) ** 2).sum(axis=-1)
            cost = (costs * w[None, :]).sum(axis=1)
            new_medoids[c] = members[int(np.argmin(cost))]
        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            if verbose:
                print(f"k-medoids converged at iter {it}")
            break
        medoids = new_medoids

    d2 = ((points[:, None, :] - points[medoids][None, :, :]) ** 2).sum(axis=-1)
    assign = d2.argmin(axis=1)
    return medoids, assign


def voronoi_owner(grid, station_xy: np.ndarray) -> np.ndarray:
    """Return (nx, ny) int32 array of station id per cell.

    For small K (<=64) we broadcast directly. For larger K we use cKDTree to
    avoid the (nx, ny, K) memory blowup that would happen at K=400+.

    Pattern lifted from aura_toy_archive/stations.py:_build_service_regions
    and extended for block-based siting where K can be in the hundreds.
    """
    centers_xy = grid.cell_centers_xy()  # (nx, ny, 2)
    K = len(station_xy)
    if K <= 64:
        diff = centers_xy[..., None, :] - station_xy[None, None, :, :]
        d2 = (diff ** 2).sum(axis=-1)
        return d2.argmin(axis=-1).astype(np.int32)
    from scipy.spatial import cKDTree
    tree = cKDTree(station_xy)
    flat = centers_xy.reshape(-1, 2)
    _, idx = tree.query(flat, k=1)
    return idx.reshape(grid.shape2).astype(np.int32)


def build_stations(candidates_xy_m: np.ndarray,
                   density: np.ndarray, grid,
                   n_stations: int = 50,
                   pop_per_unit: float = 5000.0,
                   cap_min: int = 4, cap_max: int = 16,
                   seed: int = 0) -> Tuple[List[Station], np.ndarray]:
    weights = population_at_points(candidates_xy_m, density, grid)
    medoid_idx, _ = weighted_kmedoids(candidates_xy_m, weights, n_stations, seed=seed)
    station_xy = candidates_xy_m[medoid_idx]
    owner = voronoi_owner(grid, station_xy)

    stations: List[Station] = []
    for sid, (sx, sy) in enumerate(station_xy):
        cell_mask = owner == sid
        pop_voronoi = float(density[cell_mask].sum())
        cap = int(np.clip(round(pop_voronoi / pop_per_unit), cap_min, cap_max))
        stations.append(Station(id=sid, xy_m=(float(sx), float(sy)),
                                pop_voronoi=pop_voronoi, capacity_pads=cap))
    return stations, owner
