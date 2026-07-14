"""Per-source database builder: solves the wave PDE for one source,
extracts paths to all valid destinations, and saves both the cruise-
altitude T slice (for wave-field viz) and the path bundle (for route
viz) into database/.

Run as a SLURM array task — one source per task. Reuses the JAX
persistent cache from solve_all.py.

Outputs (per source i):
    database/wave_slice_NNN.npz : T at cruise altitude (z=70 m)
        - T_slice  : (NX, NY) float32, first-arrival time at z=70m
        - z_idx    : int, cruise z-index
        - source_idx, source_xyz_m
    database/paths_NNN.npz : path bundle to all valid destinations
        - source_idx              : int
        - source_xyz_m            : (3,) m
        - dst_indices             : (K,) int   — valid destination station ids
        - path_lengths            : (K,) int   — number of points per path
        - paths_concat            : (Σlen, 3)  — all paths concatenated
        - path_offsets            : (K+1,) int — offsets into paths_concat
        - path_status             : (K,) S20   — 'reached' / 'truncated_*' / ...
        - T_at_dst                : (K,) float32 — T at destination station
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import os
_THIS = Path(__file__).resolve().parent
_CACHE_DIR = str(_THIS.parent / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit
from scipy.ndimage import map_coordinates

from model.solve_all import (
    build_static_fields, gaussian_source, solve_one,
    N_STEPS,
)

THIS_DIR = Path(__file__).resolve().parent
DB_DIR = THIS_DIR.parent / "output" / "database"
DB_DIR.mkdir(exist_ok=True)

CRUISE_Z_M = 70.0   # altitude for the 2D T-slice viz


def trace_path(T, source_xyz, dest_xyz, dx, dy, dz,
               gamma=0.3, max_steps=10000):
    finite = np.isfinite(T)
    if not finite.any():
        return np.array([source_xyz, dest_xyz]), "no_finite"
    T_max = T[finite].max()
    T_safe = np.where(finite, T, T_max + 1e6)
    gx, gy, gz = np.gradient(T_safe, dx, dy, dz)
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
        if stale > 200:
            path = path[: best_idx + 1]; path.append(src.copy())
            return np.array(path[::-1]), f"truncated_{best_d:.0f}m"
    return np.array(path[::-1]), "max_steps"


def nudge_station_indices(stations_xy_m, mask, dx, dy, dz):
    """Same horizontal nudge as solve_all.main."""
    nx, ny, nz = mask.shape
    st_ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    st_iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    st_iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)
    inside_b = mask[st_ix, st_iy, st_iz]
    for b in np.where(inside_b)[0]:
        ix0, iy0, iz0 = st_ix[b], st_iy[b], st_iz[b]
        found = False
        for r in range(1, 6):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r: continue
                    ix = ix0 + di; iy = iy0 + dj
                    if not (0 < ix < nx-1 and 0 < iy < ny-1): continue
                    if not mask[ix, iy, iz0]:
                        st_ix[b], st_iy[b] = ix, iy
                        found = True; break
                if found: break
            if found: break
        if not found:
            iz = iz0
            while iz < nz-1 and mask[ix0, iy0, iz]: iz += 1
            st_iz[b] = iz
    return st_ix, st_iy, st_iz


def build_for_source(src_idx: int):
    print(f"[load] domain.npz")
    z = np.load(THIS_DIR.parent / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)
    st_ix, st_iy, st_iz = nudge_station_indices(stations_xy_m, mask, dx, dy, dz)

    pm = np.load(THIS_DIR.parent / "output" / "wave" / "pair_matrix.npz", allow_pickle=False)
    valid_mask = pm["valid_mask"]    # (N, N)
    T_at_pair = pm["T_matrix"]        # (N, N)

    print(f"[jit] compiling for source {src_idx}")
    mask_jnp = jnp.asarray(mask)
    static = build_static_fields(mask_jnp, dx, dy, dz)
    one_minus_damp = (1.0 - static["damp"]).astype(jnp.float32)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))

    src_xyz_idx = (int(st_ix[src_idx]), int(st_iy[src_idx]), int(st_iz[src_idx]))
    src_xyz_m = (st_ix[src_idx]*dx, st_iy[src_idx]*dy, st_iz[src_idx]*dz)
    print(f"[solve] source={src_idx} idx={src_xyz_idx}")
    t0 = time.time()
    sp = gaussian_source(mask.shape, src_xyz_idx)
    T = solve_jit(static["c2_xy"], static["c2_z"],
                  static["u"], static["v"], static["w"],
                  static["ax"], static["ay"], static["az"],
                  one_minus_damp, static["free"], sp,
                  dx, dy, dz, N_STEPS)
    T_np = np.asarray(jax.device_get(T))
    print(f"[solve] done in {time.time()-t0:.1f}s")

    # Save 2D T-slice at cruise altitude
    iz_cruise = min(int(round(CRUISE_Z_M / dz)), mask.shape[2] - 1)
    T_slice = T_np[:, :, iz_cruise].astype(np.float32)
    out_wave = DB_DIR / f"wave_slice_{src_idx:03d}.npz"
    np.savez_compressed(
        out_wave,
        T_slice=T_slice,
        z_idx=np.int32(iz_cruise),
        z_m=np.float32(CRUISE_Z_M),
        source_idx=np.int32(src_idx),
        source_xyz_m=np.array(src_xyz_m, dtype=np.float32),
        dx=np.float32(dx), dy=np.float32(dy), dz=np.float32(dz),
    )
    print(f"[saved] {out_wave.name}")

    # Trace paths to all valid destinations
    valid_dst = np.where(valid_mask[src_idx])[0]
    valid_dst = valid_dst[valid_dst != src_idx]
    valid_dst = valid_dst[np.isfinite(T_at_pair[src_idx, valid_dst])]
    print(f"[trace] {len(valid_dst)} valid destinations")
    paths = []
    statuses = []
    T_dst_arr = []
    for j in valid_dst:
        dst_xyz = (st_ix[j]*dx, st_iy[j]*dy, st_iz[j]*dz)
        path, status = trace_path(T_np, src_xyz_m, dst_xyz, dx, dy, dz)
        paths.append(path.astype(np.float32))
        statuses.append(status)
        T_dst_arr.append(float(T_np[st_ix[j], st_iy[j], st_iz[j]]))

    if paths:
        path_lengths = np.array([p.shape[0] for p in paths], dtype=np.int32)
        path_offsets = np.concatenate([[0], np.cumsum(path_lengths)]).astype(np.int32)
        paths_concat = np.concatenate(paths, axis=0).astype(np.float32)
    else:
        path_lengths = np.zeros(0, dtype=np.int32)
        path_offsets = np.zeros(1, dtype=np.int32)
        paths_concat = np.zeros((0, 3), dtype=np.float32)

    out_path = DB_DIR / f"paths_{src_idx:03d}.npz"
    np.savez_compressed(
        out_path,
        source_idx=np.int32(src_idx),
        source_xyz_m=np.array(src_xyz_m, dtype=np.float32),
        dst_indices=valid_dst.astype(np.int32),
        path_lengths=path_lengths,
        path_offsets=path_offsets,
        paths_concat=paths_concat,
        path_status=np.array(statuses, dtype="S24"),
        T_at_dst=np.array(T_dst_arr, dtype=np.float32),
    )
    print(f"[saved] {out_path.name}  ({len(valid_dst)} paths, "
          f"{paths_concat.shape[0]} total points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-idx", type=int, required=True)
    args = parser.parse_args()
    build_for_source(args.source_idx)
