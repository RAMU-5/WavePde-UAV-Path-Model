"""Per-source path-database builder for the Kuwait CBD 64-station network.

Mirrors scripts/build_database.py (the Nanshan experiment) but is
self-contained: each array task solves ONE source with the baseline
no-wind WaveConfig, then traces the optimal -∇T path to every valid
destination (pair distance in [700m, 3000m], finite arrival time).

Settings copied from the previous experiment:
    wind = (0,0,0)        (solve_all baseline batch)
    c0 = 15 m/s, cz_up/down = 6/4, n_steps = 4500
    min_peak_amp = 1e-6 * (dx/50)^3   (grid-scaled threshold)
    valid pair band = [700, 3000] m
    path trace = gradient descent on T, gamma = 0.3

Outputs per source i (output/database_kuwait/):
    paths_NNN.npz  — dst_indices, paths_concat, path_offsets,
                     path_status, T_at_dst, T_at_stations (for the matrix)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = str(ROOT / "_jax_cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0, str(ROOT / "model"))
import jax  # noqa: E402
import wave_api  # noqa: E402
from wave_api import WaveConfig, build_static_fields  # noqa: E402

LO_DIST_M, HI_DIST_M = 700.0, 3000.0


def nudge_station_indices(stations_xy_m, mask, dx, dy, dz):
    nx, ny, nz = mask.shape
    ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)
    inside = mask[ix, iy, iz]
    for b in np.where(inside)[0]:
        ix0, iy0, iz0 = ix[b], iy[b], iz[b]; found = False
        for r in range(1, 6):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    a, c = ix0 + di, iy0 + dj
                    if 0 < a < nx - 1 and 0 < c < ny - 1 and not mask[a, c, iz0]:
                        ix[b], iy[b] = a, c; found = True; break
                if found:
                    break
            if found:
                break
        if not found:
            k = iz0
            while k < nz - 1 and mask[ix0, iy0, k]:
                k += 1
            iz[b] = k
    return ix, iy, iz


def trace_path(T, source_xyz, dest_xyz, dx, dy, dz, gamma=0.3, max_steps=10000):
    """Gradient descent on T from dst back to src (build_database.py)."""
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
    best_d = np.linalg.norm(pos - src); best_idx = 0; stale = 0
    for _ in range(max_steps):
        coords = np.array([[pos[0] / dx], [pos[1] / dy], [pos[2] / dz]])
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-idx", type=int, required=True)
    p.add_argument("--domain", default="kuwait_stations.npz")
    p.add_argument("--outdir", default="database_kuwait",
                   help="subdir under output/ for the path bundles")
    args = p.parse_args()
    DB_DIR = ROOT / "output" / args.outdir
    DB_DIR.mkdir(parents=True, exist_ok=True)

    out_path = DB_DIR / f"paths_{args.source_idx:03d}.npz"
    if out_path.exists():
        print(f"[skip] {out_path.name} exists"); return

    print(f"[jax] {jax.devices()}", flush=True)
    z = np.load(ROOT / "input" / args.domain, allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations = z["stations_xy_m"]
    n_st = len(stations)
    src = args.source_idx
    if not (0 <= src < n_st):
        raise ValueError(f"source {src} out of range [0,{n_st})")

    ix, iy, iz = nudge_station_indices(stations, mask, dx, dy, dz)
    src_idx = (int(ix[src]), int(iy[src]), int(iz[src]))
    src_xyz = (ix[src] * dx, iy[src] * dy, iz[src] * dz)

    min_peak = 1e-6 * (dx / 50.0) ** 3
    cfg = WaveConfig(min_peak_amp=min_peak)        # wind defaults to (0,0,0)
    print(f"[cfg] wind={cfg.wind} c0={cfg.c0} n_steps={cfg.n_steps} "
          f"min_peak={min_peak:.2e}", flush=True)

    t0 = time.time()
    static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
    T = wave_api.solve_forward(mask, dx, dy, dz, src_idx, cfg=cfg, static=static)
    print(f"[solve] src {src} -> {src_idx} in {time.time()-t0:.1f}s", flush=True)

    T_at_stations = T[ix, iy, iz].astype(np.float32)

    # Valid destinations: distance band + finite arrival.
    d = np.sqrt(((stations[:, :2] - stations[src, :2]) ** 2).sum(1))
    valid = np.where((d >= LO_DIST_M) & (d <= HI_DIST_M)
                     & np.isfinite(T_at_stations))[0]
    valid = valid[valid != src]
    print(f"[trace] {len(valid)} valid destinations", flush=True)

    paths, statuses, T_dst = [], [], []
    t1 = time.time()
    for j in valid:
        dst_xyz = (ix[j] * dx, iy[j] * dy, iz[j] * dz)
        path, status = trace_path(T, src_xyz, dst_xyz, dx, dy, dz)
        paths.append(path.astype(np.float32))
        statuses.append(status)
        T_dst.append(float(T_at_stations[j]))
    print(f"[trace] done in {time.time()-t1:.1f}s  "
          f"reached={sum(s=='reached' for s in statuses)}/{len(valid)}",
          flush=True)

    if paths:
        lengths = np.array([p.shape[0] for p in paths], dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
        concat = np.concatenate(paths, axis=0).astype(np.float32)
    else:
        lengths = np.zeros(0, np.int32); offsets = np.zeros(1, np.int32)
        concat = np.zeros((0, 3), np.float32)

    np.savez_compressed(
        out_path,
        source_idx=np.int32(src),
        source_xyz_m=np.array(src_xyz, dtype=np.float32),
        dst_indices=valid.astype(np.int32),
        path_lengths=lengths, path_offsets=offsets, paths_concat=concat,
        path_status=np.array(statuses, dtype="S24"),
        T_at_dst=np.array(T_dst, dtype=np.float32),
        T_at_stations=T_at_stations,
    )
    print(f"[saved] {out_path.name}  ({len(valid)} paths, "
          f"{concat.shape[0]} pts)", flush=True)


if __name__ == "__main__":
    main()
