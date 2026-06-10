"""Compute and save the all-pairs nominal trajectory DB for Stage 1.

Reads a Stage1Assets dir, builds the Rander cost field (with the OSM-derived
3D building wall mask), runs all-pairs FMM (anisotropic via AGD if available,
else scipy.csgraph multi-source Dijkstra), and persists the trajectory DB.

Usage:
    python -m aura.cli.build_trajectories --assets runs/v1 --out runs/v1/traj
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from aura.cost.metric import build_rander
from aura.fmm.agd_solver import save_db, solve_all_pairs
from aura.io.assets import load as load_assets
from aura.wind import ConstantWind


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--out", default=None,
                   help="Trajectory DB output dir (default: <assets>/traj).")
    p.add_argument("--coverage-m", type=float, default=5000.0,
                   help="Each station only routes to neighbours within this "
                        "Euclidean distance (m). 0 = all-pairs.")
    p.add_argument("--workers", type=int, default=1,
                   help="multiprocessing workers (>1 splits sources).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    assets = load_assets(args.assets)
    cfg = assets.config
    out_dir = Path(args.out) if args.out else Path(args.assets) / "traj"

    grid = assets.grid
    print(f"[traj] grid {grid.shape3} ({grid.size_xy[0]/1000:.1f}x"
          f"{grid.size_xy[1]/1000:.1f} km), {len(assets.stations)} stations")

    wind = ConstantWind(
        u=cfg["wind"]["constant"]["u_mps"],
        v=cfg["wind"]["constant"]["v_mps"],
        w=cfg["wind"]["constant"]["w_mps"],
    )
    print("[traj] building Rander cost field with 3D building walls...")
    field = build_rander(
        grid, wind, assets.population_density, assets.height_mask,
        v_cruise=cfg["cost"]["v_cruise_mps"],
        alpha_noise=cfg["cost"]["alpha_noise"],
        alpha_obstacle=cfg["cost"]["alpha_obstacle"],
        use_walls_flag=cfg["cost"]["use_walls_flag"],
    )
    n_walls = int(field.walls.sum()) if field.walls is not None else 0
    print(f"[traj] walls = {n_walls} cells ({n_walls/field.rho.size*100:.2f}% of grid)")

    station_xys = np.asarray([s.xy_m for s in assets.stations], dtype=np.float64)
    z_cruise = 0.5 * (cfg["fmm"]["z_floor_m"] + cfg["fmm"]["z_ceiling_m"])

    t0 = time.time()
    last = [t0]
    def _cb(done, total):
        # AGD path only — fallback runs in one batched call.
        if time.time() - last[0] > 5.0:
            print(f"[traj] {done}/{total} sources done "
                  f"({time.time()-t0:.0f}s)")
            last[0] = time.time()

    cov = args.coverage_m if args.coverage_m and args.coverage_m > 0 else None
    print(f"[traj] coverage_radius_m = {cov} ; workers = {args.workers}")
    db = solve_all_pairs(
        field, grid, station_xys, station_z_m=z_cruise,
        trajectory_dt_s=cfg["fmm"]["trajectory_dt_s"],
        progress_cb=_cb,
        coverage_radius_m=cov,
        n_workers=args.workers,
    )
    elapsed = time.time() - t0
    print(f"[traj] solved {len(db.pairs)} pairs in {elapsed:.0f}s "
          f"({len(db.pairs)/elapsed:.0f} pairs/s)")

    # Trajectory length stats.
    lens = []
    z_devs = []
    for traj in db.pairs.values():
        if not traj.xyz:
            continue
        pts = np.asarray(traj.xyz)
        d = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        lens.append(d)
        z_devs.append(float(pts[:, 2].max() - pts[:, 2].min()))
    if lens:
        print(f"[traj] path length:  median {np.median(lens):.0f} m, "
              f"max {max(lens):.0f} m")
        print(f"[traj] altitude span (per traj): "
              f"median {np.median(z_devs):.1f} m, "
              f"max {max(z_devs):.1f} m")

    save_db(db, out_dir)
    print(f"[traj] saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
