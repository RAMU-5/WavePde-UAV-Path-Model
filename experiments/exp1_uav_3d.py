"""Experiment 1: 3D rendering of UAV trajectories from station 48 to a
handful of destinations, with collision-against-buildings check.

Reads database/paths_048.npz (already produced by build_database.py).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from viz3d import (add_buildings, add_path, setup_axes,
                   collision_check)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures" / "3d_uav"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_IDX = 48
N_PICK = 6  # destinations to plot in detail


def main():
    z = np.load(ROOT / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations_xy_m = z["stations_xy_m"]

    pdata = np.load(ROOT / "output" / "database" / f"paths_{SOURCE_IDX:03d}.npz",
                    allow_pickle=False)
    src_xyz = pdata["source_xyz_m"]
    dst_indices = pdata["dst_indices"]
    offsets = pdata["path_offsets"]
    paths_concat = pdata["paths_concat"]
    statuses = [s.decode() for s in pdata["path_status"]]
    T_dst = pdata["T_at_dst"]

    print(f"[exp1] source {SOURCE_IDX} → {len(dst_indices)} dests")
    print(f"[exp1] reached / truncated counts:")
    from collections import Counter
    sc = Counter("reached" if s == "reached" else "truncated"
                 for s in statuses)
    print(f"        {dict(sc)}")

    # Build per-path arrays + run collision check
    paths = []
    n_total_violations = 0
    n_paths_with_violation = 0
    for k in range(len(dst_indices)):
        a, b = offsets[k], offsets[k + 1]
        p = paths_concat[a:b]
        paths.append(p)
        nv, _ = collision_check(p, mask, dx, dy, dz)
        if nv > 0:
            n_total_violations += nv
            n_paths_with_violation += 1
    print(f"[collision] {n_paths_with_violation}/{len(paths)} paths "
          f"contain ≥1 in-building point  "
          f"(total in-building points = {n_total_violations})")

    # Pick N_PICK paths spread around: longest distance, longest path,
    # or just spread by destination angle.
    src_xy = src_xyz[:2]
    rel = stations_xy_m[dst_indices, :2] - src_xy
    d = np.linalg.norm(rel, axis=1)
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    bins = np.floor((angles + np.pi) / (2 * np.pi) * N_PICK).astype(int)
    pick = []
    for b_ in range(N_PICK):
        in_b = np.where(bins == b_)[0]
        if len(in_b) == 0:
            continue
        # Prefer reached + far + long-enough path
        scored = sorted(
            in_b,
            key=lambda i: (
                statuses[i] != "reached",
                -d[i],
                -paths[i].shape[0],
            ),
        )
        pick.append(scored[0])
    pick = list(dict.fromkeys(pick))[:N_PICK]
    print(f"[pick] dst list-indices = {pick}, "
          f"distances = {[f'{d[i]:.0f}m' for i in pick]}")

    # ---- Render: single 3D panel with all picked paths ---------------
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection="3d")
    add_buildings(ax, mask, dx, dy, dz, alpha=0.30, colour="#888")
    cmap = cm.get_cmap("turbo")
    for j, k in enumerate(pick):
        p = paths[k]
        # subsample for cleanliness
        if p.shape[0] > 400:
            idx = np.linspace(0, p.shape[0] - 1, 400).astype(int)
            p_plot = p[idx]
        else:
            p_plot = p
        col = cmap(j / max(1, len(pick) - 1))
        nv, viol = collision_check(p, mask, dx, dy, dz)
        lbl = (f"dst {dst_indices[k]} ({d[k]:.0f}m, "
               f"T={T_dst[k]:.0f}s, viol={nv})")
        add_path(ax, p_plot, colour=col, lw=2.0, label=lbl)
        if nv > 0:
            ax.scatter(viol[:, 0], viol[:, 1], viol[:, 2],
                       c="red", s=24, marker="x", zorder=20)
    # Source / destinations
    ax.scatter(*src_xyz, color="#ffd400", s=200, marker="*",
               edgecolor="black", linewidth=1.0,
               label=f"source {SOURCE_IDX}", zorder=15)
    dst_xy = stations_xy_m[dst_indices[pick], :2]
    dst_z = stations_xy_m[dst_indices[pick], 2]
    ax.scatter(dst_xy[:, 0], dst_xy[:, 1], dst_z,
               color="#ff5544", s=70, marker="v",
               edgecolor="black", linewidth=0.6, zorder=14)
    setup_axes(ax, mask, dx, dy, dz,
               f"3D UAV trajectories — source {SOURCE_IDX} "
               f"({len(pick)} of {len(paths)} dests shown)",
               elev=28, azim=-58)
    ax.legend(loc="upper left", fontsize=8.5)
    out = OUT / f"uav_3d_src{SOURCE_IDX:03d}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # ---- Top-down view for sanity check -----------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    add_buildings(ax, mask, dx, dy, dz, alpha=0.40, colour="#888")
    for j, k in enumerate(pick):
        p = paths[k]
        if p.shape[0] > 400:
            idx = np.linspace(0, p.shape[0] - 1, 400).astype(int)
            p = p[idx]
        col = cmap(j / max(1, len(pick) - 1))
        add_path(ax, p, colour=col, lw=2.0)
    ax.scatter(*src_xyz, color="#ffd400", s=200, marker="*",
               edgecolor="black", linewidth=1.0, zorder=15)
    setup_axes(ax, mask, dx, dy, dz,
               f"Top-down — source {SOURCE_IDX}",
               elev=85, azim=-90)
    out2 = OUT / f"uav_3d_src{SOURCE_IDX:03d}_top.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
