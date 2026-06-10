"""Viz 4: for each of the 118 source stations, plot all paths from that
source to every reachable destination on the Nanshan map (with buildings).

Reads database/paths_NNN.npz. Generates 118 PNGs in
outputs/04_routes/source_NNN.png.

Each path is coloured by travel time (T at destination).

Usage:
    python3 viz_04_routes_per_station.py            # all 118
    python3 viz_04_routes_per_station.py --src 5    # one source
    python3 viz_04_routes_per_station.py --max 10   # first 10
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB = ROOT / "output" / "database"
DOMAIN = ROOT / "input" / "domain.npz"
OUT = ROOT / "output" / "figures" / "routine_system" / "04_routes"


def render_one(src_idx: int, mask, dx, dy, dz, stations_xy_m):
    p = DB / f"paths_{src_idx:03d}.npz"
    if not p.exists():
        print(f"[skip] {p.name} missing")
        return False
    z = np.load(p, allow_pickle=False)
    src_xyz = z["source_xyz_m"]
    dst_indices = z["dst_indices"]
    path_offsets = z["path_offsets"]
    paths_concat = z["paths_concat"]
    statuses = [s.decode() if isinstance(s, bytes) else s
                for s in z["path_status"]]
    T_dst = z["T_at_dst"]

    nx, ny = mask.shape[0], mask.shape[1]
    Lx, Ly = nx * dx, ny * dy

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_facecolor("#f4f1ec")
    # Buildings as filled grey footprints
    fp = mask.any(axis=2)
    # Use imshow for fast fill
    bg = np.where(fp, 0.55, 0.0)
    ax.imshow(bg.T, origin="lower", extent=[0, Lx, 0, Ly],
              cmap="Greys", vmin=0, vmax=1, alpha=0.6, zorder=1)

    # All stations as small dots
    ax.scatter(stations_xy_m[:, 0], stations_xy_m[:, 1],
               s=14, c="white", edgecolor="black", linewidths=0.4,
               zorder=5)

    # Build LineCollection from all path segments
    segs = []; tcol = []
    n_reached = 0
    for k in range(len(dst_indices)):
        a, b = path_offsets[k], path_offsets[k+1]
        path = paths_concat[a:b, :2]   # 2D projection
        if path.shape[0] < 2:
            continue
        for s in range(path.shape[0] - 1):
            segs.append([path[s], path[s+1]])
            tcol.append(T_dst[k])
        if statuses[k] == "reached":
            n_reached += 1

    if segs:
        tcol = np.asarray(tcol)
        # Normalize over this source's own destinations
        vmin = float(tcol.min())
        vmax = float(np.percentile(tcol, 95))
        lc = LineCollection(segs, cmap="plasma_r",
                            norm=plt.Normalize(vmin, max(vmax, vmin+1)),
                            linewidths=0.8, alpha=0.55, zorder=6)
        lc.set_array(tcol)
        ax.add_collection(lc)
        cbar = plt.colorbar(lc, ax=ax, fraction=0.028, pad=0.01)
        cbar.set_label("travel time T(dst) (s)")

    # Source highlighted
    ax.scatter(src_xyz[0], src_xyz[1], s=300, marker="*",
               facecolor="#ffe833", edgecolor="black", linewidths=1.0,
               zorder=12, label=f"source {src_idx}")
    # Reachable destinations
    if len(dst_indices):
        d_xy = stations_xy_m[dst_indices, :2]
        ax.scatter(d_xy[:, 0], d_xy[:, 1], s=40, marker="v",
                   facecolor="#ff5544", edgecolor="black", linewidths=0.5,
                   alpha=0.7, zorder=10, label=f"{len(dst_indices)} dests")

    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(
        f"Routes from source {src_idx} — "
        f"{n_reached}/{len(dst_indices)} reached "
        f"(distance band 700-3000m)",
        fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"source_{src_idx:03d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=None)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    z = np.load(DOMAIN, allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)

    if args.src is not None:
        render_one(args.src, mask, dx, dy, dz, stations_xy_m)
        return

    todo = list(range(n_st))
    if args.max is not None:
        todo = todo[: args.max]
    print(f"[viz4] rendering {len(todo)} route maps")

    n_done = 0
    for i in todo:
        ok = render_one(i, mask, dx, dy, dz, stations_xy_m)
        if ok:
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)}")
    print(f"[viz4] {n_done}/{len(todo)} rendered → {OUT}")


if __name__ == "__main__":
    main()
