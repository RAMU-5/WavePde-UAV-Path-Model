"""Assemble the Kuwait 64-station results: pair travel-time matrix from
the per-source path files, plus a route-overview figure.

Reads output/database_kuwait/paths_NNN.npz (from build_database_kuwait.py).
Writes:
    output/wave/pair_matrix_kuwait.npz  — T_matrix, valid_mask, distances
    output/wave/kuwait_routes.png       — all traced paths over footprint
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "output" / "database_kuwait"
OUT = ROOT / "output" / "wave"
LO, HI = 700.0, 3000.0


def main():
    z = np.load(ROOT / "input" / "kuwait_stations.npz", allow_pickle=False)
    mask = z["mask"]; dx = float(z["dx"]); dy = float(z["dy"])
    stations = z["stations_xy_m"]; lonlat = z["station_lonlat"]
    n = len(stations)

    T = np.full((n, n), np.inf, np.float32)
    all_paths = []
    found, total_paths, reached = [], 0, 0
    for i in range(n):
        f = DB / f"paths_{i:03d}.npz"
        if not f.exists():
            continue
        a = np.load(f, allow_pickle=False)
        found.append(i)
        if "T_at_stations" in a.files:
            T[i, :] = a["T_at_stations"]
        dst = a["dst_indices"]; off = a["path_offsets"]; pc = a["paths_concat"]
        status = a["path_status"].astype(str)
        for k in range(len(dst)):
            seg = pc[off[k]:off[k + 1]]
            all_paths.append(seg)
            total_paths += 1
            reached += int(status[k] == "reached")
    print(f"[aggregate] sources found {len(found)}/{n}")
    print(f"[aggregate] {total_paths} paths, reached {reached} "
          f"({reached/max(total_paths,1)*100:.1f}%)")

    diff = stations[:, None, :2] - stations[None, :, :2]
    D = np.sqrt((diff ** 2).sum(-1)).astype(np.float32); np.fill_diagonal(D, 0)
    valid = (D >= LO) & (D <= HI)
    finite = valid & np.isfinite(T)
    print(f"[aggregate] valid pairs {int(valid.sum())}, "
          f"finite T {int(finite.sum())} "
          f"({finite.sum()/max(valid.sum(),1)*100:.1f}%)")
    if finite.any():
        tt = T[finite]
        print(f"[aggregate] travel time min/median/max = "
              f"{tt.min():.1f}/{np.median(tt):.1f}/{tt.max():.1f}s")
        ratio = tt / np.maximum(D[finite] / 15.0, 1e-6)
        print(f"[aggregate] T/(d/15) ratio median={np.median(ratio):.2f} "
              f"p95={np.percentile(ratio,95):.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "pair_matrix_kuwait.npz",
        T_matrix=T, valid_mask=valid, distances_m=D,
        stations_xy_m=stations, station_lonlat=lonlat,
        n_pairs_valid=np.int32(valid.sum()),
        n_pairs_with_T=np.int32(finite.sum()))
    print(f"[saved] {OUT/'pair_matrix_kuwait.npz'}")

    # Route overview figure
    foot = mask.any(axis=2).T.astype(float); foot[foot == 0] = np.nan
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(foot, origin="lower", cmap="Greys", vmin=0, vmax=1.4,
              extent=[0, mask.shape[0] * dx, 0, mask.shape[1] * dy])
    for seg in all_paths:
        ax.plot(seg[:, 0], seg[:, 1], lw=0.25, color="tab:blue", alpha=0.35)
    ax.scatter(stations[:, 0], stations[:, 1], c="red", s=25,
               edgecolor="k", lw=0.4, zorder=5)
    ax.set_title(f"Kuwait CBD — {total_paths} station-to-station routes "
                 f"({n} stations, no wind)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout(); plt.savefig(OUT / "kuwait_routes.png", dpi=120)
    print(f"[saved] {OUT/'kuwait_routes.png'}")


if __name__ == "__main__":
    main()
