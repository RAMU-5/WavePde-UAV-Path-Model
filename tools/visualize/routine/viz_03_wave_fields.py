"""Viz 3: wave-field T(x) at cruise altitude (z=70m) for each of the 118
sources. Reads database/wave_slice_NNN.npz and overlays building footprint.
Generates 118 PNGs in outputs/03_wave_fields/source_NNN.png.

Usage:
    python3 viz_03_wave_fields.py            # all 118
    python3 viz_03_wave_fields.py --src 5    # one source
    python3 viz_03_wave_fields.py --max 10   # first 10 only
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB = ROOT / "output" / "database"
DOMAIN = ROOT / "input" / "domain.npz"
OUT = ROOT / "output" / "figures" / "routine_system" / "03_wave_fields"


def render_one(src_idx: int, mask, dx, dy, dz, stations_xy_m,
               vmax_global=None):
    p = DB / f"wave_slice_{src_idx:03d}.npz"
    if not p.exists():
        print(f"[skip] {p.name} missing")
        return False
    z = np.load(p, allow_pickle=False)
    T = z["T_slice"]
    src_xyz = z["source_xyz_m"]
    z_m = float(z["z_m"])

    nx, ny = mask.shape[0], mask.shape[1]
    Lx, Ly = nx * dx, ny * dy

    fig, ax = plt.subplots(figsize=(13, 11))
    Tnan = np.where(np.isfinite(T), T, np.nan)
    finite = T[np.isfinite(T)]
    if vmax_global is None:
        vmax = np.percentile(finite, 98) if finite.size else 1
    else:
        vmax = vmax_global
    im = ax.imshow(Tnan.T, origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="turbo", vmin=0, vmax=vmax, alpha=0.85)
    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.01)
    cbar.set_label(f"first-arrival T(x) at z={z_m:.0f}m  (s)")

    # Building footprint outline
    fp = mask.any(axis=2)
    ax.contour(fp.T, levels=[0.5], colors="black", linewidths=0.3,
               extent=[0, Lx, 0, Ly])

    # All stations as small dots
    ax.scatter(stations_xy_m[:, 0], stations_xy_m[:, 1],
               s=10, c="white", edgecolor="black", linewidths=0.2,
               alpha=0.7, zorder=5)
    # Source highlighted
    ax.scatter(src_xyz[0], src_xyz[1], s=240, marker="*",
               facecolor="#ffe833", edgecolor="black", linewidths=0.8,
               zorder=10, label=f"source {src_idx}")

    n_reached = int(np.sum(np.isfinite(T)))
    cov_pct = n_reached / T.size * 100
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Wave T-field — source {src_idx}  "
                 f"(z={z_m:.0f}m, {cov_pct:.0f}% reached, "
                 f"max={finite.max():.0f}s)", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, color="white")

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
    print(f"[viz3] rendering {len(todo)} wave fields")

    n_done = 0
    for i in todo:
        ok = render_one(i, mask, dx, dy, dz, stations_xy_m)
        if ok:
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)}")
    print(f"[viz3] {n_done}/{len(todo)} rendered → {OUT}")


if __name__ == "__main__":
    main()
