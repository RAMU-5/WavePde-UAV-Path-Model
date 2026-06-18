"""Clean visualisation of station positions vs building positions.
Left: whole domain (buildings solid, stations as markers).
Right: zoom on a dense block, with each station's clearance-to-building
circle so you can see it sits in an inter-building gap.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="kuwait_stations_v2.npz")
    args = p.parse_args()
    tag = args.domain.replace("kuwait_stations", "").replace(".npz", "").strip("_") or "v1"

    z = np.load(ROOT / "input" / args.domain)
    mask = z["mask"]; dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    st = z["stations_xy_m"]
    B = mask.any(axis=2)
    clr_m = distance_transform_edt(~B) * dx
    clr_st = clr_m[np.round(st[:, 0] / dx).astype(int).clip(0, nx - 1),
                   np.round(st[:, 1] / dy).astype(int).clip(0, ny - 1)]

    foot = B.T.astype(float); foot[foot == 0] = np.nan
    ext = [0, nx * dx, 0, ny * dy]
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    ax = axs[0]
    ax.imshow(foot, origin="lower", cmap="Greys", vmin=0, vmax=1.3, extent=ext)
    ax.scatter(st[:, 0], st[:, 1], c="red", s=45, edgecolor="k", lw=0.6, zorder=5)
    for i, (x, y, _) in enumerate(st):
        ax.text(x, y + 35, str(i), fontsize=6, ha="center", color="navy")
    ax.set_title(f"{len(st)} stations vs buildings  [{tag}]  "
                 f"clearance min/med {clr_st.min():.0f}/{np.median(clr_st):.0f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")

    # zoom on the most building-dense 1.2 km box
    win = 600
    bx = (B.sum(axis=1)); cx0 = int(np.argmax(np.convolve(bx, np.ones(int(win/dx)), "same"))) * dx
    by = (B.sum(axis=0)); cy0 = int(np.argmax(np.convolve(by, np.ones(int(win/dy)), "same"))) * dy
    ax = axs[1]
    ax.imshow(foot, origin="lower", cmap="Greys", vmin=0, vmax=1.3, extent=ext)
    sel = ((np.abs(st[:, 0] - cx0) < win) & (np.abs(st[:, 1] - cy0) < win))
    ax.scatter(st[sel, 0], st[sel, 1], c="red", s=70, edgecolor="k", lw=0.8, zorder=6)
    for i in np.where(sel)[0]:
        ax.add_patch(Circle((st[i, 0], st[i, 1]), clr_st[i], fill=False,
                            color="red", lw=1.0, alpha=0.7))
        ax.text(st[i, 0], st[i, 1] + 25, f"{i}\n{clr_st[i]:.0f}m",
                fontsize=7, ha="center", color="darkred")
    ax.set_xlim(cx0 - win, cx0 + win); ax.set_ylim(cy0 - win, cy0 + win)
    ax.set_title(f"Zoom (±{win} m): red circle = clearance to nearest building")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")

    plt.tight_layout()
    out = ROOT / "output" / "wave" / f"kuwait_stations_check_{tag}.png"
    plt.savefig(out, dpi=130)
    print(f"[saved] {out}  ({len(st)} stations, min clearance {clr_st.min():.1f} m)",
          flush=True)


if __name__ == "__main__":
    main()
