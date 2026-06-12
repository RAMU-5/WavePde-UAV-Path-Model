"""Fig.9 — scale transfer (paper Q4, claim C9): the same wave kernel
routes a 15 m/s delivery multirotor through 5 m street canyons and a
200 m/s regional aircraft around weather cells in an 80 km domain —
only the config changes.

Left : Nanshan 5 m, T-field slice at cruise height + smoothed-tracer
       path (output/runs/exp7_efficiency/fig9_multirotor.npz).
Right: regional aircraft, top-down paths + the three anisotropic
       pollution/weather clouds (output/figures/3d_aircraft/
       aircraft_regional.npz).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "Q4" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mr = np.load(ROOT / "output" / "runs" / "exp7_efficiency"
             / "fig9_multirotor.npz")
ac = np.load(ROOT / "output" / "figures" / "3d_aircraft"
             / "aircraft_regional.npz")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

# ---------------- (a) Nanshan multirotor ------------------------------
ax = axes[0]
T = np.array(mr["T_slice"], dtype=np.float64)
m = mr["mask_slice"].astype(bool)
dx = float(mr["dx"])
ext = [0, T.shape[0] * dx / 1e3, 0, T.shape[1] * dx / 1e3]
Tm = np.ma.masked_where(~np.isfinite(T) | m, T)
im = ax.imshow(Tm.T, origin="lower", extent=ext, cmap="viridis",
               vmax=np.nanpercentile(T[np.isfinite(T)], 99))
bld = np.ma.masked_where(~m, np.ones_like(T))
ax.imshow(bld.T, origin="lower", extent=ext, cmap="gray_r",
          vmin=0, vmax=1.4)
p = mr["path"] / 1e3
ax.plot(p[:, 0], p[:, 1], "r-", lw=1.8,
        label=f"$-\\nabla T$ path, {float(mr['path_time_s']):.0f} s")
st = mr["st_xyz"] / 1e3
ax.plot(st[:, 0], st[:, 1], ".", color="w", ms=2, alpha=0.5)
for tag, idx, c in (("src", int(mr["src"]), "r"),
                    ("dst", int(mr["dst"]), "orange")):
    ax.plot(st[idx, 0], st[idx, 1], "*", color=c, ms=14,
            mec="k", mew=0.5)
    ax.annotate(tag, st[idx, :2], textcoords="offset points",
                xytext=(6, 6), color="w", fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.85, label="T [s]")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title(f"(a) delivery multirotor — Nanshan 5 m grid\n"
             f"$c_0$=15 m/s, slice z={float(mr['z_m']):.0f} m, "
             f"domain 6.7$\\times$7.2 km")
ax.legend(loc="lower right", fontsize=8)

# ---------------- (b) regional aircraft -------------------------------
ax = axes[1]
cc = ac["clouds_centre"] / 1e3
sxy = ac["clouds_sigma_xy"] / 1e3
for k in range(cc.shape[0]):
    for ns, alpha in ((2, 0.15), (1, 0.35)):
        ax.add_patch(Ellipse((cc[k, 0], cc[k, 1]),
                             2 * ns * sxy[k], 2 * ns * sxy[k],
                             color="tab:brown", alpha=alpha, lw=0))
n_paths = int(ac["n_paths"])
colors = plt.cm.cool(np.linspace(0, 1, n_paths))
for i in range(n_paths):
    p = ac[f"path_{i}_xyz"] / 1e3
    ax.plot(p[:, 0], p[:, 1], "-", color=colors[i], lw=1.8)
    d = ac[f"dst_{i}_xyz"] / 1e3
    ax.plot(d[0], d[1], "o", color=colors[i], ms=7, mec="k", mew=0.5)
ap = ac["airport_xyz_m"] / 1e3
ax.plot(ap[0], ap[1], "*", color="r", ms=16, mec="k", mew=0.5)
ax.annotate("airport", ap[:2], textcoords="offset points",
            xytext=(8, 4), fontsize=9)
ax.set_xlim(0, 80)
ax.set_ylim(0, 80)
ax.set_aspect("equal")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("(b) regional aircraft — 400 m grid\n"
             "$c_0$=200 m/s, cruise FL260, domain 80$\\times$80 km, "
             "3 weather cells")

fig.text(0.5, -0.02,
         "Same PDE kernel and pipeline; a config swap only: "
         "$c_0$ 15 $\\to$ 200 m/s, $c_z$ (6,4) $\\to$ (25,20) m/s, "
         "$\\Delta x$ 5 $\\to$ 400 m, domain 7 $\\to$ 80 km.",
         ha="center", fontsize=9, style="italic")
fig.tight_layout()
for ext_ in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"fig09_transfer.{ext_}", dpi=200,
                bbox_inches="tight")
print(json.dumps(dict(
    multirotor=dict(src=int(mr["src"]), dst=int(mr["dst"]),
                    status=str(mr["path_status"]),
                    path_time_s=float(mr["path_time_s"])),
    aircraft=dict(n_paths=n_paths, c0=float(ac["c0_aircraft"]),
                  cruise_alt_m=float(ac["cruise_alt_m"]))), indent=2))
print(f"saved -> {FIG_DIR}/fig09_transfer.pdf")
