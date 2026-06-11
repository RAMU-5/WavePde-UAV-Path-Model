"""Fig. 3 — single-obstacle benchmark (E1.4).

(a) vertical (x,z) slice of the arrival-time field at the source y-row
    with the traced -∇T path overlaid; exact taut-string geodesic shown
    dashed. The first arrival climbs over the box exactly along the
    geodesic homotopy class.
(b) relative gap of the calibrated arrival time vs the exact geodesic
    time as a function of dx (1.25 m near-reference included).

Reads  output/runs/exp0_analytic/{results.json, exp0_data.npz}
Writes paper/Q1/figs/fig03_obstacle.{png,pdf}
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))
from path_trace import trace_path_3d  # noqa: E402

RUN = ROOT / "output" / "runs" / "exp0_analytic"
OUT = ROOT / "paper" / "Q1" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads((RUN / "results.json").read_text())["e14"]["runs"]
dat = np.load(RUN / "exp0_data.npz")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8),
                              gridspec_kw=dict(width_ratios=[1.5, 1]))

# ---- (a) slice + path at dx = 2.5 -------------------------------------
dx = 2.5
T = dat[f"e14_T_dx{dx:g}"]
meta = dat[f"e14_meta_dx{dx:g}"]
_, sx, sy, sz, dxm, dym, dzm, Tstar, bias = meta
iy = int(round(sy / dx - 0.5))
sl = np.where(np.isfinite(T[:, iy, :]), T[:, iy, :] - bias, np.nan)
nx, nz = sl.shape
im = ax.imshow(sl.T, origin="lower", aspect="equal", cmap="viridis",
               extent=[0, nx * dx, 0, nz * dx])
plt.colorbar(im, ax=ax, label="calibrated arrival time [s]", shrink=0.85)
ax.add_patch(plt.Rectangle((120, 0), 60, 60, fill=False,
                           edgecolor="w", lw=1.2))
path, status = trace_path_3d(T, (sx, sy, sz), (dxm, dym, dzm), dx, dx, dx)
ax.plot(path[:, 0], path[:, 2], "r-", lw=1.6,
        label=f"traced −∇T path ({status})")
geo_x = [sx, 120, 180, dxm]
geo_z = [sz, 60, 60, dzm]
ax.plot(geo_x, geo_z, "w--", lw=1.2, label="exact geodesic")
ax.plot([sx], [sz], "r*", ms=11)
ax.plot([dxm], [dzm], "rs", ms=6)
ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
ax.set_title(f"(a) T(x,z) slice at dx = {dx:g} m; T* = {Tstar:.2f} s",
             fontsize=9)
ax.legend(fontsize=8, loc="upper right")

# ---- (b) gap vs dx ------------------------------------------------------
runs = sorted(res, key=lambda r: -r["dx"])
ax2.semilogx([r["dx"] for r in runs],
             [100 * r["rel_gap"] for r in runs], "o-")
for r in runs:
    ax2.annotate(f"{100*r['rel_gap']:+.2f}%", (r["dx"], 100 * r["rel_gap"]),
                 textcoords="offset points", xytext=(6, 5), fontsize=8)
ax2.axhline(0, color="k", lw=0.7)
ax2.set_xlabel("grid spacing dx [m]")
ax2.set_ylabel("arrival-time gap vs exact geodesic [%]")
ax2.set_title("(b) calibrated gap vs taut-string time", fontsize=9)
ax2.grid(alpha=0.3, which="both")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig03_obstacle.{ext}", dpi=200)
print(f"[fig03] -> {OUT}/fig03_obstacle.png")
