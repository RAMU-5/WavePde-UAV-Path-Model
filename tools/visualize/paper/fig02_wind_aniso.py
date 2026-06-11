"""Fig. 2 — uniform wind (E1.2) and vertical anisotropy (E1.3).

(a) mid-z arrival-time contours under uniform wind q = 4.5 m/s (Mach 0.3)
    overlaid on the exact convected-medium solution (dashed): the
    wavefronts are advected ellipses, not circles.
(b) vertical-axis test (open domain): T vs vertical distance up/down
    with fitted effective speeds vs the configured cz_up = 6,
    cz_down = 4 m/s — validates the w_grav climb/descend encoding.

Reads  output/runs/exp0_analytic/{results.json, exp0_data.npz}
Writes paper/Q1/figs/fig02_wind_aniso.{png,pdf}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "runs" / "exp0_analytic"
OUT = ROOT / "paper" / "Q1" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads((RUN / "results.json").read_text())
dat = np.load(RUN / "exp0_data.npz")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

# ---- (a) wind ellipse -------------------------------------------------
C0, Q = 15.0, 4.5
dx, sx, sy, sz, bias0 = dat["e12_meta"]
T = dat["e12_Tslice_q4.5"] if "e12_Tslice_q4.5" in dat else dat["e12_Tslice_q4.5".replace(".5", "4.5")]
n = T.shape[0]
xc = (np.arange(n) + 0.5) * dx
X, Y = np.meshgrid(xc, xc, indexing="ij")
dxv, dyv = X - sx, Y - sy
r = np.hypot(dxv, dyv)
c2q2 = C0 ** 2 - Q ** 2
Tstar = (-dxv * Q + np.sqrt((dxv * Q) ** 2 + c2q2 * r ** 2)) / c2q2
levels = np.arange(2, 10, 2.0)
cs = ax.contour(X, Y, T - bias0, levels=levels, colors="C0")
ax.clabel(cs, fmt="%g s", fontsize=7)
ax.contour(X, Y, Tstar, levels=levels, colors="k",
           linestyles="--", linewidths=0.8)
ax.plot(sx, sy, "r*", ms=10)
ax.annotate("wind q = 4.5 m/s (Mach 0.3)", xy=(0.03, 0.95),
            xycoords="axes fraction", fontsize=9)
ax.annotate("", xy=(sx + 60, sy - 130), xytext=(sx, sy - 130),
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.text(sx + 30, sy - 145, "q", color="gray", ha="center")
errs = {f"M={r_['mach']:.1f}": r_["med_rel_all"]
        for r_ in res["e12"]["runs"] if r_["q"] > 0}
ax.set_title("(a) convected wavefronts vs exact (dashed)\n"
             + "  ".join(f"{k}: med {v:.2%}" for k, v in errs.items()),
             fontsize=9)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_aspect("equal")

# ---- (b) anisotropy shaft --------------------------------------------
e13 = res["e13"]
for tag, czt, color in (("up", 6.0, "C2"), ("dn", 4.0, "C3")):
    d = dat[f"e13_d_{tag}"]; Tt = dat[f"e13_T_{tag}"]
    lab = "ascend" if tag == "up" else "descend"
    ceff = e13["cz_up_eff"] if tag == "up" else e13["cz_down_eff"]
    ax2.plot(d, Tt, "o", color=color, ms=4,
             label=f"{lab}: fit {ceff:.2f} m/s (target {czt:g})")
    bias = e13["bias_up_s"] if tag == "up" else e13["bias_down_s"]
    ax2.plot(d, d / ceff + bias, "-", color=color, lw=1)
    ax2.plot(d, d / czt + bias, "k--", lw=0.7)
ax2.set_xlabel("vertical distance from source [m]")
ax2.set_ylabel("arrival time [s]")
ax2.set_title("(b) vertical-axis test: climb/descend asymmetry (w_grav)",
              fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig02_wind_aniso.{ext}", dpi=200)
print(f"[fig02] -> {OUT}/fig02_wind_aniso.png")
