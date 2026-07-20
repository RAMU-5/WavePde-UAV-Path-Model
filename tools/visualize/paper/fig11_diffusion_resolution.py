"""Plot E7.2 diffusion grid-sensitivity results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "runs" / "exp10_diffusion_resolution"
OUT = ROOT / "paper" / "Q3" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

runs = json.loads((RUN / "results.json").read_text())["runs"]
dxs = sorted({r["dx_m"] for r in runs}, reverse=True)

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
for dx in dxs:
    rows = sorted((r for r in runs if r["dx_m"] == dx),
                  key=lambda r: r["D_diff_m2_s"])
    D = np.array([r["D_diff_m2_s"] for r in rows])
    reached = np.array([r["trace_status"] == "reached" for r in rows])
    speed = np.array([r["effective_speed_m_s"] for r in rows])
    axes[0].plot(D, 100 * (speed / speed[0] - 1.0),
                 "o-", label=f"$\\Delta x={dx:g}$ m")
    clearance = np.array([r["min_clearance_m"] for r in rows])
    path_time = np.array([r["path_time_s"] for r in rows])
    axes[1].plot(D, np.where(reached, clearance, np.nan), "o-")
    axes[2].plot(D, np.where(reached, path_time, np.nan), "o-")
    if np.any(~reached):
        axes[1].scatter(D[~reached], np.zeros(np.sum(~reached)), marker="x",
                        s=55, color="crimson", zorder=5)

axes[0].axhline(0, color="k", lw=0.8, ls="--")
axes[0].set(xlabel="$D_{diff}$ [m$^2$/s]", ylabel="peak-speed shift from $D=0$ [%]",
            title="(a) empty-domain propagation")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="$D_{diff}$ [m$^2$/s]", ylabel="minimum clearance [m]",
            title="(b) obstacle clearance")
axes[2].set(xlabel="$D_{diff}$ [m$^2$/s]", ylabel="nominal path time [s]",
            title="(c) path-time sensitivity")
axes[1].text(0.72, 0.05, "$\\times$: tracer failure ($N_D=0.96$)",
             transform=axes[1].transAxes, ha="center", color="crimson",
             fontsize=8)
for ax in axes:
    ax.grid(alpha=0.25)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig11_diffusion_resolution.{ext}", dpi=220,
                facecolor="white")
print(f"[fig11] -> {OUT}/fig11_diffusion_resolution.png")
