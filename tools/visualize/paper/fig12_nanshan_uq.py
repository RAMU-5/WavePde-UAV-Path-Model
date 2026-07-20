"""Plot E7.3 zero-wind Nanshan uncertainty qualification."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "runs" / "exp11_nanshan_uq"
OUT = ROOT / "paper" / "Q3" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((RUN / "results.json").read_text())
runs = meta["runs"]
data = np.load(RUN / "paths.npz")
source = meta["source_station"]
destinations = meta["destination_stations"]
station_xyz = data["station_xyz"]

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.75))

ax = axes[0]
buildings = data["building_footprint"].T
ax.imshow(buildings, origin="lower", cmap="Greys", interpolation="nearest",
          extent=(0, buildings.shape[1] * 5 / 1000,
                  0, buildings.shape[0] * 5 / 1000), alpha=0.75)
for destination in destinations:
    path = data[f"path_D0_dst{destination}"]
    ax.plot(path[:, 0] / 1000, path[:, 1] / 1000, lw=1.5,
            label=f"50$\\to${destination}")
ax.scatter(station_xyz[source, 0] / 1000, station_xyz[source, 1] / 1000,
           marker="*", s=80, color="tab:red", zorder=5)
ax.set(xlabel="x [km]", ylabel="y [km]", title="(a) Nanshan nominal routes")
ax.legend(fontsize=8)

for destination in destinations:
    rows = sorted((r for r in runs if r["destination_station"] == destination),
                  key=lambda r: r["D_diff_m2_s"])
    D = np.array([r["D_diff_m2_s"] for r in rows])
    probability = np.array([r["obstacle_collision_probability"] for r in rows])
    low = np.array([r["obstacle_collision_ci_low"] for r in rows])
    high = np.array([r["obstacle_collision_ci_high"] for r in rows])
    axes[1].errorbar(
        D, 100 * probability,
        yerr=[100 * np.maximum(probability - low, 0),
              100 * np.maximum(high - probability, 0)],
        marker="o", capsize=2, label=f"50$\\to${destination}",
    )
    axes[2].plot(D, [r["min_clearance_margin95_m"] for r in rows],
                 "o-", label=f"50$\\to${destination}")

axes[1].set(xlabel="$D_{diff}$ [m$^2$/s]", ylabel="building-collision probability [%]",
            title="(b) building risk (95% CI)")
axes[2].axhline(0, color="k", lw=0.8, ls="--")
axes[2].set(xlabel="$D_{diff}$ [m$^2$/s]", ylabel="minimum clearance $-r_{95}$ [m]",
            title="(c) conservative safety margin")
for ax in axes[1:]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig12_nanshan_uq.{ext}", dpi=220)
print(f"[fig12] -> {OUT}/fig12_nanshan_uq.png")
