"""Plot concurrent-flight and time-varying-velocity experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MULTI = ROOT / "output" / "runs" / "exp13_multi_uav"
DYNAMIC = ROOT / "output" / "runs" / "exp14_dynamic_velocity"
OUT = ROOT / "paper" / "Q4" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

multi_meta = json.loads((MULTI / "results.json").read_text())
dynamic_meta = json.loads((DYNAMIC / "results.json").read_text())
multi_data = np.load(MULTI / "figure_data.npz")

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55))

# (a) The 20-mission reciprocal-route scenario used for the flight traces.
ax = axes[0]
footprint = multi_data["building_footprint"].T
dx, dy, _ = multi_meta["spacing_m"]
ax.imshow(
    footprint,
    origin="lower",
    cmap="Greys",
    interpolation="nearest",
    extent=(0, footprint.shape[1] * dx / 1000, 0, footprint.shape[0] * dy / 1000),
    alpha=0.72,
)
paths = multi_data["paths"]
offsets = multi_data["path_offsets"]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for mission in range(20):
    path = paths[offsets[mission] : offsets[mission + 1]]
    ax.plot(path[:, 0] / 1000, path[:, 1] / 1000, lw=1.05,
            color=colors[mission // 2], alpha=0.9)
ax.set(
    xlabel="x [km]",
    ylabel="y [km]",
    title="(a) 20 reciprocal Nanshan missions",
    aspect="equal",
)

# (b) Actual closed-loop conflicts and the scheduling cost.
ax = axes[1]
runs = multi_meta["runs"]
fleets = np.asarray([run["fleet_size"] for run in runs])
conflicts_un = np.asarray([run["uncoordinated"]["conflicting_uav_pairs"] for run in runs])
conflicts_co = np.asarray([run["coordinated"]["conflicting_uav_pairs"] for run in runs])
width = 5.4
ax.bar(fleets - width / 2, conflicts_un, width=width, color="tab:red",
       alpha=0.75, label="simultaneous launch")
ax.bar(fleets + width / 2, conflicts_co, width=width, color="tab:blue",
       alpha=0.75, label="CBS delayed launch")
ax.set(
    xlabel="missions",
    ylabel="conflicting UAV pairs",
    xticks=fleets,
    title="(b) closed-loop deconfliction",
)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=7.5, loc="upper left")
delay_ax = ax.twinx()
delay = [run["coordinated"]["mean_launch_delay_s"] for run in runs]
delay_ax.plot(fleets, delay, "ko--", ms=4, lw=1.1, label="mean delay")
delay_ax.set_ylabel("mean launch delay [s]")
delay_ax.set_ylim(0, 90)

# (c) Exact-solution error under a rotating, time-dependent velocity field.
ax = axes[2]
periods = np.asarray([run["period_s"] for run in dynamic_meta["runs"]])
for key, label, style in (
    ("dynamic_solver", "time-varying solver", "o-"),
    ("frozen_at_emission", "velocity frozen at $t_0$", "s--"),
    ("omit_dqdt", "omit $\\partial_t q$", "^-."),
):
    error = [run[key]["p95_abs_gap_pct"] for run in dynamic_meta["runs"]]
    ax.plot(periods, error, style, lw=1.5, ms=5, label=label)
ax.set(
    xlabel="velocity rotation period [s]",
    ylabel="P95 absolute arrival-time gap [%]",
    xticks=periods,
    title="(c) dynamic-velocity validation",
)
ax.set_yscale("log")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=7.5)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig13_dynamic_operations.{ext}", dpi=220)
print(f"[fig13] -> {OUT}/fig13_dynamic_operations.png")
