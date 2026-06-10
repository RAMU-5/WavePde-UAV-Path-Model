"""Viz 5 (scheduling): per-station UAV fleet inventory over time.

Shows how UAVs get pulled from stations near business circles and
deposited at residential stations after delivery — visualises the
imbalance dynamics and overflow risk.

  - Heatmap: station × time, value = idle UAV count
  - Line plot: 7 primary-circle stations + 3 most-receiving stations
  - Peak / valley table
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from business_circles import (load_circles_with_stations, NANSHAN_CIRCLES)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM = ROOT / "visualization" / "scheduling_system" / "sim_log.npz"
OUT = ROOT / "visualization" / "scheduling_system" / "05_station_fleet.png"


def main():
    z = np.load(SIM, allow_pickle=False)
    info = load_circles_with_stations()
    log_idle = z["log_idle_per_station"]    # (n_ticks, 118)
    initial = int(z["initial_per_station"])
    cap = int(z["max_per_station"])
    tick_sec = float(z["tick_sec"])
    t_start = float(z["sim_t_start"])
    n_ticks, n_st = log_idle.shape
    t_h = t_start + np.arange(n_ticks) * tick_sec / 3600.0
    deadhead_out = z.get("log_deadhead_outbound", None) \
        if hasattr(z, "get") else (z["log_deadhead_outbound"]
                                   if "log_deadhead_outbound" in z.files
                                   else None)
    deadhead_in = (z["log_deadhead_inbound"]
                   if "log_deadhead_inbound" in z.files else None)
    n_deadhead = (int(z["n_deadhead_total"])
                  if "n_deadhead_total" in z.files else 0)

    # Sort stations by total deviation from initial (most volatile first)
    deviation = np.abs(log_idle - initial).sum(axis=0)
    order = np.argsort(-deviation)
    log_sorted = log_idle[:, order]

    # Identify the 7 primary circle stations
    primary_stations = info["circle_to_station"]
    # And the 3 stations with maximum peak count (popular delivery sinks)
    peak_idle = log_idle.max(axis=0)
    receivers = np.argsort(-peak_idle)[:3]

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          width_ratios=[1.0, 1.0])
    ax_heat = fig.add_subplot(gs[0, :])
    ax_circ = fig.add_subplot(gs[1, 0])
    ax_recv = fig.add_subplot(gs[1, 1])

    # Heatmap (stations sorted by volatility)
    im = ax_heat.imshow(log_sorted.T, aspect="auto", origin="lower",
                        cmap="RdYlBu_r",
                        vmin=0, vmax=cap,
                        extent=[t_h[0], t_h[-1], 0, n_st])
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.025)
    cbar.set_label(f"idle UAVs at station   (init={initial}, cap={cap})")
    cbar.ax.axhline(initial, color="black", lw=1)
    cbar.ax.axhline(cap, color="red", lw=1)
    ax_heat.set_xlabel("hour of day")
    ax_heat.set_ylabel("station (sorted by volatility, top = most variable)")
    ax_heat.set_title(f"Per-station idle-UAV count over time  "
                      f"({n_st} stations × {n_ticks} ticks)")

    # Per-circle primary station lines
    for k, c in enumerate(NANSHAN_CIRCLES):
        s = int(primary_stations[k])
        ax_circ.plot(t_h, log_idle[:, s], lw=1.5,
                     label=f"{c.name_en} (st {s})")
    ax_circ.axhline(initial, color="black", ls="--", lw=0.8,
                    label=f"initial = {initial}")
    ax_circ.axhline(0, color="red", ls=":", lw=0.8,
                    label="depleted = 0")
    ax_circ.axhline(cap, color="orange", ls=":", lw=0.8,
                    label=f"capacity = {cap}")
    ax_circ.set_xlabel("hour of day")
    ax_circ.set_ylabel("idle UAVs")
    ax_circ.set_title("Primary circle stations — pickup demand drains pool")
    ax_circ.legend(fontsize=7, loc="lower left", ncol=2)
    ax_circ.grid(True, alpha=0.3)

    # Top receiver stations
    for s in receivers:
        ax_recv.plot(t_h, log_idle[:, s], lw=1.6,
                     label=f"station {s}  peak={peak_idle[s]}")
    ax_recv.axhline(initial, color="black", ls="--", lw=0.8,
                    label=f"initial = {initial}")
    ax_recv.axhline(cap, color="red", ls="--", lw=1.2,
                    label=f"capacity = {cap}")
    n_overflow_events = int(z["log_overflow"].sum())
    ax_recv.text(0.02, 0.95,
                 f"Stations exceeding capacity: {n_overflow_events} events",
                 transform=ax_recv.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.85))
    ax_recv.set_xlabel("hour of day")
    ax_recv.set_ylabel("idle UAVs")
    ax_recv.set_title("Top 3 receiver stations — delivery surplus accumulates")
    ax_recv.legend(fontsize=8, loc="upper left")
    ax_recv.grid(True, alpha=0.3)

    fig.suptitle("Station-level fleet dynamics  "
                 f"(initial 60 UAVs/station, cap {cap}, "
                 f"{n_deadhead:,} deadhead repositions during sim)",
                 fontsize=13)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[saved] {OUT}")
    print(f"  total overflow events: {n_overflow_events}")
    print(f"  most-drained station: peak deficit = "
          f"{(initial - log_idle.min(axis=0)).max()}")
    print(f"  most-saturated station: peak count = {peak_idle.max()}")


if __name__ == "__main__":
    main()
