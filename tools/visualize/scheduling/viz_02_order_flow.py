"""Viz 2 (scheduling): order flow over the simulated time window.

Plots:
  - Stacked area: orders/min by business circle (shows which circle is busiest when)
  - Line: pending queue length over time
  - Line: # busy UAVs over time
  - Line: completed/min and failed/min
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from business_circles import NANSHAN_CIRCLES

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM = ROOT / "output" / "figures" / "scheduling_system" / "sim_log.npz"
OUT = ROOT / "output" / "figures" / "scheduling_system" / "02_order_flow.png"

CIRCLE_COLORS_BY_TYPE = {
    "office": "#5fc8ff", "mall": "#ff7755",
    "tech":   "#3affb3", "entertainment": "#ffaa44",
}


def main():
    z = np.load(SIM, allow_pickle=False)
    t_start = float(z["sim_t_start"])
    tick_sec = float(z["tick_sec"])
    n_uavs = int(z["n_uavs"])
    log_pending = z["log_pending"]
    log_busy = z["log_busy"]
    log_completed = z["log_completed"]
    log_failed = z["log_failed"]
    log_per_circle = z["log_arriving_per_circle"]
    n_ticks = len(log_pending)
    # Scatter orders (circle_id = -1) — bin by tick
    o_circle = z["order_circle"]
    o_created = z["order_created_t"]
    is_scatter = o_circle == -1
    scatter_per_tick = np.zeros(n_ticks, dtype=np.int32)
    s_ticks = (o_created[is_scatter] / tick_sec).astype(int)
    np.add.at(scatter_per_tick, np.clip(s_ticks, 0, n_ticks-1), 1)

    t_h = t_start + np.arange(n_ticks) * tick_sec / 3600.0

    # Bin to 5-min averages for smoother plots
    bin_size = max(1, int(300 / tick_sec))
    n_bins = n_ticks // bin_size

    def binned(arr):
        return arr[:n_bins*bin_size].reshape(n_bins, bin_size).sum(axis=1)
    def binned_avg(arr):
        return arr[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)

    t_bins = t_h[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
    arrive_bin = log_per_circle[:n_bins*bin_size].reshape(
        n_bins, bin_size, log_per_circle.shape[1]).sum(axis=1)
    scatter_bin = scatter_per_tick[:n_bins*bin_size].reshape(
        n_bins, bin_size).sum(axis=1)
    completed_bin = binned(log_completed)
    failed_bin = binned(log_failed)
    pending_bin = binned_avg(log_pending)
    busy_bin = binned_avg(log_busy)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    ax_in, ax_q, ax_done = axes

    # ---- Top: stacked area of arrivals by circle ----
    bottoms = np.zeros(n_bins)
    # Scatter at the bottom
    ax_in.fill_between(t_bins, bottoms, bottoms + scatter_bin / 5.0,
                       color="#888888", edgecolor="black", linewidth=0.3,
                       alpha=0.7, label="scatter (118 stations, small shops)")
    bottoms += scatter_bin / 5.0
    for k, c in enumerate(NANSHAN_CIRCLES):
        col = CIRCLE_COLORS_BY_TYPE[c.profile]
        ax_in.fill_between(t_bins, bottoms, bottoms + arrive_bin[:, k] / 5.0,
                           color=col, edgecolor="black", linewidth=0.3,
                           alpha=0.85, label=f"{c.name_en} ({c.profile})")
        bottoms += arrive_bin[:, k] / 5.0
    ax_in.set_ylabel("orders / minute")
    total_orders = int(arrive_bin.sum() + scatter_bin.sum())
    ax_in.set_title(f"Order arrivals — hub circles + scatter (5-min bins, "
                    f"total={total_orders:,})", fontsize=11)
    ax_in.legend(loc="upper right", fontsize=8, ncol=2)
    ax_in.grid(True, alpha=0.3)
    ax_in.axvspan(17.5, 18.0, alpha=0.05, color="gray")
    ax_in.axvspan(20.0, 21.5, alpha=0.05, color="gray")

    # ---- Mid: pending queue and busy UAV count ----
    ax_q.plot(t_bins, pending_bin, color="#cc3344", lw=2.0,
              label="pending queue length")
    ax_q.fill_between(t_bins, 0, pending_bin, alpha=0.2, color="#cc3344")
    ax_q.set_ylabel("orders pending", color="#cc3344")
    ax_q.tick_params(axis="y", labelcolor="#cc3344")
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("System backlog and UAV utilisation", fontsize=11)

    ax_q2 = ax_q.twinx()
    ax_q2.plot(t_bins, busy_bin, color="#33aa66", lw=2.0,
               label="busy UAVs")
    ax_q2.fill_between(t_bins, 0, busy_bin, alpha=0.15, color="#33aa66")
    ax_q2.axhline(n_uavs, color="black", ls="--", lw=0.8,
                  label=f"fleet size = {n_uavs}")
    ax_q2.set_ylabel(f"UAVs in flight (of {n_uavs})", color="#33aa66")
    ax_q2.tick_params(axis="y", labelcolor="#33aa66")
    ax_q2.set_ylim(0, n_uavs * 1.05)

    # ---- Bottom: completed vs failed per minute ----
    ax_done.fill_between(t_bins, 0, completed_bin / 5.0,
                         alpha=0.6, color="#33aa66", label="delivered/min")
    ax_done.fill_between(t_bins, 0, -failed_bin / 5.0,
                         alpha=0.6, color="#cc3344", label="failed/min")
    ax_done.axhline(0, color="black", lw=0.5)
    ax_done.set_xlabel("hour of day")
    ax_done.set_ylabel("orders / minute")
    ax_done.set_title("Throughput (delivered above zero, failed below)",
                      fontsize=11)
    ax_done.legend(loc="upper right")
    ax_done.grid(True, alpha=0.3)

    fig.suptitle(f"Nanshan delivery sim — orders={int(arrive_bin.sum())}, "
                 f"delivered={int(completed_bin.sum())}, "
                 f"failed={int(failed_bin.sum())}",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
