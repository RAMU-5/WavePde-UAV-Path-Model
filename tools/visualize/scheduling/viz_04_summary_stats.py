"""Viz 4 (scheduling): final summary stats from the simulation log.

  - Order outcome donut: delivered / failed_no_route / failed_timeout / pending
  - Histogram of total delivery time (created → delivered)
  - Histogram of waiting time (created → assigned)
  - Per-circle order count vs delivery rate
  - UAV utilisation (% of time non-idle)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from business_circles import NANSHAN_CIRCLES

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM = ROOT / "visualization" / "scheduling_system" / "sim_log.npz"
OUT = ROOT / "visualization" / "scheduling_system" / "04_summary_stats.png"

CIRCLE_COLORS = {"office": "#5fc8ff", "mall": "#ff7755",
                 "tech": "#3affb3", "entertainment": "#ffaa44"}


def main():
    z = np.load(SIM, allow_pickle=False)
    o_circle = z["order_circle"]
    o_created = z["order_created_t"]
    o_assigned = z["order_assigned_t"]
    o_delivered = z["order_delivered_t"]
    o_status_int = z["order_status_int"]
    # 0 pending, 1 assigned, 2 loading, 3 in_transit, 4 delivered, 5 failed
    n_uavs = int(z["n_uavs"])
    log_busy = z["log_busy"]
    n_orders = len(o_circle)

    # ---- Outcome counts
    delivered = (o_status_int == 4)
    failed_route = np.zeros(n_orders, dtype=bool)
    failed_timeout = (o_status_int == 5)
    pending = ~(delivered | failed_timeout)
    n_delivered = int(delivered.sum())
    n_fail_route = int(failed_route.sum())
    n_fail_timeout = int(failed_timeout.sum())
    n_pending = int(pending.sum())

    # ---- Times
    delivery_time = (o_delivered - o_created)[delivered]      # total
    waiting_time = np.where(
        o_assigned >= 0, o_assigned - o_created, np.nan)
    waiting_time = waiting_time[delivered]

    # ---- Layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3)
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_total = fig.add_subplot(gs[0, 1])
    ax_wait = fig.add_subplot(gs[0, 2])
    ax_circle = fig.add_subplot(gs[1, 0])
    ax_util = fig.add_subplot(gs[1, 1:])
    ax_batch = fig.add_subplot(gs[2, :])

    # Pie / donut
    sizes = [n_delivered, n_fail_route, n_fail_timeout, n_pending]
    labels = [f"delivered ({n_delivered})",
              f"no route ({n_fail_route})",
              f"timeout ({n_fail_timeout})",
              f"pending @end ({n_pending})"]
    colors = ["#33cc66", "#cc4433", "#ff8833", "#888888"]
    nz = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    ax_pie.pie([s for s, _, _ in nz], labels=[l for _, l, _ in nz],
               colors=[c for _, _, c in nz], autopct="%.1f%%",
               wedgeprops={"width": 0.4, "edgecolor": "white"},
               textprops={"fontsize": 9})
    ax_pie.set_title(f"Order outcomes  (n={n_orders})", fontsize=11)

    # Total delivery time histogram
    if delivery_time.size:
        ax_total.hist(delivery_time, bins=40, color="#5fc8ff",
                      edgecolor="black", alpha=0.85)
        med = np.median(delivery_time)
        p95 = np.percentile(delivery_time, 95)
        ax_total.axvline(med, color="black", lw=1.5,
                         label=f"median {med:.0f}s")
        ax_total.axvline(p95, color="red", lw=1.5,
                         label=f"p95 {p95:.0f}s")
    ax_total.set_xlabel("total time (s)  created → delivered")
    ax_total.set_ylabel("# orders")
    ax_total.set_title("Total delivery time distribution")
    ax_total.legend(); ax_total.grid(True, alpha=0.3)

    # Waiting time histogram
    valid_wait = waiting_time[~np.isnan(waiting_time)]
    if valid_wait.size:
        ax_wait.hist(valid_wait, bins=40, color="#ff8833",
                     edgecolor="black", alpha=0.85)
        med = np.median(valid_wait)
        p95 = np.percentile(valid_wait, 95)
        ax_wait.axvline(med, color="black", lw=1.5,
                        label=f"median {med:.0f}s")
        ax_wait.axvline(p95, color="red", lw=1.5,
                        label=f"p95 {p95:.0f}s")
    ax_wait.set_xlabel("waiting time (s)  created → assigned")
    ax_wait.set_ylabel("# orders")
    ax_wait.set_title("Order waiting time")
    ax_wait.legend(); ax_wait.grid(True, alpha=0.3)

    # Per-circle bar (and scatter)
    K = len(NANSHAN_CIRCLES)
    # include scatter (-1) as last bar
    sc_mask = (o_circle == -1)
    cnt_total = np.zeros(K + 1, dtype=int)
    cnt_done = np.zeros(K + 1, dtype=int)
    for k in range(K):
        m_total = (o_circle == k)
        cnt_total[k] = int(m_total.sum())
        cnt_done[k] = int((m_total & delivered).sum())
    cnt_total[K] = int(sc_mask.sum())
    cnt_done[K] = int((sc_mask & delivered).sum())
    rate = cnt_done / np.maximum(cnt_total, 1) * 100
    x_pos = np.arange(K + 1)
    cols = [CIRCLE_COLORS[c.profile] for c in NANSHAN_CIRCLES] + ["#888888"]
    labels = [c.name_en for c in NANSHAN_CIRCLES] + ["scatter (118 stations)"]
    ax_circle.bar(x_pos, cnt_total, color="#cccccc",
                  edgecolor="black", label="total")
    ax_circle.bar(x_pos, cnt_done, color=cols,
                  edgecolor="black", label="delivered")
    for k, r in enumerate(rate):
        ax_circle.text(k, cnt_total[k] + 1, f"{r:.0f}%",
                       ha="center", fontsize=8)
    ax_circle.set_xticks(x_pos)
    ax_circle.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax_circle.set_ylabel("# orders")
    ax_circle.set_title("Order volume + delivery rate (hubs + scatter)")
    ax_circle.legend(loc="upper right")
    ax_circle.grid(True, axis="y", alpha=0.3)

    # UAV utilisation timeline
    tick_sec = float(z["tick_sec"])
    t_start = float(z["sim_t_start"])
    t_h = t_start + np.arange(len(log_busy)) * tick_sec / 3600.0
    util = log_busy / n_uavs * 100
    ax_util.fill_between(t_h, 0, util, color="#33aa66", alpha=0.6)
    ax_util.plot(t_h, util, color="#33aa66", lw=1.4)
    ax_util.axhline(util.mean(), color="black", ls="--", lw=1.0,
                    label=f"mean {util.mean():.1f}%")
    peak = util.max()
    ax_util.axhline(peak, color="red", ls=":", lw=1.0,
                    label=f"peak {peak:.0f}%")
    ax_util.set_xlabel("hour of day")
    ax_util.set_ylabel("UAV utilisation (%)")
    ax_util.set_title(f"Fleet utilisation timeline  (N_UAV = {n_uavs})")
    ax_util.set_ylim(0, 105)
    ax_util.legend(loc="upper right")
    ax_util.grid(True, alpha=0.3)

    # Batching stats — read true size distribution from sim log
    n_batches = int(z["n_batches"]) if "n_batches" in z.files else 0
    n_batched_orders = int(z["n_batched_orders"]) if "n_batched_orders" in z.files else 0
    avg_batch = (n_orders / max(n_batches, 1)) if n_batches > 0 else 1.0
    if "n_batch_size_count" in z.files:
        size_count = z["n_batch_size_count"]   # idx = batch size
    else:
        size_count = None

    if n_batches > 0 and size_count is not None:
        sizes = np.arange(1, len(size_count))
        trips_by_size = size_count[1:]
        orders_by_size = trips_by_size * sizes
        labels = [f"solo (size 1)", "pair (size 2)", "triple (size 3)",
                  "quad (size 4)", "5-batch", "6-batch"][:len(sizes)]
        x = np.arange(len(sizes))
        w = 0.4
        ax_batch.bar(x - w/2, trips_by_size, w, color="#888888",
                     edgecolor="black", label="# trips")
        ax_batch.bar(x + w/2, orders_by_size, w, color="#33aa66",
                     edgecolor="black", label="# orders carried")
        for i, (n_t, n_o) in enumerate(zip(trips_by_size, orders_by_size)):
            if n_t == 0: continue
            pct_trips = n_t * 100 / max(n_batches, 1)
            pct_orders = n_o * 100 / max(n_orders, 1)
            ax_batch.text(i - w/2, n_t, f"{pct_trips:.0f}%",
                          ha="center", va="bottom", fontsize=8)
            ax_batch.text(i + w/2, n_o, f"{pct_orders:.0f}%",
                          ha="center", va="bottom", fontsize=8)
        ax_batch.set_xticks(x); ax_batch.set_xticklabels(labels[:len(sizes)])
        ax_batch.set_ylabel("count")
        ax_batch.set_title(
            f"Order batching distribution — "
            f"{n_batches:,d} trips deliver {n_orders:,d} orders  "
            f"(avg = {avg_batch:.2f}, "
            f"effective fleet × {avg_batch:.2f})",
            fontsize=11)
        ax_batch.legend(loc="upper left")
        ax_batch.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Nanshan delivery sim — {n_orders:,d} orders, "
                 f"{n_delivered:,d} delivered "
                 f"({n_delivered/max(n_orders,1)*100:.1f}%), "
                 f"avg batch = {avg_batch:.2f}",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[saved] {OUT}")

    # Print headline numbers
    print(f"\n=== HEADLINE NUMBERS ===")
    print(f"  orders generated      : {n_orders}")
    print(f"  delivered             : {n_delivered} "
          f"({n_delivered/max(n_orders,1)*100:.1f}%)")
    if delivery_time.size:
        print(f"  median delivery time  : {np.median(delivery_time):.0f}s "
              f"({np.median(delivery_time)/60:.1f} min)")
        print(f"  p95 delivery time     : {np.percentile(delivery_time,95):.0f}s "
              f"({np.percentile(delivery_time,95)/60:.1f} min)")
    if valid_wait.size:
        print(f"  median wait to assign : {np.median(valid_wait):.0f}s")
    print(f"  fleet utilisation mean: {util.mean():.1f}%  peak: {peak:.0f}%")


if __name__ == "__main__":
    main()
