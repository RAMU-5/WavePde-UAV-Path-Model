"""5-way comparison of scheduling methods on the Nanshan T_matrix.

Methods:
    0. Mean-field (soft_lsa)  — original, may drop expensive feasible tasks
    A. Mean-field (hybrid LSA) — feasibility-first, all 22 feasible kept
    C. Mean-field (K=2)        — each UAV may do up to 2 tasks
    D. Hungarian (§4)
    E. Greedy
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

THIS_DIR = Path(__file__).resolve().parent

METHODS = [
    ("MF (soft)",        "assignment_meanfield_soft",   "#5fc8ff"),
    ("MF (hybrid α=0.5)", "assignment_meanfield_hybrid", "#3affb3"),
    ("MF (K=2)",         "assignment_meanfield_K2",     "#ffaa55"),
    ("Hungarian (§4)",   "assignment_hungarian",        "#ff7777"),
    ("Greedy",           "assignment_greedy",           "#cccccc"),
]


def picked_costs(C, a):
    out = []
    for mu, i in enumerate(a):
        if i >= 0 and np.isfinite(C[i, mu]):
            out.append(C[i, mu])
    return np.asarray(out)


def main():
    z = np.load(THIS_DIR / "outputs" / "scheduler_demo.npz", allow_pickle=False)
    C = z["C"]
    uav_idx = z["uav_idx"]; task_idx = z["task_idx"]
    d = np.load(THIS_DIR / "domain.npz", allow_pickle=False)
    stations_xy_m = d["stations_xy_m"]
    mask = d["mask"]
    dx, dy = float(d["dx"]), float(d["dy"])
    nx, ny = mask.shape[0], mask.shape[1]
    Lx, Ly = nx * dx, ny * dy

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.4, 1.0])

    # ---- Top row: 5 mini-maps showing each method's assignments -----------
    uav_xy = stations_xy_m[uav_idx, :2]
    task_xy = stations_xy_m[task_idx, :2]
    fp = mask.any(axis=2)

    summary = []
    for col, (label, key, color) in enumerate(METHODS):
        a = z[key]
        cs = picked_costs(C, a)
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#0d0d18")
        ax.contour(fp.T, levels=[0.5], colors="#666666", linewidths=0.3,
                   extent=[0, Lx, 0, Ly])
        ax.scatter(uav_xy[:, 0], uav_xy[:, 1], s=24, marker="s",
                   facecolor="#5fc8ff", edgecolor="white", linewidths=0.4,
                   zorder=8)
        ax.scatter(task_xy[:, 0], task_xy[:, 1], s=24, marker="v",
                   facecolor="#ff6f5f", edgecolor="white", linewidths=0.4,
                   zorder=8)
        segs = []; cmap_v = []
        load_count = np.zeros(len(uav_idx), dtype=int)
        for mu, i in enumerate(a):
            if i < 0 or not np.isfinite(C[i, mu]): continue
            segs.append([uav_xy[i], task_xy[mu]])
            cmap_v.append(C[i, mu])
            load_count[i] += 1
        if segs:
            cv = np.asarray(cmap_v)
            lc = LineCollection(segs, cmap="plasma_r",
                                norm=plt.Normalize(vmin=40, vmax=160),
                                linewidths=1.2, alpha=0.85, zorder=5)
            lc.set_array(cv)
            ax.add_collection(lc)
        # Mark UAVs serving 2 tasks (only relevant for K=2)
        for i, n_load in enumerate(load_count):
            if n_load > 1:
                ax.scatter(uav_xy[i, 0], uav_xy[i, 1], s=140, marker="s",
                           facecolor="none", edgecolor="#ffd633",
                           linewidths=2.2, zorder=10)
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        avg = cs.sum() / max(len(cs), 1)
        title = (f"{label}\nΣ={cs.sum():.0f}s  "
                 f"{len(cs)}/{len(task_idx)} done  "
                 f"avg={avg:.1f}  max={cs.max() if len(cs) else 0:.0f}")
        ax.set_title(title, fontsize=10, color="white",
                     backgroundcolor=color, pad=4)
        summary.append((label, cs))

    # ---- Bottom-left: bar chart of summary stats -------------------------
    ax_bar = fig.add_subplot(gs[1, :2])
    labels = [s[0] for s in summary]
    sums = [s[1].sum() for s in summary]
    counts = [len(s[1]) for s in summary]
    avgs = [s[1].mean() if len(s[1]) else 0 for s in summary]
    maxes = [s[1].max() if len(s[1]) else 0 for s in summary]
    x_pos = np.arange(len(labels))
    width = 0.22
    colors = [m[2] for m in METHODS]
    ax_bar.bar(x_pos - 1.5*width, sums, width, label="Σ total (s)",
               color=colors, edgecolor="black")
    ax_bar.bar(x_pos - 0.5*width, [a*30 for a in avgs], width,
               label="avg×30 (s)", color=colors, edgecolor="black",
               alpha=0.7, hatch="//")
    ax_bar.bar(x_pos + 0.5*width, [c*55 for c in counts], width,
               label="# done × 55", color=colors, edgecolor="black",
               alpha=0.5, hatch="..")
    ax_bar.bar(x_pos + 1.5*width, maxes, width, label="max single (s)",
               color=colors, edgecolor="black", alpha=0.4, hatch="xx")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=15, fontsize=9)
    ax_bar.set_ylabel("seconds")
    ax_bar.set_title("Method comparison")
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(True, axis="y", alpha=0.3)

    # ---- Bottom-right: cost histograms overlaid --------------------------
    ax_hist = fig.add_subplot(gs[1, 2:])
    bins = np.linspace(40, 280, 25)
    for (label, cs), color in zip(summary, colors):
        ax_hist.hist(cs, bins=bins, alpha=0.45, label=label,
                     color=color, edgecolor="black", linewidth=0.4)
    ax_hist.set_xlabel("travel cost C[i,μ] (s)")
    ax_hist.set_ylabel("# tasks")
    ax_hist.set_title("Cost distribution per method")
    ax_hist.legend(fontsize=9, loc="upper right")
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle("v3 §4-§5 scheduling — 5 methods on Nanshan 30×30 demo",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    out = THIS_DIR / "outputs" / "scheduler_demo.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
