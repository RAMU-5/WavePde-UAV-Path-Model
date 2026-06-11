"""Fig. 4 (headline) — wave arrival times vs discrete ground truth (E2.1).

(a) scatter of calibrated wave pair times vs the string-pulled
    anisotropic-Dijkstra reference on the 5 m Nanshan domain.
(b) optimality-gap histogram, median / p95 annotated; the 50 m result
    is overlaid to show that the gap is a resolution property of the
    wave solve, not of the method.

Also prints the E2.3 lower-bound table: T / (d/c) detour-ratio stats.

Reads  output/runs/exp1_dijkstra/gt_domain*.npz,
       output/wave/pair_matrix*.npz
Writes paper/Q2/figs/fig04_groundtruth.{png,pdf}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
GT = ROOT / "output" / "runs" / "exp1_dijkstra"
OUT = ROOT / "paper" / "Q2" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

BIAS_S = 0.945          # E1.1 production point (dx=5, tau=0.4), paper/Q1
C0 = 15.0


def collect(gt_file, pm_file):
    gt = np.load(GT / gt_file)
    pm = np.load(ROOT / "output" / "wave" / pm_file)
    Tw, D = pm["T_matrix"], pm["distances_m"]
    rows = []
    for a, s in enumerate(gt["src_idx"]):
        tw = Tw[s] - BIAS_S
        ts = gt["T_str"][a]
        td = gt["T_dij"][a]
        m = (np.isfinite(tw) & np.isfinite(ts)
             & (D[s] >= 700.0) & (D[s] <= 3000.0))
        for b in np.where(m)[0]:
            rows.append((s, b, D[s, b], tw[b], ts[b], td[b]))
    r = np.asarray(rows)
    return dict(src=r[:, 0], dst=r[:, 1], d=r[:, 2],
                t_wave=r[:, 3], t_ref=r[:, 4], t_dij=r[:, 5])


c5 = collect("gt_domain_5m.npz", "pair_matrix_5m.npz")
c50 = collect("gt_domain.npz", "pair_matrix.npz")

gap5 = (c5["t_wave"] - c5["t_ref"]) / c5["t_ref"]
gap50 = (c50["t_wave"] - c50["t_ref"]) / c50["t_ref"]
pull5 = (c5["t_dij"] - c5["t_ref"]) / c5["t_ref"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

ax.plot(c5["t_ref"], c5["t_wave"], ".", ms=3, alpha=0.5, color="C0")
lim = [0, max(c5["t_ref"].max(), c5["t_wave"].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ground-truth time T_ref [s]  (Dijkstra + string pulling)")
ax.set_ylabel("wave arrival time (calibrated) [s]")
ax.set_title(f"(a) {len(gap5)} station pairs, 5 m Nanshan domain",
             fontsize=9)
ax.annotate(f"median gap {np.median(gap5):+.2%}\n"
            f"p95 |gap| {np.percentile(np.abs(gap5), 95):.2%}",
            xy=(0.04, 0.86), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", fc="w", alpha=0.8))

bins = np.linspace(-0.15, 0.15, 61)
ax2.hist(np.clip(gap50, bins[0], bins[-1]), bins=bins, alpha=0.45,
         label=f"50 m: med {np.median(gap50):+.1%}, "
               f"p95 {np.percentile(np.abs(gap50), 95):.1%}",
         color="C3", density=True)
ax2.hist(np.clip(gap5, bins[0], bins[-1]), bins=bins, alpha=0.6,
         label=f"5 m: med {np.median(gap5):+.1%}, "
               f"p95 {np.percentile(np.abs(gap5), 95):.1%}",
         color="C0", density=True)
ax2.axvline(0, color="k", lw=0.7)
ax2.set_xlabel("optimality gap (T_wave − T_ref) / T_ref")
ax2.set_ylabel("density")
ax2.set_title("(b) gap distribution: resolution effect", fontsize=9)
ax2.legend(fontsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig04_groundtruth.{ext}", dpi=200)
print(f"[fig04] -> {OUT}/fig04_groundtruth.png")

# ---- console summary for paper/Q2/README.md ---------------------------
ratio5 = c5["t_wave"] / (c5["d"] / C0)
print(json.dumps(dict(
    n_pairs_5m=len(gap5), n_pairs_50m=len(gap50),
    gap5_med=float(np.median(gap5)),
    gap5_p95=float(np.percentile(np.abs(gap5), 95)),
    gap5_mean=float(np.mean(gap5)),
    gap50_med=float(np.median(gap50)),
    gap50_p95=float(np.percentile(np.abs(gap50), 95)),
    dij_over_ref_med=float(np.median(pull5)),
    dij_over_ref_max=float(np.max(pull5)),
    detour_ratio_med=float(np.median(ratio5)),
    detour_ratio_p95=float(np.percentile(ratio5, 95)),
), indent=2))
