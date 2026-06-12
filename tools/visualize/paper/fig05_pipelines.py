"""Fig.5 — efficiency: adjoint + multicolor pipelines (paper Q4,
claims C3/C4/C7; data from output/runs/exp7_efficiency/results.json).

(a) multicolor crosstalk: max |band error| of a batched solve vs the
    source separation — the empirical safe threshold lands exactly on
    the theoretical 2 x band radius (6 km).
(b) full 118x118 matrix wall time vs separation threshold (solves =
    colour count x warm solve), hatched where (a) shows corruption.
(c) the three pipelines: full matrix naive / multicolor@6km; one
    destination column forward x118 / adjoint x1, annotated with the
    measured accuracy.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = json.load(open(ROOT / "output" / "runs" / "exp7_efficiency"
                     / "results.json"))
FIG_DIR = ROOT / "paper" / "Q4" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# ---------------- (a) crosstalk vs separation ------------------------
ax = axes[0]
xt = RES["e32_crosstalk"]
sep = np.array([c["sep_m"] for c in xt]) / 1e3
mx = np.array([max(c["a_max"], c["b_max"]) for c in xt])
fb = np.array([max(c["a_frac_bad"], c["b_frac_bad"]) for c in xt])
ax.semilogy(sep, np.maximum(mx, 1e-4), "o-", color="tab:blue",
            label="max |error|, 2 sources")
for b in RES["e32_bigclass"]:
    ax.semilogy(b["sep_m"] / 1e3, b["max"], "^", ms=9, color="tab:red",
                label=f"max |error|, k={b['k']} batch")
ax.axvline(6.0, color="k", ls="--", lw=1)
ax.text(6.1, 3e-4, "2$\\times$band radius\n(theory: no overlap)",
        fontsize=8)
ax.axhline(0.01, color="gray", ls=":", lw=1)
ax.text(1.0, 0.012, "1%", fontsize=8, color="gray")
ax2 = ax.twinx()
ax2.plot(sep, fb, "s--", color="tab:orange", ms=4, alpha=0.7)
ax2.set_ylabel("corrupted station fraction", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(0, 1.05)
ax.set_xlabel("source separation [km]")
ax.set_ylabel("relative band-pair error")
ax.set_title("(a) batched-solve crosstalk")
ax.legend(loc="center right", fontsize=8)

# ---------------- (b) matrix cost vs separation ----------------------
ax = axes[1]
cols = RES["e32_colours"]
seps = np.array([c["sep_m"] for c in cols]) / 1e3
ncol = np.array([c["n_colours"] for c in cols])
warm = RES["pipelines"]["warm_solve_s"]
naive_min = 118 * warm / 60
safe = seps >= 6.0
bars = ax.bar(seps[~safe], ncol[~safe] * warm / 60, width=0.7,
              color="tab:red", alpha=0.5, hatch="//",
              label="corrupted (see a)")
ax.bar(seps[safe], ncol[safe] * warm / 60, width=0.7,
       color="tab:green", alpha=0.8, label="clean")
ax.axhline(naive_min, color="k", ls="-", lw=1.2)
ax.text(2.0, naive_min - 8, f"naive 118 solves = {naive_min:.0f} min",
        fontsize=8)
for s, n in zip(seps, ncol):
    ax.text(s, n * warm / 60 + 1.5, f"{n}", ha="center", fontsize=8)
ax.set_ylim(0, naive_min * 1.12)
ax.set_xlabel("colour separation threshold [km]")
ax.set_ylabel("full 118$\\times$118 matrix wall time [min, 1 A100]")
ax.set_title("(b) multicolor matrix cost (labels: #solves)")
ax.legend(loc="upper left", fontsize=8)

# ---------------- (c) pipelines ---------------------------------------
ax = axes[2]
e31 = RES["e31"]
t_back = RES["pipelines"]["one_destination_backward_s"] / 60
mc6 = RES["pipelines"]["full_matrix_multicolor_s"]["6000"] / 60
vals = [naive_min, mc6, naive_min, t_back]
labels = ["naive\n118 fwd", "multicolor\n@6 km, 82",
          "fwd column\n118 solves", "adjoint\n1 solve"]
colors = ["tab:gray", "tab:green", "tab:gray", "tab:blue"]
xpos = [0, 1, 2.5, 3.5]
ax.bar(xpos, vals, color=colors, width=0.8, log=True)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("wall time [min, 1 A100, log]")
bigerr = max(b["max"] for b in RES["e32_bigclass"]
             if b["sep_m"] >= 6000) if any(
    b["sep_m"] >= 6000 for b in RES["e32_bigclass"]) else None
xt6 = [max(c["a_max"], c["b_max"]) for c in RES["e32_crosstalk"]
       if c["sep_m"] >= 5990]
ax.text(1, mc6 * 1.15, f"max err {100*max(xt6):.1f}%",
        ha="center", fontsize=8)
ax.text(3.5, t_back * 2.2,
        f"{naive_min/t_back:.0f}$\\times$",
        ha="center", fontsize=10, fontweight="bold", color="tab:blue")
ax.text(3.5, t_back * 1.25,
        f"med {100*e31['med']:.2f}%  p95 {100*e31['p95']:.1f}%",
        ha="center", fontsize=7)
ax.text(0.5, max(vals) * 1.45, "full matrix", ha="center", fontsize=9)
ax.text(3.0, max(vals) * 1.45, "one destination", ha="center",
        fontsize=9)
ax.axvline(1.75, color="gray", lw=0.5, ls=":")
ax.set_ylim(bottom=0.3, top=max(vals) * 2.6)
ax.set_title("(c) pipelines (annotated: accuracy)")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"fig05_pipelines.{ext}", dpi=200,
                bbox_inches="tight")
print(json.dumps(dict(
    adjoint=dict(n_pairs=e31["n_pairs"], med=e31["med"],
                 p95=e31["p95"], max=e31["max"],
                 speedup=naive_min / t_back),
    multicolor_6km=dict(n_solves=82, speedup=naive_min / mc6,
                        max_err=max(xt6)),
    warm_solve_s=warm), indent=2))
print(f"saved -> {FIG_DIR}/fig05_pipelines.pdf")
