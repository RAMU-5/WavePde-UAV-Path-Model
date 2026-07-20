"""Generate the uncertainty qualification figure from E7.1."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "runs" / "exp9_uncertainty"
OUT = ROOT / "paper" / "Q3" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads((RUN / "results.json").read_text())["runs"]
dat = np.load(RUN / "exp9_data.npz")

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(res)))

# (a) x-z paths over the same obstacle.
ax = axes[0]
ax.add_patch(plt.Rectangle((120, 0), 60, 60, color="0.75", ec="0.25"))
for rec, colour in zip(res, colours):
    tag = str(rec["D_diff_m2_s"]).replace(".", "p")
    p = dat[f"path_D{tag}"]
    ax.plot(p[:, 0], p[:, 2], color=colour, lw=1.5,
            label=f"D={rec['D_diff_m2_s']:g}")
ax.set(xlabel="x [m]", ylabel="z [m]", title="(a) diffusion-aware paths",
       xlim=(45, 255), ylim=(15, 100))
ax.legend(fontsize=7, ncol=2)

# (b) clearance against the local 95% uncertainty radius.
ax = axes[1]
for rec, colour in zip(res, colours):
    tag = str(rec["D_diff_m2_s"]).replace(".", "p")
    arc = dat[f"arc_D{tag}"]
    clearance = dat[f"clearance_D{tag}"]
    r95 = dat[f"r95_D{tag}"]
    progress = arc / max(float(arc[-1]), 1e-9)
    ax.plot(progress, clearance - r95, color=colour, lw=1.3)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set(xlabel="normalised path progress", ylabel="clearance − $r_{95}$ [m]",
       title="(b) 95% clearance margin")

# (c) MC collision probability with Wilson confidence intervals.
ax = axes[2]
D = np.array([r["D_diff_m2_s"] for r in res])
p = np.array([r["collision_probability"] for r in res])
lo = np.array([r["collision_ci_low"] for r in res])
hi = np.array([r["collision_ci_high"] for r in res])
lower_err = 100 * np.maximum(p - lo, 0.0)
upper_err = 100 * np.maximum(hi - p, 0.0)
ax.errorbar(D, 100 * p, yerr=[lower_err, upper_err],
            fmt="o-", capsize=3)
ax.set(xlabel="$D_{diff}$ [m$^2$/s]", ylabel="collision probability [%]",
       title="(c) Monte Carlo risk (95% CI)")
ax.grid(alpha=0.25)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig10_uncertainty.{ext}", dpi=220)
print(f"[fig10] -> {OUT}/fig10_uncertainty.png")
