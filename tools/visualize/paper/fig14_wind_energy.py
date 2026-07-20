"""Plot the wind-exploitation benefit experiment (E7 / exp15)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "runs" / "exp15_energy"
OUT = ROOT / "paper" / "Q3" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((RUN / "results.json").read_text())
data = np.load(RUN / "exp15_data.npz")
machs = sorted(float(m) for m in meta["machs"])

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55))

# (a) Example pair at the strongest wind: naive vs wind-aware route over
# the vortex field (z = regulatory-band top slice) and building footprint.
ax = axes[0]
dx, dy = float(data["dx"]), float(data["dy"])
mask = data["mask_z5"]
u, v = data["u_z5"], data["v_z5"]
nx, ny = mask.shape
x = (np.arange(nx) + 0.5) * dx / 1000.0
y = (np.arange(ny) + 0.5) * dy / 1000.0
speed = np.hypot(u, v)
ax.streamplot(x, y, u.T, v.T, color=speed.T, cmap="viridis",
              density=1.15, linewidth=0.7, arrowsize=0.7)
footprint = np.ma.masked_where(~mask.T, np.ones_like(mask.T, dtype=float))
ax.pcolormesh(x, y, footprint, cmap="Greys", vmin=0.0, vmax=1.6,
              shading="nearest")
t_naive = float(data["example_t_naive"])
t_aware = float(data["example_t_aware"])
p_naive = data["example_path_naive"]
p_aware = data["example_path_aware"]
ax.plot(p_naive[:, 0] / 1000, p_naive[:, 1] / 1000, color="tab:red",
        lw=1.8, label=f"wind-ignoring: {t_naive:.0f} s")
ax.plot(p_aware[:, 0] / 1000, p_aware[:, 1] / 1000, color="tab:blue",
        lw=1.8, label=f"wind-aware: {t_aware:.0f} s")
stations = data["stations_xyz"]
for idx, marker in ((int(data["example_src"]), "o"),
                    (int(data["example_dst"]), "*")):
    ax.plot(stations[idx, 0] / 1000, stations[idx, 1] / 1000, marker,
            color="black", ms=8 if marker == "*" else 5, zorder=5)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set(xlabel="x [km]", ylabel="y [km]",
       title=f"(a) Example pair, $M=0.3$: {t_naive / t_aware:.1f}$\\times$ faster",
       aspect="equal", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))

# (b) Fraction of station pairs whose planned route is infeasible when
# flown under the true wind (regulatory band makes corridors one-way).
ax = axes[1]
attempted = [meta["machs"][f"{m:g}"] for m in machs]
totals = np.asarray([r["n_pairs"] + r["n_naive_infeasible"]
                     + r["n_aware_infeasible"] for r in attempted],
                    dtype=float)
naive_pct = 100.0 * np.asarray(
    [r["n_naive_infeasible"] for r in attempted]) / totals
aware_pct = 100.0 * np.asarray(
    [r["n_aware_infeasible"] for r in attempted]) / totals
pos = np.arange(len(machs))
width = 0.36
ax.bar(pos - width / 2, naive_pct, width=width, color="tab:red",
       alpha=0.8, label="wind-ignoring plan")
ax.bar(pos + width / 2, aware_pct, width=width, color="tab:blue",
       alpha=0.8, label="wind-aware plan")
for p, val in zip(pos, naive_pct):
    if val > 0:
        ax.text(p - width / 2, val + 1.2, f"{val:.0f}%", ha="center",
                fontsize=8)
ax.set(xticks=pos, xticklabels=[f"{m:g}" for m in machs],
       xlabel="wind strength  $M = |q|_{\\max}/c_0$",
       ylabel="infeasible plans [%]",
       title="(b) Feasibility under the true wind",
       ylim=(0, 100))
ax.legend(loc="upper left", fontsize=8)

# (c) Distribution of the per-pair time/energy saving of wind-aware over
# wind-ignoring planning, mutually feasible pairs only.
ax = axes[2]
savings = []
for m in machs:
    pairs = data[f"pairs_M{f'{m:g}'.replace('.', '')}"]
    savings.append(100.0 * (pairs[:, 3] - pairs[:, 4]) / pairs[:, 3])
box = ax.boxplot(savings, positions=pos, widths=0.5, whis=(5, 95),
                 showfliers=False, medianprops={"color": "tab:orange"})
ax.axhline(0.0, color="black", lw=0.7)
top = max(np.percentile(s, 95) for s in savings)
bottom = min(np.percentile(s, 5) for s in savings)
ax.set_ylim(bottom - 4.0, top + 9.0)
for p, s in zip(pos, savings):
    ax.text(p, np.percentile(s, 95) + 2.0,
            f"med {np.median(s):.1f}%", ha="center", fontsize=8)
ax.set(xticks=pos, xticklabels=[f"{m:g}" for m in machs],
       xlabel="wind strength  $M = |q|_{\\max}/c_0$",
       ylabel="time/energy saving [%]",
       title="(c) Saving on mutually feasible pairs")

fig.tight_layout()
for suffix in ("pdf", "png"):
    fig.savefig(OUT / f"fig14_wind_energy.{suffix}", dpi=300,
                bbox_inches="tight")
print(f"[done] {OUT / 'fig14_wind_energy.pdf'}")
