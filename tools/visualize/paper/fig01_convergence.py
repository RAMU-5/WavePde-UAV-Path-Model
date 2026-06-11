"""Fig. 1 — free-space convergence (E1.1).

(a) log-log RMS relative residual vs dx, one curve per pulse width τ,
    fitted convergence order annotated.
(b) measured arrival bias vs pulse onset t0 — the bias is ≈ t0 and is
    subtracted ("calibrated out") in every later experiment.

Reads  output/runs/exp0_analytic/results.json
Writes paper/Q1/figs/fig01_convergence.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "output" / "runs" / "exp0_analytic" / "results.json"
OUT = ROOT / "paper" / "Q1" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads(RES.read_text())["e11"]
runs = res["runs"]
taus = sorted({r["tau_s"] for r in runs})

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

for tau in taus:
    pts = sorted([(r["dx"], r["rms_rel"]) for r in runs if r["tau_s"] == tau])
    order = res["convergence_order"].get(f"tau{tau:g}")
    lab = f"τ = {tau:g} s" + (f"  (order {order:.2f})" if order else "")
    ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", label=lab)
# 2nd-order guide
dx0 = [2.5, 20.0]
ref = [r["rms_rel"] for r in runs if r["tau_s"] == taus[-1] and r["dx"] == 2.5]
if ref:
    ax.loglog(dx0, [ref[0] * (d / 2.5) ** 2 for d in dx0], "k--", lw=0.8,
              label="O(dx²) guide")
ax.set_xlabel("grid spacing dx [m]")
ax.set_ylabel("RMS relative arrival-time residual")
ax.set_title("(a) free-space convergence, T* = r/c")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

b = [(r["t0_s"], r["bias_s"]) for r in runs]
ax2.plot([min(x for x, _ in b), max(x for x, _ in b)],
         [min(x for x, _ in b), max(x for x, _ in b)],
         "k--", lw=0.8, label="bias = t₀")
for tau in taus:
    pts = [(r["t0_s"], r["bias_s"]) for r in runs if r["tau_s"] == tau]
    ax2.plot([p[0] for p in pts], [p[1] for p in pts], "o",
             label=f"τ = {tau:g} s")
ax2.set_xlabel("pulse onset t₀ [s]")
ax2.set_ylabel("fitted arrival bias [s]")
ax2.set_title("(b) peak-detection bias ≈ t₀ (calibrated out)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig01_convergence.{ext}", dpi=200)
print(f"[fig01] -> {OUT}/fig01_convergence.png")
