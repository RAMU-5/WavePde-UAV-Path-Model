"""Fig. 6 — wavelength operating point (E4.1, paper Q3).

Optimality gap vs the E2.1 discrete ground truth as a function of the
source pulse width tau, i.e. of the dominant wavelength
lambda ~= 2 pi c0 tau. Identifies the operating band and motivates
broadband (two-band combination overlaid as a star).

x-axis is lambda; the top axis shows lambda / W_street where W_street
is the median free street width of the 5 m domain at cruise height
(measured from the building mask along x and y).

Reads  output/runs/exp3_wavelength/exp3_data.npz
       output/runs/exp1_dijkstra/gt_domain_5m.npz
Writes paper/Q3/figs/fig06_wavelength.{png,pdf} + JSON summary
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "Q3" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 15.0
LO_D, HI_D = 700.0, 3000.0
STREET_Z_M = 50.0          # cruise height for the street-width statistic


def street_width_m():
    z = np.load(ROOT / "input" / "domain_5m.npz", allow_pickle=False)
    mask = z["mask"]
    dxm = float(z["dx"])
    k = int(round(STREET_Z_M / float(z["dz"])))
    sl = mask[:, :, k]
    widths = []
    for axis in (0, 1):
        m = sl if axis == 0 else sl.T
        for row in m:
            # free runs bounded by buildings on BOTH sides (street-like)
            d = np.diff(np.concatenate([[1], row.astype(np.int8), [1]]))
            starts = np.flatnonzero(d == -1)
            ends = np.flatnonzero(d == 1)
            for s, e in zip(starts, ends):
                if s > 0 and e < len(row):       # interior runs only
                    widths.append((e - s) * dxm)
    return float(np.median(widths)), float(np.mean(mask[:, :, k]))


def main():
    d = np.load(ROOT / "output" / "runs" / "exp3_wavelength" / "exp3_data.npz")
    gt = np.load(ROOT / "output" / "runs" / "exp1_dijkstra" / "gt_domain_5m.npz")
    taus = d["taus"]
    biases = d["biases"]
    T_at = d["T_at"]                        # (n_tau, 12, 118)
    src_idx = d["src_idx"]
    assert list(src_idx) == list(gt["src_idx"])

    st = d["stations_xy_m"]
    D = np.sqrt(((st[:, None, :2] - st[None, :, :2]) ** 2).sum(-1))

    w_street, solid_frac = street_width_m()
    lambdas = 2.0 * np.pi * C0 * taus

    stats, gaps_per_tau = [], []
    for a, tau in enumerate(taus):
        rows = []
        for b, s in enumerate(src_idx):
            tw = T_at[a, b] - biases[a]
            ts = gt["T_str"][b]
            m = (np.isfinite(tw) & np.isfinite(ts)
                 & (D[s] >= LO_D) & (D[s] <= HI_D))
            g = (tw[m] - ts[m]) / ts[m]
            rows.append(g)
        g = np.concatenate(rows)
        gaps_per_tau.append(g)
        stats.append(dict(
            tau_s=float(tau), lambda_m=float(2 * np.pi * C0 * tau),
            n=len(g), med=float(np.median(g)),
            p25=float(np.percentile(g, 25)), p75=float(np.percentile(g, 75)),
            p95_abs=float(np.percentile(np.abs(g), 95)),
            finite_frac=float(np.isfinite(
                T_at[a]).sum() / T_at[a].size)))

    # broadband proxy: per-pair earliest calibrated arrival of the two
    # outer bands (0.2 s and 1.6 s) — what a two-band solve would detect
    ia, ib = int(np.argmin(np.abs(taus - 0.2))), int(np.argmin(np.abs(taus - 1.6)))
    rows = []
    for b, s in enumerate(src_idx):
        twa = T_at[ia, b] - biases[ia]
        twb = T_at[ib, b] - biases[ib]
        tw = np.fmin(twa, twb)              # fmin: ignore NaN/inf one-sided
        tw = np.where(np.isfinite(tw), tw, np.inf)
        ts = gt["T_str"][b]
        m = (np.isfinite(tw) & np.isfinite(ts)
             & (D[s] >= LO_D) & (D[s] <= HI_D))
        rows.append((tw[m] - ts[m]) / ts[m])
    g_bb = np.concatenate(rows)
    bb = dict(med=float(np.median(g_bb)),
              p95_abs=float(np.percentile(np.abs(g_bb), 95)), n=len(g_bb))

    # ---- plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    med = np.array([s["med"] for s in stats])
    p25 = np.array([s["p25"] for s in stats])
    p75 = np.array([s["p75"] for s in stats])
    p95 = np.array([s["p95_abs"] for s in stats])

    ax.fill_between(lambdas, 100 * p25, 100 * p75, alpha=0.25, color="C0",
                    label="IQR of gap")
    ax.plot(lambdas, 100 * med, "o-", color="C0", label="median gap")
    ax.plot(lambdas, 100 * p95, "s--", color="C3", label="p95 |gap|")
    ax.axhline(0, color="k", lw=0.7)

    ip = int(np.argmin(np.abs(taus - 0.4)))
    ax.axvline(lambdas[ip], color="0.5", lw=0.8, ls=":")
    ax.annotate("production\n$\\tau$=0.4 s", xy=(lambdas[ip], ax.get_ylim()[1]),
                xytext=(lambdas[ip] * 1.1, 100 * p95.max() * 0.85),
                fontsize=8, color="0.35")
    lam_bb = np.sqrt(lambdas[ia] * lambdas[ib])
    ax.plot([lam_bb], [100 * bb["med"]], "*", ms=15, color="C2",
            label=f"two-band min (broadband proxy): "
                  f"med {100*bb['med']:+.1f}%, p95 {100*bb['p95_abs']:.1f}%")

    ax.set_xscale("log")
    ax.set_ylim(ax.get_ylim()[0], 100 * p95.max() + 5.5)   # room for legend
    ax.set_xlabel("dominant wavelength  $\\lambda \\approx 2\\pi c_0 \\tau$  [m]")
    ax.set_ylabel("optimality gap vs discrete ground truth  [%]")
    sec = ax.secondary_xaxis(
        "top", functions=(lambda x: x / w_street, lambda r: r * w_street))
    sec.set_xlabel(f"$\\lambda$ / median street width ({w_street:.0f} m)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(Fig. 6) wavelength operating point, 5 m Nanshan domain",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig06_wavelength.{ext}", dpi=200)
    print(f"[fig06] -> {OUT}/fig06_wavelength.png")

    print(json.dumps(dict(per_tau=stats, broadband_proxy=bb,
                          street_width_m=w_street,
                          solid_frac_at_z=solid_frac), indent=2))


if __name__ == "__main__":
    main()
