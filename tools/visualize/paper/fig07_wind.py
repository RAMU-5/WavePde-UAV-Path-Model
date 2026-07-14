"""Fig. 7 — spatially varying wind: validity and the reciprocity limit
(E4.3, paper Q3).

(a) path drift: -grad T descent from one station pair, no wind vs the
    M=0.3 vortex field (quiver overlay) — paths bend downwind.
(b) wind-induced time change per pair, wave vs wind-aware Dijkstra:
    dT = T(wind) - T(0). Bias and resolution error cancel in the
    difference, isolating the wind physics.
(c) forward/backward reciprocity error vs the measured RMS divergence
    of the (masked) wind field — the adjoint-trick operating limit.

Reads  output/runs/exp4_wind/exp4_data.npz
Writes paper/Q3/figs/fig07_wind.{png,pdf} + JSON summary
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

LO_D, HI_D = 700.0, 3000.0


def band_pairs(st_xyz, src_ids):
    D = np.sqrt(((st_xyz[:, None, :2] - st_xyz[None, :, :2]) ** 2).sum(-1))
    return D


def main():
    d = np.load(ROOT / "output" / "runs" / "exp4_wind" / "exp4_data.npz",
                allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    cases = meta["cases"]
    src_ids = d["src_ids"]
    st_xyz = d["st_xyz"]
    D = band_pairs(st_xyz, src_ids)

    fig = plt.figure(figsize=(12.6, 4.2))
    axa = fig.add_subplot(1, 3, 1)
    axb = fig.add_subplot(1, 3, 2)
    axc = fig.add_subplot(1, 3, 3)

    # ---- (a) path drift map --------------------------------------------
    z = np.load(ROOT / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    from solve_coupled import trace_path

    T0 = d["Tfield_nowind_src50"]
    T2 = d["Tfield_vortex_M03_src50"]
    a50 = int(np.where(src_ids == 50)[0][0])
    src_m = st_xyz[50]
    # destination: band pair with the largest wave |dT| from source 50
    dTw = np.abs(d["Twave_vortex_M03"][a50] - d["Twave_nowind"][a50])
    cand = np.where((D[50] >= LO_D) & (D[50] <= HI_D)
                    & np.isfinite(dTw))[0]
    b = int(cand[np.argmax(dTw[cand])])
    dst_m = st_xyz[b]
    fp = mask.any(axis=2)
    axa.set_facecolor("#f4f1ec")
    axa.imshow(np.where(fp, 0.55, 0.0).T, origin="lower",
               extent=[0, nx * dx, 0, ny * dy], cmap="Greys",
               vmin=0, vmax=1, alpha=0.6)
    u2 = d["u_vortex45_z5"]; v2 = d["v_vortex45_z5"]
    step = max(1, nx // 20)
    X = (np.arange(nx) * dx + 0.5 * dx)[::step]
    Y = (np.arange(ny) * dy + 0.5 * dy)[::step]
    axa.quiver(X, Y, u2[::step, ::step].T, v2[::step, ::step].T,
               color="#7799bb", scale=120, width=0.0035, alpha=0.8)
    for T, col, lab, tdst in (
            (T0, "C0", "no wind", d["Twave_nowind"][a50, b]),
            (T2, "C3", "vortex M=0.3", d["Twave_vortex_M03"][a50, b])):
        p, _ = trace_path(np.asarray(T, dtype=np.float64), src_m, dst_m,
                          dx, dy, dz)
        axa.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.0,
                 label=f"{lab}: T = {tdst:.0f} s")
    axa.scatter(*src_m[:2], s=160, marker="*", facecolor="#ffe833",
                edgecolor="k", zorder=10)
    axa.scatter(*dst_m[:2], s=60, marker="v", facecolor="#ff5544",
                edgecolor="k", zorder=10)
    pad = 600.0
    lo_x = max(0, min(src_m[0], dst_m[0]) - pad)
    hi_x = min(nx * dx, max(src_m[0], dst_m[0]) + pad)
    lo_y = max(0, min(src_m[1], dst_m[1]) - pad)
    hi_y = min(ny * dy, max(src_m[1], dst_m[1]) + pad)
    axa.set_xlim(lo_x, hi_x); axa.set_ylim(lo_y, hi_y)
    axa.set_aspect("equal")
    axa.legend(fontsize=7, loc="upper left")
    axa.set_title(f"(a) path drift, stations 50$\\to${b}", fontsize=9)

    # ---- (b) dT scatter wave vs Dijkstra --------------------------------
    summary = {}
    for name, col in (("vortex_M01", "C0"), ("vortex_M03", "C3")):
        dT_w, dT_r = [], []
        for a, s in enumerate(src_ids):
            tw = d[f"Twave_{name}"][a] - d["Twave_nowind"][a]
            tr = d[f"Tstr_{name}"][a] - d["Tstr_nowind"][a]
            m = (np.isfinite(tw) & np.isfinite(tr)
                 & (D[s] >= LO_D) & (D[s] <= HI_D))
            dT_w.append(tw[m]); dT_r.append(tr[m])
        dT_w = np.concatenate(dT_w); dT_r = np.concatenate(dT_r)
        q = cases[name]["qmax"]
        lab = f"|q|$_{{max}}$={q:.1f} m/s (M={q/15:.1f})"
        axb.plot(dT_r, dT_w, ".", ms=4, alpha=0.5, color=col, label=lab)
        cc = float(np.corrcoef(dT_r, dT_w)[0, 1])
        resid = dT_w - dT_r
        summary[name] = dict(
            n=len(dT_w), corr=cc,
            med_dT_ref=float(np.median(dT_r)),
            resid_med_s=float(np.median(resid)),
            resid_p95_abs_s=float(np.percentile(np.abs(resid), 95)))
    lim = np.percentile(np.abs(np.concatenate([dT_r, dT_w])), 99.5)
    axb.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    axb.set_xlim(-lim, lim); axb.set_ylim(-lim, lim)
    axb.set_xlabel("Dijkstra wind-induced $\\Delta T$ [s]")
    axb.set_ylabel("wave wind-induced $\\Delta T$ [s]")
    axb.legend(fontsize=7.5, loc="upper left")
    axb.set_title("(b) per-pair $\\Delta T$ = T(wind) $-$ T(0)", fontsize=9)

    # ---- (c) reciprocity vs divergence -----------------------------------
    fwd_ids, bwd_ids = d["fwd_ids"], d["bwd_ids"]
    pts = []
    for name in cases:
        if f"Tfwd_{name}" not in d:
            continue
        Tf, Tb = d[f"Tfwd_{name}"], d[f"Tbwd_{name}"]
        errs = []
        for a, i in enumerate(fwd_ids):
            for c, j in enumerate(bwd_ids):
                t_ij = Tf[a, j]            # forward from i, read at j
                t_ji = Tb[c, i]            # backward from j, read at i
                if not (np.isfinite(t_ij) and np.isfinite(t_ji)):
                    continue
                if not (LO_D <= D[i, j] <= HI_D):
                    continue
                errs.append(abs(t_ij - t_ji) / (0.5 * (t_ij + t_ji)))
        errs = np.asarray(errs)
        pts.append(dict(name=name, div_rms=cases[name]["div_rms"],
                        qmax=cases[name]["qmax"], n=len(errs),
                        med=float(np.median(errs)),
                        p95=float(np.percentile(errs, 95))))
    pts.sort(key=lambda p: p["div_rms"])
    xv = [max(p["div_rms"], 1e-6) for p in pts]
    axc.plot(xv, [100 * p["med"] for p in pts], "o-", color="C0",
             label="median")
    axc.plot(xv, [100 * p["p95"] for p in pts], "s--", color="C3",
             label="p95")
    for p, x in zip(pts, xv):
        axc.annotate(p["name"].replace("blend_b", "$\\beta$=")
                     .replace("uniform_M02", "uniform"),
                     xy=(x, 100 * p["med"]), fontsize=6.5,
                     xytext=(0, -11), textcoords="offset points",
                     ha="center", color="0.3")
    axc.set_xscale("log")
    axc.set_xlabel("RMS $\\nabla\\!\\cdot q$ of the masked wind field [1/s]")
    axc.set_ylabel("fwd/bwd reciprocity error [%]")
    axc.legend(fontsize=7.5)
    axc.set_title("(c) adjoint reciprocity vs divergence", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig07_wind.{ext}", dpi=200)
    print(f"[fig07] -> {OUT}/fig07_wind.png")
    print(json.dumps(dict(dT=summary, reciprocity=pts, cases=cases),
                     indent=2))


if __name__ == "__main__":
    main()
