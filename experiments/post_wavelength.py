"""Post-process the wavelength sweep npz: recompute κ_max / R_min with
the dedup-aware curvature stat, replot."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from experiments.test_wavelength import path_curvature_stats, _simplify_path

_THIS = Path(__file__).resolve().parent
DOMAIN = _THIS.parent / "input" / "domain.npz"
SWEEP_NPZ = (_THIS.parent / "output" / "figures" / "wavelength_test"
             / "wavelength_sweep_src048.npz")
OUT_PNG = SWEEP_NPZ.with_suffix(".png")

LABELS = ["short", "medium", "long"]
TAU_SIGMA = {"short": (1, 0.5), "medium": (4, 1.5), "long": (20, 6.0)}


def main():
    z = np.load(SWEEP_NPZ)
    src_idx = int(z["source_idx"])
    src_xyz = z["source_xyz_m"]
    dst_indices = z["dst_indices"]

    d = np.load(DOMAIN, allow_pickle=False)
    mask_np = d["mask"]
    dx, dy, dz = float(d["dx"]), float(d["dy"]), float(d["dz"])
    nx, ny, nz = mask_np.shape
    Lx, Ly = nx * dx, ny * dy
    stations_xy_m = d["stations_xy_m"]

    results = {}
    for label in LABELS:
        offsets = z[f"{label}_path_offsets"]
        paths_concat = z[f"{label}_paths_concat"]
        plengths = z[f"{label}_path_lengths"]
        T_dst = z[f"{label}_T_dst"]
        paths = []
        kappas = []
        for k in range(len(plengths)):
            a, b = offsets[k], offsets[k + 1]
            p = paths_concat[a:b]
            paths.append(p)
            kappas.append(path_curvature_stats(p))
        results[label] = dict(paths=paths, kappas=kappas, T_dst=T_dst)
        good_k = [k for k, p in zip(kappas, paths) if p.shape[0] >= 5]
        all_kmax = [k[1] for k in good_k if np.isfinite(k[1])]
        all_rmin = [k[2] for k in good_k if np.isfinite(k[2])]
        kmed = np.median(all_kmax) if all_kmax else 0.0
        rmed = np.median(all_rmin) if all_rmin else float("inf")
        kmean = np.mean([k[0] for k in good_k]) if good_k else 0.0
        # also compute simplified path total length
        simp_len = []
        for p in paths:
            if p.shape[0] < 2:
                continue
            sp = _simplify_path(p[:, :2])
            if sp.shape[0] < 2:
                continue
            seg = np.linalg.norm(np.diff(sp, axis=0), axis=1)
            simp_len.append(seg.sum())
        print(f"{label:6s}: τ={TAU_SIGMA[label][0]:>2}, σ={TAU_SIGMA[label][1]:>4.1f}  "
              f"κ_mean={kmean:.5f}  κ_max med={kmed:.5f}  "
              f"R_min med={rmed:7.1f} m  "
              f"path-len med={np.median(simp_len):.0f} m  "
              f"T_dst med={np.nanmedian(T_dst):.0f} s")

    # ---- Plot ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             sharex=True, sharey=True)
    fp = mask_np.any(axis=2)
    for ax_, label in zip(axes, LABELS):
        r = results[label]
        ax_.set_facecolor("#f4f1ec")
        ax_.imshow(np.where(fp, 0.55, 0.0).T, origin="lower",
                   extent=[0, Lx, 0, Ly], cmap="Greys",
                   vmin=0, vmax=1, alpha=0.6, zorder=1)
        ax_.scatter(stations_xy_m[:, 0], stations_xy_m[:, 1],
                    s=8, c="white", edgecolor="black", linewidths=0.3,
                    zorder=4)
        # Use simplified paths for plotting (drops zero-length segments).
        segs = []
        tcol = []
        for k, path in enumerate(r["paths"]):
            if path.shape[0] < 2:
                continue
            sp = _simplify_path(path[:, :2])
            if sp.shape[0] < 2:
                continue
            for s in range(sp.shape[0] - 1):
                segs.append([sp[s], sp[s + 1]])
                tcol.append(r["T_dst"][k])
        if segs:
            tcol = np.asarray(tcol)
            vmin = float(np.nanmin(tcol))
            vmax = float(np.nanpercentile(tcol, 95))
            lc = LineCollection(segs, cmap="plasma_r",
                                norm=plt.Normalize(vmin, max(vmax, vmin + 1)),
                                linewidths=1.4, alpha=0.9, zorder=6)
            lc.set_array(tcol)
            ax_.add_collection(lc)
        ax_.scatter(src_xyz[0], src_xyz[1], s=240, marker="*",
                    facecolor="#ffe833", edgecolor="black", linewidths=1.0,
                    zorder=12)
        d_xy = stations_xy_m[dst_indices, :2]
        ax_.scatter(d_xy[:, 0], d_xy[:, 1], s=40, marker="v",
                    facecolor="#ff5544", edgecolor="black", linewidths=0.4,
                    alpha=0.8, zorder=10)
        good_k = [k for k, p in zip(r["kappas"], r["paths"]) if p.shape[0] >= 5]
        all_kmax = [k[1] for k in good_k if np.isfinite(k[1])]
        all_rmin = [k[2] for k in good_k if np.isfinite(k[2])]
        kmed = np.median(all_kmax) if all_kmax else 0.0
        rmed = np.median(all_rmin) if all_rmin else float("inf")
        tau, sigma = TAU_SIGMA[label]
        ax_.set_title(
            f"{label}: τ={tau} steps, σ={sigma} cells\n"
            f"κ_max med = {kmed:.4f} 1/m,  R_min med = {rmed:.0f} m",
            fontsize=11)
        ax_.set_xlim(0, Lx)
        ax_.set_ylim(0, Ly)
        ax_.set_aspect("equal")
        ax_.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Wavelength sweep — paths from station {src_idx} "
        f"({len(dst_indices)} destinations, colour = T(dst))",
        fontsize=13)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
