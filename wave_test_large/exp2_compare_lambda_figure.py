"""Make a side-by-side comparison figure: short λ vs long λ for the
aircraft scenario (RAW trace, no path smoothing).

Layout (2 rows × 2 cols):
   row 1: top-down xy paths   (short | long)
   row 2: side view z vs dist (short | long)

Output: visualization/3d_aircraft/lambda_comparison.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path("/leonardo_scratch/large/userexternal/yli00001/Aura_Project/"
            "Levistone/visualization/3d_aircraft")
SHORT_NPZ = ROOT / "aircraft_raw_short_lambda.npz"
LONG_NPZ = ROOT / "aircraft_raw_long_lambda.npz"
OUT_PNG = ROOT / "lambda_comparison.png"

CRUISE_KM = 8.0
AIRPORT_KM = (8.0, 8.0)


def load(npz):
    z = np.load(npz)
    return {
        "n": int(z["n_paths"]),
        "paths": [z[f"path_{i}_xyz"]/1000 for i in range(int(z["n_paths"]))],
        "dsts":  [z[f"dst_{i}_xyz"]/1000  for i in range(int(z["n_paths"]))],
        "clouds_centre": z["clouds_centre"]/1000,
        "clouds_sxy": z["clouds_sigma_xy"]/1000,
        "clouds_sz":  z["clouds_sigma_z"]/1000,
    }


def k_at_5km(p):
    """Median κ@p95 (1/km) for phase 2 only, resampled at 5 km."""
    p2 = p[250:] if p.shape[0] > 260 else p
    if p2.shape[0] < 10:
        return 0.0
    seg = np.diff(p2, axis=0)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    if arc[-1] < 5:
        return 0.0
    s = np.arange(0, arc[-1], 5)
    pr = np.column_stack([np.interp(s, arc, p2[:, k]) for k in range(3)])
    seg2 = np.diff(pr[:, :2], axis=0)
    seg_len = np.maximum(np.linalg.norm(seg2, axis=1), 1e-6)
    tan = seg2 / seg_len[:, None]
    dt = np.diff(tan, axis=0)
    if dt.size == 0:
        return 0.0
    dn = np.linalg.norm(dt, axis=1)
    midlen = 0.5 * (seg_len[:-1] + seg_len[1:])
    return float((dn / midlen).max())


def main():
    short = load(SHORT_NPZ)
    long_ = load(LONG_NPZ)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    cmap = cm.get_cmap("turbo")

    for col, (D, title) in enumerate([
        (short, "RAW SHORT λ (τ=6, σ_xy=4 cells = 1.6 km)"),
        (long_, "RAW LONG λ (τ=20, σ_xy=10 cells = 4 km)"),
    ]):
        # ---- Top-down (xy) -------------------------------------------
        ax = axes[0, col]
        # Cloud rings (1σ solid, 2σ dashed)
        for c, sxy in zip(D["clouds_centre"], D["clouds_sxy"]):
            theta = np.linspace(0, 2*np.pi, 80)
            ax.plot(c[0]+sxy*np.cos(theta), c[1]+sxy*np.sin(theta),
                    color="#7028b0", lw=1.6, alpha=0.6)
            ax.plot(c[0]+2*sxy*np.cos(theta), c[1]+2*sxy*np.sin(theta),
                    color="#7028b0", lw=1.0, ls="--", alpha=0.5)
            ax.scatter(c[0], c[1], color="#7028b0", s=70,
                       edgecolor="black", linewidth=0.5, zorder=5)
        # Paths
        kappa_vals = []
        for j, p in enumerate(D["paths"]):
            col_ = cmap(0.15 + 0.7 * j / max(1, D["n"]-1))
            ax.plot(p[:, 0], p[:, 1], color=col_, lw=1.6, alpha=0.9)
            ax.scatter(D["dsts"][j][0], D["dsts"][j][1], color=col_,
                       s=70, marker="v", edgecolor="black", linewidth=0.4,
                       zorder=10)
            kappa_vals.append(k_at_5km(p))
        # Airport
        ax.scatter(*AIRPORT_KM, color="#ffd400", s=200, marker="*",
                   edgecolor="black", linewidth=1.0, zorder=15)
        ax.plot([AIRPORT_KM[0], AIRPORT_KM[0]+5], [AIRPORT_KM[1], AIRPORT_KM[1]],
                color="#222", lw=3, alpha=0.7)
        ax.set_xlim(0, 80); ax.set_ylim(0, 80)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        kmed = np.median(kappa_vals)
        ax.set_title(f"{title}\nTop-down — κ@5km median = {kmed:.3f} 1/km, "
                     f"R_min ≈ {1/max(kmed,1e-9):.1f} km", fontsize=11)
        ax.grid(True, alpha=0.3)

        # ---- Side view (z) -------------------------------------------
        ax = axes[1, col]
        # Cloud z-extent (vertical lines at horizontal distance from airport)
        from matplotlib.patches import Ellipse
        for c, sxy, sz in zip(D["clouds_centre"], D["clouds_sxy"],
                              D["clouds_sz"]):
            d = np.hypot(c[0]-AIRPORT_KM[0], c[1]-AIRPORT_KM[1])
            ax.add_patch(Ellipse((d, c[2]), 2*sxy, 2*sz,
                                 color="#a45ad9", alpha=0.20, zorder=3))
            ax.scatter(d, c[2], color="#7028b0", s=60, edgecolor="black",
                       linewidth=0.4, zorder=5)
        for j, p in enumerate(D["paths"]):
            col_ = cmap(0.15 + 0.7 * j / max(1, D["n"]-1))
            d = np.linalg.norm(p[:, :2] - np.array(AIRPORT_KM), axis=1)
            ax.plot(d, p[:, 2], color=col_, lw=1.6, alpha=0.9)
            ax.scatter(d[-1], p[-1, 2], color=col_, s=70, marker="v",
                       edgecolor="black", linewidth=0.4, zorder=10)
        ax.scatter(0, 0, color="#ffd400", s=200, marker="*",
                   edgecolor="black", linewidth=1.0, zorder=15)
        ax.axhline(CRUISE_KM, color="#888", lw=0.8, ls=":")
        ax.set_xlim(0, 110); ax.set_ylim(0, 12)
        ax.set_xlabel("ground distance from airport (km)")
        ax.set_ylabel("altitude z (km)")
        # max |dz/step| for a quick "smoothness" indicator
        dz_med = np.median([
            np.abs(np.diff(p[250:, 2] if p.shape[0] > 260 else p[:, 2])).max()
            for p in D["paths"]
        ])
        ax.set_title(f"Side view — z max-step jump median = {dz_med*1000:.0f} m",
                     fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Pure wavelength effect — RAW gradient-descent trace, "
        "no path smoothing, no descent ramp",
        fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
