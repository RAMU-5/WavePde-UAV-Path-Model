"""Pick the 3 destinations whose 'long' path differs MOST from 'short',
overlay all 3 wavelengths for each on the same panel.

Also recompute curvature with arc-length resampling so the metric
reflects path smoothness rather than grid-step jitter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_wavelength import _simplify_path

_THIS = Path(__file__).resolve().parent
DOMAIN = _THIS / "domain.npz"
SWEEP_NPZ = (_THIS.parent / "visualization" / "wavelength_test"
             / "wavelength_sweep_src048.npz")
OUT_PNG = (_THIS.parent / "visualization" / "wavelength_test"
           / "wavelength_overlay_src048.png")

LABELS = ["short", "medium", "long"]
TAU_SIGMA = {"short": (1, 0.5), "medium": (4, 1.5), "long": (20, 6.0)}
COLOURS = {"short": "#e23b3b", "medium": "#3380ff", "long": "#1aa04a"}


def resample_arclength(path_xy, ds=30.0):
    """Resample a 2D polyline at uniform arc-length spacing ds metres."""
    if path_xy.shape[0] < 2:
        return path_xy
    seg = np.diff(path_xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    if arc[-1] < ds:
        return path_xy
    s_new = np.arange(0.0, arc[-1], ds)
    if s_new[-1] < arc[-1] - 0.5 * ds:
        s_new = np.concatenate([s_new, [arc[-1]]])
    x = np.interp(s_new, arc, path_xy[:, 0])
    y = np.interp(s_new, arc, path_xy[:, 1])
    return np.column_stack([x, y])


def smoothed_curvature(path_xy, ds=30.0):
    """κ at uniform arc-length sampling (1/m). Less noisy than per-grid-step κ."""
    p = resample_arclength(path_xy, ds=ds)
    if p.shape[0] < 3:
        return np.array([])
    d1 = np.diff(p, axis=0)
    seg_len = np.linalg.norm(d1, axis=1)
    seg_len = np.maximum(seg_len, 1e-6)
    tangent = d1 / seg_len[:, None]
    dt = np.diff(tangent, axis=0)
    dn = np.linalg.norm(dt, axis=1)
    midlen = 0.5 * (seg_len[:-1] + seg_len[1:])
    return dn / midlen


def path_length(p):
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


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

    # Load paths per label (apply the simplify dedup)
    per_label = {}
    for label in LABELS:
        offsets = z[f"{label}_path_offsets"]
        plen = z[f"{label}_path_lengths"]
        paths_concat = z[f"{label}_paths_concat"]
        T_dst = z[f"{label}_T_dst"]
        paths = []
        for k in range(len(plen)):
            a, b = offsets[k], offsets[k + 1]
            p = paths_concat[a:b, :2]
            paths.append(_simplify_path(p))
        per_label[label] = dict(paths=paths, T_dst=T_dst)

    # Per-destination: divergence between short and long paths (Hausdorff)
    n_dst = len(dst_indices)
    div = np.zeros(n_dst)
    for k in range(n_dst):
        s = per_label["short"]["paths"][k]
        l = per_label["long"]["paths"][k]
        if s.shape[0] < 2 or l.shape[0] < 2:
            div[k] = 0.0
            continue
        # symmetric mean min-dist (cheaper than Hausdorff but proportional)
        from scipy.spatial.distance import cdist
        D = cdist(s, l)
        div[k] = 0.5 * (D.min(axis=0).mean() + D.min(axis=1).mean())

    pick = np.argsort(div)[::-1][:3]
    print(f"top-3 most-divergent destinations: idx in dst list = {pick}")
    for k in pick:
        print(f"  dst {dst_indices[k]}: divergence = {div[k]:.0f} m")

    # ---- Aggregate smoothed curvature per label (over ALL paths) -----
    print("\n=== smoothed curvature (arc-length sampling at 30m) ===")
    summary_lines = []
    for label in LABELS:
        all_k = []
        all_pl = []
        for p in per_label[label]["paths"]:
            ks = smoothed_curvature(p, ds=30.0)
            if ks.size:
                all_k.append(ks)
            all_pl.append(path_length(p))
        if all_k:
            ks = np.concatenate(all_k)
            kmean = float(ks.mean())
            kp95 = float(np.percentile(ks, 95))
            kmax = float(ks.max())
            r_at_p95 = 1.0 / kp95 if kp95 > 0 else float("inf")
        else:
            kmean = kp95 = kmax = 0.0
            r_at_p95 = float("inf")
        T_med = np.nanmedian(per_label[label]["T_dst"])
        pl_med = np.median(all_pl)
        line = (f"{label:6s}  τ={TAU_SIGMA[label][0]:>2}  "
                f"σ={TAU_SIGMA[label][1]:>4.1f}  "
                f"κ_mean={kmean:.5f}  κ_95%={kp95:.4f}  κ_max={kmax:.4f}  "
                f"R_at95={r_at_p95:6.0f} m  "
                f"path-len med={pl_med:.0f} m  T_dst med={T_med:.0f} s")
        print(line)
        summary_lines.append(line)

    # ---- Plot overlay: 3 destinations, 3 wavelengths each ------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             sharex=False, sharey=False)
    fp = mask_np.any(axis=2)
    for ax_, k in zip(axes, pick):
        dst_id = int(dst_indices[k])
        dst_xy = stations_xy_m[dst_id, :2]
        # Crop to bbox around src + dst with 600m padding
        xs = [src_xyz[0], dst_xy[0]]
        ys = [src_xyz[1], dst_xy[1]]
        xmin = max(0, min(xs) - 600); xmax = min(Lx, max(xs) + 600)
        ymin = max(0, min(ys) - 600); ymax = min(Ly, max(ys) + 600)
        ax_.set_facecolor("#f4f1ec")
        ax_.imshow(np.where(fp, 0.55, 0.0).T, origin="lower",
                   extent=[0, Lx, 0, Ly], cmap="Greys",
                   vmin=0, vmax=1, alpha=0.6, zorder=1)
        for label in LABELS:
            p = per_label[label]["paths"][k]
            ax_.plot(p[:, 0], p[:, 1], color=COLOURS[label],
                     lw=2.0, alpha=0.95,
                     label=(f"{label} (τ={TAU_SIGMA[label][0]}, "
                            f"σ={TAU_SIGMA[label][1]}, "
                            f"T={per_label[label]['T_dst'][k]:.0f}s)"),
                     zorder=8)
        # Source / dest markers
        ax_.scatter(src_xyz[0], src_xyz[1], s=240, marker="*",
                    facecolor="#ffe833", edgecolor="black", linewidths=1.0,
                    zorder=12)
        ax_.scatter(dst_xy[0], dst_xy[1], s=80, marker="v",
                    facecolor="#ff5544", edgecolor="black", linewidths=0.6,
                    zorder=11)
        ax_.set_xlim(xmin, xmax)
        ax_.set_ylim(ymin, ymax)
        ax_.set_aspect("equal")
        ax_.set_xlabel("x (m)")
        ax_.set_title(f"src {src_idx} → dst {dst_id}\n"
                      f"divergence(short,long) = {div[k]:.0f} m",
                      fontsize=11)
        ax_.legend(loc="best", fontsize=8.5)
        ax_.grid(True, alpha=0.3)
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        "Wavelength sweep — overlay of short / medium / long paths "
        "for 3 most-divergent destinations",
        fontsize=13)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
