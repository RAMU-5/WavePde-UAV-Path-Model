"""Re-plot the routine_sk_wind result with clearer paths and a 3D view."""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def render(npz_path, out_dir, vis_z=12):
    d = np.load(npz_path)
    mask = d["mask"]
    dx, dy, dz = float(d["dx"]), float(d["dy"]), float(d["dz"])
    u, v = d["u"], d["v"]
    T0, T1 = d["T0"], d["T1"]
    p0, p1 = d["path_nowind"], d["path_wind"]
    src, dst = d["src_xyz_m"], d["dst_xyz_m"]
    t_label = int(d["source_t"])
    nx, ny, nz = mask.shape
    Lx, Ly = nx * dx, ny * dy
    # exclude z=0 (which is the artificial solid ground floor in the source geometry);
    # only count voxels above the ground as "building"
    fp = mask[:, :, 1:].any(axis=2)

    # ---------- Top-down 2-panel ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax_, T, path, title, overlay_wind in (
        (axes[0], T0, p0, "no wind", False),
        (axes[1], T1, p1, f"with wind  t={t_label}", True),
    ):
        ax_.set_facecolor("#ece7df")
        # buildings
        ax_.imshow(np.where(fp.T, 1.0, 0.0),
                   origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="Greys", vmin=0, vmax=1, alpha=0.85, zorder=1)
        # T contours (lines, not filled — keeps path readable)
        Tslice = T[:, :, vis_z]
        finite = np.isfinite(Tslice)
        if finite.any():
            Tplot = np.where(finite, Tslice, np.nan)
            X = np.arange(nx) * dx + 0.5 * dx
            Y = np.arange(ny) * dy + 0.5 * dy
            cs = ax_.contour(X, Y, Tplot.T, levels=15, cmap="viridis", linewidths=0.8, alpha=0.7, zorder=2)
            cb = plt.colorbar(cs, ax=ax_, fraction=0.046, pad=0.02)
            cb.set_label("T (s)")
        # wind quiver
        if overlay_wind:
            step = max(1, nx // 24)
            Xq = (np.arange(nx) * dx + 0.5 * dx)[::step]
            Yq = (np.arange(ny) * dy + 0.5 * dy)[::step]
            U = u[::step, ::step, vis_z]
            V = v[::step, ::step, vis_z]
            ax_.quiver(Xq, Yq, U.T, V.T, color="#1a73e8", scale=40, width=0.0035,
                       headwidth=4, alpha=0.85, zorder=3)
        # path with high contrast: thick white halo + red core
        if path is not None and len(path) > 1:
            ax_.plot(path[:, 0], path[:, 1], "-", color="white", lw=5.0, zorder=4, solid_capstyle="round")
            ax_.plot(path[:, 0], path[:, 1], "-", color="red", lw=2.5, zorder=5, solid_capstyle="round")
        # markers
        ax_.scatter([src[0]], [src[1]], s=180, c="lime", edgecolor="black", linewidths=1.5,
                    zorder=7, label=f"src ({src[0]:.0f},{src[1]:.0f})")
        ax_.scatter([dst[0]], [dst[1]], s=180, c="red", edgecolor="black", linewidths=1.5,
                    zorder=7, label=f"dst ({dst[0]:.0f},{dst[1]:.0f})")
        T_dst = T[int(round(dst[0]/dx)), int(round(dst[1]/dy)), int(round(dst[2]/dz))]
        ax_.set_title(f"{title}   T(dst) = {float(T_dst):.1f} s", fontsize=13)
        ax_.set_xlim(0, Lx); ax_.set_ylim(0, Ly)
        ax_.set_aspect("equal")
        ax_.set_xlabel("x (m)"); ax_.set_ylabel("y (m)")
        ax_.legend(loc="upper left")

    plt.tight_layout()
    out_top = os.path.join(out_dir, f"routine_sk_t{t_label:06d}_topdown.png")
    plt.savefig(out_top, dpi=140)
    plt.close()
    print(f"[saved] {out_top}")

    # ---------- 3D view ----------
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    # Building footprint as a 'floor' image
    # Build per-column building heights (skip z=0 solid floor).
    # For each (ix, iy) the building extends from z=0 (or first solid z>0) up to
    # the highest z where mask is True. We draw bar3d boxes for visible columns.
    h_idx = np.zeros((nx, ny), dtype=np.int32)
    for iz in range(1, nz):
        h_idx = np.where(mask[:, :, iz], iz + 1, h_idx)
    heights_m = h_idx * dz   # building top elevation in metres

    # Coarsen the grid (one bar per BAR_STEP cells) so bar3d isn't too dense.
    BAR_STEP = max(1, nx // 64)
    n_bx = nx // BAR_STEP
    n_by = ny // BAR_STEP
    # Per coarse cell, take max height inside the BAR_STEP × BAR_STEP block
    h_coarse = heights_m[: n_bx*BAR_STEP, : n_by*BAR_STEP] \
                 .reshape(n_bx, BAR_STEP, n_by, BAR_STEP).max(axis=(1, 3))
    ys_b, xs_b = np.meshgrid(np.arange(n_by), np.arange(n_bx), indexing="xy")
    flat_x = (xs_b.ravel() + 0.5) * (BAR_STEP * dx) - 0.5 * (BAR_STEP * dx)
    flat_y = (ys_b.ravel() + 0.5) * (BAR_STEP * dy) - 0.5 * (BAR_STEP * dy)
    flat_h = h_coarse.ravel()
    keep = flat_h > 0
    bar_w = BAR_STEP * dx
    if keep.sum() > 0:
        # Colour the bars by height (taller buildings lighter).
        norm = plt.Normalize(vmin=0.0, vmax=max(flat_h[keep].max(), 1.0))
        colours = plt.cm.copper(norm(flat_h[keep]))
        ax.bar3d(
            flat_x[keep], flat_y[keep], np.zeros(keep.sum()),
            bar_w, bar_w, flat_h[keep],
            color=colours, edgecolor="#222", linewidth=0.15,
            shade=True, alpha=0.85, zsort="min",
        )
        # Building height colourbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap="copper")
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.05)
        cb.set_label("building height (m)")
    ax.plot(p0[:, 0], p0[:, 1], p0[:, 2], "-", color="#1f77b4", lw=2.0, label=f"no wind  T={float(d['T0_dst']):.1f}s")
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], "-", color="#d62728", lw=2.0, label=f"with wind  T={float(d['T1_dst']):.1f}s")
    ax.scatter([src[0]], [src[1]], [src[2]], s=120, c="lime", edgecolor="black")
    ax.scatter([dst[0]], [dst[1]], [dst[2]], s=120, c="red", edgecolor="black")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_zlim(0, nz * dz)
    ax.set_title(f"South Kensington routine  t={t_label}  wind_scale={float(d['wind_scale']):g}")
    ax.legend()
    ax.view_init(elev=25, azim=-130)
    plt.tight_layout()
    out_3d = os.path.join(out_dir, f"routine_sk_t{t_label:06d}_3d.png")
    plt.savefig(out_3d, dpi=140)
    plt.close()
    print(f"[saved] {out_3d}")

    # ---------- Path altitude profile ----------
    fig, ax = plt.subplots(figsize=(11, 4))
    s0 = np.linalg.norm(np.diff(p0, axis=0), axis=1).cumsum()
    s0 = np.concatenate([[0.0], s0])
    s1 = np.linalg.norm(np.diff(p1, axis=0), axis=1).cumsum()
    s1 = np.concatenate([[0.0], s1])
    ax.plot(s0, p0[:, 2], "-", color="#1f77b4", lw=2.0, label=f"no wind   length={s0[-1]:.0f}m  T={float(d['T0_dst']):.1f}s")
    ax.plot(s1, p1[:, 2], "-", color="#d62728", lw=2.0, label=f"with wind length={s1[-1]:.0f}m  T={float(d['T1_dst']):.1f}s")
    ax.set_xlabel("arc length s (m)"); ax.set_ylabel("z (m)")
    ax.set_title("Altitude profile along path")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    out_z = os.path.join(out_dir, f"routine_sk_t{t_label:06d}_zprofile.png")
    plt.savefig(out_z, dpi=140)
    plt.close()
    print(f"[saved] {out_z}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/leonardo_work/EUHPC_D25_028/Levistone/SCALED-UrbanFlowv3/inference_output/routine_sk_wind/routine_sk_t003000.npz")
    p.add_argument("--out_dir", default="/leonardo_work/EUHPC_D25_028/Levistone/SCALED-UrbanFlowv3/inference_output/routine_sk_wind")
    p.add_argument("--vis_z", type=int, default=12)
    args = p.parse_args()
    render(args.npz, args.out_dir, args.vis_z)
