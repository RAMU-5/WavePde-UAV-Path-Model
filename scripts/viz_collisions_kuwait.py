"""Visualise geometry + traced routes + collision points.

A path "collision" point = a densely-sampled path location whose cell is
inside a building voxel. Plots: (left) whole domain with all routes and
collision points in red; (right) zoom on the worst collision cluster.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
K = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="kuwait_stations.npz")
    p.add_argument("--dbdir", default="database_kuwait")
    args = p.parse_args()
    DB = ROOT / "output" / args.dbdir
    tag = args.dbdir.replace("database_", "")
    z = np.load(ROOT / "input" / args.domain)
    mask = z["mask"]; dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    st = z["stations_xy_m"]

    def inside_pts(pts):
        ix = np.clip(np.round(pts[:, 0] / dx).astype(np.int32), 0, nx - 1)
        iy = np.clip(np.round(pts[:, 1] / dy).astype(np.int32), 0, ny - 1)
        iz = np.clip(np.round(pts[:, 2] / dz).astype(np.int32), 0, nz - 1)
        return mask[ix, iy, iz]

    all_paths, coll_pts = [], []
    for i in range(64):
        f = DB / f"paths_{i:03d}.npz"
        if not f.exists():
            continue
        a = np.load(f, allow_pickle=False)
        off = a["path_offsets"]; pc = a["paths_concat"]; dst = a["dst_indices"]
        for k in range(len(dst)):
            seg = pc[off[k]:off[k + 1]]
            if len(seg) < 1:
                continue
            all_paths.append(seg)
            if len(seg) >= 2:
                a0, a1 = seg[:-1], seg[1:]
                t = np.linspace(0, 1, K, endpoint=False)
                samp = (a0[:, None, :] + (a1 - a0)[:, None, :] * t[None, :, None]
                        ).reshape(-1, 3)
            else:
                samp = seg
            hit = inside_pts(samp)
            if hit.any():
                coll_pts.append(samp[hit])
    coll = np.concatenate(coll_pts, axis=0) if coll_pts else np.zeros((0, 3))
    print(f"[viz] {len(all_paths)} paths, {len(coll)} collision sample points",
          flush=True)

    foot = mask.any(axis=2).T.astype(float); foot[foot == 0] = np.nan
    ext = [0, nx * dx, 0, ny * dy]
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    for ax in axs:
        ax.imshow(foot, origin="lower", cmap="Greys", vmin=0, vmax=1.4, extent=ext)
        for s in all_paths:
            ax.plot(s[:, 0], s[:, 1], lw=0.3, color="tab:blue", alpha=0.3)
        ax.scatter(st[:, 0], st[:, 1], c="limegreen", s=22, edgecolor="k",
                   lw=0.4, zorder=4)
        if len(coll):
            ax.scatter(coll[:, 0], coll[:, 1], c="red", s=10, zorder=6)

    axs[0].set_title(f"All routes + {len(coll)} collision points (red)")
    axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)")

    # zoom on densest collision cluster
    if len(coll):
        # grid-bin collisions to find the hottest 1km box
        cx = coll[:, 0]; cy = coll[:, 1]
        hx = np.floor(cx / 500).astype(int); hy = np.floor(cy / 500).astype(int)
        keys, cnts = np.unique(np.column_stack([hx, hy]), axis=0,
                               return_counts=True)
        bx, by = keys[np.argmax(cnts)]
        cxm, cym = (bx + 0.5) * 500, (by + 0.5) * 500
    else:
        cxm, cym = nx * dx / 2, ny * dy / 2
    R = 700
    axs[1].set_xlim(cxm - R, cxm + R); axs[1].set_ylim(cym - R, cym + R)
    for s in all_paths:
        axs[1].plot(s[:, 0], s[:, 1], lw=1.0, color="tab:blue", alpha=0.6)
    if len(coll):
        axs[1].scatter(coll[:, 0], coll[:, 1], c="red", s=40, zorder=6)
    axs[1].set_title(f"Zoom on worst collision cluster (±{R} m)")
    axs[1].set_xlabel("x (m)"); axs[1].set_ylabel("y (m)")

    plt.tight_layout()
    out = ROOT / "output" / "wave" / f"kuwait_collisions_{tag}.png"
    plt.savefig(out, dpi=130)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
