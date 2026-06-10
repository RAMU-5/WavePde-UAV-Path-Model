"""3D rendering of Stage 1 nominal trajectories over the building height field.

Output: a single static PNG with mpl_toolkits.mplot3d:
  - building height_mask drawn as a downsampled 3D surface (terrain-like)
  - trajectories drawn as 3D polylines colored by altitude
  - station markers at z=0
  - station service "5 km" sphere optional (off by default — too cluttered)

For a true interactive 3D view, use plotly when available (HTML output).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from aura.fmm.agd_solver import load_db
from aura.io.assets import load as load_assets


def _pair_metrics(traj):
    pts = np.asarray(traj.xyz, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0, 0.0, 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    plen = float(seg.sum())
    direct = float(np.linalg.norm(pts[-1] - pts[0]))
    tort = plen / max(direct, 1.0)
    zspan = float(pts[:, 2].max() - pts[:, 2].min())
    return plen, tort, zspan


def _select_pairs(db, n, rank_by, seed):
    keys = list(db.pairs.keys())
    if rank_by == "random":
        rng = np.random.default_rng(seed)
        return [keys[i] for i in rng.choice(len(keys), size=min(n, len(keys)),
                                            replace=False)]
    scored = []
    for k in keys:
        plen, tort, zspan = _pair_metrics(db.pairs[k])
        if rank_by == "altitude_span":
            score = zspan
        elif rank_by == "tortuosity":
            score = tort
        elif rank_by == "longest":
            score = plen
        else:
            score = 0.0
        scored.append((score, k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:n]]


def render_mpl(out_path: Path, assets_dir: Path, traj_dir: Path,
               n: int, rank_by: str, seed: int,
               surface_downsample: int = 8,
               elev: float = 30.0, azim: float = -60.0) -> None:
    print(f"[viz3d] loading...")
    assets = load_assets(assets_dir)
    db = load_db(traj_dir)
    grid = assets.grid

    # Grid in km for axis readability.
    nx, ny = grid.shape2
    ds = surface_downsample
    ix = np.arange(0, nx, ds); iy = np.arange(0, ny, ds)
    X = (grid.origin_xy[0] + (ix + 0.5) * grid.dx) / 1000.0
    Y = (grid.origin_xy[1] + (iy + 0.5) * grid.dy) / 1000.0
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    H = assets.height_mask[::ds, ::ds].astype(np.float32)

    fig = plt.figure(figsize=(18, 12), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Terrain — buildings as a (low) surface.
    ax.plot_surface(
        Xg, Yg, H, cmap="copper", linewidth=0, antialiased=False,
        alpha=0.55, rstride=1, cstride=1,
    )

    # Stations at z=0 (ground glyphs).
    s_xy = np.asarray([s.xy_m for s in assets.stations]) / 1000.0
    ax.scatter(s_xy[:, 0], s_xy[:, 1], np.zeros(len(s_xy)),
               c="#1a3d8f", s=22, depthshade=False, zorder=5)

    # Trajectories.
    keys = _select_pairs(db, n, rank_by, seed)
    print(f"[viz3d] plotting {len(keys)} trajectories (rank_by={rank_by})")
    cmap = plt.get_cmap("plasma")
    z_min, z_max = np.inf, -np.inf
    for k in keys:
        z = np.asarray([p[2] for p in db.pairs[k].xyz])
        if z.size:
            z_min = min(z_min, z.min()); z_max = max(z_max, z.max())
    if not np.isfinite(z_min):
        z_min, z_max = 0, 100
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    for k in keys:
        traj = db.pairs[k]
        if len(traj.xyz) < 2:
            continue
        pts = np.asarray(traj.xyz, dtype=np.float64)
        x = pts[:, 0] / 1000.0
        y = pts[:, 1] / 1000.0
        z = pts[:, 2]
        # Color by per-segment altitude.
        for i in range(len(pts) - 1):
            ax.plot(
                x[i:i + 2], y[i:i + 2], z[i:i + 2],
                color=cmap(norm(0.5 * (z[i] + z[i + 1]))),
                linewidth=1.6, alpha=0.95, zorder=6,
            )

    # Cosmetics.
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_zlabel("Altitude (m)")
    ax.set_zlim(0, max(120.0, float(assets.height_mask.max()) * 1.05))
    ax.view_init(elev=elev, azim=azim)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.05)
    cb.set_label("trajectory altitude (m)")
    ax.set_title(
        f"Nanshan Stage 1 — 3D nominal trajectories ({len(keys)} shown, "
        f"rank={rank_by})",
        fontsize=12, pad=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[viz3d] wrote {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)")


def render_plotly(html_path: Path, assets_dir: Path, traj_dir: Path,
                  n: int, rank_by: str, seed: int,
                  surface_downsample: int = 8) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[viz3d] plotly not available ({e}); skipping HTML.")
        return
    assets = load_assets(assets_dir)
    db = load_db(traj_dir)
    grid = assets.grid
    nx, ny = grid.shape2
    ds = surface_downsample
    X = (grid.origin_xy[0] + (np.arange(0, nx, ds) + 0.5) * grid.dx) / 1000.0
    Y = (grid.origin_xy[1] + (np.arange(0, ny, ds) + 0.5) * grid.dy) / 1000.0
    H = assets.height_mask[::ds, ::ds].astype(np.float32).T  # plotly uses (y, x)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=H, colorscale="copper", opacity=0.55,
        showscale=False, name="building height",
    ))
    keys = _select_pairs(db, n, rank_by, seed)
    for k in keys:
        traj = db.pairs[k]
        if len(traj.xyz) < 2:
            continue
        pts = np.asarray(traj.xyz)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0] / 1000.0, y=pts[:, 1] / 1000.0, z=pts[:, 2],
            mode="lines", line=dict(width=4, color=pts[:, 2],
                                    colorscale="Plasma"),
            showlegend=False,
        ))
    s_xy = np.asarray([s.xy_m for s in assets.stations]) / 1000.0
    fig.add_trace(go.Scatter3d(
        x=s_xy[:, 0], y=s_xy[:, 1], z=np.zeros(len(s_xy)),
        mode="markers", marker=dict(size=3, color="navy"),
        name="stations",
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (km)",
            yaxis_title="Northing (km)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
        ),
        title=f"Nanshan Stage 1 — 3D ({len(keys)} traj, rank={rank_by})",
        width=1200, height=800,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    print(f"[viz3d] wrote {html_path} "
          f"({html_path.stat().st_size/1e6:.1f} MB)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--traj", default=None)
    p.add_argument("--out", default="runs/viz/stage1_traj_3d.png")
    p.add_argument("--html", default="runs/viz/stage1_traj_3d.html")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--rank-by", default="altitude_span",
                   choices=["random", "tortuosity", "altitude_span", "longest"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ds", type=int, default=8,
                   help="height surface downsample factor (cells per pixel)")
    p.add_argument("--elev", type=float, default=30.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--no-html", action="store_true")
    args = p.parse_args(argv)
    traj_dir = Path(args.traj) if args.traj else Path(args.assets) / "traj"
    render_mpl(Path(args.out), Path(args.assets), traj_dir, args.n,
               args.rank_by, args.seed, surface_downsample=args.ds,
               elev=args.elev, azim=args.azim)
    if not args.no_html:
        render_plotly(Path(args.html), Path(args.assets), traj_dir, args.n,
                      args.rank_by, args.seed, surface_downsample=args.ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
