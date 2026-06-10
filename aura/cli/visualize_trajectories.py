"""Render Stage 1 nominal trajectories on top of the OSM map.

Two outputs:
  - <out>_top.png      : 2D top-down overlay of N sampled trajectories
                          colored by altitude.
  - <out>_profile.png  : Arc-length vs altitude profile for the same set,
                          with a horizontal building-height-along-path band
                          showing where the trajectory had to climb over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from aura.fmm.agd_solver import load_db
from aura.io.assets import load as load_assets


STAGE1_BBOX = (113.930, 22.535, 113.976, 22.571)
NANSHAN_DIR_DEFAULT = Path(
    "/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/"
    "shenzhen-osm/nanshan"
)


def _iter_polygons(feature):
    g = feature.get("geometry") or {}
    t = g.get("type")
    if t == "Polygon":
        yield g["coordinates"]
    elif t == "MultiPolygon":
        for poly in g["coordinates"]:
            yield poly


def _iter_lines(feature):
    g = feature.get("geometry") or {}
    t = g.get("type")
    if t == "LineString":
        yield g["coordinates"]
    elif t == "MultiLineString":
        for ls in g["coordinates"]:
            yield ls


def _ring_in_bbox(ring, bbox) -> bool:
    for lon, lat in ring:
        if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
            return True
    return False


def load_polys(path: Path, bbox) -> List[np.ndarray]:
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    out = []
    for feat in gj.get("features", []):
        for poly in _iter_polygons(feat):
            outer = poly[0]
            if _ring_in_bbox(outer, bbox):
                out.append(np.asarray(outer))
    return out


def load_lines(path: Path, bbox) -> List[np.ndarray]:
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    out = []
    for feat in gj.get("features", []):
        for ls in _iter_lines(feat):
            arr = np.asarray(ls)
            if (arr[:, 0].max() < bbox[0] or arr[:, 0].min() > bbox[2]
                    or arr[:, 1].max() < bbox[1] or arr[:, 1].min() > bbox[3]):
                continue
            out.append(arr)
    return out


def _xy_m_to_lonlat(xy_m: np.ndarray, grid, bbox) -> np.ndarray:
    """Inverse of build_assets' approximate Mercator (linear in bbox)."""
    lat0 = 0.5 * (bbox[1] + bbox[3])
    scale_x = 111_000.0 * float(np.cos(np.deg2rad(lat0)))
    scale_y = 111_000.0
    lon = bbox[0] + (xy_m[..., 0] - grid.origin_xy[0]) / scale_x
    lat = bbox[1] + (xy_m[..., 1] - grid.origin_xy[1]) / scale_y
    return np.stack([lon, lat], axis=-1)


def _pair_metrics(traj):
    pts = np.asarray(traj.xyz, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0, 0.0, 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path_len = float(seg.sum())
    direct = float(np.linalg.norm(pts[-1] - pts[0]))
    tort = path_len / max(direct, 1.0)
    z_span = float(pts[:, 2].max() - pts[:, 2].min())
    return path_len, tort, z_span


def render(out_prefix: Path, assets_dir: Path, traj_dir: Path,
           src_dir: Path, bbox, n_samples: int = 30,
           seed: int = 0, rank_by: str = "random") -> None:
    print(f"[viz-traj] loading assets/db...")
    assets = load_assets(assets_dir)
    db = load_db(traj_dir)
    grid = assets.grid

    pair_keys = list(db.pairs.keys())
    if rank_by == "random":
        rng = np.random.default_rng(seed)
        sampled = rng.choice(len(pair_keys), size=min(n_samples, len(pair_keys)),
                             replace=False)
        sampled_keys = [pair_keys[i] for i in sampled]
    else:
        scored = []
        for k in pair_keys:
            traj = db.pairs[k]
            plen, tort, zspan = _pair_metrics(traj)
            if rank_by == "tortuosity":
                score = tort
            elif rank_by == "altitude_span":
                score = zspan
            elif rank_by == "longest":
                score = plen
            else:
                score = 0.0
            scored.append((score, k))
        scored.sort(reverse=True)
        sampled_keys = [k for _, k in scored[:n_samples]]
    print(f"[viz-traj] sampling {len(sampled_keys)} of {len(pair_keys)} pairs "
          f"(rank_by={rank_by})")

    # ---- 2D top-down ----
    print("[viz-traj] loading OSM base layers...")
    water = load_polys(src_dir / "water.geojson", bbox)
    landuse = load_polys(src_dir / "landuse.geojson", bbox)
    buildings = load_polys(src_dir / "buildings.geojson", bbox)
    roads = load_lines(src_dir / "roads.geojson", bbox)

    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (bbox[1] + bbox[3])))
    fig_h = 12.0
    fig_w = fig_h * (bbox[2] - bbox[0]) / ((bbox[3] - bbox[1]) * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_facecolor("#f4f1ea")
    if landuse:
        ax.add_collection(PolyCollection(landuse, facecolors="#e6e2d3",
                                         edgecolors="none", alpha=0.55))
    if water:
        ax.add_collection(PolyCollection(water, facecolors="#a6cee3",
                                         edgecolors="#7fb3d5",
                                         linewidths=0.3))
    if buildings:
        ax.add_collection(PolyCollection(buildings, facecolors="#7c7a75",
                                         edgecolors="none", alpha=0.7))
    if roads:
        ax.add_collection(LineCollection(roads, colors="#888888",
                                         linewidths=0.25, alpha=0.7))

    # stations
    s_xy = np.asarray([s.xy_m for s in assets.stations])
    s_lonlat = _xy_m_to_lonlat(s_xy, grid, bbox)
    ax.scatter(s_lonlat[:, 0], s_lonlat[:, 1], s=18, c="#1a3d8f",
               marker="o", edgecolors="white", linewidths=0.5, zorder=20)

    # trajectories — colored by altitude (z_max along path).
    cmap = plt.get_cmap("plasma")
    z_min, z_max = np.inf, -np.inf
    for k in sampled_keys:
        traj = db.pairs[k]
        z = np.asarray([p[2] for p in traj.xyz], dtype=np.float64)
        if z.size:
            z_min = min(z_min, z.min())
            z_max = max(z_max, z.max())
    if not np.isfinite(z_min):
        z_min, z_max = 0, 100
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    for k in sampled_keys:
        traj = db.pairs[k]
        if not traj.xyz:
            continue
        pts_xy_m = np.asarray([(p[0], p[1]) for p in traj.xyz])
        pts_z = np.asarray([p[2] for p in traj.xyz])
        pts_ll = _xy_m_to_lonlat(pts_xy_m, grid, bbox)
        # build per-segment colored line
        segs = np.stack([pts_ll[:-1], pts_ll[1:]], axis=1)
        z_mid = 0.5 * (pts_z[:-1] + pts_z[1:])
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=1.4,
                            alpha=0.9, zorder=15)
        lc.set_array(z_mid)
        ax.add_collection(lc)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("trajectory altitude (m)")

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(aspect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    side_km_x = (bbox[2]-bbox[0]) * 111 * np.cos(np.deg2rad(0.5*(bbox[1]+bbox[3])))
    side_km_y = (bbox[3]-bbox[1]) * 111
    ax.set_title(
        f"Nanshan Stage 1 — {len(sampled_keys)} sampled nominal trajectories "
        f"({side_km_x:.1f}x{side_km_y:.1f} km)",
        fontsize=13, pad=12,
    )

    out_top = out_prefix.with_suffix(".top.png")
    fig.savefig(out_top, bbox_inches="tight", facecolor="#f4f1ea")
    plt.close(fig)
    print(f"[viz-traj] wrote {out_top} "
          f"({out_top.stat().st_size/1e6:.1f} MB)")

    # ---- altitude profile ----
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for k in sampled_keys:
        traj = db.pairs[k]
        if len(traj.xyz) < 2:
            continue
        pts = np.asarray(traj.xyz)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        ax.plot(s, pts[:, 2], linewidth=1.1, alpha=0.7)
        # building-height along path (sampled at each trajectory point)
        ijk = grid.world_to_index(pts)
        ix = ijk[..., 0]; iy = ijk[..., 1]
        bh = assets.height_mask[ix, iy]
        ax.fill_between(s, 0, bh, color="#888888", alpha=0.05, linewidth=0)
    ax.set_xlabel("arc length (m)")
    ax.set_ylabel("altitude (m)")
    ax.set_title(
        f"Altitude profile of {len(sampled_keys)} sampled trajectories "
        f"(gray = building height under path)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(120.0, float(assets.height_mask.max()) * 1.05))

    out_prof = out_prefix.with_suffix(".profile.png")
    fig.savefig(out_prof, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz-traj] wrote {out_prof} "
          f"({out_prof.stat().st_size/1e6:.1f} MB)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--traj", default=None,
                   help="Trajectory DB dir (default: <assets>/traj).")
    p.add_argument("--src", default=str(NANSHAN_DIR_DEFAULT))
    p.add_argument("--out", default="runs/viz/stage1_traj")
    p.add_argument("--bbox", nargs=4, type=float, default=list(STAGE1_BBOX))
    p.add_argument("--n", type=int, default=30,
                   help="Number of sampled trajectories to render.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank-by", default="random",
                   choices=["random", "tortuosity", "altitude_span", "longest"],
                   help="Which trajectories to surface; default = random sample.")
    args = p.parse_args(argv)
    traj_dir = Path(args.traj) if args.traj else Path(args.assets) / "traj"
    render(Path(args.out), Path(args.assets), traj_dir, Path(args.src),
           tuple(args.bbox), n_samples=args.n, seed=args.seed,
           rank_by=args.rank_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
