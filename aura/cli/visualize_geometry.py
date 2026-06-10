"""Visualize the Stage 1 20 km^2 Nanshan geometry from shenzhen-osm GeoJSONs.

Pure json + matplotlib so it runs in any env with numpy + matplotlib (no
geopandas required). Crops boundary/water/landuse/buildings/roads to the
Stage 1 bbox and renders a single PNG (and optional individual layer PNGs
for QA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle


# Stage 1 bbox = the bbox in aura/config/stage1.yaml (kept as default here).
STAGE1_BBOX = (113.930, 22.535, 113.976, 22.571)
NANSHAN_DIR_DEFAULT = Path(
    "/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/"
    "shenzhen-osm/nanshan"
)


def _ring_in_bbox(ring: List[List[float]], bbox) -> bool:
    """Cheap test: any vertex of the ring inside bbox -> keep."""
    min_lon, min_lat, max_lon, max_lat = bbox
    for lon, lat in ring:
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True
    return False


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


def _highway(feat) -> str:
    tags = feat.get("properties", {}).get("tags") or {}
    if isinstance(tags, dict):
        return tags.get("highway", "")
    return ""


def load_polygons(path: Path, bbox) -> List[np.ndarray]:
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    out: List[np.ndarray] = []
    for feat in gj.get("features", []):
        for poly in _iter_polygons(feat):
            outer = poly[0]
            if _ring_in_bbox(outer, bbox):
                out.append(np.asarray(outer))
    return out


def load_lines(path: Path, bbox) -> List[Tuple[str, np.ndarray]]:
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    out: List[Tuple[str, np.ndarray]] = []
    for feat in gj.get("features", []):
        hw = _highway(feat)
        for ls in _iter_lines(feat):
            arr = np.asarray(ls)
            min_lon, min_lat, max_lon, max_lat = bbox
            if (arr[:, 0].max() < min_lon or arr[:, 0].min() > max_lon
                    or arr[:, 1].max() < min_lat or arr[:, 1].min() > max_lat):
                continue
            out.append((hw, arr))
    return out


def render(out_path: Path, bbox, src_dir: Path,
           full_nanshan: bool = False) -> None:
    print(f"[viz] bbox = {bbox}  src = {src_dir}")
    print("[viz] loading boundary, water, landuse, buildings, roads...",
          flush=True)
    if full_nanshan:
        # widen bbox to the whole Nanshan extent for context plot
        with open(src_dir / "boundary.geojson") as f:
            d = json.load(f)
        coords = []
        for feat in d["features"]:
            for poly in _iter_polygons(feat):
                coords.extend(poly[0])
        arr = np.asarray(coords)
        plot_bbox = (arr[:, 0].min(), arr[:, 1].min(),
                     arr[:, 0].max(), arr[:, 1].max())
    else:
        plot_bbox = bbox

    boundary = load_polygons(src_dir / "boundary.geojson", plot_bbox)
    water = load_polygons(src_dir / "water.geojson", plot_bbox)
    landuse = load_polygons(src_dir / "landuse.geojson", plot_bbox)
    buildings = load_polygons(src_dir / "buildings.geojson", plot_bbox)
    roads = load_lines(src_dir / "roads.geojson", plot_bbox)

    print(f"[viz]   boundary={len(boundary)}  water={len(water)}  "
          f"landuse={len(landuse)}  buildings={len(buildings)}  "
          f"road_segments={len(roads)}")

    # crop to plot bbox
    min_lon, min_lat, max_lon, max_lat = plot_bbox
    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (min_lat + max_lat)))
    fig_h = 11.0
    fig_w = fig_h * (max_lon - min_lon) / ((max_lat - min_lat) * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_facecolor("#f4f1ea")

    if landuse:
        ax.add_collection(PolyCollection(
            landuse, facecolors="#e6e2d3", edgecolors="none", alpha=0.6))
    if water:
        ax.add_collection(PolyCollection(
            water, facecolors="#a6cee3", edgecolors="#7fb3d5", linewidths=0.3))
    if buildings:
        ax.add_collection(PolyCollection(
            buildings, facecolors="#7c7a75", edgecolors="none", alpha=0.85))

    major_keys = {"motorway", "trunk", "primary",
                  "motorway_link", "trunk_link", "primary_link"}
    secondary_keys = {"secondary", "tertiary",
                      "secondary_link", "tertiary_link"}
    minor: List[np.ndarray] = []
    secondary: List[np.ndarray] = []
    major: List[np.ndarray] = []
    for hw, arr in roads:
        if hw in major_keys:
            major.append(arr)
        elif hw in secondary_keys:
            secondary.append(arr)
        else:
            minor.append(arr)
    if minor:
        ax.add_collection(LineCollection(minor, colors="#b8b8b8", linewidths=0.25))
    if secondary:
        ax.add_collection(LineCollection(secondary, colors="#444444", linewidths=0.55))
    if major:
        ax.add_collection(LineCollection(major, colors="#d62728", linewidths=1.0))

    # district boundary
    for ring in boundary:
        ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=1.2,
                zorder=10)

    # Stage1 bbox rectangle (in red) -- only meaningful if we plotted wider
    if full_nanshan:
        sb = STAGE1_BBOX
        ax.add_patch(Rectangle(
            (sb[0], sb[1]), sb[2] - sb[0], sb[3] - sb[1],
            fill=False, edgecolor="#d62728", linewidth=2.0, zorder=11,
            label="Stage 1 bbox (~20 km^2)",
        ))
        ax.legend(loc="lower left")

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    title_extra = " (Nanshan full extent + Stage 1 bbox)" if full_nanshan else \
                  f" (Stage 1 bbox)"
    title = (f"Nanshan{title_extra} — "
             f"{(max_lon-min_lon)*111*np.cos(np.deg2rad(0.5*(min_lat+max_lat))):.1f}"
             f" x {(max_lat-min_lat)*111:.1f} km")
    ax.set_title(title, fontsize=14, pad=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="#f4f1ea")
    plt.close(fig)
    print(f"[viz] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(NANSHAN_DIR_DEFAULT),
                   help="Path to shenzhen-osm/nanshan/ directory.")
    p.add_argument("--out", default="runs/viz/stage1_geometry.png")
    p.add_argument("--context", default="runs/viz/nanshan_context.png",
                   help="Optional Nanshan-wide context plot path.")
    p.add_argument("--bbox", nargs=4, type=float, default=list(STAGE1_BBOX),
                   help="lon_min lat_min lon_max lat_max")
    p.add_argument("--no-context", action="store_true",
                   help="Skip the wider Nanshan context plot.")
    args = p.parse_args(argv)

    bbox = tuple(args.bbox)
    src_dir = Path(args.src)
    out_path = Path(args.out)

    render(out_path, bbox, src_dir, full_nanshan=False)
    if not args.no_context:
        render(Path(args.context), bbox, src_dir, full_nanshan=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
