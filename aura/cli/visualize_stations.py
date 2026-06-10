"""Render the Stage 1 bbox with blocks (颜色填充) + station markers overlaid.

Reuses the shenzhen-osm/nanshan/ source layers and aura.network.blocks for
block siting. Output: PNG with three layers — OSM base, block polygons (per
station capacity color), station glyphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import box

from aura.network.blocks import site_blocks


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
    min_lon, min_lat, max_lon, max_lat = bbox
    for lon, lat in ring:
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True
    return False


def load_polygons_lonlat(path: Path, bbox) -> List[np.ndarray]:
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


def load_lines_lonlat(path: Path, bbox) -> List[tuple]:
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    out = []
    for feat in gj.get("features", []):
        tags = feat.get("properties", {}).get("tags") or {}
        hw = tags.get("highway", "") if isinstance(tags, dict) else ""
        for ls in _iter_lines(feat):
            arr = np.asarray(ls)
            if (arr[:, 0].max() < bbox[0] or arr[:, 0].min() > bbox[2]
                    or arr[:, 1].max() < bbox[1] or arr[:, 1].min() > bbox[3]):
                continue
            out.append((hw, arr))
    return out


def render(out_path: Path, bbox, src_dir: Path,
           min_area_m2: float = 2000.0, min_buildings: int = 1,
           buildings_per_pad: int = 3,
           n_stations_target: int | None = None,
           assets_dir: Path | None = None) -> None:
    print(f"[viz-stations] bbox={bbox} src={src_dir}")
    # Block siting in WGS84.
    def _l2m(lon, lat):
        scale = 111000.0
        return (lon * scale * np.cos(np.deg2rad(0.5 * (bbox[1] + bbox[3]))),
                lat * scale)
    stations, station_blocks, all_blocks = site_blocks(
        src_dir / "roads.geojson",
        src_dir / "buildings.geojson",
        src_dir / "water.geojson",
        bbox, _l2m,
        min_area_m2=min_area_m2,
        min_buildings=min_buildings,
        buildings_per_pad=buildings_per_pad,
        n_stations_target=n_stations_target,
    )

    print("[viz-stations] loading OSM base layers...")
    water = load_polygons_lonlat(src_dir / "water.geojson", bbox)
    landuse = load_polygons_lonlat(src_dir / "landuse.geojson", bbox)
    buildings = load_polygons_lonlat(src_dir / "buildings.geojson", bbox)
    roads = load_lines_lonlat(src_dir / "roads.geojson", bbox)

    # All polygonized blocks (faint, so the user sees the neighborhood
    # structure even where stations were not chosen).
    all_block_arrays = [np.asarray(a.polygon_lonlat.exterior.coords)
                        for a in all_blocks]
    # Subset that got a station — drawn slightly bolder.
    station_block_ids = {id(a) for a in station_blocks}
    chosen_arrays = [np.asarray(a.polygon_lonlat.exterior.coords)
                     for a in station_blocks]
    caps = np.asarray([s.capacity_pads for s in stations])
    n_b = np.asarray([a.n_buildings for a in station_blocks])

    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (bbox[1] + bbox[3])))
    fig_h = 12.0
    fig_w = fig_h * (bbox[2] - bbox[0]) / ((bbox[3] - bbox[1]) * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_facecolor("#f4f1ea")

    if landuse:
        ax.add_collection(PolyCollection(landuse, facecolors="#e6e2d3",
                                         edgecolors="none", alpha=0.5))
    if water:
        ax.add_collection(PolyCollection(water, facecolors="#a6cee3",
                                         edgecolors="#7fb3d5", linewidths=0.3))

    # Optional: Voronoi service regions, drawn as a translucent raster under
    # the blocks. Loaded from a Stage1Assets dir.
    if assets_dir is not None:
        from aura.io.assets import load as load_assets
        assets = load_assets(assets_dir)
        owner = assets.voronoi_owner
        nx, ny = owner.shape
        # Project each cell's index back to lon/lat for imshow extent.
        # We rely on the same approximate Mercator that built_assets used
        # so the picture lines up.
        scale_x = 111_000.0 * float(np.cos(np.deg2rad(0.5 * (bbox[1] + bbox[3]))))
        scale_y = 111_000.0
        x_lo = bbox[0]; x_hi = bbox[0] + (assets.grid.size_xy[0] / scale_x)
        y_lo = bbox[1]; y_hi = bbox[1] + (assets.grid.size_xy[1] / scale_y)
        # Random per-station color so adjacent cells are visually distinct.
        rng = np.random.default_rng(0)
        K = int(owner.max()) + 1
        color_lut = rng.uniform(0.55, 0.95, size=(K, 3))
        rgb = color_lut[owner].transpose(1, 0, 2)  # (ny, nx, 3) for imshow
        ax.imshow(rgb, origin="lower", extent=(x_lo, x_hi, y_lo, y_hi),
                  alpha=0.30, zorder=1, interpolation="nearest")

    # All blocks: light edges only — shows the polygonized neighborhood grid.
    if all_block_arrays:
        ax.add_collection(PolyCollection(
            all_block_arrays, facecolors="none", edgecolors="#7a7a7a",
            linewidths=0.25, alpha=0.5,
        ))
    # Chosen blocks: filled by building count.
    if chosen_arrays:
        norm = Normalize(vmin=1, vmax=max(8, np.percentile(n_b, 95)))
        cmap = plt.get_cmap("YlOrBr")
        face_colors = [cmap(norm(n)) for n in n_b]
        ax.add_collection(PolyCollection(
            chosen_arrays, facecolors=face_colors, edgecolors="#2a2a2a",
            linewidths=0.7, alpha=0.65,
        ))
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("buildings per chosen block")

    # buildings on top (faint)
    if buildings:
        ax.add_collection(PolyCollection(
            buildings, facecolors="#3a3a3a", edgecolors="none", alpha=0.35))

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
        ax.add_collection(LineCollection(minor, colors="#888888",
                                         linewidths=0.25, alpha=0.7))
    if secondary:
        ax.add_collection(LineCollection(secondary, colors="#444444",
                                         linewidths=0.5))
    if major:
        ax.add_collection(LineCollection(major, colors="#d62728",
                                         linewidths=1.0))

    # station markers — size scales with capacity_pads
    if stations:
        xs = np.asarray([a.centroid_lonlat[0] for a in station_blocks])
        ys = np.asarray([a.centroid_lonlat[1] for a in station_blocks])
        cap_range = max(1, caps.max() - caps.min())
        sizes = 9.0 + 7.0 * (caps - caps.min()) / cap_range
        ax.scatter(xs, ys, s=sizes ** 2, c="#1a3d8f", marker="o",
                   edgecolors="white", linewidths=0.7, zorder=20)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    side_km_x = (bbox[2] - bbox[0]) * 111 * np.cos(np.deg2rad(0.5 * (bbox[1] + bbox[3])))
    side_km_y = (bbox[3] - bbox[1]) * 111
    title = (f"Nanshan Stage 1 — {len(stations)} stations across {len(all_blocks)} blocks "
             f"  ({side_km_x:.1f} x {side_km_y:.1f} km, "
             f"~{side_km_x*side_km_y:.1f} km^2)")
    ax.set_title(title, fontsize=13, pad=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="#f4f1ea")
    plt.close(fig)
    print(f"[viz-stations] wrote {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"[viz-stations] capacity distribution: "
          f"min={caps.min()} med={int(np.median(caps))} "
          f"max={caps.max()} mean={caps.mean():.1f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(NANSHAN_DIR_DEFAULT))
    p.add_argument("--out", default="runs/viz/stage1_stations.png")
    p.add_argument("--bbox", nargs=4, type=float, default=list(STAGE1_BBOX))
    p.add_argument("--min-area-m2", type=float, default=2000.0)
    p.add_argument("--min-buildings", type=int, default=1)
    p.add_argument("--buildings-per-pad", type=int, default=3)
    p.add_argument("--with-voronoi", default=None,
                   help="Path to a Stage1Assets dir; overlays Voronoi service regions.")
    p.add_argument("--n-stations", type=int, default=None,
                   help="Thin to this many stations via weighted k-medoids.")
    args = p.parse_args(argv)
    assets_dir = Path(args.with_voronoi) if args.with_voronoi else None
    render(Path(args.out), tuple(args.bbox), Path(args.src),
           min_area_m2=args.min_area_m2,
           min_buildings=args.min_buildings,
           buildings_per_pad=args.buildings_per_pad,
           n_stations_target=args.n_stations,
           assets_dir=assets_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
