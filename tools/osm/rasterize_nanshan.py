# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "geopandas>=0.14",
#   "rasterio>=1.3",
#   "shapely>=2.0",
#   "pyproj>=3.6",
#   "numpy",
#   "pillow",
# ]
# ///
"""Rasterize Nanshan OSM layers to a 4 m/grid stack.

Output: nanshan/raster_4m.npz with
    buildings        (H, W) uint8     binary mask
    building_height  (H, W) float32   metres (0 outside buildings)
    water            (H, W) uint8
    roads            (H, W) uint8     buffered by highway class
    landuse_class    (H, W) uint8     class id, 0 = none
    boundary         (H, W) uint8     1 inside Nanshan polygon
    transform        (6,)   float64   affine [a, b, c, d, e, f]
    bounds           (4,)   float64   [minx, miny, maxx, maxy] in EPSG:32650
    crs              str              "EPSG:32650"
    res              float64          4.0
    landuse_classes  list[str]        index -> tag value
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine
from PIL import Image
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2] / ".cache" / "shenzhen-osm"
SRC = ROOT / "nanshan"
OUT_NPZ = SRC / "raster_4m.npz"
OUT_PNG = SRC / "raster_4m_preview.png"

TARGET_CRS = "EPSG:32650"
RES = 4.0

ROAD_HALF_WIDTH = {
    "motorway": 12.0, "motorway_link": 6.0,
    "trunk": 10.0,    "trunk_link": 5.0,
    "primary": 8.0,   "primary_link": 4.0,
    "secondary": 6.0, "secondary_link": 3.0,
    "tertiary": 5.0,  "tertiary_link": 3.0,
    "residential": 4.0,
    "unclassified": 4.0,
    "service": 3.0,
    "living_street": 3.0,
}
ROAD_DEFAULT_HALF = 2.5


def parse_height(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def get_tag(props, key):
    t = props.get("tags") if isinstance(props, dict) else None
    if isinstance(t, dict):
        v = t.get(key)
        if v is not None:
            return v
    return props.get(key) if isinstance(props, dict) else None


def load(path):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(4326)
    return g.to_crs(TARGET_CRS)


def main():
    t0 = time.time()
    print(f"[load] boundary", flush=True)
    boundary = load(SRC / "boundary.geojson")
    mask_geom = boundary.geometry.union_all()

    minx, miny, maxx, maxy = mask_geom.bounds
    # snap to 4 m grid (expand outward)
    minx = np.floor(minx / RES) * RES
    miny = np.floor(miny / RES) * RES
    maxx = np.ceil(maxx / RES) * RES
    maxy = np.ceil(maxy / RES) * RES
    W = int(round((maxx - minx) / RES))
    H = int(round((maxy - miny) / RES))
    # rasterio convention: y origin at top, negative dy
    transform = Affine(RES, 0.0, minx, 0.0, -RES, maxy)
    print(f"  grid {H} x {W}  ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m)", flush=True)

    def rasterize(shapes, dtype=np.uint8, fill=0, all_touched=False):
        return rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(H, W),
            transform=transform,
            fill=fill,
            dtype=dtype,
            all_touched=all_touched,
        )

    print(f"[rast] boundary mask", flush=True)
    boundary_mask = rasterize([(g, 1) for g in boundary.geometry], dtype=np.uint8)

    # --- buildings ----------------------------------------------------------
    print(f"[load] buildings", flush=True)
    bld = load(SRC / "buildings.geojson")
    print(f"       {len(bld)} features", flush=True)

    def building_height(props):
        h = parse_height(get_tag(props, "height"))
        if h is not None and h > 0:
            return float(h)
        lv = parse_height(get_tag(props, "building:levels"))
        if lv is not None and lv > 0:
            return float(lv) * 3.0
        return 6.0

    bld_mask_shapes = [(g, 1) for g in bld.geometry if not g.is_empty]
    bld_h_shapes = [
        (g, building_height(row.drop("geometry").to_dict()))
        for _, row in bld.iterrows()
        for g in [row.geometry]
        if g is not None and not g.is_empty
    ]
    print(f"[rast] buildings mask", flush=True)
    buildings = rasterize(bld_mask_shapes, dtype=np.uint8)
    print(f"[rast] buildings height", flush=True)
    building_h = rasterize(bld_h_shapes, dtype=np.float32, fill=0.0)

    # --- water --------------------------------------------------------------
    print(f"[load] water", flush=True)
    water_g = load(SRC / "water.geojson")
    print(f"       {len(water_g)} features", flush=True)
    water = rasterize(
        [(g, 1) for g in water_g.geometry if g is not None and not g.is_empty],
        dtype=np.uint8,
    )

    # --- roads (buffered by class) -----------------------------------------
    print(f"[load] roads", flush=True)
    roads_g = load(SRC / "roads.geojson")
    print(f"       {len(roads_g)} features", flush=True)
    road_shapes = []
    for _, row in roads_g.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        hwy = get_tag(row.drop("geometry").to_dict(), "highway")
        half = ROAD_HALF_WIDTH.get(hwy, ROAD_DEFAULT_HALF)
        try:
            buf = g.buffer(half, cap_style=2, join_style=2)
        except Exception:
            continue
        if buf.is_empty:
            continue
        road_shapes.append((buf, 1))
    print(f"[rast] roads ({len(road_shapes)} buffered)", flush=True)
    roads = rasterize(road_shapes, dtype=np.uint8)

    # --- landuse (with class id) -------------------------------------------
    print(f"[load] landuse", flush=True)
    lu = load(SRC / "landuse.geojson")
    print(f"       {len(lu)} features", flush=True)
    lu_vals = []
    for _, row in lu.iterrows():
        v = get_tag(row.drop("geometry").to_dict(), "landuse")
        if v is None:
            v = get_tag(row.drop("geometry").to_dict(), "natural")
        lu_vals.append(v if isinstance(v, str) else "unknown")
    uniq = sorted({v for v in lu_vals})
    # reserve 0 for "none / background"; classes start at 1
    cls_to_id = {v: i + 1 for i, v in enumerate(uniq)}
    print(f"       {len(uniq)} landuse classes (max id {len(uniq)})", flush=True)
    lu_shapes = []
    for v, geom in zip(lu_vals, lu.geometry):
        if geom is None or geom.is_empty:
            continue
        lu_shapes.append((geom, cls_to_id[v]))
    landuse_class = rasterize(lu_shapes, dtype=np.uint8 if len(uniq) < 255 else np.uint16)
    landuse_classes = [""] + [v for v in uniq]  # index 0 -> ""

    # --- save ---------------------------------------------------------------
    print(f"[save] {OUT_NPZ}", flush=True)
    np.savez_compressed(
        OUT_NPZ,
        buildings=buildings,
        building_height=building_h,
        water=water,
        roads=roads,
        landuse_class=landuse_class,
        boundary=boundary_mask,
        transform=np.asarray(transform[:6], dtype=np.float64),
        bounds=np.asarray([minx, miny, maxx, maxy], dtype=np.float64),
        crs=np.asarray(TARGET_CRS),
        res=np.asarray(RES, dtype=np.float64),
        landuse_classes=np.asarray(landuse_classes, dtype=object),
    )
    print(f"  size: {OUT_NPZ.stat().st_size/1e6:.1f} MB", flush=True)

    # --- preview PNG -------------------------------------------------------
    print(f"[save] preview {OUT_PNG}", flush=True)
    # RGB: water blue, landuse light, roads dark grey, buildings red
    rgb = np.full((H, W, 3), 244, dtype=np.uint8)  # background
    rgb[boundary_mask == 0] = (255, 255, 255)
    lu_present = landuse_class > 0
    rgb[lu_present] = (230, 226, 211)
    rgb[water == 1] = (166, 206, 227)
    rgb[roads == 1] = (90, 90, 90)
    rgb[buildings == 1] = (140, 50, 50)
    # downsample for a smaller preview if big
    img = Image.fromarray(rgb)
    if max(img.size) > 4096:
        scale = 4096 / max(img.size)
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.NEAREST,
        )
    img.save(OUT_PNG, optimize=True)
    print(f"  preview {img.size} -> {OUT_PNG.stat().st_size/1e6:.1f} MB", flush=True)

    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
