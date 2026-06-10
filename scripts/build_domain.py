"""Build the large 6.6×7.2 km Nanshan upper-right domain for the
118-station path planning batch.

Outputs `domain.npz` (default; override with --out):
    mask  : (NX, NY, NZ) bool building voxel mask
    info  : grid metadata (dx, dy, dz, etc.)
    stations_xy_m : (N, 3) station positions in local metres (x, y, z=5)

CLI:
    python setup.py                    # 50m default (fast dev)
    python setup.py --dx 5 --dy 5 --dz 5 --out domain_5m.npz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, box

THIS_DIR = Path(__file__).resolve().parent
OSM_BUILDINGS = THIS_DIR.parent / ".cache" / "shenzhen-osm" / "nanshan" / "buildings.geojson"
STATIONS_JSON = THIS_DIR.parent / "input" / "stations.json"

# Domain extent — exactly the bbox used to deploy the 118 stations.
DOMAIN_BBOX_LONLAT = (113.900, 22.520, 113.965, 22.585)

# Grid resolution. Coarse 50m × 14m chosen because (a) the framework
# kills python after ~5 min so we need wave solves to finish in <60s,
# and (b) for 6km city-scale routing we don't need to resolve every
# small building — only major obstacle clusters.
DX = 50.0   # m
DY = 50.0
DZ = 14.0   # m
SIZE_Z_M = 210.0  # vertical extent (covers up to typical 100-200m buildings)

# Station altitude (last-mile pickup at ground level)
STATION_Z_M = 5.0


def parse_tags(s):
    if isinstance(s, dict):
        return s
    if isinstance(s, str) and s.startswith("{"):
        try:
            return eval(s)
        except Exception:
            return {}
    return {}


def building_height_m(tags):
    h = tags.get("height")
    if h:
        try: return float(str(h).split()[0])
        except Exception: pass
    lv = tags.get("building:levels")
    if lv:
        try: return 3.0 * float(lv)
        except Exception: pass
    return 9.0


def lonlat_to_local_m(lons, lats, lon0, lat0):
    LAT_M = 111000.0
    LON_M = LAT_M * np.cos(np.deg2rad(lat0))
    return ((np.asarray(lons) - lon0) * LON_M,
            (np.asarray(lats) - lat0) * LAT_M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dx", type=float, default=DX)
    p.add_argument("--dy", type=float, default=DY)
    p.add_argument("--dz", type=float, default=DZ)
    p.add_argument("--size-z", type=float, default=SIZE_Z_M)
    p.add_argument("--out", type=str, default="domain.npz")
    args = p.parse_args()
    dx, dy, dz, size_z = args.dx, args.dy, args.dz, args.size_z

    lon0, lat0, lon1, lat1 = DOMAIN_BBOX_LONLAT
    LAT_M = 111000.0
    LON_M = LAT_M * np.cos(np.deg2rad(lat0))
    SIZE_X_M = (lon1 - lon0) * LON_M
    SIZE_Y_M = (lat1 - lat0) * LAT_M
    NX = int(round(SIZE_X_M / dx))
    NY = int(round(SIZE_Y_M / dy))
    NZ = int(round(size_z / dz))
    print(f"[domain] bbox lon [{lon0}, {lon1}], lat [{lat0}, {lat1}]")
    print(f"[domain] size {SIZE_X_M:.0f}m × {SIZE_Y_M:.0f}m × {size_z:.0f}m")
    print(f"[domain] grid {NX} × {NY} × {NZ}  "
          f"(dx={dx}m dy={dy}m dz={dz}m, {NX*NY*NZ:,d} cells)")
    n_cells = NX * NY * NZ
    print(f"[domain] memory estimate per psi-buffer: "
          f"{n_cells * 4 / 1e9:.2f} GB float32")

    # ---- Load buildings inside the domain bbox -------------------------
    print("[buildings] loading...")
    with open(OSM_BUILDINGS) as f:
        gj = json.load(f)
    domain_box = box(lon0, lat0, lon1, lat1)
    rings_xy_m = []
    heights = []
    for feat in gj["features"]:
        g = feat["geometry"]
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            rings = [poly[0] for poly in g["coordinates"]]
        else:
            continue
        tags = parse_tags(feat.get("properties", {}).get("tags", ""))
        h = building_height_m(tags)
        for ring in rings:
            if len(ring) < 4:
                continue
            try:
                poly = Polygon(ring)
            except Exception:
                continue
            if not poly.is_valid:
                continue
            if not poly.intersects(domain_box):
                continue
            lons = np.array([p[0] for p in ring])
            lats = np.array([p[1] for p in ring])
            xs_m, ys_m = lonlat_to_local_m(lons, lats, lon0, lat0)
            rings_xy_m.append(np.column_stack([xs_m, ys_m]))
            heights.append(h)
    print(f"[buildings] {len(rings_xy_m)} polygons inside domain")

    # ---- Voxelize ------------------------------------------------------
    print("[voxelize] rasterising...")
    mask = np.zeros((NX, NY, NZ), dtype=bool)
    cell_x = (np.arange(NX) + 0.5) * dx
    cell_y = (np.arange(NY) + 0.5) * dy
    for i, (ring, h) in enumerate(zip(rings_xy_m, heights)):
        x_min, y_min = ring[:, 0].min(), ring[:, 1].min()
        x_max, y_max = ring[:, 0].max(), ring[:, 1].max()
        ix0 = max(0, int(np.floor(x_min / dx)) - 1)
        ix1 = min(NX, int(np.ceil(x_max / dx)) + 1)
        iy0 = max(0, int(np.floor(y_min / dy)) - 1)
        iy1 = min(NY, int(np.ceil(y_max / dy)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        XX, YY = np.meshgrid(cell_x[ix0:ix1], cell_y[iy0:iy1], indexing="ij")
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
        path = MplPath(ring)
        inside = path.contains_points(pts).reshape(XX.shape)
        if not inside.any():
            continue
        k_top = max(1, min(NZ, int(np.ceil(h / dz))))
        sub = mask[ix0:ix1, iy0:iy1, 0:k_top]
        sub |= inside[..., None]
        mask[ix0:ix1, iy0:iy1, 0:k_top] = sub
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(rings_xy_m)}")
    print(f"[voxelize] mask filled fraction: {mask.mean()*100:.2f}%  "
          f"({mask.sum():,d} of {mask.size:,d} cells)")

    # ---- Stations to local coordinates --------------------------------
    print("[stations] loading + projecting...")
    with open(STATIONS_JSON) as f:
        sjs = json.load(f)
    stations = sjs["stations"]
    s_x_m, s_y_m = lonlat_to_local_m(
        [s["lon"] for s in stations],
        [s["lat"] for s in stations], lon0, lat0)
    s_z_m = np.full(len(stations), STATION_Z_M)
    stations_xy_m = np.column_stack([s_x_m, s_y_m, s_z_m])
    print(f"[stations] {len(stations)} stations at z={STATION_Z_M}m")

    # ---- Save ----------------------------------------------------------
    out = THIS_DIR.parent / "input" / args.out
    np.savez_compressed(
        out, mask=mask,
        dx=dx, dy=dy, dz=dz,
        nx=NX, ny=NY, nz=NZ,
        size_x_m=SIZE_X_M, size_y_m=SIZE_Y_M, size_z_m=size_z,
        domain_bbox_lonlat=np.array(DOMAIN_BBOX_LONLAT),
        station_z_m=STATION_Z_M,
        stations_xy_m=stations_xy_m,
        station_lonlat=np.array([[s["lon"], s["lat"]] for s in stations]),
        station_source=np.array([s["source"] for s in stations]),
        station_n_buildings_within_max=np.array(
            [s["n_buildings_within_max"] for s in stations]),
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
