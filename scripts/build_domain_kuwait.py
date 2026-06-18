"""Build a building voxel domain from the Kuwait CBD buildings GeoJSON.

Adapted from build_domain.py. Differences:
  * heights are read directly from the `height_m` property (already
    derived upstream); null heights fall back to DEFAULT_HEIGHT_M.
  * the domain bbox is derived from the GeoJSON coordinate extent
    (optionally padded) rather than a hard-coded Shenzhen bbox.
  * no stations (this dataset is geometry only).

Outputs an .npz with:
    mask  : (NX, NY, NZ) bool building voxel mask
    grid metadata (dx, dy, dz, nx, ny, nz, sizes, bbox)

CLI:
    python build_domain_kuwait.py                       # 5m default
    python build_domain_kuwait.py --dx 50 --dy 50 --dz 14 --out kuwait_50m.npz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, box

THIS_DIR = Path(__file__).resolve().parent
GEOJSON = THIS_DIR.parent / "input" / "kuwait_cbd_buildings.geojson"

# Grid resolution (match domain_5m.npz by default).
DX = 5.0   # m
DY = 5.0
DZ = 5.0   # m
SIZE_Z_M = 320.0  # vertical extent (tallest building here is ~302 m)

DEFAULT_HEIGHT_M = 9.0  # fallback when height_m is null


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
    p.add_argument("--pad-m", type=float, default=0.0,
                   help="padding added around the data extent, in metres")
    p.add_argument("--out", type=str, default="kuwait_cbd.npz")
    args = p.parse_args()
    dx, dy, dz, size_z = args.dx, args.dy, args.dz, args.size_z

    # ---- Load buildings ------------------------------------------------
    print("[buildings] loading...")
    with open(GEOJSON) as f:
        gj = json.load(f)

    rings_lonlat = []
    heights = []
    lon_min = lat_min = +1e9
    lon_max = lat_max = -1e9
    for feat in gj["features"]:
        g = feat["geometry"]
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            rings = [poly[0] for poly in g["coordinates"]]
        else:
            continue
        h = feat.get("properties", {}).get("height_m")
        h = float(h) if h is not None else DEFAULT_HEIGHT_M
        for ring in rings:
            if len(ring) < 4:
                continue
            arr = np.asarray(ring, dtype=float)
            rings_lonlat.append(arr)
            heights.append(h)
            lon_min = min(lon_min, arr[:, 0].min())
            lon_max = max(lon_max, arr[:, 0].max())
            lat_min = min(lat_min, arr[:, 1].min())
            lat_max = max(lat_max, arr[:, 1].max())
    print(f"[buildings] {len(rings_lonlat)} polygons loaded")

    # ---- Domain bbox from data extent ----------------------------------
    lat0c = 0.5 * (lat_min + lat_max)
    LAT_M = 111000.0
    LON_M = LAT_M * np.cos(np.deg2rad(lat0c))
    pad_lon = args.pad_m / LON_M
    pad_lat = args.pad_m / LAT_M
    lon0, lat0 = lon_min - pad_lon, lat_min - pad_lat
    lon1, lat1 = lon_max + pad_lon, lat_max + pad_lat

    SIZE_X_M = (lon1 - lon0) * LON_M
    SIZE_Y_M = (lat1 - lat0) * LAT_M
    NX = int(round(SIZE_X_M / dx))
    NY = int(round(SIZE_Y_M / dy))
    NZ = int(round(size_z / dz))
    print(f"[domain] bbox lon [{lon0:.5f}, {lon1:.5f}], "
          f"lat [{lat0:.5f}, {lat1:.5f}]")
    print(f"[domain] size {SIZE_X_M:.0f}m × {SIZE_Y_M:.0f}m × {size_z:.0f}m")
    print(f"[domain] grid {NX} × {NY} × {NZ}  "
          f"(dx={dx}m dy={dy}m dz={dz}m, {NX*NY*NZ:,d} cells)")
    print(f"[domain] memory estimate per psi-buffer: "
          f"{NX*NY*NZ * 4 / 1e9:.2f} GB float32")

    # ---- Project rings to local metres ---------------------------------
    rings_xy_m = []
    for ring in rings_lonlat:
        xs_m, ys_m = lonlat_to_local_m(ring[:, 0], ring[:, 1], lon0, lat0)
        rings_xy_m.append(np.column_stack([xs_m, ys_m]))

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
        mask[ix0:ix1, iy0:iy1, 0:k_top] |= inside[..., None]
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(rings_xy_m)}")
    print(f"[voxelize] mask filled fraction: {mask.mean()*100:.2f}%  "
          f"({mask.sum():,d} of {mask.size:,d} cells)")

    # ---- Save ----------------------------------------------------------
    out = THIS_DIR.parent / "input" / args.out
    np.savez_compressed(
        out, mask=mask,
        dx=dx, dy=dy, dz=dz,
        nx=NX, ny=NY, nz=NZ,
        size_x_m=SIZE_X_M, size_y_m=SIZE_Y_M, size_z_m=size_z,
        domain_bbox_lonlat=np.array([lon0, lat0, lon1, lat1]),
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
