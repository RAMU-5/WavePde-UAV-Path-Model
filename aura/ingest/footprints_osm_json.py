"""Rasterize OSM building footprints into a (nx, ny) max-height mask.

Pure-Python path (json + shapely + matplotlib.Path), no rasterio/geopandas
needed. Heights resolved per OSM convention:
  - tags.height   (numeric or '<n> m')
  - tags.building:levels x 3 m
  - class-conditional defaults (residential 12 m / commercial 30 m / 18 m).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from matplotlib.path import Path as MplPath


def _iter_polygons(feature) -> Iterable[list]:
    g = feature.get("geometry") or {}
    t = g.get("type")
    if t == "Polygon":
        yield g["coordinates"]
    elif t == "MultiPolygon":
        for poly in g["coordinates"]:
            yield poly


def _resolve_height(tags: dict,
                    default_residential: float = 12.0,
                    default_commercial: float = 30.0,
                    default_other: float = 18.0,
                    height_per_level: float = 3.0) -> float:
    if not isinstance(tags, dict):
        return default_other
    h = tags.get("height")
    if h is not None:
        try:
            s = str(h).strip().rstrip("m").strip()
            return float(s)
        except (TypeError, ValueError):
            pass
    levels = tags.get("building:levels")
    if levels is not None:
        try:
            return float(levels) * height_per_level
        except (TypeError, ValueError):
            pass
    btype = str(tags.get("building", "")).lower()
    if btype in {"residential", "apartments", "house", "yes", "dormitory",
                 "hotel"}:
        return default_residential
    if btype in {"commercial", "office", "retail", "industrial", "warehouse",
                 "supermarket"}:
        return default_commercial
    return default_other


def rasterize_heights(buildings_geojson: Path, grid,
                      lonlat_to_xy_m,
                      bbox_lonlat: Tuple[float, float, float, float],
                      default_residential: float = 12.0,
                      default_commercial: float = 30.0,
                      default_other: float = 18.0,
                      height_per_level: float = 3.0,
                      verbose: bool = True) -> np.ndarray:
    """Return a (nx, ny) float32 max-height array aligned with `grid`.

    `lonlat_to_xy_m` must match the coordinate frame used elsewhere (the same
    function passed to aura.network.blocks.site_blocks for consistency).
    """
    with open(buildings_geojson) as f:
        gj = json.load(f)
    nx, ny = grid.shape2
    height_mask = np.zeros((nx, ny), dtype=np.float32)
    n_used = 0; n_total = 0
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        tags = props.get("tags") if isinstance(props, dict) else None
        if not isinstance(tags, dict):
            tags = props if isinstance(props, dict) else {}
        h_m = _resolve_height(
            tags,
            default_residential=default_residential,
            default_commercial=default_commercial,
            default_other=default_other,
            height_per_level=height_per_level,
        )
        for poly_coords in _iter_polygons(feat):
            n_total += 1
            outer = poly_coords[0]
            if len(outer) < 4:
                continue
            # bbox cull
            lon_min = min(c[0] for c in outer)
            lon_max = max(c[0] for c in outer)
            lat_min = min(c[1] for c in outer)
            lat_max = max(c[1] for c in outer)
            if (lon_max < bbox_lonlat[0] or lon_min > bbox_lonlat[2]
                    or lat_max < bbox_lonlat[1] or lat_min > bbox_lonlat[3]):
                continue
            outer_xy = np.asarray([lonlat_to_xy_m(lon, lat)
                                   for (lon, lat) in outer], dtype=np.float64)
            x_lo, y_lo = outer_xy.min(axis=0)
            x_hi, y_hi = outer_xy.max(axis=0)
            ix_lo = max(0, int((x_lo - grid.origin_xy[0]) / grid.dx) - 1)
            ix_hi = min(nx, int((x_hi - grid.origin_xy[0]) / grid.dx) + 2)
            iy_lo = max(0, int((y_lo - grid.origin_xy[1]) / grid.dy) - 1)
            iy_hi = min(ny, int((y_hi - grid.origin_xy[1]) / grid.dy) + 2)
            if ix_hi <= ix_lo or iy_hi <= iy_lo:
                continue
            ix_arr = np.arange(ix_lo, ix_hi)
            iy_arr = np.arange(iy_lo, iy_hi)
            gx, gy = np.meshgrid(ix_arr, iy_arr, indexing="ij")
            cx = grid.origin_xy[0] + (gx + 0.5) * grid.dx
            cy = grid.origin_xy[1] + (gy + 0.5) * grid.dy
            pts = np.stack([cx.ravel(), cy.ravel()], axis=1)
            mpath = MplPath(outer_xy)
            inside = mpath.contains_points(pts).reshape(gx.shape)
            if not inside.any():
                continue
            local = height_mask[ix_lo:ix_hi, iy_lo:iy_hi]
            stamp = inside.astype(np.float32) * float(h_m)
            np.maximum(local, stamp, out=local)
            height_mask[ix_lo:ix_hi, iy_lo:iy_hi] = local
            n_used += 1
    if verbose:
        nonzero = int((height_mask > 0).sum())
        cov = nonzero / float(nx * ny) * 100.0
        peak = float(height_mask.max())
        print(f"[footprints-osm] used {n_used}/{n_total} polygons; "
              f"coverage {nonzero} cells ({cov:.1f}% of grid); "
              f"peak height {peak:.1f} m")
    return height_mask
