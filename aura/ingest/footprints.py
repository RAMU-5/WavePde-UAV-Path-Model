"""Building footprints + heights for Nanshan, projected to EPSG:32650.

Primary source: Microsoft Global ML Building Footprints. OSM is the fallback.
Height inference order: explicit `height` tag -> `building:levels` x 3.0 m ->
class-conditional default (residential 12 m, commercial 30 m).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BuildingMeta:
    """Mirrors UrbanGen's BuildingMeta but trimmed for routing-only use."""
    id: int
    footprint_xy: np.ndarray  # (V, 2) UTM polygon vertices, closed ring
    height_m: float
    centroid_xy: Tuple[float, float]
    bldg_class: str = "unknown"


@dataclass
class FootprintBundle:
    buildings: List[BuildingMeta]
    crs: str = "EPSG:32650"
    source: str = "ms_global"

    def height_max(self) -> float:
        return max((b.height_m for b in self.buildings), default=0.0)


def _resolve_height(tags: dict, default_residential: float = 12.0,
                    default_commercial: float = 30.0,
                    height_per_level: float = 3.0) -> Tuple[float, str]:
    """Return (height_m, class)."""
    htag = tags.get("height")
    if htag is not None:
        try:
            return float(str(htag).rstrip(" m")), tags.get("building", "unknown")
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels is not None:
        try:
            return float(levels) * height_per_level, tags.get("building", "unknown")
        except ValueError:
            pass
    btype = str(tags.get("building", "unknown")).lower()
    if btype in {"residential", "apartments", "house", "yes"}:
        return default_residential, "residential"
    if btype in {"commercial", "office", "retail", "industrial"}:
        return default_commercial, "commercial"
    return default_residential, btype or "unknown"


def fetch_osm_footprints(bbox_wgs84: Tuple[float, float, float, float],
                         target_crs: str = "EPSG:32650",
                         height_per_level: float = 3.0,
                         default_residential: float = 12.0,
                         default_commercial: float = 30.0) -> FootprintBundle:
    import osmnx as ox
    import geopandas as gpd
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    gdf = ox.features.features_from_bbox(
        bbox=(max_lat, min_lat, max_lon, min_lon),
        tags={"building": True},
    )
    gdf = gdf.to_crs(target_crs)
    out: List[BuildingMeta] = []
    for i, row in enumerate(gdf.itertuples(index=False)):
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)
            xs, ys = geom.exterior.xy
            ring = np.stack([np.asarray(xs), np.asarray(ys)], axis=1)
        except Exception:
            continue
        tags = row._asdict() if hasattr(row, "_asdict") else dict(row.__dict__)
        h, cls = _resolve_height(tags, default_residential, default_commercial, height_per_level)
        cx, cy = float(geom.centroid.x), float(geom.centroid.y)
        out.append(BuildingMeta(id=i, footprint_xy=ring, height_m=h,
                                centroid_xy=(cx, cy), bldg_class=cls))
    return FootprintBundle(buildings=out, crs=target_crs, source="osm")


def fetch_ms_global_footprints(bbox_wgs84: Tuple[float, float, float, float],
                               local_csv_gz: Optional[str] = None,
                               target_crs: str = "EPSG:32650",
                               default_height_m: float = 18.0) -> FootprintBundle:
    """MS Global Footprints. The release does not carry heights; assume per-tile default.

    For Stage 1 we accept that MS Global is height-less and use a single
    constant per the config (18 m default for Nanshan mixed urban).
    Override later via building-class GIS join in Stage 2.
    """
    if local_csv_gz is None:
        raise FileNotFoundError(
            "MS Global Footprints CSV.gz not yet downloaded. Set "
            "data.footprints.ms_global_url_template path or fall back to OSM."
        )
    import geopandas as gpd
    from shapely import wkt
    import pandas as pd
    df = pd.read_csv(local_csv_gz, compression="gzip")
    geom_col = "geometry" if "geometry" in df.columns else "WKT"
    geom = df[geom_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat].to_crs(target_crs)
    out: List[BuildingMeta] = []
    for i, row in enumerate(gdf.itertuples(index=False)):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        xs, ys = geom.exterior.xy
        ring = np.stack([np.asarray(xs), np.asarray(ys)], axis=1)
        cx, cy = float(geom.centroid.x), float(geom.centroid.y)
        out.append(BuildingMeta(id=i, footprint_xy=ring, height_m=default_height_m,
                                centroid_xy=(cx, cy), bldg_class="ms_global"))
    return FootprintBundle(buildings=out, crs=target_crs, source="ms_global")


def rasterize_height_mask(bundle: FootprintBundle, grid) -> np.ndarray:
    """Rasterize footprint heights to a (nx, ny) max-height-per-cell array.

    `grid` is an aura.geom.grid.SimGrid. Cells where no footprint covers them
    get 0.0. Used downstream by `aura.cost.metric` to build the obstacle mask
    for the FMM domain.
    """
    from rasterio.features import rasterize
    from shapely.geometry import Polygon
    nx, ny = grid.nx, grid.ny
    transform = _grid_to_affine(grid)
    shapes = []
    for b in bundle.buildings:
        if len(b.footprint_xy) < 3:
            continue
        shapes.append((Polygon(b.footprint_xy), float(b.height_m)))
    if not shapes:
        return np.zeros((nx, ny), dtype=np.float32)
    raster = rasterize(
        shapes,
        out_shape=(ny, nx),  # rasterio uses (rows, cols) = (y, x)
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype="float32",
        merge_alg=__import__("rasterio").enums.MergeAlg.replace,
    )
    return raster.T  # back to (nx, ny)


def _grid_to_affine(grid):
    from rasterio.transform import from_origin
    return from_origin(grid.origin_xy[0], grid.origin_xy[1] + grid.size_xy[1],
                       grid.dx, grid.dy)


def save_bundle(bundle: FootprintBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str | Path) -> FootprintBundle:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
