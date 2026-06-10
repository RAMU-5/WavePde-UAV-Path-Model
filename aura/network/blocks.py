"""Block-based station siting.

A "block" (街区) is a face of the OSM road network's planar graph: the area
enclosed by roads. We polygonize the road LineStrings (after noding them with
shapely's unary_union), filter out water-dominated and tiny slivers, and place
one station at each block's representative point.

Capacity for a block scales with the number of buildings whose centroid lies
inside the block (proxy for residential/commercial demand). The plan's
existing pop_voronoi field is repurposed to hold building_count for downstream
re-balancing logic — that field is only used as a relative weight, so the
substitution is transparent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box, shape
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from aura.schemas import Station


@dataclass
class BlockAttrs:
    block_id: int
    polygon_lonlat: Polygon
    centroid_lonlat: Tuple[float, float]
    area_m2: float
    n_buildings: int
    building_area_m2: float
    is_water: bool


def _iter_lines(feature) -> Iterable[List[List[float]]]:
    g = feature.get("geometry") or {}
    t = g.get("type")
    if t == "LineString":
        yield g["coordinates"]
    elif t == "MultiLineString":
        for ls in g["coordinates"]:
            yield ls


def _iter_polygons(feature) -> Iterable[List[List[float]]]:
    g = feature.get("geometry") or {}
    t = g.get("type")
    if t == "Polygon":
        yield g["coordinates"]
    elif t == "MultiPolygon":
        for poly in g["coordinates"]:
            yield poly


def _local_meters_scale(lat0: float) -> Tuple[float, float]:
    """Approximate lon/lat -> meters scale (for areas, not absolute positions)."""
    return (111_000.0 * float(np.cos(np.deg2rad(lat0))), 111_000.0)


def _polygon_area_m2(poly: Polygon, lat0: float) -> float:
    sx, sy = _local_meters_scale(lat0)
    # area in degree^2 -> approx meters^2 via lon-scale * lat-scale.
    return float(poly.area) * sx * sy


def polygonize_roads(roads_path: Path,
                     bbox_lonlat: Tuple[float, float, float, float]
                     ) -> List[Polygon]:
    """Read roads.geojson, clip to bbox, node + polygonize."""
    with open(roads_path) as f:
        gj = json.load(f)
    bbox_poly = box(*bbox_lonlat)
    lines: List[LineString] = []
    for feat in gj.get("features", []):
        for raw in _iter_lines(feat):
            ls = LineString(raw)
            inter = ls.intersection(bbox_poly)
            if inter.is_empty:
                continue
            if inter.geom_type == "LineString":
                lines.append(inter)
            elif inter.geom_type == "MultiLineString":
                for sub in inter.geoms:
                    lines.append(sub)
    # close outer faces by adding the bbox boundary itself
    lines.append(LineString(list(bbox_poly.exterior.coords)))
    print(f"[blocks] {len(lines)} road segments inside bbox; noding...")
    noded = unary_union(lines)
    print(f"[blocks] polygonizing...")
    polys = list(polygonize(noded))
    print(f"[blocks] {len(polys)} raw blocks")
    return polys


def annotate_blocks(blocks: List[Polygon],
                    buildings_path: Path,
                    water_path: Path | None,
                    lat0: float) -> List[BlockAttrs]:
    """Compute area, building counts, and water flag for each block."""
    bldg_centroids: List[Tuple[float, float]] = []
    bldg_areas: List[float] = []
    if buildings_path.exists():
        with open(buildings_path) as f:
            gj = json.load(f)
        sx, sy = _local_meters_scale(lat0)
        for feat in gj.get("features", []):
            for poly_coords in _iter_polygons(feat):
                outer = poly_coords[0]
                if len(outer) < 4:
                    continue
                p = Polygon(outer)
                if not p.is_valid or p.is_empty:
                    continue
                c = p.centroid
                bldg_centroids.append((c.x, c.y))
                bldg_areas.append(float(p.area) * sx * sy)
    print(f"[blocks] {len(bldg_centroids)} buildings indexed")

    # Build STRtree over building centroids represented as Points.
    from shapely.geometry import Point
    pts = [Point(x, y) for (x, y) in bldg_centroids]
    tree = STRtree(pts) if pts else None

    water_polys: List[Polygon] = []
    if water_path is not None and water_path.exists():
        with open(water_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            for poly_coords in _iter_polygons(feat):
                if len(poly_coords[0]) < 4:
                    continue
                p = Polygon(poly_coords[0])
                if p.is_valid and not p.is_empty:
                    water_polys.append(p)
    water_union = unary_union(water_polys) if water_polys else None
    print(f"[blocks] {len(water_polys)} water polygons")

    out: List[BlockAttrs] = []
    for i, blk in enumerate(blocks):
        area_m2 = _polygon_area_m2(blk, lat0)
        is_water = False
        if water_union is not None:
            inter = blk.intersection(water_union)
            if not inter.is_empty:
                ratio = inter.area / max(blk.area, 1e-12)
                if ratio > 0.5:
                    is_water = True
        n_b = 0
        a_b = 0.0
        if tree is not None:
            cand_idx = tree.query(blk)
            for j in cand_idx:
                if blk.contains(pts[j]):
                    n_b += 1
                    a_b += bldg_areas[j]
        rep = blk.representative_point()
        out.append(BlockAttrs(
            block_id=i,
            polygon_lonlat=blk,
            centroid_lonlat=(rep.x, rep.y),
            area_m2=area_m2,
            n_buildings=n_b,
            building_area_m2=a_b,
            is_water=is_water,
        ))
    return out


def filter_blocks(attrs: List[BlockAttrs],
                  min_area_m2: float = 2000.0,
                  min_buildings: int = 1) -> List[BlockAttrs]:
    """Drop slivers, water blocks, and empty (no-building) blocks."""
    out = []
    for a in attrs:
        if a.is_water:
            continue
        if a.area_m2 < min_area_m2:
            continue
        if a.n_buildings < min_buildings:
            continue
        out.append(a)
    return out


def block_to_stations(attrs: List[BlockAttrs],
                      lonlat_to_xy_m,
                      cap_min: int = 4, cap_max: int = 16,
                      buildings_per_pad: int = 10) -> List[Station]:
    """Convert filtered block attrs into Station dataclasses."""
    out: List[Station] = []
    for sid, a in enumerate(attrs):
        x, y = lonlat_to_xy_m(*a.centroid_lonlat)
        cap = int(np.clip(round(a.n_buildings / max(1, buildings_per_pad)),
                          cap_min, cap_max))
        out.append(Station(
            id=sid,
            xy_m=(float(x), float(y)),
            pop_voronoi=float(a.n_buildings),
            capacity_pads=cap,
        ))
    return out


def thin_blocks_kmedoids(attrs: List[BlockAttrs],
                         lonlat_to_xy_m,
                         n_target: int,
                         seed: int = 0) -> List[BlockAttrs]:
    """Pick `n_target` representative blocks via building-count-weighted k-medoids.

    Returns a subset of `attrs` ordered by k-medoids medoid index. Intended for
    the case where polygonization yields more blocks than the user wants.
    """
    if len(attrs) <= n_target:
        return attrs
    pts = np.asarray([
        lonlat_to_xy_m(*a.centroid_lonlat) for a in attrs
    ], dtype=np.float64)
    weights = np.asarray([max(1.0, a.n_buildings) for a in attrs],
                         dtype=np.float64)
    from aura.network.siting import weighted_kmedoids
    medoids, _ = weighted_kmedoids(pts, weights, n_target, seed=seed)
    return [attrs[int(i)] for i in medoids]


def site_blocks(roads_path: Path, buildings_path: Path,
                water_path: Path | None,
                bbox_lonlat: Tuple[float, float, float, float],
                lonlat_to_xy_m,
                min_area_m2: float = 2000.0,
                min_buildings: int = 1,
                cap_min: int = 4, cap_max: int = 16,
                buildings_per_pad: int = 10,
                n_stations_target: int | None = None,
                seed: int = 0,
                ) -> Tuple[List[Station], List[BlockAttrs], List[BlockAttrs]]:
    """End-to-end block-based siting.

    Returns (stations, station_block_attrs, all_block_attrs).
    `all_block_attrs` keeps every kept block (for visualization), while
    `station_block_attrs` is the subset that actually got a station — equal to
    `all` if `n_stations_target is None or n_stations_target >= len(kept)`,
    otherwise a k-medoids subset.
    """
    raw_blocks = polygonize_roads(roads_path, bbox_lonlat)
    lat0 = 0.5 * (bbox_lonlat[1] + bbox_lonlat[3])
    attrs = annotate_blocks(raw_blocks, buildings_path, water_path, lat0)
    kept = filter_blocks(attrs, min_area_m2=min_area_m2,
                         min_buildings=min_buildings)
    print(f"[blocks] kept {len(kept)} / {len(attrs)} blocks "
          f"(min_area={min_area_m2:.0f} m^2, min_buildings={min_buildings})")
    if n_stations_target is not None and n_stations_target < len(kept):
        chosen = thin_blocks_kmedoids(kept, lonlat_to_xy_m, n_stations_target,
                                      seed=seed)
        print(f"[blocks] thinned to {len(chosen)} via weighted k-medoids "
              f"(target={n_stations_target})")
    else:
        chosen = kept
    stations = block_to_stations(chosen, lonlat_to_xy_m,
                                 cap_min=cap_min, cap_max=cap_max,
                                 buildings_per_pad=buildings_per_pad)
    return stations, chosen, kept
