"""Per-station feature vectors driving order origin/destination weights.

Inputs:
  - shenzhen-osm/<sub>/pois.geojson — POI features with `tags` dict
  - station list with xy_m in projected meters
  - height_mask (nx, ny) — building footprint heights, used for residential proxy
  - bbox_lonlat used to project pois to the same UTM frame

Outputs (per station, dict):
  - n_food_pois     : int
  - n_shop_pois     : float (mall/super-market/convenience get 3x weight)
  - n_residential_units : float (floor area within Voronoi cell × 0.7 / 80 m²/unit)
  - is_hub          : 0 or 1, top-N stations by (n_shop_pois + 2 × building_area_m2)
  - n_pois_total    : int (debug)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


FOOD_AMENITIES = {"restaurant", "cafe", "fast_food", "food_court", "ice_cream"}
SHOP_BAKERY = {"bakery"}
SHOP_HEAVY = {"mall", "supermarket", "department_store", "convenience"}
RESIDENTIAL_FRACTION = 0.7   # Nanshan land-use proxy
UNIT_AREA_M2 = 80.0


@dataclass
class StationFeatures:
    n_food_pois: int = 0
    n_shop_pois: float = 0.0
    n_residential_units: float = 0.0
    n_pois_total: int = 0
    building_area_m2: float = 0.0
    is_hub: int = 0


def _classify_poi(tags: dict) -> Tuple[int, float]:
    """Return (food_count_increment, shop_weight_increment) for one POI."""
    if not isinstance(tags, dict):
        return (0, 0.0)
    amenity = str(tags.get("amenity", "")).lower()
    shop = str(tags.get("shop", "")).lower()
    food = 1 if amenity in FOOD_AMENITIES or shop in SHOP_BAKERY else 0
    shop_w = 0.0
    if shop:
        shop_w = 3.0 if shop in SHOP_HEAVY else 1.0
    # treat takeaway-tagged amenities as food signal too
    if tags.get("takeaway") in {"yes", "only"}:
        food = max(food, 1)
    return (food, shop_w)


def _iter_points(features) -> Iterable[Tuple[float, float, dict]]:
    for feat in features:
        g = feat.get("geometry") or {}
        gtype = g.get("type")
        if gtype == "Point":
            x, y = g["coordinates"][:2]
            props = feat.get("properties", {}) or {}
            tags = props.get("tags") if isinstance(props, dict) else None
            if not isinstance(tags, dict):
                tags = props if isinstance(props, dict) else {}
            yield (float(x), float(y), tags)


def compute_features(stations,
                     pois_geojson: Path,
                     height_mask: np.ndarray,
                     grid,
                     voronoi_owner: np.ndarray,
                     lonlat_to_xy_m,
                     bbox_lonlat: Tuple[float, float, float, float],
                     poi_radius_m: float = 250.0,
                     hub_top_n: int = 8) -> List[StationFeatures]:
    """Build per-station features. Returns list aligned with `stations`.

    Side-effect: also writes the resulting dict into `station.features` so
    callers can save() the assets without an extra step.
    """
    K = len(stations)
    sxy = np.asarray([s.xy_m for s in stations], dtype=np.float64)
    feats = [StationFeatures() for _ in range(K)]

    # POIs -> nearest station by (sxy) within radius_m.
    if pois_geojson.exists():
        with open(pois_geojson) as f:
            gj = json.load(f)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(sxy)
        except Exception:
            tree = None
        n_in = 0; n_food = 0; n_shop = 0
        for lon, lat, tags in _iter_points(gj.get("features", [])):
            if not (bbox_lonlat[0] <= lon <= bbox_lonlat[2]
                    and bbox_lonlat[1] <= lat <= bbox_lonlat[3]):
                continue
            x, y = lonlat_to_xy_m(lon, lat)
            n_in += 1
            if tree is not None:
                d, idx = tree.query([x, y], k=1)
                if d > poi_radius_m:
                    continue
                sid = int(idx)
            else:
                d2 = (sxy[:, 0] - x) ** 2 + (sxy[:, 1] - y) ** 2
                sid = int(np.argmin(d2))
                if d2[sid] > poi_radius_m ** 2:
                    continue
            food_inc, shop_inc = _classify_poi(tags)
            feats[sid].n_pois_total += 1
            feats[sid].n_food_pois += food_inc
            feats[sid].n_shop_pois += shop_inc
            if food_inc:
                n_food += 1
            if shop_inc > 0:
                n_shop += 1
        print(f"[poi_features] {n_in} POIs in bbox; food={n_food} shop={n_shop}")

    # Residential units & building area per station Voronoi cell.
    if height_mask is not None and voronoi_owner is not None:
        cell_area = float(grid.dx * grid.dy)
        # Floor area approx = footprint area * (height / floor_height). Use 3 m/floor.
        floors = np.maximum(0.0, height_mask / 3.0)
        floor_area_per_cell = (height_mask > 0).astype(np.float32) * floors * cell_area
        building_area_per_cell = (height_mask > 0).astype(np.float32) * cell_area
        for sid in range(K):
            mask = voronoi_owner == sid
            if not mask.any():
                continue
            farea = float(floor_area_per_cell[mask].sum())
            barea = float(building_area_per_cell[mask].sum())
            feats[sid].n_residential_units = farea * RESIDENTIAL_FRACTION / UNIT_AREA_M2
            feats[sid].building_area_m2 = barea

    # Hub flag: top-N by (n_shop_pois + 2 * building_area_m2 / 1e4 [hectare]).
    score = np.asarray([
        f.n_shop_pois + 2.0 * f.building_area_m2 / 1e4 for f in feats
    ])
    hub_idx = np.argsort(-score)[:hub_top_n]
    for i in hub_idx:
        feats[int(i)].is_hub = 1
    print(f"[poi_features] hubs (top {hub_top_n} by shop+area): "
          f"{[int(i) for i in hub_idx]}")

    # Persist into station.features dict so build_assets save() picks it up.
    for s, f in zip(stations, feats):
        s.features = {
            "n_food_pois": int(f.n_food_pois),
            "n_shop_pois": float(f.n_shop_pois),
            "n_residential_units": float(f.n_residential_units),
            "n_pois_total": int(f.n_pois_total),
            "building_area_m2": float(f.building_area_m2),
            "is_hub": int(f.is_hub),
        }
    return feats


def features_to_arrays(stations) -> dict:
    """Stack station.features dicts into per-key numpy arrays."""
    keys = ["n_food_pois", "n_shop_pois", "n_residential_units",
            "is_hub", "building_area_m2"]
    out = {}
    for k in keys:
        out[k] = np.asarray([float((s.features or {}).get(k, 0.0))
                             for s in stations], dtype=np.float64)
    return out
