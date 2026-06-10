"""Stage 1 asset builder.

Pulls OSM streets, building footprints, and the population raster,
constructs the SimGrid, sites stations, and writes a Stage1Assets bundle.

Usage:
    python -m aura.cli.build_assets --config aura/config/stage1.yaml --out runs/v1/

If `--synthetic` is passed, ingest steps are replaced with offline synthetic
substitutes (useful for environments without the geo stack installed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from aura.geom.grid import SimGrid
from aura.io.assets import Stage1Assets, compute_version_hash, save


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="aura/config/stage1.yaml")
    p.add_argument("--out", default="runs/v1/")
    p.add_argument("--synthetic", action="store_true",
                   help="Use offline synthetic data; no osmnx/rasterio needed.")
    p.add_argument("--osm-dir", default=None,
                   help="Path to a shenzhen-osm/<sub>/ directory of GeoJSONs. "
                        "When set, candidates and footprints are read from "
                        "these files (no osmnx required).")
    p.add_argument("--siting", default="auto", choices=["auto", "blocks", "kmedoids"],
                   help="Station siting strategy. 'auto' = blocks if --osm-dir "
                        "is set, else kmedoids.")
    p.add_argument("--min-block-area-m2", type=float, default=2000.0)
    p.add_argument("--min-buildings", type=int, default=1)
    p.add_argument("--buildings-per-pad", type=int, default=3)
    p.add_argument("--n-stations", type=int, default=None,
                   help="Target station count. If set and #blocks > target, "
                        "thin via building-count-weighted k-medoids.")
    return p.parse_args(argv)


def _bbox_to_utm(bbox_wgs84, target_crs="EPSG:32650"):
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        x0, y0 = tr.transform(bbox_wgs84[0], bbox_wgs84[1])
        x1, y1 = tr.transform(bbox_wgs84[2], bbox_wgs84[3])
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    except Exception:
        # Approx Mercator scaling for tests when pyproj is missing.
        scale = 111_000.0
        x0 = bbox_wgs84[0] * scale * np.cos(np.deg2rad(bbox_wgs84[1]))
        y0 = bbox_wgs84[1] * scale
        x1 = bbox_wgs84[2] * scale * np.cos(np.deg2rad(bbox_wgs84[1]))
        y1 = bbox_wgs84[3] * scale
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    bbox = cfg["region"]["bbox_wgs84"]
    target_crs = cfg["region"]["crs_projected"]

    x0, y0, x1, y1 = _bbox_to_utm(bbox, target_crs)
    grid = SimGrid.from_bbox_meters(
        x0, y0, x1, y1,
        dx=cfg["grid"]["dx_m"], dy=cfg["grid"]["dy_m"],
        z_min=cfg["region"]["z_min_m"], z_max=cfg["region"]["z_max_m"],
        dz=cfg["grid"]["dz_m"], crs=target_crs,
    )
    print(f"[build_assets] grid {grid.shape3} ({grid.size_xy[0]/1000:.1f}x"
          f"{grid.size_xy[1]/1000:.1f} km, {grid.size_xy[0]*grid.size_xy[1]/1e6:.2f} km^2)")

    # ----- candidates --------------------------------------------------------
    if args.synthetic:
        rng = np.random.default_rng(cfg.get("seed", 0))
        n_cand = cfg["network"]["candidate_pool_target"]
        candidates = rng.uniform(
            [grid.origin_xy[0], grid.origin_xy[1]],
            [grid.origin_xy[0] + grid.size_xy[0], grid.origin_xy[1] + grid.size_xy[1]],
            size=(n_cand, 2),
        ).astype(np.float64)
    else:
        from aura.ingest.osm import build_candidates
        sc = build_candidates(
            tuple(bbox),
            network_type=cfg["data"]["osm"]["network_type"],
            dedupe_radius_m=cfg["data"]["osm"]["candidate_dedupe_radius_m"],
            cache_path=cfg["data"]["osm"].get("cache_path"),
            target_crs=target_crs,
        )
        candidates = sc.xy_m
    print(f"[build_assets] {len(candidates)} candidates")

    # ----- population --------------------------------------------------------
    pop_cfg = cfg["data"]["population"]
    if args.synthetic or not Path(pop_cfg.get("worldpop_path", "")).exists():
        from aura.ingest.pop import synth_pop
        pr = synth_pop(grid, mean=5.0, hotspots=8, hotspot_strength=120.0,
                       hotspot_sigma_m=600.0, seed=cfg.get("seed", 0))
        print(f"[build_assets] using synthetic pop (sum={pr.density.sum():.0f})")
    else:
        from aura.ingest.pop import load_worldpop
        pr = load_worldpop(pop_cfg["worldpop_path"], grid)
        print(f"[build_assets] worldpop sum={pr.density.sum():.0f}")

    # ----- footprints --------------------------------------------------------
    if args.osm_dir:
        from aura.ingest.footprints_osm_json import rasterize_heights
        from pathlib import Path as _P
        osm_dir_p = _P(args.osm_dir)
        lat0 = 0.5 * (bbox[1] + bbox[3])
        scale_x = 111_000.0 * float(np.cos(np.deg2rad(lat0)))
        scale_y = 111_000.0
        ref_lon = bbox[0]
        ref_lat = bbox[1]
        def _l2m_h(lon, lat):
            x = grid.origin_xy[0] + (lon - ref_lon) * scale_x
            y = grid.origin_xy[1] + (lat - ref_lat) * scale_y
            return (x, y)
        height_mask = rasterize_heights(
            osm_dir_p / "buildings.geojson", grid, _l2m_h, tuple(bbox),
            default_residential=cfg["data"]["footprints"]["fallback_height_residential_m"],
            default_commercial=cfg["data"]["footprints"]["fallback_height_commercial_m"],
            height_per_level=cfg["data"]["footprints"]["height_per_level_m"],
        )
        footprint_path = str(osm_dir_p / "buildings.geojson")
    elif args.synthetic:
        height_mask = np.zeros(grid.shape2, dtype=np.float32)
        rng = np.random.default_rng(cfg.get("seed", 0) + 1)
        n_blocks = 100
        for _ in range(n_blocks):
            cx = rng.integers(0, grid.nx)
            cy = rng.integers(0, grid.ny)
            r = rng.integers(2, 6)
            h = float(rng.uniform(15, 45))
            i_lo = max(0, cx - r); i_hi = min(grid.nx, cx + r + 1)
            j_lo = max(0, cy - r); j_hi = min(grid.ny, cy + r + 1)
            height_mask[i_lo:i_hi, j_lo:j_hi] = np.maximum(
                height_mask[i_lo:i_hi, j_lo:j_hi], h)
        footprint_path = str(out_dir / "footprints_synth.pkl")
    else:
        from aura.ingest.footprints import (
            fetch_osm_footprints, rasterize_height_mask, save_bundle,
        )
        bundle = fetch_osm_footprints(tuple(bbox), target_crs=target_crs,
                                      height_per_level=cfg["data"]["footprints"]["height_per_level_m"],
                                      default_residential=cfg["data"]["footprints"]["fallback_height_residential_m"],
                                      default_commercial=cfg["data"]["footprints"]["fallback_height_commercial_m"])
        height_mask = rasterize_height_mask(bundle, grid)
        footprint_path = str(out_dir / "footprints.pkl")
        save_bundle(bundle, footprint_path)

    # ----- network -----------------------------------------------------------
    siting = args.siting
    if siting == "auto":
        siting = "blocks" if args.osm_dir else "kmedoids"
    from aura.network.siting import build_stations, voronoi_owner

    if siting == "blocks":
        from pathlib import Path as _P
        from aura.network.blocks import site_blocks
        osm_dir = _P(args.osm_dir) if args.osm_dir else _P(
            "/leonardo_scratch/large/userexternal/yli00001/Aura_Project/"
            "Levistone/shenzhen-osm/nanshan")
        lat0 = 0.5 * (bbox[1] + bbox[3])
        scale_x = 111_000.0 * float(np.cos(np.deg2rad(lat0)))
        scale_y = 111_000.0
        # Use the grid origin as the local frame so block centroids land on
        # the correct simulation cells.
        ref_lon = bbox[0]
        ref_lat = bbox[1]
        def _l2m(lon, lat):
            x = grid.origin_xy[0] + (lon - ref_lon) * scale_x
            y = grid.origin_xy[1] + (lat - ref_lat) * scale_y
            return (x, y)
        n_target = args.n_stations
        if n_target is None:
            n_target = cfg["network"].get("n_stations")
        stations, _, _ = site_blocks(
            osm_dir / "roads.geojson",
            osm_dir / "buildings.geojson",
            osm_dir / "water.geojson",
            tuple(bbox), _l2m,
            min_area_m2=args.min_block_area_m2,
            min_buildings=args.min_buildings,
            cap_min=cfg["network"]["capacity"]["cap_min"],
            cap_max=cfg["network"]["capacity"]["cap_max"],
            buildings_per_pad=args.buildings_per_pad,
            n_stations_target=n_target,
            seed=cfg.get("seed", 0),
        )
        station_xy = np.asarray([s.xy_m for s in stations])
        owner = voronoi_owner(grid, station_xy)
    else:
        stations, owner = build_stations(
            candidates, pr.density, grid,
            n_stations=cfg["network"]["n_stations"],
            pop_per_unit=cfg["network"]["capacity"]["population_per_unit"],
            cap_min=cfg["network"]["capacity"]["cap_min"],
            cap_max=cfg["network"]["capacity"]["cap_max"],
            seed=cfg.get("seed", 0),
        )
    print(f"[build_assets] siting={siting} -> {len(stations)} stations")

    # ----- POI features (writes into station.features) -----------------------
    if args.osm_dir:
        from aura.network.poi_features import compute_features
        from pathlib import Path as _P
        osm_dir_p = _P(args.osm_dir)
        lat0 = 0.5 * (bbox[1] + bbox[3])
        scx = 111_000.0 * float(np.cos(np.deg2rad(lat0)))
        scy = 111_000.0
        ref_lon = bbox[0]
        ref_lat = bbox[1]
        def _l2m_p(lon, lat):
            return (grid.origin_xy[0] + (lon - ref_lon) * scx,
                    grid.origin_xy[1] + (lat - ref_lat) * scy)
        compute_features(
            stations,
            osm_dir_p / "pois.geojson",
            height_mask, grid, owner,
            _l2m_p, tuple(bbox),
            poi_radius_m=cfg.get("network", {}).get("poi_radius_m", 250.0),
            hub_top_n=cfg.get("network", {}).get("hub_top_n", 8),
        )
    else:
        # Synthetic: no POIs; still seed minimal features so OrderBook works.
        for s in stations:
            s.features = {
                "n_food_pois": 1, "n_shop_pois": 1.0,
                "n_residential_units": float(s.pop_voronoi),
                "n_pois_total": 0, "building_area_m2": 0.0, "is_hub": 0,
            }
        # Random hubs in synthetic mode
        rng_h = np.random.default_rng(cfg.get("seed", 0) + 7)
        hub_ids = rng_h.choice(len(stations), size=min(8, len(stations)), replace=False)
        for hid in hub_ids:
            stations[int(hid)].features["is_hub"] = 1

    version_payload = {
        "config": cfg, "grid_shape": grid.shape3,
        "n_candidates": int(len(candidates)),
        "n_stations": len(stations),
        "pop_source": pr.source,
    }
    version_hash = compute_version_hash(version_payload)
    assets = Stage1Assets(
        grid=grid,
        candidates_xy_m=candidates,
        population_density=pr.density,
        population_source=pr.source,
        footprint_path=footprint_path,
        height_mask=height_mask,
        stations=stations,
        voronoi_owner=owner,
        version_hash=version_hash,
        config=cfg,
    )
    save(assets, out_dir)
    print(f"[build_assets] saved -> {out_dir} (version {version_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
