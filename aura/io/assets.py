"""Stage1Assets: a versioned, pickled bundle of all ingest outputs.

Members:
  - grid: SimGrid
  - candidates: StreetCentroids (street midpoints in UTM)
  - footprints: FootprintBundle
  - pop: PopulationRaster
  - stations: list[Station]
  - building_height_mask: (nx, ny) max footprint height per cell
  - version_hash: sha256 of all input parameters

The hash is computed over the asset config dict so a Stage1Assets instance can
be detected as stale if any upstream choice changed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np

from aura.geom.grid import SimGrid
from aura.schemas import Station


@dataclass
class Stage1Assets:
    grid: SimGrid
    candidates_xy_m: np.ndarray              # (N, 2)
    population_density: np.ndarray            # (nx, ny)
    population_source: str
    footprint_path: str                       # pickle path; payload too large for in-memory dataclass
    height_mask: np.ndarray                   # (nx, ny) float32 — max building height per cell
    stations: List[Station] = field(default_factory=list)
    voronoi_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))  # (nx, ny) int32
    version_hash: str = ""
    config: dict = field(default_factory=dict)


def compute_version_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save(assets: Stage1Assets, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "rasters.npz",
        candidates_xy_m=assets.candidates_xy_m,
        population_density=assets.population_density,
        height_mask=assets.height_mask,
        voronoi_owner=assets.voronoi_owner,
    )
    meta = {
        "grid": {
            "origin_xy": assets.grid.origin_xy,
            "size_xy": assets.grid.size_xy,
            "dx": assets.grid.dx, "dy": assets.grid.dy,
            "z_min": assets.grid.z_min, "z_max": assets.grid.z_max, "dz": assets.grid.dz,
            "crs": assets.grid.crs,
        },
        "population_source": assets.population_source,
        "footprint_path": assets.footprint_path,
        "stations": [
            {"id": s.id, "xy_m": list(s.xy_m), "pop_voronoi": s.pop_voronoi,
             "capacity_pads": s.capacity_pads, "n_uavs": s.n_uavs,
             "features": s.features}
            for s in assets.stations
        ],
        "version_hash": assets.version_hash,
        "config": assets.config,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load(out_dir: str | Path) -> Stage1Assets:
    out_dir = Path(out_dir)
    with open(out_dir / "meta.json") as f:
        meta = json.load(f)
    grid = SimGrid(
        origin_xy=tuple(meta["grid"]["origin_xy"]),
        size_xy=tuple(meta["grid"]["size_xy"]),
        dx=meta["grid"]["dx"], dy=meta["grid"]["dy"],
        z_min=meta["grid"]["z_min"], z_max=meta["grid"]["z_max"], dz=meta["grid"]["dz"],
        crs=meta["grid"]["crs"],
    )
    z = np.load(out_dir / "rasters.npz", allow_pickle=False)
    stations = [
        Station(id=s["id"], xy_m=tuple(s["xy_m"]),
                pop_voronoi=s["pop_voronoi"],
                capacity_pads=s["capacity_pads"],
                n_uavs=s["n_uavs"],
                features=s.get("features"))
        for s in meta["stations"]
    ]
    return Stage1Assets(
        grid=grid,
        candidates_xy_m=z["candidates_xy_m"],
        population_density=z["population_density"],
        population_source=meta["population_source"],
        footprint_path=meta["footprint_path"],
        height_mask=z["height_mask"],
        stations=stations,
        voronoi_owner=z["voronoi_owner"],
        version_hash=meta["version_hash"],
        config=meta["config"],
    )
