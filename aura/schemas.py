"""Shared dataclasses passed between layers.

These are intentionally plain (no behavior, no numpy slots) so they serialize
cleanly to pickle/zarr/JSON for assets and trajectory DBs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Station:
    id: int
    xy_m: Tuple[float, float]      # projected meters (UTM 50N)
    pop_voronoi: float             # integral of pop density over Voronoi cell
    capacity_pads: int             # max simultaneous landings/launches
    n_uavs: int = 0
    features: Optional[dict] = None  # POI/footprint features for order routing


@dataclass
class Order:
    id: int
    origin: int                    # station id
    destination: int               # station id
    born_at_s: float
    deadline_s: float
    payload_kg: float
    type: str = "parcel"           # food | urgent_doc | parcel | heavy_cargo
    priority: int = 2              # 0 = highest, lower number = more urgent
    required_class: int = 0        # 0 small, 1 large; UAVs serve >= this class
    status: str = "queued"         # queued | assigned | in_flight | completed | failed
    completed_at_s: Optional[float] = None


@dataclass
class HazardEvent:
    id: int
    born_at_s: float
    cx_m: float
    cy_m: float
    radius_m: float
    tau_s: float                   # exponential decay timescale


@dataclass
class TubeCorridor:
    src: int
    dst: int
    eps_m: float                   # tube radius
    is_trunk: bool


@dataclass
class NominalTrajectory:
    src: int
    dst: int
    xyz: List[Tuple[float, float, float]]  # resampled at trajectory_dt_s
    dt_s: float
    metric_version: int = 0
