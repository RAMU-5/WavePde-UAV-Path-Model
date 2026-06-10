"""Hazard injection: scripted cylindrical no-fly events triggering reroute."""

from __future__ import annotations

from typing import List

import numpy as np

from aura.geom.noflyzone import CylinderZone, NoFlyRegistry
from aura.schemas import HazardEvent


def schedule_to_events(schedule: list[dict],
                       lonlat_to_xy_m,
                       z_min_m: float = 0.0,
                       z_max_m: float = 200.0) -> List[CylinderZone]:
    """Convert YAML hazard schedule entries into CylinderZone objects.

    `lonlat_to_xy_m` is a callable (lon, lat) -> (x, y) projecting to grid CRS.
    """
    out: List[CylinderZone] = []
    for h in schedule:
        x, y = lonlat_to_xy_m(h["lon"], h["lat"])
        out.append(CylinderZone(
            cx=float(x), cy=float(y),
            radius_m=float(h["radius_m"]),
            z_min_m=z_min_m, z_max_m=z_max_m,
            born_at_s=float(h["t_s"]),
            expires_at_s=float(h["t_s"]) + 3.0 * float(h["tau_s"]),
        ))
    return out


def step_registry(registry: NoFlyRegistry, scheduled: List[CylinderZone],
                  fired_idx: set, t_s: float) -> List[CylinderZone]:
    """Activate any cylinder whose born_at <= t and not fired yet.

    Returns the list of zones newly activated this tick (callers use this list
    to trigger `aura.fmm.incremental.resolve_affected`).
    """
    fresh: List[CylinderZone] = []
    for i, z in enumerate(scheduled):
        if i in fired_idx:
            continue
        if z.born_at_s <= t_s:
            registry.add(z)
            fired_idx.add(i)
            fresh.append(z)
    registry.remove_expired(t_s)
    return fresh
