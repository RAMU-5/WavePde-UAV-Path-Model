"""Dynamic no-fly registry — hazards, NOTAMs, weather-driven closures.

In Stage 1 the only producer is `aura.missions.hazard`. Consumers are
`aura.cost.metric` (rebuilds rho mask) and `aura.fmm.incremental` (computes
affected source set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class CylinderZone:
    cx: float
    cy: float
    radius_m: float
    z_min_m: float = 0.0
    z_max_m: float = 1000.0
    born_at_s: float = 0.0
    expires_at_s: float | None = None

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        dx = x - self.cx
        dy = y - self.cy
        in_xy = dx * dx + dy * dy <= self.radius_m ** 2
        in_z = (z >= self.z_min_m) & (z <= self.z_max_m)
        return in_xy & in_z


@dataclass
class NoFlyRegistry:
    zones: List[CylinderZone] = field(default_factory=list)
    version: int = 0

    def add(self, zone: CylinderZone) -> None:
        self.zones.append(zone)
        self.version += 1

    def remove_expired(self, t: float) -> int:
        before = len(self.zones)
        self.zones = [z for z in self.zones
                      if z.expires_at_s is None or z.expires_at_s > t]
        if len(self.zones) != before:
            self.version += 1
        return before - len(self.zones)

    def mask(self, xyz: np.ndarray, t: float) -> np.ndarray:
        """Return boolean mask (broadcast on xyz[..., 3]) of points inside ANY active zone."""
        out = np.zeros(xyz.shape[:-1], dtype=bool)
        for z in self.zones:
            if z.expires_at_s is not None and z.expires_at_s <= t:
                continue
            out |= z.contains(xyz)
        return out
