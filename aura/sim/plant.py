"""Vectorized point-mass UAV plant.

x_dot = v
v_dot = (u + w_wind - gamma v) / m

State arrays are kept on plain numpy so plant.py runs with no extra deps and
fits the layered scheduler. Boundary handling: clamp to grid AABB; out-of-domain
positions are reflected with a 0.5x velocity damping (pattern from
aura_toy_archive/uav.py:integrate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FleetState:
    pos: np.ndarray            # (N, 3)
    vel: np.ndarray            # (N, 3)
    home_station: np.ndarray   # (N,) int32
    state_code: np.ndarray     # (N,) int32: 0 idle, 1 in_flight, 2 returning, 3 charging
    soc: np.ndarray            # (N,) float32 — battery state of charge in [0, 1]
    mission_id: np.ndarray     # (N,) int64 — -1 if idle
    target: np.ndarray         # (N, 3) — current MPC reference / nominal sample
    class_id: np.ndarray = None  # (N,) int8: 0 = small, 1 = large

    @property
    def n(self) -> int:
        return self.pos.shape[0]


# State codes (mirrors aura_toy_archive but extended).
IDLE = 0
IN_FLIGHT = 1
RETURNING = 2
CHARGING = 3
RESPONDING = 4

STATE_NAMES = {IDLE: "idle", IN_FLIGHT: "in_flight",
               RETURNING: "returning", CHARGING: "charging",
               RESPONDING: "responding"}


def make_fleet(stations, n_uavs: int, z_cruise: float = 60.0,
               soc_init: float = 1.0,
               seed: int = 1,
               class_split: tuple | None = None) -> FleetState:
    """Spawn UAVs at home stations, distributed by station capacity_pads.

    `class_split` = (n_small, n_large) optional split for heterogeneous fleet.
    None or sum != n_uavs falls back to all class_id=0 (small).
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray([s.capacity_pads for s in stations], dtype=np.float64)
    weights = weights / weights.sum()
    home_idx = rng.choice(len(stations), size=n_uavs, p=weights)
    pos = np.zeros((n_uavs, 3), dtype=np.float64)
    for i, sid in enumerate(home_idx):
        sx, sy = stations[sid].xy_m
        jit = rng.uniform(-1.0, 1.0, size=2)
        pos[i, 0] = sx + jit[0]
        pos[i, 1] = sy + jit[1]
        pos[i, 2] = z_cruise
    class_id = np.zeros(n_uavs, dtype=np.int8)
    if class_split is not None and sum(class_split) == n_uavs:
        n_small, n_large = class_split
        # Take a random subset of indices and mark as large class.
        large_idx = rng.choice(n_uavs, size=n_large, replace=False)
        class_id[large_idx] = 1
    return FleetState(
        pos=pos,
        vel=np.zeros((n_uavs, 3), dtype=np.float64),
        home_station=home_idx.astype(np.int32),
        state_code=np.full(n_uavs, IDLE, dtype=np.int32),
        soc=np.full(n_uavs, float(soc_init), dtype=np.float32),
        mission_id=np.full(n_uavs, -1, dtype=np.int64),
        target=pos.copy(),
        class_id=class_id,
    )


def integrate(fleet: FleetState, force: np.ndarray, wind_uvw: np.ndarray,
              dt: float, mass: float = 1.5, gamma: float = 0.6,
              v_max: float = 18.0,
              bounds_xy: tuple = None,
              bounds_z: tuple = (0.0, 120.0)) -> None:
    """Semi-implicit Euler. Pattern: aura_toy_archive/uav.py:integrate."""
    f = force + wind_uvw - gamma * fleet.vel
    fleet.vel += (f / mass) * dt
    speed = np.linalg.norm(fleet.vel, axis=1, keepdims=True)
    over = speed > v_max
    if np.any(over):
        scale = np.where(over, v_max / np.maximum(speed, 1e-9), 1.0)
        fleet.vel *= scale
    fleet.pos += fleet.vel * dt
    if bounds_xy is not None:
        (x_lo, y_lo), (x_hi, y_hi) = bounds_xy
        for ax, lo, hi in ((0, x_lo, x_hi), (1, y_lo, y_hi)):
            below = fleet.pos[:, ax] < lo
            fleet.pos[below, ax] = 2 * lo - fleet.pos[below, ax]
            fleet.vel[below, ax] *= -0.5
            above = fleet.pos[:, ax] > hi
            fleet.pos[above, ax] = 2 * hi - fleet.pos[above, ax]
            fleet.vel[above, ax] *= -0.5
    z_lo, z_hi = bounds_z
    below = fleet.pos[:, 2] < z_lo
    fleet.pos[below, 2] = z_lo
    fleet.vel[below, 2] = np.maximum(fleet.vel[below, 2], 0.0)
    above = fleet.pos[:, 2] > z_hi
    fleet.pos[above, 2] = z_hi
    fleet.vel[above, 2] = np.minimum(fleet.vel[above, 2], 0.0)


def proportional_track(fleet: FleetState, kp: float = 1.0,
                       kd: float = 0.5) -> np.ndarray:
    """A trivial PD controller toward `fleet.target` — used until MPC is wired in."""
    err = fleet.target - fleet.pos
    return kp * err - kd * fleet.vel
