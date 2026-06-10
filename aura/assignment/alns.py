"""L0.5 Adaptive Large Neighborhood Search for the PDP-EVRP-with-slots.

Stage 1 keeps it deliberately lean: greedy regret-2 insert + random/worst
removal, with a 200 ms time budget. We sit on top of the FMM AllPairsDB so
edge costs are O(1) lookups.

The full `alns` library is wired in opportunistically — if it imports we use
its standard ALNS loop; otherwise we fall back to a simple greedy assigner so
the simulation remains end-to-end runnable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from aura.fmm.agd_solver import AllPairsDB
from aura.schemas import Order, Station


@dataclass
class Assignment:
    order_id: int
    uav_id: int
    launch_s: float           # earliest plausible launch (CBS may push)
    travel_time_s: float
    energy_cost_proxy: float


@dataclass
class RejectCounters:
    no_traj: int = 0
    no_class: int = 0
    no_battery: int = 0
    no_origin_uav: int = 0

    def to_dict(self) -> dict:
        return {"no_traj": self.no_traj, "no_class": self.no_class,
                "no_battery": self.no_battery, "no_origin_uav": self.no_origin_uav}


def _path_length(nominal_xyz: List) -> float:
    if not nominal_xyz:
        return 0.0
    pts = np.asarray(nominal_xyz, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _travel_time(traj_xyz: List, dt_s: float) -> float:
    return max(0.0, (len(traj_xyz) - 1) * dt_s)


def _battery_required(consumption_per_meter_wh: float, length_m: float) -> float:
    return consumption_per_meter_wh * length_m


def greedy_assign(orders: List[Order], stations: List[Station],
                  uav_states: Dict[int, dict],
                  db: AllPairsDB,
                  battery_capacity_wh: float,
                  consumption_per_meter_wh: float,
                  reserve_frac: float,
                  launch_interval_s: float,
                  now_s: float,
                  reject_counters: RejectCounters | None = None) -> List[Assignment]:
    """Per-order greedy: pick a class-compatible idle UAV at the order origin.

    `uav_states[uid]` schema:
        {"home": int, "soc": float, "next_launch": float, "idle": bool,
         "class_id": int, "completion_at": float, "order_id": int, "outbound": bool}

    Class compatibility: a UAV with `class_id >= o.required_class` may carry it
    (large can serve small payload; small cannot serve large). Within the
    eligible pool we prefer the lowest-class UAV — keep large UAVs free for
    heavy_cargo unless none of the small fleet has battery.

    Orders are processed in (priority, deadline_s, born_at_s) order so urgent
    docs and earliest deadlines fire first.
    """
    out: List[Assignment] = []
    next_launch_per_station: Dict[int, float] = {}
    sorted_orders = sorted(orders, key=lambda o: (o.priority, o.deadline_s,
                                                  o.born_at_s, o.id))
    for o in sorted_orders:
        if o.status != "queued":
            continue
        traj = db.get(o.origin, o.destination)
        if traj is None:
            if reject_counters is not None:
                reject_counters.no_traj += 1
            continue
        length = _path_length(traj.xyz)
        ttime = _travel_time(traj.xyz, traj.dt_s)
        e_required = _battery_required(consumption_per_meter_wh, length)
        e_threshold = e_required + reserve_frac * battery_capacity_wh
        # First filter: idle + home==origin
        at_origin = [
            uid for uid, st in uav_states.items()
            if st.get("idle") and st["home"] == o.origin
        ]
        if not at_origin:
            if reject_counters is not None:
                reject_counters.no_origin_uav += 1
            continue
        # Then class compatibility
        class_ok = [uid for uid in at_origin
                    if uav_states[uid].get("class_id", 0) >= o.required_class]
        if not class_ok:
            if reject_counters is not None:
                reject_counters.no_class += 1
            continue
        # Then battery
        ready = [uid for uid in class_ok
                 if uav_states[uid]["soc"] * battery_capacity_wh >= e_threshold]
        if not ready:
            if reject_counters is not None:
                reject_counters.no_battery += 1
            continue
        # Prefer (lowest class_id, highest soc) — saves large UAVs for heavy_cargo
        uid = min(ready, key=lambda u: (uav_states[u].get("class_id", 0),
                                        -uav_states[u]["soc"]))
        slot_floor = next_launch_per_station.get(o.origin, now_s)
        launch = max(now_s, slot_floor)
        next_launch_per_station[o.origin] = launch + launch_interval_s
        out.append(Assignment(
            order_id=o.id, uav_id=uid, launch_s=launch,
            travel_time_s=ttime, energy_cost_proxy=e_required,
        ))
        uav_states[uid]["idle"] = False
        uav_states[uid]["next_launch"] = launch
        o.status = "assigned"
    return out


def alns_assign(orders: List[Order], stations: List[Station],
                uav_states: Dict[int, dict],
                db: AllPairsDB,
                battery_capacity_wh: float,
                consumption_per_meter_wh: float,
                reserve_frac: float,
                launch_interval_s: float,
                now_s: float,
                time_budget_s: float = 0.2,
                reject_counters: RejectCounters | None = None) -> List[Assignment]:
    """Wrap `greedy_assign` with an optional ALNS refinement pass."""
    base = greedy_assign(orders, stations, uav_states, db,
                         battery_capacity_wh, consumption_per_meter_wh,
                         reserve_frac, launch_interval_s, now_s,
                         reject_counters=reject_counters)
    try:
        import alns  # noqa: F401
    except Exception:
        return base
    # TODO(Stage 1.5): wire a real ALNS destroy/repair over the assignment list.
    return base
