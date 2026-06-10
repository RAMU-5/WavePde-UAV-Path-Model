"""Delay-only CBS-style conflict resolution.

We do not invoke the full cbs-py constraint tree because (a) Stage 1 mission
volume is modest, (b) we always resolve conflicts by departure delay (paths
are immutable nominal trajectories). The algorithm:

1. Collect cell reservations for each mission given its candidate launch time.
2. While any (cell, t_bin) is reserved by >1 mission:
   - Pick the conflict involving the mission with the latest launch time.
   - Push that mission's launch time by `bin_s`.
   - Re-stamp its reservations.
3. Stop after `max_delay_s` (mission rejected back to ALNS).

This is functionally equivalent to a CBS that only adds vertex constraints
on the second mission. cbs-py is wired in as a future optimisation in the
TODO at the bottom — it would replace the manual loop here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from aura.cbs.discretize import (
    ReservationTable,
    cells_along_trajectory,
)
from aura.schemas import NominalTrajectory


@dataclass
class MissionRequest:
    mission_id: int
    nominal: NominalTrajectory
    eps_m: float
    requested_launch_s: float


def resolve_delays(requests: List[MissionRequest], grid,
                   bin_s: float = 1.0,
                   max_delay_s: float = 60.0) -> Dict[int, float]:
    """Return a {mission_id: granted_launch_s} map.

    Mission ids are kept; missions that exceed `max_delay_s` are dropped from
    the result (caller treats that as "rejected").
    """
    table = ReservationTable(bin_s=bin_s)
    granted: Dict[int, float] = {}
    pending = sorted(requests, key=lambda r: r.requested_launch_s)
    for req in pending:
        launch = req.requested_launch_s
        accepted = False
        while launch - req.requested_launch_s <= max_delay_s:
            keys = set(cells_along_trajectory(req.nominal, grid, req.eps_m,
                                              bin_s=bin_s, t_launch_s=launch))
            collisions = [k for k in keys if k in table.table]
            if not collisions:
                for k in keys:
                    table.add(k, req.mission_id)
                granted[req.mission_id] = launch
                accepted = True
                break
            launch += bin_s
        if not accepted:
            continue
    return granted


# TODO(Stage 2): plug cbs-py over the same reservation interface for
# constraint-tree exploration when conflicts grow dense (>30/min).
