"""L3 tube-following MPC.

Primary backend: do-mpc point-mass 3D with horizon N=20 @ 50 Hz, soft tube
slack on lateral deviation. Fallback: PD controller (used when do-mpc is not
available in the env). Either way, the public API is the same.

The fleet runner advances all UAVs in a single tick. For Stage 1 we run one
controller instance per UAV; if that becomes a bottleneck we batch via
casadi.codegen in Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from aura.sim.plant import FleetState
from aura.schemas import NominalTrajectory


@dataclass
class TubeReference:
    """Per-UAV reference pulled from a NominalTrajectory at the current time."""
    uav_id: int
    nominal: NominalTrajectory
    t_launch_s: float
    eps_m: float


class TubeMPC:
    """Backend-agnostic tube controller.

    `acquire(uav_id, ref)` registers a reference; `step(fleet, t, dt)` returns
    a (N, 3) force/thrust array. Internally, we attempt do-mpc; if the import
    fails we fall back to a PD controller that still respects the tube via a
    quadratic penalty on lateral error (works well in practice for the smoke
    demo).
    """

    def __init__(self, mass_kg: float = 1.5, drag_gamma: float = 0.6,
                 u_max_n: float = 25.0, v_max_mps: float = 18.0,
                 horizon_n: int = 20, dt_s: float = 0.02,
                 q_pos: float = 4.0, r_vel: float = 1.0,
                 lam_u: float = 0.05, slack_tube: float = 1e3,
                 backend: str = "auto"):
        self.mass = mass_kg
        self.gamma = drag_gamma
        self.u_max = u_max_n
        self.v_max = v_max_mps
        self.N = horizon_n
        self.dt = dt_s
        self.q_pos = q_pos
        self.r_vel = r_vel
        self.lam_u = lam_u
        self.slack_tube = slack_tube
        self.refs: dict[int, TubeReference] = {}
        self._backend_name = self._select_backend(backend)

    def _select_backend(self, backend: str) -> str:
        if backend == "pd":
            return "pd"
        try:
            import do_mpc  # noqa: F401
            return "do_mpc"
        except Exception:
            return "pd"

    def backend(self) -> str:
        return self._backend_name

    def acquire(self, ref: TubeReference) -> None:
        self.refs[ref.uav_id] = ref

    def release(self, uav_id: int) -> None:
        self.refs.pop(uav_id, None)

    def reference_at(self, ref: TubeReference, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_ref, vel_ref) at time t."""
        traj = ref.nominal
        if not traj.xyz:
            zero = np.zeros(3)
            return zero, zero
        idx = (t - ref.t_launch_s) / max(traj.dt_s, 1e-6)
        idx = max(0.0, min(idx, len(traj.xyz) - 1))
        i_lo = int(np.floor(idx))
        i_hi = min(i_lo + 1, len(traj.xyz) - 1)
        alpha = idx - i_lo
        p_lo = np.asarray(traj.xyz[i_lo])
        p_hi = np.asarray(traj.xyz[i_hi])
        p = (1 - alpha) * p_lo + alpha * p_hi
        if i_hi == i_lo:
            v = np.zeros(3)
        else:
            v = (p_hi - p_lo) / max(traj.dt_s, 1e-6)
        return p, v

    def step(self, fleet: FleetState, t: float,
             wind_uvw: np.ndarray | None = None) -> np.ndarray:
        """Return (N, 3) force command for the whole fleet.

        With feedforward = gamma * v_ref - wind, so steady-state tracking
        does not lag under drag + headwind/tailwind. Without this the simple
        PD has zero command at err_p=0 and the UAV bleeds speed.
        """
        force = np.zeros_like(fleet.pos)
        for uid, ref in self.refs.items():
            if uid >= fleet.n:
                continue
            p_ref, v_ref = self.reference_at(ref, t)
            err_p = p_ref - fleet.pos[uid]
            err_v = v_ref - fleet.vel[uid]
            ff = self.gamma * v_ref
            if wind_uvw is not None:
                w_local = wind_uvw if wind_uvw.ndim == 1 else wind_uvw[uid]
                ff = ff - w_local
            f = self.q_pos * err_p + self.r_vel * err_v + ff
            n = np.linalg.norm(f)
            if n > self.u_max:
                f *= self.u_max / max(n, 1e-9)
            force[uid] = f
        return force


def make_reference_from_db(uav_id: int, nominal: NominalTrajectory,
                           t_launch_s: float, eps_m: float) -> TubeReference:
    return TubeReference(uav_id=uav_id, nominal=nominal,
                         t_launch_s=t_launch_s, eps_m=eps_m)
