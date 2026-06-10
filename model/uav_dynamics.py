"""Paper v3 §3: UAV dynamics along a prescribed wave-derived path.

The wave solver gives the path *geometry* x_d(s) (Eq. 18) but assumes the
UAV instantly travels at c_0 + u in horizontal and c_z0 + w in vertical.
This module integrates the actual 1D dynamics along the path including
thrust/drag/gravity (Eq. 33-35) so we can:

    1. compute realistic time-of-flight along a wave path
    2. report energy / battery use along a path
    3. distinguish ascent vs descent timing from a single wave solve

Inputs:
    path : (M, 3) array of points along the path in metres (src→dst order)
    params : UAVDynParams with mass, drag area, power, etc.

Outputs:
    s   : (M,) arc-length at each point
    V   : (M,) speed at each point (m/s) — integrated from Eq. 33
    t   : (M,) cumulative time at each point (s)
    e   : (M,) cumulative energy expended (J) = ∫ P dt
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class UAVDynParams:
    """Defaults tuned so terminal V≈c0=15 m/s (matches wave-model assumption).

    From Eq. 33 with steady state dV/dt=0 on level ground:
        V_terminal = (2P / (ρ·CDA))^(1/3)
        15 m/s ⇒ P/CDA ≈ 2070 W·m^-2

    Picking CDA=0.10 (multirotor frontal × 0.5 Cd ≈ Mavic-3-class) gives
    P=200W cruise, which matches Meituan / DJI delivery drone numbers.
    Mass 2 kg matches their 1.7-2.5 kg payload class.
    """
    mass: float = 2.0          # kg
    drag_area: float = 0.10    # C_D · A (m²)
    rho: float = 1.225         # air density kg/m³
    power: float = 220.0       # cruise power W
    g: float = 9.81            # m/s²
    v_min: float = 0.5         # m/s — avoid T=P/V singularity at V→0
    v_max: float = 18.0        # m/s — slightly above c0 to allow tail-wind


def integrate_dynamics(path: np.ndarray, params: UAVDynParams = None,
                       v0: float = 1.0, dt: float = 0.05):
    """Eq. 33-35 explicit Euler along the path.

    Returns dict with arrays of length M:
        s, V, t, e (arc length, speed, time, energy)
        slope (dot(t̂, ê_z) at each segment — for inspection)
    """
    if params is None:
        params = UAVDynParams()
    M = path.shape[0]
    if M < 2:
        raise ValueError("path must have at least 2 points")

    # Arc-length and unit tangent per segment
    seg = np.diff(path, axis=0)                  # (M-1, 3)
    seg_len = np.linalg.norm(seg, axis=1)        # (M-1,)
    tangent = seg / np.maximum(seg_len[:, None], 1e-9)
    slope_seg = tangent[:, 2]                    # t̂ · ê_z, per segment

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    V = np.zeros(M)
    t = np.zeros(M)
    e = np.zeros(M)

    V[0] = v0
    s_target = s[1:]    # path s-values to land on
    s_curr = 0.0
    V_curr = v0
    t_curr = 0.0
    e_curr = 0.0

    seg_idx = 0
    out_V = [v0]; out_t = [0.0]; out_e = [0.0]
    half_rho_CDA = 0.5 * params.rho * params.drag_area

    # Sub-step in time, advancing s_curr; whenever we cross a node,
    # snapshot V/t/e there.
    max_substeps = int(60.0 / dt + 5)  # 1 minute hard cap
    for _ in range(max_substeps):
        # Per Eq. 33: dV/dt = P/(m·V) - (½ρCDA/m) V² - g·(t̂·ê_z)
        slope = slope_seg[min(seg_idx, len(slope_seg) - 1)]
        V_safe = max(V_curr, params.v_min)
        dV = (params.power / (params.mass * V_safe)
              - (half_rho_CDA / params.mass) * V_safe ** 2
              - params.g * slope) * dt
        V_new = np.clip(V_curr + dV, params.v_min, params.v_max)
        ds = 0.5 * (V_curr + V_new) * dt
        s_new = s_curr + ds
        # Power expended over the sub-step (constant P assumption)
        e_new = e_curr + params.power * dt

        # Cross any nodes? Linearly interpolate snapshot
        while seg_idx < M - 1 and s_new >= s_target[seg_idx]:
            frac = ((s_target[seg_idx] - s_curr)
                    / max(s_new - s_curr, 1e-12))
            V_at = V_curr + frac * (V_new - V_curr)
            t_at = t_curr + frac * dt
            e_at = e_curr + frac * (e_new - e_curr)
            out_V.append(V_at); out_t.append(t_at); out_e.append(e_at)
            seg_idx += 1
        s_curr = s_new; V_curr = V_new
        t_curr += dt; e_curr = e_new
        if seg_idx >= M - 1:
            break

    # Pad if dynamics didn't reach the end (extreme uphill stalled)
    while len(out_V) < M:
        out_V.append(out_V[-1]); out_t.append(out_t[-1]);
        out_e.append(out_e[-1])

    V = np.asarray(out_V[:M]); t = np.asarray(out_t[:M])
    e = np.asarray(out_e[:M])

    return dict(s=s, V=V, t=t, e=e, slope_seg=slope_seg, params=params)


def report(path: np.ndarray, params: UAVDynParams = None) -> dict:
    """One-shot: integrate and return summary stats."""
    res = integrate_dynamics(path, params)
    summary = dict(
        path_length_m=float(res["s"][-1]),
        time_dynamics_s=float(res["t"][-1]),
        energy_J=float(res["e"][-1]),
        v_mean_mps=float(res["s"][-1] / max(res["t"][-1], 1e-9)),
        v_min_mps=float(res["V"].min()),
        v_max_mps=float(res["V"].max()),
        ascent_m=float(np.maximum(np.diff(path[:, 2]), 0).sum()),
        descent_m=float(np.maximum(-np.diff(path[:, 2]), 0).sum()),
    )
    return summary, res


if __name__ == "__main__":
    # Smoke test: 1 km horizontal at z=50m
    path = np.linspace([0, 0, 50], [1000, 0, 50], 200)
    s, r = report(path)
    print("[demo] 1 km level path:")
    for k, v in s.items(): print(f"  {k}: {v:.2f}")
