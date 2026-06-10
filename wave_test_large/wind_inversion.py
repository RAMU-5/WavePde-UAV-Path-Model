"""Wind estimation from observed UAV flight (paper §2.6.2, Eq. 38-41).

Given an observed UAV trajectory:
  - position(t)  -> ground velocity q_g(t) by finite-differencing
  - airspeed(t) (from onboard sensors / dynamics model) -> q_a(t)

the wind velocity along the trajectory is recovered as

    u_est(t) = q_g(t) - q_a(t)

If a predicted wind field u_p(x,t) is available (e.g. from a numerical
weather surrogate), the prediction-error residual is

    epsilon(t) = u_est(t) - u_p(x(t), t)

This residual is the data-assimilation hook: it tells the user where
the predicted wind field is wrong along the actual flight track.

The current codebase has wind interfaces in `aura/wind/base.py` but
no inversion. This module fills that gap.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class WindEstimate:
    """Per-track-point wind reconstruction.

    Attributes
    ----------
    t_mid : (M-1,) — segment midpoint times (s)
    pos_mid : (M-1, 3) — segment midpoint positions (m)
    q_g : (M-1, 3) — ground velocity by central difference
    q_a : (M-1, 3) — air-relative velocity (passed in or modelled)
    u_est : (M-1, 3) — wind estimate q_g - q_a
    u_pred : (M-1, 3) or None — predicted wind sampled along track
    epsilon : (M-1, 3) or None — u_est - u_pred (None if no prediction)
    speed_g : (M-1,) — |q_g|
    speed_a : (M-1,) — |q_a|
    """
    t_mid: np.ndarray
    pos_mid: np.ndarray
    q_g: np.ndarray
    q_a: np.ndarray
    u_est: np.ndarray
    u_pred: Optional[np.ndarray]
    epsilon: Optional[np.ndarray]
    speed_g: np.ndarray
    speed_a: np.ndarray

    def summary(self) -> dict:
        out = dict(
            n_segments=int(len(self.q_g)),
            mean_wind_speed_mps=float(np.linalg.norm(self.u_est, axis=1).mean()),
            std_wind_speed_mps=float(np.linalg.norm(self.u_est, axis=1).std()),
            mean_wind_xyz=tuple(float(v) for v in self.u_est.mean(axis=0)),
        )
        if self.epsilon is not None:
            out["rmse_vs_predicted_mps"] = float(
                np.sqrt((self.epsilon ** 2).sum(axis=1).mean()))
        return out


def estimate_wind(track_xyz: np.ndarray,
                  track_t: np.ndarray,
                  q_a: np.ndarray,
                  u_pred_fn: Optional[Callable] = None) -> WindEstimate:
    """Recover the wind along an observed UAV track.

    Parameters
    ----------
    track_xyz : (M, 3) array of GPS-observed positions in metres.
    track_t : (M,) sample times (s). Must be strictly increasing.
    q_a : (M, 3) array of air-relative velocity at each sample
        (from IMU / motor-thrust model / pitot tube).
    u_pred_fn : optional callable (pos, t) -> (3,) wind vector. If
        provided, returns a residual against this prediction.

    Returns
    -------
    WindEstimate dataclass with arrays of length M-1 (segment-centred).
    """
    track_xyz = np.asarray(track_xyz, dtype=np.float64)
    track_t = np.asarray(track_t, dtype=np.float64)
    q_a = np.asarray(q_a, dtype=np.float64)
    M = track_xyz.shape[0]
    if M < 2:
        raise ValueError("track must have at least 2 points")
    if track_t.shape != (M,) or q_a.shape != (M, 3):
        raise ValueError("inconsistent input shapes")
    dt = np.diff(track_t)
    if np.any(dt <= 0):
        raise ValueError("track_t must be strictly increasing")

    # Central / segment-midpoint quantities
    pos_mid = 0.5 * (track_xyz[:-1] + track_xyz[1:])
    t_mid = 0.5 * (track_t[:-1] + track_t[1:])
    q_g = (track_xyz[1:] - track_xyz[:-1]) / dt[:, None]
    q_a_mid = 0.5 * (q_a[:-1] + q_a[1:])
    u_est = q_g - q_a_mid

    if u_pred_fn is not None:
        u_pred = np.array([u_pred_fn(pos_mid[i], t_mid[i])
                           for i in range(len(t_mid))], dtype=np.float64)
        epsilon = u_est - u_pred
    else:
        u_pred = None
        epsilon = None

    return WindEstimate(
        t_mid=t_mid, pos_mid=pos_mid,
        q_g=q_g, q_a=q_a_mid, u_est=u_est,
        u_pred=u_pred, epsilon=epsilon,
        speed_g=np.linalg.norm(q_g, axis=1),
        speed_a=np.linalg.norm(q_a_mid, axis=1),
    )


def estimate_wind_from_dynamics_model(track_xyz, track_t,
                                      airspeed_dir_fn,
                                      airspeed_mag_fn,
                                      u_pred_fn=None) -> WindEstimate:
    """Variant when q_a is built from (heading, airspeed) measurements.

    airspeed_dir_fn : (i) -> (3,) unit-vector heading at sample i
    airspeed_mag_fn : (i) -> scalar airspeed magnitude at sample i
    """
    M = track_xyz.shape[0]
    q_a = np.zeros((M, 3), dtype=np.float64)
    for i in range(M):
        d = np.asarray(airspeed_dir_fn(i), dtype=np.float64)
        d = d / max(np.linalg.norm(d), 1e-9)
        q_a[i] = airspeed_mag_fn(i) * d
    return estimate_wind(track_xyz, track_t, q_a, u_pred_fn)
