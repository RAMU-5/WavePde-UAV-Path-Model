"""Diffusion-based positional uncertainty along a UAV trajectory
(paper §2.6.3 / §2.7, Eqs. 42-49).

Premise: turbulent velocity fluctuations along the flight path act as
a diffusive perturbation to the UAV's nominal position. Treating the
UAV as a passive tracer (inertia neglected at the uncertainty scale),
the position PDF p(r, t) satisfies the spherically symmetric
diffusion equation

    dp/dt = D(t) * [d^2p/dr^2 + (2/r) dp/dr]

with diffusion coefficient

    D(t) = <q'(t) . q'(t)>_{T_L} * T_L
         = sigma_q^2 * T_L,         T_L = L_eddy / |q_uav|

(paper Eq. 44-45). The closed-form Green's function with a delta
initial condition is

    p(r, t) = (4*pi*tau(t))^{-3/2} * exp(-r^2 / (4 * tau(t)))     (Eq. 49)
    tau(t) = integral_0^t D(s) ds

and the radial standard deviation of the UAV position is

    sigma_position(t) = 2 * sqrt(tau(t))        (paper Eq. line 859)

This module:
  - computes tau(t), sigma_position(t) along a discretised trajectory
  - estimates sigma_q^2 and T_L from a sample of velocity vectors
    along the trajectory (paper Eqs. 46-48 — vector autocorrelation)
  - provides a sanity-check Monte-Carlo Brownian-motion solver

The existing aura/wind/base.py exposes an `uncertainty()` API but
never had a real PDE solution; this fills that in.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UncertaintyProfile:
    s: np.ndarray         # arc length (m)
    t: np.ndarray         # cumulative time (s)
    D: np.ndarray         # diffusion coeff D(t) (m^2/s)
    tau: np.ndarray       # cumulative integral of D (m^2)
    sigma: np.ndarray     # 2 * sqrt(tau) — paper's legacy radius proxy (m)
    sigma_axis: np.ndarray  # sqrt(2*tau) — exact per-axis std (m)
    r95: np.ndarray       # exact 95% radial confidence radius in 3D (m)
    L_eddy: float
    sigma_q2: np.ndarray  # variance of q'(t) along path (m^2/s^2)


def lagrangian_T_L(q_uav_speed: np.ndarray, L_eddy: float) -> np.ndarray:
    """T_L = L_eddy / |q_uav|, paper Eq. line 764."""
    return L_eddy / np.maximum(q_uav_speed, 1e-3)


def estimate_q_fluct_var(q_along: np.ndarray, t: np.ndarray,
                         window_T: float) -> np.ndarray:
    """Window-averaged variance ||q'(t)||^2 = ||q - <q>_window||^2.

    Computes per-sample local mean over a centred ~window_T window,
    then returns the squared deviation. This is the discrete analogue
    of <q'.q'>_{T_L} in paper Eq. 44.

    q_along : (N, 3) velocity samples along trajectory
    t : (N,) sample times
    window_T : window size in seconds (typically T_L)
    """
    N = q_along.shape[0]
    var = np.zeros(N, dtype=np.float64)
    for i in range(N):
        lo = np.searchsorted(t, t[i] - 0.5 * window_T)
        hi = np.searchsorted(t, t[i] + 0.5 * window_T)
        lo = max(lo, 0)
        hi = max(hi, lo + 1)
        local = q_along[lo:hi]
        mu = local.mean(axis=0)
        diff = q_along[i] - mu
        var[i] = float(diff @ diff)
    return var


def compute_uncertainty(s: np.ndarray, t: np.ndarray,
                        q_along: np.ndarray,
                        L_eddy: float = 1.0,
                        sigma_q2_override: Optional[np.ndarray] = None
                        ) -> UncertaintyProfile:
    """Build the full UncertaintyProfile from a discretised trajectory.

    s, t : (N,) arc length and cumulative time along trajectory
    q_along : (N, 3) velocity along trajectory (used for both T_L
                     via |q| and sigma_q2 via fluctuation)
    L_eddy : characteristic eddy size (m). Default 1 m matches the
             urban-canopy estimate in paper §2.8 (line 766).
    sigma_q2_override : optional pre-computed variance per sample
                        (from a higher-fidelity model than the windowed
                        local-mean shortcut here).
    """
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    q_along = np.asarray(q_along, dtype=np.float64)
    if t.shape != s.shape or q_along.shape[0] != s.shape[0]:
        raise ValueError("inconsistent input shapes")
    N = len(t)

    speed = np.linalg.norm(q_along, axis=1)
    T_L = lagrangian_T_L(speed, L_eddy)

    if sigma_q2_override is not None:
        sigma_q2 = sigma_q2_override.astype(np.float64)
    else:
        # Use the local mean over a T_L window; for paths with small N
        # we fall back to global variance.
        T_L_typ = float(np.median(T_L)) if N > 1 else 1.0
        sigma_q2 = estimate_q_fluct_var(q_along, t, window_T=max(T_L_typ, 1e-3))

    D = sigma_q2 * T_L

    # tau(t) = trapezoidal integral of D over time
    tau = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        dt = t[i] - t[i-1]
        tau[i] = tau[i-1] + 0.5 * (D[i-1] + D[i]) * dt

    sigma = 2.0 * np.sqrt(np.maximum(tau, 0.0))
    sigma_axis = np.sqrt(2.0 * np.maximum(tau, 0.0))
    r95 = confidence_radius(tau, confidence=0.95, dimensions=3)

    return UncertaintyProfile(s=s, t=t, D=D, tau=tau, sigma=sigma,
                              sigma_axis=sigma_axis, r95=r95,
                              L_eddy=L_eddy, sigma_q2=sigma_q2)


def confidence_radius(tau: np.ndarray, confidence: float = 0.95,
                      dimensions: int = 3) -> np.ndarray:
    """Radial confidence radius for isotropic diffusion.

    Each Cartesian component has variance ``2*tau``. Consequently
    ``||X||^2 / (2*tau)`` follows a chi-square distribution with
    ``dimensions`` degrees of freedom. This avoids interpreting
    ``2*sqrt(tau)`` as a standard deviation, which it is not in 3D.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    from scipy.stats import chi2

    tau = np.asarray(tau, dtype=np.float64)
    return np.sqrt(2.0 * np.maximum(tau, 0.0)
                   * chi2.ppf(confidence, df=dimensions))


def sample_trajectory_ensemble(path: np.ndarray, t: np.ndarray,
                               D: np.ndarray, n_samples: int = 2000,
                               seed: int = 0) -> np.ndarray:
    """Sample Brownian perturbations around a nominal timed trajectory.

    The increment on segment ``i`` is isotropic Gaussian with variance
    ``2 * D_mid * dt`` per axis. The initial position is held fixed.
    Returns an array shaped ``(n_samples, n_path_points, 3)``.
    """
    path = np.asarray(path, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must have shape (N, 3)")
    if t.shape != (len(path),) or D.shape != t.shape:
        raise ValueError("t and D must have one value per path point")
    if np.any(np.diff(t) < 0.0) or np.any(D < 0.0):
        raise ValueError("time must be monotone and D non-negative")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    dt = np.diff(t)
    D_mid = 0.5 * (D[:-1] + D[1:])
    step_sigma = np.sqrt(2.0 * D_mid * dt)
    increments = rng.normal(size=(n_samples, len(dt), 3))
    increments *= step_sigma[None, :, None]
    perturb = np.concatenate([
        np.zeros((n_samples, 1, 3), dtype=np.float64),
        np.cumsum(increments, axis=1),
    ], axis=1)
    return path[None, :, :] + perturb


def path_clearance(path: np.ndarray, mask: np.ndarray,
                   spacing: tuple[float, float, float]) -> np.ndarray:
    """Distance from every path point to the nearest occupied voxel."""
    from scipy.ndimage import distance_transform_edt, map_coordinates

    path = np.asarray(path, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("expected a 3D mask and an (N, 3) path")
    clearance_grid = distance_transform_edt(~mask, sampling=spacing)
    # Physical coordinates refer to voxel centres at (i + 1/2) * h.
    coords = np.stack([
        path[:, 0] / spacing[0] - 0.5,
        path[:, 1] / spacing[1] - 0.5,
        path[:, 2] / spacing[2] - 0.5,
    ])
    return map_coordinates(clearance_grid, coords, order=1,
                           mode="constant", cval=0.0)


def ensemble_collision_statistics(samples: np.ndarray, mask: np.ndarray,
                                  spacing: tuple[float, float, float],
                                  confidence: float = 0.95) -> dict:
    """Estimate building collision, domain exit and their union.

    The three event probabilities are reported separately because a sample
    may leave the computational domain (most commonly below ground) without
    intersecting an occupied building voxel. ``collision_probability`` keeps
    the conservative union used by earlier callers.
    """
    samples = np.asarray(samples, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if samples.ndim != 3 or samples.shape[2] != 3:
        raise ValueError("samples must have shape (M, N, 3)")

    # Voxel i occupies [i*h, (i+1)*h); floor therefore maps physical
    # coordinates consistently with the cell-centred solver grid.
    idx = np.floor(
        samples / np.asarray(spacing)[None, None, :]
    ).astype(int)
    shape = np.asarray(mask.shape)
    outside = np.any((idx < 0) | (idx >= shape[None, None, :]), axis=2)
    clipped = np.clip(idx, 0, shape - 1)
    occupied = mask[clipped[:, :, 0], clipped[:, :, 1], clipped[:, :, 2]]
    obstacle_collided = np.any(occupied & ~outside, axis=1)
    outside_collided = np.any(outside, axis=1)
    collided = obstacle_collided | outside_collided
    k = int(collided.sum())
    n = int(len(collided))

    from scipy.stats import norm

    z = float(norm.ppf(0.5 + 0.5 * confidence))
    def estimate(event: np.ndarray) -> tuple[int, float, float, float]:
        count = int(event.sum())
        probability = count / n
        denom = 1.0 + z * z / n
        centre = (probability + z * z / (2.0 * n)) / denom
        half = z * np.sqrt(
            probability * (1.0 - probability) / n
            + z * z / (4.0 * n * n)
        ) / denom
        low = 0.0 if count == 0 else max(0.0, centre - half)
        high = 1.0 if count == n else min(1.0, centre + half)
        return count, float(probability), float(low), float(high)

    k, p, low, high = estimate(collided)
    k_obstacle, p_obstacle, low_obstacle, high_obstacle = estimate(obstacle_collided)
    k_outside, p_outside, low_outside, high_outside = estimate(outside_collided)
    return {
        "n_samples": n,
        "n_collided": k,
        "collision_probability": p,
        "collision_ci_low": low,
        "collision_ci_high": high,
        "n_obstacle_collided": k_obstacle,
        "obstacle_collision_probability": p_obstacle,
        "obstacle_collision_ci_low": low_obstacle,
        "obstacle_collision_ci_high": high_obstacle,
        "n_domain_exited": k_outside,
        "domain_exit_probability": p_outside,
        "domain_exit_ci_low": low_outside,
        "domain_exit_ci_high": high_outside,
    }


def monte_carlo_dispersion(D_t_fn, t_max: float, dt: float = 0.05,
                           n_walkers: int = 2000,
                           rng: Optional[np.random.Generator] = None):
    """3D Brownian-motion sampler with time-varying isotropic diffusion.

    Returns the empirical std of |x(t_max) - x(0)| across n_walkers,
    which should agree with the analytical 2*sqrt(tau(t_max)) at large
    n_walkers if the closed-form solution is correct.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_steps = int(t_max / dt)
    x = np.zeros((n_walkers, 3), dtype=np.float64)
    for k in range(n_steps):
        t = (k + 0.5) * dt
        D = float(D_t_fn(t))
        # Diffusion of variance D dt per dimension over dt:
        # x += sqrt(2*D*dt) * N(0, I)   gives <x.x>_per_dim = 2 D dt
        sigma_step = np.sqrt(2.0 * D * dt)
        x += sigma_step * rng.normal(size=x.shape)
    r = np.linalg.norm(x, axis=1)
    return dict(r_mean=float(r.mean()), r_std=float(r.std()),
                r_rms=float(np.sqrt((r ** 2).mean())))
