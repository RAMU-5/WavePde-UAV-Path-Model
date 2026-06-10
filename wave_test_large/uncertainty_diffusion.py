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
    sigma: np.ndarray     # 2 * sqrt(tau) — position std (m)
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
        lo = max(lo, 0); hi = max(hi, lo + 1)
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

    return UncertaintyProfile(s=s, t=t, D=D, tau=tau, sigma=sigma,
                              L_eddy=L_eddy, sigma_q2=sigma_q2)


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
