"""Convected-wave PDE solver for UAV travel-time fields.

Given a 3D building mask + a source station, solves an anisotropic
convected wave equation by explicit leapfrog (JAX) and reads off the
arrival-time field T(x,y,z). T is then consumed by `routine.path` to
trace the actual flight path between two stations.

Reference: `neural_physics_uav_flight_path_optimisation/main_article.tex`
(forward solver) and `solve_backward.py` (adjoint/backward variant).

Physics summary:
    D^2 psi/Dt^2 - div(c^2 grad psi) + damping = src
    D = d_t + q·grad       (Lagrangian derivative under wind q)
    c  = anisotropic UAV speed (c_xy horizontal, c_z vertical with
         distinct ascending/descending speeds)
    src = Gaussian pulse at the source station

After ~N_STEPS leapfrog steps, T[i,j,k] is the time at which |psi| first
peaks at cell (i,j,k) — the wavefront's first arrival, which by Fermat's
principle is the optimal travel time.

The kernel is parameterised: every dynamic dependency is passed as a
positional argument so JIT caches across sources.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import os
from pathlib import Path

# Persistent JAX compile cache (first solve pays ~30-60s compile, rest
# reuse). Set BEFORE importing jax. Callers may override via the env var.
_DEFAULT_CACHE = str(Path(os.getcwd()) / "_jax_cache")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _DEFAULT_CACHE)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)   # float32 — big speed boost
import jax.numpy as jnp
from jax import jit, lax

__all__ = [
    "WaveConfig",
    "StaticFields",
    "build_static_fields",
    "gaussian_source",
    "solve_one",
    "solve_forward",
    "solve_backward",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaveConfig:
    """Wave-equation parameters. Defaults match `solve_all.py` (Meituan
    delivery-UAV regime, paper v3 §2.1.1).
    """
    c0: float = 15.0             # horizontal cruise speed [m/s]
    cz_up: float = 6.0           # ascent speed (faster than descent, paper Eq. 7)
    cz_down: float = 4.0         # descent speed (vortex-ring stability)
    wind: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # (u, v, w) in m/s

    dt: float = 0.10
    n_steps: int = 4500          # ~450s — covers a 7 km diagonal at 15 m/s

    pulse_t0_steps: int = 10
    pulse_tau_steps: int = 4
    source_sigma_cells: float = 1.5

    sponge_layers: int = 8
    sponge_strength: float = 0.30
    min_peak_amp: float = 1e-6

    # Regulatory altitude band (paper §5.9): horizontal speed is reduced
    # below 30 m, ramping to full at 40 m. Forces near-vertical takeoff/
    # landing, matching Meituan ops where horizontal motion is forbidden
    # below ~30 m.
    reg_z_low: float = 30.0
    reg_z_full: float = 40.0
    reg_low_factor: float = 0.05


@dataclass(frozen=True)
class StaticFields:
    """Spatial fields built once per (mask, dx, dy, dz, c, wind) tuple
    and reused across every source on that domain.
    """
    c2_xy: jnp.ndarray
    c2_z: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray
    ax: jnp.ndarray
    ay: jnp.ndarray
    az: jnp.ndarray
    one_minus_damp: jnp.ndarray
    free: jnp.ndarray


# ---------------------------------------------------------------------------
# Stencil operators (jnp)
# ---------------------------------------------------------------------------

def grad_central(f, axis, h):
    return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, +1, axis=axis)) / (2.0 * h)


def lap_var_coef_axis(c2, psi, h, axis):
    """Variable-coefficient Laplacian along `axis`, conservative form
    (flux at cell faces uses the geometric-mean c^2)."""
    c2_e = 0.5 * (c2 + jnp.roll(c2, -1, axis=axis))
    c2_w = 0.5 * (c2 + jnp.roll(c2, +1, axis=axis))
    flux_e = c2_e * (jnp.roll(psi, -1, axis=axis) - psi) / h
    flux_w = c2_w * (psi - jnp.roll(psi, +1, axis=axis)) / h
    return (flux_e - flux_w) / h


def adv(qx, qy, qz, psi, dx, dy, dz):
    """Convection q·∇psi."""
    return (qx * grad_central(psi, 0, dx)
            + qy * grad_central(psi, 1, dy)
            + qz * grad_central(psi, 2, dz))


# ---------------------------------------------------------------------------
# Static-field builder (run once per domain)
# ---------------------------------------------------------------------------

def build_static_fields(mask, dx, dy, dz,
                        cfg: Optional[WaveConfig] = None) -> StaticFields:
    """Build wave-speed, advection and sponge fields from a (NX,NY,NZ)
    boolean building mask.

    The c-field encodes:
      - 0 inside buildings (no propagation),
      - reduced horizontal speed below `cfg.reg_z_full` (regulatory),
      - asymmetric vertical speed (cz_up vs cz_down) baked in via a
        gravity-like z-component `w_grav = (cz_down - cz_up)/2`.
    """
    if cfg is None:
        cfg = WaveConfig()
    mask_jnp = jnp.asarray(mask)
    nx, ny, nz = mask_jnp.shape

    cz0 = 0.5 * (cfg.cz_up + cfg.cz_down)
    w_grav = 0.5 * (cfg.cz_down - cfg.cz_up)

    # Regulatory altitude profile (z-only)
    z_centers = (jnp.arange(nz) + 0.5) * dz
    reg = jnp.where(
        z_centers >= cfg.reg_z_full, 1.0,
        jnp.where(z_centers < cfg.reg_z_low, cfg.reg_low_factor,
                  cfg.reg_low_factor + (1.0 - cfg.reg_low_factor)
                      * (z_centers - cfg.reg_z_low)
                      / max(1e-9, cfg.reg_z_full - cfg.reg_z_low)),
    )[None, None, :]

    free = (~mask_jnp).astype(jnp.float32)
    c0_arr = cfg.c0 * free * reg
    cz0_arr = cz0 * free
    c2_xy = c0_arr ** 2
    c2_z = cz0_arr ** 2

    # Wind (constant for now; replace u/v/w with arrays for spatial wind).
    u = jnp.full((nx, ny, nz), float(cfg.wind[0])) * free
    v = jnp.full((nx, ny, nz), float(cfg.wind[1])) * free
    w = jnp.full((nx, ny, nz), float(cfg.wind[2])) * free - w_grav

    # Steady-flow acceleration (q·∇)q
    ax = u*grad_central(u, 0, dx) + v*grad_central(u, 1, dy) + w*grad_central(u, 2, dz)
    ay = u*grad_central(v, 0, dx) + v*grad_central(v, 1, dy) + w*grad_central(v, 2, dz)
    az = u*grad_central(w, 0, dx) + v*grad_central(w, 1, dy) + w*grad_central(w, 2, dz)

    # Sponge layers (absorbing BC) — lateral + top only, ground reflects freely.
    def edge_ramp(n):
        d = jnp.minimum(jnp.arange(n), n - 1 - jnp.arange(n))
        return jnp.where(d < cfg.sponge_layers,
                         (1.0 - d / max(1, cfg.sponge_layers - 1)) ** 2, 0.0)
    def top_only(n):
        d = n - 1 - jnp.arange(n)
        return jnp.where(d < cfg.sponge_layers,
                         (1.0 - d / max(1, cfg.sponge_layers - 1)) ** 2, 0.0)
    rx = edge_ramp(nx)[:, None, None]
    ry = edge_ramp(ny)[None, :, None]
    rz = top_only(nz)[None, None, :]
    damp = cfg.sponge_strength * jnp.maximum(jnp.maximum(rx, ry), rz)

    return StaticFields(
        c2_xy=c2_xy, c2_z=c2_z, u=u, v=v, w=w,
        ax=ax, ay=ay, az=az,
        one_minus_damp=(1.0 - damp).astype(jnp.float32),
        free=free,
    )


# ---------------------------------------------------------------------------
# Source pulse + leapfrog kernel
# ---------------------------------------------------------------------------

def gaussian_source(shape, idx, sigma: float = 1.5) -> jnp.ndarray:
    """Soft point source centered at cell `idx` with Gaussian width `sigma`."""
    nx, ny, nz = shape
    ix = jnp.arange(nx)[:, None, None]
    iy = jnp.arange(ny)[None, :, None]
    iz = jnp.arange(nz)[None, None, :]
    r2 = ((ix - idx[0])**2 + (iy - idx[1])**2 + (iz - idx[2])**2)
    g = jnp.exp(-0.5 * r2 / (sigma**2))
    return (g / g.sum()).astype(jnp.float32)


def _step_kernel(carry, n, c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
                 one_minus_damp, free, dx, dy, dz,
                 dt, pulse_t0_steps, pulse_tau_steps, min_peak_amp):
    psi_prev, psi_curr, amp_prev, was_rising, T, src_spatial = carry
    t = n.astype(jnp.float32) * dt
    pulse_t0 = pulse_t0_steps * dt
    pulse_tau = pulse_tau_steps * dt
    dt2 = jnp.float32(dt * dt)
    diff = (lap_var_coef_axis(c2_xy, psi_curr, dx, 0)
            + lap_var_coef_axis(c2_xy, psi_curr, dy, 1)
            + lap_var_coef_axis(c2_z, psi_curr, dz, 2))
    src = src_spatial * jnp.exp(-0.5 * ((t - pulse_t0) / pulse_tau) ** 2)
    psi_dot = (psi_curr - psi_prev) / dt
    adv1 = adv(u, v, w, psi_dot, dx, dy, dz)
    qg = adv(u, v, w, psi_curr, dx, dy, dz)
    adv2 = adv(u, v, w, qg, dx, dy, dz)
    adv_a = (ax_f*grad_central(psi_curr, 0, dx)
             + ay_f*grad_central(psi_curr, 1, dy)
             + az_f*grad_central(psi_curr, 2, dz))
    rhs = diff + src - 2.0*adv1 - adv2 - adv_a
    psi_next = 2.0*psi_curr - psi_prev + dt2 * rhs
    psi_next = psi_next * one_minus_damp * free
    # Hard zero on the outermost cells (sponge handles soft attenuation,
    # this guarantees no reflection from the box boundary).
    psi_next = psi_next.at[0, :, :].set(0.0)
    psi_next = psi_next.at[-1, :, :].set(0.0)
    psi_next = psi_next.at[:, 0, :].set(0.0)
    psi_next = psi_next.at[:, -1, :].set(0.0)
    psi_next = psi_next.at[:, :, 0].set(0.0)
    psi_next = psi_next.at[:, :, -1].set(0.0)

    # First-peak detector → arrival time T
    amp = jnp.abs(psi_next)
    rising_now = amp > amp_prev
    peak_now = (was_rising & (~rising_now)
                & jnp.isinf(T)
                & (amp_prev > min_peak_amp))
    T_new = jnp.where(peak_now, t - dt, T)
    return (psi_curr, psi_next, amp, rising_now, T_new, src_spatial), None


def solve_one(static: StaticFields,
              src_spatial: jnp.ndarray,
              dx: float, dy: float, dz: float,
              cfg: Optional[WaveConfig] = None) -> jnp.ndarray:
    """Run one leapfrog wave solve and return the arrival-time field T.

    Returns
    -------
    T : (NX, NY, NZ) jnp.ndarray
        T[i,j,k] is the time at which the wavefront first peaks at cell
        (i,j,k). Cells the wave never reaches stay at +inf.
    """
    if cfg is None:
        cfg = WaveConfig()
    shape = src_spatial.shape
    init = (
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=bool),
        jnp.full(shape, jnp.inf, dtype=jnp.float32),
        src_spatial,
    )
    def step_fn(carry, n):
        return _step_kernel(
            carry, n,
            static.c2_xy, static.c2_z,
            static.u, static.v, static.w,
            static.ax, static.ay, static.az,
            static.one_minus_damp, static.free,
            dx, dy, dz,
            cfg.dt, cfg.pulse_t0_steps, cfg.pulse_tau_steps, cfg.min_peak_amp,
        )
    n_arr = jnp.arange(cfg.n_steps, dtype=jnp.int32)
    final, _ = lax.scan(step_fn, init, n_arr)
    return final[4]   # T field


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def solve_forward(mask, dx, dy, dz, src_idx,
                  cfg: Optional[WaveConfig] = None,
                  static: Optional[StaticFields] = None) -> np.ndarray:
    """Forward wave solve: source at `src_idx`, returns T(x,y,z) as numpy.

    Parameters
    ----------
    mask    : (NX, NY, NZ) boolean building mask. True = blocked.
    dx,dy,dz: cell sizes in metres.
    src_idx : 3-tuple of integer cell indices for the source station.
    cfg     : WaveConfig (defaults match Meituan delivery UAV).
    static  : pre-built StaticFields (skip rebuild when re-using a domain).

    Returns
    -------
    T : (NX, NY, NZ) float32 numpy array.
        T[i,j,k] = arrival time at cell (i,j,k). +inf if never reached.
    """
    if cfg is None:
        cfg = WaveConfig()
    if static is None:
        static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
    src = gaussian_source(mask.shape, src_idx, sigma=cfg.source_sigma_cells)
    T = solve_one(static, src, dx, dy, dz, cfg=cfg)
    jax.block_until_ready(T)
    return np.asarray(jax.device_get(T))


def solve_backward(mask, dx, dy, dz, dst_idx,
                   cfg: Optional[WaveConfig] = None) -> np.ndarray:
    """Adjoint wave solve: source at the destination, q -> -q.

    The resulting T-field gives travel time *to* `dst_idx` from every
    cell in the domain — equivalent to forward in the absence of damping
    and wind, but lets you serve K candidate origins with one solve
    instead of K solves (paper §2.5).
    """
    if cfg is None:
        cfg = WaveConfig()
    flipped = WaveConfig(
        c0=cfg.c0, cz_up=cfg.cz_up, cz_down=cfg.cz_down,
        wind=(-cfg.wind[0], -cfg.wind[1], -cfg.wind[2]),
        dt=cfg.dt, n_steps=cfg.n_steps,
        pulse_t0_steps=cfg.pulse_t0_steps,
        pulse_tau_steps=cfg.pulse_tau_steps,
        source_sigma_cells=cfg.source_sigma_cells,
        sponge_layers=cfg.sponge_layers,
        sponge_strength=cfg.sponge_strength,
        min_peak_amp=cfg.min_peak_amp,
        reg_z_low=cfg.reg_z_low, reg_z_full=cfg.reg_z_full,
        reg_low_factor=cfg.reg_low_factor,
    )
    return solve_forward(mask, dx, dy, dz, dst_idx, cfg=flipped)
