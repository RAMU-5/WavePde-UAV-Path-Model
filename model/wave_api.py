"""Parameterised API over the wave solver kernel in solve_all.py.

This replaces the former `routine.wave` module. There is exactly ONE
implementation of the PDE step (solve_all.step_kernel / solve_one /
build_static_fields_param); this file only adds the WaveConfig dataclass,
the StaticFields container, numpy-in/numpy-out wrappers and the adjoint
(backward) solve.

Note on the backward solve: reciprocity requires flipping the WHOLE
advection field w -> -w, including the climb/descend asymmetry term
w_grav (the backward wave ascending through a segment the UAV descends
must travel at the descend speed). The old routine.wave.solve_backward
only flipped cfg.wind and therefore did NOT flip w_grav — with zero wind
it silently degenerated to a forward solve. This implementation mirrors
the validated solve_backward.build_backward_fields semantics:
negate u, v, w; keep ax, ay, az (the (q·grad)q term is q-quadratic).

Usage:
    import sys; sys.path.insert(0, "<root>/model")
    from wave_api import WaveConfig, solve_forward, solve_backward
    T = solve_forward(mask, dx, dy, dz, src_idx, cfg=WaveConfig(wind=(3,0,0)))
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

import solve_all as _sa

__all__ = [
    "WaveConfig",
    "StaticFields",
    "build_static_fields",
    "gaussian_source",
    "solve_forward",
    "solve_backward",
]

gaussian_source = _sa.gaussian_source


@dataclass(frozen=True)
class WaveConfig:
    """Wave-equation parameters. Defaults equal the solve_all.py module
    constants (Meituan delivery-UAV regime, paper v3 §2.1.1) — the
    default config reproduces the batch scripts bit-for-bit."""
    c0: float = _sa.C0
    cz_up: float = _sa.CZ_UP
    cz_down: float = _sa.CZ_DOWN
    wind: Tuple[float, float, float] = _sa.WIND

    dt: float = _sa.DT
    n_steps: int = _sa.N_STEPS

    pulse_t0_steps: int = _sa.PULSE_T0_STEPS
    pulse_tau_steps: int = _sa.PULSE_TAU_STEPS
    source_sigma_cells: float = _sa.SOURCE_SIGMA_CELLS

    sponge_layers: int = _sa.SPONGE_LAYERS
    sponge_strength: float = _sa.SPONGE_STRENGTH
    # NB: source amplitudes scale with dx^3 (sum=1 cell normalisation);
    # the default anchor is tuned for the 50 m grid — pass
    # min_peak_amp = 1e-6 * (dx/50)**3 on other grids or the detector
    # misses weak first arrivals (see paper/Q2).
    min_peak_amp: float = _sa.MIN_PEAK_AMP

    reg_z_low: float = _sa.REG_Z_LOW
    reg_z_full: float = _sa.REG_Z_FULL
    reg_low_factor: float = _sa.REG_LOW_FACTOR


@dataclass(frozen=True)
class StaticFields:
    """Spatial fields built once per (mask, dx, dy, dz, cfg) and reused
    across every source on that domain."""
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


def build_static_fields(mask, dx, dy, dz,
                        cfg: Optional[WaveConfig] = None) -> StaticFields:
    """Build wave-speed, advection and sponge fields from a (NX,NY,NZ)
    boolean building mask via the single kernel implementation."""
    if cfg is None:
        cfg = WaveConfig()
    s = _sa.build_static_fields_param(
        jnp.asarray(mask), dx, dy, dz,
        c0=cfg.c0, cz_up=cfg.cz_up, cz_down=cfg.cz_down, wind=cfg.wind,
        sponge_layers=cfg.sponge_layers, sponge_strength=cfg.sponge_strength,
        reg_z_low=cfg.reg_z_low, reg_z_full=cfg.reg_z_full,
        reg_low_factor=cfg.reg_low_factor,
    )
    return StaticFields(
        c2_xy=s["c2_xy"], c2_z=s["c2_z"], u=s["u"], v=s["v"], w=s["w"],
        ax=s["ax"], ay=s["ay"], az=s["az"],
        one_minus_damp=(1.0 - s["damp"]).astype(jnp.float32),
        free=s["free"],
    )


# dx, dy, dz, n_steps, dt, pulse_t0_steps, pulse_tau_steps, min_peak_amp
# are compile-time constants for the scan.
_solve_jit = jit(_sa.solve_one, static_argnums=(11, 12, 13, 14, 15, 16, 17, 18))


def _run(static: StaticFields, src_idx, dx, dy, dz,
         cfg: WaveConfig) -> np.ndarray:
    src = _sa.gaussian_source(static.free.shape, src_idx,
                              sigma=cfg.source_sigma_cells)
    T = _solve_jit(static.c2_xy, static.c2_z,
                   static.u, static.v, static.w,
                   static.ax, static.ay, static.az,
                   static.one_minus_damp, static.free, src,
                   dx, dy, dz, cfg.n_steps,
                   cfg.dt, cfg.pulse_t0_steps, cfg.pulse_tau_steps,
                   cfg.min_peak_amp)
    jax.block_until_ready(T)
    return np.asarray(jax.device_get(T))


def solve_forward(mask, dx, dy, dz, src_idx,
                  cfg: Optional[WaveConfig] = None,
                  static: Optional[StaticFields] = None) -> np.ndarray:
    """Forward wave solve: source at `src_idx`, returns T(x,y,z) as numpy.

    Pass a pre-built `static` when solving many sources on one domain to
    amortise the field build and the JIT compile."""
    if cfg is None:
        cfg = WaveConfig()
    if static is None:
        static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
    return _run(static, src_idx, dx, dy, dz, cfg)


def flip_static_fields(static: StaticFields) -> StaticFields:
    """q -> -q for the adjoint solve: negate u, v, w (wind AND the
    climb/descend asymmetry w_grav); ax, ay, az are q-quadratic and
    stay unchanged. Mirrors solve_backward.build_backward_fields."""
    return StaticFields(
        c2_xy=static.c2_xy, c2_z=static.c2_z,
        u=-static.u, v=-static.v, w=-static.w,
        ax=static.ax, ay=static.ay, az=static.az,
        one_minus_damp=static.one_minus_damp, free=static.free,
    )


def solve_backward(mask, dx, dy, dz, dst_idx,
                   cfg: Optional[WaveConfig] = None,
                   static: Optional[StaticFields] = None) -> np.ndarray:
    """Adjoint wave solve (paper §2.5): source at the destination,
    full advection flip q -> -q. The returned T-field gives the travel
    time *to* `dst_idx` from every cell — one solve serves K origins.

    `static`, if given, must be the FORWARD fields (it is flipped here).
    Exact reciprocity holds for divergence-free q; see validation/."""
    if cfg is None:
        cfg = WaveConfig()
    if static is None:
        static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
    return _run(flip_static_fields(static), dst_idx, dx, dy, dz, cfg)
