"""Broadband / multi-scale wave-routing source (paper §2.9.4–2.9.5).

The current solve_all.py uses a single narrow Gaussian source:

    s(t) = exp(-0.5 * ((t - t0)/tau)^2)

This produces a wave at one characteristic wavelength. The paper argues
for a SUPERPOSITION of long- and short-wave components in a single
solve:

    s_hat(t) = A_L exp(-(t-t_L)^2 / tau_L^2)              # long, smooth, broad in time
             + A_S exp(-(t-t_S)^2 / tau_S^2) cos(omega_S t + phi)
                                                          # short Gabor packet

Linearity of the wave equation makes this equivalent to solving twice
(once with the long source, once with the short) and adding — but a
single solve is half the cost. The resulting T-field combines the
long-wave robustness (diffraction fills shadows behind narrow gaps)
with short-wave directional sharpness (precise gradient near the
destination). See paper Eq. broadband-source.

This module provides:
  - `solve_one_broadband(...)`: same signature as solve_all.solve_one
    but with extra source-shape parameters.
  - a CLI that solves and saves T-field for one source station.

Validation: see validate_broadband_minimal.py (linearity check
ψ_broadband == w_L * ψ_long + w_S * ψ_short on a 2D toy domain).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import os
_THIS = Path(__file__).resolve().parent
_CACHE_DIR = str(_THIS.parent / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit, lax

from solve_all import (
    build_static_fields, gaussian_source,
    grad_central, lap_var_coef_axis, adv,
    DT, N_STEPS, MIN_PEAK_AMP,
)

THIS_DIR = _THIS


# Default source parameters (paper §2.9.5).
# Long Gaussian: temporally broad -> low frequency -> long wavelength
# Short Gabor: narrow envelope, oscillatory -> high frequency -> short wavelength
DEF_A_L = 1.0
DEF_TAU_L_STEPS = 24       # broad Gaussian
DEF_T0_L_STEPS = 32

DEF_A_S = 1.0
DEF_TAU_S_STEPS = 4        # narrow envelope
DEF_T0_S_STEPS = 16
DEF_OMEGA_S = 6.0          # rad/s — picks short wavelength via lambda ~ 2*pi*c0/omega
DEF_PHI = 0.0


def broadband_amp(t,
                  A_L=DEF_A_L, t0_L=DEF_T0_L_STEPS * DT, tau_L=DEF_TAU_L_STEPS * DT,
                  A_S=DEF_A_S, t0_S=DEF_T0_S_STEPS * DT, tau_S=DEF_TAU_S_STEPS * DT,
                  omega_S=DEF_OMEGA_S, phi=DEF_PHI):
    """Time-domain amplitude at scalar time t.

    Returns: A_L * gauss_L(t) + A_S * gabor_S(t)
    """
    g_L = jnp.exp(-((t - t0_L) ** 2) / (tau_L ** 2))
    g_S = jnp.exp(-((t - t0_S) ** 2) / (tau_S ** 2)) * jnp.cos(omega_S * (t - t0_S) + phi)
    return A_L * g_L + A_S * g_S


def step_kernel_bb(carry, n, c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
                   one_minus_damp, free, dx, dy, dz,
                   A_L, t0_L, tau_L, A_S, t0_S, tau_S, omega_S, phi):
    psi_prev, psi_curr, amp_prev, was_rising, T, src_spatial = carry
    t = n.astype(jnp.float32) * DT
    dt2 = jnp.float32(DT * DT)
    diff = (lap_var_coef_axis(c2_xy, psi_curr, dx, 0)
            + lap_var_coef_axis(c2_xy, psi_curr, dy, 1)
            + lap_var_coef_axis(c2_z, psi_curr, dz, 2))
    src_amp = broadband_amp(t, A_L, t0_L, tau_L, A_S, t0_S, tau_S, omega_S, phi)
    src = src_spatial * src_amp
    psi_dot = (psi_curr - psi_prev) / DT
    adv1 = adv(u, v, w, psi_dot, dx, dy, dz)
    qg = adv(u, v, w, psi_curr, dx, dy, dz)
    adv2 = adv(u, v, w, qg, dx, dy, dz)
    adv_a = (ax_f*grad_central(psi_curr,0,dx)
             + ay_f*grad_central(psi_curr,1,dy)
             + az_f*grad_central(psi_curr,2,dz))
    rhs = diff + src - 2.0*adv1 - adv2 - adv_a
    psi_next = 2.0*psi_curr - psi_prev + dt2 * rhs
    psi_next = psi_next * one_minus_damp * free
    psi_next = psi_next.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    psi_next = psi_next.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    psi_next = psi_next.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

    amp = jnp.abs(psi_next)
    rising_now = amp > amp_prev
    peak_now = (was_rising & (~rising_now)
                & jnp.isinf(T) & (amp_prev > MIN_PEAK_AMP))
    T_new = jnp.where(peak_now, t - DT, T)
    return (psi_curr, psi_next, amp, rising_now, T_new, src_spatial), None


def solve_one_broadband(c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
                        one_minus_damp, free, src_spatial, dx, dy, dz,
                        n_steps,
                        A_L=DEF_A_L, t0_L=DEF_T0_L_STEPS * DT, tau_L=DEF_TAU_L_STEPS * DT,
                        A_S=DEF_A_S, t0_S=DEF_T0_S_STEPS * DT, tau_S=DEF_TAU_S_STEPS * DT,
                        omega_S=DEF_OMEGA_S, phi=DEF_PHI):
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
        return step_kernel_bb(carry, n, c2_xy, c2_z, u, v, w,
                              ax_f, ay_f, az_f, one_minus_damp, free,
                              dx, dy, dz,
                              A_L, t0_L, tau_L, A_S, t0_S, tau_S, omega_S, phi)
    n_arr = jnp.arange(n_steps, dtype=jnp.int32)
    final, _ = lax.scan(step_fn, init, n_arr)
    return final[4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-idx", type=int, required=True)
    p.add_argument("--out", type=str, default="broadband_T.npz")
    p.add_argument("--A-L", type=float, default=DEF_A_L)
    p.add_argument("--A-S", type=float, default=DEF_A_S)
    p.add_argument("--tau-L-steps", type=float, default=DEF_TAU_L_STEPS)
    p.add_argument("--tau-S-steps", type=float, default=DEF_TAU_S_STEPS)
    p.add_argument("--omega-S", type=float, default=DEF_OMEGA_S)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    args = p.parse_args()

    z = np.load(THIS_DIR.parent / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations = z["stations_xy_m"]

    print(f"[domain] {nx}x{ny}x{nz}, {len(stations)} stations")
    mask_jnp = jnp.asarray(mask)
    s = build_static_fields(mask_jnp, dx, dy, dz)
    one_minus_damp = (1.0 - s["damp"]).astype(jnp.float32)

    src_xyz = stations[args.src_idx]
    src_idx = (int(round(src_xyz[0]/dx)),
               int(round(src_xyz[1]/dy)),
               int(round(src_xyz[2]/dz)))
    src_spatial = gaussian_source(mask_jnp.shape, src_idx)
    print(f"[src] station {args.src_idx} at cell {src_idx}")
    print(f"[bb] A_L={args.A_L} tau_L_steps={args.tau_L_steps} "
          f"A_S={args.A_S} tau_S_steps={args.tau_S_steps} "
          f"omega_S={args.omega_S}")

    solve_jit = jit(solve_one_broadband, static_argnums=(11, 12, 13, 14))
    t0 = time.time()
    T = solve_jit(s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
                  s["ax"], s["ay"], s["az"], one_minus_damp,
                  s["free"], src_spatial, dx, dy, dz, args.n_steps,
                  args.A_L, DEF_T0_L_STEPS * DT, args.tau_L_steps * DT,
                  args.A_S, DEF_T0_S_STEPS * DT, args.tau_S_steps * DT,
                  args.omega_S, DEF_PHI)
    jax.block_until_ready(T)
    T_np = np.asarray(jax.device_get(T))
    print(f"[solve] done in {time.time()-t0:.1f}s")

    np.savez_compressed(THIS_DIR / args.out, T_field=T_np,
                        src_idx=np.int32(args.src_idx),
                        A_L=np.float32(args.A_L), A_S=np.float32(args.A_S),
                        tau_L_steps=np.float32(args.tau_L_steps),
                        tau_S_steps=np.float32(args.tau_S_steps),
                        omega_S=np.float32(args.omega_S))
    print(f"[saved] {THIS_DIR / args.out}")


if __name__ == "__main__":
    main()
