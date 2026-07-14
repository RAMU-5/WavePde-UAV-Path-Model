"""Experiment 2: regional fixed-wing aircraft routing with pollution
clouds — at proper aircraft scale.

Differences vs the multi-rotor UAV setup:
  * Domain is 80×80×12 km (regional flight scale), dx=400m dz=600m,
    so the cruise altitude (8 km) is well above the planetary boundary
    layer and matches typical regional jet operations.
  * Aircraft cruise speed c0 = 200 m/s (typical jet),
    climb rate cz_up = 25 m/s, descent cz_down = 20 m/s.
  * Source pulse and source-spread are scaled accordingly so the
    guidance field is smooth at the multi-km scale relevant for
    aircraft turning radii (R_min ≈ 5 km for airliner).
  * Phase 1 climb-out: 30 km horizontal, climbing from runway alt
    to 8 km cruise (6° climb angle, conservative).
  * No buildings — irrelevant at this altitude.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_THIS.parent / "_jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit, lax
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import model.solve_all as SA

ROOT = _THIS.parent
OUT = ROOT / "output" / "figures" / "3d_aircraft"
OUT.mkdir(parents=True, exist_ok=True)


# ===== Domain (regional aircraft scale) ==============================
NX, NY, NZ = 200, 200, 20            # cells
DX, DY, DZ = 400.0, 400.0, 600.0     # metres → 80 × 80 × 12 km
LX, LY, LZ = NX * DX, NY * DY, NZ * DZ
DT = 0.1                              # s (matches solve_all)


# ===== Aircraft physics ==============================================
AC_C0 = 200.0       # cruise horizontal speed (m/s) — typical jet
AC_CZ_UP = 25.0     # climb rate (m/s) ≈ 5000 ft/min
AC_CZ_DOWN = 20.0   # descent rate (m/s)
# Asymmetry → effective gravity term in solve_all:
#   c_z0 = (c_up + c_down)/2 = 22.5
#   w_grav = (c_down − c_up)/2 = −2.5  (so wave is biased downward)


# ===== Scenario ======================================================
AIRPORT_XYZ_M = (8000.0, 8000.0, 0.0)        # runway start (sea level)
RUNWAY_DIR = np.array([1.0, 0.0, 0.0])

CRUISE_ALT_M = 8000.0                         # 8 km cruise (FL260)
PHASE1_LENGTH_M = 30000.0                     # 30 km climb-out
# 8 km climb over 30 km horizontal → 15° climb angle (steep but works
# for the wave-routing demo; commercial would be 5–7°).

# Destinations at cruise altitude — these represent en-route hand-off
# fixes (where the aircraft passes control to the destination ATC for
# its own descent procedure). The aircraft never descends in phase 2.
DST_CANDIDATES_M = [
    # Each destination ≥ 2 σ_xy from EVERY cloud centre — well outside
    # the cloud cores so paths can land in clear airspace.
    (74000.0, 72000.0, 8000.0),   # NE  — 34 km from cloud C
    (74000.0, 12000.0, 8000.0),   # E   — 25 km from cloud B
    (72000.0,  6000.0, 8000.0),   # SE  — 28 km from cloud B
    ( 5000.0, 72000.0, 8000.0),   # NW  — 44 km from cloud C, 46 km from A
]

# Pollution clouds — ellipsoidal: σ_xy ≫ σ_z, but σ_z large enough that
# the cloud TOP reaches cruise altitude (8 km). Physically these
# represent wide low-altitude phenomena that pollute the column up to
# cruise:
#   • wildfire pyroCb / smoke plumes: 5 km wide × 4–10 km tall
#   • volcanic ash trails: 5 km wide × 5–15 km tall
#   • Saharan dust columns: 10–20 km wide × 3–5 km tall
# Aircraft cruise (8 km) is at the upper edge of these layers, so
# routing must avoid them horizontally even though their bulk is below.
CLOUDS = [
    # Wildfire smoke plume (wide, mid-tall)
    {"centre": (32000.0, 35000.0, 5000.0),
     "sigma_xy": 18000.0, "sigma_z": 3500.0, "alpha": 0.55},
    # Volcanic ash trail (narrower, taller — strongest at cruise)
    {"centre": (55000.0, 28000.0, 6500.0),
     "sigma_xy": 10000.0, "sigma_z": 3000.0, "alpha": 0.65},
    # Saharan dust column (wide, mid-altitude)
    {"centre": (45000.0, 55000.0, 4500.0),
     "sigma_xy": 15000.0, "sigma_z": 3000.0, "alpha": 0.45},
]


# ===== Solver knobs ==================================================
AC_N_STEPS = 12000  # 1200 s sim — gives wave time to propagate THROUGH
                    # low-α cloud layers up to cruise altitude.

# Wavelength: spatial source σ ≈ 1.5 km so guidance field is smooth at
# multi-km. Temporal pulse τ stays short so first-peak detection is
# clean.
AC_PULSE_TAU = 6           # τ_t = 0.6 s → λ_temp ≈ 2π·c·τ ≈ 750 m (still grid-limited; spatial σ dominates)
AC_SOURCE_SIGMA_CELLS = 4.0   # 4 cells × 400 m = 1.6 km source spread

# Path smoothing — Dubins-lite. σ_path = 5 km → R_min ≈ 5 km
# (matches airliner turning radius at typical bank angle).
PATH_SMOOTH_SIGMA_M = 5000.0


# ===== 2D wave kernel (cruise-altitude routing) =======================

def make_2d_solve(pulse_tau_steps: int, n_steps: int):
    """Standalone 2D wave solver for routing at a fixed altitude.
    The 3D-with-floor approach kept hitting numerical artefacts because
    the 3D T-field at cruise altitude was a partial slice; this is
    cleaner and faster (≪ 3D cost) and gives a single well-defined T
    field on the cruise plane."""
    pulse_t0 = SA.PULSE_T0_STEPS * DT
    pulse_tau = pulse_tau_steps * DT

    def grad_x(f, h):
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, +1, axis=0)) / (2.0 * h)

    def grad_y(f, h):
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, +1, axis=1)) / (2.0 * h)

    def lap_var(c2, psi, h, axis):
        c2_e = 0.5 * (c2 + jnp.roll(c2, -1, axis=axis))
        c2_w = 0.5 * (c2 + jnp.roll(c2, +1, axis=axis))
        flux_e = c2_e * (jnp.roll(psi, -1, axis=axis) - psi) / h
        flux_w = c2_w * (psi - jnp.roll(psi, +1, axis=axis)) / h
        return (flux_e - flux_w) / h

    def step(carry, n, c2, one_minus_damp, dx, dy):
        psi_prev, psi_curr, amp_prev, was_rising, T, src_spatial = carry
        t = n.astype(jnp.float32) * DT
        dt2 = jnp.float32(DT * DT)
        diff = lap_var(c2, psi_curr, dx, 0) + lap_var(c2, psi_curr, dy, 1)
        src = src_spatial * jnp.exp(-0.5 * ((t - pulse_t0) / pulse_tau) ** 2)
        rhs = diff + src
        psi_next = 2.0 * psi_curr - psi_prev + dt2 * rhs
        psi_next = psi_next * one_minus_damp
        psi_next = psi_next.at[0, :].set(0.0)
        psi_next = psi_next.at[-1, :].set(0.0)
        psi_next = psi_next.at[:, 0].set(0.0)
        psi_next = psi_next.at[:, -1].set(0.0)
        amp = jnp.abs(psi_next)
        rising_now = amp > amp_prev
        peak_now = (was_rising & (~rising_now) & jnp.isinf(T)
                    & (amp_prev > SA.MIN_PEAK_AMP))
        T_new = jnp.where(peak_now, t - DT, T)
        return (psi_curr, psi_next, amp, rising_now, T_new, src_spatial), None

    def solve(c2, one_minus_damp, src_spatial, dx, dy):
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
            return step(carry, n, c2, one_minus_damp, dx, dy)
        n_arr = jnp.arange(n_steps, dtype=jnp.int32)
        final, _ = lax.scan(step_fn, init, n_arr)
        return final[4]

    return jit(solve)


def build_static_2d(clouds, cruise_alt):
    """2D c² field at cruise altitude, with anisotropic cloud
    suppression projected to z=cruise."""
    ix = jnp.arange(NX)[:, None] * DX
    iy = jnp.arange(NY)[None, :] * DY
    c0_arr = jnp.full((NX, NY), AC_C0, dtype=jnp.float32)
    if clouds:
        supp = jnp.ones((NX, NY), dtype=jnp.float32)
        for cl in clouds:
            cx, cy, cz_ = cl["centre"]
            sxy = cl["sigma_xy"]
            sz = cl["sigma_z"]
            # 2D suppression at z=cruise: integrate Gaussian z-profile
            # at cruise altitude. The xy Gaussian stays the same.
            z_falloff = float(np.exp(-0.5 * ((cruise_alt - cz_) / sz) ** 2))
            r2_xy = ((ix - cx) / sxy) ** 2 + ((iy - cy) / sxy) ** 2
            supp = supp * (1.0 - cl["alpha"] * z_falloff
                           * jnp.exp(-0.5 * r2_xy))
        supp = jnp.clip(supp, 0.05, 1.0).astype(jnp.float32)
        c0_arr = c0_arr * supp
    c2 = c0_arr ** 2

    def edge_ramp(n):
        d = jnp.minimum(jnp.arange(n), n - 1 - jnp.arange(n))
        return jnp.where(d < SA.SPONGE_LAYERS,
                         (1.0 - d / max(1, SA.SPONGE_LAYERS - 1)) ** 2, 0.0)
    rx = edge_ramp(NX)[:, None]
    ry = edge_ramp(NY)[None, :]
    damp = SA.SPONGE_STRENGTH * jnp.maximum(rx, ry)
    one_minus_damp = (1.0 - damp).astype(jnp.float32)

    return c2, one_minus_damp


def gaussian_source_2d(shape, idx_xy, sigma_cells):
    nx, ny = shape
    ix = jnp.arange(nx)[:, None]
    iy = jnp.arange(ny)[None, :]
    r2 = (ix - idx_xy[0]) ** 2 + (iy - idx_xy[1]) ** 2
    g = jnp.exp(-0.5 * r2 / (sigma_cells ** 2))
    return (g / g.sum()).astype(jnp.float32)


# ===== Wave kernel (parametrised for aircraft physics) ===============

def make_aircraft_solve(pulse_tau_steps: int, n_steps: int):
    """Wave PDE kernel with our own constants, not solve_all's globals."""
    pulse_t0 = SA.PULSE_T0_STEPS * DT
    pulse_tau = pulse_tau_steps * DT

    def step_kernel(carry, n, c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
                    one_minus_damp, free, dx, dy, dz):
        psi_prev, psi_curr, amp_prev, was_rising, T, src_spatial = carry
        t = n.astype(jnp.float32) * DT
        dt2 = jnp.float32(DT * DT)
        diff = (SA.lap_var_coef_axis(c2_xy, psi_curr, dx, 0)
                + SA.lap_var_coef_axis(c2_xy, psi_curr, dy, 1)
                + SA.lap_var_coef_axis(c2_z, psi_curr, dz, 2))
        src = src_spatial * jnp.exp(-0.5 * ((t - pulse_t0) / pulse_tau) ** 2)
        psi_dot = (psi_curr - psi_prev) / DT
        adv1 = SA.adv(u, v, w, psi_dot, dx, dy, dz)
        qg = SA.adv(u, v, w, psi_curr, dx, dy, dz)
        adv2 = SA.adv(u, v, w, qg, dx, dy, dz)
        adv_a = (ax_f * SA.grad_central(psi_curr, 0, dx)
                 + ay_f * SA.grad_central(psi_curr, 1, dy)
                 + az_f * SA.grad_central(psi_curr, 2, dz))
        rhs = diff + src - 2.0 * adv1 - adv2 - adv_a
        psi_next = 2.0 * psi_curr - psi_prev + dt2 * rhs
        psi_next = psi_next * one_minus_damp * free
        psi_next = psi_next.at[0, :, :].set(0.0)
        psi_next = psi_next.at[-1, :, :].set(0.0)
        psi_next = psi_next.at[:, 0, :].set(0.0)
        psi_next = psi_next.at[:, -1, :].set(0.0)
        psi_next = psi_next.at[:, :, 0].set(0.0)
        psi_next = psi_next.at[:, :, -1].set(0.0)
        amp = jnp.abs(psi_next)
        rising_now = amp > amp_prev
        peak_now = (was_rising & (~rising_now)
                    & jnp.isinf(T)
                    & (amp_prev > SA.MIN_PEAK_AMP))
        T_new = jnp.where(peak_now, t - DT, T)
        return (psi_curr, psi_next, amp, rising_now, T_new, src_spatial), None

    def solve(c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
              one_minus_damp, free, src_spatial, dx, dy, dz):
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
            return step_kernel(carry, n, c2_xy, c2_z, u, v, w,
                               ax_f, ay_f, az_f, one_minus_damp, free,
                               dx, dy, dz)
        n_arr = jnp.arange(n_steps, dtype=jnp.int32)
        final, _ = lax.scan(step_fn, init, n_arr)
        return final[4]

    return jit(solve)


def gaussian_source_3d(shape, idx, sigma_cells):
    nx, ny, nz = shape
    ix = jnp.arange(nx)[:, None, None]
    iy = jnp.arange(ny)[None, :, None]
    iz = jnp.arange(nz)[None, None, :]
    r2 = ((ix - idx[0]) ** 2 + (iy - idx[1]) ** 2 + (iz - idx[2]) ** 2)
    g = jnp.exp(-0.5 * r2 / (sigma_cells ** 2))
    return (g / g.sum()).astype(jnp.float32)


# ===== Static fields (aircraft physics + clouds) =====================

def build_static_aircraft(clouds):
    cz0 = 0.5 * (AC_CZ_UP + AC_CZ_DOWN)
    w_grav = 0.5 * (AC_CZ_DOWN - AC_CZ_UP)

    free = jnp.ones((NX, NY, NZ), dtype=jnp.float32)
    free = free.at[:, :, 0].set(0.0)

    c0_arr = AC_C0 * free
    cz0_arr = cz0 * free

    if clouds:
        ix = jnp.arange(NX)[:, None, None] * DX
        iy = jnp.arange(NY)[None, :, None] * DY
        iz = jnp.arange(NZ)[None, None, :] * DZ
        supp = jnp.ones((NX, NY, NZ), dtype=jnp.float32)
        for cl in clouds:
            cx, cy, cz_ = cl["centre"]
            sxy = cl["sigma_xy"]
            sz = cl["sigma_z"]
            # Anisotropic Gaussian: pancake-like ellipsoid
            r2_norm = (((ix - cx) / sxy) ** 2
                       + ((iy - cy) / sxy) ** 2
                       + ((iz - cz_) / sz) ** 2)
            supp = supp * (1.0 - cl["alpha"] * jnp.exp(-0.5 * r2_norm))
        supp = jnp.clip(supp, 0.05, 1.0).astype(jnp.float32)
        c0_arr = c0_arr * supp
        cz0_arr = cz0_arr * supp

    c2_xy = c0_arr ** 2
    c2_z = cz0_arr ** 2

    u = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
    v = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
    w = jnp.full((NX, NY, NZ), float(-w_grav), dtype=jnp.float32) * free

    ax_f = jnp.zeros_like(u); ay_f = jnp.zeros_like(u); az_f = jnp.zeros_like(u)

    def edge_ramp(n):
        d = jnp.minimum(jnp.arange(n), n - 1 - jnp.arange(n))
        return jnp.where(d < SA.SPONGE_LAYERS,
                         (1.0 - d / max(1, SA.SPONGE_LAYERS - 1)) ** 2, 0.0)

    # NO top-z sponge — would damp the wave at z=10–12 km where the
    # path tracer needs T-info to find climb-over corridors.
    rx = edge_ramp(NX)[:, None, None]
    ry = edge_ramp(NY)[None, :, None]
    damp = SA.SPONGE_STRENGTH * jnp.maximum(rx, ry)

    return dict(c2_xy=c2_xy, c2_z=c2_z, u=u, v=v, w=w,
                ax=ax_f, ay=ay_f, az=az_f, damp=damp, free=free)


# ===== Path smoothing (Dubins-lite) ==================================

def smooth_path(path_xyz, sigma_m=PATH_SMOOTH_SIGMA_M, step_size=None):
    if path_xyz.shape[0] < 4:
        return path_xyz
    seg = np.diff(path_xyz, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    if step_size is None:
        step_size = max(50.0, sigma_m / 12.0)
    s_new = np.arange(0.0, arc[-1] + 1e-6, step_size)
    if s_new.size < 4:
        return path_xyz
    rs = np.column_stack([
        np.interp(s_new, arc, path_xyz[:, 0]),
        np.interp(s_new, arc, path_xyz[:, 1]),
        np.interp(s_new, arc, path_xyz[:, 2]),
    ])
    sigma_steps = max(1.5, sigma_m / step_size)
    half = int(np.ceil(3.0 * sigma_steps))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_steps) ** 2)
    k = k / k.sum()
    out = np.empty_like(rs)
    for axis in range(3):
        v = rs[:, axis]
        if half > 0:
            head = v[half:0:-1]
            tail = v[-2:-half - 2:-1]
            padded = np.concatenate([head, v, tail])
            if padded.size != v.size + 2 * half:
                padded = np.pad(v, half, mode="edge")
        else:
            padded = v
        out[:, axis] = np.convolve(padded, k, mode="valid")[: v.size]
    out[0] = rs[0]; out[-1] = rs[-1]
    return out


# ===== Two-phase trace ===============================================

def trace_aircraft(T_field, src_xyz_m, dst_xyz_m, dx, dy, dz,
                   runway_dir, phase1_length, cruise_alt,
                   gamma=0.3, max_steps=40000):
    """Phase 1 = straight runway climb-out (3D, ground → cruise alt).
    Phase 2 = 3D gradient-descent with an altitude floor:
        z >= cruise_alt at all times.
    The aircraft can fly HIGHER (over pollution clouds that reach into
    cruise altitude), but must never descend below cruise. This is
    enforced after each step by clamping pos[2] = max(pos[2], cruise_alt).
    """
    src = np.array(src_xyz_m, dtype=np.float64)
    dst = np.array(dst_xyz_m, dtype=np.float64)

    # Phase 1 — straight 3D climb along runway
    n_p1 = max(2, int(phase1_length / (gamma * min(dx, dy, dz))))
    s_arr = np.linspace(0.0, phase1_length, n_p1)
    x_p1 = src[0] + runway_dir[0] * s_arr
    y_p1 = src[1] + runway_dir[1] * s_arr
    z_p1 = np.linspace(src[2], cruise_alt, n_p1)
    phase1 = np.column_stack([x_p1, y_p1, z_p1])
    align_xyz = phase1[-1]   # at cruise altitude

    # Phase 2 — full 3D gradient descent with altitude FLOOR only:
    #     z >= cruise_alt   (aircraft cannot descend)
    #     z <= top of usable domain   (no artificial ceiling)
    # This lets the routing climb ABOVE cruise to fly over clouds whose
    # tops reach into the cruise layer.
    finite = np.isfinite(T_field)
    if not finite.any():
        empty = np.empty((0, 3))
        return phase1, "no_finite", phase1, empty, empty
    T_max = T_field[finite].max()
    from scipy.ndimage import distance_transform_edt
    inf_mask = ~finite
    dist_to_finite = distance_transform_edt(inf_mask)
    fill_alpha = T_max / 30.0
    T_safe = np.where(finite, T_field, T_max + fill_alpha * dist_to_finite)
    gx, gy, gz = np.gradient(T_safe, dx, dy, dz)

    step = gamma * min(dx, dy, dz)
    pos = dst.copy()
    path2 = [pos.copy()]
    status = "max_steps"
    align_xy = align_xyz[:2]
    best_d_align = np.inf
    best_idx = 0
    stale = 0
    floor_hits = 0
    z_max_allowed = (T_field.shape[2] - 2) * dz
    for _ in range(max_steps):
        coords = np.array([[pos[0] / dx], [pos[1] / dy], [pos[2] / dz]])
        gv = np.array([
            float(map_coordinates(gx, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gy, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gz, coords, order=1, mode="nearest")[0]),
        ])
        # Block downward motion at the cruise floor.
        on_floor = pos[2] <= cruise_alt + 0.5 * dz
        if on_floor and gv[2] > 0.0:   # gv[2]>0 → motion -gv goes down
            gv[2] = 0.0
            floor_hits += 1
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            status = "zero_grad"; break
        pos = pos - step * gv / gn
        if pos[2] < cruise_alt:
            pos[2] = cruise_alt
        if pos[2] > z_max_allowed:
            pos[2] = z_max_allowed
        path2.append(pos.copy())
        d_a = np.linalg.norm(pos[:2] - align_xy)
        if d_a < best_d_align - 0.5 * step:
            best_d_align = d_a; best_idx = len(path2) - 1; stale = 0
        else:
            stale += 1
        if d_a < 2.0 * max(dx, dy):
            status = "reached_align"; break
        if stale > 4000:
            status = f"stale_align{best_d_align:.0f}m"; break
    full_back = np.array(path2[::-1])
    cut_back = max(0, len(full_back) - 1 - best_idx)
    phase2_raw = full_back[cut_back:]
    phase2_raw = np.vstack([align_xyz[None, :],
                            phase2_raw[1:] if phase2_raw.shape[0] > 1
                            else phase2_raw])
    phase2 = smooth_path(phase2_raw, sigma_m=PATH_SMOOTH_SIGMA_M)
    if phase2.size:
        phase2[:, 2] = np.maximum(phase2[:, 2], cruise_alt)
        phase2[0] = align_xyz
    full = np.vstack([phase1, phase2])
    status = (f"{status}_align{best_d_align:.0f}m_"
              f"floor{floor_hits}_zmax{phase2[:,2].max() if phase2.size else 0:.0f}m")
    return full, status, phase1, phase2, phase2_raw


def _curv_max(path):
    if path.shape[0] < 5:
        return 0.0
    seg = np.diff(path[:, :2], axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len = np.maximum(seg_len, 1e-6)
    tan = seg / seg_len[:, None]
    dt = np.diff(tan, axis=0)
    dn = np.linalg.norm(dt, axis=1)
    midlen = 0.5 * (seg_len[:-1] + seg_len[1:])
    return float((dn / midlen).max())


# ===== Main ==========================================================

def main():
    print(f"[domain] {NX}×{NY}×{NZ}  "
          f"({LX/1000:.0f}×{LY/1000:.0f}×{LZ/1000:.0f} km)  "
          f"dx={DX:.0f}m dz={DZ:.0f}m  c0={AC_C0} m/s")

    print("[build] 3D static fields with cloud suppression (anisotropic)")
    static = build_static_aircraft(CLOUDS)
    one_minus_damp = (1.0 - static["damp"]).astype(jnp.float32)

    # Place the 3D source at the Phase-1 endpoint (top-of-climb).
    align_xyz_M = (
        AIRPORT_XYZ_M[0] + RUNWAY_DIR[0] * PHASE1_LENGTH_M,
        AIRPORT_XYZ_M[1] + RUNWAY_DIR[1] * PHASE1_LENGTH_M,
        CRUISE_ALT_M,
    )
    src_idx = (
        int(np.clip(round(align_xyz_M[0] / DX), 1, NX - 2)),
        int(np.clip(round(align_xyz_M[1] / DY), 1, NY - 2)),
        int(np.clip(round(align_xyz_M[2] / DZ), 1, NZ - 2)),
    )
    print(f"[source] top-of-climb idx {src_idx} = "
          f"({src_idx[0]*DX/1000}, {src_idx[1]*DY/1000}, "
          f"{src_idx[2]*DZ/1000}) km  "
          f"(airport ({AIRPORT_XYZ_M[0]/1000:.0f}, "
          f"{AIRPORT_XYZ_M[1]/1000:.0f}) km, "
          f"climb-out {PHASE1_LENGTH_M/1000:.0f} km)")

    print(f"[solve 3D] τ={AC_PULSE_TAU} steps, σ={AC_SOURCE_SIGMA_CELLS} cells, "
          f"N={AC_N_STEPS} (sim time = {AC_N_STEPS*DT:.0f}s)")
    solve_jit = make_aircraft_solve(AC_PULSE_TAU, AC_N_STEPS)
    sp = gaussian_source_3d((NX, NY, NZ), src_idx,
                            sigma_cells=AC_SOURCE_SIGMA_CELLS)
    jax.block_until_ready(sp)
    t0 = time.time()
    T = solve_jit(static["c2_xy"], static["c2_z"],
                  static["u"], static["v"], static["w"],
                  static["ax"], static["ay"], static["az"],
                  one_minus_damp, static["free"], sp,
                  DX, DY, DZ)
    jax.block_until_ready(T)
    T_np = np.asarray(jax.device_get(T))
    iz_cruise = int(round(CRUISE_ALT_M / DZ))
    print(f"[solve 3D] done in {time.time()-t0:.1f}s; "
          f"T finite frac = {np.isfinite(T_np).mean():.3f}  "
          f"(cruise slice: {np.isfinite(T_np[:,:,iz_cruise]).mean():.3f})")
    for label, xyz in [("near (15,15,8)", (15, 15, 8)),
                       ("mid  (35,35,8)", (35, 35, 8)),
                       ("over (35,35,10)", (35, 35, 10)),
                       ("far  (65,55,8)", (65, 55, 8))]:
        ix = int(round(xyz[0] * 1000 / DX))
        iy = int(round(xyz[1] * 1000 / DY))
        iz = int(round(xyz[2] * 1000 / DZ))
        print(f"          T@{label} km = {T_np[ix, iy, iz]:.1f}s")

    paths = []
    for dst_m in DST_CANDIDATES_M:
        ix = int(np.clip(round(dst_m[0] / DX), 0, NX - 1))
        iy = int(np.clip(round(dst_m[1] / DY), 0, NY - 1))
        iz = int(np.clip(round(dst_m[2] / DZ), 0, NZ - 1))
        T_at_dst = float(T_np[ix, iy, iz])
        if not np.isfinite(T_at_dst):
            print(f"[skip] dst {dst_m} unreachable (T=inf)")
            continue
        dst_snapped = (ix * DX, iy * DY, iz * DZ)
        full, status, p1, p2, p2_raw = trace_aircraft(
            T_np, AIRPORT_XYZ_M, dst_snapped, DX, DY, DZ,
            RUNWAY_DIR, PHASE1_LENGTH_M, CRUISE_ALT_M,
        )
        # In-1σ check using anisotropic norm
        n_in = 0
        for cl in CLOUDS:
            cx, cy, cz_ = cl["centre"]
            r_norm = np.sqrt(((full[:, 0] - cx) / cl["sigma_xy"]) ** 2
                             + ((full[:, 1] - cy) / cl["sigma_xy"]) ** 2
                             + ((full[:, 2] - cz_) / cl["sigma_z"]) ** 2)
            n_in += int((r_norm < 1.0).sum())
        kr = _curv_max(p2_raw); ks = _curv_max(p2)
        Rr = 1 / max(kr, 1e-9); Rs = 1 / max(ks, 1e-9)
        print(f"[trace] dst {tuple(round(x/1000,1) for x in dst_m)}km  "
              f"status={status} T={T_at_dst:.0f}s  in-1σ-cloud={n_in}  "
              f"R_min raw={Rr/1000:.2f}km → smooth={Rs/1000:.2f}km")
        paths.append((dst_snapped, full, T_at_dst, p1, p2, p2_raw))

    if not paths:
        print("[fail] no reachable destinations")
        return

    # ===== Plot ======================================================
    fig = plt.figure(figsize=(20, 13))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axtp = fig.add_subplot(2, 2, 2)
    axsv = fig.add_subplot(2, 2, 3)
    axleg = fig.add_subplot(2, 2, 4)
    axleg.axis("off")
    cmap = cm.get_cmap("turbo")

    def _draw_clouds_3d(ax_):
        rng = np.random.default_rng(7)
        for cl in CLOUDS:
            cx, cy, cz_ = cl["centre"]
            sxy = cl["sigma_xy"]; sz = cl["sigma_z"]
            # Anisotropic Gaussian samples for ellipsoid look
            rx = rng.normal(0, sxy, 1500)
            ry = rng.normal(0, sxy, 1500)
            rz = rng.normal(0, sz, 1500)
            r2_norm = (rx / sxy) ** 2 + (ry / sxy) ** 2 + (rz / sz) ** 2
            keep = r2_norm < 1.5 ** 2
            rx, ry, rz = rx[keep], ry[keep], rz[keep]
            ax_.scatter(cx + rx, cy + ry, cz_ + rz,
                        c="#a45ad9", s=10, alpha=0.06,
                        edgecolors="none", depthshade=True)

    # 3D oblique
    _draw_clouds_3d(ax3d)
    for j, (dst_snapped, full, T_at_dst, p1, p2, p2_raw) in enumerate(paths):
        col = cmap(0.15 + 0.7 * j / max(1, len(paths) - 1))
        ax3d.plot(p1[:, 0], p1[:, 1], p1[:, 2], color=col, lw=1.6, ls="--",
                  alpha=0.85)
        ax3d.plot(p2[:, 0], p2[:, 1], p2[:, 2], color=col, lw=2.5, alpha=0.95)
    ax3d.scatter(*AIRPORT_XYZ_M, color="#ffd400", s=240, marker="*",
                 edgecolor="black", linewidth=1.0, zorder=15)
    for cl in CLOUDS:
        ax3d.scatter(*cl["centre"], color="#7028b0", s=80, alpha=0.7,
                     edgecolor="black", linewidth=0.4)
    ax3d.set_xlim(0, LX); ax3d.set_ylim(0, LY); ax3d.set_zlim(0, LZ)
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.set_box_aspect((LX, LY, max(LZ, 0.10 * LX)))
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_title("3D — regional aircraft trajectories around clouds")

    # Top-down — show cloud horizontal σ extent + the EFFECTIVE
    # suppression footprint at cruise altitude (since the σ_z falloff
    # weakens the cloud at z=cruise, the "effective σ_xy" is what
    # matters for routing).
    iz_cruise = int(round(CRUISE_ALT_M / DZ))
    for cl in CLOUDS:
        theta = np.linspace(0, 2 * np.pi, 80)
        # 1σ_xy (full strength) ring
        axtp.plot(cl["centre"][0]/1000 + cl["sigma_xy"]/1000 * np.cos(theta),
                  cl["centre"][1]/1000 + cl["sigma_xy"]/1000 * np.sin(theta),
                  color="#7028b0", lw=1.6, ls="-", alpha=0.6, zorder=4)
        # 2σ_xy ring (dashed)
        axtp.plot(cl["centre"][0]/1000 + 2 * cl["sigma_xy"]/1000 * np.cos(theta),
                  cl["centre"][1]/1000 + 2 * cl["sigma_xy"]/1000 * np.sin(theta),
                  color="#7028b0", lw=1.0, ls="--", alpha=0.6, zorder=4)
        # Effective suppression at cruise altitude (cloud strength × falloff_z)
        falloff = np.exp(-0.5 * ((CRUISE_ALT_M - cl["centre"][2]) / cl["sigma_z"]) ** 2)
        eff_alpha = cl["alpha"] * falloff
        axtp.scatter(cl["centre"][0]/1000, cl["centre"][1]/1000,
                     color="#7028b0", s=70, edgecolor="black",
                     linewidth=0.5, zorder=5,
                     label=(f"cloud (eff α@cruise = {eff_alpha:.0%})"
                            if cl is CLOUDS[0] else None))
    for j, (dst_snapped, full, T_at_dst, p1, p2, p2_raw) in enumerate(paths):
        col = cmap(0.15 + 0.7 * j / max(1, len(paths) - 1))
        axtp.plot(p1[:, 0]/1000, p1[:, 1]/1000, color=col, lw=1.8,
                  ls="--", alpha=0.85, zorder=7)
        if p2_raw.size:
            axtp.plot(p2_raw[:, 0]/1000, p2_raw[:, 1]/1000, color=col,
                      lw=0.7, ls=":", alpha=0.5, zorder=6)
        axtp.plot(p2[:, 0]/1000, p2[:, 1]/1000, color=col, lw=2.4,
                  alpha=0.95, zorder=8)
        axtp.scatter(dst_snapped[0]/1000, dst_snapped[1]/1000,
                     color=col, s=70, marker="v",
                     edgecolor="black", linewidth=0.5, zorder=10)
    axtp.scatter(AIRPORT_XYZ_M[0]/1000, AIRPORT_XYZ_M[1]/1000,
                 color="#ffd400", s=200, marker="*",
                 edgecolor="black", linewidth=1.0, zorder=15)
    axtp.plot([AIRPORT_XYZ_M[0]/1000,
               (AIRPORT_XYZ_M[0] + RUNWAY_DIR[0]*5000)/1000],
              [AIRPORT_XYZ_M[1]/1000,
               (AIRPORT_XYZ_M[1] + RUNWAY_DIR[1]*5000)/1000],
              color="#222", lw=3.5, alpha=0.7, zorder=14)
    axtp.set_xlim(0, LX/1000); axtp.set_ylim(0, LY/1000)
    axtp.set_aspect("equal")
    axtp.set_xlabel("x (km)"); axtp.set_ylabel("y (km)")
    axtp.set_title("Top-down — solid rings 1σ, dashed 2σ; "
                   "—— smoothed, ⋯ raw")
    axtp.grid(True, alpha=0.3)

    # Side view (z vs ground distance)
    for j, (dst_snapped, full, T_at_dst, p1, p2, p2_raw) in enumerate(paths):
        col = cmap(0.15 + 0.7 * j / max(1, len(paths) - 1))
        d1 = np.linalg.norm(p1[:, :2] - np.array(AIRPORT_XYZ_M[:2]), axis=1)
        d2 = np.linalg.norm(p2[:, :2] - np.array(AIRPORT_XYZ_M[:2]), axis=1)
        axsv.plot(d1/1000, p1[:, 2]/1000, color=col, lw=1.8, ls="--",
                  alpha=0.85)
        if p2_raw.size:
            d2r = np.linalg.norm(p2_raw[:, :2] - np.array(AIRPORT_XYZ_M[:2]),
                                 axis=1)
            axsv.plot(d2r/1000, p2_raw[:, 2]/1000, color=col, lw=0.8,
                      ls=":", alpha=0.55, zorder=6)
        axsv.plot(d2/1000, p2[:, 2]/1000, color=col, lw=2.4, alpha=0.95,
                  zorder=8)
        axsv.scatter(d2[-1]/1000, p2[-1, 2]/1000, color=col, s=70,
                     marker="v", edgecolor="black", linewidth=0.5, zorder=10)
    for cl in CLOUDS:
        d_cl = np.linalg.norm(np.array(cl["centre"][:2])
                              - np.array(AIRPORT_XYZ_M[:2]))
        # Ellipse: σ_xy (horizontal) × σ_z (vertical)
        from matplotlib.patches import Ellipse
        axsv.add_patch(Ellipse(
            (d_cl/1000, cl["centre"][2]/1000),
            width=2 * cl["sigma_xy"] / 1000,
            height=2 * cl["sigma_z"] / 1000,
            color="#a45ad9", alpha=0.20, zorder=3))
        axsv.scatter(d_cl/1000, cl["centre"][2]/1000, color="#7028b0",
                     s=60, edgecolor="black", linewidth=0.4, zorder=5)
    axsv.scatter(0, AIRPORT_XYZ_M[2]/1000, color="#ffd400", s=200,
                 marker="*", edgecolor="black", linewidth=1.0, zorder=15)
    axsv.axhline(CRUISE_ALT_M/1000, color="#888", lw=0.8, ls=":")
    axsv.set_xlim(0, np.hypot(LX, LY)/1000)
    axsv.set_ylim(0, LZ/1000)
    axsv.set_xlabel("ground distance from airport (km)")
    axsv.set_ylabel("altitude z (km)")
    axsv.set_title("Side view — z vs distance "
                   "(— smoothed, ⋯ raw, -- phase 1 climb-out)")
    axsv.grid(True, alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#ffd400", markeredgecolor="black",
                   markersize=14, label="airport (runway start)"),
        plt.Line2D([0], [0], color="#222", lw=3, label="runway direction"),
        plt.Line2D([0], [0], color="black", lw=1.6, ls="--",
                   label=f"phase 1 — straight climb-out {PHASE1_LENGTH_M/1000:.0f} km"),
        plt.Line2D([0], [0], color="black", lw=2.4,
                   label="phase 2 — gradient routing −∇T (smoothed)"),
        plt.Line2D([0], [0], color="black", lw=0.8, ls=":",
                   label="phase 2 — raw trace before smoothing"),
    ]
    for ci, cl in enumerate(CLOUDS):
        falloff = np.exp(-0.5 * ((CRUISE_ALT_M - cl["centre"][2])
                                 / cl["sigma_z"]) ** 2)
        eff_alpha = cl["alpha"] * falloff
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
            markerfacecolor="#7028b0", markersize=10,
            label=(f"cloud {ci}: ({cl['centre'][0]/1000:.0f},"
                   f"{cl['centre'][1]/1000:.0f},"
                   f"{cl['centre'][2]/1000:.1f}) km  "
                   f"σxy={cl['sigma_xy']/1000:.0f} σz={cl['sigma_z']/1000:.1f} km  "
                   f"α={cl['alpha']:.0%} (@cruise={eff_alpha:.0%})")))
    for j, (dst_snapped, full, T_at_dst, p1, p2, p2_raw) in enumerate(paths):
        col = cmap(0.15 + 0.7 * j / max(1, len(paths) - 1))
        n_in = 0
        for cl in CLOUDS:
            cx, cy, cz_ = cl["centre"]
            r_norm = np.sqrt(((full[:, 0] - cx) / cl["sigma_xy"]) ** 2
                             + ((full[:, 1] - cy) / cl["sigma_xy"]) ** 2
                             + ((full[:, 2] - cz_) / cl["sigma_z"]) ** 2)
            n_in += int((r_norm < 1.0).sum())
        ks = _curv_max(p2)
        Rs = 1 / max(ks, 1e-9)
        handles.append(plt.Line2D([0], [0], color=col, lw=2.6,
            label=(f"path {j} → ({dst_snapped[0]/1000:.0f},"
                   f"{dst_snapped[1]/1000:.0f},"
                   f"{dst_snapped[2]/1000:.0f}) km  "
                   f"T={T_at_dst:.0f}s  in-1σ={n_in}  "
                   f"R_min={Rs/1000:.1f}km")))
    axleg.legend(handles=handles, loc="center", fontsize=10, frameon=True)

    fig.suptitle(
        "Regional fixed-wing aircraft routing\n"
        f"airport=({AIRPORT_XYZ_M[0]/1000:.0f},{AIRPORT_XYZ_M[1]/1000:.0f},"
        f"{AIRPORT_XYZ_M[2]:.0f}m), runway +x, "
        f"cruise z={CRUISE_ALT_M/1000:.0f}km, "
        f"c0={AC_C0:.0f} m/s, {len(CLOUDS)} pollution clouds",
        fontsize=13, y=0.99)
    out = OUT / "aircraft_regional.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    np.savez_compressed(
        OUT / "aircraft_regional.npz",
        airport_xyz_m=np.array(AIRPORT_XYZ_M, dtype=np.float32),
        runway_dir=RUNWAY_DIR.astype(np.float32),
        cruise_alt_m=np.float32(CRUISE_ALT_M),
        clouds_centre=np.array([cl["centre"] for cl in CLOUDS], dtype=np.float32),
        clouds_sigma_xy=np.array([cl["sigma_xy"] for cl in CLOUDS], dtype=np.float32),
        clouds_sigma_z=np.array([cl["sigma_z"] for cl in CLOUDS], dtype=np.float32),
        clouds_alpha=np.array([cl["alpha"] for cl in CLOUDS], dtype=np.float32),
        c0_aircraft=np.float32(AC_C0),
        n_paths=np.int32(len(paths)),
        **{f"path_{i}_xyz": p[1].astype(np.float32) for i, p in enumerate(paths)},
        **{f"dst_{i}_xyz": np.array(p[0], dtype=np.float32) for i, p in enumerate(paths)},
    )
    print(f"[saved] {OUT/'aircraft_regional.npz'}")


if __name__ == "__main__":
    main()
