"""Solve the wave PDE with each of the 118 stations as source, save the
arrival-time array for each. Per-source checkpointing so the framework
killing the process mid-loop doesn't cost more than the in-progress one.

The clever bit: from each source's full T-field, we can read off T at
all 117 destination stations in O(1), giving the entire 118×118
travel-time matrix in just 118 wave solves (instead of 5498).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import os
# Persistent JIT cache via env vars (set BEFORE importing jax). After the
# first invocation populates the cache, every subsequent invocation skips
# the 50-60s compile tax — critical because the framework hard-kills
# python processes at ~5 min wall-clock.
_THIS = Path(__file__).resolve().parent
_CACHE_DIR = str(_THIS / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)  # float32 — big speed boost
import jax.numpy as jnp
from jax import jit, lax

THIS_DIR = Path(__file__).resolve().parent
# Result npz files live under outputs/ (inputs domain*.npz stay in THIS_DIR)
OUT_DIR = THIS_DIR / "outputs"

# ---- Wave config (matches the small-domain demo's tuned params) -------

C0 = 15.0          # horizontal UAV speed in still air (m/s)
# Paper v3 §2.1.1 Eq. 7: c_uav-up > c_uav-down (vortex ring state — descent
# is slower than ascent for stability). Earlier v1 had this flipped.
# Typical Meituan/DJI UAV: ascend 3-6 m/s, descend 2-5 m/s.
CZ_UP = 6.0        # ascending: faster
CZ_DOWN = 4.0      # descending: slower (vortex ring stability)
WIND = (0.0, 0.0, 0.0)  # no wind in this baseline batch

DT = 0.10
N_STEPS = 4500     # ≈ 450s — covers worst-case 7km diagonal at 15 m/s

PULSE_T0_STEPS = 10
PULSE_TAU_STEPS = 4
SOURCE_SIGMA_CELLS = 1.5
SPONGE_LAYERS = 8
SPONGE_STRENGTH = 0.30
MIN_PEAK_AMP = 1e-6

# Regulatory altitude band (paper §5.9): horizontal speed reduced
# below 30m, ramping to full at 40m. Below 30m → only 5% of cruise
# speed (≈0.75 m/s) so the optimal path is nearly vertical at takeoff/
# landing — matches Meituan ops where horizontal motion is forbidden
# below ~30m.
REG_Z_LOW = 30.0
REG_Z_FULL = 40.0
REG_LOW_FACTOR = 0.05


# ---- Stencil operators (jnp) ------------------------------------------

def grad_central(f, axis, h):
    return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, +1, axis=axis)) / (2.0 * h)


def lap_var_coef_axis(c2, psi, h, axis):
    c2_e = 0.5 * (c2 + jnp.roll(c2, -1, axis=axis))
    c2_w = 0.5 * (c2 + jnp.roll(c2, +1, axis=axis))
    flux_e = c2_e * (jnp.roll(psi, -1, axis=axis) - psi) / h
    flux_w = c2_w * (psi - jnp.roll(psi, +1, axis=axis)) / h
    return (flux_e - flux_w) / h


def adv(qx, qy, qz, psi, dx, dy, dz):
    return (qx * grad_central(psi, 0, dx)
            + qy * grad_central(psi, 1, dy)
            + qz * grad_central(psi, 2, dz))


# ---- Spatial fields (built once, reused across sources) ---------------

def build_static_fields_param(mask, dx, dy, dz,
                              c0=C0, cz_up=CZ_UP, cz_down=CZ_DOWN, wind=WIND):
    """Parameterised version that lets the caller override the UAV speeds
    and wind (used by the coupling-iteration solver — paper §2.10 — which
    updates c0/cz_up/cz_down from observed dynamics each iteration)."""
    nx, ny, nz = mask.shape
    cz0 = 0.5 * (cz_up + cz_down)
    w_grav = 0.5 * (cz_down - cz_up)

    # Regulatory profile (z-only)
    z_centers = (jnp.arange(nz) + 0.5) * dz
    reg = jnp.where(
        z_centers >= REG_Z_FULL, 1.0,
        jnp.where(z_centers < REG_Z_LOW, REG_LOW_FACTOR,
                  REG_LOW_FACTOR + (1.0 - REG_LOW_FACTOR)
                      * (z_centers - REG_Z_LOW)
                      / max(1e-9, REG_Z_FULL - REG_Z_LOW)),
    )[None, None, :]

    free = (~mask).astype(jnp.float32)
    c0_arr = c0 * free * reg
    cz0_arr = cz0 * free
    c2_xy = c0_arr ** 2
    c2_z = cz0_arr ** 2

    # Wind + gravity advection
    u = jnp.full((nx, ny, nz), float(wind[0])) * free
    v = jnp.full((nx, ny, nz), float(wind[1])) * free
    w = jnp.full((nx, ny, nz), float(wind[2])) * free - w_grav

    # Steady-flow acceleration term (q·∇)q
    ax = u*grad_central(u,0,dx) + v*grad_central(u,1,dy) + w*grad_central(u,2,dz)
    ay = u*grad_central(v,0,dx) + v*grad_central(v,1,dy) + w*grad_central(v,2,dz)
    az = u*grad_central(w,0,dx) + v*grad_central(w,1,dy) + w*grad_central(w,2,dz)

    # Sponge (lateral + top only — let ground reflect freely)
    def edge_ramp(n):
        d = jnp.minimum(jnp.arange(n), n - 1 - jnp.arange(n))
        return jnp.where(d < SPONGE_LAYERS,
                         (1.0 - d / max(1, SPONGE_LAYERS - 1)) ** 2, 0.0)
    def top_only(n):
        d = n - 1 - jnp.arange(n)
        return jnp.where(d < SPONGE_LAYERS,
                         (1.0 - d / max(1, SPONGE_LAYERS - 1)) ** 2, 0.0)
    rx = edge_ramp(nx)[:, None, None]
    ry = edge_ramp(ny)[None, :, None]
    rz = top_only(nz)[None, None, :]
    damp = SPONGE_STRENGTH * jnp.maximum(jnp.maximum(rx, ry), rz)

    return dict(c2_xy=c2_xy, c2_z=c2_z, u=u, v=v, w=w,
                ax=ax, ay=ay, az=az, damp=damp, free=free)


def build_static_fields(mask, dx, dy, dz):
    """Original module-constant interface — preserved so existing call
    sites in solve_one_source/build_database/exp2_aircraft etc. don't
    need to change."""
    return build_static_fields_param(mask, dx, dy, dz)


# ---- Per-source solve --------------------------------------------------

def step_kernel(carry, n, c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
                one_minus_damp, free, dx, dy, dz):
    psi_prev, psi_curr, amp_prev, was_rising, T, src_spatial = carry
    t = n.astype(jnp.float32) * DT
    pulse_t0 = PULSE_T0_STEPS * DT
    pulse_tau = PULSE_TAU_STEPS * DT
    dt2 = jnp.float32(DT * DT)
    diff = (lap_var_coef_axis(c2_xy, psi_curr, dx, 0)
            + lap_var_coef_axis(c2_xy, psi_curr, dy, 1)
            + lap_var_coef_axis(c2_z, psi_curr, dz, 2))
    src = src_spatial * jnp.exp(-0.5 * ((t - pulse_t0) / pulse_tau) ** 2)
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
                & (amp_prev > MIN_PEAK_AMP))
    T_new = jnp.where(peak_now, t - DT, T)
    return (psi_curr, psi_next, amp, rising_now, T_new, src_spatial), None


def gaussian_source(shape, idx, sigma=SOURCE_SIGMA_CELLS):
    nx, ny, nz = shape
    ix = jnp.arange(nx)[:, None, None]
    iy = jnp.arange(ny)[None, :, None]
    iz = jnp.arange(nz)[None, None, :]
    r2 = ((ix - idx[0])**2 + (iy - idx[1])**2 + (iz - idx[2])**2)
    g = jnp.exp(-0.5 * r2 / (sigma**2))
    return (g / g.sum()).astype(jnp.float32)


def solve_one(c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
              one_minus_damp, free, src_spatial, dx, dy, dz, n_steps):
    """All-explicit-arg version: every dynamic dependency is a positional
    arg. Closures are avoided so the JIT cache hits across sources."""
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
    return final[4]   # T field


# ---- Main loop ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=118)
    parser.add_argument("--time-budget-s", type=float, default=240.0,
                        help="Stop launching new sources after this much "
                             "elapsed time so the framework doesn't kill us "
                             "mid-source. Framework kill ≈ 5 min, so default "
                             "is 4 min.")
    args = parser.parse_args()

    print("[load] domain.npz")
    z = np.load(THIS_DIR / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations_xy_m = z["stations_xy_m"]   # (N, 3) in metres
    n_st = len(stations_xy_m)
    print(f"[domain] {nx}×{ny}×{nz} cells, {n_st} stations")

    # Build static fields once (reused across all sources).
    print("[jit] compiling static fields + step kernel...")
    mask_jnp = jnp.asarray(mask)
    static = build_static_fields(mask_jnp, dx, dy, dz)

    # Pre-compute station integer indices and their float positions.
    st_ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    st_iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    st_iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)
    print(f"[stations] z idx range: [{st_iz.min()}, {st_iz.max()}]")

    # Some block centroids and hex patches happen to fall inside a
    # building footprint (≈20% of stations). Last-mile pickup has to
    # be on the ground, so nudge HORIZONTALLY to the nearest free cell
    # at the same altitude — only fall back to climbing if there's no
    # free cell within 5 cells (≈150m).
    inside_b = mask[st_ix, st_iy, st_iz]
    if inside_b.any():
        bad = np.where(inside_b)[0]
        n_h_nudge = 0
        n_v_nudge = 0
        for b in bad:
            ix0, iy0, iz0 = st_ix[b], st_iy[b], st_iz[b]
            found = False
            for r in range(1, 6):
                # Try cells in a square shell of radius r around (ix0, iy0)
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        if max(abs(di), abs(dj)) != r:
                            continue
                        ix = ix0 + di; iy = iy0 + dj
                        if not (0 < ix < nx - 1 and 0 < iy < ny - 1):
                            continue
                        if not mask[ix, iy, iz0]:
                            st_ix[b], st_iy[b] = ix, iy
                            found = True; break
                    if found:
                        break
                if found:
                    break
            if not found:
                # Fallback: climb in z
                iz = iz0
                while iz < nz - 1 and mask[ix0, iy0, iz]:
                    iz += 1
                st_iz[b] = iz
                n_v_nudge += 1
            else:
                n_h_nudge += 1
        print(f"[stations] {n_h_nudge} horizontally nudged "
              f"to nearest free cell, {n_v_nudge} climbed up")

    # JIT-compile (n_steps fixed, dx/dy/dz fixed → static).
    # Pre-extract static fields so we pass plain jnp arrays into the JIT.
    one_minus_damp = (1.0 - static["damp"]).astype(jnp.float32)
    s_c2_xy = static["c2_xy"]; s_c2_z = static["c2_z"]
    s_u = static["u"]; s_v = static["v"]; s_w = static["w"]
    s_ax = static["ax"]; s_ay = static["ay"]; s_az = static["az"]
    s_free = static["free"]
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))

    OUT_DIR.mkdir(exist_ok=True)
    out_npz = OUT_DIR / "pair_times.npz"
    if out_npz.exists():
        z = np.load(out_npz)
        T_at_stations = z["T_at_stations"]
        done_mask = z["done"]
        print(f"[resume] {done_mask.sum()}/{n_st} sources already done")
    else:
        T_at_stations = np.full((n_st, n_st), np.inf, dtype=np.float32)
        done_mask = np.zeros(n_st, dtype=bool)

    print("[jit] first call will compile (~1 min)...")
    t_start = time.time()

    for i in range(args.start, min(args.end, n_st)):
        if done_mask[i]:
            continue
        elapsed = time.time() - t_start
        if elapsed > args.time_budget_s:
            print(f"[budget] {elapsed:.0f}s elapsed; stopping before "
                  f"source {i} to leave headroom for save")
            break

        print(f"  -- entering source {i} --", flush=True)
        t0 = time.time()
        idx = (int(st_ix[i]), int(st_iy[i]), int(st_iz[i]))
        print(f"     idx {idx}", flush=True)
        ta = time.time()
        src_spatial = gaussian_source(mask_jnp.shape, idx)
        jax.block_until_ready(src_spatial)
        tb = time.time()
        print(f"     src_spatial computed in {tb-ta:.2f}s, "
              f"calling solve_jit...", flush=True)
        T = solve_jit(s_c2_xy, s_c2_z, s_u, s_v, s_w,
                      s_ax, s_ay, s_az,
                      one_minus_damp, s_free, src_spatial,
                      dx, dy, dz, N_STEPS)
        jax.block_until_ready(T)
        tc = time.time()
        print(f"     solve done in {tc-tb:.2f}s, copying...", flush=True)
        T_np = np.asarray(jax.device_get(T))
        td = time.time()
        print(f"    [timing] gauss {tb-ta:.2f}s, solve {tc-tb:.2f}s, "
              f"copy {td-tc:.2f}s", flush=True)
        # Read T at all station cells. Use trilinear-ish nearest cell.
        T_vec = T_np[st_ix, st_iy, st_iz]
        T_at_stations[i, :] = T_vec
        done_mask[i] = True
        elapsed_src = time.time() - t0
        n_arrived = int(np.isfinite(T_vec).sum())
        print(f"  source {i:3d}/{n_st}  ({elapsed_src:5.1f}s)  "
              f"reached {n_arrived-1}/{n_st-1} other stations  "
              f"min={np.nanmin(T_vec[T_vec > 0]):.0f}s  "
              f"max={np.nanmax(T_vec[np.isfinite(T_vec)]):.0f}s")

        # Checkpoint every source — IO is ~10ms, framework kills at ~5min.
        if True:
            np.savez_compressed(
                out_npz, T_at_stations=T_at_stations, done=done_mask,
                station_xy_m=stations_xy_m,
                dx=dx, dy=dy, dz=dz, n_steps=N_STEPS,
            )
            print(f"  [chk] saved {out_npz.name}, {done_mask.sum()}/{n_st} done")

    np.savez_compressed(
        out_npz, T_at_stations=T_at_stations, done=done_mask,
        station_xy_m=stations_xy_m,
        dx=dx, dy=dy, dz=dz, n_steps=N_STEPS,
    )
    total = time.time() - t_start
    print(f"\n[done] processed {done_mask.sum()}/{n_st} sources in "
          f"{total:.0f}s = {total/60:.1f} min  "
          f"(~{total/max(1, done_mask.sum()):.1f}s per source)")


if __name__ == "__main__":
    main()
