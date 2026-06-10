"""Wavelength sweep: run the wave solver with short / medium / long
source pulses from the same station, trace gradient-descent paths to a
handful of destinations, and produce a side-by-side comparison plot.

Wavelength is set by two knobs:
  * PULSE_TAU_STEPS   - temporal Gaussian width of the source pulse
                        (spatial dominant wavelength ≈ 2π c0 τ)
  * SOURCE_SIGMA      - spatial Gaussian width of the source (cells)

We sweep both together to get a clean 'short / medium / long' axis.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_CACHE_DIR = str(_THIS.parent / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
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
from matplotlib.collections import LineCollection

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "model"))
import solve_all as SA


# ---- Sweep config ------------------------------------------------------

# Reduced step count — 3500 steps × 0.1s = 350s sim time. Production
# uses 4500 (450s); we trim a bit for speed.
N_STEPS = 3500

# (label, pulse_tau_steps, source_sigma_cells)
# c0 = 15 m/s, dt = 0.1 s, dx = 50 m
#   short : tau=1  -> τ=0.1s,  λ_temp ≈ 2π·15·0.1 ≈ 9 m   (sub-grid! → grid-limited)
#                     σ_xy = 0.5·dx = 25 m
#   med   : tau=4  -> τ=0.4s,  λ_temp ≈ 38 m,  σ_xy = 1.5·dx = 75 m
#   long  : tau=20 -> τ=2.0s,  λ_temp ≈ 188 m, σ_xy = 6·dx = 300 m
SWEEP = [
    ("short", 1, 0.5),
    ("medium", 4, 1.5),
    ("long",  20, 6.0),
]

SOURCE_IDX = 48           # central station
N_DESTINATIONS = 12       # spread destinations evenly around source


# ---- Parametrised wave kernel ------------------------------------------

def make_step_kernel(pulse_tau_steps: int):
    """Closure-based kernel that bakes pulse_tau into the JIT trace."""

    pulse_t0 = SA.PULSE_T0_STEPS * SA.DT
    pulse_tau = pulse_tau_steps * SA.DT

    def step_kernel(carry, n, c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
                    one_minus_damp, free, dx, dy, dz):
        psi_prev, psi_curr, amp_prev, was_rising, T, src_spatial = carry
        t = n.astype(jnp.float32) * SA.DT
        dt2 = jnp.float32(SA.DT * SA.DT)
        diff = (SA.lap_var_coef_axis(c2_xy, psi_curr, dx, 0)
                + SA.lap_var_coef_axis(c2_xy, psi_curr, dy, 1)
                + SA.lap_var_coef_axis(c2_z, psi_curr, dz, 2))
        src = src_spatial * jnp.exp(-0.5 * ((t - pulse_t0) / pulse_tau) ** 2)
        psi_dot = (psi_curr - psi_prev) / SA.DT
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
        T_new = jnp.where(peak_now, t - SA.DT, T)
        return (psi_curr, psi_next, amp, rising_now, T_new, src_spatial), None

    return step_kernel


def make_solve(pulse_tau_steps: int, n_steps: int):
    step_kernel = make_step_kernel(pulse_tau_steps)

    def solve_one(c2_xy, c2_z, u, v, w, ax_f, ay_f, az_f,
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

    return jit(solve_one)


def gaussian_source(shape, idx, sigma):
    nx, ny, nz = shape
    ix = jnp.arange(nx)[:, None, None]
    iy = jnp.arange(ny)[None, :, None]
    iz = jnp.arange(nz)[None, None, :]
    r2 = ((ix - idx[0])**2 + (iy - idx[1])**2 + (iz - idx[2])**2)
    g = jnp.exp(-0.5 * r2 / (sigma**2))
    return (g / g.sum()).astype(jnp.float32)


# ---- Path tracing (copied from build_database.py) ----------------------

def trace_path(T, source_xyz, dest_xyz, dx, dy, dz,
               gamma=0.3, max_steps=10000):
    finite = np.isfinite(T)
    if not finite.any():
        return np.array([source_xyz, dest_xyz]), "no_finite"
    T_max = T[finite].max()
    T_safe = np.where(finite, T, T_max + 1e6)
    gx, gy, gz = np.gradient(T_safe, dx, dy, dz)
    step = gamma * min(dx, dy, dz)
    pos = np.array(dest_xyz, dtype=np.float64)
    src = np.array(source_xyz, dtype=np.float64)
    path = [pos.copy()]
    best_d = np.linalg.norm(pos - src)
    best_idx = 0
    stale = 0
    for _ in range(max_steps):
        coords = np.array([[pos[0]/dx], [pos[1]/dy], [pos[2]/dz]])
        gv = np.array([
            float(map_coordinates(gx, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gy, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gz, coords, order=1, mode="nearest")[0]),
        ])
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            break
        pos = pos - step * gv / gn
        path.append(pos.copy())
        d = np.linalg.norm(pos - src)
        if d < best_d - 0.5 * step:
            best_d = d
            best_idx = len(path) - 1
            stale = 0
        else:
            stale += 1
        if d < 2.0 * max(dx, dy, dz):
            path.append(src.copy())
            return np.array(path[::-1]), "reached"
        if stale > 200:
            path = path[: best_idx + 1]
            path.append(src.copy())
            return np.array(path[::-1]), f"truncated_{best_d:.0f}m"
    return np.array(path[::-1]), "max_steps"


# ---- Curvature diagnostic ----------------------------------------------

def _simplify_path(path_xy, min_seg=5.0):
    """Drop consecutive points closer than min_seg metres (the gradient
    descent loop emits hundreds of near-duplicate points near the
    destination once it stalls — those segments dominate any per-step
    κ stat with numerical garbage)."""
    if path_xy.shape[0] < 2:
        return path_xy
    keep = [path_xy[0]]
    for i in range(1, path_xy.shape[0]):
        if np.linalg.norm(path_xy[i] - keep[-1]) >= min_seg:
            keep.append(path_xy[i])
    if len(keep) == 1 or not np.allclose(keep[-1], path_xy[-1]):
        keep.append(path_xy[-1])
    return np.asarray(keep)


def path_curvature_stats(path, min_seg=5.0):
    """Return (mean κ, max κ, min radius) for a 2D path. κ in 1/m.
    Path is first simplified to drop near-duplicate points."""
    if path.shape[0] < 3:
        return 0.0, 0.0, float("inf")
    p = _simplify_path(path[:, :2].astype(np.float64), min_seg=min_seg)
    if p.shape[0] < 3:
        return 0.0, 0.0, float("inf")
    d1 = np.diff(p, axis=0)
    seg_len = np.linalg.norm(d1, axis=1)
    seg_len = np.maximum(seg_len, 1e-6)
    tangent = d1 / seg_len[:, None]
    dt = np.diff(tangent, axis=0)
    dn = np.linalg.norm(dt, axis=1)
    midlen = 0.5 * (seg_len[:-1] + seg_len[1:])
    kappa = dn / midlen
    if kappa.size == 0:
        return 0.0, 0.0, float("inf")
    k_mean = float(kappa.mean())
    k_max = float(kappa.max())
    r_min = 1.0 / k_max if k_max > 0 else float("inf")
    return k_mean, k_max, r_min


# ---- Main ----------------------------------------------------------------

def pick_destinations(stations_xy_m, src_idx, n,
                      d_min=600.0, d_max=2200.0):
    """Pick destinations spread by angle around the source. Distance band
    kept moderate so the wave reliably reaches the destination within
    N_STEPS·DT seconds even with detours around buildings."""
    src_xy = stations_xy_m[src_idx, :2]
    rel = stations_xy_m[:, :2] - src_xy
    d = np.linalg.norm(rel, axis=1)
    ok = (d > d_min) & (d < d_max)
    ok[src_idx] = False
    cand = np.where(ok)[0]
    if len(cand) <= n:
        return cand
    angles = np.arctan2(rel[cand, 1], rel[cand, 0])
    # Group by n angular bins, pick farthest in each (most diverse paths)
    bins = np.floor((angles + np.pi) / (2 * np.pi) * n).astype(int) % n
    chosen = []
    for b in range(n):
        in_b = cand[bins == b]
        if len(in_b) == 0:
            continue
        chosen.append(in_b[d[in_b].argmax()])
    return np.array(chosen, dtype=int)


def main():
    print("[load] domain.npz")
    z = np.load(_THIS.parent / "input" / "domain.npz", allow_pickle=False)
    mask_np = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask_np.shape
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)
    print(f"[domain] {nx}×{ny}×{nz}  dx={dx} dy={dy} dz={dz}  "
          f"{n_st} stations")

    # Source / destinations
    src_idx = SOURCE_IDX
    dst_indices = pick_destinations(stations_xy_m, src_idx, N_DESTINATIONS)
    print(f"[setup] source={src_idx} → {len(dst_indices)} destinations")

    # Station integer indices (re-use solve_all's nudging logic by copy)
    st_ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    st_iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    st_iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)
    inside_b = mask_np[st_ix, st_iy, st_iz]
    for b in np.where(inside_b)[0]:
        ix0, iy0, iz0 = st_ix[b], st_iy[b], st_iz[b]
        found = False
        for r in range(1, 6):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ix = ix0 + di
                    iy = iy0 + dj
                    if not (0 < ix < nx - 1 and 0 < iy < ny - 1):
                        continue
                    if not mask_np[ix, iy, iz0]:
                        st_ix[b], st_iy[b] = ix, iy
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            iz = iz0
            while iz < nz - 1 and mask_np[ix0, iy0, iz]:
                iz += 1
            st_iz[b] = iz

    src_xyz_idx = (int(st_ix[src_idx]),
                   int(st_iy[src_idx]),
                   int(st_iz[src_idx]))
    src_xyz_m = (st_ix[src_idx] * dx,
                 st_iy[src_idx] * dy,
                 st_iz[src_idx] * dz)

    # Static fields (same for every solve)
    print("[build] static fields")
    mask_jnp = jnp.asarray(mask_np)
    static = SA.build_static_fields(mask_jnp, dx, dy, dz)
    one_minus_damp = (1.0 - static["damp"]).astype(jnp.float32)

    # Run the sweep
    results = {}
    for label, tau_steps, sigma_cells in SWEEP:
        print(f"\n=== {label}: τ={tau_steps} steps, "
              f"σ_xy={sigma_cells} cells ===")
        solve_jit = make_solve(tau_steps, N_STEPS)
        sp = gaussian_source(mask_np.shape, src_xyz_idx, sigma_cells)
        jax.block_until_ready(sp)
        t0 = time.time()
        T = solve_jit(static["c2_xy"], static["c2_z"],
                      static["u"], static["v"], static["w"],
                      static["ax"], static["ay"], static["az"],
                      one_minus_damp, static["free"], sp,
                      dx, dy, dz)
        jax.block_until_ready(T)
        T_np = np.asarray(jax.device_get(T))
        dt_solve = time.time() - t0
        print(f"[solve] {dt_solve:.1f}s  T finite frac="
              f"{np.isfinite(T_np).mean():.3f}  "
              f"T(src nbrs) = {T_np[src_xyz_idx]:.2f}")

        # Filter to only destinations the wave actually reached
        T_dst_full = np.array(
            [float(T_np[st_ix[j], st_iy[j], st_iz[j]]) for j in dst_indices])
        finite_mask = np.isfinite(T_dst_full)
        active_dst = dst_indices[finite_mask]
        print(f"[paths] {len(active_dst)}/{len(dst_indices)} dests reachable "
              f"in {N_STEPS * SA.DT:.0f}s")

        paths = []
        statuses = []
        T_dst = []
        kappas = []
        for j in active_dst:
            dst_m = (st_ix[j] * dx, st_iy[j] * dy, st_iz[j] * dz)
            path, status = trace_path(T_np, src_xyz_m, dst_m, dx, dy, dz)
            paths.append(path)
            statuses.append(status)
            T_dst.append(float(T_np[st_ix[j], st_iy[j], st_iz[j]]))
            kappas.append(path_curvature_stats(path))
        n_reached = sum(1 for s in statuses if s == "reached")
        # Only consider κ for paths with enough samples to be meaningful
        good_k = [k for k, p in zip(kappas, paths) if p.shape[0] >= 5]
        all_kmax = [k[1] for k in good_k if np.isfinite(k[1])]
        all_rmin = [k[2] for k in good_k if np.isfinite(k[2])]
        kmed = float(np.median(all_kmax)) if all_kmax else 0.0
        rmed = float(np.median(all_rmin)) if all_rmin else float('inf')
        print(f"[paths] reached {n_reached}/{len(active_dst)}  "
              f"κ_max median = {kmed:.4f} 1/m  "
              f"R_min median = {rmed:.0f} m")
        results[label] = dict(
            T=T_np, paths=paths, statuses=statuses, T_dst=T_dst,
            kappas=kappas, tau=tau_steps, sigma=sigma_cells,
            active_dst=active_dst,
        )

    # ---- Plot --------------------------------------------------------
    fig, axes = plt.subplots(1, len(SWEEP), figsize=(6 * len(SWEEP), 6.5),
                             sharex=True, sharey=True)
    if len(SWEEP) == 1:
        axes = [axes]
    fp = mask_np.any(axis=2)
    Lx, Ly = nx * dx, ny * dy
    for ax_, (label, _, _) in zip(axes, SWEEP):
        r = results[label]
        ax_.set_facecolor("#f4f1ec")
        ax_.imshow(np.where(fp, 0.55, 0.0).T, origin="lower",
                   extent=[0, Lx, 0, Ly], cmap="Greys",
                   vmin=0, vmax=1, alpha=0.6, zorder=1)
        ax_.scatter(stations_xy_m[:, 0], stations_xy_m[:, 1],
                    s=8, c="white", edgecolor="black", linewidths=0.3,
                    zorder=4)
        # Paths coloured by destination travel time
        segs = []
        tcol = []
        for k, path in enumerate(r["paths"]):
            if path.shape[0] < 2:
                continue
            for s in range(path.shape[0] - 1):
                segs.append([path[s, :2], path[s + 1, :2]])
                tcol.append(r["T_dst"][k])
        if segs:
            tcol = np.asarray(tcol)
            vmin = float(np.nanmin(tcol))
            vmax = float(np.nanpercentile(tcol, 95))
            lc = LineCollection(segs, cmap="plasma_r",
                                norm=plt.Normalize(vmin, max(vmax, vmin + 1)),
                                linewidths=1.2, alpha=0.85, zorder=6)
            lc.set_array(tcol)
            ax_.add_collection(lc)
        # Source / destinations
        ax_.scatter(src_xyz_m[0], src_xyz_m[1], s=240, marker="*",
                    facecolor="#ffe833", edgecolor="black", linewidths=1.0,
                    zorder=12)
        d_xy = stations_xy_m[r["active_dst"], :2]
        ax_.scatter(d_xy[:, 0], d_xy[:, 1], s=40, marker="v",
                    facecolor="#ff5544", edgecolor="black", linewidths=0.4,
                    alpha=0.8, zorder=10)
        n_reached = sum(1 for s in r["statuses"] if s == "reached")
        good_k = [k for k, p in zip(r["kappas"], r["paths"]) if p.shape[0] >= 5]
        all_kmax = [k[1] for k in good_k if np.isfinite(k[1])]
        all_rmin = [k[2] for k in good_k if np.isfinite(k[2])]
        kmed = np.median(all_kmax) if all_kmax else 0.0
        rmed = np.median(all_rmin) if all_rmin else float('inf')
        ax_.set_title(
            f"{label}: τ={r['tau']} steps, σ={r['sigma']} cells\n"
            f"reachable {len(r['active_dst'])}/{len(dst_indices)},  "
            f"κ_max med={kmed:.4f} 1/m,  R_min med={rmed:.0f} m",
            fontsize=11)
        ax_.set_xlim(0, Lx)
        ax_.set_ylim(0, Ly)
        ax_.set_aspect("equal")
        ax_.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Wavelength sweep — paths from station {src_idx} "
        f"({len(dst_indices)} destinations)",
        fontsize=13)
    out_dir = _THIS.parent / "output" / "figures" / "wavelength_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"wavelength_sweep_src{src_idx:03d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")

    # Save the raw bundle for later inspection
    bundle = dict(
        source_idx=np.int32(src_idx),
        source_xyz_m=np.array(src_xyz_m, dtype=np.float32),
        dst_indices=dst_indices.astype(np.int32),
        sweep_labels=np.array([s[0] for s in SWEEP]),
        sweep_tau=np.array([s[1] for s in SWEEP], dtype=np.int32),
        sweep_sigma=np.array([s[2] for s in SWEEP], dtype=np.float32),
    )
    for label, _, _ in SWEEP:
        r = results[label]
        path_lengths = np.array([p.shape[0] for p in r["paths"]],
                                dtype=np.int32)
        path_offsets = np.concatenate([[0], np.cumsum(path_lengths)])
        paths_concat = (np.concatenate(r["paths"], axis=0).astype(np.float32)
                        if r["paths"] else np.zeros((0, 3), dtype=np.float32))
        bundle[f"{label}_path_lengths"] = path_lengths
        bundle[f"{label}_path_offsets"] = path_offsets.astype(np.int32)
        bundle[f"{label}_paths_concat"] = paths_concat
        bundle[f"{label}_T_dst"] = np.array(r["T_dst"], dtype=np.float32)
        bundle[f"{label}_kappa_max"] = np.array(
            [k[1] for k in r["kappas"]], dtype=np.float32)
        bundle[f"{label}_R_min"] = np.array(
            [k[2] for k in r["kappas"]], dtype=np.float32)
    out_npz = out_dir / f"wavelength_sweep_src{src_idx:03d}.npz"
    np.savez_compressed(out_npz, **bundle)
    print(f"[saved] {out_npz}")


if __name__ == "__main__":
    main()
