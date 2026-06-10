"""Routine planning on the South Kensington area with a real flow field.

This experiment:
  1. Builds a domain mask from INHALE_1280_64x1024x1024.npy (4 m grid).
  2. Loads velocity (uvw) at a chosen timestep from the urbanflow dataset.
  3. Picks two points (source, destination) inside the free region.
  4. Solves the convected wave PDE twice — once with the spatial wind field
     and once with zero wind — and traces the gradient-descent path of the
     arrival-time field T(x).
  5. Compares travel times and saves a 2x2 visual panel.

The local function `build_static_fields_with_wind_array` is a drop-in
replacement for `solve_all.build_static_fields_param` that accepts full
(NX,NY,NZ) wind arrays instead of a single 3-tuple.

Usage (login node, CPU, small patch ~ 1-2 min):
  $VENV/bin/python experiments/exp_sk_wind.py --nx 192 --ny 192 --nz 32

Usage (SLURM GPU, full patch):
  see jobs/job_sk_wind.sh

The domain is *not* saved to disk; the script builds + solves + plots in
one shot. If you want to reuse with `solve_one_source.py`, pass
`--save_domain south_kensington_t3000.npz`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

_CACHE_DIR = str(THIS.parent / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import h5py
import numpy as np
import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "model"))
from solve_all import (
    C0, CZ_UP, CZ_DOWN, DT, PULSE_T0_STEPS, PULSE_TAU_STEPS,
    SOURCE_SIGMA_CELLS, SPONGE_LAYERS, SPONGE_STRENGTH,
    REG_Z_LOW, REG_Z_FULL, REG_LOW_FACTOR,
    grad_central, gaussian_source, solve_one,
)
from solve_coupled import trace_path
from scipy.ndimage import map_coordinates, gaussian_filter


def trace_path_2d_at_z(T, iz, src_xy_m, dst_xy_m, dx, dy,
                       gamma=0.3, max_steps=20000, max_stale=400):
    """Pure 2D gradient descent on T[:, :, iz] from dst_xy_m back to src_xy_m.

    Returns a (M, 2) array of (x, y) in metres, in src→dst order.
    """
    T2 = np.asarray(T[:, :, iz])
    finite = np.isfinite(T2)
    if not finite.any():
        return None
    T_max = T2[finite].max()
    T_safe = np.where(finite, T2, T_max + 1e6)
    # Smooth out reflection-induced noise so gradient descent has a clean field.
    T_safe = gaussian_filter(T_safe, sigma=3.0)
    gx, gy = np.gradient(T_safe, dx, dy)
    step = gamma * min(dx, dy)
    pos = np.asarray(dst_xy_m, dtype=np.float64)
    src = np.asarray(src_xy_m, dtype=np.float64)
    path = [pos.copy()]
    best_d = np.linalg.norm(pos - src)
    best_idx = 0
    stale = 0
    for _ in range(max_steps):
        coords = np.array([[pos[0] / dx], [pos[1] / dy]])
        gv = np.array([
            float(map_coordinates(gx, coords, order=1, mode="nearest")[0]),
            float(map_coordinates(gy, coords, order=1, mode="nearest")[0]),
        ])
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            break
        pos = pos - step * gv / gn
        path.append(pos.copy())
        d = np.linalg.norm(pos - src)
        if d < best_d - 0.5 * step:
            best_d = d; best_idx = len(path) - 1; stale = 0
        else:
            stale += 1
        if d < 2.0 * max(dx, dy):
            path.append(src.copy())
            return np.array(path[::-1])
        if stale > max_stale:
            path = path[: best_idx + 1]
            path.append(src.copy())
            return np.array(path[::-1])
    return np.array(path[::-1])


def build_three_phase_path(T, src_xyz_m, dst_xyz_m, dx, dy, dz, cruise_z_m=40.0):
    """Build a takeoff–cruise–landing trajectory for low-altitude endpoints.

    1) vertical climb from src to (src_x, src_y, cruise_z_m)
    2) horizontal trace at z=cruise_z_idx through T[:, :, cruise_z_idx]
    3) vertical descent at (dst_x, dst_y) from cruise_z_m to dst

    Returns (N, 3) path in (x, y, z) metres.
    """
    nz = T.shape[2]
    cruise_iz = int(np.clip(round(cruise_z_m / dz), 1, nz - 1))
    cruise_z_actual = cruise_iz * dz

    # Phase 2: horizontal cruise trace on the 2D slice
    xy_path = trace_path_2d_at_z(T, cruise_iz,
                                  src_xyz_m[:2], dst_xyz_m[:2], dx, dy)
    if xy_path is None or len(xy_path) < 2:
        return None
    # Phase 1: climb at src (N samples between src_z and cruise_z)
    climb_N = max(4, int(round((cruise_z_actual - src_xyz_m[2]) / dz)) + 1)
    climb = np.linspace(src_xyz_m[2], cruise_z_actual, climb_N)
    climb_xyz = np.stack([np.full(climb_N, src_xyz_m[0]),
                          np.full(climb_N, src_xyz_m[1]),
                          climb], axis=1)
    # Cruise xyz (z fixed)
    cruise_xyz = np.column_stack([xy_path,
                                   np.full(len(xy_path), cruise_z_actual)])
    # Phase 3: descent at dst
    desc_N = max(4, int(round((cruise_z_actual - dst_xyz_m[2]) / dz)) + 1)
    desc = np.linspace(cruise_z_actual, dst_xyz_m[2], desc_N)
    desc_xyz = np.stack([np.full(desc_N, dst_xyz_m[0]),
                         np.full(desc_N, dst_xyz_m[1]),
                         desc], axis=1)
    # Stitch (avoid duplicate endpoints)
    return np.vstack([climb_xyz, cruise_xyz[1:-1], desc_xyz])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "/leonardo_scratch/large/userexternal/yli00001/SCALED_Project/data/urbanflow_dataset/southkensington"
DEFAULT_GEOMETRY = "/leonardo_scratch/large/userexternal/yli00001/SCALED_Project/data/geometry/INHALE_1280_64x1024x1024.npy"


def load_uvw(data_dir, t):
    fn = os.path.join(data_dir, f"{t:06d}.h5")
    with h5py.File(fn, "r") as f:
        uvw = f["uvw"][...]  # (3, D, H, W), float16
    return np.asarray(uvw, dtype=np.float32)  # (3, D=64, H=1280, W=1280)


def load_geometry(geom_path):
    return np.load(geom_path, mmap_mode="r")  # (D=64, H=1024, W=1024) uint8


def crop_and_axes_transform(uvw, geom, hi, wi, nx, ny, nz):
    """
    Source convention: (3=C, D, H, W) with D=vertical, H/W horizontal.
    Wave-solver convention: (NX, NY, NZ) with NZ=vertical.

    Crop H, W → nx, ny starting at (hi, wi). Crop D → nz (from bottom).
    Then reshape to (NX, NY, NZ):
        mask[ix, iy, iz] = geom_DHW[iz, hi+ix, wi+iy]
        u_wave           = uvw[0, :nz, hi:hi+nx, wi:wi+ny].transpose(1, 2, 0)
        v_wave, w_wave   similarly for channels 1, 2.

    Mapping of velocity channels to spatial axes:
        channel 0 (u) → +H direction → wave-x  (NX)
        channel 1 (v) → +W direction → wave-y  (NY)
        channel 2 (w) → +D direction → wave-z  (NZ)
    """
    geom_crop = np.array(geom[:nz, hi : hi + nx, wi : wi + ny], dtype=np.uint8)
    mask = geom_crop.transpose(1, 2, 0).astype(bool)  # (nx, ny, nz)

    uvw_crop = uvw[:, :nz, hi : hi + nx, wi : wi + ny]  # (3, nz, nx, ny)
    u_wave = uvw_crop[0].transpose(1, 2, 0)  # (nx, ny, nz)
    v_wave = uvw_crop[1].transpose(1, 2, 0)
    w_wave = uvw_crop[2].transpose(1, 2, 0)
    return mask, u_wave, v_wave, w_wave


# ---------------------------------------------------------------------------
# Static fields with spatial wind
# ---------------------------------------------------------------------------

def build_static_fields_with_wind_array(mask, dx, dy, dz, u_field, v_field, w_field,
                                        c0=C0, cz_up=CZ_UP, cz_down=CZ_DOWN,
                                        reg_low_factor=REG_LOW_FACTOR,
                                        reg_z_low=REG_Z_LOW, reg_z_full=REG_Z_FULL):
    """Like solve_all.build_static_fields_param but with full (NX,NY,NZ)
    wind arrays and overridable regulatory profile."""
    nx, ny, nz = mask.shape
    cz0 = 0.5 * (cz_up + cz_down)
    w_grav = 0.5 * (cz_down - cz_up)

    z_centers = (jnp.arange(nz) + 0.5) * dz
    reg = jnp.where(
        z_centers >= reg_z_full, 1.0,
        jnp.where(z_centers < reg_z_low, reg_low_factor,
                  reg_low_factor + (1.0 - reg_low_factor)
                      * (z_centers - reg_z_low)
                      / max(1e-9, reg_z_full - reg_z_low)),
    )[None, None, :]

    free = (~mask).astype(jnp.float32)
    c0_arr = c0 * free * reg
    cz0_arr = cz0 * free
    c2_xy = c0_arr ** 2
    c2_z = cz0_arr ** 2

    u = jnp.asarray(u_field, dtype=jnp.float32) * free
    v = jnp.asarray(v_field, dtype=jnp.float32) * free
    w = jnp.asarray(w_field, dtype=jnp.float32) * free - w_grav

    ax = u*grad_central(u,0,dx) + v*grad_central(u,1,dy) + w*grad_central(u,2,dz)
    ay = u*grad_central(v,0,dx) + v*grad_central(v,1,dy) + w*grad_central(v,2,dz)
    az = u*grad_central(w,0,dx) + v*grad_central(w,1,dy) + w*grad_central(w,2,dz)

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


# ---------------------------------------------------------------------------
# Helper: nudge a point out of buildings
# ---------------------------------------------------------------------------

def nudge_out_of_solid(mask, ix, iy, iz, max_r=10):
    """If (ix,iy,iz) is inside a building, walk outward in xy then up in z."""
    nx, ny, nz = mask.shape
    if not mask[ix, iy, iz]:
        return ix, iy, iz
    for r in range(1, max_r + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                jx, jy = ix + di, iy + dj
                if not (0 < jx < nx - 1 and 0 < jy < ny - 1):
                    continue
                if not mask[jx, jy, iz]:
                    return jx, jy, iz
    # Last resort: walk up
    jz = iz
    while jz < nz - 1 and mask[ix, iy, jz]:
        jz += 1
    return ix, iy, jz


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_panel(mask, path_wind, path_nowind, T_wind, T_nowind,
                 src_xyz_m, dst_xyz_m, dx, dy, dz, out_png,
                 u_field, v_field, t_label, vis_z=2):
    nx, ny, nz = mask.shape
    Lx, Ly = nx * dx, ny * dy

    fp = mask.any(axis=2)  # footprint
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax_, path, T, title, wind_overlay in (
        (axes[0], path_nowind, T_nowind, "no wind", False),
        (axes[1], path_wind, T_wind, f"with wind  t={t_label}", True),
    ):
        ax_.set_facecolor("#f4f1ec")
        ax_.imshow(np.where(fp.T, 0.55, 0.0),
                   origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="Greys", vmin=0, vmax=1, alpha=0.7, zorder=1)

        # T contours at vis_z slice
        T_slice = np.asarray(T)[:, :, vis_z]
        finite = np.isfinite(T_slice)
        if finite.any():
            Tplot = np.where(finite, T_slice, np.nan)
            cs = ax_.contourf(
                np.arange(nx)*dx + 0.5*dx,
                np.arange(ny)*dy + 0.5*dy,
                Tplot.T, levels=20, cmap="viridis", alpha=0.55, zorder=2,
            )
            plt.colorbar(cs, ax=ax_, label="T (s)", fraction=0.046)

        # wind quiver overlay
        if wind_overlay:
            step = max(1, nx // 24)
            X = (np.arange(nx)*dx + 0.5*dx)[::step]
            Y = (np.arange(ny)*dy + 0.5*dy)[::step]
            U = np.asarray(u_field)[::step, ::step, vis_z]
            V = np.asarray(v_field)[::step, ::step, vis_z]
            ax_.quiver(X, Y, U.T, V.T, color="white", scale=30, width=0.003,
                       headwidth=4, zorder=4)

        # path
        if path is not None and len(path) > 1:
            ax_.plot(path[:, 0], path[:, 1], "-", color="red", lw=2.0, zorder=5)
            T_at_dst = float(np.asarray(T)[
                int(round(dst_xyz_m[0]/dx)),
                int(round(dst_xyz_m[1]/dy)),
                int(round(dst_xyz_m[2]/dz)),
            ])
            ax_.set_title(f"{title}   T(dst) = {T_at_dst:.1f} s")
        else:
            ax_.set_title(title)

        ax_.scatter([src_xyz_m[0]], [src_xyz_m[1]], s=120, c="lime",
                    edgecolor="black", linewidths=1.0, zorder=6, label="src")
        ax_.scatter([dst_xyz_m[0]], [dst_xyz_m[1]], s=120, c="red",
                    edgecolor="black", linewidths=1.0, zorder=6, label="dst")
        ax_.set_xlim(0, Lx); ax_.set_ylim(0, Ly)
        ax_.set_aspect("equal")
        ax_.set_xlabel("x (m)"); ax_.set_ylabel("y (m)")
        ax_.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"[saved] {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=int, default=3000)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    p.add_argument("--hi", type=int, default=320, help="top-left H crop index (in geom 0..1024)")
    p.add_argument("--wi", type=int, default=320, help="top-left W crop index")
    p.add_argument("--nx", type=int, default=384)
    p.add_argument("--ny", type=int, default=384)
    p.add_argument("--nz", type=int, default=48)
    p.add_argument("--dx", type=float, default=4.0, help="grid spacing in metres")
    p.add_argument("--dt", type=float, default=0.1, help="time step (s); override solve_all DT")
    p.add_argument("--n_steps", type=int, default=2000, help="wave-solver iterations")
    p.add_argument("--wind_scale", type=float, default=5.0,
                   help="multiplier applied to raw uvw to convert to m/s (raw mag~0.5)")
    p.add_argument("--src", type=float, nargs=3, default=None,
                   help="source x y z in metres (default: 15%% , 15%% , 5 m)")
    p.add_argument("--dst", type=float, nargs=3, default=None,
                   help="dest   x y z in metres (default: 85%% , 85%% , 5 m)")
    p.add_argument("--out_dir", default=str(THIS.parent / "SCALED-UrbanFlowv3" / "inference_output" / "routine_sk_wind"))
    p.add_argument("--vis_z", type=int, default=2)
    p.add_argument("--reg_low_factor", type=float, default=0.0,
                   help="horizontal speed factor below reg_z_low (default 0 = strict no-fly horiz below low-z)")
    p.add_argument("--reg_z_low", type=float, default=REG_Z_LOW)
    p.add_argument("--reg_z_full", type=float, default=REG_Z_FULL)
    p.add_argument("--save_domain", default=None,
                   help="if given, write the cropped domain as an npz (mask + dx/dy/dz)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dx = dy = dz = args.dx

    # ---- load data ---------------------------------------------------------
    print(f"[info] jax devices: {jax.devices()}")
    print(f"[load] uvw t={args.t}")
    uvw = load_uvw(args.data_dir, args.t)  # (3, 64, 1280, 1280)
    geom = load_geometry(args.geometry)    # (64, 1024, 1024)

    # bounds check
    if args.hi + args.nx > geom.shape[1] or args.wi + args.ny > geom.shape[2]:
        raise SystemExit("[error] crop window outside geometry 1024x1024")
    if args.nz > geom.shape[0]:
        raise SystemExit("[error] nz > geom depth (64)")

    mask, uF, vF, wF = crop_and_axes_transform(uvw, geom, args.hi, args.wi,
                                               args.nx, args.ny, args.nz)
    # scale to m/s
    uF = (uF * args.wind_scale).astype(np.float32)
    vF = (vF * args.wind_scale).astype(np.float32)
    wF = (wF * args.wind_scale).astype(np.float32)

    print(f"[crop] mask {mask.shape}  solid frac {mask.mean():.3%}  free cells {(~mask).sum():,}")
    print(f"[wind] scaled  u: [{uF.min():.2f}, {uF.max():.2f}]  v: [{vF.min():.2f}, {vF.max():.2f}]  "
          f"w: [{wF.min():.2f}, {wF.max():.2f}]  mean|U|={np.sqrt(uF**2+vF**2+wF**2).mean():.2f} m/s")

    nx, ny, nz = mask.shape
    Lx, Ly, Lz = nx*dx, ny*dy, nz*dz
    print(f"[grid] {nx} x {ny} x {nz}  →  {Lx:.0f} m x {Ly:.0f} m x {Lz:.0f} m  (dx={dx}m)")

    # ---- pick source / dest -----------------------------------------------
    src_xyz_m = np.asarray(args.src) if args.src else np.array([0.15*Lx, 0.15*Ly, 5.0])
    dst_xyz_m = np.asarray(args.dst) if args.dst else np.array([0.85*Lx, 0.85*Ly, 5.0])

    src_idx = (int(round(src_xyz_m[0]/dx)),
               int(round(src_xyz_m[1]/dy)),
               int(round(src_xyz_m[2]/dz)))
    dst_idx = (int(round(dst_xyz_m[0]/dx)),
               int(round(dst_xyz_m[1]/dy)),
               int(round(dst_xyz_m[2]/dz)))
    src_idx = nudge_out_of_solid(mask, *src_idx)
    dst_idx = nudge_out_of_solid(mask, *dst_idx)
    src_xyz_m = np.array([src_idx[0]*dx, src_idx[1]*dy, src_idx[2]*dz], dtype=np.float64)
    dst_xyz_m = np.array([dst_idx[0]*dx, dst_idx[1]*dy, dst_idx[2]*dz], dtype=np.float64)
    print(f"[points] src idx={src_idx} m={src_xyz_m}")
    print(f"[points] dst idx={dst_idx} m={dst_xyz_m}")

    # ---- save domain (optional) -------------------------------------------
    if args.save_domain:
        np.savez_compressed(
            args.save_domain, mask=mask, dx=dx, dy=dy, dz=dz,
            stations_xy_m=np.stack([src_xyz_m, dst_xyz_m]),
            u=uF, v=vF, w=wF, source_t=args.t,
        )
        print(f"[saved] {args.save_domain}")

    # ---- solve no-wind ----------------------------------------------------
    mask_jnp = jnp.asarray(mask)
    zero = jnp.zeros_like(mask_jnp, dtype=jnp.float32)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))

    print("[solve] no-wind build + solve...")
    t0 = time.time()
    static0 = build_static_fields_with_wind_array(mask_jnp, dx, dy, dz, zero, zero, zero,
                                                  reg_low_factor=args.reg_low_factor,
                                                  reg_z_low=args.reg_z_low,
                                                  reg_z_full=args.reg_z_full)
    one_minus_damp0 = (1.0 - static0["damp"]).astype(jnp.float32)
    src_spatial = gaussian_source(mask_jnp.shape, src_idx)
    T0 = solve_jit(static0["c2_xy"], static0["c2_z"],
                   static0["u"], static0["v"], static0["w"],
                   static0["ax"], static0["ay"], static0["az"],
                   one_minus_damp0, static0["free"], src_spatial,
                   dx, dy, dz, args.n_steps)
    jax.block_until_ready(T0)
    T0_np = np.asarray(jax.device_get(T0))
    print(f"[solve] no-wind done in {time.time()-t0:.1f}s")

    # ---- solve with wind --------------------------------------------------
    print("[solve] with-wind build + solve...")
    t0 = time.time()
    static1 = build_static_fields_with_wind_array(
        mask_jnp, dx, dy, dz,
        jnp.asarray(uF), jnp.asarray(vF), jnp.asarray(wF),
        reg_low_factor=args.reg_low_factor,
        reg_z_low=args.reg_z_low,
        reg_z_full=args.reg_z_full,
    )
    one_minus_damp1 = (1.0 - static1["damp"]).astype(jnp.float32)
    T1 = solve_jit(static1["c2_xy"], static1["c2_z"],
                   static1["u"], static1["v"], static1["w"],
                   static1["ax"], static1["ay"], static1["az"],
                   one_minus_damp1, static1["free"], src_spatial,
                   dx, dy, dz, args.n_steps)
    jax.block_until_ready(T1)
    T1_np = np.asarray(jax.device_get(T1))
    print(f"[solve] with-wind done in {time.time()-t0:.1f}s")

    # ---- T at dst ----------------------------------------------------------
    T0_dst = float(T0_np[dst_idx])
    T1_dst = float(T1_np[dst_idx])
    print(f"[result] T(dst | no wind)  = {T0_dst:.2f} s")
    print(f"[result] T(dst | with wind) = {T1_dst:.2f} s  "
          f"(Δ = {T1_dst - T0_dst:+.2f} s, {(T1_dst-T0_dst)/T0_dst:+.2%})")

    # ---- trace paths -------------------------------------------------------
    # If endpoints are below cruise altitude (regulatory low-z zone), build a
    # three-phase climb/cruise/descent path. Otherwise do 3D gradient descent.
    cruise_z_m = args.reg_z_full + 5.0  # just above the regulatory free-fly threshold
    use_three_phase = (src_xyz_m[2] < cruise_z_m - 1.0) or (dst_xyz_m[2] < cruise_z_m - 1.0)
    if use_three_phase:
        print(f"[path] using three-phase (climb→cruise@z={cruise_z_m}m→descend)")
        path0 = build_three_phase_path(T0_np, src_xyz_m, dst_xyz_m, dx, dy, dz, cruise_z_m)
        path1 = build_three_phase_path(T1_np, src_xyz_m, dst_xyz_m, dx, dy, dz, cruise_z_m)
        st0 = "three_phase" if path0 is not None else "no_finite"
        st1 = "three_phase" if path1 is not None else "no_finite"
    else:
        path0, st0 = trace_path(T0_np, src_xyz_m, dst_xyz_m, dx, dy, dz)
        path1, st1 = trace_path(T1_np, src_xyz_m, dst_xyz_m, dx, dy, dz)
    print(f"[path] no-wind   status={st0}  npts={len(path0) if path0 is not None else 0}")
    print(f"[path] with-wind status={st1}  npts={len(path1) if path1 is not None else 0}")
    if path0 is None: path0 = np.stack([src_xyz_m, dst_xyz_m])
    if path1 is None: path1 = np.stack([src_xyz_m, dst_xyz_m])

    # ---- save metrics + paths ---------------------------------------------
    np.savez_compressed(
        os.path.join(args.out_dir, f"routine_sk_t{args.t:06d}.npz"),
        mask=mask, dx=dx, dy=dy, dz=dz,
        u=uF, v=vF, w=wF, wind_scale=args.wind_scale, source_t=args.t,
        src_xyz_m=src_xyz_m, dst_xyz_m=dst_xyz_m,
        T0=T0_np.astype(np.float32),
        T1=T1_np.astype(np.float32),
        path_nowind=path0.astype(np.float32),
        path_wind=path1.astype(np.float32),
        T0_dst=np.float32(T0_dst), T1_dst=np.float32(T1_dst),
    )
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"timestep   {args.t}\n")
        f.write(f"grid       {nx} x {ny} x {nz}  dx={dx}m\n")
        f.write(f"wind_scale {args.wind_scale}\n")
        f.write(f"src_m      {src_xyz_m}\n")
        f.write(f"dst_m      {dst_xyz_m}\n")
        f.write(f"T_no_wind  {T0_dst:.2f} s   status={st0}\n")
        f.write(f"T_wind     {T1_dst:.2f} s   status={st1}\n")
        f.write(f"delta      {T1_dst-T0_dst:+.2f} s ({(T1_dst-T0_dst)/T0_dst:+.2%})\n")
    print(f"[saved] {os.path.join(args.out_dir, 'summary.txt')}")

    # ---- visualise --------------------------------------------------------
    render_panel(mask, path1, path0, T1_np, T0_np,
                 src_xyz_m, dst_xyz_m, dx, dy, dz,
                 out_png=os.path.join(args.out_dir, f"routine_sk_t{args.t:06d}.png"),
                 u_field=uF, v_field=vF, t_label=args.t, vis_z=args.vis_z)


if __name__ == "__main__":
    main()
