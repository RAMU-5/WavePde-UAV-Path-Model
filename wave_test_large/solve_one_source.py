"""Single-source wave solve, designed for SLURM array job.

Usage (one task per source):
    python solve_one_source.py --source-idx 0
    python solve_one_source.py --source-idx 117

Each invocation reads `domain.npz`, runs the JAX wave solve with one
station as source, then saves
    T_source_<idx>.npz containing
        T_at_stations : (N,) float32 — arrival times at every station
        T_full        : (NX, NY, NZ) float32 — full T-field (optional, for
                        path tracing). Set --no-full to skip and save disk.

After all 118 are done, run `aggregate_pair_matrix.py` to assemble the
full pair-time matrix.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

# Persistent JAX cache (so all 118 array tasks reuse the same compiled
# program — first task pays ~30s compile, rest read it from disk).
_THIS = Path(__file__).resolve().parent
_CACHE_DIR = str(_THIS / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit, lax

# Import shared kernel functions from solve_all.py
from solve_all import (
    grad_central, lap_var_coef_axis, adv,
    gaussian_source, build_static_fields,
    step_kernel, solve_one,
    C0, CZ_UP, CZ_DOWN, WIND, DT, N_STEPS,
    PULSE_T0_STEPS, PULSE_TAU_STEPS, SOURCE_SIGMA_CELLS,
    SPONGE_LAYERS, SPONGE_STRENGTH, MIN_PEAK_AMP,
    REG_Z_LOW, REG_Z_FULL, REG_LOW_FACTOR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-idx", type=int, required=True)
    parser.add_argument("--domain", default="domain.npz",
                        help="Domain file (use domain_5m.npz for 5m grid).")
    parser.add_argument("--no-full", action="store_true",
                        help="Skip saving the (NX,NY,NZ) T-field; only "
                             "save T_at_stations vector.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"[task] source-idx={args.source_idx}  "
          f"jax devices: {jax.devices()}")

    z = np.load(_THIS / args.domain, allow_pickle=False)
    mask_np = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask_np.shape
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)

    if not (0 <= args.source_idx < n_st):
        raise ValueError(f"source-idx {args.source_idx} out of range [0, {n_st})")

    out_dir = _THIS / "outputs"
    out_dir.mkdir(exist_ok=True)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
    else:
        out_path = out_dir / f"T_source_{args.source_idx:03d}.npz"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    # Station integer indices, with horizontal nudge if inside a building
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
                    ix = ix0 + di; iy = iy0 + dj
                    if not (0 < ix < nx - 1 and 0 < iy < ny - 1):
                        continue
                    if not mask_np[ix, iy, iz0]:
                        st_ix[b], st_iy[b] = ix, iy
                        found = True; break
                if found: break
            if found: break
        if not found:
            iz = iz0
            while iz < nz - 1 and mask_np[ix0, iy0, iz]:
                iz += 1
            st_iz[b] = iz

    # Build static fields (jit-compiled lazily)
    mask_jnp = jnp.asarray(mask_np)
    static = build_static_fields(mask_jnp, dx, dy, dz)
    one_minus_damp = (1.0 - static["damp"]).astype(jnp.float32)

    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))

    i = args.source_idx
    idx = (int(st_ix[i]), int(st_iy[i]), int(st_iz[i]))
    print(f"[task] station {i} → idx {idx} (z={idx[2]*dz:.1f}m)")

    t0 = time.time()
    src_spatial = gaussian_source(mask_jnp.shape, idx)
    jax.block_until_ready(src_spatial)
    t1 = time.time()
    T = solve_jit(static["c2_xy"], static["c2_z"],
                  static["u"], static["v"], static["w"],
                  static["ax"], static["ay"], static["az"],
                  one_minus_damp, static["free"], src_spatial,
                  dx, dy, dz, N_STEPS)
    jax.block_until_ready(T)
    t2 = time.time()
    T_np = np.asarray(jax.device_get(T))
    t3 = time.time()
    T_at_stations = T_np[st_ix, st_iy, st_iz]

    print(f"[task] gauss {t1-t0:.2f}s, solve {t2-t1:.2f}s, "
          f"copy {t3-t2:.2f}s")
    n_arrived = int(np.isfinite(T_at_stations).sum())
    print(f"[task] reached {n_arrived-1}/{n_st-1} other stations")

    save_dict = {
        "T_at_stations": T_at_stations.astype(np.float32),
        "source_idx": np.int32(i),
        "n_steps": np.int32(N_STEPS),
    }
    if not args.no_full:
        save_dict["T_full"] = T_np.astype(np.float32)
    np.savez_compressed(out_path, **save_dict)
    print(f"[task] saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
