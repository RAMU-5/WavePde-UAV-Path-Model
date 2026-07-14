"""5 m grid smoke test: ONE wave solve from one source on the
1333 × 1443 × 42 = 80.8 M-cell domain. Reports timing, memory, and
how many cells the wave actually reached.

Run via submit_smoke_5m.sh on a GPU node — needs ~10-15 GB GPU memory
and ~25-40 min wall time on A100.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(THIS_DIR.parents[1] / "_jax_cache"))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit

from model.solve_all import (build_static_fields, gaussian_source, solve_one,
                       N_STEPS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="domain_5m.npz")
    p.add_argument("--src-idx", type=int, default=48)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--out", default="smoke_5m_T.npz")
    args = p.parse_args()

    print(f"[smoke] jax devices: {jax.devices()}")
    print(f"[smoke] backend:     {jax.default_backend()}")

    t_load = time.time()
    z = np.load(THIS_DIR.parents[1] / "input" / args.domain, allow_pickle=False)
    mask_np = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask_np.shape
    stations = z["stations_xy_m"]
    print(f"[smoke] loaded {args.domain} in {time.time()-t_load:.1f}s")
    print(f"[smoke] grid {nx}×{ny}×{nz} = {nx*ny*nz/1e6:.1f}M cells, "
          f"dx={dx} dy={dy} dz={dz}")

    src_xyz = stations[args.src_idx]
    idx = (int(np.clip(round(src_xyz[0]/dx), 1, nx-2)),
           int(np.clip(round(src_xyz[1]/dy), 1, ny-2)),
           int(np.clip(round(src_xyz[2]/dz), 1, nz-2)))
    print(f"[smoke] source station {args.src_idx} at xyz={src_xyz} "
          f"-> cell {idx}")

    print("[smoke] building static fields...")
    t0 = time.time()
    mask = jnp.asarray(mask_np)
    s = build_static_fields(mask, dx, dy, dz)
    one_minus_damp = (1.0 - s["damp"]).astype(jnp.float32)
    src_spatial = gaussian_source(mask.shape, idx)
    jax.block_until_ready(src_spatial)
    print(f"[smoke] static fields ready in {time.time()-t0:.1f}s")

    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))
    print(f"[smoke] starting JIT compile + {args.n_steps}-step solve...")
    t_compile = time.time()
    T = solve_jit(s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
                  s["ax"], s["ay"], s["az"], one_minus_damp, s["free"],
                  src_spatial, dx, dy, dz, args.n_steps)
    jax.block_until_ready(T)
    elapsed = time.time() - t_compile
    print(f"[smoke] solve done: {elapsed:.1f}s "
          f"(equiv. {args.n_steps/elapsed:.0f} steps/s on this device)")

    print("[smoke] copying T to host...")
    t_cp = time.time()
    T_np = np.asarray(jax.device_get(T))
    print(f"[smoke] copy: {time.time()-t_cp:.1f}s")

    n_finite = int(np.isfinite(T_np).sum())
    coverage = n_finite / T_np.size
    if n_finite > 0:
        T_finite = T_np[np.isfinite(T_np)]
        print(f"[smoke] T finite cells: {n_finite:,} / {T_np.size:,} "
              f"({coverage*100:.1f}%)")
        print(f"[smoke] T range: [{T_finite.min():.1f}, "
              f"{T_finite.max():.1f}] s")

    # Read T at every other station — gives the entire row of the
    # forward T-matrix from this single source in one solve.
    st_ix = np.clip(np.round(stations[:, 0] / dx).astype(int), 1, nx-2)
    st_iy = np.clip(np.round(stations[:, 1] / dy).astype(int), 1, ny-2)
    st_iz = np.clip(np.round(stations[:, 2] / dz).astype(int), 1, nz-2)
    T_row = T_np[st_ix, st_iy, st_iz]
    n_reached = int(np.isfinite(T_row).sum())
    print(f"[smoke] reached {n_reached-1}/{len(stations)-1} other "
          f"stations from source {args.src_idx}")
    if n_reached > 1:
        T_other = T_row[np.isfinite(T_row) & (np.arange(len(stations)) != args.src_idx)]
        print(f"[smoke] travel times to other stations: "
              f"min {T_other.min():.1f}s, max {T_other.max():.1f}s, "
              f"mean {T_other.mean():.1f}s")

    out_dir = THIS_DIR.parents[1] / "output" / "wave"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / args.out, T_field=T_np.astype(np.float32),
                        T_at_stations=T_row.astype(np.float32),
                        src_idx=np.int32(args.src_idx),
                        dx=dx, dy=dy, dz=dz, n_steps=args.n_steps)
    print(f"[smoke] saved {args.out}")


if __name__ == "__main__":
    main()
