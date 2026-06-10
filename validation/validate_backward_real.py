"""Real-domain validation of the backward-wave symmetry (paper §2.5).

Picks a handful of (origin, destination) station pairs, runs the
backward solver from each destination, and compares the recovered
travel time against the corresponding entry in the forward
pair_times.npz / pair_matrix.npz produced by solve_all.py.

Pass criterion: |T_back - T_fwd| / T_fwd < 5% on average across the
sampled pairs (5% accommodates discretisation differences from peak-
time detection in the leapfrog scheme; the math is exact in the
continuum limit).

Usage on a GPU node:
    python3 validate_backward_real.py \
        --pair-file pair_matrix.npz --n-pairs 6
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
from jax import jit

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "model"))
from solve_all import gaussian_source, solve_one, N_STEPS
from solve_backward import build_backward_fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="domain.npz",
                   help="Domain mask file (use domain_5m.npz for 5m grid).")
    p.add_argument("--pair-file", default="pair_matrix.npz",
                   help="Forward T matrix file (T_at_stations[i,j] = "
                        "travel time from src i to dst j)")
    p.add_argument("--n-pairs", type=int, default=4,
                   help="Number of distinct destinations to backward-solve.")
    p.add_argument("--n-origins-per-dst", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-t-fwd", type=float, default=30.0,
                   help="Skip pairs whose forward T is below this (s) — "
                        "too close to the source, peak detection picks up "
                        "the initial pulse's spatial tail rather than a "
                        "propagated wave.")
    p.add_argument("--rel-tol", type=float, default=0.05,
                   help="Pass threshold on mean relative error (default 5%).")
    args = p.parse_args()

    print(f"[load] {args.domain}")
    z = np.load(_THIS.parent / "input" / args.domain, allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)

    pair_path = Path(args.pair_file)
    if not pair_path.is_absolute():
        pair_path = _THIS.parent / "output" / "wave" / pair_path
    if not pair_path.exists():
        raise SystemExit(f"missing {pair_path}: run solve_all.py first")

    pf = np.load(pair_path)
    # Different builders use different keys: aggregate_pair_matrix.py
    # writes 'T_matrix', solve_all.py writes 'T_at_stations'.
    for key in ("T_matrix", "T_at_stations", "T"):
        if key in pf.files:
            Tfwd = pf[key]
            print(f"[fwd] using key '{key}'")
            break
    else:
        raise SystemExit(f"no recognised T-matrix key in {pair_path.name}; "
                         f"keys={list(pf.files)}")
    print(f"[fwd] forward matrix shape {Tfwd.shape}, "
          f"{np.isfinite(Tfwd).sum()} finite entries")

    rng = np.random.default_rng(args.seed)

    # Pick destinations whose forward column has at least n_origins finite
    # entries — otherwise we have nothing to compare to.
    finite_per_col = np.isfinite(Tfwd).sum(axis=0)
    candidates = np.where(finite_per_col >= args.n_origins_per_dst + 1)[0]
    if len(candidates) < args.n_pairs:
        raise SystemExit(f"only {len(candidates)} dsts with >= "
                         f"{args.n_origins_per_dst} reachable origins")
    dst_indices = rng.choice(candidates, size=args.n_pairs, replace=False)
    print(f"[pairs] sampled destinations: {dst_indices.tolist()}")

    # ---- station cell indices (must match solve_all.py's mapping) ----
    st_ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx-2)
    st_iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny-2)
    st_iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz-2)

    inside_b = mask[st_ix, st_iy, st_iz]
    if inside_b.any():
        bad = np.where(inside_b)[0]
        for b in bad:
            ix0, iy0, iz0 = st_ix[b], st_iy[b], st_iz[b]
            found = False
            for r in range(1, 6):
                for di in range(-r, r+1):
                    for dj in range(-r, r+1):
                        if max(abs(di), abs(dj)) != r:
                            continue
                        ix = ix0 + di; iy = iy0 + dj
                        if not (0 < ix < nx-1 and 0 < iy < ny-1):
                            continue
                        if not mask[ix, iy, iz0]:
                            st_ix[b], st_iy[b] = ix, iy
                            found = True; break
                    if found:
                        break
                if found:
                    break
            if not found:
                iz = iz0
                while iz < nz-1 and mask[ix0, iy0, iz]:
                    iz += 1
                st_iz[b] = iz

    # ---- one-time JIT setup ----
    mask_jnp = jnp.asarray(mask)
    s = build_backward_fields(mask_jnp, dx, dy, dz)
    one_minus_damp = (1.0 - s["damp"]).astype(jnp.float32)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))

    print()
    print("dst  origin       T_fwd     T_bwd    abs_err  rel_err")
    print("-" * 60)

    abs_errs = []
    rel_errs = []
    for dst in dst_indices:
        idx = (int(st_ix[dst]), int(st_iy[dst]), int(st_iz[dst]))
        src_spatial = gaussian_source(mask_jnp.shape, idx)
        t0 = time.time()
        T = solve_jit(s["c2_xy"], s["c2_z"],
                      s["u"], s["v"], s["w"],
                      s["ax"], s["ay"], s["az"],
                      one_minus_damp, s["free"], src_spatial,
                      dx, dy, dz, args.n_steps)
        jax.block_until_ready(T)
        T_np = np.asarray(jax.device_get(T))
        T_bwd_at_origins = T_np[st_ix, st_iy, st_iz]
        elapsed = time.time() - t0

        # Pick origins to compare: finite forward time, T_fwd >= min,
        # and exclude self.
        col = Tfwd[:, dst]
        valid = np.where(np.isfinite(col)
                         & (col >= args.min_t_fwd)
                         & (np.arange(n_st) != dst))[0]
        sel = rng.choice(valid, size=min(args.n_origins_per_dst, len(valid)),
                         replace=False)
        for o in sel:
            t_fwd = float(col[o])
            t_bwd = float(T_bwd_at_origins[o])
            if not np.isfinite(t_bwd):
                print(f"{int(dst):3d}  {int(o):3d}    {t_fwd:8.2f}  "
                      f"{t_bwd:>8}  ---     ---     (skip: bwd inf)")
                continue
            ae = abs(t_fwd - t_bwd)
            re = ae / max(abs(t_fwd), 1e-9)
            abs_errs.append(ae)
            rel_errs.append(re)
            print(f"{int(dst):3d}  {int(o):3d}    {t_fwd:8.2f}  "
                  f"{t_bwd:8.2f}  {ae:7.2f}  {re:6.2%}")
        print(f"     [solve {elapsed:.1f}s]")

    print()
    print(f"summary  n={len(rel_errs)}  "
          f"mean_rel_err={np.mean(rel_errs):.2%}  "
          f"max_rel_err={np.max(rel_errs):.2%}  "
          f"mean_abs_err={np.mean(abs_errs):.3f}s")

    threshold = args.rel_tol
    if np.mean(rel_errs) < threshold:
        print(f"PASS (mean rel error < {threshold:.0%})")
    else:
        print(f"FAIL (mean rel error >= {threshold:.0%})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
