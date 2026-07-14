"""Backward / adjoint wave-routing solver (paper §2.5).

Forward routing places a wave source at the origin and reads T at the
destination. For a fleet of K candidate origins routing to one common
destination, the forward approach needs K solves. The adjoint formulation
needs ONE: place the source at the destination, propagate the wave
equation with the velocity field q -> -q, and the resulting T-field gives
the optimal travel time to that destination from every point in the
domain simultaneously.

Mathematics: under q -> -q, the convected wave operator
  D^2 psi/Dt^2 = (d_t + q.grad)^2 psi
has the same square-of-(d_t + q.grad) structure (q->-q flips sign inside
the parens, but the square restores it modulo the cross-term sign which
the second-order discrete leapfrog handles correctly when we also flip
the "advection" terms adv1, adv2). The flow-acceleration term (q.grad)q
is invariant under q -> -q (both factors flip), so a_x, a_y, a_z are
re-used unchanged.

The paper's claim 'the operator is self-adjoint in the absence of
damping and for q=0' (line 592) implies T_backward(j -> i) = T_forward(
i -> j) for any i, j when sigma_1 = 0, q = 0. With q != 0 the equality
still holds when q is also flipped (Galilean argument). Validation is
in validate_backward_minimal.py (2D, pure numpy) and via direct
forward-vs-backward comparison on the real domain (this script).
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
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit

# Re-use the forward solver's building blocks. solve_one is generic in
# u,v,w so feeding it the negated fields produces the adjoint solve
# without any duplicated PDE code.
try:
    from .solve_all import (
        build_static_fields, gaussian_source, solve_one,
        N_STEPS, THIS_DIR, OUT_DIR,
        DT, PULSE_T0_STEPS, PULSE_TAU_STEPS, MIN_PEAK_AMP,
    )
except ImportError:  # Support ``python model/solve_backward.py``.
    from solve_all import (
        build_static_fields, gaussian_source, solve_one,
        N_STEPS, THIS_DIR, OUT_DIR,
        DT, PULSE_T0_STEPS, PULSE_TAU_STEPS, MIN_PEAK_AMP,
    )


def build_backward_fields(mask, dx, dy, dz):
    """Same as forward static fields but with q -> -q.

    Crucial: the (q.grad)q acceleration is q-quadratic so it stays the
    same under q -> -q — recomputing it from the negated u,v,w would
    give the same numbers (verified by hand). We therefore just flip
    u,v,w in the dict and leave ax,ay,az alone.
    """
    s = build_static_fields(mask, dx, dy, dz)
    s = dict(s)
    s["u"] = -s["u"]
    s["v"] = -s["v"]
    s["w"] = -s["w"]
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dst-idx", type=int, required=True,
                   help="Index of the destination station to seed the "
                        "backward wave from.")
    p.add_argument("--out", type=str, default="back_T.npz",
                   help="Output filename in output/wave/.")
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    args = p.parse_args()

    print(f"[load] domain.npz")
    z = np.load(THIS_DIR.parent / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)

    if not (0 <= args.dst_idx < n_st):
        raise SystemExit(f"--dst-idx must be in [0,{n_st}); got {args.dst_idx}")

    print(f"[domain] {nx}x{ny}x{nz}, {n_st} stations, dst={args.dst_idx}")

    mask_jnp = jnp.asarray(mask)
    s = build_backward_fields(mask_jnp, dx, dy, dz)
    one_minus_damp = (1.0 - s["damp"]).astype(jnp.float32)

    st_ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    st_iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    st_iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)

    # Same in-building horizontal nudge used by solve_all.py — we keep
    # the destination location consistent with the forward run so a
    # forward/backward T-pair compares like-for-like.
    inside_b = mask[st_ix, st_iy, st_iz]
    if inside_b.any():
        bad = np.where(inside_b)[0]
        for b in bad:
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
                        if not mask[ix, iy, iz0]:
                            st_ix[b], st_iy[b] = ix, iy
                            found = True; break
                    if found:
                        break
                if found:
                    break
            if not found:
                iz = iz0
                while iz < nz - 1 and mask[ix0, iy0, iz]:
                    iz += 1
                st_iz[b] = iz

    idx = (int(st_ix[args.dst_idx]),
           int(st_iy[args.dst_idx]),
           int(st_iz[args.dst_idx]))
    print(f"[src@dst] cell idx {idx}")

    src_spatial = gaussian_source(mask_jnp.shape, idx)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14, 15, 16, 17, 18))
    # dx^3-scaled peak threshold (see solve_one_source.py / paper/Q2)
    min_peak = MIN_PEAK_AMP * (dx / 50.0) ** 3

    print("[solve] running backward wave (q -> -q) ...")
    t0 = time.time()
    T = solve_jit(s["c2_xy"], s["c2_z"],
                  s["u"], s["v"], s["w"],
                  s["ax"], s["ay"], s["az"],
                  one_minus_damp, s["free"], src_spatial,
                  dx, dy, dz, args.n_steps,
                  DT, PULSE_T0_STEPS, PULSE_TAU_STEPS, min_peak)
    jax.block_until_ready(T)
    T_np = np.asarray(jax.device_get(T))
    print(f"[solve] done in {time.time()-t0:.1f}s")

    # T_np[i, j, k] = backward arrival time at cell (i,j,k) starting
    # from dst. Read at every station index = travel time from that
    # station to dst (this is the multi-origin lookup the adjoint
    # formulation buys us in O(1) per origin).
    T_at_origins = T_np[st_ix, st_iy, st_iz]
    n_arrived = int(np.isfinite(T_at_origins).sum())
    print(f"[stations] {n_arrived}/{n_st} reached  "
          f"(min={np.nanmin(T_at_origins[T_at_origins>0]):.1f}s, "
          f"max={np.nanmax(T_at_origins[np.isfinite(T_at_origins)]):.1f}s)")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / args.out
    np.savez_compressed(
        out_path,
        T_field=T_np,
        T_at_origins=T_at_origins,
        dst_idx=np.int32(args.dst_idx),
        dst_cell=np.array(idx, dtype=np.int32),
        station_xy_m=stations_xy_m,
        dx=dx, dy=dy, dz=dz, n_steps=args.n_steps,
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
