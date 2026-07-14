"""Wave / dynamics fixed-point coupling iteration (paper §2.10).

Paper claim: the wave-equation parameters (c, c_z, q) and the UAV
dynamics must be self-consistent. Initial wave solve uses spec-sheet
speeds; the path it produces may demand climb rates the UAV cannot
sustain. The iteration alternates:

    wave-solve(c, c_z, w_grav)
        --> path = grad-descent on T
            --> dynamics(path) gives V(s) and V_z(s)
                --> c0_new, cz_up_new, cz_down_new from observed V
                    --> back to wave-solve

repeated until path geometry and parameters stabilise.

Currently the rest of the codebase runs ONE wave solve and integrates
dynamics on the result — no feedback loop. This script adds the loop.

Algorithm
---------
Given params p_k = (c0_k, cz_up_k, cz_down_k):
  1. Build static fields from p_k.
  2. Solve wave from src.
  3. Trace path src->dst on T-field via grad-descent.
  4. Integrate dynamics along path -> V(s), tangent(s).
  5. Decompose:
        V_horiz_seg  = V * sqrt(1 - slope^2)
        V_up_seg     = V * slope         when slope > 0 (ascending)
        V_down_seg   = V * (-slope)      when slope < 0 (descending)
  6. New params (with relaxation alpha):
        c0_new   = (1-a)*c0_k   + a * mean(V_horiz_seg)         (always)
        cz_up    = (1-a)*cz_up  + a * mean(V_up_seg)            (only if any ascending seg)
        cz_down  = (1-a)*cz_dn  + a * mean(V_down_seg)          (only if any descending)
  7. Stop when ||p_new - p_k||/||p_k|| < tol AND
              path geometry change < tol (Frechet-style mean displacement).

Outputs
-------
  - converged (c0*, cz_up*, cz_down*)
  - converged path
  - per-iteration log: params, path RMS displacement, dynamics V_max
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
from scipy.ndimage import map_coordinates

try:
    from .solve_all import (
        build_static_fields_param, gaussian_source, solve_one,
        N_STEPS, C0 as C0_DEFAULT, CZ_UP as CZ_UP_DEFAULT,
        CZ_DOWN as CZ_DOWN_DEFAULT, WIND as WIND_DEFAULT,
    )
    from .uav_dynamics import integrate_dynamics, UAVDynParams
except ImportError:  # Support ``python model/solve_coupled.py``.
    from solve_all import (
        build_static_fields_param, gaussian_source, solve_one,
        N_STEPS, C0 as C0_DEFAULT, CZ_UP as CZ_UP_DEFAULT,
        CZ_DOWN as CZ_DOWN_DEFAULT, WIND as WIND_DEFAULT,
    )
    from uav_dynamics import integrate_dynamics, UAVDynParams

THIS_DIR = _THIS


# ----- helpers ---------------------------------------------------------

def trace_path(T, source_xyz, dest_xyz, dx, dy, dz,
               gamma=0.3, max_steps=10000):
    """Vendored from build_database.py (gradient-descent on -∇T)."""
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
            best_d = d; best_idx = len(path) - 1; stale = 0
        else:
            stale += 1
        if d < 2.0 * max(dx, dy, dz):
            path.append(src.copy())
            return np.array(path[::-1]), "reached"
        if stale > 200:
            path = path[: best_idx + 1]; path.append(src.copy())
            return np.array(path[::-1]), f"truncated_{best_d:.0f}m"
    return np.array(path[::-1]), "max_steps"


def decompose_speeds(path, V):
    """Per segment: V_horiz, V_up, V_down (the latter two non-negative)."""
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    tangent = seg / np.maximum(seg_len[:, None], 1e-9)
    slope = tangent[:, 2]                                   # cos(angle)
    sin_xy = np.sqrt(np.maximum(1.0 - slope ** 2, 0.0))
    V_seg = 0.5 * (V[:-1] + V[1:])
    V_h = V_seg * sin_xy
    V_z = V_seg * slope                                     # signed
    asc = slope > 1e-3
    desc = slope < -1e-3
    return dict(V_h=V_h, V_z=V_z, slope=slope, seg_len=seg_len,
                asc=asc, desc=desc)


def update_params(decomp, p_old, alpha=0.5,
                  bounds_horiz=(2.0, 25.0),
                  bounds_vert=(0.5, 10.0)):
    """Move the param triple a fraction alpha toward observed values.

    Returns (p_new, observed_dict). p_new is clamped to physically
    reasonable bounds so that no observation outlier (e.g. one-cell
    wobble) collapses the wave solver."""
    c0_old, cz_up_old, cz_down_old = p_old

    # Length-weighted means
    L = decomp["seg_len"]
    V_h = decomp["V_h"]
    asc = decomp["asc"]; desc = decomp["desc"]
    V_z = decomp["V_z"]

    horiz_total = float(np.sum(L * (1 - decomp["slope"] ** 2)))
    if horiz_total > 1e-3:
        c0_obs = float(np.sum(L * V_h) / np.sum(L))
        c0_new = (1 - alpha) * c0_old + alpha * c0_obs
    else:
        c0_obs = float("nan"); c0_new = c0_old

    if asc.any():
        cz_up_obs = float(np.sum(L[asc] * V_z[asc]) / np.sum(L[asc]))
        cz_up_new = (1 - alpha) * cz_up_old + alpha * cz_up_obs
    else:
        cz_up_obs = float("nan"); cz_up_new = cz_up_old

    if desc.any():
        cz_down_obs = float(np.sum(L[desc] * (-V_z[desc])) / np.sum(L[desc]))
        cz_down_new = (1 - alpha) * cz_down_old + alpha * cz_down_obs
    else:
        cz_down_obs = float("nan"); cz_down_new = cz_down_old

    c0_new = float(np.clip(c0_new, *bounds_horiz))
    cz_up_new = float(np.clip(cz_up_new, *bounds_vert))
    cz_down_new = float(np.clip(cz_down_new, *bounds_vert))

    return ((c0_new, cz_up_new, cz_down_new),
            dict(c0_obs=c0_obs, cz_up_obs=cz_up_obs, cz_down_obs=cz_down_obs))


def path_change(p1, p2):
    """Mean nearest-neighbour distance from path 1 to path 2 (Frechet-lite).
    Returns RMS in metres."""
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return float("inf")
    # For each point in p1, distance to nearest in p2.
    diffs = p1[:, None, :] - p2[None, :, :]
    d = np.linalg.norm(diffs, axis=2)
    return float(np.sqrt((d.min(axis=1) ** 2).mean()))


# ----- main loop -------------------------------------------------------

def run_coupled(mask, dx, dy, dz, src_xyz, dst_xyz,
                p0=(C0_DEFAULT, CZ_UP_DEFAULT, CZ_DOWN_DEFAULT),
                wind=WIND_DEFAULT,
                max_iter=8, alpha=0.5, tol_param=0.01, tol_path_m=2.0,
                n_steps=N_STEPS, dyn_params=None,
                verbose=True):
    """Returns dict with params*, path*, history."""
    if dyn_params is None:
        dyn_params = UAVDynParams()
    mask_jnp = jnp.asarray(mask)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))
    src_idx = (int(round(src_xyz[0]/dx)),
               int(round(src_xyz[1]/dy)),
               int(round(src_xyz[2]/dz)))
    src_spatial = gaussian_source(mask_jnp.shape, src_idx)

    p = tuple(float(x) for x in p0)
    history = []
    prev_path = None
    # Defaults so the early-break paths still return a sensible dict.
    path = np.array([src_xyz, dst_xyz], dtype=np.float64)
    dyn = None
    T_np = None

    for k in range(max_iter):
        c0, cz_up, cz_down = p
        if verbose:
            print(f"[iter {k}] c0={c0:.3f}  cz_up={cz_up:.3f}  "
                  f"cz_down={cz_down:.3f}")
        s = build_static_fields_param(mask_jnp, dx, dy, dz,
                                      c0=c0, cz_up=cz_up, cz_down=cz_down,
                                      wind=wind)
        one_minus_damp = (1.0 - s["damp"]).astype(jnp.float32)
        T = solve_jit(s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
                      s["ax"], s["ay"], s["az"], one_minus_damp,
                      s["free"], src_spatial, dx, dy, dz, n_steps)
        T_np = np.asarray(jax.device_get(T))

        path, status = trace_path(T_np, src_xyz, dst_xyz, dx, dy, dz)
        if status not in ("reached",) and not status.startswith("truncated_"):
            if verbose:
                print(f"  [warn] path-trace status={status}")
        if path.shape[0] < 2:
            history.append(dict(
                iter=k, params=p, params_new=p, status="no_path",
                rel_change=0.0, path_rms=float("inf"),
                path_len_m=float("nan"), t_dynamics_s=float("nan"),
                v_max=float("nan"), v_min=float("nan"),
                n_path_pts=int(path.shape[0]),
                observed=dict(c0_obs=float("nan"),
                              cz_up_obs=float("nan"),
                              cz_down_obs=float("nan")),
            ))
            break

        dyn = integrate_dynamics(path, dyn_params)
        decomp = decompose_speeds(path, dyn["V"])
        p_new, obs = update_params(decomp, p, alpha=alpha)

        # Convergence checks
        rel_change = max(
            abs(p_new[i] - p[i]) / max(p[i], 1e-9) for i in range(3))
        path_rms = path_change(path, prev_path) if prev_path is not None else float("inf")

        record = dict(
            iter=k, params=p, params_new=p_new, observed=obs, status=status,
            path_len_m=float(dyn["s"][-1]),
            t_dynamics_s=float(dyn["t"][-1]),
            v_max=float(dyn["V"].max()), v_min=float(dyn["V"].min()),
            rel_change=rel_change, path_rms=path_rms,
            n_path_pts=int(path.shape[0]),
        )
        history.append(record)

        if verbose:
            print(f"  [obs] c0_obs={obs['c0_obs']:.3f} "
                  f"cz_up_obs={obs['cz_up_obs']:.3f} "
                  f"cz_down_obs={obs['cz_down_obs']:.3f}  "
                  f"rel_change={rel_change:.4f}  path_rms={path_rms:.2f}m")

        prev_path = path
        p = p_new

        if rel_change < tol_param and path_rms < tol_path_m:
            if verbose:
                print(f"[converged] after {k+1} iterations")
            break
    else:
        if verbose:
            print(f"[max-iter reached] returning iter {max_iter} state")

    return dict(params=p, path=path, dynamics=dyn, history=history,
                T_field=T_np)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-idx", type=int, default=48)
    p.add_argument("--dst-idx", type=int, default=83)
    p.add_argument("--max-iter", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out", type=str, default="coupled_result.npz")
    # Allow forcing a deliberately-wrong starting cz_up for the test
    p.add_argument("--cz-up-init", type=float, default=CZ_UP_DEFAULT)
    p.add_argument("--cz-down-init", type=float, default=CZ_DOWN_DEFAULT)
    args = p.parse_args()

    z = np.load(THIS_DIR.parent / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations = z["stations_xy_m"]

    src_xyz = stations[args.src_idx]
    dst_xyz = stations[args.dst_idx]
    print(f"[domain] {nx}x{ny}x{nz}, stations {len(stations)}")
    print(f"[pair] src={args.src_idx} {src_xyz}  dst={args.dst_idx} {dst_xyz}")
    print(f"[init] c0={C0_DEFAULT} cz_up={args.cz_up_init} "
          f"cz_down={args.cz_down_init}")

    t0 = time.time()
    result = run_coupled(mask, dx, dy, dz, src_xyz, dst_xyz,
                        p0=(C0_DEFAULT, args.cz_up_init, args.cz_down_init),
                        max_iter=args.max_iter, alpha=args.alpha)
    elapsed = time.time() - t0

    print(f"\n[done] {elapsed:.1f}s, "
          f"{len(result['history'])} iterations")
    print(f"[final params] c0={result['params'][0]:.3f}  "
          f"cz_up={result['params'][1]:.3f}  "
          f"cz_down={result['params'][2]:.3f}")

    np.savez_compressed(
        THIS_DIR / args.out,
        params_final=np.array(result["params"], dtype=np.float32),
        path_final=result["path"].astype(np.float32),
        T_field=result["T_field"].astype(np.float32),
        history=np.array([(h.get("iter", -1),
                           h.get("params", (0,0,0))[0],
                           h.get("params", (0,0,0))[1],
                           h.get("params", (0,0,0))[2],
                           h.get("rel_change", float("nan")),
                           h.get("path_rms", float("nan")),
                           h.get("path_len_m", float("nan")),
                           h.get("t_dynamics_s", float("nan")))
                          for h in result["history"]],
                         dtype=np.float32),
        src_xyz=src_xyz.astype(np.float32),
        dst_xyz=dst_xyz.astype(np.float32),
    )
    print(f"[saved] {THIS_DIR / args.out}")


if __name__ == "__main__":
    main()
