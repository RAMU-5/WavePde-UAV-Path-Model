"""E7 — wind-exploitation benefit (flight time & energy) on the 50 m domain.

Supports the claim that embedding wind in the planner reduces flight
time and energy versus wind-ignoring planning. Protocol, per station
pair and per wind strength:

  1. plan AWARE  : wave solve with the truth vortex field, trace -grad T;
  2. plan NAIVE  : wave solve with q = 0, trace -grad T;
  3. fly BOTH paths under the truth wind with the same constant-airspeed
     policy (seg_times_wind — the identical metric used by the wind-aware
     Dijkstra reference of exp4), and compare arrival times.

Under the constant-airspeed cruise model the aerodynamic power draw is
constant, so relative energy savings equal relative time savings; the
absolute energy figure uses a representative delivery-multirotor cruise
power P_CRUISE. Naive paths that cross a corridor rendered one-way by
the regulatory band ("supersonic" upwind segments -> infinite metric
time) are counted separately as infeasible.

Truth field: the solenoidal vortex array of exp4 (3 km cells,
analytically divergence-free) at |q|_max in {1.5, 3.0, 4.5} m/s =
Mach {0.1, 0.2, 0.3} of c0 = 15 m/s, multiplied by an urban-canopy
profile: zero below the 30 m regulatory band, linear ramp to full
strength at 40 m. Without the profile, the z-independent vortex makes
every take-off/landing leg upwind-infeasible inside the low-altitude
band (speed cap 0.05*c0 = 0.75 m/s < |q|), which exp4 characterised
separately as the deliberately supersonic regime; the canopy profile
matches the sheltered street-canyon winds described in the
introduction and is seen identically by planner and evaluator.

Outputs output/runs/exp15_energy/{exp15_data.npz, results.json}.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scipy.ndimage import gaussian_filter  # noqa: E402

from experiments.exp1_dijkstra_gt import nudge_stations  # noqa: E402
from experiments.exp4_wind import (vortex_wind, build_static_wind,  # noqa: E402
                                   seg_times_wind)
from experiments.exp5_broadband import path_collides  # noqa: E402
import model.wave_api as wave_api  # noqa: E402
from model.wave_api import WaveConfig  # noqa: E402
from model.solve_coupled import trace_path  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp15_energy"

SRC_IDS = list(range(0, 118, 10))
MACHS = {0.1: 1.5, 0.2: 3.0, 0.3: 4.5}   # Mach -> |q|_max [m/s]
LO_D, HI_D = 700.0, 3000.0
P_CRUISE = 500.0                          # representative cruise power [W]
CANOPY_Z0, CANOPY_Z1 = 30.0, 40.0        # wind ramp [m], matches reg band


def canopy_profile(nz, dz):
    zc = (np.arange(nz) + 0.5) * dz
    return np.clip((zc - CANOPY_Z0) / (CANOPY_Z1 - CANOPY_Z0),
                   0.0, 1.0).astype(np.float32)


def flight_time(path, u3, v3, w3, dx, dy, dz):
    t = seg_times_wind(path, u3, v3, w3, dx, dy, dz)
    return float(np.sum(t)) if np.all(np.isfinite(t)) else np.inf


def smooth_T(T):
    """Normalised-convolution smoothing of the finite region (sigma = 2
    cells); buildings keep a sharp repulsive plateau (exp5 tracer fix)."""
    fin = np.isfinite(T)
    num = gaussian_filter(np.where(fin, T, 0.0), 2.0)
    den = gaussian_filter(fin.astype(np.float64), 2.0)
    Tsm = num / np.maximum(den, 1e-9)
    return np.where(fin, Tsm, T[fin].max() + 1e6)


def trace_ok(T, src_xyz, dst_xyz, mask, dx, dy, dz):
    path, status = trace_path(T, src_xyz, dst_xyz, dx, dy, dz)
    if status != "reached" or path_collides(path, mask, dx, dy, dz):
        return None
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations = z["stations_xy_m"]
    ix, iy, iz = nudge_stations(mask, stations, dx, dy, dz)
    st_xyz = np.stack([(ix + 0.5) * dx, (iy + 0.5) * dy,
                       (iz + 0.5) * dz], axis=1)
    st2 = st_xyz[:, :2]

    cfg = WaveConfig()
    mask_j = jnp.asarray(mask)
    zero = np.zeros((nx, ny, nz), dtype=np.float32)

    # ---- naive plan: one zero-wind solve + trace per source ------------
    static0 = build_static_wind(mask_j, dx, dy, dz, zero, zero, zero)
    paths0 = {}
    for s in SRC_IDS:
        t0 = time.time()
        T = np.asarray(wave_api._run(static0, (int(ix[s]), int(iy[s]),
                                               int(iz[s])),
                                     dx, dy, dz, cfg), np.float64)
        Ts = smooth_T(T)
        d = np.hypot(*(st2 - st2[s]).T)
        cand = np.where((d >= LO_D) & (d <= HI_D))[0]
        for b in cand:
            if b == s:
                continue
            p = trace_ok(Ts, st_xyz[s], st_xyz[b], mask, dx, dy, dz)
            if p is not None:
                paths0[(s, int(b))] = p
        print(f"[naive src {s}] {len(cand)} candidates, "
              f"{sum(1 for k in paths0 if k[0] == s)} traced, "
              f"{time.time()-t0:.1f}s", flush=True)

    # ---- per wind strength: aware plan + paired fly-off -----------------
    results = {}
    per_pair = {}
    example = None
    winds = {}
    prof = canopy_profile(nz, dz)
    for mach, qmax in MACHS.items():
        u3, v3, w3 = vortex_wind(nx, ny, nz, dx, dy, qmax)
        u3 = (u3 * prof[None, None, :]).astype(np.float32)
        v3 = (v3 * prof[None, None, :]).astype(np.float32)
        winds[mach] = (u3, v3, w3)
        static_w = build_static_wind(mask_j, dx, dy, dz, u3, v3, w3)
        rows = []
        n_naive_inf = n_aware_inf = n_trace_fail = 0
        for s in SRC_IDS:
            Tw = smooth_T(np.asarray(
                wave_api._run(static_w,
                              (int(ix[s]), int(iy[s]), int(iz[s])),
                              dx, dy, dz, cfg), np.float64))
            for (ss, b), p_naive in paths0.items():
                if ss != s:
                    continue
                p_aware = trace_ok(Tw, st_xyz[s], st_xyz[b],
                                   mask, dx, dy, dz)
                if p_aware is None:
                    n_trace_fail += 1
                    continue
                t_aware = flight_time(p_aware, u3, v3, w3, dx, dy, dz)
                t_naive = flight_time(p_naive, u3, v3, w3, dx, dy, dz)
                if not np.isfinite(t_naive):
                    n_naive_inf += 1
                    continue
                if not np.isfinite(t_aware):
                    n_aware_inf += 1
                    continue
                d2 = float(np.hypot(*(st2[b] - st2[s])))
                rows.append((s, b, d2, t_naive, t_aware))
        if not rows:
            raise RuntimeError(f"no valid pairs at M={mach}")
        arr = np.array([(r[2], r[3], r[4]) for r in rows])
        dt_abs = arr[:, 1] - arr[:, 2]
        rel = dt_abs / arr[:, 1]
        results[mach] = dict(
            n_pairs=len(rows),
            n_naive_infeasible=n_naive_inf,
            n_aware_infeasible=n_aware_inf,
            n_trace_fail=n_trace_fail,
            rel_med=float(np.median(rel)),
            rel_p95=float(np.percentile(rel, 95)),
            rel_mean=float(rel.mean()),
            frac_negative=float((rel < -1e-3).mean()),
            abs_med_s=float(np.median(dt_abs)),
            abs_p95_s=float(np.percentile(dt_abs, 95)),
            energy_med_Wh=float(np.median(dt_abs) * P_CRUISE / 3600.0),
            energy_p95_Wh=float(np.percentile(dt_abs, 95)
                                * P_CRUISE / 3600.0),
        )
        per_pair[mach] = np.array(rows)
        print(f"[M={mach}] n={len(rows)} rel med={results[mach]['rel_med']:.2%} "
              f"p95={results[mach]['rel_p95']:.2%} "
              f"abs med={results[mach]['abs_med_s']:.1f}s "
              f"naive_inf={n_naive_inf} neg={results[mach]['frac_negative']:.1%}",
              flush=True)
        # illustrative pair: largest absolute saving at the strongest wind
        if mach == max(MACHS):
            k = int(np.argmax(dt_abs))
            s, b = int(rows[k][0]), int(rows[k][1])
            Tw = smooth_T(np.asarray(
                wave_api._run(static_w,
                              (int(ix[s]), int(iy[s]), int(iz[s])),
                              dx, dy, dz, cfg), np.float64))
            p_aware = trace_ok(Tw, st_xyz[s], st_xyz[b], mask, dx, dy, dz)
            example = dict(src=s, dst=b,
                           path_aware=p_aware, path_naive=paths0[(s, b)],
                           t_naive=rows[k][3], t_aware=rows[k][4])

    # ---- save ------------------------------------------------------------
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    u3, v3, _ = winds[max(MACHS)]
    np.savez_compressed(
        OUT_DIR / "exp15_data.npz",
        mask_z5=mask[:, :, 5],
        u_z5=u3[:, :, 5], v_z5=v3[:, :, 5],
        dx=dx, dy=dy, dz=dz,
        stations_xyz=st_xyz,
        example_src=example["src"], example_dst=example["dst"],
        example_path_aware=example["path_aware"],
        example_path_naive=example["path_naive"],
        example_t_naive=example["t_naive"],
        example_t_aware=example["t_aware"],
        **{f"pairs_M{str(m).replace('.', '')}": per_pair[m]
           for m in MACHS})
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(dict(machs={str(m): results[m] for m in MACHS},
                       p_cruise_W=P_CRUISE, src_ids=SRC_IDS,
                       band_m=[LO_D, HI_D], git_sha=sha), f, indent=2)
    print(f"[done] -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
