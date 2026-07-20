"""E2.3 — single-GPU runtime scaling of FMM vs the wave solver.

The Nanshan 5 m building mask is coarsened by max-pooling to 80, 40, 20,
10 and 5 m, giving problem sizes spanning roughly three decades of cell
count on identical physical geometry.  At every resolution one
single-source solve from station 50 is timed for

* the AGD CUDA ``Rander3`` fast-marching baseline (the exp12 zero-wind
  asymmetric-vertical metric), and
* the ConvFEM wave solver (the exp11 Nanshan configuration with all
  time-step counts scaled to keep the physical setup fixed:
  ``dt = 0.02 dx``, ``t_end = 340 s``, 1.0 s pulse centre, 0.4 s pulse
  width, 40 m physical sponge).

The first call at each size is reported separately (JIT/compile warm-up);
steady-state timing is the median of the remaining repeats.  Per-solver
log-log slopes of steady time versus cell count quantify the observed
scaling exponents.  Run each solver in its own process so CuPy and JAX
never share the GPU:

    venv/bin/python -m experiments.exp16_scaling --solver fmm
    venv/bin/python -m experiments.exp16_scaling --solver wave
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.exp1_dijkstra_gt import nudge_stations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp16_scaling"
DOMAIN = ROOT / "input" / "domain_5m.npz"

SOURCE = 50
T_END_S = 340.0
FACTORS = (16, 8, 4, 2, 1)


def coarsen_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Conservative max-pool: a coarse cell is solid if any fine cell is."""
    if factor == 1:
        return mask
    pad = [(0, (-n) % factor) for n in mask.shape]
    padded = np.pad(mask, pad, constant_values=False)
    nx, ny, nz = (n // factor for n in padded.shape)
    return padded.reshape(nx, factor, ny, factor, nz, factor).any(axis=(1, 3, 5))


def source_index(mask: np.ndarray, stations_xy_m: np.ndarray, h: float
                 ) -> tuple[int, int, int]:
    ix, iy, iz = nudge_stations(mask, stations_xy_m, h, h, h)
    return int(ix[SOURCE]), int(iy[SOURCE]), int(iz[SOURCE])


def run_fmm(mask: np.ndarray, h: float, repeats: int) -> dict:
    import cupy as cp

    from experiments.exp12_fmm_baseline import exact_rander_field
    from aura.fmm.agd_solver import solve_single_source

    field, grid = exact_rander_field(mask, h, h, h)
    src = source_index(mask, _stations(), h)
    src_xyz = tuple((np.asarray(src) + 0.5) * h)
    times = []
    finite = 0.0
    for _ in range(repeats + 1):
        started = time.perf_counter()
        arrival = solve_single_source(field, grid, src_xyz, backend="cuda")
        times.append(time.perf_counter() - started)
        finite = float(np.isfinite(arrival.T).mean())
    return {
        "times_s": times,
        "finite_fraction": finite,
        "memory_pool_peak_bytes": int(cp.get_default_memory_pool().total_bytes()),
    }


def run_wave(mask: np.ndarray, h: float, repeats: int) -> dict:
    from model.wave_api import WaveConfig, build_static_fields, solve_forward

    dt = 0.02 * h
    cfg = WaveConfig(
        dt=dt,
        n_steps=math.ceil(T_END_S / dt),
        pulse_t0_steps=max(2, round(1.0 / dt)),
        pulse_tau_steps=max(1, round(0.4 / dt)),
        sponge_layers=max(2, round(40.0 / h)),
        min_peak_amp=1e-6 * (h / 50.0) ** 3,
    )
    src = source_index(mask, _stations(), h)
    build_started = time.perf_counter()
    static = build_static_fields(mask, h, h, h, cfg=cfg)
    build_s = time.perf_counter() - build_started
    times = []
    finite = 0.0
    for _ in range(repeats + 1):
        started = time.perf_counter()
        T = solve_forward(mask, h, h, h, src, cfg=cfg, static=static)
        times.append(time.perf_counter() - started)
        finite = float(np.isfinite(T).mean())
    return {
        "times_s": times,
        "finite_fraction": finite,
        "dt_s": dt,
        "n_steps": cfg.n_steps,
        "static_build_s": build_s,
    }


_STATIONS: np.ndarray | None = None


def _stations() -> np.ndarray:
    global _STATIONS
    if _STATIONS is None:
        _STATIONS = np.load(DOMAIN, allow_pickle=False)["stations_xy_m"]
    return _STATIONS


def gpu_name() -> str:
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip().splitlines()[0] if probe.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("fmm", "wave"), required=True)
    parser.add_argument("--factors", default=",".join(str(f) for f in FACTORS))
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    factors = [int(f) for f in args.factors.split(",")]

    domain = np.load(DOMAIN, allow_pickle=False)
    mask5 = domain["mask"]
    h5 = float(domain["dx"])
    runner = run_fmm if args.solver == "fmm" else run_wave

    runs = []
    for factor in factors:
        mask = coarsen_mask(mask5, factor)
        h = h5 * factor
        print(f"[{args.solver}] dx={h:g} m shape={mask.shape} "
              f"cells={mask.size:.3g}", flush=True)
        rec = runner(mask, h, args.repeats)
        steady = rec["times_s"][1:]
        rec.update({
            "dx_m": h,
            "shape": list(mask.shape),
            "n_cells": int(mask.size),
            "solid_fraction": float(mask.mean()),
            "first_call_s": rec["times_s"][0],
            "steady_median_s": float(np.median(steady)),
        })
        runs.append(rec)
        print(f"  first={rec['first_call_s']:.3f}s "
              f"steady={rec['steady_median_s']:.3f}s "
              f"finite={rec['finite_fraction']:.3f}", flush=True)

    n = np.log([rec["n_cells"] for rec in runs])
    t = np.log([rec["steady_median_s"] for rec in runs])
    slope = float(np.polyfit(n, t, 1)[0]) if len(runs) > 1 else float("nan")
    pair_slopes = [
        {
            "from_dx_m": a["dx_m"], "to_dx_m": b["dx_m"],
            "slope": float((math.log(b["steady_median_s"])
                            - math.log(a["steady_median_s"]))
                           / (math.log(b["n_cells"]) - math.log(a["n_cells"]))),
        }
        for a, b in zip(runs, runs[1:])
    ]
    result = {
        "experiment": "E2.3 FMM vs wave single-GPU runtime scaling",
        "solver": args.solver,
        "source_station": SOURCE,
        "t_end_s": T_END_S if args.solver == "wave" else None,
        "repeats": args.repeats,
        "gpu": gpu_name(),
        "runs": runs,
        "global_loglog_slope": slope,
        "pairwise_loglog_slopes": pair_slopes,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=False,
        ).stdout.strip(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"results_{args.solver}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[done] slope={slope:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
