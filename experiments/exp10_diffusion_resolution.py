"""E7.2 — grid sensitivity of constant uncertainty diffusion.

The experiment repeats the E1.4 single-obstacle benchmark at 10, 5 and
2.5 m resolution, without prescribed wind.  For every physical diffusion
coefficient it performs an empty-domain companion solve and an obstacle
solve.  The empty solve measures effective propagation speed and source-peak
bias; the obstacle solve measures traced-path time and geometric clearance.

This separates a physical ``D_diff`` response from grid-scale smoothing and
tests whether the clearance trend reported by E7.1 survives refinement.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from model.path_trace import smooth_path, trace_path_3d
from model.uncertainty_diffusion import path_clearance
from model.wave_api import WaveConfig, solve_forward


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp10_diffusion_resolution"

C0 = 15.0
TAU_S = 0.4
DOMAIN_M = (300.0, 300.0, 120.0)
DIFFUSIVITIES = (0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0)


def centres(n: int, h: float) -> np.ndarray:
    return (np.arange(n) + 0.5) * h


def make_domain(dx: float) -> tuple[np.ndarray, tuple[int, int, int],
                                      tuple[int, int, int], np.ndarray,
                                      np.ndarray]:
    shape = tuple(int(round(length / dx)) for length in DOMAIN_M)
    x, y, z = (centres(n, dx) for n in shape)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    mask = ((X > 120.0) & (X < 180.0)
            & (Y > 90.0) & (Y < 210.0)
            & (Z < 60.0))

    def nearest(xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(int(round(v / dx - 0.5)) for v in xyz)

    src = nearest((60.0, 150.0, 30.0))
    dst = nearest((240.0, 150.0, 30.0))
    src_xyz = np.array([x[src[0]], y[src[1]], z[src[2]]])
    dst_xyz = np.array([x[dst[0]], y[dst[1]], z[dst[2]]])
    return mask, src, dst, src_xyz, dst_xyz


def make_cfg(dx: float, diffusivity: float) -> WaveConfig:
    dt = 0.02 * dx
    t_end = 21.0
    return WaveConfig(
        c0=C0, cz_up=C0, cz_down=C0,
        dt=dt, n_steps=math.ceil(t_end / dt),
        pulse_t0_steps=max(2, round(2.5 * TAU_S / dt)),
        pulse_tau_steps=max(1, round(TAU_S / dt)),
        sponge_layers=max(2, round(40.0 / dx)),
        sponge_strength=0.30,
        min_peak_amp=1e-6 * (dx / 5.0) ** 3,
        diffusivity_m2_s=diffusivity,
        reg_z_low=-2.0, reg_z_full=-1.0, reg_low_factor=1.0,
    )


def calibrate_empty(T: np.ndarray, src_xyz: np.ndarray,
                    dx: float) -> tuple[float, float, float]:
    x, y, z = (centres(n, dx) for n in T.shape)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt((X - src_xyz[0]) ** 2 + (Y - src_xyz[1]) ** 2
                + (Z - src_xyz[2]) ** 2)
    # Fit a source-height annulus only.  The source is intentionally low in
    # the obstacle benchmark, so a spherical shell would mix the direct wave
    # with the ground reflection and bias the inferred propagation speed.
    lateral_margin = 40.0
    use = ((r >= 50.0) & (r <= 100.0)
           & (np.abs(Z - src_xyz[2]) <= 0.51 * dx)
           & (X >= lateral_margin) & (X <= DOMAIN_M[0] - lateral_margin)
           & (Y >= lateral_margin) & (Y <= DOMAIN_M[1] - lateral_margin)
           & np.isfinite(T))
    A = np.column_stack([r[use], np.ones(int(use.sum()))])
    (slope, bias), *_ = np.linalg.lstsq(A, T[use], rcond=None)
    residual = T[use] - (slope * r[use] + bias)
    return float(1.0 / slope), float(bias), float(np.sqrt(np.mean(residual**2)))


def resample(path: np.ndarray, step_m: float) -> np.ndarray:
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    if arc[-1] <= 0.0:
        return path.copy()
    s = np.linspace(0.0, arc[-1], max(2, math.ceil(arc[-1] / step_m) + 1))
    return np.column_stack([np.interp(s, arc, path[:, k]) for k in range(3)])


def run_case(dx: float, diffusivity: float) -> tuple[dict, np.ndarray]:
    mask, src, dst, src_xyz, dst_xyz = make_domain(dx)
    cfg = make_cfg(dx, diffusivity)
    empty = np.zeros_like(mask)

    started = time.perf_counter()
    T_empty = solve_forward(empty, dx, dx, dx, src, cfg=cfg)
    c_eff, bias, fit_rms = calibrate_empty(T_empty, src_xyz, dx)
    T = solve_forward(mask, dx, dx, dx, src, cfg=cfg)
    wall = time.perf_counter() - started

    raw, status = trace_path_3d(
        T, src_xyz, dst_xyz, dx, dx, dx, gamma=0.25,
    )
    path = smooth_path(raw, sigma_m=7.5, step_size=max(dx / 2.0, 1.25))
    path = resample(path, step_m=max(dx / 2.0, 1.25))
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    clearance = np.maximum(
        path_clearance(path, mask, (dx, dx, dx)) - 0.5 * dx, 0.0,
    )
    diffusion_number = 2.0 * diffusivity * cfg.dt * 3.0 / dx**2
    rec = {
        "dx_m": dx,
        "D_diff_m2_s": diffusivity,
        "dt_s": cfg.dt,
        "n_steps": cfg.n_steps,
        "diffusion_number": diffusion_number,
        "trace_status": status,
        "effective_speed_m_s": c_eff,
        "effective_speed_rel_error": (c_eff - C0) / C0,
        "arrival_bias_s": bias,
        "empty_fit_rms_s": fit_rms,
        "arrival_calibrated_s": float(T[dst] - bias),
        "path_length_m": length,
        "path_time_s": length / C0,
        "min_clearance_m": float(clearance.min()),
        "mean_clearance_m": float(clearance.mean()),
        "max_path_z_m": float(path[:, 2].max()),
        "wall_s_two_solves": wall,
    }
    return rec, path.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    dxs = (10.0, 5.0) if args.quick else (10.0, 5.0, 2.5)
    diffusivities = (0.0, 5.0, 20.0) if args.quick else DIFFUSIVITIES

    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    result = {
        "experiment": "E7.2 diffusion grid sensitivity",
        "git_sha": sha,
        "quick": args.quick,
        "domain_m": DOMAIN_M,
        "zero_prescribed_wind": True,
        "runs": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for dx in dxs:
        for diffusivity in diffusivities:
            print(f"[dx={dx:g} m, D={diffusivity:g} m^2/s]", flush=True)
            rec, path = run_case(dx, diffusivity)
            result["runs"].append(rec)
            tag = f"dx{dx:g}_D{diffusivity:g}".replace(".", "p")
            arrays[f"path_{tag}"] = path
            print(
                f"  c_eff={rec['effective_speed_m_s']:.3f} m/s  "
                f"time={rec['path_time_s']:.3f} s  "
                f"clearance={rec['min_clearance_m']:.3f} m  "
                f"status={rec['trace_status']}", flush=True,
            )

    suffix = "_quick" if args.quick else ""
    (OUT / f"results{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(OUT / f"paths{suffix}.npz", **arrays)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
