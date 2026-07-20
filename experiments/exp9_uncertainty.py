"""E7.1 — uncertainty-aware routing and quantitative safety margins.

This experiment complements the single-obstacle benchmark without changing
the paper's method. It activates the optional

    div(D_diff grad(d psi / dt))

term already present in the manuscript and measures two distinct effects:

1. deterministic guidance-field smoothing as ``D_diff`` is increased;
2. stochastic positional uncertainty around each nominal path.

For each diffusion level we report calibrated arrival time, traced-path time,
minimum building clearance, the exact 3D 95% confidence radius, conservative
clearance margin, Monte-Carlo obstacle-collision probability with a Wilson
confidence interval, and terminal position error. Results are written to
``output/runs/exp9_uncertainty``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from model.path_trace import smooth_path, trace_path_3d
from model.uncertainty_diffusion import (
    confidence_radius,
    ensemble_collision_statistics,
    path_clearance,
    sample_trajectory_ensemble,
)
from model.wave_api import WaveConfig, solve_forward


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp9_uncertainty"

DX = 5.0
C0 = 15.0
TAU_S = 0.4
SHAPE = (60, 60, 24)  # 300 x 300 x 120 m


def cell_centres(n: int, h: float) -> np.ndarray:
    return (np.arange(n) + 0.5) * h


def make_domain() -> tuple[np.ndarray, tuple[int, int, int],
                           tuple[int, int, int], np.ndarray, np.ndarray]:
    """Match the paper's E1.4 box geometry at 5 m resolution."""
    xc = cell_centres(SHAPE[0], DX)
    yc = cell_centres(SHAPE[1], DX)
    zc = cell_centres(SHAPE[2], DX)
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    mask = ((X > 120.0) & (X < 180.0)
            & (Y > 90.0) & (Y < 210.0)
            & (Z < 60.0))

    def nearest(xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(int(round(v / DX - 0.5)) for v in xyz)

    src = nearest((60.0, 150.0, 30.0))
    dst = nearest((240.0, 150.0, 30.0))
    src_xyz = np.array([xc[src[0]], yc[src[1]], zc[src[2]]])
    dst_xyz = np.array([xc[dst[0]], yc[dst[1]], zc[dst[2]]])
    return mask, src, dst, src_xyz, dst_xyz


def make_cfg(D: float) -> WaveConfig:
    # The longest path is below 25 s; leave margin for the source wavelet.
    n_steps = 360
    return WaveConfig(
        c0=C0, cz_up=C0, cz_down=C0,
        dt=0.1, n_steps=n_steps,
        pulse_t0_steps=10, pulse_tau_steps=4,
        sponge_layers=8, sponge_strength=0.30,
        min_peak_amp=1e-6,
        reg_z_low=-2.0, reg_z_full=-1.0, reg_low_factor=1.0,
        diffusivity_m2_s=D,
    )


def calibrate_bias(T: np.ndarray, src: tuple[int, int, int]) -> float:
    """Fit T = r/c_eff + bias on a source-clear spherical shell."""
    x = cell_centres(SHAPE[0], DX)
    y = cell_centres(SHAPE[1], DX)
    z = cell_centres(SHAPE[2], DX)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    s = np.array([x[src[0]], y[src[1]], z[src[2]]])
    r = np.sqrt((X - s[0]) ** 2 + (Y - s[1]) ** 2 + (Z - s[2]) ** 2)
    use = (r >= 50.0) & (r <= 100.0) & np.isfinite(T)
    A = np.column_stack([r[use], np.ones(int(use.sum()))])
    (_, bias), *_ = np.linalg.lstsq(A, T[use], rcond=None)
    return float(bias)


def resample_path(path: np.ndarray, step_m: float = 2.5) -> np.ndarray:
    seg = np.diff(path, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    if arc[-1] <= 0.0:
        return path.copy()
    s = np.linspace(0.0, arc[-1], max(2, int(np.ceil(arc[-1] / step_m)) + 1))
    return np.column_stack([np.interp(s, arc, path[:, k]) for k in range(3)])


def run_level(mask: np.ndarray, empty: np.ndarray,
              src: tuple[int, int, int], dst: tuple[int, int, int],
              src_xyz: np.ndarray, dst_xyz: np.ndarray,
              D: float, n_samples: int, seed: int) -> tuple[dict, dict]:
    cfg = make_cfg(D)
    t0 = time.time()
    T_empty = solve_forward(empty, DX, DX, DX, src, cfg=cfg)
    bias = calibrate_bias(T_empty, src)
    T = solve_forward(mask, DX, DX, DX, src, cfg=cfg)
    wall = time.time() - t0

    raw_path, raw_status = trace_path_3d(
        T, src_xyz, dst_xyz, DX, DX, DX, gamma=0.25,
    )
    path = smooth_path(raw_path, sigma_m=7.5, step_size=2.5)
    path = resample_path(path)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    t_path = arc / C0

    # Convert centre-to-centre EDT distance to an approximate surface
    # clearance by subtracting half a voxel.
    clearance = np.maximum(
        path_clearance(path, mask, (DX, DX, DX)) - 0.5 * DX, 0.0,
    )
    D_path = np.full(len(path), D, dtype=np.float64)
    tau = D * t_path
    r95 = confidence_radius(tau, confidence=0.95, dimensions=3)
    samples = sample_trajectory_ensemble(
        path, t_path, D_path, n_samples=n_samples, seed=seed,
    )
    mc = ensemble_collision_statistics(samples, mask, (DX, DX, DX))
    terminal_error = np.linalg.norm(samples[:, -1] - path[-1], axis=1)
    terminal_error_p95 = float(np.percentile(terminal_error, 95))

    rec = {
        "D_diff_m2_s": D,
        "diffusion_number": 2.0 * D * cfg.dt * 3.0 / DX ** 2,
        "trace_status": raw_status,
        "n_path_points": int(len(path)),
        "arrival_raw_s": float(T[dst]),
        "arrival_bias_s": bias,
        "arrival_calibrated_s": float(T[dst] - bias),
        "path_time_nominal_s": float(t_path[-1]),
        "path_length_m": float(arc[-1]),
        "min_clearance_m": float(clearance.min()),
        "mean_clearance_m": float(clearance.mean()),
        "final_r95_m": float(r95[-1]),
        "terminal_error_mean_m": float(terminal_error.mean()),
        "terminal_error_p95_m": terminal_error_p95,
        "terminal_r95_rel_error": float(
            0.0 if r95[-1] == 0.0
            else (terminal_error_p95 - r95[-1]) / r95[-1]
        ),
        "min_clearance_margin95_m": float(np.min(clearance - r95)),
        "fraction_path_outside_95_tube": float(np.mean(clearance < r95)),
        "wall_s_two_solves": wall,
        **mc,
    }
    arrays = {
        "T": T.astype(np.float32),
        "path": path.astype(np.float32),
        "arc": arc.astype(np.float32),
        "time": t_path.astype(np.float32),
        "clearance": clearance.astype(np.float32),
        "r95": r95.astype(np.float32),
    }
    return rec, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Run D={0,5} with 250 Monte-Carlo paths")
    ap.add_argument("--samples", type=int, default=4000)
    args = ap.parse_args()

    levels = [0.0, 5.0] if args.quick else [0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
    n_samples = 250 if args.quick else args.samples
    mask, src, dst, src_xyz, dst_xyz = make_domain()
    empty = np.zeros_like(mask)
    OUT.mkdir(parents=True, exist_ok=True)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    results = {
        "experiment": "E7.1 uncertainty-aware obstacle clearance",
        "git_sha": sha,
        "quick": args.quick,
        "n_samples": n_samples,
        "geometry": {
            "shape": SHAPE,
            "spacing_m": [DX, DX, DX],
            "box_m": {"x": [120, 180], "y": [90, 210], "z": [0, 60]},
            "source_xyz_m": src_xyz.tolist(),
            "destination_xyz_m": dst_xyz.tolist(),
        },
        "runs": [],
    }
    npz = {"mask": mask, "src_xyz": src_xyz, "dst_xyz": dst_xyz}

    for i, D in enumerate(levels):
        print(f"[D={D:g} m^2/s] solving and sampling...", flush=True)
        rec, arrays = run_level(
            mask, empty, src, dst, src_xyz, dst_xyz,
            D=D, n_samples=n_samples, seed=1701 + i,
        )
        results["runs"].append(rec)
        tag = str(D).replace(".", "p")
        for key, value in arrays.items():
            npz[f"{key}_D{tag}"] = value
        print(
            f"  path={rec['path_time_nominal_s']:.2f}s  "
            f"clearance={rec['min_clearance_m']:.2f}m  "
            f"r95(final)={rec['final_r95_m']:.2f}m  "
            f"P(collision)={rec['collision_probability']:.2%} "
            f"[{rec['collision_ci_low']:.2%}, {rec['collision_ci_high']:.2%}]",
            flush=True,
        )

    suffix = "_quick" if args.quick else ""
    (OUT / f"results{suffix}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(OUT / f"exp9_data{suffix}.npz", **npz)
    print(f"[done] results written to {OUT}")


if __name__ == "__main__":
    main()
