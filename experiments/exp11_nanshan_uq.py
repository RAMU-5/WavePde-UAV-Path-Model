"""E7.3 — zero-wind uncertainty qualification on Nanshan geometry.

One wave solve from station 50 serves three destinations spanning distinct
directions and 1.0--2.5 km Euclidean separation.  Constant ``D_diff`` is
swept without a prescribed wind field.  For each nominal path, Brownian
position ensembles quantify collision probability against the real 5 m
building mask and the analytical 3D confidence radius supplies a conservative
clearance margin.

No full arrival-time fields are written: only paths, metrics and a building
slice needed by the paper figure are retained.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from experiments.exp1_dijkstra_gt import metric_time, reg_of_z
from experiments.exp7_efficiency import production_station_cells
from model.uncertainty_diffusion import (
    confidence_radius,
    ensemble_collision_statistics,
    sample_trajectory_ensemble,
)
from model.wave_api import WaveConfig, solve_forward


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp11_nanshan_uq"
DOMAIN = ROOT / "input" / "domain_5m.npz"

SOURCE = 50
DESTINATIONS = (17, 19, 20)
DIFFUSIVITIES = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5)


def make_cfg(diffusivity: float) -> WaveConfig:
    return WaveConfig(
        dt=0.1,
        n_steps=3400,
        pulse_t0_steps=10,
        pulse_tau_steps=4,
        sponge_layers=8,
        min_peak_amp=1e-6 * (5.0 / 50.0) ** 3,
        diffusivity_m2_s=diffusivity,
    )


def smooth_arrival_and_gradients(T: np.ndarray, spacing: tuple[float, float, float]
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised smoothing from E4.2, followed by one shared gradient."""
    finite = np.isfinite(T)
    if not finite.any():
        raise RuntimeError("arrival-time field has no finite cells")
    values = np.where(finite, T, 0.0).astype(np.float32)
    weights = finite.astype(np.float32)
    numerator = gaussian_filter(values, sigma=2.0)
    denominator = gaussian_filter(weights, sigma=2.0)
    T_smooth = np.where(
        finite,
        numerator / np.maximum(denominator, 1e-9),
        float(T[finite].max()) + 1e6,
    ).astype(np.float32)
    del values, weights, numerator, denominator, finite
    return tuple(np.gradient(T_smooth, *spacing))


def trace_gradient(grads: tuple[np.ndarray, np.ndarray, np.ndarray],
                   source_xyz: np.ndarray, destination_xyz: np.ndarray,
                   spacing: tuple[float, float, float], gamma: float = 0.3,
                   max_steps: int = 20000) -> tuple[np.ndarray, str]:
    """Trace ``-grad(T)`` with cell-centred coordinate interpolation."""
    step = gamma * min(spacing)
    position = destination_xyz.astype(np.float64).copy()
    source = source_xyz.astype(np.float64)
    path = [position.copy()]
    best_distance = float(np.linalg.norm(position - source))
    best_index = 0
    stale = 0
    for _ in range(max_steps):
        coords = np.array([
            [position[0] / spacing[0] - 0.5],
            [position[1] / spacing[1] - 0.5],
            [position[2] / spacing[2] - 0.5],
        ])
        gradient = np.array([
            float(map_coordinates(g, coords, order=1, mode="nearest")[0])
            for g in grads
        ])
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-12:
            break
        position -= step * gradient / norm
        path.append(position.copy())
        distance = float(np.linalg.norm(position - source))
        if distance < best_distance - 0.5 * step:
            best_distance = distance
            best_index = len(path) - 1
            stale = 0
        else:
            stale += 1
        if distance < 2.0 * max(spacing):
            path.append(source.copy())
            return np.asarray(path[::-1]), "reached"
        if stale > 300:
            path = path[:best_index + 1]
            path.append(source.copy())
            return np.asarray(path[::-1]), f"truncated_{best_distance:.0f}m"
    return np.asarray(path[::-1]), "max_steps"


def resample(path: np.ndarray, step_m: float = 5.0) -> np.ndarray:
    length = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(length)])
    if arc[-1] <= 0.0:
        return path.copy()
    s = np.linspace(0.0, arc[-1], max(2, int(np.ceil(arc[-1] / step_m)) + 1))
    return np.column_stack([np.interp(s, arc, path[:, k]) for k in range(3)])


def cumulative_path_time(path: np.ndarray) -> np.ndarray:
    segment = np.diff(path, axis=0)
    midpoint_z = 0.5 * (path[:-1, 2] + path[1:, 2])
    horizontal_speed = 15.0 * reg_of_z(midpoint_z)
    dt = metric_time(
        segment[:, 0], segment[:, 1], segment[:, 2], horizontal_speed,
    )
    return np.concatenate([[0.0], np.cumsum(dt)])


def clearance_on_grid(path: np.ndarray, clearance_grid: np.ndarray,
                      spacing: tuple[float, float, float]) -> np.ndarray:
    coords = np.stack([
        path[:, axis] / spacing[axis] - 0.5 for axis in range(3)
    ])
    centre_distance = map_coordinates(
        clearance_grid, coords, order=1, mode="constant", cval=0.0,
    )
    return np.maximum(centre_distance - 0.5 * min(spacing), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()
    diffusivities = (0.0, 0.1) if args.quick else DIFFUSIVITIES
    n_samples = 250 if args.quick else args.samples

    domain = np.load(DOMAIN, allow_pickle=False)
    mask = domain["mask"]
    spacing = tuple(float(domain[key]) for key in ("dx", "dy", "dz"))
    ix, iy, iz = production_station_cells(
        mask, domain["stations_xy_m"], *spacing,
    )
    station_xyz = np.column_stack([
        (ix + 0.5) * spacing[0],
        (iy + 0.5) * spacing[1],
        (iz + 0.5) * spacing[2],
    ])
    source_cell = (int(ix[SOURCE]), int(iy[SOURCE]), int(iz[SOURCE]))

    print("[clearance] computing Nanshan distance transform", flush=True)
    clearance_grid = distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    result = {
        "experiment": "E7.3 Nanshan zero-wind uncertainty qualification",
        "git_sha": sha,
        "quick": args.quick,
        "n_samples": n_samples,
        "source_station": SOURCE,
        "destination_stations": DESTINATIONS,
        "spacing_m": spacing,
        "shape": mask.shape,
        "zero_prescribed_wind": True,
        "runs": [],
    }
    arrays: dict[str, np.ndarray] = {
        "building_footprint": np.any(mask, axis=2),
        "station_xyz": station_xyz.astype(np.float32),
    }

    for diffusivity in diffusivities:
        cfg = make_cfg(diffusivity)
        print(f"[D={diffusivity:g} m^2/s] solving Nanshan", flush=True)
        started = time.perf_counter()
        T = solve_forward(mask, *spacing, source_cell, cfg=cfg)
        solve_wall = time.perf_counter() - started
        print(f"  solve={solve_wall:.1f}s; smoothing and tracing", flush=True)
        grads = smooth_arrival_and_gradients(T, spacing)

        for destination in DESTINATIONS:
            path, status = trace_gradient(
                grads, station_xyz[SOURCE], station_xyz[destination], spacing,
            )
            path = resample(path)
            path_time = cumulative_path_time(path)
            clearance = clearance_on_grid(path, clearance_grid, spacing)
            tau = diffusivity * path_time
            r95 = confidence_radius(tau)
            samples = sample_trajectory_ensemble(
                path, path_time, np.full(len(path), diffusivity),
                n_samples=n_samples, seed=9100 + destination,
            )
            mc = ensemble_collision_statistics(samples, mask, spacing)
            terminal_error = np.linalg.norm(samples[:, -1] - path[-1], axis=1)
            rec = {
                "D_diff_m2_s": diffusivity,
                "source_station": SOURCE,
                "destination_station": destination,
                "euclidean_distance_m": float(np.linalg.norm(
                    station_xyz[destination] - station_xyz[SOURCE]
                )),
                "trace_status": status,
                "arrival_raw_s": float(T[ix[destination], iy[destination], iz[destination]]),
                "path_length_m": float(np.linalg.norm(
                    np.diff(path, axis=0), axis=1,
                ).sum()),
                "path_time_s": float(path_time[-1]),
                "min_clearance_m": float(clearance.min()),
                "mean_clearance_m": float(clearance.mean()),
                "final_r95_m": float(r95[-1]),
                "min_clearance_margin95_m": float(np.min(clearance - r95)),
                "fraction_path_outside_95_tube": float(np.mean(clearance < r95)),
                "terminal_error_p95_m": float(np.percentile(terminal_error, 95)),
                "solve_wall_s": solve_wall,
                **mc,
            }
            result["runs"].append(rec)
            tag = f"D{diffusivity:g}_dst{destination}".replace(".", "p")
            arrays[f"path_{tag}"] = path.astype(np.float32)
            arrays[f"time_{tag}"] = path_time.astype(np.float32)
            arrays[f"clearance_{tag}"] = clearance.astype(np.float32)
            arrays[f"r95_{tag}"] = r95.astype(np.float32)
            print(
                f"  dst={destination}: {status}, time={rec['path_time_s']:.1f}s, "
                f"clearance={rec['min_clearance_m']:.1f}m, "
                f"P(building)={rec['obstacle_collision_probability']:.1%}, "
                f"P(exit)={rec['domain_exit_probability']:.1%}", flush=True,
            )
        del T, grads

    suffix = "_quick" if args.quick else ""
    (OUT / f"results{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(OUT / f"paths{suffix}.npz", **arrays)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
