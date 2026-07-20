"""E2.2 — AGD CUDA Fast Marching baseline on the Nanshan geometry.

The experiment solves the same zero-prescribed-wind, asymmetric vertical
navigation metric used by E2.1.  The control-velocity ellipsoid has horizontal
semi-axis ``15 * reg(z)`` and vertical semi-axis 5 m/s, with a +1 m/s vertical
drift (6 m/s climb, 4 m/s descent).  It is converted exactly to a Randers
metric and solved with AGD's CUDA ``Rander3`` implementation.

By default the full 118-source matrix is evaluated on the 50 m Nanshan grid
and compared pairwise with the Dijkstra + metric-aware string-pulling upper
bound from E2.1.  The 5 m grid can be used for a selected-source performance
check, but results from different resolutions are never pooled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from aura.cost.metric import RanderField
from aura.fmm.agd_solver import solve_single_source
from aura.geom.grid import SimGrid
from experiments.exp1_dijkstra_gt import (
    A_Z,
    C0,
    W_UP,
    metric_time,
    nudge_stations,
    reg_of_z,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp12_fmm"


def exact_rander_field(mask: np.ndarray, dx: float, dy: float, dz: float
                       ) -> tuple[RanderField, SimGrid]:
    """Convert the E2.1 Zermelo metric to its exact Randers form."""
    nx, ny, nz = mask.shape
    z = (np.arange(nz, dtype=np.float32) + 0.5) * dz
    a_h = (C0 * reg_of_z(z)).astype(np.float32)
    q_h = 1.0 / np.square(a_h)
    q_z = np.float32(1.0 / A_Z**2)
    lam = np.float32(1.0 - (W_UP / A_Z) ** 2)

    # F(d) = sqrt(d^T m d) + omega.d, obtained exactly from
    # ||d / t - b||_Q = 1 with b=(0, 0, W_UP).
    diag_z = q_z / lam + np.square(q_z * np.float32(W_UP)) / np.square(lam)
    omega_z = -q_z * np.float32(W_UP) / lam

    diag = np.empty((nx, ny, nz, 3), dtype=np.float32)
    diag[..., 0] = (q_h / lam)[None, None, :]
    diag[..., 1] = (q_h / lam)[None, None, :]
    diag[..., 2] = diag_z
    omega = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    omega[..., 2] = omega_z
    field = RanderField(
        riemann_diag=diag,
        rander_omega=omega,
        rho=np.ones(mask.shape, dtype=np.float32),
        walls=mask,
        v_cruise=C0,
    )
    grid = SimGrid(
        origin_xy=(0.0, 0.0),
        size_xy=(nx * dx, ny * dy),
        dx=dx,
        dy=dy,
        z_min=0.0,
        z_max=nz * dz,
        dz=dz,
    )
    return field, grid


def check_metric_conversion() -> float:
    """Return max relative error against the E2.1 closed-form edge time."""
    rng = np.random.default_rng(1202)
    d = rng.normal(size=(1000, 3)) * np.array([80.0, 80.0, 30.0])
    a_h = rng.uniform(3.0, C0, size=len(d))
    q_h = 1.0 / np.square(a_h)
    q_z = 1.0 / A_Z**2
    lam = 1.0 - (W_UP / A_Z) ** 2
    m_xy = q_h / lam
    m_z = q_z / lam + (q_z * W_UP) ** 2 / lam**2
    omega_z = -q_z * W_UP / lam
    rander = np.sqrt(
        m_xy * (np.square(d[:, 0]) + np.square(d[:, 1]))
        + m_z * np.square(d[:, 2])
    ) + omega_z * d[:, 2]
    exact = metric_time(d[:, 0], d[:, 1], d[:, 2], a_h)
    return float(np.max(np.abs(rander - exact) / np.maximum(exact, 1e-12)))


def parse_sources(text: str | None, n: int, quick: bool) -> list[int]:
    if text:
        out = [int(value) for value in text.split(",")]
    elif quick:
        out = [0, 17, 50]
    else:
        out = list(range(n))
    if not out or min(out) < 0 or max(out) >= n:
        raise ValueError(f"source indices must lie in [0, {n - 1}]")
    return out


def comparison_stats(T_fmm: np.ndarray, sources: list[int], station_xyz: np.ndarray,
                     gt_path: Path) -> dict | None:
    if not gt_path.exists():
        return None
    gt = np.load(gt_path, allow_pickle=False)
    gt_row = {int(source): row for row, source in enumerate(gt["src_idx"])}
    fmm_values: list[float] = []
    ref_values: list[float] = []
    raw_values: list[float] = []
    for row, source in enumerate(sources):
        if source not in gt_row:
            continue
        distance = np.linalg.norm(station_xyz[:, :2] - station_xyz[source, :2], axis=1)
        valid = (
            (distance >= 700.0)
            & (distance <= 3000.0)
            & np.isfinite(T_fmm[row])
            & np.isfinite(gt["T_str"][gt_row[source]])
        )
        fmm_values.extend(T_fmm[row, valid].tolist())
        ref_values.extend(gt["T_str"][gt_row[source], valid].tolist())
        raw_values.extend(gt["T_dij"][gt_row[source], valid].tolist())
    if not fmm_values:
        return None
    fmm = np.asarray(fmm_values)
    ref = np.asarray(ref_values)
    raw = np.asarray(raw_values)
    gap = (fmm - ref) / ref
    raw_gap = (fmm - raw) / raw
    return {
        "valid_pairs": int(len(gap)),
        "median_gap_vs_string_pull_pct": float(100.0 * np.median(gap)),
        "p95_abs_gap_vs_string_pull_pct": float(100.0 * np.percentile(np.abs(gap), 95)),
        "median_gap_vs_raw_dijkstra_pct": float(100.0 * np.median(raw_gap)),
        "p95_abs_gap_vs_raw_dijkstra_pct": float(100.0 * np.percentile(np.abs(raw_gap), 95)),
        "finite_fraction": float(np.isfinite(fmm).mean()),
    }


def wave_anchor_stats(T_fmm: np.ndarray, sources: list[int],
                      domain_name: str) -> dict | None:
    """Compare selected 5 m endpoints with the existing D=0 wave solve."""
    if domain_name != "domain_5m.npz":
        return None
    path = OUT.parent / "exp11_nanshan_uq" / "results.json"
    if not path.exists():
        return None
    with path.open() as handle:
        wave = json.load(handle)
    source = int(wave["source_station"])
    if source not in sources:
        return None
    row = sources.index(source)
    pulse_bias_s = 0.945
    records = []
    for record in wave["runs"]:
        if record["D_diff_m2_s"] != 0.0:
            continue
        destination = int(record["destination_station"])
        wave_time = float(record["arrival_raw_s"] - pulse_bias_s)
        fmm_time = float(T_fmm[row, destination])
        records.append({
            "destination_station": destination,
            "fmm_arrival_s": fmm_time,
            "calibrated_wave_arrival_s": wave_time,
            "fmm_minus_wave_pct": 100.0 * (fmm_time - wave_time) / wave_time,
        })
    if not records:
        return None
    gaps = np.asarray([record["fmm_minus_wave_pct"] for record in records])
    return {
        "source_station": source,
        "pulse_bias_s": pulse_bias_s,
        "n_routes": len(records),
        "median_abs_difference_pct": float(np.median(np.abs(gaps))),
        "max_abs_difference_pct": float(np.max(np.abs(gaps))),
        "routes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain.npz")
    parser.add_argument("--sources", help="comma-separated station indices")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each source solve for timing")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    import cupy as cp

    domain_path = ROOT / "input" / args.domain
    domain = np.load(domain_path, allow_pickle=False)
    mask = domain["mask"]
    dx, dy, dz = (float(domain[key]) for key in ("dx", "dy", "dz"))
    ix, iy, iz = nudge_stations(mask, domain["stations_xy_m"], dx, dy, dz)
    station_xyz = np.column_stack([
        (ix + 0.5) * dx,
        (iy + 0.5) * dy,
        (iz + 0.5) * dz,
    ])
    sources = parse_sources(args.sources, len(station_xyz), args.quick)
    conversion_error = check_metric_conversion()
    if conversion_error > 1e-12:
        raise RuntimeError(f"Randers conversion check failed: {conversion_error:.3e}")

    print(f"[domain] {args.domain}: {mask.shape}, sources={len(sources)}", flush=True)
    print(f"[metric] closed-form conversion max rel error={conversion_error:.3e}", flush=True)
    field, grid = exact_rander_field(mask, dx, dy, dz)
    T_fmm = np.full((len(sources), len(station_xyz)), np.inf, dtype=np.float32)
    runtime_by_source: list[list[float]] = []
    pool_peak = 0
    free_min = cp.cuda.runtime.memGetInfo()[0]

    for row, source in enumerate(sources):
        source_runtimes = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            arrival = solve_single_source(
                field, grid, tuple(station_xyz[source]), backend="cuda",
            )
            source_runtimes.append(time.perf_counter() - started)
        runtime_by_source.append(source_runtimes)
        T_fmm[row] = arrival.T[ix, iy, iz]
        pool_peak = max(pool_peak, cp.get_default_memory_pool().total_bytes())
        free_min = min(free_min, cp.cuda.runtime.memGetInfo()[0])
        print(
            f"[src {source:3d}] "
            f"{np.median(source_runtimes):.3f}s median of {args.repeats}; "
            f"finite stations={np.isfinite(T_fmm[row]).sum()}/{len(station_xyz)}",
            flush=True,
        )

    gt_path = OUT.parent / "exp1_dijkstra" / f"gt_{domain_path.stem}.npz"
    comparison = comparison_stats(T_fmm, sources, station_xyz, gt_path)
    device = cp.cuda.runtime.getDeviceProperties(0)
    runtimes = np.asarray(runtime_by_source, dtype=np.float64).ravel()
    runtime_steady = runtimes[1:] if len(runtimes) > 1 else runtimes
    result = {
        "experiment": "E2.2 AGD CUDA Rander3 FMM baseline",
        "domain": args.domain,
        "shape": mask.shape,
        "spacing_m": [dx, dy, dz],
        "n_cells": int(mask.size),
        "sources": sources,
        "n_stations": int(len(station_xyz)),
        "zero_prescribed_wind": True,
        "metric": {
            "horizontal_speed_m_s": C0,
            "vertical_control_speed_m_s": A_Z,
            "vertical_drift_m_s": W_UP,
            "climb_speed_m_s": A_Z + W_UP,
            "descent_speed_m_s": A_Z - W_UP,
            "conversion_max_relative_error": conversion_error,
        },
        "repeats_per_source": args.repeats,
        "runtime_by_source_s": runtime_by_source,
        "runtime_first_s": float(runtimes[0]),
        "runtime_steady_median_s": float(np.median(runtime_steady)),
        "runtime_steady_p95_s": float(np.percentile(runtime_steady, 95)),
        "gpu": {
            "name": device["name"].decode(),
            "cupy_version": cp.__version__,
            "agd_version": version("agd"),
            "memory_pool_peak_bytes": int(pool_peak),
            "minimum_free_memory_bytes": int(free_min),
        },
        "ground_truth_file": str(gt_path.relative_to(ROOT)) if gt_path.exists() else None,
        "comparison": comparison,
        "wave_5m_anchors": wave_anchor_stats(T_fmm, sources, args.domain),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=False,
        ).stdout.strip(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = domain_path.stem + ("_quick" if args.quick else "")
    np.savez_compressed(
        OUT / f"fmm_{suffix}.npz",
        T_fmm=T_fmm,
        sources=np.asarray(sources),
        station_xyz=station_xyz.astype(np.float32),
        runtime_by_source_s=np.asarray(runtime_by_source),
    )
    with (OUT / f"results_{suffix}.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
