"""E8.2 — analytic validation under a genuinely time-varying velocity.

An empty isotropic domain is subjected to a spatially uniform rotating flow

    q(t) = A [cos(2 pi (t-t0)/P), sin(2 pi (t-t0)/P), 0].

Because the flow is uniform in space, the exact first-arrival travel time T
to displacement d satisfies

    ||d - integral_{t0}^{t0+T} q(s) ds|| = c0 T.

The experiment advances the existing wave step kernel with q(t) and the
material-acceleration term dq/dt at every time step.  It compares the dynamic
solve with (i) the analytic root, (ii) a frozen-at-emission velocity solve,
and (iii) an ablation that omits dq/dt.  No real wind data are used.
"""
from __future__ import annotations

import json
import subprocess
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.optimize import brentq

from model import solve_all as sa
from model.wave_api import WaveConfig, build_static_fields, solve_forward


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp14_dynamic_velocity"

C0 = 15.0
QMAX = 4.5
PERIODS_S = (20.0, 40.0, 80.0)
DX = 5.0
DOMAIN_M = 600.0
RADIUS_M = 200.0
DT = 0.05
N_STEPS = 600
TAU_S = 0.4
T0_S = 1.0


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12))
def solve_time_varying(c2_xy, c2_z, one_minus_damp, free, source,
                       velocity_t, acceleration_t,
                       dx, dy, dz, n_steps, dt, min_peak_amp):
    """Run the shared step kernel with uniform q(t) and dq/dt."""
    shape = source.shape
    initial = (
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=bool),
        jnp.full(shape, jnp.inf, dtype=jnp.float32),
        source,
    )

    def step(carry, n):
        q = velocity_t[n]
        dq = acceleration_t[n]
        u, v, w = (q[axis] * free for axis in range(3))
        ax, ay, az = (dq[axis] * free for axis in range(3))
        return sa.step_kernel(
            carry, n, c2_xy, c2_z, u, v, w, ax, ay, az,
            one_minus_damp, free, dx, dy, dz,
            dt, round(T0_S / dt), round(TAU_S / dt), min_peak_amp, None,
        )

    final, _ = lax.scan(step, initial, jnp.arange(n_steps, dtype=jnp.int32))
    return final[4]


def velocity_series(period_s: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N_STEPS, dtype=np.float64) * DT
    omega = 2.0 * np.pi / period_s
    phase = omega * (t - T0_S)
    velocity = np.column_stack([
        QMAX * np.cos(phase),
        QMAX * np.sin(phase),
        np.zeros_like(phase),
    ]).astype(np.float32)
    acceleration = np.column_stack([
        -QMAX * omega * np.sin(phase),
        QMAX * omega * np.cos(phase),
        np.zeros_like(phase),
    ]).astype(np.float32)
    return velocity, acceleration


def integrated_velocity(travel_time: float, period_s: float) -> np.ndarray:
    omega = 2.0 * np.pi / period_s
    end_phase = omega * travel_time
    return np.asarray([
        QMAX * np.sin(end_phase) / omega,
        QMAX * (1.0 - np.cos(end_phase)) / omega,
        0.0,
    ])


def exact_dynamic_time(displacement: np.ndarray, period_s: float) -> float:
    def residual(travel_time: float) -> float:
        relative = displacement - integrated_velocity(travel_time, period_s)
        return float(np.linalg.norm(relative) - C0 * travel_time)

    return float(brentq(residual, 0.0, 2.0 * np.linalg.norm(displacement) / (C0 - QMAX)))


def endpoint_cells(n: int, source: tuple[int, int, int]
                   ) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    source_xyz = (np.asarray(source) + 0.5) * DX
    requested = source_xyz + np.column_stack([
        RADIUS_M * np.cos(angles),
        RADIUS_M * np.sin(angles),
        np.zeros_like(angles),
    ])
    cells = np.floor(requested / DX).astype(int)
    cells = np.clip(cells, 1, n - 2)
    xyz = (cells + 0.5) * DX
    return cells, xyz


def summarise_gap(predicted: np.ndarray, exact: np.ndarray) -> dict:
    gap = (predicted - exact) / exact
    return {
        "median_signed_gap_pct": float(100.0 * np.median(gap)),
        "median_abs_gap_pct": float(100.0 * np.median(np.abs(gap))),
        "p95_abs_gap_pct": float(100.0 * np.percentile(np.abs(gap), 95)),
        "max_abs_gap_pct": float(100.0 * np.max(np.abs(gap))),
    }


def main() -> None:
    n = int(round(DOMAIN_M / DX))
    mask = np.zeros((n, n, n), dtype=bool)
    source = (n // 2, n // 2, n // 2)
    endpoint_ijk, endpoint_xyz = endpoint_cells(n, source)
    source_xyz = (np.asarray(source) + 0.5) * DX
    displacement = endpoint_xyz - source_xyz
    distance = np.linalg.norm(displacement, axis=1)
    config_zero = WaveConfig(
        c0=C0, cz_up=C0, cz_down=C0, wind=(0.0, 0.0, 0.0),
        dt=DT, n_steps=N_STEPS,
        pulse_t0_steps=round(T0_S / DT), pulse_tau_steps=round(TAU_S / DT),
        sponge_layers=8, min_peak_amp=1e-6,
        reg_z_low=-2.0, reg_z_full=-1.0, reg_low_factor=1.0,
    )
    static_zero = build_static_fields(mask, DX, DX, DX, config_zero)
    print("[bias] zero-flow companion solve", flush=True)
    zero_arrival = solve_forward(
        mask, DX, DX, DX, source, cfg=config_zero, static=static_zero,
    )
    zero_endpoint = zero_arrival[
        endpoint_ijk[:, 0], endpoint_ijk[:, 1], endpoint_ijk[:, 2]
    ]
    bias_s = float(np.median(zero_endpoint - distance / C0))
    source_field = sa.gaussian_source(mask.shape, source, sigma=1.5)
    result = {
        "experiment": "E8.2 time-varying uniform velocity validation",
        "domain_shape": mask.shape,
        "dx_m": DX,
        "dt_s": DT,
        "n_steps": N_STEPS,
        "c0_m_s": C0,
        "qmax_m_s": QMAX,
        "source_peak_time_s": T0_S,
        "measured_zero_flow_bias_s": bias_s,
        "n_directions": len(endpoint_xyz),
        "runs": [],
    }
    arrays = {
        "angles_rad": np.arctan2(displacement[:, 1], displacement[:, 0]),
        "endpoint_xyz": endpoint_xyz.astype(np.float32),
    }

    frozen_config = WaveConfig(**{
        **config_zero.__dict__,
        "wind": (QMAX, 0.0, 0.0),
    })
    print("[frozen] q(t0) companion", flush=True)
    frozen_arrival = solve_forward(mask, DX, DX, DX, source, cfg=frozen_config)
    frozen_time = frozen_arrival[
        endpoint_ijk[:, 0], endpoint_ijk[:, 1], endpoint_ijk[:, 2]
    ] - bias_s

    for period_s in PERIODS_S:
        velocity, acceleration = velocity_series(period_s)
        exact = np.asarray([
            exact_dynamic_time(delta, period_s) for delta in displacement
        ])
        started = time.perf_counter()
        dynamic_arrival = solve_time_varying(
            static_zero.c2_xy, static_zero.c2_z,
            static_zero.one_minus_damp, static_zero.free, source_field,
            jnp.asarray(velocity), jnp.asarray(acceleration),
            DX, DX, DX, N_STEPS, DT, config_zero.min_peak_amp,
        )
        jax.block_until_ready(dynamic_arrival)
        dynamic_wall = time.perf_counter() - started
        dynamic_arrival = np.asarray(jax.device_get(dynamic_arrival))
        dynamic_time = dynamic_arrival[
            endpoint_ijk[:, 0], endpoint_ijk[:, 1], endpoint_ijk[:, 2]
        ] - bias_s

        no_acceleration_arrival = solve_time_varying(
            static_zero.c2_xy, static_zero.c2_z,
            static_zero.one_minus_damp, static_zero.free, source_field,
            jnp.asarray(velocity), jnp.zeros_like(jnp.asarray(acceleration)),
            DX, DX, DX, N_STEPS, DT, config_zero.min_peak_amp,
        )
        jax.block_until_ready(no_acceleration_arrival)
        no_acceleration_arrival = np.asarray(jax.device_get(no_acceleration_arrival))
        no_acceleration_time = no_acceleration_arrival[
            endpoint_ijk[:, 0], endpoint_ijk[:, 1], endpoint_ijk[:, 2]
        ] - bias_s

        rec = {
            "period_s": period_s,
            "dynamic_solver": summarise_gap(dynamic_time, exact),
            "frozen_at_emission": summarise_gap(frozen_time, exact),
            "omit_dqdt": summarise_gap(no_acceleration_time, exact),
            "dynamic_solve_wall_s": dynamic_wall,
            "finite_directions": int(np.isfinite(dynamic_time).sum()),
        }
        result["runs"].append(rec)
        key = f"P{period_s:g}"
        arrays[f"exact_{key}"] = exact.astype(np.float32)
        arrays[f"dynamic_{key}"] = dynamic_time.astype(np.float32)
        arrays[f"frozen_{key}"] = frozen_time.astype(np.float32)
        arrays[f"no_acceleration_{key}"] = no_acceleration_time.astype(np.float32)
        print(
            f"[P={period_s:4.0f}s] p95 |gap| dynamic="
            f"{rec['dynamic_solver']['p95_abs_gap_pct']:.2f}%, frozen="
            f"{rec['frozen_at_emission']['p95_abs_gap_pct']:.2f}%, "
            f"omit dq/dt={rec['omit_dqdt']['p95_abs_gap_pct']:.2f}%",
            flush=True,
        )

    result["git_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(OUT / "arrival_data.npz", **arrays)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
