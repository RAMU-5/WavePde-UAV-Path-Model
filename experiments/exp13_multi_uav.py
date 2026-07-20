"""E8.1 — concurrent multi-UAV flight and delay-based deconfliction.

Reciprocal station pairs on the 50 m Nanshan geometry create deterministic
head-on conflicts along real obstacle-aware routes.  The same nominal paths
are flown by the vectorised point-mass plant and tube controller under two
launch policies: all missions launch at t=0, or the existing delay-only CBS
reservation layer assigns conflict-free launch times.

The experiment is deliberately zero-wind so that it isolates inter-UAV
coordination.  It reports actual (not merely reserved) minimum separation,
pair-seconds below the separation threshold, building violations, tracking
error, accepted missions and delay.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

from aura.cbs.solver import MissionRequest, resolve_delays
from aura.geom.grid import SimGrid
from aura.mpc.tube_mpc import TubeMPC, make_reference_from_db
from aura.schemas import NominalTrajectory
from aura.sim.plant import FleetState, IN_FLIGHT, integrate
from experiments.exp1_dijkstra_gt import (
    C0,
    build_graph,
    metric_time,
    nudge_stations,
    reconstruct,
    reg_of_z,
    straighten,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "runs" / "exp13_multi_uav"
DOMAIN = ROOT / "input" / "domain.npz"
FLEET_SIZES = (10, 20, 40)
TARGET_ROUTE_M = 1800.0
TRAJECTORY_DT_S = 1.0
SIM_DT_S = 0.1
SEPARATION_M = 50.0
TUBE_RADIUS_M = 20.0


def select_disjoint_pairs(station_xyz: np.ndarray, n_pairs: int) -> list[tuple[int, int]]:
    """Greedily choose disjoint station pairs near the target distance."""
    xy = station_xyz[:, :2]
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    candidates = []
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if 1000.0 <= d[i, j] <= 2600.0:
                candidates.append((abs(d[i, j] - TARGET_ROUTE_M), i, j))
    candidates.sort()
    used: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.update((i, j))
        if len(pairs) == n_pairs:
            break
    if len(pairs) != n_pairs:
        raise RuntimeError(f"could select only {len(pairs)} disjoint pairs")
    return pairs


def time_resample(path: np.ndarray, dt_s: float = TRAJECTORY_DT_S
                  ) -> tuple[np.ndarray, float]:
    """Resample a feasible polyline at uniform continuous-metric time."""
    segment = np.diff(path, axis=0)
    midpoint_z = 0.5 * (path[:-1, 2] + path[1:, 2])
    segment_time = metric_time(
        segment[:, 0], segment[:, 1], segment[:, 2], C0 * reg_of_z(midpoint_z),
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_time)])
    duration = float(cumulative[-1])
    sample_t = np.arange(0.0, duration, dt_s)
    if sample_t.size == 0 or sample_t[-1] < duration:
        sample_t = np.append(sample_t, duration)
    sampled = np.column_stack([
        np.interp(sample_t, cumulative, path[:, axis]) for axis in range(3)
    ])
    return sampled, duration


def build_nominals(mask: np.ndarray, station_xyz: np.ndarray,
                   pairs: list[tuple[int, int]], dx: float, dy: float, dz: float
                   ) -> list[NominalTrajectory]:
    """Build one route and its exact reverse for every reciprocal pair."""
    graph = build_graph(mask, dx, dy, dz)
    ny, nz = mask.shape[1:]
    nominals: list[NominalTrajectory] = []
    for pair_id, (source, destination) in enumerate(pairs):
        source_ijk = np.floor(station_xyz[source] / np.array([dx, dy, dz])).astype(int)
        destination_ijk = np.floor(
            station_xyz[destination] / np.array([dx, dy, dz])
        ).astype(int)
        source_node = int((source_ijk[0] * ny + source_ijk[1]) * nz + source_ijk[2])
        destination_node = int(
            (destination_ijk[0] * ny + destination_ijk[1]) * nz + destination_ijk[2]
        )
        _, predecessors = dijkstra(
            graph, indices=source_node, directed=True, return_predecessors=True,
        )
        nodes = reconstruct(predecessors, source_node, destination_node)
        if nodes is None:
            raise RuntimeError(f"no route for station pair {source}->{destination}")
        nodes = np.asarray(nodes, dtype=np.int64)
        ijk = np.column_stack([
            nodes // (ny * nz),
            (nodes // nz) % ny,
            nodes % nz,
        ])
        path = (ijk + 0.5) * np.array([dx, dy, dz])
        path, _ = straighten(path, mask, dx, dy, dz)
        for mission_id, (src, dst, oriented) in enumerate((
            (source, destination, path),
            (destination, source, path[::-1].copy()),
        ), start=2 * pair_id):
            sampled, _ = time_resample(oriented)
            nominals.append(NominalTrajectory(
                src=src,
                dst=dst,
                xyz=sampled.tolist(),
                dt_s=TRAJECTORY_DT_S,
                metric_version=0,
            ))
        print(
            f"[pair {pair_id:02d}] {source}<->{destination}; "
            f"{len(path)} vertices, {len(nominals[-2].xyz)} samples",
            flush=True,
        )
    return nominals


def make_fleet(nominals: list[NominalTrajectory]) -> FleetState:
    starts = np.asarray([trajectory.xyz[0] for trajectory in nominals], dtype=np.float64)
    n = len(nominals)
    return FleetState(
        pos=starts.copy(),
        vel=np.zeros((n, 3), dtype=np.float64),
        home_station=np.asarray([trajectory.src for trajectory in nominals], dtype=np.int32),
        state_code=np.full(n, IN_FLIGHT, dtype=np.int32),
        soc=np.ones(n, dtype=np.float32),
        mission_id=np.arange(n, dtype=np.int64),
        target=starts.copy(),
        class_id=np.zeros(n, dtype=np.int8),
    )


def simulate(nominals: list[NominalTrajectory], launches: np.ndarray,
             mask: np.ndarray, grid: SimGrid) -> tuple[dict, np.ndarray]:
    """Fly every mission and audit separation/building/tracking metrics."""
    fleet = make_fleet(nominals)
    controller = TubeMPC(
        mass_kg=1.5, drag_gamma=0.6, u_max_n=25.0, v_max_mps=18.0,
        dt_s=SIM_DT_S, q_pos=4.0, r_vel=2.0, backend="pd",
    )
    durations = np.asarray([
        (len(trajectory.xyz) - 1) * trajectory.dt_s for trajectory in nominals
    ])
    for mission_id, trajectory in enumerate(nominals):
        controller.acquire(make_reference_from_db(
            mission_id, trajectory, float(launches[mission_id]), TUBE_RADIUS_M,
        ))
    horizon = float(np.max(launches + durations) + 5.0)
    n_steps = int(np.ceil(horizon / SIM_DT_S)) + 1
    traces = np.full((n_steps, len(nominals), 3), np.nan, dtype=np.float32)
    min_separation = np.inf
    conflict_pair_seconds = 0.0
    conflict_pairs: set[tuple[int, int]] = set()
    tracking_error = []
    building_samples = 0
    active_samples = 0

    for step in range(n_steps):
        now = step * SIM_DT_S
        active = (now >= launches) & (now <= launches + durations)
        wind = np.zeros_like(fleet.pos)
        force = controller.step(fleet, now, wind_uvw=wind)
        force[~active] = 0.0
        integrate(
            fleet, force, wind, dt=SIM_DT_S, mass=1.5, gamma=0.6,
            v_max=18.0,
            bounds_xy=((0.0, 0.0), grid.size_xy),
            bounds_z=(0.0, grid.z_max),
        )
        traces[step, active] = fleet.pos[active]
        active_ids = np.where(active)[0]
        for uid in active_ids:
            reference = controller.refs[uid]
            target, _ = controller.reference_at(reference, now)
            tracking_error.append(float(np.linalg.norm(fleet.pos[uid] - target)))
        if len(active_ids) >= 2:
            positions = fleet.pos[active_ids]
            distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
            iu, ju = np.triu_indices(len(active_ids), 1)
            pair_distance = distance[iu, ju]
            min_separation = min(min_separation, float(pair_distance.min()))
            for local_index in np.where(pair_distance < SEPARATION_M)[0]:
                a = int(active_ids[iu[local_index]])
                b = int(active_ids[ju[local_index]])
                conflict_pairs.add((min(a, b), max(a, b)))
                conflict_pair_seconds += SIM_DT_S
        if len(active_ids):
            ijk = grid.world_to_index(fleet.pos[active_ids])
            building_samples += int(mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]].sum())
            active_samples += len(active_ids)

    errors = np.asarray(tracking_error)
    endpoint_error = []
    for uid, trajectory in enumerate(nominals):
        end_step = min(n_steps - 1, int(round((launches[uid] + durations[uid]) / SIM_DT_S)))
        endpoint_error.append(float(np.linalg.norm(
            traces[end_step, uid] - np.asarray(trajectory.xyz[-1]),
        )))
    metrics = {
        "accepted_missions": len(nominals),
        "mean_launch_delay_s": float(np.mean(launches)),
        "p95_launch_delay_s": float(np.percentile(launches, 95)),
        "makespan_s": float(np.max(launches + durations)),
        "minimum_active_separation_m": float(min_separation),
        "conflicting_uav_pairs": len(conflict_pairs),
        "conflict_pair_seconds": float(conflict_pair_seconds),
        "building_violation_fraction": building_samples / max(active_samples, 1),
        "tracking_error_median_m": float(np.median(errors)),
        "tracking_error_p95_m": float(np.percentile(errors, 95)),
        "endpoint_error_p95_m": float(np.percentile(endpoint_error, 95)),
    }
    return metrics, traces


def main() -> None:
    domain = np.load(DOMAIN, allow_pickle=False)
    mask = domain["mask"]
    dx, dy, dz = (float(domain[key]) for key in ("dx", "dy", "dz"))
    ix, iy, iz = nudge_stations(mask, domain["stations_xy_m"], dx, dy, dz)
    station_xyz = np.column_stack([
        (ix + 0.5) * dx,
        (iy + 0.5) * dy,
        (iz + 0.5) * dz,
    ])
    grid = SimGrid(
        origin_xy=(0.0, 0.0), size_xy=(mask.shape[0] * dx, mask.shape[1] * dy),
        dx=dx, dy=dy, z_min=0.0, z_max=mask.shape[2] * dz, dz=dz,
    )
    pairs = select_disjoint_pairs(station_xyz, max(FLEET_SIZES) // 2)
    nominals_all = build_nominals(mask, station_xyz, pairs, dx, dy, dz)
    result = {
        "experiment": "E8.1 Nanshan concurrent multi-UAV flight",
        "domain": "domain.npz",
        "shape": mask.shape,
        "spacing_m": [dx, dy, dz],
        "zero_wind": True,
        "separation_threshold_m": SEPARATION_M,
        "tube_radius_m": TUBE_RADIUS_M,
        "simulation_dt_s": SIM_DT_S,
        "runs": [],
    }
    figure_arrays = {}

    for fleet_size in FLEET_SIZES:
        nominals = nominals_all[:fleet_size]
        requests = [MissionRequest(
            mission_id=mission_id,
            nominal=trajectory,
            eps_m=TUBE_RADIUS_M,
            requested_launch_s=0.0,
        ) for mission_id, trajectory in enumerate(nominals)]
        granted = resolve_delays(requests, grid, bin_s=1.0, max_delay_s=600.0)
        coordinated_launches = np.asarray([
            granted.get(mission_id, np.nan) for mission_id in range(fleet_size)
        ])
        accepted = np.isfinite(coordinated_launches)
        if not accepted.all():
            nominals_coord = [trajectory for trajectory, keep in zip(nominals, accepted) if keep]
            coordinated_launches = coordinated_launches[accepted]
        else:
            nominals_coord = nominals

        uncoordinated, traces_uncoordinated = simulate(
            nominals, np.zeros(fleet_size), mask, grid,
        )
        coordinated, traces_coordinated = simulate(
            nominals_coord, coordinated_launches, mask, grid,
        )
        coordinated["rejected_missions"] = int(fleet_size - accepted.sum())
        rec = {
            "fleet_size": fleet_size,
            "station_pairs": pairs[:fleet_size // 2],
            "uncoordinated": uncoordinated,
            "coordinated": coordinated,
        }
        result["runs"].append(rec)
        print(
            f"[N={fleet_size}] conflicts {uncoordinated['conflicting_uav_pairs']} -> "
            f"{coordinated['conflicting_uav_pairs']}; min separation "
            f"{uncoordinated['minimum_active_separation_m']:.1f} -> "
            f"{coordinated['minimum_active_separation_m']:.1f} m; "
            f"mean delay {coordinated['mean_launch_delay_s']:.1f} s",
            flush=True,
        )
        if fleet_size == 20:
            paths = [np.asarray(trajectory.xyz, dtype=np.float32) for trajectory in nominals]
            offsets = np.concatenate([[0], np.cumsum([len(path) for path in paths])])
            figure_arrays = {
                "paths": np.concatenate(paths),
                "path_offsets": offsets,
                "uncoordinated_launches": np.zeros(fleet_size),
                "coordinated_launches": coordinated_launches,
                "traces_uncoordinated": traces_uncoordinated,
                "traces_coordinated": traces_coordinated,
                "building_footprint": np.any(mask, axis=2),
            }

    result["git_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(OUT / "figure_data.npz", **figure_arrays)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
