"""Stage 1 simulation entry point.

Wires Stage1Assets + ConstantWind -> RanderField -> all-pairs FMM ->
SimClock(plant, MPC, CBS, ALNS, rebalance) -> Mission generators -> KPI/GIF.

Usage:
    python -m aura.cli.run_stage1 --assets runs/v1 --duration 1800 --gif runs/demo.gif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from aura.assignment.alns import RejectCounters, alns_assign
from aura.cbs.solver import MissionRequest, resolve_delays
from aura.cost.metric import build_rander, patch_rho_for_zone
from aura.fmm.agd_solver import load_db as load_traj_db, solve_all_pairs
from aura.fmm.incremental import resolve_affected
from aura.geom.noflyzone import NoFlyRegistry
from aura.io.assets import load as load_assets
from aura.metrics.kpi import KpiAccumulator, write_csv
from aura.missions.hazard import schedule_to_events, step_registry
from aura.missions.order_book import OrderBook
from aura.missions.order_types import default_specs
from aura.missions.passive_sensing import PassiveSensingLogger
from aura.mpc.tube_mpc import TubeMPC, make_reference_from_db
from aura.schemas import Order
from aura.sim.plant import (
    CHARGING, IDLE, IN_FLIGHT, RETURNING,
    integrate, make_fleet,
)
from aura.sim.scheduler import SimClock
from aura.wind import ConstantWind


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--traj", default=None,
                   help="Pre-built trajectory DB dir (default: <assets>/traj). "
                        "If absent, falls back to FMM at start (slow) or --no-fmm.")
    p.add_argument("--duration", type=float, default=None,
                   help="Override sim duration_s.")
    p.add_argument("--gif", default=None)
    p.add_argument("--kpi", default=None)
    p.add_argument("--demo", action="store_true",
                   help="Shorter run + viz on for the demo target.")
    p.add_argument("--no-fmm", action="store_true",
                   help="Skip FMM all-pairs (use straight-line nominals).")
    p.add_argument("--class-split", type=int, nargs=2, default=None,
                   metavar=("N_SMALL", "N_LARGE"),
                   help="Heterogeneous fleet split (overrides yaml).")
    p.add_argument("--mpc-active-only", action="store_true", default=True,
                   help="Skip MPC for idle UAVs (24h sim accelerator).")
    return p.parse_args(argv)


def _identity_lonlat_to_xy_m(lon, lat):
    """For synthetic Stage 1 we degrade hazard scheduling to UTM-ish meters."""
    scale = 111_000.0
    x = lon * scale * np.cos(np.deg2rad(22.55))
    y = lat * scale
    return (x, y)


def main(argv=None) -> int:
    args = _parse_args(argv)
    assets = load_assets(args.assets)
    cfg = assets.config

    duration_s = args.duration if args.duration is not None else cfg["sim"]["duration_s"]
    if args.demo:
        duration_s = min(duration_s, 120.0)

    grid = assets.grid
    stations = assets.stations
    print(f"[run] grid {grid.shape3}, {len(stations)} stations, duration {duration_s:.0f}s")

    wind = ConstantWind(
        u=cfg["wind"]["constant"]["u_mps"],
        v=cfg["wind"]["constant"]["v_mps"],
        w=cfg["wind"]["constant"]["w_mps"],
    )
    field = build_rander(
        grid, wind, assets.population_density, assets.height_mask,
        v_cruise=cfg["cost"]["v_cruise_mps"],
        alpha_noise=cfg["cost"]["alpha_noise"],
        alpha_obstacle=cfg["cost"]["alpha_obstacle"],
        use_walls_flag=cfg["cost"]["use_walls_flag"],
    )

    station_xys = np.asarray([s.xy_m for s in stations], dtype=np.float64)
    z_cruise = 0.5 * (cfg["fmm"]["z_floor_m"] + cfg["fmm"]["z_ceiling_m"])

    traj_dir = Path(args.traj) if args.traj else (Path(args.assets) / "traj")
    if traj_dir.exists() and (traj_dir / "trajectories.pkl").exists():
        db = load_traj_db(traj_dir)
        print(f"[run] loaded {len(db.pairs)} prebuilt trajectory pairs from {traj_dir}")
    elif args.no_fmm:
        from aura.schemas import NominalTrajectory
        from aura.fmm.agd_solver import AllPairsDB
        db = AllPairsDB(grid=grid)
        cov = cfg.get("fmm", {}).get("coverage_radius_m", 5000.0)
        for i in range(len(stations)):
            for j in range(len(stations)):
                if i == j:
                    continue
                xi = np.array([station_xys[i, 0], station_xys[i, 1], z_cruise])
                xj = np.array([station_xys[j, 0], station_xys[j, 1], z_cruise])
                if np.linalg.norm(xj[:2] - xi[:2]) > cov:
                    continue
                n = max(2, int(np.linalg.norm(xj - xi) / cfg["cost"]["v_cruise_mps"] /
                              cfg["fmm"]["trajectory_dt_s"]) + 1)
                pts = np.linspace(xi, xj, n)
                db.pairs[(i, j)] = NominalTrajectory(
                    src=i, dst=j, xyz=pts.tolist(),
                    dt_s=cfg["fmm"]["trajectory_dt_s"], metric_version=0,
                )
        print(f"[run] using straight-line nominals ({len(db.pairs)} pairs)")
    else:
        t0 = time.time()
        db = solve_all_pairs(field, grid, station_xys, station_z_m=z_cruise,
                             trajectory_dt_s=cfg["fmm"]["trajectory_dt_s"],
                             coverage_radius_m=cfg.get("fmm", {}).get("coverage_radius_m", 5000.0))
        print(f"[run] FMM all-pairs: {len(db.pairs)} pairs in {time.time()-t0:.1f}s")

    # Heterogeneous fleet: 70/30 split unless overridden via --class-split.
    n_uavs = cfg["fleet"]["n_uavs"]
    if args.class_split is not None:
        class_split = tuple(args.class_split)
    else:
        cls_cfg = cfg.get("fleet", {}).get("classes")
        if isinstance(cls_cfg, list) and len(cls_cfg) == 2:
            class_split = (int(cls_cfg[0].get("count", int(0.7 * n_uavs))),
                           int(cls_cfg[1].get("count", n_uavs - int(0.7 * n_uavs))))
        else:
            class_split = (int(0.7 * n_uavs), n_uavs - int(0.7 * n_uavs))
    fleet = make_fleet(stations, n_uavs=n_uavs, z_cruise=z_cruise,
                       class_split=class_split)
    n_small = int((fleet.class_id == 0).sum())
    n_large = int((fleet.class_id == 1).sum())
    print(f"[run] fleet {fleet.n}  small={n_small}  large={n_large}")

    mpc = TubeMPC(
        mass_kg=cfg["mpc"]["uav"]["mass_kg"],
        drag_gamma=cfg["mpc"]["uav"]["drag_gamma"],
        u_max_n=cfg["mpc"]["uav"]["u_max_n"],
        v_max_mps=cfg["mpc"]["uav"]["v_max_mps"],
        horizon_n=cfg["mpc"]["horizon_n"],
        dt_s=cfg["mpc"]["dt_s"],
        q_pos=cfg["mpc"]["cost_weights"]["Q_pos"],
        r_vel=cfg["mpc"]["cost_weights"]["R_vel"],
        lam_u=cfg["mpc"]["cost_weights"]["lam_u"],
        slack_tube=cfg["mpc"]["cost_weights"]["slack_tube"],
    )
    print(f"[run] MPC backend = {mpc.backend()}")

    reg = NoFlyRegistry()
    haz_zones = schedule_to_events(cfg["missions"]["hazard"]["schedule"],
                                   _identity_lonlat_to_xy_m,
                                   z_min_m=0.0, z_max_m=200.0)
    fired = set()

    rng = np.random.default_rng(cfg.get("seed", 0))
    orders: list[Order] = []
    kpi = KpiAccumulator()
    rejects = RejectCounters()

    uav_states = {
        i: {"home": int(fleet.home_station[i]), "soc": 1.0,
            "idle": True, "next_launch": 0.0,
            "completion_at": -1.0, "order_id": -1,
            "outbound": True,
            "class_id": int(fleet.class_id[i])}
        for i in range(fleet.n)
    }

    # OrderBook orchestrator — multi-type Poisson generator.
    order_book = OrderBook(
        specs=default_specs(),
        stations=stations,
        db=db,
        master_rng=rng,
        coverage_radius_m=cfg.get("fmm", {}).get("coverage_radius_m", 5000.0),
    )

    bounds_xy = (
        (grid.origin_xy[0], grid.origin_xy[1]),
        (grid.origin_xy[0] + grid.size_xy[0],
         grid.origin_xy[1] + grid.size_xy[1]),
    )

    clock = SimClock(tick_s=cfg["sim"]["tick_s"])

    def physics_step(now: float, dt: float) -> None:
        F = mpc.step(fleet, now)
        wind_uvw = np.broadcast_to(
            np.asarray([cfg["wind"]["constant"]["u_mps"],
                        cfg["wind"]["constant"]["v_mps"],
                        cfg["wind"]["constant"]["w_mps"]]),
            fleet.pos.shape,
        ).copy()
        integrate(fleet, F, wind_uvw, dt=dt,
                  mass=cfg["mpc"]["uav"]["mass_kg"],
                  gamma=cfg["mpc"]["uav"]["drag_gamma"],
                  v_max=cfg["mpc"]["uav"]["v_max_mps"],
                  bounds_xy=bounds_xy,
                  bounds_z=(cfg["region"]["z_min_m"], cfg["region"]["z_max_m"]))

    def assign_step(now: float, dt: float) -> None:
        # Spawn new orders via OrderBook (multi-type Poisson).
        n_before = len(orders)
        order_book.step(orders, now, dt)
        # Per-type seen counts.
        for o in orders[n_before:]:
            kpi.record_seen(1, o.type)
        # Greedy assign + delay-only CBS
        asg = alns_assign(orders, stations, uav_states, db,
                          battery_capacity_wh=cfg["assignment"]["battery"]["capacity_wh"],
                          consumption_per_meter_wh=cfg["assignment"]["battery"]["consumption_per_meter_wh"],
                          reserve_frac=cfg["assignment"]["battery"]["reserve_frac"],
                          launch_interval_s=cfg["assignment"]["station_slot"]["launch_interval_s"],
                          now_s=now,
                          reject_counters=rejects)
        if not asg:
            return
        eps_default = cfg["tube"]["eps_branch_m"]
        reqs = [MissionRequest(mission_id=a.order_id,
                               nominal=db.get(orders[a.order_id].origin,
                                              orders[a.order_id].destination),
                               eps_m=eps_default,
                               requested_launch_s=a.launch_s)
                for a in asg]
        delays = resolve_delays(reqs, grid, bin_s=cfg["cbs"]["time_bin_s"],
                                max_delay_s=cfg["cbs"]["max_delay_s"])
        for a in asg:
            if a.order_id not in delays:
                continue
            launch = delays[a.order_id]
            traj = db.get(orders[a.order_id].origin, orders[a.order_id].destination)
            if traj is None:
                continue
            ref = make_reference_from_db(a.uav_id, traj, t_launch_s=launch,
                                         eps_m=eps_default)
            mpc.acquire(ref)
            ttime = max(0.0, (len(traj.xyz) - 1) * traj.dt_s)
            uav_states[a.uav_id]["completion_at"] = launch + ttime
            uav_states[a.uav_id]["order_id"] = a.order_id
            uav_states[a.uav_id]["outbound"] = True
            uav_states[a.uav_id]["idle"] = False
            fleet.state_code[a.uav_id] = IN_FLIGHT
            fleet.mission_id[a.uav_id] = a.order_id
            kpi.record_energy(a.energy_cost_proxy)

    def completion_step(now: float, dt: float) -> None:
        # Check arrivals
        for uid, st in uav_states.items():
            if st["idle"]:
                continue
            if st["completion_at"] < 0 or now < st["completion_at"]:
                continue
            if st["outbound"] and st["order_id"] >= 0:
                o = orders[st["order_id"]]
                o.status = "completed"
                o.completed_at_s = now
                kpi.record_complete(o, now)
                # Schedule return leg (back to home) using db lookup.
                traj = db.get(o.destination, st["home"])
                if traj is None:
                    st["idle"] = True
                    fleet.state_code[uid] = IDLE
                    mpc.release(uid)
                    continue
                eps_default = cfg["tube"]["eps_branch_m"]
                ref = make_reference_from_db(uid, traj, t_launch_s=now, eps_m=eps_default)
                mpc.acquire(ref)
                st["completion_at"] = now + max(0.0, (len(traj.xyz) - 1) * traj.dt_s)
                st["outbound"] = False
                fleet.state_code[uid] = RETURNING
            else:
                # Returned home — go idle.
                st["idle"] = True
                st["completion_at"] = -1.0
                st["order_id"] = -1
                fleet.state_code[uid] = IDLE
                mpc.release(uid)

    def hazard_step(now: float, dt: float) -> None:
        fresh = step_registry(reg, haz_zones, fired, now)
        for z in fresh:
            print(f"[run] hazard at t={now:.1f}: ({z.cx:.0f},{z.cy:.0f}) r={z.radius_m:.0f}")
            resolve_affected(field, grid, db, reg, z, station_xys,
                             station_z_m=z_cruise,
                             buffer_m=cfg["fmm"]["incremental"]["affected_zone_buffer_m"],
                             trajectory_dt_s=cfg["fmm"]["trajectory_dt_s"],
                             now_t_s=now)

    clock.add_layer("plant", cfg["sim"]["layers"]["plant_hz"], physics_step)
    clock.add_layer("alns", cfg["sim"]["layers"]["alns_hz"], assign_step)
    clock.add_layer("complete", cfg["sim"]["layers"]["alns_hz"], completion_step)
    clock.add_layer("hazard", 1.0, hazard_step)

    t0 = time.time()
    clock.run(duration_s)
    print(f"[run] sim wallclock {time.time()-t0:.1f}s")

    # Surface OrderBook + ALNS reject counters into the KPI dict.
    kpi.record_reject(rejects.to_dict())
    book_rejects = {f"book_{kk}_{tt}": vv
                    for kk, dct in (
                        ("rejected_geometry", order_book.counters.rejected_geometry),
                        ("rejected_no_traj", order_book.counters.rejected_no_traj),
                    )
                    for tt, vv in dct.items()}
    kpi.record_reject(book_rejects)

    kpi_path = args.kpi or cfg["kpi"]["log_path"]
    write_csv(kpi, duration_s, kpi_path)
    print(f"[run] kpi -> {kpi_path}")
    summary = kpi.to_dict(duration_s)
    overall_keys = ["orders_seen", "orders_completed", "orders_late",
                    "throughput_per_min", "avg_delivery_s", "sla_pct"]
    print("[run] overall:", {k: summary[k] for k in overall_keys if k in summary})
    type_names = ["food", "urgent_doc", "parcel", "heavy_cargo"]
    for t in type_names:
        ks = [f"{t}_seen", f"{t}_completed", f"{t}_late", f"{t}_sla"]
        if all(k in summary for k in ks):
            print(f"[run]   {t:12s} seen={summary[ks[0]]:5d} done={summary[ks[1]]:5d} "
                  f"late={summary[ks[2]]:4d} sla={summary[ks[3]]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
