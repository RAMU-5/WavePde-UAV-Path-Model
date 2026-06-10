"""Rolling-horizon UAV dispatch simulation with Poisson order stream.

Compares GREEDY vs MEAN-FIELD dispatchers end-to-end over a 1-hour
dinner-peak window on the real Nanshan station / travel-time data.

Mini-scale (compared to simulation.py):
  - 500 UAVs across 118 stations (vs 7080)
  - 1 hour window 18:00 → 19:00 (vs 4 hours)
  - Order rate scaled to give ~3000 orders/hour (vs ~85k)
  - No batching (1 UAV = 1 order) to keep the MF cost matrix tractable
  - No proactive deadhead rebalancing — dispatchers see whatever is idle

What this DOES test (that test_mf_dispatch.py doesn't):
  - Poisson arrivals with real time-of-day profile (lunch/dinner peaks)
    differentiated per business circle type (office/mall/tech/etc.)
  - Rolling horizon: re-solve every dispatch tick, UAV state and order
    backlog evolve continuously
  - End-to-end KPIs: wait time, delivery time, fleet utilisation,
    completion rate over the whole window
  - Decision latency: per-tick wallclock of the dispatcher
"""
from __future__ import annotations
import heapq, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Callable

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT / "tools" / "visualize" / "scheduling"))

from mf_scheduler import mf_hybrid_lsa_assignment, MFConfig
from business_circles import (load_circles_with_stations, demand_profile,
                              NANSHAN_CIRCLES)

PAIR_MATRIX = ROOT / "wave_test_large" / "outputs" / "pair_matrix.npz"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    n_uavs: int = 500
    n_stations: int = 118
    sim_hours: tuple = (18.0, 19.0)        # dinner peak window
    tick_sec: float = 30.0                 # event step
    dispatch_interval_sec: float = 30.0    # one dispatch per tick
    load_sec: float = 60.0                 # merchant pickup wait
    dispatch_radius_m: float = 4000.0
    target_orders_per_hour: int = 3000     # tuning param
    queue_timeout_sec: float = 1800.0      # 30 min before order is "failed"
    seed: int = 0


# ---------------------------------------------------------------------------
# Travel-time matrix loader
# ---------------------------------------------------------------------------

def load_T_eff(dispatch_radius_m):
    pm = np.load(PAIR_MATRIX, allow_pickle=False)
    T = pm["T_matrix"].astype(np.float32)
    dist = pm["distances_m"].astype(np.float32)
    fallback = dist / 15.0
    T_eff = np.where(np.isfinite(T), T, fallback)
    T_eff[dist > dispatch_radius_m] = np.inf
    np.fill_diagonal(T_eff, np.inf)
    return T_eff


# ---------------------------------------------------------------------------
# Order arrival rate calibration
# ---------------------------------------------------------------------------

def calibrate_base_rate(target_per_hour, sim_hours):
    """Solve for BASE so that the expected hub orders over [t0,t1]
    equals target_per_hour × (t1 - t0)."""
    t0, t1 = sim_hours
    t_grid = np.linspace(t0, t1, 64)
    dt = (t1 - t0) / len(t_grid)
    total_weight = 0.0
    for c in NANSHAN_CIRCLES:
        prof = np.array([demand_profile(t, c.profile) for t in t_grid])
        total_weight += c.weight * prof.sum() * dt
    target_total = target_per_hour * (t1 - t0)
    return target_total / total_weight


# ---------------------------------------------------------------------------
# Fleet & order state (numpy arrays for speed)
# ---------------------------------------------------------------------------

# UAV state codes
IDLE, FLY_PICKUP, LOADING, FLY_CUSTOMER = 0, 1, 2, 3

# Order status codes
O_PENDING, O_ASSIGNED, O_LOADING, O_FLIGHT, O_DELIVERED, O_FAILED = 0,1,2,3,4,5


@dataclass
class Fleet:
    station: np.ndarray            # (N_uav,) current/home station
    state: np.ndarray              # (N_uav,) UAV state code
    busy_until: np.ndarray         # (N_uav,) sim_sec when state transition next
    next_station: np.ndarray       # (N_uav,) destination station of current move
    current_order: np.ndarray      # (N_uav,) order id this UAV is serving (-1 if idle)


def init_fleet(cfg: SimConfig, rng):
    # distribute UAVs across stations (roughly uniform)
    base = cfg.n_uavs // cfg.n_stations
    rem = cfg.n_uavs - base * cfg.n_stations
    per_st = np.full(cfg.n_stations, base, dtype=np.int32)
    per_st[:rem] += 1
    station = np.repeat(np.arange(cfg.n_stations), per_st)[:cfg.n_uavs]
    rng.shuffle(station)
    return Fleet(
        station=station.astype(np.int32),
        state=np.full(cfg.n_uavs, IDLE, dtype=np.int8),
        busy_until=np.zeros(cfg.n_uavs, dtype=np.float64),
        next_station=np.full(cfg.n_uavs, -1, dtype=np.int32),
        current_order=np.full(cfg.n_uavs, -1, dtype=np.int32),
    )


@dataclass
class OrderBook:
    pickup: np.ndarray
    customer: np.ndarray
    created_t: np.ndarray
    assigned_t: np.ndarray
    pickup_t: np.ndarray
    delivered_t: np.ndarray
    status: np.ndarray
    uav_id: np.ndarray            # which UAV took this order (-1 if pending)
    n_orders: int = 0


def new_orderbook(max_orders):
    return OrderBook(
        pickup=np.zeros(max_orders, dtype=np.int32),
        customer=np.zeros(max_orders, dtype=np.int32),
        created_t=np.full(max_orders, -1, dtype=np.float32),
        assigned_t=np.full(max_orders, -1, dtype=np.float32),
        pickup_t=np.full(max_orders, -1, dtype=np.float32),
        delivered_t=np.full(max_orders, -1, dtype=np.float32),
        status=np.zeros(max_orders, dtype=np.int8),
        uav_id=np.full(max_orders, -1, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Order generation (Poisson per business circle + scatter)
# ---------------------------------------------------------------------------

def generate_arrivals(sim_h, dt_sec, base_rate, circle_info, T_eff,
                      residential_score, rng, n_stations):
    """Return list of (pickup_station, customer_station) for orders
    arriving in this [sim_h, sim_h + dt_sec/3600] window."""
    new = []
    circle_to_st = circle_info["circle_to_station"]
    for k, c in enumerate(NANSHAN_CIRCLES):
        rate_per_h = base_rate * c.weight * demand_profile(sim_h, c.profile)
        n_new = rng.poisson(rate_per_h * dt_sec / 3600.0)
        if n_new == 0:
            continue
        ps = int(circle_to_st[k])
        feasible = np.isfinite(T_eff[ps]) & (np.arange(n_stations) != ps)
        p = residential_score * feasible
        psum = p.sum()
        if psum < 1e-9:
            continue
        custs = rng.choice(n_stations, size=n_new, p=p / psum)
        for cust in custs:
            new.append((ps, int(cust)))
    return new


# ---------------------------------------------------------------------------
# Dispatchers (pluggable)
# ---------------------------------------------------------------------------

def build_dispatch_cost(idle_uav_stations, pending_pickups, pending_customers,
                        T_eff):
    """C[i, j] = T_eff[uav_i_station, pickup_j] + T_eff[pickup_j, cust_j].
    Infeasible pairs get a large finite cost so MF can run."""
    leg1 = T_eff[idle_uav_stations[:, None], pending_pickups[None, :]]
    leg2 = T_eff[pending_pickups, pending_customers]
    C = leg1 + leg2[None, :]
    feas_mask = np.isfinite(C)
    INF_COST = 1e6
    C = np.where(feas_mask, C, INF_COST).astype(np.float64)
    return C, feas_mask


def dispatch_greedy(idle_uav_ids, idle_uav_stations,
                    pending_order_ids, pending_pickups, pending_customers,
                    pending_ages, T_eff):
    """Serve oldest order first; pick closest reachable idle UAV.
    Returns list of (uav_id, order_id, leg1_cost, leg2_cost)."""
    if len(idle_uav_ids) == 0 or len(pending_order_ids) == 0:
        return []
    N = len(idle_uav_ids)
    M = len(pending_order_ids)
    C, feas = build_dispatch_cost(idle_uav_stations,
                                  pending_pickups, pending_customers, T_eff)
    leg1 = T_eff[idle_uav_stations[:, None], pending_pickups[None, :]]
    leg2 = T_eff[pending_pickups, pending_customers]

    used = np.zeros(N, dtype=bool)
    order_priority = np.argsort(-pending_ages)
    assignments = []
    for j in order_priority:
        cands = np.where((~used) & feas[:, j])[0]
        if len(cands) == 0:
            continue
        best = cands[np.argmin(leg1[cands, j])]
        used[best] = True
        assignments.append((int(idle_uav_ids[best]),
                            int(pending_order_ids[j]),
                            float(leg1[best, j]), float(leg2[j])))
    return assignments


def dispatch_mf(idle_uav_ids, idle_uav_stations,
                pending_order_ids, pending_pickups, pending_customers,
                pending_ages, T_eff, age_weight=0.5, seed=0):
    """Global mean-field hybrid-LSA assignment.

    Adds an age-priority term to the cost so older orders are slightly
    preferred (otherwise MF cherry-picks cheap orders and old ones rot).
    """
    if len(idle_uav_ids) == 0 or len(pending_order_ids) == 0:
        return []
    N = len(idle_uav_ids)
    M = len(pending_order_ids)
    C, feas = build_dispatch_cost(idle_uav_stations,
                                  pending_pickups, pending_customers, T_eff)
    # priority bonus: each order's effective cost is reduced by age*weight
    age_bonus = age_weight * pending_ages.astype(np.float64)  # (M,)
    C_eff = C - age_bonus[None, :]
    # sentinel column = stay idle (always feasible, "cost" between real
    # and infeasible so it's chosen only by truly excess UAVs)
    sentinel = 5e5
    Cext = np.hstack([C_eff, np.full((N, 1), sentinel)])

    # Jacobi (fully vectorised) for speed at large N; LSA cleanup at the
    # end recovers the strict 1-to-1 anyway, so per-iter quality matters
    # less than wall-clock.
    cfg = MFConfig(theta0=50.0, gamma=0.97, n_iter=60,
                   eta=0.4, perturb=0.1, tol=1e-5, seed=seed,
                   update="jacobi")
    task_of_uav, _ = mf_hybrid_lsa_assignment(Cext, lam_one=200.0, cfg=cfg)
    # extract assignments — drop sentinel and infeasible picks
    leg1 = T_eff[idle_uav_stations[:, None], pending_pickups[None, :]]
    leg2 = T_eff[pending_pickups, pending_customers]
    assignments = []
    for i in range(N):
        j = int(task_of_uav[i])
        if j < 0 or j >= M:           # unassigned or sentinel
            continue
        if not feas[i, j]:             # infeasible — MF was forced to pick
            continue
        assignments.append((int(idle_uav_ids[i]),
                            int(pending_order_ids[j]),
                            float(leg1[i, j]), float(leg2[j])))
    return assignments


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(cfg: SimConfig, dispatcher, dispatcher_name,
                   verbose=True):
    rng = np.random.default_rng(cfg.seed)
    info = load_circles_with_stations()
    T_eff = load_T_eff(cfg.dispatch_radius_m)
    base_rate = calibrate_base_rate(cfg.target_orders_per_hour, cfg.sim_hours)

    fleet = init_fleet(cfg, rng)
    MAX_ORDERS = int(cfg.target_orders_per_hour
                     * (cfg.sim_hours[1] - cfg.sim_hours[0]) * 3.0)
    book = new_orderbook(MAX_ORDERS)
    pending_ids: list[int] = []          # order ids currently in dispatch queue

    t0_h, t1_h = cfg.sim_hours
    t_total = (t1_h - t0_h) * 3600.0
    n_ticks = int(t_total / cfg.tick_sec)

    # ---- KPI accumulators
    n_dispatch_calls = 0
    dispatch_wallclock = 0.0
    per_tick_problem_size = []           # (N_idle, N_pending) per dispatch
    log_pending = np.zeros(n_ticks, dtype=np.int32)
    log_idle = np.zeros(n_ticks, dtype=np.int32)
    log_in_flight = np.zeros(n_ticks, dtype=np.int32)
    log_new_orders = np.zeros(n_ticks, dtype=np.int32)
    log_dispatched = np.zeros(n_ticks, dtype=np.int32)
    log_delivered = np.zeros(n_ticks, dtype=np.int32)
    log_failed = np.zeros(n_ticks, dtype=np.int32)

    if verbose:
        print(f"[{dispatcher_name}] base_rate={base_rate:.1f}/h  "
              f"target {cfg.target_orders_per_hour}/h  "
              f"fleet {cfg.n_uavs} UAV  {n_ticks} ticks")

    t_wall0 = time.time()

    for tick in range(n_ticks):
        sim_sec = tick * cfg.tick_sec
        sim_h = t0_h + sim_sec / 3600.0

        # --- 1. UAV state transitions that have completed by now
        ready = np.where(fleet.busy_until <= sim_sec)[0]
        for u in ready:
            st = fleet.state[u]
            if st == IDLE:
                continue
            elif st == FLY_PICKUP:
                # arrived at merchant → start loading
                fleet.state[u] = LOADING
                fleet.station[u] = fleet.next_station[u]
                fleet.busy_until[u] = sim_sec + cfg.load_sec
                oid = fleet.current_order[u]
                if oid >= 0:
                    book.pickup_t[oid] = sim_sec
                    book.status[oid] = O_LOADING
            elif st == LOADING:
                # finished loading → fly to customer
                oid = fleet.current_order[u]
                if oid >= 0:
                    leg2 = T_eff[book.pickup[oid], book.customer[oid]]
                    if np.isfinite(leg2):
                        fleet.state[u] = FLY_CUSTOMER
                        fleet.next_station[u] = book.customer[oid]
                        fleet.busy_until[u] = sim_sec + float(leg2)
                        book.status[oid] = O_FLIGHT
                    else:
                        # shouldn't happen — defensive: deliver immediately
                        fleet.state[u] = IDLE
                        book.status[oid] = O_DELIVERED
                        book.delivered_t[oid] = sim_sec
                        log_delivered[tick] += 1
                        fleet.current_order[u] = -1
            elif st == FLY_CUSTOMER:
                # arrived at customer → order delivered, UAV idle at customer
                oid = fleet.current_order[u]
                fleet.station[u] = fleet.next_station[u]
                fleet.state[u] = IDLE
                fleet.current_order[u] = -1
                if oid >= 0:
                    book.delivered_t[oid] = sim_sec
                    book.status[oid] = O_DELIVERED
                    log_delivered[tick] += 1

        # --- 2. Generate new Poisson arrivals
        new_orders = generate_arrivals(sim_h, cfg.tick_sec, base_rate,
                                        info, T_eff, info["residential_score"],
                                        rng, cfg.n_stations)
        for ps, cust in new_orders:
            if book.n_orders >= MAX_ORDERS:
                break
            oid = book.n_orders
            book.pickup[oid] = ps
            book.customer[oid] = cust
            book.created_t[oid] = sim_sec
            book.status[oid] = O_PENDING
            pending_ids.append(oid)
            book.n_orders += 1
        log_new_orders[tick] = len(new_orders)

        # --- 3. Time out stale orders
        if pending_ids:
            survivors = []
            for oid in pending_ids:
                if sim_sec - book.created_t[oid] > cfg.queue_timeout_sec:
                    book.status[oid] = O_FAILED
                    log_failed[tick] += 1
                else:
                    survivors.append(oid)
            pending_ids = survivors

        # --- 4. Dispatch
        idle_mask = fleet.state == IDLE
        idle_ids = np.where(idle_mask)[0].astype(np.int32)
        idle_stations = fleet.station[idle_ids]

        if pending_ids and len(idle_ids):
            pend_arr = np.array(pending_ids, dtype=np.int32)
            pickups = book.pickup[pend_arr]
            custs = book.customer[pend_arr]
            ages = sim_sec - book.created_t[pend_arr]

            per_tick_problem_size.append((len(idle_ids), len(pend_arr)))
            t_disp0 = time.time()
            assignments = dispatcher(
                idle_ids, idle_stations,
                pend_arr, pickups, custs, ages,
                T_eff)
            dispatch_wallclock += time.time() - t_disp0
            n_dispatch_calls += 1

            assigned_oids = set()
            for uav_id, oid, leg1, _ in assignments:
                # commit: UAV flies to pickup
                fleet.state[uav_id] = FLY_PICKUP
                fleet.next_station[uav_id] = book.pickup[oid]
                fleet.busy_until[uav_id] = sim_sec + leg1
                fleet.current_order[uav_id] = oid
                book.assigned_t[oid] = sim_sec
                book.status[oid] = O_ASSIGNED
                book.uav_id[oid] = uav_id
                assigned_oids.add(oid)
            pending_ids = [o for o in pending_ids if o not in assigned_oids]
            log_dispatched[tick] = len(assignments)

        # --- 5. Snapshot stats
        log_pending[tick] = len(pending_ids)
        log_idle[tick] = int(idle_mask.sum())
        log_in_flight[tick] = int(((fleet.state == FLY_PICKUP)
                                   | (fleet.state == FLY_CUSTOMER)).sum())

        if verbose and tick % 20 == 0:
            print(f"  t={sim_h:5.2f}h  pend={log_pending[tick]:4d}  "
                  f"idle={log_idle[tick]:4d}  "
                  f"flight={log_in_flight[tick]:4d}  "
                  f"new={log_new_orders[tick]:3d}  "
                  f"disp={log_dispatched[tick]:3d}  "
                  f"deliv={log_delivered[tick]:3d}")

    wall = time.time() - t_wall0

    # ---- KPI calculation
    n = book.n_orders
    delivered_mask = book.status[:n] == O_DELIVERED
    failed_mask = book.status[:n] == O_FAILED
    n_delivered = int(delivered_mask.sum())
    n_failed = int(failed_mask.sum())
    n_pending_end = int(((book.status[:n] != O_DELIVERED)
                          & (book.status[:n] != O_FAILED)).sum())

    if n_delivered > 0:
        wait_to_pickup = (book.pickup_t[:n][delivered_mask]
                          - book.created_t[:n][delivered_mask])
        delivery_time = (book.delivered_t[:n][delivered_mask]
                         - book.created_t[:n][delivered_mask])
    else:
        wait_to_pickup = np.array([0.0])
        delivery_time = np.array([0.0])

    fleet_util = 1.0 - (log_idle.mean() / cfg.n_uavs)

    kpis = dict(
        dispatcher=dispatcher_name,
        n_orders=n,
        n_delivered=n_delivered,
        n_failed=n_failed,
        n_pending_end=n_pending_end,
        completion_pct=100.0 * n_delivered / max(n, 1),
        mean_wait_pickup_s=float(wait_to_pickup.mean()),
        p95_wait_pickup_s=float(np.percentile(wait_to_pickup, 95)),
        mean_delivery_s=float(delivery_time.mean()),
        p95_delivery_s=float(np.percentile(delivery_time, 95)),
        fleet_util_mean=float(fleet_util),
        peak_pending=int(log_pending.max()),
        n_dispatch_calls=n_dispatch_calls,
        dispatch_wallclock_s=dispatch_wallclock,
        mean_dispatch_ms=1000.0 * dispatch_wallclock / max(n_dispatch_calls, 1),
        sim_wallclock_s=wall,
        per_tick_problem_size=per_tick_problem_size,
        log_pending=log_pending,
        log_idle=log_idle,
        log_in_flight=log_in_flight,
        log_new_orders=log_new_orders,
        log_dispatched=log_dispatched,
        log_delivered=log_delivered,
        log_failed=log_failed,
    )
    return kpis


def print_kpi_table(kpi_list):
    print("\n" + "=" * 78)
    print(f" SUMMARY  ({len(kpi_list)} dispatcher{'s' if len(kpi_list)>1 else ''})")
    print("=" * 78)
    cols = [
        ("dispatcher",            "{:>12s}",   "dispatcher"),
        ("n_orders",              "{:>8d}",    "orders"),
        ("n_delivered",           "{:>8d}",    "delivered"),
        ("completion_pct",        "{:>7.1f}%", "compl%"),
        ("mean_wait_pickup_s",    "{:>7.1f}",  "wait/s"),
        ("p95_wait_pickup_s",     "{:>7.1f}",  "p95w/s"),
        ("mean_delivery_s",       "{:>8.1f}",  "deliv/s"),
        ("p95_delivery_s",        "{:>8.1f}",  "p95d/s"),
        ("fleet_util_mean",       "{:>7.1%}",  "util"),
        ("peak_pending",          "{:>7d}",    "pkpend"),
        ("mean_dispatch_ms",      "{:>9.2f}",  "disp/ms"),
        ("sim_wallclock_s",       "{:>7.2f}",  "wall/s"),
    ]
    hdr = "  ".join(f"{h:>{max(len(h),7)}s}" for _, _, h in cols)
    print(hdr)
    for k in kpi_list:
        row = "  ".join(fmt.format(k[key]) for key, fmt, _ in cols)
        print(row)


def main():
    cfg = SimConfig()
    print(f"[sim] window {cfg.sim_hours[0]:.1f}→{cfg.sim_hours[1]:.1f}h  "
          f"fleet {cfg.n_uavs} UAV  target {cfg.target_orders_per_hour}/h")

    print("\n--- Running GREEDY ---")
    k_g = run_simulation(cfg, dispatch_greedy, "greedy", verbose=True)
    print("\n--- Running MEAN-FIELD ---")
    k_m = run_simulation(cfg, dispatch_mf, "mean-field", verbose=True)

    print_kpi_table([k_g, k_m])

    # comparison block
    print("\n" + "=" * 78)
    print(" COMPARISON  (mean-field vs greedy)")
    print("=" * 78)
    def pct(a, b):
        return f"{100*(a-b)/max(b,1e-9):+.2f}%"
    print(f"  completion       : {k_m['completion_pct']:.1f}% vs "
          f"{k_g['completion_pct']:.1f}%  "
          f"(Δ {k_m['completion_pct']-k_g['completion_pct']:+.2f}pp)")
    print(f"  mean wait        : {k_m['mean_wait_pickup_s']:.1f}s vs "
          f"{k_g['mean_wait_pickup_s']:.1f}s  "
          f"({pct(k_m['mean_wait_pickup_s'], k_g['mean_wait_pickup_s'])})")
    print(f"  mean delivery    : {k_m['mean_delivery_s']:.1f}s vs "
          f"{k_g['mean_delivery_s']:.1f}s  "
          f"({pct(k_m['mean_delivery_s'], k_g['mean_delivery_s'])})")
    print(f"  p95 delivery     : {k_m['p95_delivery_s']:.1f}s vs "
          f"{k_g['p95_delivery_s']:.1f}s  "
          f"({pct(k_m['p95_delivery_s'], k_g['p95_delivery_s'])})")
    print(f"  fleet util       : {k_m['fleet_util_mean']:.1%} vs "
          f"{k_g['fleet_util_mean']:.1%}")
    print(f"  peak pending q   : {k_m['peak_pending']} vs "
          f"{k_g['peak_pending']}")
    print(f"  dispatch latency : {k_m['mean_dispatch_ms']:.2f}ms vs "
          f"{k_g['mean_dispatch_ms']:.2f}ms  "
          f"({k_m['mean_dispatch_ms']/max(k_g['mean_dispatch_ms'],1e-9):.0f}×)")


if __name__ == "__main__":
    main()
