"""Rolling-horizon UAV dispatch simulation v2 — with order batching.

Differences from rolling_sim.py:
  - Each dispatch step groups orders sharing the same pickup station
    into batches of up to MAX_BATCH_SIZE (default 3) using greedy
    destination-clustering. A batch is delivered by a single UAV.
  - UAV state machine handles multi-stop trips:
      IDLE → FLY_PICKUP → LOADING (60s × batch_size) →
      FLY_CUSTOMER_1 → … → FLY_CUSTOMER_k → IDLE at last customer.
  - Dispatchers (greedy + MF) operate on (N_uav, N_batch) cost matrix
    instead of (N_uav, N_orders). N_batch ≈ N_orders / mean_batch_size,
    typically 2-3× smaller — makes MF dispatch faster too.

This matches the paper Table 3 setup (which says "Order batching
collapses 342498 orders into 155223 trips", an avg batch size of 2.21).
"""
from __future__ import annotations
import sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT / "tools" / "visualize" / "scheduling"))

from business_circles import (load_circles_with_stations, demand_profile,
                              NANSHAN_CIRCLES)

PAIR_MATRIX = ROOT / "wave_test_large" / "outputs" / "pair_matrix.npz"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfigV2:
    n_uavs: int = 500
    n_stations: int = 118
    sim_hours: tuple = (18.0, 19.0)
    tick_sec: float = 30.0
    dispatch_interval_sec: float = 30.0
    load_sec: float = 60.0
    dispatch_radius_m: float = 4000.0
    target_orders_per_hour: int = 3000
    queue_timeout_sec: float = 1800.0
    seed: int = 0
    # batching
    max_batch_size: int = 3
    batch_detour_sec: float = 150.0
    # predictive rebalance (set enable_rebalance=False to disable)
    enable_rebalance: bool = False
    rebalance_interval_sec: float = 60.0
    rebalance_radius_m: float = 8000.0      # empty-flight radius (looser)
    rebalance_max_moves: int = 1500
    rebalance_local_reserve: int = 5
    rebalance_lookahead_sec: float = 600.0
    rebalance_ema_tau_sec: float = 300.0
    rebalance_batch_factor: float = 2.0
    rebalance_trigger_gap: int = 3
    # charging: after each completed delivery batch, the UAV must charge
    # at its current station for this many seconds before becoming IDLE
    # again. Set to 0 to disable.
    charge_sec_per_trip: float = 300.0


# ---------------------------------------------------------------------------
# Travel-time matrix loader (same as v1)
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


def load_T_reb(rebalance_radius_m):
    """Empty-flight travel time matrix — UAVs without payload can take
    direct straight-line routes, so we use distance/15.0 within a looser
    radius (default 8 km vs 4 km for payload dispatch)."""
    pm = np.load(PAIR_MATRIX, allow_pickle=False)
    dist = pm["distances_m"].astype(np.float32)
    T_reb = dist / 15.0
    T_reb[dist > rebalance_radius_m] = np.inf
    np.fill_diagonal(T_reb, np.inf)
    return T_reb


def calibrate_base_rate(target_per_hour, sim_hours):
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
# Fleet & order state (extended for batching)
# ---------------------------------------------------------------------------

IDLE, FLY_PICKUP, LOADING, FLY_CUSTOMER, DEADHEAD, CHARGING = 0, 1, 2, 3, 4, 5
O_PENDING, O_ASSIGNED, O_LOADING, O_FLIGHT, O_DELIVERED, O_FAILED = 0,1,2,3,4,5


@dataclass
class Fleet:
    station: np.ndarray
    state: np.ndarray
    busy_until: np.ndarray
    next_station: np.ndarray
    # batch tracking — per-UAV ragged lists
    batch_route: list          # list[list[int]]: remaining customer stations
    batch_orders: list         # list[list[int]]: parallel order ids


def init_fleet(cfg: SimConfigV2, rng):
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
        batch_route=[[] for _ in range(cfg.n_uavs)],
        batch_orders=[[] for _ in range(cfg.n_uavs)],
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
    uav_id: np.ndarray
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
# Order arrivals
# ---------------------------------------------------------------------------

def generate_arrivals(sim_h, dt_sec, base_rate, circle_info, T_eff,
                      residential_score, rng, n_stations):
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
# Batching (port from simulation.py)
# ---------------------------------------------------------------------------

def make_batches(order_ids, customer_lookup, T_eff, max_size, max_detour):
    """Greedy destination-clustering. Orders are assumed to share the
    same pickup station. Returns list of [order_id, ...] batches."""
    if not order_ids:
        return []
    ids = np.asarray(order_ids, dtype=np.int64)
    custs = customer_lookup[ids]
    n = len(ids)
    used = np.zeros(n, dtype=bool)
    batches = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        seed_c = int(custs[i])
        if max_size <= 1:
            batches.append([int(ids[i])])
            continue
        free = ~used[i+1:]
        if not free.any():
            batches.append([int(ids[i])])
            continue
        rel_idx = np.where(free)[0] + (i + 1)
        rel_custs = custs[rel_idx]
        ts = T_eff[seed_c, rel_custs]
        valid = np.isfinite(ts) & (ts < max_detour)
        if not valid.any():
            batches.append([int(ids[i])])
            continue
        cand_idx = rel_idx[valid]
        cand_ts = ts[valid]
        order = np.argsort(cand_ts)[:max_size - 1]
        chosen = cand_idx[order]
        used[chosen] = True
        batches.append([int(ids[i])] + [int(ids[j]) for j in chosen])
    return batches


def batch_route_and_cost(batch_orders, pickup_station, customer_lookup, T_eff,
                          load_sec):
    """Given a batch (all sharing pickup_station), compute nearest-neighbor
    delivery route through customers and the total trip cost (excluding
    leg1 from UAV to pickup, which depends on the UAV)."""
    custs = [int(customer_lookup[o]) for o in batch_orders]
    n = len(custs)
    if n == 0:
        return [], 0.0, []
    if n == 1:
        leg = float(T_eff[pickup_station, custs[0]])
        return custs, load_sec * 1 + leg, list(batch_orders)
    # NN traversal starting from pickup
    remaining = list(range(n))
    route_idx = []
    legs = []
    cur = pickup_station
    while remaining:
        best = remaining[0]; best_t = float(T_eff[cur, custs[remaining[0]]])
        for r in remaining[1:]:
            t = float(T_eff[cur, custs[r]])
            if t < best_t:
                best_t = t; best = r
        route_idx.append(best)
        legs.append(best_t)
        remaining.remove(best)
        cur = custs[best]
    ordered_custs = [custs[i] for i in route_idx]
    ordered_orders = [batch_orders[i] for i in route_idx]
    total = load_sec * n + sum(legs)
    return ordered_custs, total, ordered_orders


# ---------------------------------------------------------------------------
# Predictive rebalance (ported from simulation.py)
# ---------------------------------------------------------------------------

def predictive_rebalance(station_idle, station_in_flight_to,
                          ema_pickup_rate, T_reb, sim_sec,
                          max_moves, max_per_station,
                          local_reserve, lookahead_sec,
                          batch_factor, trigger_gap):
    """Returns list of (src_station, dst_station, t_arrive) moves and the
    computed per-station target idle count.

    For each station:
      target_idle = max(local_reserve, ema_pickup_rate × lookahead / batch_factor)
    Surplus stations (idle > target) send UAVs to deficit stations
    (idle+in_flight < target). Most-needy targets served first by nearest
    source. Avoids jitter via trigger_gap.
    """
    n_st = station_idle.shape[0]
    raw_target = ema_pickup_rate * lookahead_sec / max(batch_factor, 1e-6)
    target = np.maximum(local_reserve, raw_target).astype(np.int32)
    target = np.minimum(target, max_per_station)

    eff_inv = station_idle + station_in_flight_to
    total_fleet = int(eff_inv.sum())
    sum_t = int(target.sum())
    if sum_t > total_fleet and sum_t > 0:
        target = (target.astype(np.float32) * total_fleet / sum_t).astype(np.int32)
        target = np.maximum(target, local_reserve).clip(max=max_per_station)

    surplus = (eff_inv - target).clip(min=0)
    surplus = np.minimum(surplus, station_idle)        # can only move idle UAVs
    deficit = (target - eff_inv).clip(min=0)
    surplus[surplus <= trigger_gap] = 0
    deficit[deficit <= trigger_gap] = 0
    src_ids = np.where(surplus > 0)[0]
    dst_ids = np.where(deficit > 0)[0]
    if src_ids.size == 0 or dst_ids.size == 0:
        return [], target

    dst_ids = dst_ids[np.argsort(-deficit[dst_ids])]
    moves = []
    n_total = 0
    for d in dst_ids:
        need = int(deficit[d])
        if need <= 0:
            continue
        live_src = src_ids[(surplus[src_ids] > 0)
                            & np.isfinite(T_reb[src_ids, d])]
        if live_src.size == 0:
            continue
        s_order = live_src[np.argsort(T_reb[live_src, d])]
        for s in s_order:
            if n_total >= max_moves:
                break
            avail = int(surplus[s])
            if avail <= 0:
                continue
            n_move = int(min(need, avail, max_moves - n_total))
            if n_move <= 0:
                continue
            station_idle[s] -= n_move
            station_in_flight_to[d] += n_move
            surplus[s] -= n_move
            t_arrive = sim_sec + float(T_reb[s, d])
            for _ in range(n_move):
                moves.append((int(s), int(d), t_arrive))
            need -= n_move
            n_total += n_move
            if need <= 0:
                break
        if n_total >= max_moves:
            break
    return moves, target


# ---------------------------------------------------------------------------
# Dispatch cost builder (UAV i -> batch b)
# ---------------------------------------------------------------------------

INFEASIBLE = 1e6


def build_batch_cost(idle_uav_stations, batches, T_eff, customer_lookup,
                      load_sec):
    """Returns:
       C[i, b]   = leg1[uav_i_st, batch_pickup_b] + trip_cost_b
       trip_cost[b] : independent of UAV (loading + sequential delivery)
       batch_pickup[b], batch_orders_ordered[b], batch_route[b]
    """
    N = len(idle_uav_stations)
    B = len(batches)
    if B == 0:
        return np.zeros((N, 0)), [], [], []
    pickups = []; trip_costs = []; ordered_orders_all = []; routes_all = []
    for orders in batches:
        # all orders in batch share pickup station
        ps = orders[0]   # placeholder; replace next line
        # caller passes us "orders" but we need pickup; pull from caller
        pickups.append(orders[-1])   # not used directly here, recomputed
    # Compute per-batch trip and pickup at caller-level instead — fix:
    return None


def dispatch_pipeline(idle_uav_ids, idle_uav_stations,
                      pending_ids_by_ps,           # dict pickup_station -> [order_ids]
                      book, T_eff, max_batch_size, batch_detour_sec, load_sec,
                      pending_ages_arr, dispatcher, **dispatcher_kwargs):
    """Form batches per pickup station, build (N_uav, N_batch) cost,
    delegate assignment to `dispatcher`, return list of (uav_id, batch_meta)."""
    if not idle_uav_ids.size or not pending_ids_by_ps:
        return []

    batches = []
    batch_pickup = []
    batch_trip_cost = []
    batch_route = []           # ordered customer stations per batch
    batch_orders_ordered = []  # ordered order ids per batch
    batch_age = []             # max age across orders in batch (for priority)

    for ps, order_ids in pending_ids_by_ps.items():
        if not order_ids:
            continue
        bs = make_batches(order_ids, book.customer, T_eff,
                          max_batch_size, batch_detour_sec)
        for b in bs:
            route, trip, ordered = batch_route_and_cost(
                b, ps, book.customer, T_eff, load_sec)
            if not route:
                continue
            batches.append(b)
            batch_pickup.append(ps)
            batch_trip_cost.append(trip)
            batch_route.append(route)
            batch_orders_ordered.append(ordered)
            ages = [pending_ages_arr[oid] for oid in b]
            batch_age.append(max(ages))

    B = len(batches)
    if B == 0:
        return []

    pickups_arr = np.array(batch_pickup, dtype=np.int32)
    trip_arr = np.array(batch_trip_cost, dtype=np.float64)
    ages_arr = np.array(batch_age, dtype=np.float64)

    # leg1[i, b] = T_eff[uav_station_i, pickup_b]
    leg1 = T_eff[idle_uav_stations[:, None], pickups_arr[None, :]]
    C = leg1 + trip_arr[None, :]
    feas = np.isfinite(C)
    C = np.where(feas, C, INFEASIBLE).astype(np.float64)

    # Dispatcher returns list of (uav_idx_in_idle_list, batch_idx)
    pairs = dispatcher(C, feas, ages_arr, **dispatcher_kwargs)

    # Build assignment outputs
    out = []
    for uav_local_idx, b_idx in pairs:
        uav_id = int(idle_uav_ids[uav_local_idx])
        leg1_val = float(leg1[uav_local_idx, b_idx])
        out.append(dict(
            uav_id=uav_id,
            pickup=int(pickups_arr[b_idx]),
            leg1=leg1_val,
            trip_cost=float(trip_arr[b_idx]),
            route=batch_route[b_idx],
            orders=batch_orders_ordered[b_idx],
        ))
    return out


# ---------------------------------------------------------------------------
# Dispatchers (operate on the (N_uav, N_batch) cost matrix)
# ---------------------------------------------------------------------------

def greedy_dispatcher(C, feas, ages):
    """Process batches in age-descending order; each picks the closest
    reachable idle UAV (leg1-minimising = first column of C minus
    trip_cost, but since trip_cost is constant per batch, argmin C is
    equivalent for picking the UAV)."""
    N, B = C.shape
    if N == 0 or B == 0:
        return []
    order = np.argsort(-ages)
    used = np.zeros(N, dtype=bool)
    out = []
    for b in order:
        cands = np.where(~used & feas[:, b])[0]
        if cands.size == 0:
            continue
        best = cands[np.argmin(C[cands, b])]
        used[best] = True
        out.append((int(best), int(b)))
    return out


def mf_gpu_dispatcher(C, feas, ages, age_weight=0.5, seed=0,
                      demand_bias=None):
    """Mean-field hybrid-LSA assignment on GPU.

    demand_bias : optional (N_uav,) array giving an extra reward for
    UAVs ending up at high-demand stations after dispatch — passed
    through by the caller from the demand forecast (a simplified
    F_balance influence).
    """
    from mf_scheduler_torch import mf_hybrid_lsa_fast_t, MFConfigT
    N, B = C.shape
    if N == 0 or B == 0:
        return []
    # Age priority: subtract age*weight so older batches look cheaper.
    C_eff = C - (age_weight * ages)[None, :]
    if demand_bias is not None:
        C_eff = C_eff - demand_bias[:, None]
    # sentinel column = stay idle
    sentinel = 5e5
    Cext = np.hstack([C_eff, np.full((N, 1), sentinel)]).astype(np.float64)
    cfg = MFConfigT(theta0=50.0, gamma=0.97, n_iter=60, eta=0.4,
                    perturb=0.1, tol=1e-5, seed=seed, device="cuda")
    task_of_uav, _ = mf_hybrid_lsa_fast_t(Cext, lam_one=200.0, cfg=cfg)
    out = []
    for i in range(N):
        j = int(task_of_uav[i])
        if j < 0 or j >= B:
            continue
        if not feas[i, j]:
            continue
        out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_simulation(cfg: SimConfigV2, dispatcher_fn, dispatcher_name,
                   verbose=True, use_demand_bias=False):
    rng = np.random.default_rng(cfg.seed)
    info = load_circles_with_stations()
    T_eff = load_T_eff(cfg.dispatch_radius_m)
    T_reb = load_T_reb(cfg.rebalance_radius_m)
    base_rate = calibrate_base_rate(cfg.target_orders_per_hour, cfg.sim_hours)

    fleet = init_fleet(cfg, rng)
    MAX_ORDERS = int(cfg.target_orders_per_hour
                     * (cfg.sim_hours[1] - cfg.sim_hours[0]) * 3.0)
    book = new_orderbook(MAX_ORDERS)
    pending_by_ps: dict[int, list[int]] = {}

    t0_h, t1_h = cfg.sim_hours
    t_total = (t1_h - t0_h) * 3600.0
    n_ticks = int(t_total / cfg.tick_sec)

    # --- Per-station inventory accounting (needed for rebalance)
    # station_idle[s]      : # IDLE UAVs at station s
    # station_in_flight_to : # UAVs heading to s (FLY_PICKUP→pickup, FLY_CUSTOMER→customer, DEADHEAD→dest)
    station_idle = np.bincount(fleet.station, minlength=cfg.n_stations) \
                     .astype(np.int32)
    station_in_flight_to = np.zeros(cfg.n_stations, dtype=np.int32)
    MAX_PER_STATION = max(cfg.n_uavs // 4, 50)

    # --- EMA pickup rate per station (orders/sec)
    ema_pickup_rate = np.zeros(cfg.n_stations, dtype=np.float32)
    ema_alpha = cfg.tick_sec / max(cfg.rebalance_ema_tau_sec, cfg.tick_sec)
    pickups_this_tick = np.zeros(cfg.n_stations, dtype=np.int32)
    last_rebalance_sec = -1e9

    n_dispatch_calls = 0
    dispatch_wallclock = 0.0
    log_pending = np.zeros(n_ticks, dtype=np.int32)
    log_idle = np.zeros(n_ticks, dtype=np.int32)
    log_in_flight = np.zeros(n_ticks, dtype=np.int32)
    log_new_orders = np.zeros(n_ticks, dtype=np.int32)
    log_dispatched_orders = np.zeros(n_ticks, dtype=np.int32)
    log_delivered = np.zeros(n_ticks, dtype=np.int32)
    log_failed = np.zeros(n_ticks, dtype=np.int32)
    log_deadheads = np.zeros(n_ticks, dtype=np.int32)
    log_charging = np.zeros(n_ticks, dtype=np.int32)
    log_batch_sizes = []
    n_deadhead_total = 0

    if verbose:
        print(f"[{dispatcher_name}] base_rate={base_rate:.1f}/h  "
              f"target {cfg.target_orders_per_hour}/h  "
              f"fleet {cfg.n_uavs} UAV  {n_ticks} ticks  "
              f"max_batch={cfg.max_batch_size}  "
              f"rebalance={cfg.enable_rebalance}")

    t_wall0 = time.time()

    for tick in range(n_ticks):
        sim_sec = tick * cfg.tick_sec
        sim_h = t0_h + sim_sec / 3600.0
        pickups_this_tick.fill(0)

        # --- 1. Process UAV transitions (multi-leg + deadhead aware)
        ready = np.where(fleet.busy_until <= sim_sec)[0]
        for u in ready:
            st = fleet.state[u]
            if st == IDLE:
                continue
            elif st == FLY_PICKUP:
                # arrived at pickup → loading
                arrived_at = int(fleet.next_station[u])
                fleet.station[u] = arrived_at
                station_in_flight_to[arrived_at] -= 1
                fleet.state[u] = LOADING
                n_load = len(fleet.batch_orders[u])
                fleet.busy_until[u] = sim_sec + cfg.load_sec * n_load
                for oid in fleet.batch_orders[u]:
                    book.pickup_t[oid] = sim_sec
                    book.status[oid] = O_LOADING
                    pickups_this_tick[arrived_at] += 1
            elif st == LOADING:
                # finished loading → fly to first customer
                if not fleet.batch_route[u]:
                    fleet.state[u] = IDLE
                    station_idle[fleet.station[u]] += 1
                    continue
                first_cust = fleet.batch_route[u][0]
                leg = float(T_eff[fleet.station[u], first_cust])
                if not np.isfinite(leg):
                    for oid in fleet.batch_orders[u]:
                        book.delivered_t[oid] = sim_sec
                        book.status[oid] = O_DELIVERED
                        log_delivered[tick] += 1
                    fleet.batch_route[u] = []
                    fleet.batch_orders[u] = []
                    fleet.state[u] = IDLE
                    station_idle[fleet.station[u]] += 1
                else:
                    fleet.state[u] = FLY_CUSTOMER
                    fleet.next_station[u] = first_cust
                    fleet.busy_until[u] = sim_sec + leg
                    station_in_flight_to[first_cust] += 1
                    for oid in fleet.batch_orders[u]:
                        book.status[oid] = O_FLIGHT
            elif st == FLY_CUSTOMER:
                arrived_at = int(fleet.next_station[u])
                fleet.station[u] = arrived_at
                station_in_flight_to[arrived_at] -= 1
                if fleet.batch_orders[u]:
                    delivered_oid = fleet.batch_orders[u].pop(0)
                    fleet.batch_route[u].pop(0)
                    book.delivered_t[delivered_oid] = sim_sec
                    book.status[delivered_oid] = O_DELIVERED
                    log_delivered[tick] += 1
                if fleet.batch_route[u]:
                    next_cust = fleet.batch_route[u][0]
                    leg = float(T_eff[fleet.station[u], next_cust])
                    if np.isfinite(leg):
                        fleet.next_station[u] = next_cust
                        fleet.busy_until[u] = sim_sec + leg
                        station_in_flight_to[next_cust] += 1
                    else:
                        for oid in fleet.batch_orders[u]:
                            book.delivered_t[oid] = sim_sec
                            book.status[oid] = O_DELIVERED
                            log_delivered[tick] += 1
                        fleet.batch_route[u] = []
                        fleet.batch_orders[u] = []
                        # batch complete → CHARGING (or IDLE if disabled)
                        if cfg.charge_sec_per_trip > 0:
                            fleet.state[u] = CHARGING
                            fleet.busy_until[u] = sim_sec + cfg.charge_sec_per_trip
                        else:
                            fleet.state[u] = IDLE
                            station_idle[arrived_at] += 1
                else:
                    # entire batch delivered → CHARGING (or IDLE if disabled)
                    if cfg.charge_sec_per_trip > 0:
                        fleet.state[u] = CHARGING
                        fleet.busy_until[u] = sim_sec + cfg.charge_sec_per_trip
                    else:
                        fleet.state[u] = IDLE
                        station_idle[arrived_at] += 1
            elif st == DEADHEAD:
                # empty-repositioning trip done — no charging needed (lightly loaded)
                arrived_at = int(fleet.next_station[u])
                fleet.station[u] = arrived_at
                station_in_flight_to[arrived_at] -= 1
                fleet.state[u] = IDLE
                station_idle[arrived_at] += 1
            elif st == CHARGING:
                # finished charging at current station → idle
                fleet.state[u] = IDLE
                station_idle[fleet.station[u]] += 1

        # --- 2. Generate Poisson arrivals
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
            pending_by_ps.setdefault(ps, []).append(oid)
            book.n_orders += 1
        log_new_orders[tick] = len(new_orders)

        # --- 3. Time out stale orders
        n_failed_tick = 0
        for ps in list(pending_by_ps.keys()):
            survivors = []
            for oid in pending_by_ps[ps]:
                if sim_sec - book.created_t[oid] > cfg.queue_timeout_sec:
                    book.status[oid] = O_FAILED
                    n_failed_tick += 1
                else:
                    survivors.append(oid)
            if survivors:
                pending_by_ps[ps] = survivors
            else:
                del pending_by_ps[ps]
        log_failed[tick] = n_failed_tick

        # --- 4. Dispatch
        idle_mask = fleet.state == IDLE
        idle_ids = np.where(idle_mask)[0].astype(np.int32)
        idle_stations = fleet.station[idle_ids]

        if pending_by_ps and idle_ids.size:
            # Build flat pending-age lookup so dispatcher can compute per-batch age
            ages_arr = np.maximum(sim_sec - book.created_t, 0.0)

            kwargs = {}
            if dispatcher_name.startswith("mf"):
                kwargs["seed"] = cfg.seed
                if use_demand_bias:
                    # Simplified F_balance hint: each UAV gets a bonus based
                    # on the demand-weighted "attractiveness" of where it'd
                    # end up. Computed per-UAV before dispatch.
                    # (Skipped for v0 — left for future K>1 extension.)
                    kwargs["demand_bias"] = None

            t_disp0 = time.time()
            assignments = dispatch_pipeline(
                idle_ids, idle_stations,
                pending_by_ps, book, T_eff,
                cfg.max_batch_size, cfg.batch_detour_sec, cfg.load_sec,
                ages_arr,
                dispatcher_fn, **kwargs)
            dispatch_wallclock += time.time() - t_disp0
            n_dispatch_calls += 1

            n_dispatched_orders = 0
            for a in assignments:
                u = a["uav_id"]
                fleet.state[u] = FLY_PICKUP
                fleet.next_station[u] = a["pickup"]
                fleet.busy_until[u] = sim_sec + a["leg1"]
                fleet.batch_route[u] = list(a["route"])
                fleet.batch_orders[u] = list(a["orders"])
                # station inventory accounting: UAV leaves its current station,
                # heads to pickup.
                station_idle[fleet.station[u]] -= 1
                station_in_flight_to[a["pickup"]] += 1
                for oid in a["orders"]:
                    book.assigned_t[oid] = sim_sec
                    book.status[oid] = O_ASSIGNED
                    book.uav_id[oid] = u
                ps = a["pickup"]
                if ps in pending_by_ps:
                    pending_by_ps[ps] = [o for o in pending_by_ps[ps]
                                          if o not in set(a["orders"])]
                    if not pending_by_ps[ps]:
                        del pending_by_ps[ps]
                n_dispatched_orders += len(a["orders"])
                log_batch_sizes.append(len(a["orders"]))
            log_dispatched_orders[tick] = n_dispatched_orders

        # --- 5. EMA pickup-rate update + predictive rebalance
        rate_now = pickups_this_tick.astype(np.float32) / max(cfg.tick_sec, 1e-6)
        ema_pickup_rate = ((1.0 - ema_alpha) * ema_pickup_rate
                            + ema_alpha * rate_now)

        if (cfg.enable_rebalance
                and sim_sec - last_rebalance_sec >= cfg.rebalance_interval_sec):
            last_rebalance_sec = sim_sec
            moves, _ = predictive_rebalance(
                station_idle, station_in_flight_to,
                ema_pickup_rate, T_reb, sim_sec,
                max_moves=cfg.rebalance_max_moves,
                max_per_station=MAX_PER_STATION,
                local_reserve=cfg.rebalance_local_reserve,
                lookahead_sec=cfg.rebalance_lookahead_sec,
                batch_factor=cfg.rebalance_batch_factor,
                trigger_gap=cfg.rebalance_trigger_gap,
            )
            # assign actual UAVs to the abstract (src, dst) moves.
            # predictive_rebalance already decremented station_idle / incremented
            # station_in_flight_to. We just need to mark UAVs as DEADHEAD.
            moves_by_src: dict[int, list] = {}
            for s, d, t_arr in moves:
                moves_by_src.setdefault(s, []).append((d, t_arr))
            for s, dest_list in moves_by_src.items():
                # pick currently-IDLE UAVs at s
                candidates = np.where((fleet.state == IDLE)
                                       & (fleet.station == s))[0]
                k = min(len(candidates), len(dest_list))
                for i in range(k):
                    u = int(candidates[i])
                    d, t_arr = dest_list[i]
                    fleet.state[u] = DEADHEAD
                    fleet.next_station[u] = d
                    fleet.busy_until[u] = t_arr
                # if the count diverged (predictive_rebalance and reality
                # out of sync), unwind the over-decrement on station_idle
                # and over-increment on in_flight_to.
                over_count = len(dest_list) - k
                if over_count > 0:
                    station_idle[s] += over_count
                    for d, _ in dest_list[k:]:
                        station_in_flight_to[d] -= 1
            n_deadhead_total += sum(len(v) for v in moves_by_src.values())
            log_deadheads[tick] = len(moves)

        # recompute idle mask after rebalance
        idle_mask = fleet.state == IDLE
        log_pending[tick] = sum(len(v) for v in pending_by_ps.values())
        log_idle[tick] = int(idle_mask.sum())
        log_in_flight[tick] = int(((fleet.state == FLY_PICKUP)
                                   | (fleet.state == FLY_CUSTOMER)
                                   | (fleet.state == DEADHEAD)).sum())
        log_charging[tick] = int((fleet.state == CHARGING).sum())

        if verbose and tick % 20 == 0:
            print(f"  t={sim_h:5.2f}h  pend={log_pending[tick]:5d}  "
                  f"idle={log_idle[tick]:5d}  "
                  f"flight={log_in_flight[tick]:5d}  "
                  f"chg={log_charging[tick]:5d}  "
                  f"new={log_new_orders[tick]:4d}  "
                  f"disp={log_dispatched_orders[tick]:4d}  "
                  f"deliv={log_delivered[tick]:4d}  "
                  f"dh={int(log_deadheads[tick]):4d}")

    wall = time.time() - t_wall0
    n = book.n_orders
    delivered_mask = book.status[:n] == O_DELIVERED
    n_delivered = int(delivered_mask.sum())
    n_failed = int((book.status[:n] == O_FAILED).sum())
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
    mean_batch = float(np.mean(log_batch_sizes)) if log_batch_sizes else 0.0

    return dict(
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
        mean_batch_size=mean_batch,
        n_batches_total=len(log_batch_sizes),
        n_deadhead_total=int(n_deadhead_total),
        mean_charging_count=float(log_charging.mean()),
        peak_charging_count=int(log_charging.max()),
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def main():
    cfg = SimConfigV2(n_uavs=500, sim_hours=(18.0, 19.0),
                       target_orders_per_hour=3000, seed=0)
    print(f"[v2 smoke test] {cfg.n_uavs} UAV, "
          f"{cfg.target_orders_per_hour}/h, "
          f"max_batch={cfg.max_batch_size}")
    print("\n--- Greedy with batching ---")
    k_g = run_simulation(cfg, greedy_dispatcher, "greedy-batch", verbose=True)
    print("\n--- MF-CPU with batching (note: dispatcher tries cuda which "
          "will fail on login node — expected, this is just for the API) ---")
    try:
        k_m = run_simulation(cfg, mf_gpu_dispatcher, "mf-batch", verbose=True)
    except Exception as e:
        print(f"  expected failure on no-GPU login node: {e}")
        k_m = None

    print("\n=== SMOKE TEST RESULTS ===")
    print(f"  greedy-batch: orders={k_g['n_orders']} "
          f"delivered={k_g['n_delivered']} ({k_g['completion_pct']:.1f}%)  "
          f"mean_batch={k_g['mean_batch_size']:.2f}  "
          f"wall={k_g['sim_wallclock_s']:.1f}s")
    if k_m is not None:
        print(f"  mf-batch    : orders={k_m['n_orders']} "
              f"delivered={k_m['n_delivered']} ({k_m['completion_pct']:.1f}%)  "
              f"mean_batch={k_m['mean_batch_size']:.2f}  "
              f"wall={k_m['sim_wallclock_s']:.1f}s")


if __name__ == "__main__":
    main()
