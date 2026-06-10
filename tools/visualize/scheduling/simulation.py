"""End-to-end Nanshan delivery simulation at Shenzhen scale.

Fleet configuration:
    - Each of 118 stations holds up to 100 UAVs (capacity).
    - Initially 60 UAVs are deployed per station → 7,080 active UAVs.

Order volume target: ~1,000,000 orders / day across the 7 business
circles (matches Meituan-class single-platform Shenzhen scale, scaled
to our 6.6×7.2 km bbox).

Per circle daily totals are determined by:
    daily_orders_circle = BASE × weight × ∫₀²⁴ profile(t, type) dt

Time horizon: 4h dinner peak window (17:30 → 21:30), which by integration
captures roughly 25-35% of daily volume depending on profile.

Architecture
------------
Vectorised event-driven core (heapq) for scalability. State per
station, not per UAV — UAVs treated as fungible "fleet pool" items
since they're identical. Order-level timing still tracked per-order.

Per dispatch tick (every 30s):
    - Generate Poisson arrivals at each circle
    - Greedy assignment per pickup-station: take the cheapest idle
      UAV reachable. With ~7000 UAVs and finite T_eff, this is near
      optimal and orders of magnitude faster than full mean-field over
      thousands of (UAV, order) pairs.

Future events processed via priority queue:
    1. UAV arrives at pickup → starts 60s loading
    2. UAV finishes loading → departs for customer
    3. UAV arrives at customer → station fleet ++; order delivered
"""
from __future__ import annotations
import heapq
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent.parent
PAIR_MATRIX = ROOT / "wave_test_large" / "outputs" / "pair_matrix.npz"

from business_circles import (load_circles_with_stations, demand_profile,
                              NANSHAN_CIRCLES)

OUT_DIR = ROOT / "visualization" / "scheduling_system"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Simulation parameters -----------------------------------------------

SIM_HOURS = (17.5, 21.5)        # 17:30 → 21:30 dinner peak
TICK_SEC = 30.0                 # discrete logging step
DISPATCH_INTERVAL_SEC = 30.0    # one dispatch per tick (greedy is fast)
LOAD_SEC = 60.0                 # merchant pickup duration (per item)
INITIAL_UAVS_PER_STATION = 60
MAX_UAVS_PER_STATION = 200      # raised — capacity for delivery surplus
N_STATIONS = 118
N_UAVS = INITIAL_UAVS_PER_STATION * N_STATIONS    # 7,080

# Order batching ("拼单") — one UAV may carry multiple orders in one trip
# if the destinations are close enough that the detour is cheap.
MAX_BATCH_SIZE = 3              # up to 3 orders per UAV per trip
BATCH_DETOUR_SEC = 150.0        # extra T (in s) between any two
                                # destinations to be batchable (~1500-2200m).
                                # Larger threshold → more 3-order batches

TARGET_ORDERS_PER_DAY = 1_000_000
QUEUE_TIMEOUT_SEC = 3600.0      # 60 min — order is "failed" only if it
                                # sits unassigned in the queue this long

# Order origin split:
#   60% concentrated in the 7 big business-circle hubs
#   40% scattered across all 118 stations (small restaurants / shops)
HUB_ORDER_FRACTION = 0.60
SCATTER_ORDER_FRACTION = 0.40

# Rebalance / deadhead repositioning — demand-driven, spare-UAV based
#   target:  station with unmet pending orders
#            unmet = n_pending - idle - in_flight_to > 0
#   source:  station whose idle UAVs exceed its own demand + reserve
#            spare = idle - max(n_pending, LOCAL_RESERVE) > 0
#            (i.e. if you have no pending, "max" = reserve; if you have
#             some pending, you must serve those first)
REBALANCE_INTERVAL_SEC = 60.0
LOCAL_RESERVE = 5                  # always keep this many idle for self
MAX_DEADHEADS_PER_REBALANCE = 1500

# Predictive (target-following) rebalance config.
#   "reactive" : original demand-driven (only fires when pending > supply)
#   "predictive": each station maintains a target idle pool sized by its
#                 EMA pickup rate × lookahead; surplus stations push UAVs
#                 to deficit stations every tick to track the target.
REBALANCE_MODE = "predictive"
PRED_LOOKAHEAD_SEC = 600.0          # 10 min lookahead window
PRED_EMA_TAU_SEC = 300.0            # 5 min EMA half-life-ish for rate
PRED_BATCH_FACTOR = 2.0             # ≈ avg batch size; converts orders→UAVs
PRED_TRIGGER_GAP = 3                # only act when |idle - target| > this


def _compute_base_rate(target_total=None):
    """Solve for BASE so daily hub total matches `target_total`."""
    if target_total is None:
        target_total = TARGET_ORDERS_PER_DAY * HUB_ORDER_FRACTION
    t_grid = np.linspace(0, 24, 24*60)
    dt = 24.0 / len(t_grid)
    total = 0.0
    for c in NANSHAN_CIRCLES:
        prof = np.array([demand_profile(t, c.profile) for t in t_grid])
        total += c.weight * prof.sum() * dt
    return target_total / total


def _compute_scatter_rate(target_total, res_score):
    """Per-station baseline pickup rate for the 'scatter' (small shops)
    component. Uses (1 - residential_score) as commercial intensity."""
    commercial = np.clip(1.0 - res_score, 0.05, 1.0)   # everyone gets some
    # Scatter has a milder profile than circle hotspots — call it
    # an average of mall + office.
    t_grid = np.linspace(0, 24, 24*60)
    dt = 24.0 / len(t_grid)
    avg_prof = 0.5 * (np.array([demand_profile(t, "mall") for t in t_grid])
                      + np.array([demand_profile(t, "office") for t in t_grid]))
    int_profile = avg_prof.sum() * dt              # weight·hours
    # daily orders per station = rate_per_st × int_profile
    # total over all stations = target_total
    weight_sum = commercial.sum() * int_profile
    rate_per_st_unit = target_total / weight_sum
    rates = commercial * rate_per_st_unit          # per-station per-hour
    return rates, avg_prof, t_grid


def _scatter_profile_value(t_h):
    """Scalar avg-of-mall-and-office multiplier at time t_h."""
    return 0.5 * (demand_profile(t_h, "mall")
                  + demand_profile(t_h, "office"))


def make_batches(order_ids, customer_lookup, T_eff,
                 max_size=MAX_BATCH_SIZE, max_detour=BATCH_DETOUR_SEC):
    """Vectorised greedy destination-clustering. For each seed (FIFO
    order), find up to `max_size - 1` other orders whose destinations
    are within `max_detour` seconds.

    O(N) numpy ops per seed instead of O(N) Python ops, ~50× faster
    for queues of thousands."""
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
        # consider only later, unused entries
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
        batches.append([int(ids[i])]
                       + [int(ids[j]) for j in chosen])
    return batches


def predictive_rebalance(station_idle, station_in_flight_to,
                         ema_pickup_rate, T_reb, sim_sec,
                         max_moves, MAX, lookahead_sec=PRED_LOOKAHEAD_SEC,
                         batch_factor=PRED_BATCH_FACTOR,
                         trigger_gap=PRED_TRIGGER_GAP):
    """Predictive (target-following) rebalance.

    For each station, a target idle UAV count is computed from its
    EMA pickup rate (orders/sec) × lookahead window / batch_factor.
    Stations whose effective inventory (idle + in_flight) exceeds target
    are sources; stations below target are sinks. We greedily route UAVs
    from the nearest source to the most-needy sink.

    Returns moves (s, d, t_arrive) — the caller schedules type-3 events.
    """
    n_st = station_idle.shape[0]
    # 1. desired idle per station = EMA rate × lookahead / batch_factor
    raw_target = ema_pickup_rate * lookahead_sec / max(batch_factor, 1e-6)
    target = np.maximum(LOCAL_RESERVE, raw_target).astype(np.int32)
    target = np.minimum(target, MAX)
    # 2. global conservation: if Σ target > fleet, scale down proportionally
    eff_inv = station_idle + station_in_flight_to
    total_fleet = int(eff_inv.sum())
    sum_t = int(target.sum())
    if sum_t > total_fleet and sum_t > 0:
        target = (target.astype(np.float32) * total_fleet / sum_t).astype(np.int32)
        target = np.maximum(target, LOCAL_RESERVE).clip(max=MAX)
    # 3. surplus / deficit
    surplus = (eff_inv - target).clip(min=0)
    # only take from physically-idle UAVs (can't move in-flight ones)
    surplus = np.minimum(surplus, station_idle)
    deficit = (target - eff_inv).clip(min=0)
    # require gap > trigger to avoid jitter
    surplus[surplus <= trigger_gap] = 0
    deficit[deficit <= trigger_gap] = 0
    src_ids = np.where(surplus > 0)[0]
    dst_ids = np.where(deficit > 0)[0]
    if src_ids.size == 0 or dst_ids.size == 0:
        return [], target
    # 4. process most-deficient first
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


def land_or_divert(target, station_idle, station_in_flight_to,
                   T_reb, sim_sec, MAX, events):
    """Land a UAV at `target`. If `target` is at cap, divert to the
    nearest station with spare capacity; the UAV pays an extra
    repositioning flight time `T_reb[target, alt]` before landing.

    Returns (final_station, extra_seconds, was_diverted).
    """
    if station_idle[target] < MAX:
        station_idle[target] += 1
        station_in_flight_to[target] -= 1
        return int(target), 0.0, False
    # target full → divert to nearest open station
    station_in_flight_to[target] -= 1
    open_mask = station_idle < MAX
    if not open_mask.any():
        # global saturation — fall back to overflow at original target
        station_idle[target] += 1
        return int(target), 0.0, False
    distances = np.where(open_mask, T_reb[target], np.inf)
    s_alt = int(np.argmin(distances))
    extra = float(distances[s_alt])
    if not np.isfinite(extra):
        # nothing reachable from `target` within T_reb — overflow at target
        station_idle[target] += 1
        return int(target), 0.0, False
    station_in_flight_to[s_alt] += 1
    heapq.heappush(events, (sim_sec + extra, 5, s_alt))
    return s_alt, extra, True


def rebalance_step(station_idle, station_in_flight_to, pending_q,
                   T_eff, sim_sec, max_moves):
    """Demand-driven deadhead repositioning.

    Source:  station with `n_pending == 0` AND `idle > LOCAL_RESERVE`
             (no current demand, has spare UAVs above reserve)
    Target:  station with pending orders AND
             `idle + in_flight_to < n_pending`
             (still under-provisioned even counting incoming UAVs)

    For each target (sorted by unmet need, most-needy first), pull
    UAVs from the nearest sources by T_eff.
    """
    n_pending = np.array([len(q) for q in pending_q], dtype=np.int32)
    unmet = n_pending - station_idle - station_in_flight_to
    targets_mask = unmet > 0
    needs = np.maximum(n_pending, LOCAL_RESERVE)
    spare = station_idle - needs
    sources_mask = (~targets_mask) & (spare > 0)
    sources = np.where(sources_mask)[0]
    targets = np.where(targets_mask)[0]
    if len(sources) == 0 or len(targets) == 0:
        return []
    # Most-needy targets first
    targets = targets[np.argsort(-unmet[targets])]

    moves = []
    n_total = 0
    for d in targets:
        need = int(unmet[d])
        if need <= 0: continue
        # Sort current sources by T(s, d) ascending
        valid_src = sources[np.isfinite(T_eff[sources, d])]
        if len(valid_src) == 0: continue
        s_order = valid_src[np.argsort(T_eff[valid_src, d])]
        for s in s_order:
            if n_total >= max_moves: break
            # Respect "spare" budget — recompute since station_idle changed
            available = int(station_idle[s]
                            - max(int(n_pending[s]), LOCAL_RESERVE))
            if available <= 0: continue
            t_fly = float(T_eff[s, d])
            n_move = min(need, available, max_moves - n_total)
            if n_move <= 0: continue
            station_idle[s] -= n_move
            station_in_flight_to[d] += n_move
            t_arrive = sim_sec + t_fly
            for _ in range(n_move):
                moves.append((int(s), int(d), t_arrive))
            need -= n_move
            n_total += n_move
            if need <= 0: break
        if n_total >= max_moves: break
    return moves


def main(seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    info = load_circles_with_stations()
    n_st = info["n_st"]
    circle_to_st = info["circle_to_station"]
    res_score = info["residential_score"]

    pm = np.load(PAIR_MATRIX, allow_pickle=False)
    T = pm["T_matrix"].astype(np.float32)
    distances = pm["distances_m"].astype(np.float32)
    fallback = distances / 15.0
    T_eff = np.where(np.isfinite(T), T, fallback)
    T_eff[distances > 4000.0] = np.inf
    np.fill_diagonal(T_eff, np.inf)
    # Rebalance T uses straight-line for empty repositioning out to 8km
    # (UAVs flying without payload can take direct routes). This lets
    # rebalance reach further than payload dispatch.
    T_reb = distances / 15.0
    T_reb[distances > 8000.0] = np.inf
    np.fill_diagonal(T_reb, np.inf)

    # Pre-compute, for each pickup station ps, the sorted list of source
    # stations by ascending T_eff[S, ps]. Truncate at first inf.
    sorted_source_for_pickup = np.argsort(T_eff, axis=0).astype(np.int32)

    base_rate = _compute_base_rate()
    scatter_rates, _, _ = _compute_scatter_rate(
        TARGET_ORDERS_PER_DAY * SCATTER_ORDER_FRACTION, res_score)
    print(f"[sim] target {TARGET_ORDERS_PER_DAY:,d} orders/day "
          f"({HUB_ORDER_FRACTION*100:.0f}% hub + "
          f"{SCATTER_ORDER_FRACTION*100:.0f}% scatter)")
    print(f"[sim] hub BASE = {base_rate:.0f}/hr per weight-unit; "
          f"scatter rate range "
          f"[{scatter_rates.min():.1f}, {scatter_rates.max():.1f}]/hr/station")
    print(f"[sim] fleet: {N_UAVS:,d} UAVs "
          f"({INITIAL_UAVS_PER_STATION}/station × {N_STATIONS} stations, "
          f"cap {MAX_UAVS_PER_STATION}/station)")

    # --- station fleet
    station_idle = np.full(N_STATIONS, INITIAL_UAVS_PER_STATION,
                           dtype=np.int32)
    station_in_flight_to = np.zeros(N_STATIONS, dtype=np.int32)  # arriving here

    # --- order arrays (preallocate)
    MAX_ORDERS = 1_500_000
    arr_circle    = np.zeros(MAX_ORDERS, dtype=np.int32)
    arr_pickup    = np.zeros(MAX_ORDERS, dtype=np.int32)
    arr_customer  = np.zeros(MAX_ORDERS, dtype=np.int32)
    arr_created   = np.full(MAX_ORDERS, -1, dtype=np.float32)
    arr_assigned  = np.full(MAX_ORDERS, -1, dtype=np.float32)
    arr_pickup_t  = np.full(MAX_ORDERS, -1, dtype=np.float32)
    arr_delivered = np.full(MAX_ORDERS, -1, dtype=np.float32)
    arr_status    = np.zeros(MAX_ORDERS, dtype=np.int8)
    # 0 pending, 1 assigned, 2 loading, 3 in_transit, 4 delivered, 5 failed
    n_orders = 0
    pending_q = [[] for _ in range(N_STATIONS)]   # FIFO per pickup station

    # --- time
    t_start, t_end = SIM_HOURS
    t_total = (t_end - t_start) * 3600.0
    n_ticks = int(t_total / TICK_SEC)
    print(f"[sim] {SIM_HOURS[0]}h → {SIM_HOURS[1]}h, "
          f"{n_ticks} ticks of {TICK_SEC:.0f}s")

    # --- logs
    log_pending = np.zeros(n_ticks, dtype=np.int32)
    log_busy = np.zeros(n_ticks, dtype=np.int32)
    log_completed = np.zeros(n_ticks, dtype=np.int32)
    log_failed = np.zeros(n_ticks, dtype=np.int32)
    log_arr_per_circle = np.zeros((n_ticks, len(NANSHAN_CIRCLES)),
                                  dtype=np.int32)
    log_idle_per_station = np.zeros((n_ticks, N_STATIONS), dtype=np.int32)
    log_overflow = np.zeros(n_ticks, dtype=np.int32)   # # stations > capacity
    log_diversions = np.zeros(n_ticks, dtype=np.int32)  # # cap-induced diversions per tick
    log_deadhead_outbound = np.zeros((n_ticks, N_STATIONS), dtype=np.int32)
    log_deadhead_inbound = np.zeros((n_ticks, N_STATIONS), dtype=np.int32)
    log_target_per_station = np.zeros((n_ticks, N_STATIONS), dtype=np.int32)
    log_ema_pickup_rate = np.zeros((n_ticks, N_STATIONS), dtype=np.float32)
    rebalance_history = []   # list of (sim_sec, n_moves)

    # --- predictive rebalance state: EMA of per-station pickup rate (orders/sec)
    ema_pickup_rate = np.zeros(N_STATIONS, dtype=np.float32)
    ema_alpha = TICK_SEC / max(PRED_EMA_TAU_SEC, TICK_SEC)
    pickups_this_tick = np.zeros(N_STATIONS, dtype=np.int32)

    # --- event queue: (time, type_int, payload)
    # type 2 = order_delivered (payload = order_id) — no station effect
    # type 3 = deadhead_arrived (payload = dest_station)
    # type 5 = uav_landed (payload = station)  — fires once per trip,
    #          including the very last stop of a batch
    events: list[tuple] = []
    n_batched_orders = 0
    n_batches = 0
    n_batch_size_count = np.zeros(MAX_BATCH_SIZE + 1, dtype=np.int64)

    n_completed_total = 0
    n_failed_total = 0
    n_deadhead_total = 0
    n_diversions_total = 0
    last_rebalance_sec = -1e9
    t_wall0 = time.time()

    for tick in range(n_ticks):
        sim_sec = float(tick * TICK_SEC)
        sim_h = t_start + sim_sec / 3600.0
        pickups_this_tick.fill(0)

        # 1a. Hub arrivals (7 circles)
        for k, c in enumerate(NANSHAN_CIRCLES):
            rate_per_h = (base_rate * c.weight
                          * demand_profile(sim_h, c.profile))
            n_new = rng.poisson(rate_per_h * TICK_SEC / 3600.0)
            log_arr_per_circle[tick, k] = n_new
            ps = int(circle_to_st[k])
            for _ in range(n_new):
                if n_orders >= MAX_ORDERS: break
                feasible = np.isfinite(T_eff[ps]) & (np.arange(n_st) != ps)
                p = res_score * feasible
                if p.sum() < 1e-9: continue
                cust = int(rng.choice(n_st, p=p / p.sum()))
                arr_circle[n_orders] = k
                arr_pickup[n_orders] = ps
                arr_customer[n_orders] = cust
                arr_created[n_orders] = sim_sec
                arr_status[n_orders] = 0
                pending_q[ps].append(n_orders)
                pickups_this_tick[ps] += 1
                n_orders += 1

        # 1b. Scatter arrivals (every station has small-shop demand)
        scatter_mult = _scatter_profile_value(sim_h)
        scatter_rate_per_st = scatter_rates * scatter_mult
        n_new_scatter = rng.poisson(scatter_rate_per_st * TICK_SEC / 3600.0)
        for ps in range(N_STATIONS):
            n_new = int(n_new_scatter[ps])
            if n_new == 0: continue
            for _ in range(n_new):
                if n_orders >= MAX_ORDERS: break
                feasible = np.isfinite(T_eff[ps]) & (np.arange(n_st) != ps)
                p = res_score * feasible
                if p.sum() < 1e-9: continue
                cust = int(rng.choice(n_st, p=p / p.sum()))
                arr_circle[n_orders] = -1     # scatter, no specific circle
                arr_pickup[n_orders] = ps
                arr_customer[n_orders] = cust
                arr_created[n_orders] = sim_sec
                arr_status[n_orders] = 0
                pending_q[ps].append(n_orders)
                pickups_this_tick[ps] += 1
                n_orders += 1

        # 1c. Update EMA pickup rate (orders/sec) used by predictive rebalance
        rate_now = pickups_this_tick.astype(np.float32) / max(TICK_SEC, 1e-6)
        ema_pickup_rate = ((1.0 - ema_alpha) * ema_pickup_rate
                           + ema_alpha * rate_now)
        log_ema_pickup_rate[tick] = ema_pickup_rate

        # 2. Dispatch (greedy per pickup station, with batching)
        for ps in range(N_STATIONS):
            q = pending_q[ps]
            if not q: continue
            # 2a. Form batches of up to MAX_BATCH_SIZE orders whose
            #     destinations are clustered.
            batches = make_batches(q, arr_customer, T_eff)
            sorted_S = sorted_source_for_pickup[:, ps]
            si = 0
            new_q = []
            for batch in batches:
                # find next idle UAV
                while si < N_STATIONS:
                    S = int(sorted_S[si])
                    if not np.isfinite(T_eff[S, ps]):
                        si = N_STATIONS
                        break
                    if station_idle[S] > 0:
                        break
                    si += 1
                if si >= N_STATIONS:
                    new_q.extend(batch)   # whole batch waits
                    continue
                S = int(sorted_S[si])
                # 2b. assign UAV to this batch
                station_idle[S] -= 1
                t_arrive_ps = sim_sec + float(T_eff[S, ps])
                # Loading time scales with batch size (each item adds load)
                t_loaded = t_arrive_ps + LOAD_SEC * len(batch)
                # 2c. nearest-neighbor sequence through the customers
                cur = ps
                t_curr = t_loaded
                for oid in batch:
                    arr_assigned[oid] = sim_sec
                    arr_pickup_t[oid] = t_arrive_ps
                    arr_status[oid] = 1
                remaining_b = list(batch)
                final_station = ps
                while remaining_b:
                    # pick remaining order whose customer is closest to cur
                    best = remaining_b[0]; best_t = float(T_eff[
                        cur, int(arr_customer[best])])
                    for o in remaining_b[1:]:
                        c = int(arr_customer[o])
                        t_to = float(T_eff[cur, c])
                        if t_to < best_t:
                            best = o; best_t = t_to
                    if not np.isfinite(best_t):
                        # cannot reach this customer from cur — skip,
                        # fail the order (rare, only if dispatch was off)
                        arr_status[best] = 5
                        remaining_b.remove(best)
                        continue
                    c = int(arr_customer[best])
                    t_curr = t_curr + best_t
                    heapq.heappush(events, (t_curr, 2, best))
                    cur = c
                    final_station = c
                    remaining_b.remove(best)
                # UAV becomes idle at the LAST stop
                station_in_flight_to[final_station] += 1
                heapq.heappush(events, (t_curr, 5, final_station))
                if len(batch) > 1:
                    n_batched_orders += len(batch)
                n_batches += 1
                n_batch_size_count[len(batch)] += 1
            pending_q[ps] = new_q

        # 3. Rebalance / deadhead repositioning
        if sim_sec - last_rebalance_sec >= REBALANCE_INTERVAL_SEC:
            if REBALANCE_MODE == "predictive":
                moves, target_now = predictive_rebalance(
                    station_idle, station_in_flight_to,
                    ema_pickup_rate, T_reb, sim_sec,
                    max_moves=MAX_DEADHEADS_PER_REBALANCE,
                    MAX=MAX_UAVS_PER_STATION)
                log_target_per_station[tick] = target_now
            else:
                moves = rebalance_step(station_idle, station_in_flight_to,
                                       pending_q, T_reb, sim_sec,
                                       max_moves=MAX_DEADHEADS_PER_REBALANCE)
            for s_, d_, t_arr in moves:
                heapq.heappush(events, (t_arr, 3, d_))
                log_deadhead_outbound[tick, s_] += 1
                n_deadhead_total += 1
            if moves:
                rebalance_history.append((sim_sec, len(moves)))
            last_rebalance_sec = sim_sec

        # 4. Process arrival events up to end of tick
        tick_end = (tick + 1) * TICK_SEC
        while events and events[0][0] <= tick_end:
            t_evt, type_evt, payload = heapq.heappop(events)
            if type_evt == 2:
                # order delivered at intermediate or final stop
                oid = int(payload)
                arr_delivered[oid] = float(t_evt)
                arr_status[oid] = 4
                n_completed_total += 1
                log_completed[tick] += 1
            elif type_evt == 3:
                d = int(payload)
                final, extra, diverted = land_or_divert(
                    d, station_idle, station_in_flight_to,
                    T_reb, t_evt, MAX_UAVS_PER_STATION, events)
                if diverted:
                    n_diversions_total += 1
                    log_diversions[tick] += 1
                else:
                    log_deadhead_inbound[tick, d] += 1
            elif type_evt == 5:
                # UAV lands at this station (final stop of a trip)
                s = int(payload)
                final, extra, diverted = land_or_divert(
                    s, station_idle, station_in_flight_to,
                    T_reb, t_evt, MAX_UAVS_PER_STATION, events)
                if diverted:
                    n_diversions_total += 1
                    log_diversions[tick] += 1

        # 5. Drop timed-out pending orders
        for ps in range(N_STATIONS):
            new_q = []
            for oid in pending_q[ps]:
                if sim_sec - arr_created[oid] > QUEUE_TIMEOUT_SEC:
                    arr_status[oid] = 5
                    n_failed_total += 1
                    log_failed[tick] += 1
                else:
                    new_q.append(oid)
            pending_q[ps] = new_q

        # 6. Logs
        log_pending[tick] = sum(len(q) for q in pending_q)
        log_busy[tick] = N_UAVS - int(station_idle.sum())
        log_idle_per_station[tick] = station_idle.copy()
        log_overflow[tick] = int((station_idle > MAX_UAVS_PER_STATION).sum())

        if verbose and tick % 30 == 0:
            elapsed = time.time() - t_wall0
            arr_h = log_arr_per_circle[max(tick-30, 0):tick+1].sum() * (
                3600.0 / max(min(31, tick+1) * TICK_SEC, 1))
            print(f"  tick {tick:4d}  t={sim_h:5.2f}h  "
                  f"orders={n_orders:>7,d}  "
                  f"complete={n_completed_total:>7,d}  "
                  f"failed={n_failed_total:>5,d}  "
                  f"pending={log_pending[tick]:>5,d}  "
                  f"busy={log_busy[tick]:>5,d}/{N_UAVS}  "
                  f"rate={arr_h:>6,.0f}/hr  "
                  f"({elapsed:.1f}s)")

    avg_batch = n_orders / max(n_batches, 1)
    print(f"\n[done] {n_orders:,d} orders, "
          f"{n_completed_total:,d} delivered "
          f"({n_completed_total/max(n_orders,1)*100:.1f}%), "
          f"{n_failed_total:,d} failed, "
          f"{n_deadhead_total:,d} deadhead moves, "
          f"{n_diversions_total:,d} cap-induced diversions")
    print(f"[batching] {n_batches:,d} trips, "
          f"{n_batched_orders:,d} orders carried as multi-order trips "
          f"(avg batch size = {avg_batch:.2f})")
    for k in range(1, MAX_BATCH_SIZE + 1):
        cnt = int(n_batch_size_count[k])
        if cnt:
            print(f"  size {k}: {cnt:>7,d} trips  "
                  f"({cnt*100/max(n_batches,1):.1f}% of trips, "
                  f"{cnt*k:>7,d} orders)")

    # Trim to actual orders
    n = n_orders
    out = OUT_DIR / "sim_log.npz"
    np.savez_compressed(
        out,
        sim_t_start=np.float32(t_start),
        sim_t_end=np.float32(t_end),
        tick_sec=np.float32(TICK_SEC),
        n_uavs=np.int32(N_UAVS),
        n_stations=np.int32(N_STATIONS),
        initial_per_station=np.int32(INITIAL_UAVS_PER_STATION),
        max_per_station=np.int32(MAX_UAVS_PER_STATION),
        target_orders_per_day=np.int32(TARGET_ORDERS_PER_DAY),
        base_rate=np.float32(base_rate),
        log_pending=log_pending,
        log_busy=log_busy,
        log_completed=log_completed,
        log_failed=log_failed,
        log_arriving_per_circle=log_arr_per_circle,
        log_idle_per_station=log_idle_per_station,
        log_overflow=log_overflow,
        log_diversions=log_diversions,
        n_diversions_total=np.int32(n_diversions_total),
        log_target_per_station=log_target_per_station,
        log_ema_pickup_rate=log_ema_pickup_rate,
        rebalance_mode=np.array(REBALANCE_MODE),
        log_deadhead_outbound=log_deadhead_outbound,
        log_deadhead_inbound=log_deadhead_inbound,
        rebalance_history=np.array(rebalance_history if rebalance_history
                                   else [[0, 0]], dtype=np.float32),
        n_deadhead_total=np.int32(n_deadhead_total),
        n_batches=np.int32(n_batches),
        n_batched_orders=np.int32(n_batched_orders),
        n_batch_size_count=n_batch_size_count.astype(np.int64),
        order_circle=arr_circle[:n],
        order_pickup_station=arr_pickup[:n],
        order_customer_station=arr_customer[:n],
        order_created_t=arr_created[:n],
        order_assigned_t=arr_assigned[:n],
        order_pickup_t=arr_pickup_t[:n],
        order_delivered_t=arr_delivered[:n],
        order_status_int=arr_status[:n],
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
