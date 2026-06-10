"""Integration test: mean-field scheduler vs greedy baseline on a small,
self-contained order-system simulation using the real pair_matrix.npz
travel-time matrix.

Each dispatch tick:
  - some idle UAVs are sitting at various stations
  - some pending orders have (pickup_station, customer_station)
  - we solve a global rectangular assignment:
        cost[i, j] = T_eff[uav_i_station, order_j_pickup]
                   + T_eff[order_j_pickup, order_j_customer]
    (the second term is identical for the order regardless of UAV, so it
    only shifts costs uniformly per-order — we still include it for
    realistic absolute values.)
  - infeasible pairs (no finite travel time within 4 km dispatch radius)
    are masked with a large finite cost so MF can still process them but
    they're never preferred.

Compared methods:
  - GREEDY: for each pending order in FIFO order, take the closest
            reachable idle UAV.  (Same heuristic as simulation.py:25-27.)
  - MF    : single global mean-field hybrid-LSA assignment.

KPIs reported:
  - # orders dispatched / # orders pending
  - mean & total dispatch travel cost (seconds)
  - mean wait time (sim seconds from order creation to UAV departure)
  - wallclock solve time

Run:
    /leonardo_work/EUHPC_D25_028/Levistone/venv/bin/python \
        tools/scheduling/test_mf_dispatch.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

from mf_scheduler import (mean_field_schedule, mf_hybrid_lsa_assignment,
                          MFConfig)

PAIR_MATRIX = ROOT / "output" / "wave" / "pair_matrix.npz"
N_STATIONS = 118
DISPATCH_RADIUS_M = 4000.0       # same as simulation.py


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_T_eff():
    pm = np.load(PAIR_MATRIX, allow_pickle=False)
    T = pm["T_matrix"].astype(np.float32)
    dist = pm["distances_m"].astype(np.float32)
    fallback = dist / 15.0
    T_eff = np.where(np.isfinite(T), T, fallback)
    T_eff[dist > DISPATCH_RADIUS_M] = np.inf
    np.fill_diagonal(T_eff, np.inf)
    return T_eff, dist


# ---------------------------------------------------------------------------
# Synthetic order/fleet generator
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    uav_station: np.ndarray   # (N_uav,) station of each idle UAV
    order_pickup: np.ndarray  # (N_ord,) pickup station of each pending order
    order_cust:   np.ndarray  # (N_ord,) customer station of each order
    order_age:    np.ndarray  # (N_ord,) seconds already waited
    T_eff:        np.ndarray  # (118, 118)


def generate_scene(n_uav, n_orders, T_eff, seed=0, max_age=600.0):
    rng = np.random.default_rng(seed)
    # UAVs distributed across stations (some clustering allowed)
    uav_station = rng.integers(0, N_STATIONS, size=n_uav)
    # Orders: pickup at a station with at least one reachable neighbour
    feasible_stations = [s for s in range(N_STATIONS)
                         if np.isfinite(T_eff[s]).any()]
    pickup = rng.choice(feasible_stations, size=n_orders)
    cust = np.zeros(n_orders, dtype=np.int32)
    for j, ps in enumerate(pickup):
        reach = np.where(np.isfinite(T_eff[ps]) &
                         (np.arange(N_STATIONS) != ps))[0]
        cust[j] = rng.choice(reach)
    age = rng.uniform(0, max_age, size=n_orders)
    return Scene(uav_station=uav_station, order_pickup=pickup,
                 order_cust=cust, order_age=age, T_eff=T_eff)


# ---------------------------------------------------------------------------
# Cost matrix builder
# ---------------------------------------------------------------------------

INFEASIBLE_COST = 1e6        # large finite to keep MF numerically happy


def build_cost(scene: Scene):
    """Return C[i, j] = travel time UAV_i_station -> order_j_pickup
                       + travel time pickup -> customer.
    Infeasible pairs (T=inf) get INFEASIBLE_COST so they're soft-blocked.
    """
    N = len(scene.uav_station)
    M = len(scene.order_pickup)
    T = scene.T_eff
    # leg1: UAV station -> pickup station
    leg1 = T[scene.uav_station[:, None], scene.order_pickup[None, :]]  # (N,M)
    # leg2: pickup -> customer (constant per order)
    leg2 = T[scene.order_pickup, scene.order_cust]                     # (M,)
    C = leg1 + leg2[None, :]
    C = np.where(np.isfinite(C), C, INFEASIBLE_COST)
    return C, np.isfinite(leg1)


# ---------------------------------------------------------------------------
# Greedy baseline (matches simulation.py logic, rewritten standalone)
# ---------------------------------------------------------------------------

def greedy_dispatch(scene: Scene):
    """For each pending order in (age-descending) order, take the closest
    reachable idle UAV. Each UAV used at most once."""
    N = len(scene.uav_station)
    M = len(scene.order_pickup)
    C, feas = build_cost(scene)
    # serve oldest orders first
    order_idx = np.argsort(-scene.order_age)
    used_uav = np.zeros(N, dtype=bool)
    assign = np.full(M, -1, dtype=np.int32)
    for j in order_idx:
        # candidates: not used AND feasible
        cands = np.where((~used_uav) & feas[:, j])[0]
        if len(cands) == 0:
            continue
        best = cands[np.argmin(C[cands, j])]
        assign[j] = best
        used_uav[best] = True
    return assign, C


# ---------------------------------------------------------------------------
# Mean-field dispatch
# ---------------------------------------------------------------------------

def mf_dispatch(scene: Scene, lam_one=200.0, theta0=50.0,
                gamma=0.99, n_iter=300, eta=0.4, seed=0):
    """Global mean-field assignment + Hungarian cleanup (hybrid-LSA).

    We add a sentinel "no-op" task (column M) with cost 0 so excess UAVs
    can be left idle. After LSA, any UAV assigned to the sentinel is
    treated as unassigned.
    """
    N = len(scene.uav_station)
    M = len(scene.order_pickup)
    C, feas = build_cost(scene)
    # Sentinel column: any UAV picking it = remain idle.
    sentinel_cost = INFEASIBLE_COST * 0.5   # large but below INFEASIBLE,
                                            # so feasible orders preferred
    Cext = np.hstack([C, np.full((N, 1), sentinel_cost)])

    cfg = MFConfig(theta0=theta0, gamma=gamma, n_iter=n_iter,
                   eta=eta, perturb=0.1, tol=1e-7, seed=seed,
                   update="seidel")
    col, _ = mf_hybrid_lsa_assignment(Cext, lam_one=lam_one, cfg=cfg)
    # Build (M,) assignment: order j -> uav i, -1 if unassigned
    assign = np.full(M, -1, dtype=np.int32)
    for i, j in enumerate(col):
        if j < M and C[i, j] < INFEASIBLE_COST:
            assign[j] = i
    return assign, C


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(assign, C, scene, name=""):
    M = len(assign)
    served = assign >= 0
    n_served = int(served.sum())
    n_pending = M - n_served
    if n_served > 0:
        costs = C[assign[served], np.where(served)[0]]
        total_cost = float(costs.sum())
        mean_cost = float(costs.mean())
        # waiting time = age at dispatch + leg1 travel
        leg1_times = scene.T_eff[scene.uav_station[assign[served]],
                                 scene.order_pickup[served]]
        wait_to_pickup = scene.order_age[served] + leg1_times
        mean_wait = float(np.mean(wait_to_pickup))
    else:
        total_cost = mean_cost = mean_wait = 0.0
    return {
        "name": name,
        "served": n_served,
        "pending": n_pending,
        "served_pct": 100.0 * n_served / max(M, 1),
        "total_cost_s": total_cost,
        "mean_cost_s": mean_cost,
        "mean_wait_s": mean_wait,
    }


def print_kpis(kpis):
    print(f"  [{kpis['name']:8s}]  served {kpis['served']:3d}  "
          f"({kpis['served_pct']:5.1f}%)  "
          f"total cost {kpis['total_cost_s']:8.1f}s  "
          f"mean cost {kpis['mean_cost_s']:5.1f}s  "
          f"mean wait {kpis['mean_wait_s']:5.1f}s")


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def run_scenario(n_uav, n_orders, T_eff, seed=0, label=""):
    scene = generate_scene(n_uav, n_orders, T_eff, seed=seed)
    print(f"\n--- Scenario {label}: N_uav={n_uav}  N_order={n_orders}  "
          f"seed={seed} ---")

    t0 = time.time(); ga, gC = greedy_dispatch(scene); gt = time.time() - t0
    t0 = time.time(); ma, mC = mf_dispatch(scene, seed=seed); mt = time.time() - t0

    kg = evaluate(ga, gC, scene, "GREEDY")
    km = evaluate(ma, mC, scene, "MF")
    print_kpis(kg)
    print_kpis(km)
    cost_delta_pct = ((km["total_cost_s"] - kg["total_cost_s"])
                       / max(kg["total_cost_s"], 1e-9) * 100.0)
    serve_delta = km["served"] - kg["served"]
    print(f"  Δ:  served {serve_delta:+d}  "
          f"total cost {cost_delta_pct:+.2f}%   "
          f"wallclock greedy={gt*1000:.1f}ms  mf={mt*1000:.1f}ms")
    return kg, km


def main():
    print("Loading travel-time matrix...")
    T_eff, _ = load_T_eff()
    print(f"  T_eff shape {T_eff.shape}, "
          f"{np.isfinite(T_eff).sum()}/{T_eff.size} feasible pairs")

    print("\n" + "=" * 72)
    print(" Integration test: mean-field dispatcher vs greedy baseline")
    print("=" * 72)

    scenarios = [
        # (label, n_uav, n_orders)
        ("A: UAV surplus",    30, 15),     # plenty of UAVs
        ("B: balanced",       20, 20),     # 1:1
        ("C: order surplus",  10, 30),     # too many orders
        ("D: realistic peak", 60, 80),     # near-peak demand pressure
    ]

    summary = []
    for label, nU, nO in scenarios:
        kg, km = run_scenario(nU, nO, T_eff, seed=0, label=label)
        summary.append((label, kg, km))

    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"  {'scenario':22s}  {'greedy cost':>12s}  {'mf cost':>10s}  "
          f"{'Δ%':>8s}  {'greedy served':>13s}  {'mf served':>10s}")
    for label, kg, km in summary:
        delta = ((km["total_cost_s"] - kg["total_cost_s"])
                  / max(kg["total_cost_s"], 1e-9) * 100.0)
        print(f"  {label:22s}  {kg['total_cost_s']:12.1f}  "
              f"{km['total_cost_s']:10.1f}  {delta:+7.2f}%  "
              f"{kg['served']:8d}/{kg['served']+kg['pending']:<4d}  "
              f"{km['served']:5d}/{km['served']+km['pending']:<4d}")

    # smoke test: MF should at least never serve dramatically fewer orders
    # than greedy (within 10%) and total cost should be within 50% of greedy.
    all_ok = True
    for label, kg, km in summary:
        if km["served"] < 0.9 * kg["served"] - 1:
            print(f"  WARN: MF served too few in {label}")
            all_ok = False
        if (kg["served"] > 0 and
            km["total_cost_s"] > 1.5 * kg["total_cost_s"]):
            print(f"  WARN: MF cost > 1.5x greedy in {label}")
            all_ok = False
    print(f"\n  Overall: {'PASS' if all_ok else 'WARN'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
