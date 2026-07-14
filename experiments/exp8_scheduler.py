"""E5.1 — scheduler comparison on the real wave cost matrix
(paper Q4 / Table 5, §5.6 system level).

Instances: N UAVs and N orders placed at stations sampled WITH
replacement from the 118 Nanshan stations (multiple UAVs/orders per
station is the realistic city regime); cost C[i,j] =
T_matrix_v2[station(uav_i), station(order_j)], non-finite -> inf.

Algorithms (model/scheduler_meanfield.py):
  hungarian   scipy linear_sum_assignment on the penalised matrix
              (exact optimum of the assignment LP)
  greedy      global cheapest-pair first (vectorised re-implementation
              of greedy_min_assign — identical output ordering)
  mf_soft     Boltzmann annealing, -log(x) LSA readout (paper's
              original Eq. 52 pipeline)
  mf_hybrid   same annealing, feasibility-first readout
              C + alpha*(-log x), alpha = 0.5

Objective: sum over ALL orders, an unassigned order costs
P = 2 * max finite cost of the instance ("give up" penalty) — without
this an algorithm that strands orders scores a spuriously LOW total
(evaluate.total_finite only sums assigned ones).

Metrics per (size, seed): penalised total, gap vs Hungarian, wall
time, unassigned count. Outputs output/runs/exp8_scheduler/results.json.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from model.scheduler_meanfield import (  # noqa: E402
    mean_field_assign, hungarian_assign, evaluate)

OUT_DIR = ROOT / "output" / "runs" / "exp8_scheduler"

SIZES = [10, 30, 100, 300, 1000, 3000, 5000]
SEEDS = {10: 3, 30: 3, 100: 3, 300: 3, 1000: 3, 3000: 1, 5000: 1}
MF_ITER = 600


def penalised_objective(C, assign, P):
    """Sum over ALL orders; unassigned or infeasible costs P."""
    M = len(assign)
    c = np.array([C[assign[m], m] if assign[m] >= 0 else np.inf
                  for m in range(M)])
    return float(np.where(np.isfinite(c), c, P).sum())


def greedy_min_assign_fast(C):
    """Vectorised global-cheapest-first; same semantics as
    scheduler_meanfield.greedy_min_assign (which sorts a Python list
    of N*M tuples and is too slow beyond ~1000^2)."""
    N, M = C.shape
    flat = np.where(np.isfinite(C), C, np.inf).ravel()
    order = np.argsort(flat, kind="stable")
    assign = np.full(M, -1, dtype=int)
    used_i = np.zeros(N, dtype=bool)
    n_done = 0
    for o in order:
        if not np.isfinite(flat[o]):
            break
        i, mu = divmod(int(o), M)
        if assign[mu] == -1 and not used_i[i]:
            assign[mu] = i
            used_i[i] = True
            n_done += 1
            if n_done == min(N, M):
                break
    return assign


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "output" / "wave" / "pair_matrix_5m_v2.npz")
    T = z["T_matrix"]
    n_st = T.shape[0]

    # warm up scipy (first linear_sum_assignment call pays the import)
    hungarian_assign(np.eye(3))

    rows = []
    for size in SIZES:
        for seed in range(SEEDS[size]):
            rng = np.random.default_rng(1000 * size + seed)
            uav_st = rng.integers(0, n_st, size)
            ord_st = rng.integers(0, n_st, size)
            C = T[np.ix_(uav_st, ord_st)].astype(np.float64)
            C[~np.isfinite(C)] = np.inf
            P = 2.0 * float(np.nanmax(C[np.isfinite(C)]))

            algos = {}
            t0 = time.time()
            a_h = hungarian_assign(C)
            algos["hungarian"] = (a_h, time.time() - t0)

            t0 = time.time()
            a_g = greedy_min_assign_fast(C)
            algos["greedy"] = (a_g, time.time() - t0)

            t0 = time.time()
            r = mean_field_assign(C, lam=1.0, n_iter=MF_ITER,
                                  seed=seed, hard_step="soft_lsa")
            algos["mf_soft"] = (r["assignment"], time.time() - t0)

            t0 = time.time()
            r = mean_field_assign(C, lam=1.0, n_iter=MF_ITER,
                                  seed=seed, hard_step="hybrid_lsa",
                                  alpha=0.5)
            algos["mf_hybrid"] = (r["assignment"], time.time() - t0)

            ref = penalised_objective(C, a_h, P)
            for name, (a, wall) in algos.items():
                tot = penalised_objective(C, a, P)
                e = evaluate(C, a)
                rows.append(dict(
                    size=size, seed=seed, algo=name,
                    total_s=tot,
                    gap_vs_hungarian=(tot - ref) / ref if ref > 0
                    else None,
                    n_unassigned=e["n_unassigned"],
                    wall_s=wall))
                print(f"[{size:5d} s{seed} {name:10s}] "
                      f"sum {tot:10.1f}s "
                      f"gap {rows[-1]['gap_vs_hungarian']:+.4f} "
                      f"unass {e['n_unassigned']:3d} "
                      f"wall {wall:8.2f}s", flush=True)

    summary = {}
    for size in SIZES:
        for name in ("hungarian", "greedy", "mf_soft", "mf_hybrid"):
            rs = [r for r in rows if r["size"] == size
                  and r["algo"] == name]
            summary[f"{size}_{name}"] = dict(
                med_gap=float(np.median([r["gap_vs_hungarian"]
                                         for r in rs])),
                med_wall_s=float(np.median([r["wall_s"] for r in rs])),
                max_unassigned=int(max(r["n_unassigned"] for r in rs)))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(dict(rows=rows, summary=summary, mf_iter=MF_ITER,
                       git_sha=sha), f, indent=2)
    print(f"[done] -> {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
