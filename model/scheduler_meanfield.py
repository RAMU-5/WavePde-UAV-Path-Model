"""Paper v3 §5: Mean-field neural network scheduler with simulated annealing.

Given a cost matrix C[i, j] (UAV i → task j travel cost — typically the
wave-derived T(x) field at the UAV's location read from the j-th task's
backward-wave solve), find an assignment x[i, j] ∈ [0, 1] that minimises:

    F = ½ Σ_μ x_i^μ C_ij x_j^μ + (λ/2) Σ_μ (Σ_i x_i^μ - N/M)²

where μ indexes tasks, i indexes UAVs. The first term penalises high-cost
assignments (note the paper's "interaction" framing where C_ij is between
*UAVs* — we use the simpler diagonal form C_iμ in practice). The second
term enforces equal load across tasks.

Iterative Boltzmann update with annealing T (Eq. 52):

    f_i^μ = ∂F / ∂x_i^μ = C_iμ + λ·(Σ_k x_k^μ - N/M)
    x_i^μ = exp(-f_i^μ / T) / Σ_ν exp(-f_i^ν / T)

T anneals from T_0 (≈ Eq. 56 critical T) → T_min, then we take argmax over μ
to get a hard assignment.

For our 118-station Nanshan demo:
  - "UAV" = a station (warehouse) where a drone is parked
  - "task" = a delivery destination station
  - C_iμ = T_matrix[i, μ] from wave solve

This script can also handle the M < N (over-supply) and M > N
(over-demand, using K_i capacity) regimes.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def critical_temperature(C):
    """Paper Eq. 56: T_c = mean(C). Standard reference scale."""
    return float(np.mean(C[np.isfinite(C)]))


def mean_field_assign(
    C,                          # (N, M) cost — UAV i to task μ
    K=None,                     # (N,) per-UAV capacity, default = ceil(M/N)
    lam=1.0,                    # load-balance weight
    T0=None,                    # initial temperature (None → critical T)
    T_min=None,                 # final temperature (None → 0.01·T0)
    n_iter=300,                 # annealing steps
    seed=0,
    hard_step="soft_lsa",       # 'soft_lsa' | 'hybrid_lsa' | 'greedy'
    alpha=1.0,                  # weight on -log(x) in hybrid_lsa
    verbose=False,
):
    """Mean-field annealing for the assignment problem."""
    rng = np.random.default_rng(seed)
    N, M = C.shape
    finite = np.isfinite(C)
    if not finite.any():
        raise ValueError("All cost entries are infeasible")
    C_finite = C[finite]
    # T_0 = std of finite costs (≈ critical T for the feasible block).
    # Earlier we used "100·max" as inf penalty which blew up T0.
    if T0 is None:
        T0 = float(np.std(C_finite)) if C_finite.size > 1 else 1.0
        T0 = max(T0, 1.0)
    if T_min is None:
        T_min = 0.01 * T0
    # Effective C used inside the softmax: keep finite entries, mask
    # infeasible with a moderate penalty so softmax is defined but f-inf
    # entries are heavily suppressed at any T < penalty.
    C_eff = np.where(finite, C, 10.0 * float(C_finite.max()))

    if K is None:
        K = np.full(N, max(1, int(np.ceil(M / N))))

    # Initialise uniformly
    x = np.full((N, M), 1.0 / M)

    # Per-task target: M tasks distributed across N UAVs ⇒ each UAV gets
    # M/N tasks on average. Per-task: each task should be served by ~1
    # UAV (Σ_i x_i^μ = 1).
    target_per_task = 1.0

    schedule_T = T0 * (T_min / T0) ** (np.arange(n_iter) / max(n_iter - 1, 1))

    history = []
    for step, T in enumerate(schedule_T):
        # f_i^μ = C_iμ + λ · (Σ_k x_k^μ - target_per_task)
        col_sum = x.sum(axis=0, keepdims=True)        # (1, M)
        f = C_eff + lam * (col_sum - target_per_task) # (N, M)
        # capacity penalty: each UAV picks ≤ K_i tasks → softmax over μ
        # naturally ensures Σ_μ x_i^μ = 1 already; for K_i > 1 we relax to
        # picking the K_i lowest-f tasks per UAV (handled at the end).
        x_new = softmax(-f / max(T, 1e-9), axis=1)
        x = x_new

        if verbose and (step % max(n_iter // 10, 1) == 0):
            cost = (x * C_eff).sum()
            ent = -(x * np.log(np.clip(x, 1e-12, 1.0))).sum()
            history.append((step, float(T), float(cost), float(ent)))
            print(f"  step {step:4d}  T={T:7.3f}  cost={cost:.1f}  H={ent:.2f}")

    # ---- hard assignment ----
    # For each UAV, take the K_i tasks with highest x. But we also want
    # each task served by exactly one UAV (Σ_i x_iμ = 1). Use Hungarian
    # for the unconstrained K_i = 1 case, otherwise greedy.
    assign = np.full(M, -1, dtype=int)   # task μ → UAV i
    if (K == 1).all() and hard_step in ("soft_lsa", "hybrid_lsa"):
        from scipy.optimize import linear_sum_assignment
        if hard_step == "soft_lsa":
            # original: pick by mean-field confidence only
            lsa_cost = -np.log(np.clip(x, 1e-12, 1.0))
        else:  # hybrid_lsa: feasibility-priority + soft tie-break
            # Hungarian on real C dominates; -log(x) only breaks ties
            # among equally-cheap feasible options. α controls the mix.
            lsa_cost = C_eff + alpha * (
                -np.log(np.clip(x, 1e-12, 1.0)))
        ri, cj = linear_sum_assignment(lsa_cost.T)
        for mu, i in zip(ri, cj):
            if mu < M and i < N:
                assign[mu] = i
    else:
        # Greedy with capacity K_i > 1 (or generic fallback)
        load = np.zeros(N, dtype=int)
        order = np.argsort(-x.max(axis=0))   # most-confident task first
        for mu in order:
            ranking = np.argsort(-x[:, mu])
            for i in ranking:
                if load[i] < K[i]:
                    assign[mu] = int(i)
                    load[i] += 1
                    break
    return dict(x_soft=x, assignment=assign, K=K,
                T0=T0, T_min=T_min, n_iter=n_iter, history=history)


def evaluate(C, assignment):
    """Sum of costs along the assignment vector (task μ → UAV i)."""
    M = len(assignment)
    sel = []
    for mu in range(M):
        i = assignment[mu]
        if i < 0:
            sel.append(np.inf)
        else:
            sel.append(C[i, mu])
    arr = np.asarray(sel)
    finite = np.isfinite(arr)
    return dict(total_finite=float(arr[finite].sum()),
                n_assigned=int(finite.sum()),
                n_unassigned=int(M - finite.sum()),
                max_cost=float(arr[finite].max()) if finite.any() else np.inf,
                mean_cost=float(arr[finite].mean()) if finite.any() else np.inf)


# ---- baselines for comparison --------------------------------------------

def hungarian_assign(C):
    """Paper §4: classical assignment via Hungarian on the same C matrix."""
    from scipy.optimize import linear_sum_assignment
    finite = np.isfinite(C)
    C_eff = np.where(finite, C, 1e9)
    # Square: pad with ghosts if needed
    N, M = C.shape
    row_ind, col_ind = linear_sum_assignment(C_eff)
    assign = np.full(M, -1, dtype=int)
    for i, mu in zip(row_ind, col_ind):
        if mu < M and finite[i, mu]:
            assign[mu] = int(i)
    return assign


def greedy_min_assign(C):
    """Sort all (i, μ) by C; pick smallest, mark UAV/task busy."""
    finite = np.isfinite(C)
    N, M = C.shape
    pairs = [(C[i, mu], i, mu) for i in range(N) for mu in range(M)
             if finite[i, mu]]
    pairs.sort()
    assign = np.full(M, -1, dtype=int)
    used_i = set()
    for c, i, mu in pairs:
        if assign[mu] == -1 and i not in used_i:
            assign[mu] = i
            used_i.add(i)
    return assign


# ---- demo on our 118-station pair_matrix -----------------------------------

def demo():
    p = THIS_DIR.parent / "output" / "wave" / "pair_matrix.npz"
    if not p.exists():
        print(f"[demo] {p} not found — run aggregate_pair_matrix.py first")
        return
    z = np.load(p, allow_pickle=False)
    T_full = z["T_matrix"]
    valid = z["valid_mask"]
    n = T_full.shape[0]

    # Pretend the first N_uavs stations are warehouses (have a drone),
    # the last M_tasks stations are delivery destinations needing service.
    N_uavs = 30
    M_tasks = 30
    uav_idx = np.arange(N_uavs)
    task_idx = np.arange(n - M_tasks, n)
    C = T_full[np.ix_(uav_idx, task_idx)]
    valid_C = valid[np.ix_(uav_idx, task_idx)]
    C = np.where(valid_C, C, np.inf)

    print(f"[demo] cost matrix C: {C.shape}, "
          f"finite frac {np.isfinite(C).mean()*100:.0f}%, "
          f"min={np.nanmin(C[np.isfinite(C)]):.1f}s "
          f"median={np.nanmedian(C[np.isfinite(C)]):.1f}s "
          f"max={np.nanmax(C[np.isfinite(C)]):.1f}s")

    def report(name, a):
        e = evaluate(C, a)
        n_ass = e["n_assigned"]
        avg = e["total_finite"] / max(n_ass, 1)
        print(f"  {name:<28}  Σ={e['total_finite']:7.1f}s  "
              f"{n_ass}/{M_tasks} done  avg={avg:5.1f}s/task  "
              f"max={e['max_cost']:.0f}s")
        return a, e

    print("\n[Method 0  — mean-field, soft-only LSA  (original)]")
    r0 = mean_field_assign(C, lam=1.0, n_iter=2000, T_min=0.1,
                           hard_step="soft_lsa")
    a0, e0 = report("Mean-field (soft_lsa)", r0["assignment"])

    print("\n[Method A — mean-field + hybrid LSA  (feasibility-first)]")
    r_a = mean_field_assign(C, lam=1.0, n_iter=2000, T_min=0.1,
                            hard_step="hybrid_lsa", alpha=0.5)
    a_a, e_a = report("Mean-field (hybrid α=0.5)", r_a["assignment"])

    print("\n[Method C — mean-field with K=2 (each UAV does up to 2 tasks)]")
    K2 = np.full(C.shape[0], 2)
    r_c = mean_field_assign(C, K=K2, lam=1.0, n_iter=2000, T_min=0.1)
    a_c, e_c = report("Mean-field K=2", r_c["assignment"])

    print("\n[Baselines]")
    a_hu, e_hu = report("Hungarian", hungarian_assign(C))
    a_gr, e_gr = report("Greedy", greedy_min_assign(C))

    out = THIS_DIR.parent / "output" / "wave" / "scheduler_demo.npz"
    np.savez_compressed(
        out, C=C,
        assignment_meanfield_soft=a0,
        assignment_meanfield_hybrid=a_a,
        assignment_meanfield_K2=a_c,
        assignment_hungarian=a_hu,
        assignment_greedy=a_gr,
        x_soft=r0["x_soft"], x_hybrid=r_a["x_soft"], x_K2=r_c["x_soft"],
        uav_idx=uav_idx, task_idx=task_idx,
    )
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        # Tiny self-test
        rng = np.random.default_rng(0)
        N, M = 8, 8
        C = rng.uniform(10, 100, size=(N, M))
        # Plant an obvious near-diagonal optimum
        for i in range(min(N, M)): C[i, i] *= 0.3
        res = mean_field_assign(C, lam=1.0, n_iter=200, verbose=True)
        print(f"[selftest] mean-field assign: {res['assignment']}")
        print(f"           Hungarian:        {hungarian_assign(C)}")
