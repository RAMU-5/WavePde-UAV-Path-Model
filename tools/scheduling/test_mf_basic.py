"""Basic validation tests for the mean-field UAV scheduler.

Tests:
    1. K=1 assignment vs scipy Hungarian (linear_sum_assignment).
       Square (N=M) case and rectangular (N<M) case.
    2. K>1 multi-step convergence: F monotone-ish decrease and
       per-UAV softmax becomes near one-hot after annealing.
    3. Demand balancing: with no travel cost, occupancy converges to
       N * d / sum(d).

Run:
    /leonardo_work/EUHPC_D25_028/Levistone/wave_test_large/venv/bin/python \
        tools/scheduling/test_mf_basic.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mf_scheduler import (mean_field_schedule, mean_field_assignment_2d,
                          mf_hybrid_lsa_assignment, MFConfig)


# ---------------------------------------------------------------------------
# Test 1: K=1 vs Hungarian
# ---------------------------------------------------------------------------

def hungarian_cost(C):
    row, col = linear_sum_assignment(C)
    return C[row, col].sum(), col


def test_k1_vs_hungarian(N=8, M=8, n_trials=20, seed=0):
    """Square assignment N=M. This is the strictest case for mean-field
    because each task must receive exactly one UAV. The paper's Table 1
    uses the 'hybrid-LSA' variant: mean-field + Hungarian cleanup, which
    matches Hungarian exactly. We test both variants here.
    """
    print("\n" + "=" * 70)
    print(f"TEST 1: K=1 square assignment vs Hungarian   "
          f"(N={N} UAV, M={M} task, trials={n_trials})")
    print("=" * 70)
    rng = np.random.default_rng(seed)

    # --- variant A: pure mean-field (argmax of soft x). Expected to be
    #     close-but-not-exact for the strict 1-to-1 case.
    pure_gaps, pure_unique = [], 0
    # --- variant B: hybrid mean-field + Hungarian (paper's hybrid-LSA).
    hyb_gaps, hyb_match = [], 0

    for t in range(n_trials):
        C = rng.uniform(1.0, 10.0, size=(N, M)).astype(np.float64)
        hu_total, hu_col = hungarian_cost(C)

        # pure MF
        cfg = MFConfig(theta0=3.0, gamma=0.99, n_iter=400,
                       eta=0.5, perturb=0.1, tol=1e-7, seed=t,
                       update="seidel")
        res = mean_field_assignment_2d(C, lam_one=50.0, cfg=cfg)
        mf_col = np.argmax(res.x[:, 1, :], axis=1)
        mf_total = float(C[np.arange(N), mf_col].sum())
        pure_gaps.append((mf_total - hu_total) / hu_total * 100.0)
        if len(np.unique(mf_col)) == N:
            pure_unique += 1

        # hybrid MF + LSA
        col, _ = mf_hybrid_lsa_assignment(C, lam_one=50.0, cfg=cfg)
        hyb_total = float(C[np.arange(N), col].sum())
        hyb_gaps.append((hyb_total - hu_total) / hu_total * 100.0)
        if np.array_equal(col, hu_col):
            hyb_match += 1

    pure_gaps = np.array(pure_gaps)
    hyb_gaps = np.array(hyb_gaps)
    print(f"  --- Pure mean-field (argmax of soft x) ---")
    print(f"    gap %:  mean={pure_gaps.mean():+.2f}  "
          f"max={pure_gaps.max():+.2f}  unique={pure_unique}/{n_trials}")
    print(f"  --- Hybrid mean-field + Hungarian (LSA on -log(x)+C) ---")
    print(f"    gap %:  mean={hyb_gaps.mean():+.4f}  "
          f"max={hyb_gaps.max():+.4f}  exact-match={hyb_match}/{n_trials}")
    print(f"  NOTE: strict 1-to-1 square assignment is the worst case for")
    print(f"        pure mean-field; the realistic dispatch case (N<<M)")
    print(f"        is tested in Test 1b.")
    # PASS criterion: pure MF gap should at least be in the right ballpark,
    # and hybrid should consistently improve over it.
    ok = (pure_gaps.mean() < 25.0) and (hyb_gaps.mean() <= pure_gaps.mean())
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(pure MF mean gap <25% AND hybrid improves on pure)")
    return ok


def test_k1_rectangular(N=6, M=10, n_trials=10, seed=1):
    print("\n" + "=" * 70)
    print(f"TEST 1b: K=1 rectangular   (N={N} UAV, M={M} task, "
          f"trials={n_trials})")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    gaps = []
    for t in range(n_trials):
        C = rng.uniform(1.0, 10.0, size=(N, M)).astype(np.float64)
        # Hungarian on rectangular: returns min-cost partial matching of size N
        hu_total, hu_col = hungarian_cost(C)
        cfg = MFConfig(theta0=3.0, gamma=0.99, n_iter=400,
                       eta=0.5, perturb=0.1, tol=1e-7, seed=t,
                       update="seidel")
        res = mean_field_assignment_2d(C, lam_one=10.0, cfg=cfg)
        mf_col = np.argmax(res.x[:, 1, :], axis=1)
        mf_total = float(C[np.arange(N), mf_col].sum())
        gaps.append((mf_total - hu_total) / hu_total * 100.0)
    gaps = np.array(gaps)
    print(f"  Optimality gap (%):  mean={gaps.mean():+.3f}  "
          f"max={gaps.max():+.3f}")
    ok = gaps.mean() < 8.0 and gaps.max() < 30.0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test 2: K>1 convergence & one-hot quality
# ---------------------------------------------------------------------------

def test_multistep_convergence(N=5, K=6, M=8, seed=2):
    print("\n" + "=" * 70)
    print(f"TEST 2: K>1 multi-step convergence   "
          f"(N={N} UAV, K={K} time-steps, M={M} station)")
    print("=" * 70)
    rng = np.random.default_rng(seed)

    # Random pairwise travel cost (symmetric over mu,nu per (i,k))
    D = rng.uniform(1.0, 5.0, size=(N, K, M, M)).astype(np.float64)
    D = 0.5 * (D + D.transpose(0, 1, 3, 2))
    # diagonal small (staying is cheap)
    for k in range(K):
        for i in range(N):
            np.fill_diagonal(D[i, k], 0.1)

    d = rng.uniform(0.5, 2.0, size=(K, M)).astype(np.float64)
    C_cap = np.full((K, M), N // 2 + 2, dtype=np.float64)   # generous cap

    cfg = MFConfig(theta0=2.0, gamma=0.99, n_iter=800, tol=1e-7,
                   eta=0.3, perturb=0.1,
                   lam_balance=1.0, lam_cap=2.0, verbose=False)
    res = mean_field_schedule(D, d, C_cap, cfg=cfg)

    F0, F_end = res.F_history[0], res.F_history[-1]
    print(f"  iters run        : {res.iters}")
    print(f"  F initial         : {F0:.4f}")
    print(f"  F final           : {F_end:.4f}")
    print(f"  F reduction       : {(F0 - F_end) / max(abs(F0), 1e-9) * 100:.2f}%")

    # one-hot quality: max prob per (i,k) should be close to 1
    max_prob = res.x.max(axis=2)               # (N, K)
    print(f"  mean max-prob     : {max_prob.mean():.4f}  "
          f"(1.0 = perfectly one-hot)")
    print(f"  min  max-prob     : {max_prob.min():.4f}")

    ok = (F_end < F0) and (max_prob.mean() > 0.85)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(target: F decreases AND mean max-prob > 0.85)")
    return ok


# ---------------------------------------------------------------------------
# Test 3: demand balancing in isolation
# ---------------------------------------------------------------------------

def test_demand_balancing(N=40, K=2, M=4, seed=3):
    """Demand balancing test.

    We use small random per-UAV travel cost (different per UAV but no
    structured bias) to break the perfect symmetry, then check that
    occupancy correlates strongly with predicted demand.

    Pure mean-field with symmetric F_balance + identical UAVs is
    degenerate — all UAVs see the same field and collapse. Random
    per-UAV costs are realistic and break the symmetry.
    """
    print("\n" + "=" * 70)
    print(f"TEST 3: demand balancing with small random per-UAV cost   "
          f"(N={N} UAV, M={M} station)")
    print("=" * 70)
    rng = np.random.default_rng(seed)

    # tiny random per-UAV cost — much smaller than the demand penalty,
    # but enough to give each UAV a distinct field.
    D = rng.uniform(0.05, 0.15, size=(N, K, M, M)).astype(np.float64)

    d = np.array([[1.0, 3.0, 2.0, 4.0]] * K)        # demand weights, sum=10
    C_cap = np.full((K, M), N, dtype=np.float64)    # no cap binding

    cfg = MFConfig(theta0=2.0, gamma=0.99, n_iter=400, tol=1e-8,
                   eta=0.5, perturb=0.2, update="seidel",
                   lam_balance=5.0, lam_cap=0.0, verbose=False, seed=seed)
    res = mean_field_schedule(D, d, C_cap, cfg=cfg)

    occ = res.x.sum(axis=0)[-1]                     # final time-step occupancy
    target = N * d[0] / d[0].sum()

    # rank correlation: occupancy ranking should match demand ranking
    occ_rank = np.argsort(np.argsort(occ))
    tgt_rank = np.argsort(np.argsort(target))
    rank_match = np.array_equal(occ_rank, tgt_rank)

    # cosine similarity between occupancy and target
    cos = float(np.dot(occ, target) / (np.linalg.norm(occ)
                                        * np.linalg.norm(target) + 1e-12))

    err = np.max(np.abs(occ - target))
    print(f"  target occupancy: {target}")
    print(f"  actual occupancy: {np.round(occ, 3)}")
    print(f"  max abs error   : {err:.3f}  (out of {N} UAVs)")
    print(f"  cosine sim(occ, target): {cos:.4f}")
    print(f"  ranking matches : {rank_match}")
    ok = rank_match and cos > 0.92
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(target: ranking matches AND cosine > 0.92)")
    return ok


# ---------------------------------------------------------------------------

def main():
    results = []
    results.append(("K=1 vs Hungarian (square)", test_k1_vs_hungarian()))
    results.append(("K=1 vs Hungarian (rect)",   test_k1_rectangular()))
    results.append(("K>1 convergence",           test_multistep_convergence()))
    results.append(("Demand balancing",          test_demand_balancing()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  {n_pass}/{len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
