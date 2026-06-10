"""Mean-Field UAV Scheduler — minimal implementation of Eqs 44 + 45.

For strict 1-to-1 assignment (square N=M case), the paper uses a
"hybrid-LSA" variant: mean-field produces a soft assignment x, and the
Hungarian algorithm (linear_sum_assignment) is run on -log(x + eps) to
extract the exact bijection. See `mf_hybrid_lsa_assignment` below.


Reference: neural_physics_uav_shedualer_optimisation/main_article.tex,
"Definition of the mean-field" and "Mean-field equations" sections.

Decision variable:
    x[i, k, mu]  in [0, 1]   — soft assignment of UAV i to location mu
                                at time step k. Normalised so
                                sum_mu x[i, k, mu] = 1 by softmax (Eq 45).

Functional (minimised):
    F = F_tbc  + F_balance + F_cap
      where
        F_tbc     : travel cost / benefit  (uses pairwise cost D[i,k,nu,mu])
        F_balance : pulls station occupancy toward predicted demand d[k,mu]
        F_cap     : penalises exceeding station capacity C_cap[k,mu]

Algorithm (per outer iteration):
    1. f = grad_F(x)                   # Eq 44
    2. x = softmax(-f / theta, mu)     # Eq 45
    3. theta <- gamma * theta          # annealing
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _softmax_neg(f: np.ndarray, theta: float, axis: int) -> np.ndarray:
    """softmax(-f/theta) along `axis`, numerically stable."""
    z = -f / max(theta, 1e-12)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Functional and gradient
# ---------------------------------------------------------------------------

def compute_F(x, D, d, C_cap, s,
              lam_balance: float, lam_cap: float) -> float:
    """Total functional F = F_tbc + F_balance + F_cap.

    Shapes:
      x      : (N, K, M)
      D      : (N, K, M, M)   D[i, k, nu, mu] = cost from nu (at k) to mu (at k+1)
      d      : (K, M)         predicted demand
      C_cap  : (K, M)
      s      : (N, K, M)      availability switch (0/1)
    """
    N, K, M = x.shape
    y = s * x

    # F_tbc = sum_{i,k} sum_{nu,mu} D[i,k,nu,mu] x[i,k,nu] x[i,k+1,mu]
    # Only k = 0..K-2 contribute (no k+1 for k=K-1).
    F_tbc = 0.0
    if K >= 2:
        # einsum over (nu, mu) per (i, k)
        F_tbc = float(np.einsum(
            "iknm,ikn,ikm->",
            D[:, :-1], x[:, :-1], x[:, 1:]
        ))

    # F_balance = (lam_balance / 2) * sum_{k,mu} (sum_j y[j,k,mu] - N*d_hat)^2
    occ = y.sum(axis=0)                          # (K, M)
    d_norm = d / np.maximum(d.sum(axis=1, keepdims=True), 1e-12)
    target = N * d_norm
    F_balance = 0.5 * lam_balance * float(np.sum((occ - target) ** 2))

    # F_cap = (lam_cap / 2) * sum_{k,mu} max(occ - C_cap, 0)^2
    # (we use one-sided quadratic; gradient still matches Eq 43 when occ > C_cap)
    over = np.maximum(occ - C_cap, 0.0)
    F_cap = 0.5 * lam_cap * float(np.sum(over ** 2))

    return F_tbc + F_balance + F_cap


def compute_f(x, D, d, C_cap, s,
              lam_balance: float, lam_cap: float) -> np.ndarray:
    """Mean field f[i,k,mu] = dF/dx[i,k,mu] (Eq 44).

    Returns shape (N, K, M).
    """
    N, K, M = x.shape
    y = s * x
    occ = y.sum(axis=0)                          # (K, M)

    # --- (i) travel cost contribution (Eq 41)
    #   f_tbc[i,k,mu] = sum_nu D[i,k-1,nu,mu] x[i,k-1,nu]
    #                 + sum_nu D[i,k  ,mu,nu] x[i,k+1,nu]
    f_tbc = np.zeros_like(x)
    if K >= 2:
        # contribution from arriving at (k, mu) from k-1
        # for each k in 1..K-1: sum_nu D[i,k-1,nu,mu] x[i,k-1,nu]
        f_tbc[:, 1:] += np.einsum(
            "iknm,ikn->ikm", D[:, :-1], x[:, :-1]
        )
        # contribution from departing (k, mu) to k+1
        # for each k in 0..K-2: sum_nu D[i,k,mu,nu] x[i,k+1,nu]
        f_tbc[:, :-1] += np.einsum(
            "ikmn,ikn->ikm", D[:, :-1], x[:, 1:]
        )

    # --- (ii) demand balancing (Eq 42)
    d_norm = d / np.maximum(d.sum(axis=1, keepdims=True), 1e-12)
    target = N * d_norm                          # (K, M)
    f_balance = lam_balance * (occ - target)     # (K, M)
    f_balance = f_balance[None, :, :] * s        # broadcast to (N,K,M)

    # --- (iii) capacity (Eq 43, one-sided)
    over = np.maximum(occ - C_cap, 0.0)          # (K, M)
    f_cap = lam_cap * over                       # (K, M)
    f_cap = f_cap[None, :, :] * s

    return f_tbc + f_balance + f_cap


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class MFConfig:
    theta0: float = 1.0          # initial temperature
    gamma: float = 0.99          # annealing decay per outer iter
    n_iter: int = 600            # outer iterations
    lam_balance: float = 1.0
    lam_cap: float = 1.0
    eta: float = 0.3             # damping for Jacobi iter: x = (1-eta)x + eta*sm
    perturb: float = 0.05        # random perturbation on uniform init (per-UAV)
    tol: float = 1e-6            # ||x_new - x_old||_inf early stop
    update: str = "jacobi"       # "jacobi" (all UAVs together) or
                                 # "seidel" (one UAV at a time, breaks symmetry)
    verbose: bool = False
    seed: int = 0


@dataclass
class MFResult:
    x: np.ndarray                # (N, K, M) soft assignment
    assignment: np.ndarray       # (N, K) hard argmax over mu
    F_history: list = field(default_factory=list)
    theta_history: list = field(default_factory=list)
    iters: int = 0


def mean_field_schedule(D, d, C_cap, s=None,
                        x0=None, cfg: Optional[MFConfig] = None) -> MFResult:
    """Solve the mean-field scheduling problem.

    Parameters
    ----------
    D      : (N, K, M, M) travel cost tensor. D[i,k,nu,mu] is cost for UAV i
             moving from nu at time k to mu at time k+1. The last time slice
             D[:, K-1, :, :] is unused but must be present (for shape).
    d      : (K, M) predicted demand per (time, station).
    C_cap  : (K, M) station capacity.
    s      : (N, K, M) availability switch (default all ones).
    x0     : (N, K, M) initial soft assignment (default uniform 1/M).
    cfg    : MFConfig.

    Returns
    -------
    MFResult with soft x, hard argmax assignment, and convergence history.
    """
    if cfg is None:
        cfg = MFConfig()
    N, K, M = (D.shape[0], D.shape[1], D.shape[2])
    assert D.shape == (N, K, M, M)
    assert d.shape == (K, M)
    assert C_cap.shape == (K, M)

    if s is None:
        s = np.ones((N, K, M), dtype=np.float32)
    if x0 is None:
        # Uniform + small per-UAV random perturbation to break symmetry.
        rng = np.random.default_rng(cfg.seed)
        x = np.full((N, K, M), 1.0 / M, dtype=np.float64)
        if cfg.perturb > 0:
            noise = rng.uniform(-cfg.perturb, cfg.perturb, size=(N, K, M))
            x = x + noise
            x = np.clip(x, 1e-6, None)
            x = x / x.sum(axis=2, keepdims=True)
    else:
        x = x0.astype(np.float64, copy=True)

    theta = float(cfg.theta0)
    F_hist, theta_hist = [], []
    rng = np.random.default_rng(cfg.seed + 7919)

    for it in range(cfg.n_iter):
        if cfg.update == "seidel":
            # Gauss-Seidel: update one UAV at a time, using already-updated
            # neighbours. Breaks the inter-UAV symmetry that causes Jacobi
            # to collapse all UAVs onto the same task.
            x_old = x.copy()
            order = rng.permutation(N)
            for i in order:
                f_i = compute_f(x, D, d, C_cap, s,
                                cfg.lam_balance, cfg.lam_cap)[i]
                x_target_i = _softmax_neg(f_i, theta, axis=1)   # over mu
                x[i] = (1.0 - cfg.eta) * x[i] + cfg.eta * x_target_i
            x_new = x
            diff = float(np.max(np.abs(x_new - x_old)))
        else:   # jacobi
            f = compute_f(x, D, d, C_cap, s, cfg.lam_balance, cfg.lam_cap)
            x_target = _softmax_neg(f, theta, axis=2)
            x_new = (1.0 - cfg.eta) * x + cfg.eta * x_target
            diff = float(np.max(np.abs(x_new - x)))
            x = x_new

        F = compute_F(x, D, d, C_cap, s, cfg.lam_balance, cfg.lam_cap)
        F_hist.append(F)
        theta_hist.append(theta)

        if cfg.verbose and (it < 5 or it % 50 == 0):
            print(f"[MF] iter {it:3d}  theta={theta:.4f}  "
                  f"F={F:.4f}  |dx|_inf={diff:.2e}")

        theta *= cfg.gamma

        if diff < cfg.tol and it > 20:
            if cfg.verbose:
                print(f"[MF] converged at iter {it}  (|dx|<{cfg.tol})")
            break

    assignment = np.argmax(x, axis=2)
    return MFResult(x=x, assignment=assignment, F_history=F_hist,
                    theta_history=theta_hist, iters=len(F_hist))


# ---------------------------------------------------------------------------
# Convenience: 2D (single time step) wrapper for the assignment problem.
# This is the K=1 case used to validate against the Hungarian algorithm.
# ---------------------------------------------------------------------------

def mean_field_assignment_2d(C, lam_one=10.0,
                             cfg: Optional[MFConfig] = None) -> MFResult:
    """K=1 assignment: assign N UAVs to M >= N tasks using cost matrix C[N, M].

    Builds D so that the only term left is the per-UAV cost C, with a
    one-task-per-UAV constraint enforced softly via a capacity-style penalty
    so multiple UAVs don't pile on the same task.

    Specifically we use the K=2 trick:
        x[i, 0, mu] = "which task does UAV i go to"   (softmax over mu)
        D[i, 0, mu, nu] = C[i, mu] if nu == mu else big  (forces self-loop)
    and add a global capacity C_cap=1 with strong lam_cap so each task is
    occupied by ~1 UAV (one-hot in mu, distinct across i).
    """
    if cfg is None:
        cfg = MFConfig(theta0=1.0, gamma=0.99, n_iter=600,
                       eta=0.3, perturb=0.05, tol=1e-6)
    N, M = C.shape
    K = 2

    # D[i, 0, nu, mu] = C[i, mu] (cost only depends on destination mu)
    # This makes F_tbc = sum_i sum_nu sum_mu C[i,mu] x[i,0,nu] x[i,1,mu]
    #                  = sum_i sum_mu C[i,mu] x[i,1,mu]   (since sum_nu x=1)
    D = np.zeros((N, K, M, M), dtype=np.float64)
    D[:, 0, :, :] = C[:, None, :]                # broadcast over nu

    # demand: irrelevant -> use uniform
    d = np.ones((K, M), dtype=np.float64)
    # capacity: each task ≤ 1 UAV (one-sided penalty)
    C_cap = np.ones((K, M), dtype=np.float64)

    cfg2 = MFConfig(theta0=cfg.theta0, gamma=cfg.gamma,
                    n_iter=cfg.n_iter, tol=cfg.tol,
                    lam_balance=0.0, lam_cap=lam_one,
                    eta=cfg.eta, perturb=cfg.perturb,
                    update=cfg.update,
                    verbose=cfg.verbose, seed=cfg.seed)
    return mean_field_schedule(D, d, C_cap, cfg=cfg2)


def mf_hybrid_lsa_assignment(C, lam_one=50.0,
                             cfg: Optional[MFConfig] = None,
                             eps=1e-9):
    """Mean-field + Hungarian post-processing for exact 1-to-1 assignment.

    The mean-field pass biases the soft assignment x toward low-cost
    matches and respects the global capacity constraint. The Hungarian
    step then extracts a strict bijection using -log(x) + C as the cost
    matrix, breaking ties in favour of the mean-field's preferred pairings.

    This is the "hybrid-LSA" variant referenced in the paper's Table 1
    which matches Hungarian exactly on the wave-derived cost block.

    Returns
    -------
    task_of_uav  : (N,) int array, task index assigned to each UAV, or
                   -1 if that UAV got no task (when N > M).
    mf_result    : the underlying MFResult.
    """
    from scipy.optimize import linear_sum_assignment
    if cfg is None:
        cfg = MFConfig(theta0=3.0, gamma=0.99, n_iter=300,
                       eta=0.5, perturb=0.1, update="seidel")
    res = mean_field_assignment_2d(C, lam_one=lam_one, cfg=cfg)
    x_soft = res.x[:, 1, :]                              # (N, M)
    score = C - np.log(x_soft + eps)                     # lower = better
    row, col = linear_sum_assignment(score)
    # row[k] = uav index, col[k] = task index. Invert to per-UAV array.
    N = C.shape[0]
    task_of_uav = np.full(N, -1, dtype=np.int64)
    task_of_uav[row] = col
    return task_of_uav, res
