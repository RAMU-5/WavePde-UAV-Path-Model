"""PyTorch port of mf_scheduler — runs on CPU or CUDA GPU.

Mirrors the numpy implementation but uses torch tensors throughout so
that the core einsums + softmax run on whatever device the input
tensors live on.

Hungarian post-processing (mf_hybrid_lsa_assignment) still uses scipy's
linear_sum_assignment which is CPU-only — the soft x is copied to CPU
for the cleanup step (cheap, runs once per dispatch).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_tensor(x: ArrayLike, device, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _softmax_neg(f: torch.Tensor, theta: float, dim: int) -> torch.Tensor:
    """softmax(-f/theta) along `dim`. torch.softmax is already numerically stable."""
    return torch.softmax(-f / max(theta, 1e-12), dim=dim)


# ---------------------------------------------------------------------------
# Functional and gradient (torch versions of compute_F, compute_f)
# ---------------------------------------------------------------------------

def compute_F_t(x, D, d, C_cap, s, lam_balance: float, lam_cap: float) -> float:
    N, K, M = x.shape
    y = s * x

    F_tbc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if K >= 2:
        F_tbc = torch.einsum("iknm,ikn,ikm->",
                              D[:, :-1], x[:, :-1], x[:, 1:])

    occ = y.sum(dim=0)                                     # (K, M)
    d_norm = d / torch.clamp(d.sum(dim=1, keepdim=True), min=1e-12)
    target = N * d_norm
    F_balance = 0.5 * lam_balance * ((occ - target) ** 2).sum()

    over = torch.clamp(occ - C_cap, min=0.0)
    F_cap = 0.5 * lam_cap * (over ** 2).sum()

    return float((F_tbc + F_balance + F_cap).item())


def compute_f_t(x, D, d, C_cap, s,
                lam_balance: float, lam_cap: float) -> torch.Tensor:
    N, K, M = x.shape
    y = s * x
    occ = y.sum(dim=0)                                     # (K, M)

    f_tbc = torch.zeros_like(x)
    if K >= 2:
        # arrived at (k, mu) from k-1
        f_tbc[:, 1:]  += torch.einsum("iknm,ikn->ikm",
                                       D[:, :-1], x[:, :-1])
        # depart (k, mu) to k+1
        f_tbc[:, :-1] += torch.einsum("ikmn,ikn->ikm",
                                       D[:, :-1], x[:, 1:])

    d_norm = d / torch.clamp(d.sum(dim=1, keepdim=True), min=1e-12)
    target = N * d_norm                                    # (K, M)
    f_balance = lam_balance * (occ - target)               # (K, M)
    f_balance = f_balance.unsqueeze(0) * s

    over = torch.clamp(occ - C_cap, min=0.0)               # (K, M)
    f_cap = lam_cap * over
    f_cap = f_cap.unsqueeze(0) * s

    return f_tbc + f_balance + f_cap


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class MFConfigT:
    theta0: float = 1.0
    gamma: float = 0.99
    n_iter: int = 200
    lam_balance: float = 1.0
    lam_cap: float = 1.0
    eta: float = 0.4
    perturb: float = 0.05
    tol: float = 1e-6
    verbose: bool = False
    seed: int = 0
    device: str = "cpu"          # "cpu" or "cuda" or "cuda:0"


@dataclass
class MFResultT:
    x: torch.Tensor              # soft assignment on device
    assignment: torch.Tensor     # hard argmax over mu
    F_history: list = field(default_factory=list)
    theta_history: list = field(default_factory=list)
    iters: int = 0


def mean_field_schedule_t(D, d, C_cap, s=None, x0=None,
                          cfg: Optional[MFConfigT] = None) -> MFResultT:
    """Jacobi mean-field iteration on torch tensors (CPU or GPU).

    All inputs may be numpy or torch; they are moved to cfg.device.
    """
    if cfg is None:
        cfg = MFConfigT()
    device = torch.device(cfg.device)

    D = _as_tensor(D, device)
    d = _as_tensor(d, device)
    C_cap = _as_tensor(C_cap, device)
    N, K, M = D.shape[0], D.shape[1], D.shape[2]
    assert D.shape == (N, K, M, M)
    assert d.shape == (K, M)
    assert C_cap.shape == (K, M)

    if s is None:
        s = torch.ones((N, K, M), device=device, dtype=D.dtype)
    else:
        s = _as_tensor(s, device)

    if x0 is None:
        # uniform + small per-UAV random perturbation
        g = torch.Generator(device=device).manual_seed(int(cfg.seed))
        x = torch.full((N, K, M), 1.0 / M, device=device, dtype=D.dtype)
        if cfg.perturb > 0:
            noise = (torch.rand((N, K, M), generator=g, device=device,
                                 dtype=D.dtype) - 0.5) * 2 * cfg.perturb
            x = torch.clamp(x + noise, min=1e-6)
            x = x / x.sum(dim=2, keepdim=True)
    else:
        x = _as_tensor(x0, device).clone()

    theta = float(cfg.theta0)
    F_hist, theta_hist = [], []

    for it in range(cfg.n_iter):
        f = compute_f_t(x, D, d, C_cap, s, cfg.lam_balance, cfg.lam_cap)
        x_target = _softmax_neg(f, theta, dim=2)
        x_new = (1.0 - cfg.eta) * x + cfg.eta * x_target
        diff = float((x_new - x).abs().max().item())
        x = x_new

        F = compute_F_t(x, D, d, C_cap, s, cfg.lam_balance, cfg.lam_cap)
        F_hist.append(F)
        theta_hist.append(theta)

        if cfg.verbose and (it < 5 or it % 50 == 0):
            print(f"[MF-t] iter {it:3d}  theta={theta:.4f}  "
                  f"F={F:.4f}  |dx|_inf={diff:.2e}")

        theta *= cfg.gamma
        if diff < cfg.tol and it > 20:
            break

    assignment = x.argmax(dim=2)
    return MFResultT(x=x, assignment=assignment, F_history=F_hist,
                     theta_history=theta_hist, iters=len(F_hist))


def mean_field_assignment_2d_t(C, lam_one=10.0,
                               cfg: Optional[MFConfigT] = None) -> MFResultT:
    """K=1 assignment wrapper, torch version."""
    if cfg is None:
        cfg = MFConfigT(theta0=1.0, gamma=0.99, n_iter=600,
                        eta=0.3, perturb=0.05, tol=1e-6)
    device = torch.device(cfg.device)
    C = _as_tensor(C, device)
    N, M = C.shape
    K = 2

    D = torch.zeros((N, K, M, M), device=device, dtype=C.dtype)
    D[:, 0, :, :] = C.unsqueeze(1)                         # broadcast over nu

    d = torch.ones((K, M), device=device, dtype=C.dtype)
    C_cap = torch.ones((K, M), device=device, dtype=C.dtype)

    cfg2 = MFConfigT(theta0=cfg.theta0, gamma=cfg.gamma,
                     n_iter=cfg.n_iter, tol=cfg.tol,
                     lam_balance=0.0, lam_cap=lam_one,
                     eta=cfg.eta, perturb=cfg.perturb,
                     verbose=cfg.verbose, seed=cfg.seed,
                     device=cfg.device)
    return mean_field_schedule_t(D, d, C_cap, cfg=cfg2)


def mean_field_assignment_fast_t(C, lam_one=10.0,
                                  cfg: Optional[MFConfigT] = None,
                                  dtype=torch.float32
                                 ) -> torch.Tensor:
    """Memory-efficient K=1 assignment for the dispatch use case.

    Solves the same problem as mean_field_assignment_2d_t but does NOT
    materialise the (N, 2, M, M) D tensor — which blows up to hundreds
    of GB at paper-scale problem sizes.

    Reduces to a direct fixed-point iteration:
        f[i, mu]  = C[i, mu] + lam_one * max(occ[mu] - 1, 0)
        x[i, mu] <- (1-eta) x[i, mu] + eta * softmax(-f[i,mu] / theta) over mu
        theta    <- gamma * theta

    Memory: O(N*M) per state tensor (vs O(N*M^2) before).

    Returns the soft assignment x (N, M) on the target device.
    """
    if cfg is None:
        cfg = MFConfigT(theta0=10.0, gamma=0.97, n_iter=80,
                        eta=0.4, perturb=0.1)
    device = torch.device(cfg.device)
    # Force float32 for memory / speed at paper scale, and use CPU-side
    # numpy noise + copy to dodge any CUDA-generator edge cases on very
    # large fast-path tensors.
    C_t = _as_tensor(C, device, dtype=dtype)
    N, M = C_t.shape

    # Sanitise: replace any non-finite cost with a large finite sentinel
    # so softmax / arithmetic stays well-behaved.
    big = torch.finfo(dtype).max / 1e3
    C_t = torch.where(torch.isfinite(C_t), C_t,
                       torch.full_like(C_t, big))

    rng = np.random.default_rng(int(cfg.seed))
    x_np = np.full((N, M), 1.0 / M, dtype=np.float32)
    if cfg.perturb > 0:
        x_np = x_np + (rng.random((N, M), dtype=np.float32) - 0.5) \
                       * 2 * cfg.perturb
        np.clip(x_np, 1e-6, None, out=x_np)
        x_np /= x_np.sum(axis=1, keepdims=True)
    x = torch.from_numpy(x_np).to(device=device, dtype=dtype)

    theta = float(cfg.theta0)
    # Check convergence less often to avoid per-iter .item() sync.
    check_every = 10
    prev_x = x.clone()
    for it in range(cfg.n_iter):
        occ = x.sum(dim=0)                           # (M,)
        cap_penalty = lam_one * torch.clamp(occ - 1.0, min=0.0)   # (M,)
        f = C_t + cap_penalty.unsqueeze(0)           # (N, M)
        x_target = _softmax_neg(f, theta, dim=1)
        x = (1.0 - cfg.eta) * x + cfg.eta * x_target
        theta *= cfg.gamma
        if (it + 1) % check_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            diff = float((x - prev_x).abs().max().item())
            if diff < cfg.tol and it > 20:
                break
            prev_x = x.clone()
    return x


def mf_hybrid_lsa_fast_t(C, lam_one=50.0,
                         cfg: Optional[MFConfigT] = None,
                         eps=1e-9):
    """Memory-efficient mean-field + Hungarian cleanup.

    Same API as mf_hybrid_lsa_assignment_t but uses the fast K=1 solver
    (no (N, 2, M, M) tensor materialisation). Suitable for paper-scale
    problems (N, M in the thousands).
    """
    if cfg is None:
        cfg = MFConfigT(theta0=10.0, gamma=0.97, n_iter=80,
                        eta=0.4, perturb=0.1)
    x_soft = mean_field_assignment_fast_t(C, lam_one=lam_one, cfg=cfg)
    x_soft_np = x_soft.detach().cpu().numpy()
    C_np = C if isinstance(C, np.ndarray) else C.detach().cpu().numpy()
    score = C_np - np.log(x_soft_np + eps)
    row, col = linear_sum_assignment(score)
    N = C_np.shape[0]
    task_of_uav = np.full(N, -1, dtype=np.int64)
    task_of_uav[row] = col
    return task_of_uav, x_soft


def mf_hybrid_lsa_assignment_t(C, lam_one=50.0,
                               cfg: Optional[MFConfigT] = None,
                               eps=1e-9):
    """Torch mean-field + (CPU) Hungarian cleanup.

    Returns task_of_uav (numpy int array) and MFResultT.
    """
    if cfg is None:
        cfg = MFConfigT(theta0=3.0, gamma=0.99, n_iter=200,
                        eta=0.4, perturb=0.1)
    res = mean_field_assignment_2d_t(C, lam_one=lam_one, cfg=cfg)
    x_soft = res.x[:, 1, :]                                # (N, M) on device

    # Hungarian is CPU-only. Move the (N, M) soft assignment to CPU.
    x_soft_np = x_soft.detach().cpu().numpy()
    C_np = C if isinstance(C, np.ndarray) else C.detach().cpu().numpy()
    score = C_np - np.log(x_soft_np + eps)
    row, col = linear_sum_assignment(score)
    N = C_np.shape[0]
    task_of_uav = np.full(N, -1, dtype=np.int64)
    task_of_uav[row] = col
    return task_of_uav, res
