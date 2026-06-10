"""Minimal 2D pure-numpy validation of the backward-wave symmetry
(paper §2.5).

Claim being tested:
    T_backward(seed=B, q->-q)  evaluated at A
    ==
    T_forward (seed=A, q     )  evaluated at B

We test it twice:
  Case A: q = 0     -> tests the self-adjoint property of the bare
                       wave operator (line 592 of the paper).
  Case B: q != 0    -> tests the full convected-wave symmetry under
                       q -> -q (the actual claim used by the adjoint
                       routing scheme).

This file deliberately avoids JAX/JIT so it runs on a login node in
seconds. The numerics mirror solve_all.py:
    leapfrog psi_next = 2 psi - psi_prev + dt^2 * rhs
    rhs = div(c^2 grad psi) + src - 2 (q.grad) psi_dot
            - (q.grad)^2 psi - ((q.grad)q . grad) psi
"""
from __future__ import annotations
import numpy as np

# ---- domain ------------------------------------------------------------
NX, NY = 80, 80
DX = DY = 1.0
DT = 0.05
N_STEPS = 1500       # ~75 t.u. — enough for 50-cell domain crossing at c=1
C0 = 1.0

PULSE_T0 = 0.5
PULSE_TAU = 0.2
SIGMA_SRC = 1.5
MIN_PEAK = 1e-6


def grad_central(f, axis, h):
    return (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) / (2*h)


def lap_var_coef(c2, psi, h, axis):
    c2_e = 0.5 * (c2 + np.roll(c2, -1, axis=axis))
    c2_w = 0.5 * (c2 + np.roll(c2, +1, axis=axis))
    flux_e = c2_e * (np.roll(psi, -1, axis=axis) - psi) / h
    flux_w = c2_w * (psi - np.roll(psi, +1, axis=axis)) / h
    return (flux_e - flux_w) / h


def adv(qx, qy, psi):
    return qx * grad_central(psi, 0, DX) + qy * grad_central(psi, 1, DY)


def gaussian_src(shape, idx, sigma=SIGMA_SRC):
    nx, ny = shape
    ix = np.arange(nx)[:, None]
    iy = np.arange(ny)[None, :]
    g = np.exp(-0.5 * ((ix-idx[0])**2 + (iy-idx[1])**2) / sigma**2)
    return (g / g.sum()).astype(np.float32)


def solve(c2, qx, qy, ax_f, ay_f, src_idx, n_steps=N_STEPS):
    shape = c2.shape
    psi_prev = np.zeros(shape, dtype=np.float32)
    psi_curr = np.zeros(shape, dtype=np.float32)
    amp_prev = np.zeros(shape, dtype=np.float32)
    was_rising = np.zeros(shape, dtype=bool)
    T = np.full(shape, np.inf, dtype=np.float32)
    src_spatial = gaussian_src(shape, src_idx)

    dt2 = DT * DT
    for n in range(n_steps):
        t = n * DT
        diff = (lap_var_coef(c2, psi_curr, DX, 0)
                + lap_var_coef(c2, psi_curr, DY, 1))
        src = src_spatial * np.exp(-0.5 * ((t - PULSE_T0)/PULSE_TAU)**2)
        psi_dot = (psi_curr - psi_prev) / DT
        adv1 = adv(qx, qy, psi_dot)                         # 2 q.grad psi_dot
        qg = adv(qx, qy, psi_curr)
        adv2 = adv(qx, qy, qg)                              # (q.grad)^2 psi
        adv_a = (ax_f * grad_central(psi_curr, 0, DX)
                 + ay_f * grad_central(psi_curr, 1, DY))    # ((q.grad)q).grad psi
        rhs = diff + src - 2.0*adv1 - adv2 - adv_a
        psi_next = 2.0*psi_curr - psi_prev + dt2 * rhs

        # zero edges (Dirichlet)
        psi_next[0, :] = 0; psi_next[-1, :] = 0
        psi_next[:, 0] = 0; psi_next[:, -1] = 0

        amp = np.abs(psi_next)
        rising_now = amp > amp_prev
        peak_now = (was_rising & (~rising_now)
                    & np.isinf(T) & (amp_prev > MIN_PEAK))
        T = np.where(peak_now, t - DT, T)

        psi_prev = psi_curr
        psi_curr = psi_next
        amp_prev = amp
        was_rising = rising_now
    return T


def setup_static(qx_const, qy_const):
    """Build c2, q, and (q.grad)q for a uniform flow."""
    c2 = np.full((NX, NY), C0**2, dtype=np.float32)
    qx = np.full((NX, NY), qx_const, dtype=np.float32)
    qy = np.full((NX, NY), qy_const, dtype=np.float32)
    # Uniform q -> grad(q) = 0 -> (q.grad)q = 0 exactly.
    ax = np.zeros_like(qx)
    ay = np.zeros_like(qy)
    return c2, qx, qy, ax, ay


def run_case(label, qx0, qy0):
    A = (20, 30)
    B = (60, 50)

    c2, qx, qy, ax, ay = setup_static(qx0, qy0)

    # Forward: source at A
    T_fwd = solve(c2, qx, qy, ax, ay, A)
    t_AB = T_fwd[B[0], B[1]]

    # Backward: source at B, q -> -q
    T_bwd = solve(c2, -qx, -qy, ax, ay, B)
    t_BA = T_bwd[A[0], A[1]]

    rel_err = abs(t_AB - t_BA) / max(t_AB, t_BA, 1e-9)
    pass_str = "PASS" if rel_err < 0.02 else "FAIL"
    print(f"  [{pass_str}] {label}: q=({qx0:+.2f},{qy0:+.2f})  "
          f"T_forward(A->B) = {t_AB:.3f}   "
          f"T_backward(B->A) = {t_BA:.3f}   "
          f"rel_err = {rel_err:.2e}")
    return rel_err < 0.02


def main():
    print("validate_backward_minimal: 2D wave-equation symmetry test")
    print(f"  domain {NX}x{NY}, c0={C0}, dt={DT}, n_steps={N_STEPS}")
    print()

    results = []

    # ----- Case 1: no flow ------------------------------------------------
    # Self-adjoint claim: T_fwd(A->B) == T_bwd(B->A) when q=0, sigma=0.
    print("Case 1: zero flow (self-adjoint operator)")
    results.append(run_case("q=0", 0.0, 0.0))
    print()

    # ----- Case 2: with flow ----------------------------------------------
    # Convected-wave claim: T equality holds when we ALSO flip q.
    print("Case 2: uniform flow (convected wave)")
    for qx0, qy0 in [(0.20, 0.0), (0.0, 0.20), (0.15, -0.10), (-0.20, 0.15)]:
        results.append(run_case(f"q=({qx0:+.2f},{qy0:+.2f})", qx0, qy0))
    print()

    # ----- Case 3: sanity — forward-only asymmetry ------------------------
    # Without flipping q, forward and backward should DISAGREE under flow.
    # This confirms the test isn't trivially passing.
    print("Case 3: backward without flipping q (should DISAGREE under flow)")
    A = (20, 30); B = (60, 50)
    c2, qx, qy, ax, ay = setup_static(0.20, 0.0)
    T_fwd = solve(c2, qx, qy, ax, ay, A)
    T_bwd_no_flip = solve(c2, qx, qy, ax, ay, B)  # NO flip
    t_AB = T_fwd[B[0], B[1]]
    t_BA_noflip = T_bwd_no_flip[A[0], A[1]]
    rel_err = abs(t_AB - t_BA_noflip) / max(t_AB, t_BA_noflip, 1e-9)
    expect_disagree = rel_err > 0.05
    print(f"  [{'PASS' if expect_disagree else 'FAIL'}] no-flip control: "
          f"T_fwd = {t_AB:.3f}  T_bwd_no_flip = {t_BA_noflip:.3f}  "
          f"rel_err = {rel_err:.2e} (expected > 5%)")
    results.append(expect_disagree)
    print()

    n_pass = sum(results)
    n_total = len(results)
    print(f"=== {n_pass}/{n_total} checks passed ===")
    if n_pass < n_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
