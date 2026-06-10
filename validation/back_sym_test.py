"""Pure-python (no numpy/jax/anything) symmetry test for the convected
wave equation. Tiny 25x25 grid, 400 timesteps, ~2 seconds on a login
node. Mirrors the math in solve_all.py / solve_backward.py exactly.

Tests the claim:
    T_backward(seed=B, q -> -q)  at A
    ==
    T_forward (seed=A, q     )   at B

If the bare wave-routing solver is correct, both cases (q=0 and q!=0)
must pass.
"""
from math import exp, sqrt

NX, NY = 25, 25
DX = DY = 1.0
DT = 0.05
N_STEPS = 400
C0 = 1.0
PULSE_T0 = 0.5
PULSE_TAU = 0.2
SIGMA_SRC = 1.5
MIN_PEAK = 1e-6


def alloc(default=0.0):
    return [[float(default) for _ in range(NY)] for _ in range(NX)]


def gauss_src(idx, sigma=SIGMA_SRC):
    g = alloc()
    s = 0.0
    for i in range(NX):
        for j in range(NY):
            r2 = (i - idx[0])**2 + (j - idx[1])**2
            v = exp(-0.5 * r2 / (sigma * sigma))
            g[i][j] = v
            s += v
    inv = 1.0 / s
    for i in range(NX):
        for j in range(NY):
            g[i][j] *= inv
    return g


def grad_x(f, i, j):
    ip = (i + 1) % NX; im = (i - 1) % NX
    return (f[ip][j] - f[im][j]) / (2 * DX)


def grad_y(f, i, j):
    jp = (j + 1) % NY; jm = (j - 1) % NY
    return (f[i][jp] - f[i][jm]) / (2 * DY)


def lap_var(c2, psi, axis):
    """Variable-coefficient Laplacian in 2D, matching solve_all.lap_var_coef_axis."""
    out = alloc()
    if axis == 0:
        for i in range(NX):
            ip = (i + 1) % NX; im = (i - 1) % NX
            for j in range(NY):
                c2_e = 0.5 * (c2[i][j] + c2[ip][j])
                c2_w = 0.5 * (c2[i][j] + c2[im][j])
                fe = c2_e * (psi[ip][j] - psi[i][j]) / DX
                fw = c2_w * (psi[i][j] - psi[im][j]) / DX
                out[i][j] = (fe - fw) / DX
    else:
        for i in range(NX):
            for j in range(NY):
                jp = (j + 1) % NY; jm = (j - 1) % NY
                c2_n = 0.5 * (c2[i][j] + c2[i][jp])
                c2_s = 0.5 * (c2[i][j] + c2[i][jm])
                fn = c2_n * (psi[i][jp] - psi[i][j]) / DY
                fs = c2_s * (psi[i][j] - psi[i][jm]) / DY
                out[i][j] = (fn - fs) / DY
    return out


def adv_field(qx, qy, psi):
    out = alloc()
    for i in range(NX):
        for j in range(NY):
            out[i][j] = qx[i][j] * grad_x(psi, i, j) + qy[i][j] * grad_y(psi, i, j)
    return out


def solve(c2, qx, qy, src_idx):
    """Identical to validate_backward_minimal.py logic, pure python.
    Note: ax,ay set to zero (uniform q -> (q.grad)q = 0)."""
    psi_prev = alloc()
    psi_curr = alloc()
    amp_prev = alloc()
    was_rising = [[False] * NY for _ in range(NX)]
    T = [[float('inf')] * NY for _ in range(NX)]
    src_spatial = gauss_src(src_idx)
    dt2 = DT * DT

    for n in range(N_STEPS):
        t = n * DT
        diff_x = lap_var(c2, psi_curr, 0)
        diff_y = lap_var(c2, psi_curr, 1)
        psi_dot = alloc()
        for i in range(NX):
            for j in range(NY):
                psi_dot[i][j] = (psi_curr[i][j] - psi_prev[i][j]) / DT
        adv1 = adv_field(qx, qy, psi_dot)
        qg = adv_field(qx, qy, psi_curr)
        adv2 = adv_field(qx, qy, qg)

        src_amp = exp(-0.5 * ((t - PULSE_T0) / PULSE_TAU) ** 2)

        psi_next = alloc()
        for i in range(NX):
            for j in range(NY):
                rhs = (diff_x[i][j] + diff_y[i][j]
                       + src_spatial[i][j] * src_amp
                       - 2.0 * adv1[i][j]
                       - adv2[i][j])
                psi_next[i][j] = 2.0 * psi_curr[i][j] - psi_prev[i][j] + dt2 * rhs

        # zero edges
        for j in range(NY):
            psi_next[0][j] = 0.0; psi_next[NX-1][j] = 0.0
        for i in range(NX):
            psi_next[i][0] = 0.0; psi_next[i][NY-1] = 0.0

        for i in range(NX):
            for j in range(NY):
                a = abs(psi_next[i][j])
                rising = a > amp_prev[i][j]
                if (was_rising[i][j] and not rising
                        and T[i][j] == float('inf') and amp_prev[i][j] > MIN_PEAK):
                    T[i][j] = t - DT
                amp_prev[i][j] = a
                was_rising[i][j] = rising

        psi_prev = psi_curr
        psi_curr = psi_next
    return T


def make_uniform(qx0, qy0):
    c2 = [[C0 * C0 for _ in range(NY)] for _ in range(NX)]
    qx = [[qx0 for _ in range(NY)] for _ in range(NX)]
    qy = [[qy0 for _ in range(NY)] for _ in range(NX)]
    return c2, qx, qy


def case(label, qx0, qy0, A=(7, 9), B=(17, 15)):
    c2, qx, qy = make_uniform(qx0, qy0)

    Tf = solve(c2, qx, qy, A)                                         # forward
    nqx = [[-qx[i][j] for j in range(NY)] for i in range(NX)]
    nqy = [[-qy[i][j] for j in range(NY)] for i in range(NX)]
    Tb = solve(c2, nqx, nqy, B)                                       # backward (q -> -q)

    t_AB = Tf[B[0]][B[1]]
    t_BA = Tb[A[0]][A[1]]
    if t_AB == float('inf') or t_BA == float('inf'):
        print(f"  [SKIP] {label}: wave didn't reach (t_AB={t_AB}, t_BA={t_BA})")
        return None
    err = abs(t_AB - t_BA) / max(t_AB, t_BA, 1e-9)
    ok = err < 0.04
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  q=({qx0:+.2f},{qy0:+.2f})  "
          f"T_fwd(A->B)={t_AB:.3f}  T_bwd(B->A)={t_BA:.3f}  rel_err={err:.2%}")
    return ok


def main():
    print(f"pure-python backward symmetry test on {NX}x{NY} grid, "
          f"{N_STEPS} steps")
    print()

    results = []

    print("Case 1: zero flow (self-adjoint)")
    results.append(case("q=0", 0.0, 0.0))

    print("\nCase 2: uniform flow (convected, q -> -q)")
    for qx0, qy0 in [(0.20, 0.0), (0.0, 0.20), (0.15, -0.10)]:
        results.append(case(f"q=({qx0:+.2f},{qy0:+.2f})", qx0, qy0))

    print("\nCase 3: control — backward without flipping q (should DISAGREE)")
    A = (7, 9); B = (17, 15)
    c2, qx, qy = make_uniform(0.20, 0.0)
    Tf = solve(c2, qx, qy, A)
    Tbnf = solve(c2, qx, qy, B)
    t_AB = Tf[B[0]][B[1]]
    t_BA_nf = Tbnf[A[0]][A[1]]
    err = abs(t_AB - t_BA_nf) / max(t_AB, t_BA_nf, 1e-9)
    expect = err > 0.05
    print(f"  [{'PASS' if expect else 'FAIL'}] no-flip control  "
          f"T_fwd={t_AB:.3f}  T_bwd_no_flip={t_BA_nf:.3f}  rel_err={err:.2%} "
          f"(expected > 5%)")
    results.append(expect)

    n_ok = sum(1 for r in results if r is True)
    n_total = sum(1 for r in results if r is not None)
    print(f"\n=== {n_ok}/{n_total} checks passed ===")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
