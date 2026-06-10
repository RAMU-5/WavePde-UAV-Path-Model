"""Validate the multi-scale superposition claim (paper §2.9.4–2.9.5):

    Wave equation is LINEAR in the source. Therefore solving once with
    a broadband source s_long + s_short MUST give the same psi-field as
    solving twice (once with s_long alone, once with s_short alone) and
    summing the two psi-fields.

This is the foundational identity the paper relies on to argue that
the broadband-source formulation costs HALF as much as explicit
superposition. If the claim breaks, the entire §2.9.5 argument
collapses.

Test: run three pure-python 2D wave solves on a 30x30 grid:
    long_only:    A_L=1, A_S=0        -> psi_1
    short_only:   A_L=0, A_S=1        -> psi_2
    broadband:    A_L=1, A_S=1        -> psi_3

Then check at every grid cell and every recorded timestep that
psi_3 ~= psi_1 + psi_2 (within float-32 round-off, ~1e-5).

Pass: max relative error < 1e-4 over the full space-time domain.
"""
from math import exp, cos

NX, NY = 30, 30
DX = DY = 1.0
DT = 0.05
N_STEPS = 300
C0 = 1.0
SIGMA_SRC = 1.5
RECORD_AT = (50, 100, 200, 280)        # check linearity at these timesteps

# Source params
A_L_DEF = 1.0; T0_L = 1.6; TAU_L = 1.2
A_S_DEF = 1.0; T0_S = 0.8; TAU_S = 0.2; OMEGA_S = 6.0; PHI = 0.0


def alloc():
    return [[0.0] * NY for _ in range(NX)]


def gauss_src(idx):
    g = alloc(); s = 0.0
    for i in range(NX):
        for j in range(NY):
            r2 = (i - idx[0])**2 + (j - idx[1])**2
            v = exp(-0.5 * r2 / (SIGMA_SRC * SIGMA_SRC))
            g[i][j] = v; s += v
    inv = 1.0 / s
    for i in range(NX):
        for j in range(NY):
            g[i][j] *= inv
    return g


def lap(c2, psi, axis):
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


def src_amp(t, A_L, A_S):
    g_L = exp(-((t - T0_L) ** 2) / (TAU_L ** 2))
    g_S = exp(-((t - T0_S) ** 2) / (TAU_S ** 2)) * cos(OMEGA_S * (t - T0_S) + PHI)
    return A_L * g_L + A_S * g_S


def solve(A_L, A_S, src_idx, snapshots):
    c2 = [[C0 * C0] * NY for _ in range(NX)]
    psi_prev = alloc(); psi_curr = alloc()
    src_spatial = gauss_src(src_idx)
    dt2 = DT * DT
    snaps = {}

    for n in range(N_STEPS):
        t = n * DT
        diff_x = lap(c2, psi_curr, 0)
        diff_y = lap(c2, psi_curr, 1)
        amp = src_amp(t, A_L, A_S)
        psi_next = alloc()
        for i in range(NX):
            for j in range(NY):
                rhs = diff_x[i][j] + diff_y[i][j] + src_spatial[i][j] * amp
                psi_next[i][j] = 2.0 * psi_curr[i][j] - psi_prev[i][j] + dt2 * rhs

        for j in range(NY):
            psi_next[0][j] = 0.0; psi_next[NX-1][j] = 0.0
        for i in range(NX):
            psi_next[i][0] = 0.0; psi_next[i][NY-1] = 0.0

        psi_prev = psi_curr; psi_curr = psi_next
        if n in snapshots:
            snaps[n] = [row[:] for row in psi_curr]
    return snaps


def max_amp(field):
    m = 0.0
    for row in field:
        for v in row:
            a = abs(v)
            if a > m: m = a
    return m


def diff_norm(a, b):
    """Returns (max_abs_diff, ref_max). Caller computes relative norm."""
    md = 0.0; mr = 0.0
    for i in range(NX):
        for j in range(NY):
            d = abs(a[i][j] - b[i][j])
            if d > md: md = d
            ar = abs(a[i][j]) + abs(b[i][j])
            if ar > mr: mr = ar
    return md, mr


def main():
    print(f"validate_broadband: pure-python linearity test, "
          f"{NX}x{NY} grid, {N_STEPS} steps")
    print(f"  long source: A_L={A_L_DEF}, tau_L={TAU_L}")
    print(f"  short Gabor: A_S={A_S_DEF}, tau_S={TAU_S}, omega_S={OMEGA_S}")
    print()

    src_idx = (15, 15)
    snaps = list(RECORD_AT)

    print("solve 1: long only ...")
    L = solve(A_L_DEF, 0.0, src_idx, snaps)
    print("solve 2: short only ...")
    S = solve(0.0, A_S_DEF, src_idx, snaps)
    print("solve 3: broadband ...")
    B = solve(A_L_DEF, A_S_DEF, src_idx, snaps)
    print()

    all_pass = True
    print(f"{'step':>5} {'|psi_L|max':>12} {'|psi_S|max':>12} "
          f"{'|psi_B|max':>12} {'|err|max':>12} {'rel_err':>10} {'verdict':>8}")
    print("-" * 75)
    for n in snaps:
        psi_L = L[n]; psi_S = S[n]; psi_B = B[n]
        # construct psi_L + psi_S
        sum_field = [[psi_L[i][j] + psi_S[i][j] for j in range(NY)]
                     for i in range(NX)]
        ml = max_amp(psi_L); ms = max_amp(psi_S); mb = max_amp(psi_B)
        md, mr = diff_norm(psi_B, sum_field)
        rel = md / max(mb, 1e-9)
        ok = rel < 1e-4
        all_pass = all_pass and ok
        print(f"{n:5d} {ml:12.4e} {ms:12.4e} {mb:12.4e} {md:12.4e} "
              f"{rel:10.2e}   {'PASS' if ok else 'FAIL'}")

    print()
    if all_pass:
        print("=== ALL CHECKS PASSED — linearity confirmed, "
              "broadband == long+short ===")
        return 0
    else:
        print("=== FAIL — broadband solve disagrees with sum of components ===")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
