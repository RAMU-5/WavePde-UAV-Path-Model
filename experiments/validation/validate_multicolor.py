"""Validate the four-colour batched-source approach (paper §2.13.1).

Two checks:

A. GRAPH-COLOURING correctness on a contrived adversarial graph:
   - K_4 clique (4 mutually-adjacent nodes) must use all 4 colours,
     no two adjacent share.
   - A 6-cycle must be 2-colourable (greedy may use 3, accept 2 or 3).
   - A 'wheel' graph (5-cycle + hub) is 4-chromatic — must NOT exceed 4.

B. WAVE SUPERPOSITION equivalence:
   pure-python 2D wave solve with a sum-of-Gaussians source vs
   the sum of 4 individual solves. By linearity these MUST match
   exactly, so the first-arrival-time read at well-separated cells
   matches between the batched solve and the corresponding single-
   source solve.

Pure python (no numpy)."""
import math


# --------------------------- A: graph colouring -----------------------

def conflict_from_edges(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        if v not in adj[u]: adj[u].append(v)
        if u not in adj[v]: adj[v].append(u)
    return adj


def greedy_four_color(adj, n):
    order = sorted(range(n), key=lambda i: -len(adj[i]))
    colour = [-1] * n
    for v in order:
        forbidden = set(colour[u] for u in adj[v] if colour[u] >= 0)
        c = 0
        while c in forbidden:
            c += 1
        colour[v] = c
    return colour


def is_proper(colour, edges):
    for u, v in edges:
        if colour[u] == colour[v]:
            return False
    return True


def case_K4():
    print("Case A1: K_4 clique (must use exactly 4 colours)")
    n = 4
    edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
    adj = conflict_from_edges(n, edges)
    c = greedy_four_color(adj, n)
    n_col = len(set(c))
    proper = is_proper(c, edges)
    print(f"  colours: {c}  distinct = {n_col}  proper = {proper}")
    return proper and n_col == 4


def case_C6():
    print("\nCase A2: C_6 (6-cycle, chromatic number 2)")
    n = 6
    edges = [(i, (i+1) % n) for i in range(n)]
    adj = conflict_from_edges(n, edges)
    c = greedy_four_color(adj, n)
    n_col = len(set(c))
    proper = is_proper(c, edges)
    print(f"  colours: {c}  distinct = {n_col}  proper = {proper}")
    # greedy may use 2 or 3 (depending on tie-breaking)
    return proper and n_col <= 3


def case_wheel():
    print("\nCase A3: wheel W_5 (chromatic number 4)")
    n = 6  # 5 rim + 1 hub (index 5)
    edges = [(i, (i+1) % 5) for i in range(5)]   # 5-cycle on 0..4
    edges += [(i, 5) for i in range(5)]           # hub to all rim
    adj = conflict_from_edges(n, edges)
    c = greedy_four_color(adj, n)
    n_col = len(set(c))
    proper = is_proper(c, edges)
    print(f"  colours: {c}  distinct = {n_col}  proper = {proper}")
    return proper and n_col <= 4


# --------------------------- B: superposition --------------------------

NX, NY = 50, 50
DX = DY = 1.0
DT = 0.05
N_STEPS = 600
C0 = 1.0
PULSE_T0 = 0.5; PULSE_TAU = 0.2; SIGMA_SRC = 1.5
MIN_PEAK = 1e-6


def alloc(): return [[0.0] * NY for _ in range(NX)]


def gauss(idx):
    g = alloc(); s = 0.0
    for i in range(NX):
        for j in range(NY):
            r2 = (i-idx[0])**2 + (j-idx[1])**2
            v = math.exp(-0.5 * r2 / (SIGMA_SRC*SIGMA_SRC))
            g[i][j] = v; s += v
    inv = 1.0 / s
    for i in range(NX):
        for j in range(NY):
            g[i][j] *= inv
    return g


def src_sum(idxs):
    out = alloc()
    for idx in idxs:
        g = gauss(idx)
        for i in range(NX):
            for j in range(NY):
                out[i][j] += g[i][j]
    return out


def solve_2d(src_spatial):
    psi_prev = alloc(); psi_curr = alloc()
    amp_prev = alloc()
    was_rising = [[False]*NY for _ in range(NX)]
    T = [[float('inf')]*NY for _ in range(NX)]
    dt2 = DT * DT
    for n in range(N_STEPS):
        t = n * DT
        # Constant-c Laplacian (much faster than var-coef for the test)
        diff = alloc()
        for i in range(NX):
            ip = (i+1) % NX; im = (i-1) % NX
            for j in range(NY):
                jp = (j+1) % NY; jm = (j-1) % NY
                diff[i][j] = (C0*C0) * (
                    (psi_curr[ip][j] - 2*psi_curr[i][j] + psi_curr[im][j]) / (DX*DX)
                  + (psi_curr[i][jp] - 2*psi_curr[i][j] + psi_curr[i][jm]) / (DY*DY)
                )
        amp = math.exp(-0.5 * ((t - PULSE_T0)/PULSE_TAU) ** 2)
        psi_next = alloc()
        for i in range(NX):
            for j in range(NY):
                rhs = diff[i][j] + src_spatial[i][j] * amp
                psi_next[i][j] = 2.0 * psi_curr[i][j] - psi_prev[i][j] + dt2 * rhs
        for j in range(NY): psi_next[0][j] = 0.0; psi_next[NX-1][j] = 0.0
        for i in range(NX): psi_next[i][0] = 0.0; psi_next[i][NY-1] = 0.0

        for i in range(NX):
            for j in range(NY):
                a = abs(psi_next[i][j])
                rising = a > amp_prev[i][j]
                if (was_rising[i][j] and not rising
                        and T[i][j] == float('inf') and amp_prev[i][j] > MIN_PEAK):
                    T[i][j] = t - DT
                amp_prev[i][j] = a
                was_rising[i][j] = rising

        psi_prev = psi_curr; psi_curr = psi_next
    return T


def case_superposition():
    print("\nCase B: batched 4-source solve vs 4 individual solves")
    print(f"  grid {NX}x{NY}, sources at corners")
    sources = [(8, 8), (NX-9, 8), (8, NY-9), (NX-9, NY-9)]
    # Probe points: each strongly belongs to exactly one source's
    # influence region.
    probes = [
        (12, 12),   # near source 0
        (NX-13, 12),
        (12, NY-13),
        (NX-13, NY-13),
    ]
    # Solve each individually
    T_indep = []
    for s in sources:
        print(f"  solving for source {s} ...")
        T_indep.append(solve_2d(gauss(s)))
    # Batched
    print(f"  solving batched (sum of {len(sources)} sources) ...")
    T_batch = solve_2d(src_sum(sources))

    print(f"  reading T at probes:")
    print(f"  {'src':>4} {'probe':>10} {'T_indep':>10} {'T_batched':>10} {'diff':>10}")
    all_close = True
    for k, (s, p) in enumerate(zip(sources, probes)):
        ti = T_indep[k][p[0]][p[1]]
        tb = T_batch[p[0]][p[1]]
        d = abs(ti - tb)
        ok = math.isfinite(ti) and math.isfinite(tb) and d < 0.5 * DT
        all_close = all_close and ok
        print(f"  {k:>4} {str(p):>10} {ti:>10.3f} {tb:>10.3f} {d:>10.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_close


def main():
    print("validate_multicolor: graph-colouring + wave superposition")
    print()
    results = []
    results.append(case_K4())
    results.append(case_C6())
    results.append(case_wheel())
    results.append(case_superposition())
    print()
    n_ok = sum(1 for r in results if r)
    print(f"=== {n_ok}/{len(results)} cases passed ===")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
