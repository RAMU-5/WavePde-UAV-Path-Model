"""Validate the diffusion-based positional uncertainty model
(paper §2.6.3, Eqs. 42-49).

Two checks:
  A. CLOSED-FORM tau(t) accumulation matches a brute-force trapezoidal
     integration on a piecewise-constant D(t) profile (no surprise,
     just sanity-checks the integrator).
  B. MONTE CARLO: simulate n_walkers=4000 random walkers under the
     same D(t) and compare empirical position spread against the
     paper's analytical Gaussian.

Note (paper bug): the paper states sigma_position = 2*sqrt(tau).
For a 3D Gaussian with p(r,t) = (4*pi*tau)^{-3/2} exp(-r^2/(4*tau)),
the variance PER AXIS is 2*tau, so:
   - per-axis std            = sqrt(2*tau)
   - 3D RMS displacement     = sqrt(6*tau)   ~= 2.449 * sqrt(tau)
   - Maxwell std of |r|      = sqrt((6 - 16/pi)*tau) ~= 0.952 * sqrt(tau)
None of these equal 2*sqrt(tau). The MC test reports all three so the
discrepancy is visible and the user can decide which the paper meant.

Pure-python (no numpy needed for the integrals; we use the random
module for the MC walk)."""
import math
import random


# ---- Paper formula: tau(t) and sigma(t) ------------------------------

def compute_tau(times, D):
    tau = [0.0]
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        tau.append(tau[-1] + 0.5 * (D[i-1] + D[i]) * dt)
    return tau


# ---- Monte-Carlo random walk in 3D -----------------------------------

def mc_walk(D_fn, t_max, dt=0.02, n_walkers=4000, seed=0):
    rnd = random.Random(seed)
    walks = [[0.0, 0.0, 0.0] for _ in range(n_walkers)]
    n_steps = int(t_max / dt)
    # Pre-sample standard normals using Box-Muller for speed.
    def nrand():
        # Marsaglia polar
        while True:
            u = 2.0 * rnd.random() - 1.0
            v = 2.0 * rnd.random() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                f = (-2.0 * math.log(s) / s) ** 0.5
                return u * f
    for k in range(n_steps):
        t = (k + 0.5) * dt
        D = D_fn(t)
        sigma_step = (2.0 * D * dt) ** 0.5
        for w in walks:
            for i in range(3):
                w[i] += sigma_step * nrand()
    # stats
    r = [(w[0]**2 + w[1]**2 + w[2]**2) ** 0.5 for w in walks]
    n = len(r)
    r_mean = sum(r) / n
    r_var = sum((x - r_mean) ** 2 for x in r) / n
    r_rms = (sum(x * x for x in r) / n) ** 0.5
    # per-axis variance
    x_var = sum(w[0] * w[0] for w in walks) / n
    y_var = sum(w[1] * w[1] for w in walks) / n
    z_var = sum(w[2] * w[2] for w in walks) / n
    per_axis_std = ((x_var + y_var + z_var) / 3.0) ** 0.5
    return dict(r_mean=r_mean, r_std=r_var ** 0.5, r_rms=r_rms,
                per_axis_std=per_axis_std, n=n)


# ---- Cases ------------------------------------------------------------

def case_const_D():
    """Constant D=0.5 m^2/s, t_max=10s -> tau=5 m^2."""
    print("Case A: constant D = 0.5 m^2/s, t_max = 10 s")
    times = [i * 0.05 for i in range(201)]
    D = [0.5] * len(times)
    tau = compute_tau(times, D)
    tau_final = tau[-1]
    tau_expected = 0.5 * 10.0
    err = abs(tau_final - tau_expected)
    print(f"  computed tau(10) = {tau_final:.4f},  expected = {tau_expected:.4f}, "
          f"|err| = {err:.2e}")
    sigma_paper = 2.0 * tau_final ** 0.5
    sigma_per_axis = (2.0 * tau_final) ** 0.5
    sigma_3d_rms = (6.0 * tau_final) ** 0.5
    sigma_maxwell_std = ((6.0 - 16.0 / math.pi) * tau_final) ** 0.5
    print(f"  paper formula    sigma = 2*sqrt(tau)              = {sigma_paper:.4f} m")
    print(f"  per-axis std                                   = {sigma_per_axis:.4f} m")
    print(f"  3D RMS displacement                            = {sigma_3d_rms:.4f} m")
    print(f"  Maxwell std of |r|                             = {sigma_maxwell_std:.4f} m")
    return err < 1e-9


def case_mc_check():
    """Run MC to confirm which 'sigma' the analytical formula matches."""
    print("\nCase B: Monte-Carlo dispersion test, D=0.5 m^2/s, t_max = 10 s, "
          "4000 walkers")
    mc = mc_walk(lambda t: 0.5, t_max=10.0, dt=0.05, n_walkers=4000, seed=42)
    tau = 0.5 * 10.0
    sigma_paper = 2.0 * tau ** 0.5
    sigma_per_axis = (2.0 * tau) ** 0.5
    sigma_3d_rms = (6.0 * tau) ** 0.5
    sigma_maxwell_std = ((6.0 - 16.0 / math.pi) * tau) ** 0.5
    print(f"  MC per-axis std                = {mc['per_axis_std']:.4f} m  "
          f"(analytical: {sigma_per_axis:.4f})")
    print(f"  MC 3D RMS displacement         = {mc['r_rms']:.4f} m  "
          f"(analytical: {sigma_3d_rms:.4f})")
    print(f"  MC Maxwell std of |r|          = {mc['r_std']:.4f} m  "
          f"(analytical: {sigma_maxwell_std:.4f})")
    print(f"  MC mean of |r|                 = {mc['r_mean']:.4f} m")
    print(f"  paper formula 2*sqrt(tau)       = {sigma_paper:.4f} m")

    err_axis = abs(mc['per_axis_std'] - sigma_per_axis) / sigma_per_axis
    err_rms = abs(mc['r_rms'] - sigma_3d_rms) / sigma_3d_rms
    err_maxwell = abs(mc['r_std'] - sigma_maxwell_std) / sigma_maxwell_std

    print()
    print("  agreement (rel error):")
    print(f"    per-axis std       : {err_axis:.2%}  "
          f"[{'PASS' if err_axis < 0.05 else 'FAIL'}]")
    print(f"    3D RMS             : {err_rms:.2%}  "
          f"[{'PASS' if err_rms < 0.05 else 'FAIL'}]")
    print(f"    Maxwell std        : {err_maxwell:.2%}  "
          f"[{'PASS' if err_maxwell < 0.10 else 'FAIL'}]")

    err_paper_rms = abs(sigma_paper - sigma_3d_rms) / sigma_3d_rms
    err_paper_axis = abs(sigma_paper - sigma_per_axis) / sigma_per_axis
    print()
    print(f"  paper's 2*sqrt(tau) vs 3D RMS:  diff = {err_paper_rms:.2%}")
    print(f"  paper's 2*sqrt(tau) vs per-axis std: diff = {err_paper_axis:.2%}")
    print()
    if err_paper_rms < 0.05:
        print("  -> paper formula 2*sqrt(tau) APPROXIMATES 3D RMS "
              "(off by ~18%)")
    elif err_paper_rms > 0.10:
        print("  -> paper formula 2*sqrt(tau) does NOT match any "
              "standard interpretation; recommend using one of the "
              "MC-confirmed values above.")

    # Pass criterion: at least the per-axis std and 3D RMS analytical
    # forms agree with MC. The paper formula is informational.
    return (err_axis < 0.05) and (err_rms < 0.05)


def case_piecewise_D():
    print("\nCase C: piecewise-constant D, two regimes")
    # First half: D=0.2 (calm), second half D=2.0 (turbulent)
    times = [i * 0.05 for i in range(201)]
    D = [0.2 if t < 5.0 else 2.0 for t in times]
    tau = compute_tau(times, D)
    tau_expected = 0.2 * 5.0 + 2.0 * 5.0
    err = abs(tau[-1] - tau_expected)
    # Trapezoidal rule on a discontinuity has an O(dt * jump) error;
    # for dt=0.05 and jump of 1.8 the expected error is ~0.045.
    expected_trap_err = 0.5 * (2.0 - 0.2) * 0.05
    sigma = (6.0 * tau[-1]) ** 0.5  # 3D RMS
    print(f"  computed tau(10)   = {tau[-1]:.4f}, expected = {tau_expected:.4f}, "
          f"|err| = {err:.4f}  (trap-rule discontinuity error ~ "
          f"{expected_trap_err:.4f})")
    print(f"  3D RMS sigma(10)   = {sigma:.4f} m")
    return err < 2 * expected_trap_err   # <= 2x predicted discretisation error


def main():
    print("validate_uncertainty_diffusion: paper §2.6.3 / §2.7")
    print()
    results = []
    results.append(("tau accumulation (const D)", case_const_D()))
    results.append(("MC vs analytical Gaussian", case_mc_check()))
    results.append(("tau accumulation (piecewise D)", case_piecewise_D()))
    print()
    n_ok = sum(1 for _, r in results if r)
    print(f"=== {n_ok}/{len(results)} checks passed ===")
    for name, r in results:
        print(f"  [{'PASS' if r else 'FAIL'}] {name}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
