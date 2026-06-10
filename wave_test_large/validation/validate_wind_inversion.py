"""Validate the wind-inversion code (paper §2.6.2):

Synthesise a UAV flight under a KNOWN wind field, then check the
inversion recovers the wind to high precision.

Cases:
  1. Uniform wind, straight-line flight, constant airspeed.
     -> trivial, must recover exactly.
  2. Uniform wind, curved flight (S-shaped), varying airspeed.
     -> non-trivial: wind is constant but UAV motion is varied.
  3. Spatially varying wind (linear in x), straight flight.
     -> validates that the residual against a wrong-prediction is
        non-zero in a structured way.

Pure python; no numpy needed for the math, only for the dataclass
output. We re-implement estimate_wind here so we can run on a python
without numpy installed.
"""
import sys

# Pure-python re-implementation of estimate_wind that mirrors
# wind_inversion.estimate_wind line-for-line (so this validates the
# *logic*, not just the numpy version).
def estimate_wind_pp(track, t, q_a, u_pred_fn=None):
    M = len(track)
    out_t = []; out_pos = []; out_qg = []; out_qa = []; out_u = []
    out_pred = []; out_eps = []
    for i in range(M - 1):
        dt = t[i+1] - t[i]
        pos_mid = tuple(0.5 * (track[i+1][k] + track[i][k]) for k in range(3))
        t_mid = 0.5 * (t[i+1] + t[i])
        q_g = tuple((track[i+1][k] - track[i][k]) / dt for k in range(3))
        q_a_mid = tuple(0.5 * (q_a[i][k] + q_a[i+1][k]) for k in range(3))
        u_est = tuple(q_g[k] - q_a_mid[k] for k in range(3))
        out_t.append(t_mid); out_pos.append(pos_mid)
        out_qg.append(q_g); out_qa.append(q_a_mid); out_u.append(u_est)
        if u_pred_fn is not None:
            up = u_pred_fn(pos_mid, t_mid)
            eps = tuple(u_est[k] - up[k] for k in range(3))
            out_pred.append(up); out_eps.append(eps)
    return dict(t_mid=out_t, pos_mid=out_pos, q_g=out_qg, q_a=out_qa,
                u_est=out_u, u_pred=out_pred or None,
                epsilon=out_eps or None)


def synth_uniform_straight(u_wind=(2.0, 1.0, 0.0), q_a_const=(10.0, 0.0, 0.0),
                            n=20, dt=1.0):
    track = [(0.0, 0.0, 50.0)]
    t = [0.0]
    q_a = [q_a_const]
    pos = list(track[0])
    for i in range(1, n):
        for k in range(3):
            pos[k] += (q_a_const[k] + u_wind[k]) * dt
        track.append(tuple(pos))
        t.append(i * dt)
        q_a.append(q_a_const)
    return track, t, q_a, u_wind


def synth_uniform_curved(u_wind=(1.5, 0.5, 0.0), n=40, dt=0.5):
    """S-curve flight, airspeed direction rotates, magnitude varies.

    Uses TRAPEZOIDAL integration so position increments use the
    segment-averaged airspeed — matches the estimator's central-
    difference / midpoint-average discretisation."""
    import math
    speed = 12.0
    q_a = []
    for i in range(n):
        theta = 0.6 * math.sin(2.0 * math.pi * i / max(n - 1, 1))
        q_a.append((speed * math.cos(theta), speed * math.sin(theta), 0.0))

    track = [(0.0, 0.0, 50.0)]
    t = [0.0]
    pos = list(track[0])
    for i in range(n - 1):
        # trapezoidal: pos += 0.5*(q_a[i] + q_a[i+1]) * dt + u_wind * dt
        for k in range(3):
            pos[k] += 0.5 * (q_a[i][k] + q_a[i+1][k]) * dt + u_wind[k] * dt
        track.append(tuple(pos))
        t.append((i + 1) * dt)
    return track, t, q_a, u_wind


def synth_spatial_wind_straight(beta=0.001, q_a_const=(10.0, 0.0, 0.0),
                                 n=30, dt=1.0):
    """Wind increases linearly in x: u_wind(x) = (2 + beta*x, 0, 0).

    Position update uses wind sampled at the SEGMENT MIDPOINT so that
    the estimator (which midpoint-averages) recovers exactly."""
    track = [(0.0, 0.0, 50.0)]
    t = [0.0]
    q_a = [q_a_const for _ in range(n)]
    pos = list(track[0])
    for i in range(1, n):
        # implicit midpoint: pos_new such that
        # pos_new = pos + (q_a + u(0.5*(pos+pos_new)))*dt
        # Linear u(x): u_x = 2 + beta * 0.5 * (pos[0] + pos_new[0])
        # pos_new[0] = pos[0] + (q_a[0] + 2 + beta*0.5*(pos[0]+pos_new[0]))*dt
        # solve for pos_new[0]:
        a = 1 - 0.5 * beta * dt
        rhs = pos[0] + (q_a_const[0] + 2.0 + 0.5 * beta * pos[0]) * dt
        pos_new_0 = rhs / a
        pos_new_1 = pos[1] + q_a_const[1] * dt
        pos_new_2 = pos[2] + q_a_const[2] * dt
        pos = [pos_new_0, pos_new_1, pos_new_2]
        track.append(tuple(pos))
        t.append(i * dt)
    return track, t, q_a, None


def case1():
    print("Case 1: uniform wind (2,1,0), straight flight, constant airspeed")
    track, t, q_a, u_truth = synth_uniform_straight()
    res = estimate_wind_pp(track, t, q_a)
    err = max(max(abs(u[k] - u_truth[k]) for k in range(3))
              for u in res["u_est"])
    print(f"  truth wind = {u_truth}")
    print(f"  recovered  = {tuple(round(res['u_est'][0][k], 6) for k in range(3))}  "
          f"(first segment)")
    print(f"  max |err|  = {err:.3e}")
    ok = err < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] expect zero (round-off only)")
    return ok


def case2():
    print("\nCase 2: uniform wind (1.5,0.5,0), S-curve flight")
    track, t, q_a, u_truth = synth_uniform_curved()
    res = estimate_wind_pp(track, t, q_a)
    errs = [max(abs(u[k] - u_truth[k]) for k in range(3)) for u in res["u_est"]]
    err_mean = sum(errs) / len(errs)
    err_max = max(errs)
    print(f"  truth wind = {u_truth}")
    avg = tuple(sum(u[k] for u in res["u_est"]) / len(res["u_est"]) for k in range(3))
    print(f"  mean recov = {tuple(round(a, 4) for a in avg)}")
    print(f"  max |err|  = {err_max:.3e}, mean = {err_mean:.3e}")
    # Some discretization error from the airspeed midpoint averaging on
    # a curved heading — should still be sub-1% of 12 m/s nominal speed.
    ok = err_max < 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}] expect < 0.05 m/s (midpoint discretisation)")
    return ok


def case3():
    print("\nCase 3: spatially-varying wind, straight flight, "
          "compared against wrong prediction")
    beta = 0.0008
    track, t, q_a, wind_log = synth_spatial_wind_straight(beta=beta)
    # Wrong predicted wind: assumes constant 2 m/s in x (misses the
    # spatial gradient).
    def u_pred_fn(pos, t):
        return (2.0, 0.0, 0.0)
    res = estimate_wind_pp(track, t, q_a, u_pred_fn=u_pred_fn)
    # Truth recovery (no prediction error)
    u_truth_seg = []
    for i in range(len(track) - 1):
        x_mid = 0.5 * (track[i+1][0] + track[i][0])
        u_truth_seg.append((2.0 + beta * x_mid, 0.0, 0.0))
    truth_err = max(
        max(abs(res["u_est"][i][k] - u_truth_seg[i][k]) for k in range(3))
        for i in range(len(u_truth_seg))
    )
    # Residual against wrong prediction: should grow linearly along x
    residual_x = [eps[0] for eps in res["epsilon"]]
    pos_x_mid = [pm[0] for pm in res["pos_mid"]]
    # Linear fit of residual ~ beta * x
    n_seg = len(residual_x)
    sum_x = sum(pos_x_mid); sum_r = sum(residual_x)
    sum_xx = sum(x * x for x in pos_x_mid)
    sum_xr = sum(x * r for x, r in zip(pos_x_mid, residual_x))
    denom = n_seg * sum_xx - sum_x * sum_x
    slope = (n_seg * sum_xr - sum_x * sum_r) / denom
    intercept = (sum_r - slope * sum_x) / n_seg
    print(f"  wind truth at first seg = {u_truth_seg[0]}")
    print(f"  recovered at first seg  = {tuple(round(v, 4) for v in res['u_est'][0])}")
    print(f"  truth recovery max |err| = {truth_err:.3e}")
    print(f"  residual eps_x ~= slope * x + offset  "
          f"(slope={slope:.6f}, expected={beta:.6f}, "
          f"offset={intercept:.4f})")
    ok_truth = truth_err < 1e-9
    ok_slope = abs(slope - beta) < 0.05 * beta
    print(f"  [{'PASS' if ok_truth else 'FAIL'}] truth recovery (round-off)")
    print(f"  [{'PASS' if ok_slope else 'FAIL'}] residual slope matches gradient "
          f"(within 5%)")
    return ok_truth and ok_slope


def main():
    print("validate_wind_inversion: synthetic-truth recovery for "
          "estimate_wind (paper Eq. 38-41)")
    print()
    results = [case1(), case2(), case3()]
    n_ok = sum(results)
    n = len(results)
    print()
    print(f"=== {n_ok}/{n} cases passed ===")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
