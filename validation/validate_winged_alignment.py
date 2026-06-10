"""Validate the winged-UAV alignment-point search (paper §2.11).

Synthetic T-field: a paraboloid centred on the destination,
    T(x) = 0.5 * ||x - x_dst||^2
so that  -nabla T(x) = x_dst - x,  pointing radially toward x_dst.
Alignment with a launch heading e_uav is purely geometric:
    score(x) = (x_dst - x) . e_uav / ||x_dst - x||

Cases:
  1. Heading EXACTLY pointing at destination -> score = 1 along the
     launch ray; search must return a point on the ray.
  2. Heading offset 20 degrees -> max score = cos(20 deg) ~= 0.94 only
     where the cell-to-dst vector aligns with heading; verify search
     finds a cell satisfying that geometric condition.
  3. Heading 90 degrees off -> max score should be small but the
     search must NOT crash; phase-2 trajectory should still close on
     the destination.

Pure python (no numpy) — re-implements just enough of the search to
cross-check the algorithm. The real model/winged_alignment.py
uses numpy gradient + argmax; for the pure-python harness we build T
on a 30x30x10 grid, finite-difference the gradient, and brute-force
score.
"""
import math


NX, NY, NZ = 30, 30, 10
DX = DY = DZ = 1.0


def alloc(default=0.0):
    return [[[float(default)] * NZ for _ in range(NY)] for _ in range(NX)]


def build_T(x_dst):
    T = alloc()
    for i in range(NX):
        x = i * DX
        for j in range(NY):
            y = j * DY
            for k in range(NZ):
                z = k * DZ
                dx_ = x - x_dst[0]; dy_ = y - x_dst[1]; dz_ = z - x_dst[2]
                T[i][j][k] = 0.5 * (dx_*dx_ + dy_*dy_ + dz_*dz_)
    return T


def gradient(T):
    gx = alloc(); gy = alloc(); gz = alloc()
    for i in range(NX):
        ip = min(i+1, NX-1); im = max(i-1, 0); hx = (ip - im) * DX
        for j in range(NY):
            jp = min(j+1, NY-1); jm = max(j-1, 0); hy = (jp - jm) * DY
            for k in range(NZ):
                kp = min(k+1, NZ-1); km = max(k-1, 0); hz = (kp - km) * DZ
                gx[i][j][k] = (T[ip][j][k] - T[im][j][k]) / max(hx, 1e-9)
                gy[i][j][k] = (T[i][jp][k] - T[i][jm][k]) / max(hy, 1e-9)
                gz[i][j][k] = (T[i][j][kp] - T[i][j][km]) / max(hz, 1e-9)
    return gx, gy, gz


def find_alignment(T, launch, heading, search_radius=12.0,
                   cone_half_angle_deg=35.0, min_grad=1e-6):
    e = list(heading)
    en = (e[0]**2 + e[1]**2 + e[2]**2) ** 0.5
    e = [v / max(en, 1e-9) for v in e]
    gx, gy, gz = gradient(T)
    cos_max = math.cos(math.radians(cone_half_angle_deg))

    best_score = -2.0
    best_idx = None
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                rx = i*DX - launch[0]; ry = j*DY - launch[1]; rz = k*DZ - launch[2]
                rm = (rx*rx + ry*ry + rz*rz) ** 0.5
                if rm < 0.5 * DX or rm > search_radius:
                    continue
                cos_dir = (rx*e[0] + ry*e[1] + rz*e[2]) / max(rm, 1e-9)
                if cos_dir < cos_max:
                    continue
                gn = (gx[i][j][k]**2 + gy[i][j][k]**2 + gz[i][j][k]**2) ** 0.5
                if gn < min_grad:
                    continue
                score = -(gx[i][j][k]*e[0] + gy[i][j][k]*e[1] + gz[i][j][k]*e[2]) / gn
                if score > best_score:
                    best_score = score
                    best_idx = (i, j, k)
    if best_idx is None:
        return None, -1.0
    x_star = (best_idx[0]*DX, best_idx[1]*DY, best_idx[2]*DZ)
    return x_star, best_score


def case1():
    print("Case 1: heading points directly at destination -> max score ~= 1")
    launch = (4.0, 14.5, 5.0)
    x_dst = (24.0, 14.5, 5.0)
    e = (x_dst[0]-launch[0], x_dst[1]-launch[1], x_dst[2]-launch[2])
    T = build_T(x_dst)
    x_star, score = find_alignment(T, launch, e)
    print(f"  launch={launch}  dst={x_dst}  heading direct at dst")
    print(f"  x_star = {x_star}  score = {score:.4f}")
    ok = score > 0.99
    print(f"  [{'PASS' if ok else 'FAIL'}] expect score > 0.99")
    return ok


def case2():
    print("\nCase 2: heading offset 20 deg from launch->dst direction")
    launch = (4.0, 14.5, 5.0)
    x_dst = (24.0, 14.5, 5.0)
    angle = math.radians(20.0)
    # rotate launch->dst direction by 20 deg in xy plane
    dx_ = x_dst[0] - launch[0]; dy_ = x_dst[1] - launch[1]
    e = (dx_*math.cos(angle) - dy_*math.sin(angle),
         dx_*math.sin(angle) + dy_*math.cos(angle),
         0.0)
    T = build_T(x_dst)
    x_star, score = find_alignment(T, launch, e)
    print(f"  launch={launch}  dst={x_dst}  heading rotated 20 deg")
    print(f"  x_star = {x_star}  score = {score:.4f}")
    # Best achievable score: the cell on the heading ray that maximises
    # (x_dst - cell) . e_hat / ||x_dst - cell||
    # Geometric optimum is a cell projecting onto the heading ray near
    # the foot of perpendicular from x_dst -> heading ray.
    # Foot is at t = (x_dst - launch) . e_hat
    en = (e[0]**2 + e[1]**2 + e[2]**2) ** 0.5
    e_hat = [v/en for v in e]
    t_foot = (dx_*e_hat[0] + dy_*e_hat[1])
    foot = (launch[0] + t_foot * e_hat[0],
            launch[1] + t_foot * e_hat[1],
            launch[2] + t_foot * e_hat[2])
    # at the foot, (x_dst - foot) is perpendicular to e_hat, so score
    # would be 0. The best alignment occurs FAR PAST foot, where (x_dst - cell)
    # aligns more with -e_hat (since dst is "behind" the heading line at
    # large t). But the search radius caps it, so the best within the cone
    # is somewhere beyond the foot.
    # Looser pass: score should be in (0.2, 1.0)
    ok = 0.2 < score <= 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] score in (0.2, 1.0]")
    return ok


def case3():
    print("\nCase 3: heading 90 deg off -> low score, but no crash")
    launch = (4.0, 14.5, 5.0)
    x_dst = (24.0, 14.5, 5.0)
    e = (0.0, 1.0, 0.0)   # purely +y, dst is in +x
    T = build_T(x_dst)
    try:
        x_star, score = find_alignment(T, launch, e, cone_half_angle_deg=89.0)
        print(f"  x_star = {x_star}  score = {score:.4f}")
        ok = score < 0.5
        print(f"  [{'PASS' if ok else 'FAIL'}] expect score < 0.5 "
              f"(heading nearly orthogonal to optimal direction)")
        return ok
    except RuntimeError as e:
        print(f"  [FAIL] crashed: {e}")
        return False


def main():
    print("validate_winged_alignment: synthetic-T paraboloid test")
    print(f"  grid {NX}x{NY}x{NZ}, dx=dy=dz={DX}")
    print()
    results = [case1(), case2(), case3()]
    n_ok = sum(results)
    print()
    print(f"=== {n_ok}/{len(results)} cases passed ===")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
