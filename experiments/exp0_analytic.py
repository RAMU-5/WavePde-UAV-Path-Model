"""exp0 — Tier-1 analytic accuracy suite (paper §5.2, experiments E1.1–E1.4).

Everything here has an exact (or taut-string-exact) reference solution,
so this is the experiment that calibrates the method's two intrinsic
error sources — spatial discretisation and peak-detection/pulse bias —
before any real-domain comparison.

  E1.1 free space     : empty cube, T*(x) = |x−x0|/c.  dx × τ sweep.
                        Fit T vs r → slope (⇒ c_eff, bias-free) and
                        intercept (⇒ arrival bias ≈ pulse t0).
  E1.2 uniform wind   : moving medium, exact convected arrival time
                        T*(x) = (−d·q + sqrt((d·q)² + (c²−q²)|d|²))/(c²−q²),
                        Mach q/c ∈ {0.1, 0.3, 0.5}.
  E1.3 anisotropy     : open domain, arrival times along the vertical
                        axis above/below the source; slope of T vs
                        distance ⇒ effective cz_up / cz_down. Valid
                        because the vertical group speed of the
                        anisotropic+advected dispersion relation is
                        extremal for purely vertical wavevectors
                        (max upward = cz0 + |w_grav|, the first arrival
                        on the axis IS the vertical ray). NOTE a walled
                        shaft does NOT work here: a Dirichlet duct has
                        no TEM mode, so a low-pass Gaussian pulse is
                        high-pass filtered at the duct cutoff and
                        propagates at the (much slower) modal group
                        velocity — we measured a 26–32%% deficit before
                        switching to the open-domain design.
  E1.4 single obstacle: one box between src and dst; reference = exact
                        taut-string geodesic around the box (isotropic
                        medium ⇒ Fermat). dx sweep incl. a 1.25 m
                        near-reference grid; bias calibrated with an
                        empty-domain companion solve on the same grid.

Conventions: regulatory altitude band OFF (reg_z_low<0), sponge held at
~40 m physical thickness across resolutions, dt = 0.02·dx (CFL 0.3, the
production operating point), pulse width τ specified in *seconds* and
converted to steps per grid.

Results → output/runs/exp0_analytic/{results.json, e1*.npz}.
Figures are generated separately by tools/visualize/paper/fig0*.py.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import wave_api  # noqa: E402  (imports solve_all → sets JIT cache env)
from wave_api import WaveConfig, build_static_fields, solve_forward  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp0_analytic"

C0 = 15.0
SPONGE_M = 40.0          # physical sponge thickness, all resolutions
# Peak threshold, anchored at the production value for dx = 5 m and
# scaled by (dx/5)^3: the source is normalised to sum=1 over CELLS, so
# its continuum strength (and every far-field amplitude) scales with
# dx^3 — a fixed threshold makes fine grids miss weak diffracted first
# arrivals (E1.4 showed +10%/+29% phantom gaps at dx=2.5/1.25 before
# this fix). Do NOT lower the anchor: the source's spatial tail
# (sigma = 1.5 cells, ~1e-5 of peak at 5-7 cells, dx-independent) rings
# up at the pulse maximum t0 and must stay below threshold — hence also
# the >= 8.5-cell clearance for every measurement point.
MIN_PEAK_AMP_5M = 1e-6
SRC_CLEAR_CELLS = 8.5
REG_OFF = dict(reg_z_low=-2.0, reg_z_full=-1.0, reg_low_factor=1.0)


def make_cfg(dx: float, tau_s: float, *, c0=C0, cz_up=None, cz_down=None,
             wind=(0.0, 0.0, 0.0), n_steps: int, dt: float | None = None
             ) -> WaveConfig:
    dt = 0.02 * dx if dt is None else dt
    tau_steps = max(1, round(tau_s / dt))
    t0_steps = max(2, round(2.5 * tau_s / dt))
    return WaveConfig(
        c0=c0,
        cz_up=c0 if cz_up is None else cz_up,
        cz_down=c0 if cz_down is None else cz_down,
        wind=wind, dt=dt, n_steps=n_steps,
        pulse_t0_steps=t0_steps, pulse_tau_steps=tau_steps,
        sponge_layers=max(2, round(SPONGE_M / dx)),
        min_peak_amp=MIN_PEAK_AMP_5M * (dx / 5.0) ** 3, **REG_OFF,
    )


def timed_solve(mask, dx, src_idx, cfg):
    t0 = time.time()
    T = solve_forward(mask, dx, dx, dx, src_idx, cfg=cfg)
    wall = time.time() - t0
    cellsteps = mask.size * cfg.n_steps / wall
    return T, wall, cellsteps


def linfit(r, T):
    """T ≈ slope·r + b  →  c_eff = 1/slope (bias-free), b = arrival bias."""
    A = np.stack([r, np.ones_like(r)], axis=1)
    (slope, b), *_ = np.linalg.lstsq(A, T, rcond=None)
    resid = T - (slope * r + b)
    return float(slope), float(b), resid


def cell_centers(n, dx):
    return (np.arange(n) + 0.5) * dx


# ----------------------------------------------------------------- E1.1
def run_e11(quick: bool, npz: dict) -> dict:
    L = 600.0          # shell 170-250 m clears the source tail even at dx=20
    dxs = [20.0, 10.0] if quick else [20.0, 10.0, 5.0, 2.5]
    taus = [0.4, 1.6] if quick else [0.2, 0.4, 0.8, 1.6]
    rng = np.random.default_rng(0)
    out = []
    for dx in dxs:
        n = int(round(L / dx))
        mask = np.zeros((n, n, n), dtype=bool)
        src = (n // 2, n // 2, n // 2)
        xc = cell_centers(n, dx)
        X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
        sx, sy, sz = xc[src[0]], xc[src[1]], xc[src[2]]
        r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + (Z - sz) ** 2)
        shell = (r >= max(170.0, SRC_CLEAR_CELLS * dx)) & (r <= 250.0)
        idx = np.flatnonzero(shell)
        if idx.size > 4000:
            idx = rng.choice(idx, 4000, replace=False)
        static = None
        for tau in taus:
            dt = 0.02 * dx
            t_end = 250.0 / C0 + 6.5 * tau + 4.0
            cfg = make_cfg(dx, tau, n_steps=math.ceil(t_end / dt))
            if static is None:
                static = build_static_fields(mask, dx, dx, dx, cfg=cfg)
            T, wall, cps = timed_solve(mask, dx, src, cfg)
            Ts = T.ravel()[idx]
            rs = r.ravel()[idx]
            fin = np.isfinite(Ts)
            slope, bias, resid = linfit(rs[fin], Ts[fin])
            rel = resid / (rs[fin] / C0)
            rec = dict(
                dx=dx, tau_s=tau, dt=cfg.dt,
                tau_steps=cfg.pulse_tau_steps, t0_s=cfg.pulse_t0_steps * cfg.dt,
                n=n, n_steps=cfg.n_steps,
                finite_frac=float(fin.mean()),
                c_eff=1.0 / slope, c_eff_rel_err=abs(1.0 / slope - C0) / C0,
                bias_s=bias,
                rms_rel=float(np.sqrt(np.mean(rel ** 2))),
                p95_rel=float(np.percentile(np.abs(rel), 95)),
                max_rel=float(np.max(np.abs(rel))),
                wall_s=wall, cellsteps_per_s=cps,
            )
            out.append(rec)
            npz[f"e11_r_dx{dx:g}_tau{tau:g}"] = rs[fin].astype(np.float32)
            npz[f"e11_T_dx{dx:g}_tau{tau:g}"] = Ts[fin].astype(np.float32)
            print(f"[E1.1] dx={dx:5.2g} tau={tau:.1f}s  c_eff={rec['c_eff']:.3f} "
                  f"bias={bias:.3f}s (t0={rec['t0_s']:.2f}s)  "
                  f"rms_rel={rec['rms_rel']:.4f}  wall={wall:.1f}s", flush=True)
        del static
    # convergence order per tau: fit log(rms) vs log(dx)
    orders = {}
    for tau in taus:
        pts = [(m["dx"], m["rms_rel"]) for m in out if m["tau_s"] == tau]
        if len(pts) >= 3:
            lx = np.log([p[0] for p in pts]); ly = np.log([p[1] for p in pts])
            orders[f"tau{tau:g}"] = float(np.polyfit(lx, ly, 1)[0])
    return dict(runs=out, convergence_order=orders)


# ----------------------------------------------------------------- E1.2
def run_e12(quick: bool, npz: dict) -> dict:
    L, dx, tau = 600.0, 5.0, 0.4
    n = int(round(L / dx))
    qs = [0.0, 4.5] if quick else [0.0, 1.5, 4.5, 7.5]
    mask = np.zeros((n, n, n), dtype=bool)
    src = (n // 2, n // 2, n // 2)
    xc = cell_centers(n, dx)
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    sx, sy, sz = xc[src[0]], xc[src[1]], xc[src[2]]
    dxv, dyv, dzv = X - sx, Y - sy, Z - sz
    r = np.sqrt(dxv ** 2 + dyv ** 2 + dzv ** 2)
    shell = (r >= 170.0) & (r <= 250.0)
    idx = np.flatnonzero(shell)
    rng = np.random.default_rng(1)
    if idx.size > 6000:
        idx = rng.choice(idx, 6000, replace=False)
    dt = 0.05
    bias0 = None
    out = []
    for q in qs:
        t_end = 250.0 / (C0 - q) + 6.5 * tau + 3.0
        cfg = make_cfg(dx, tau, wind=(q, 0.0, 0.0),
                       n_steps=math.ceil(t_end / dt), dt=dt)
        T, wall, cps = timed_solve(mask, dx, src, cfg)
        c2q2 = C0 ** 2 - q ** 2
        dq = dxv.ravel()[idx] * q
        rr = r.ravel()[idx]
        Tstar = (-dq + np.sqrt(dq ** 2 + c2q2 * rr ** 2)) / c2q2
        Ts = T.ravel()[idx]
        fin = np.isfinite(Ts)
        if q == 0.0:
            slope, bias0, _ = linfit(rr[fin], Ts[fin])
        err = (Ts[fin] - bias0 - Tstar[fin]) / Tstar[fin]
        cosw = (dxv.ravel()[idx] / rr)[fin]
        sel = dict(all=np.ones_like(err, bool), along=cosw > 0.8,
                   against=cosw < -0.8, cross=np.abs(cosw) < 0.2)
        rec = dict(q=q, mach=q / C0, dt=dt, n_steps=cfg.n_steps,
                   finite_frac=float(fin.mean()), wall_s=wall)
        for k, m in sel.items():
            if m.any():
                rec[f"med_rel_{k}"] = float(np.median(np.abs(err[m])))
                rec[f"p95_rel_{k}"] = float(np.percentile(np.abs(err[m]), 95))
        out.append(rec)
        print(f"[E1.2] q={q:.1f} (M={q/C0:.2f})  med={rec['med_rel_all']:.4f} "
              f"p95={rec['p95_rel_all']:.4f}  wall={wall:.1f}s", flush=True)
        if q in (0.0, 4.5):  # slices for Fig 2(a)
            npz[f"e12_Tslice_q{q:g}"] = T[:, :, src[2]].astype(np.float32)
    npz["e12_meta"] = np.array([dx, sx, sy, sz, bias0], dtype=np.float64)
    return dict(runs=out, bias0_s=bias0)


# ----------------------------------------------------------------- E1.3
def run_e13(quick: bool, npz: dict) -> dict:
    """Vertical climb/descend asymmetry on the open-domain axis.

    Headline fit on 45–100 m, where the vertical pulse is resolved.
    Beyond ~120 m at dx=5 the on-axis T(d) flattens (apparent speed >
    cz): the vertical wavelength cz·tau is only 1.3–1.9 cells while the
    horizontal operator (c_xy=15, lambda 24 m) is well resolved, so an
    oblique numerical mode overtakes the under-resolved vertical one.
    We run a second grid at dx=2.5 to show this far-field artifact
    recede with resolution — it is documented, not hidden.
    """
    tau = 1.6
    dxs = [5.0] if quick else [5.0, 2.5]
    out = []
    for dx in dxs:
        nxy, nz = int(round(300.0 / dx)), int(round(600.0 / dx))
        mask = np.zeros((nxy, nxy, nz), dtype=bool)
        src = (nxy // 2, nxy // 2, nz // 2)          # z ≈ 300 m
        dt = 0.02 * dx
        d_max = 100.0 if quick else 250.0
        n_steps = math.ceil((d_max / 4.0 + 6.5 * tau + 3.0) / dt)
        cfg = make_cfg(dx, tau, cz_up=6.0, cz_down=4.0,
                       n_steps=n_steps, dt=dt)
        T, wall, _ = timed_solve(mask, dx, src, cfg)
        zc = cell_centers(nz, dx)
        sz = zc[src[2]]
        k0 = max(9, math.ceil(45.0 / dx))
        ks = np.arange(k0, int(d_max / dx) + 1)
        d_up = zc[src[2] + ks] - sz
        T_up = T[src[0], src[1], src[2] + ks]
        d_dn = sz - zc[src[2] - ks]
        T_dn = T[src[0], src[1], src[2] - ks]
        rec = dict(dx=dx, dt=dt, n_steps=n_steps, wall_s=wall)
        for tag, dd, TT, czt in (("up", d_up, T_up, 6.0),
                                 ("down", d_dn, T_dn, 4.0)):
            fin = np.isfinite(TT)
            near = fin & (dd <= 100.0)
            s, b, _ = linfit(dd[near], TT[near])
            rec[f"cz_{tag}_eff"] = 1.0 / s
            rec[f"cz_{tag}_rel_err"] = abs(1.0 / s - czt) / czt
            rec[f"bias_{tag}_s"] = b
            far = fin & (dd >= 160.0)
            if far.sum() > 3:
                sf, _, _ = linfit(dd[far], TT[far])
                rec[f"cz_{tag}_far_eff"] = 1.0 / sf
        npz[f"e13_d_up_dx{dx:g}"], npz[f"e13_T_up_dx{dx:g}"] = d_up, T_up
        npz[f"e13_d_dn_dx{dx:g}"], npz[f"e13_T_dn_dx{dx:g}"] = d_dn, T_dn
        out.append(rec)
        far_note = (f"  far: up {rec.get('cz_up_far_eff', float('nan')):.2f}"
                    f"/down {rec.get('cz_down_far_eff', float('nan')):.2f}"
                    if "cz_up_far_eff" in rec else "")
        print(f"[E1.3] dx={dx:g}  cz_up_eff={rec['cz_up_eff']:.3f} "
              f"(err {rec['cz_up_rel_err']:.2%})  "
              f"cz_down_eff={rec['cz_down_eff']:.3f} "
              f"(err {rec['cz_down_rel_err']:.2%}){far_note}", flush=True)
    return dict(runs=out)


# ----------------------------------------------------------------- E1.4
def _geodesic_around_box(s, d):
    """Exact taut-string length around box x∈[120,180], y∈[90,210],
    z∈[0,60] for src/dst at equal z=30 on either side in x.
    Two candidate homotopy classes: over the top (vertical plane, valid
    while the y-line stays inside the blocked band) and around the +y /
    −y sides (horizontal plane z=30)."""
    over = (math.hypot(120 - s[0], 60 - s[2]) + 60.0
            + math.hypot(d[0] - 180, 60 - d[2]))
    side_n = (math.hypot(120 - s[0], 210 - s[1]) + 60.0
              + math.hypot(d[0] - 180, 210 - d[1]))
    side_s = (math.hypot(120 - s[0], s[1] - 90) + 60.0
              + math.hypot(d[0] - 180, d[1] - 90))
    return min(over, side_n, side_s), dict(over=over, side_north=side_n,
                                           side_south=side_s)


def run_e14(quick: bool, npz: dict) -> dict:
    L, H, tau = 300.0, 120.0, 0.4
    dxs = [10.0, 5.0] if quick else [10.0, 5.0, 2.5, 1.25]
    rng = np.random.default_rng(2)
    out = []
    for dx in dxs:
        n, nzc = int(round(L / dx)), int(round(H / dx))
        xc = cell_centers(n, dx); zc = cell_centers(nzc, dx)
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        mask = np.zeros((n, n, nzc), dtype=bool)
        box = ((X > 120) & (X < 180))[:, :, None] \
            & ((Y > 90) & (Y < 210))[:, :, None] \
            & (zc < 60)[None, None, :]
        mask |= box
        def nearest(p):
            return (int(round(p[0] / dx - 0.5)), int(round(p[1] / dx - 0.5)),
                    int(round(p[2] / dx - 0.5)))
        src, dst = nearest((60, 150, 30)), nearest((240, 150, 30))
        s_m = (xc[src[0]], xc[src[1]], zc[src[2]])
        d_m = (xc[dst[0]], xc[dst[1]], zc[dst[2]])
        Lgeo, classes = _geodesic_around_box(s_m, d_m)
        Tstar = Lgeo / C0
        dt = 0.02 * dx
        n_steps = math.ceil((Tstar + 6.5 * tau + 3.0) / dt)
        cfg = make_cfg(dx, tau, n_steps=n_steps)

        # bias companion: same grid, no building
        empty = np.zeros_like(mask)
        Te, _, _ = timed_solve(empty, dx, src, cfg)
        Z3 = np.broadcast_to(zc[None, None, :], Te.shape)
        X3 = np.broadcast_to(xc[:, None, None], Te.shape)
        Y3 = np.broadcast_to(xc[None, :, None], Te.shape)
        r = np.sqrt((X3 - s_m[0])**2 + (Y3 - s_m[1])**2 + (Z3 - s_m[2])**2)
        safe = ((X3 > 45) & (X3 < 255) & (Y3 > 45) & (Y3 < 255)
                & (Z3 < H - SPONGE_M - dx)
                & (r > max(30, SRC_CLEAR_CELLS * dx)) & (r < 110))
        idx = np.flatnonzero(safe)
        if idx.size > 4000:
            idx = rng.choice(idx, 4000, replace=False)
        fin = np.isfinite(Te.ravel()[idx])
        _, bias, _ = linfit(r.ravel()[idx][fin], Te.ravel()[idx][fin])

        T, wall, cps = timed_solve(mask, dx, src, cfg)
        T_dst = float(T[dst])
        gap = (T_dst - bias - Tstar) / Tstar
        rec = dict(dx=dx, dt=dt, n_steps=n_steps, lambda_m=C0 * tau,
                   geodesic_m=Lgeo, classes=classes, T_star_s=Tstar,
                   T_dst_raw_s=T_dst, bias_s=bias,
                   T_dst_cal_s=T_dst - bias, rel_gap=gap,
                   wall_s=wall, cellsteps_per_s=cps)
        out.append(rec)
        print(f"[E1.4] dx={dx:5.3g}  T*={Tstar:.3f}s  T_cal={T_dst-bias:.3f}s "
              f"gap={gap:+.3%}  wall={wall:.1f}s", flush=True)
        if dx <= 2.5:
            npz[f"e14_T_dx{dx:g}"] = T.astype(np.float32)
            npz[f"e14_mask_dx{dx:g}"] = mask
            npz[f"e14_meta_dx{dx:g}"] = np.array(
                [dx, *s_m, *d_m, Tstar, bias], dtype=np.float64)
    return dict(runs=out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="e11,e12,e13,e14")
    ap.add_argument("--quick", action="store_true",
                    help="small CPU smoke subset")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    npz, results = {}, {"git_sha": sha, "quick": args.quick}
    runners = dict(e11=run_e11, e12=run_e12, e13=run_e13, e14=run_e14)
    t0 = time.time()
    for key in args.only.split(","):
        print(f"=== {key} ===", flush=True)
        results[key] = runners[key](args.quick, npz)
    results["total_wall_s"] = time.time() - t0
    suffix = "_quick" if args.quick else ""
    np.savez_compressed(OUT_DIR / f"exp0_data{suffix}.npz", **npz)
    with open(OUT_DIR / f"results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {results['total_wall_s']:.0f}s → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
