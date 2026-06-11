"""E4.1 — wavelength operating point on the real 5 m domain (paper Q3).

Sweep the source pulse width tau (the only wavelength knob used in
production; spatial sigma stays at 1.5 cells) and measure the
optimality gap against the E2.1 discrete ground truth (anisotropic
Dijkstra + metric-aware string pulling, gt_domain_5m.npz) for the same
12 sources x 118 stations and the same 700-3000 m valid-pair band.

For each tau we also run an empty-domain companion (600 m cube, dx=5,
isotropic c0, regulation off) and fit T vs r on a 170-250 m shell:
the intercept gives the per-tau arrival bias (~= pulse t0, cf. E1.1),
which is subtracted before computing gaps — Q2 used the tau=0.4
production value 0.945 s; here every tau gets its own calibration.

Dominant wavelength convention (matches test_wavelength.py):
    lambda ~= 2 pi c0 tau     (low-pass Gaussian pulse, c0 = 15 m/s)

Outputs output/runs/exp3_wavelength/{exp3_data.npz, results.json};
figure by tools/visualize/paper/fig06_wavelength.py.
"""
from __future__ import annotations
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp1_dijkstra_gt import nudge_stations  # noqa: E402
import wave_api  # noqa: E402
from wave_api import WaveConfig, build_static_fields, solve_forward  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp3_wavelength"
CHK = OUT_DIR / "exp3_chk.npz"

C0 = 15.0
DT = 0.1                      # production time step on the 5 m grid
TAUS = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]   # s; production point is 0.4
MIN_PEAK = 1e-6 * (5.0 / 50.0) ** 3      # dx^3-scaled production threshold
REG_OFF = dict(reg_z_low=-2.0, reg_z_full=-1.0, reg_low_factor=1.0)


def pulse_steps(tau_s: float) -> tuple[int, int]:
    tau_steps = max(1, round(tau_s / DT))
    t0_steps = max(2, round(2.5 * tau_s / DT))
    return tau_steps, t0_steps


def make_cfg(tau_s: float) -> WaveConfig:
    """Production 5 m config except for the pulse width; the simulated
    window is extended by t0 so late tau keeps the full 450 s reach."""
    tau_steps, t0_steps = pulse_steps(tau_s)
    return WaveConfig(dt=DT, n_steps=4500 + t0_steps,
                      pulse_t0_steps=t0_steps, pulse_tau_steps=tau_steps,
                      min_peak_amp=MIN_PEAK)


def bias_companion(tau_s: float) -> dict:
    """Empty 600 m cube at dx=5: linfit T vs r on a 170-250 m shell.
    Intercept = arrival bias for this tau (subtract before gaps)."""
    dx = 5.0
    n = int(round(600.0 / dx))
    mask = np.zeros((n, n, n), dtype=bool)
    src = (n // 2, n // 2, n // 2)
    xc = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r = np.sqrt((X - xc[src[0]]) ** 2 + (Y - xc[src[1]]) ** 2
                + (Z - xc[src[2]]) ** 2)
    shell = (r >= 170.0) & (r <= 250.0)
    idx = np.flatnonzero(shell)
    rng = np.random.default_rng(0)
    if idx.size > 4000:
        idx = rng.choice(idx, 4000, replace=False)

    tau_steps, t0_steps = pulse_steps(tau_s)
    t_end = 250.0 / C0 + 2.5 * tau_s + 6.0 * tau_s + 4.0
    cfg = WaveConfig(c0=C0, cz_up=C0, cz_down=C0, dt=DT,
                     n_steps=math.ceil(t_end / DT),
                     pulse_t0_steps=t0_steps, pulse_tau_steps=tau_steps,
                     min_peak_amp=MIN_PEAK, **REG_OFF)
    T = solve_forward(mask, dx, dx, dx, src, cfg=cfg)
    Ts, rs = T.ravel()[idx], r.ravel()[idx]
    fin = np.isfinite(Ts)
    A = np.stack([rs[fin], np.ones(fin.sum())], axis=1)
    (slope, bias), *_ = np.linalg.lstsq(A, Ts[fin], rcond=None)
    return dict(tau_s=tau_s, bias_s=float(bias), c_eff=float(1.0 / slope),
                finite_frac=float(fin.mean()), t0_s=t0_steps * DT)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] domain_5m.npz", flush=True)
    z = np.load(ROOT / "input" / "domain_5m.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations = z["stations_xy_m"]
    n_st = len(stations)
    ix, iy, iz = nudge_stations(mask, stations, dx, dy, dz)

    gt = np.load(ROOT / "output" / "runs" / "exp1_dijkstra" / "gt_domain_5m.npz")
    src_list = [int(s) for s in gt["src_idx"]]
    n_src, n_tau = len(src_list), len(TAUS)
    print(f"[setup] {n_tau} tau x {n_src} sources, {n_st} stations", flush=True)

    # -- per-tau bias calibration (cheap, empty cube) -------------------
    print("[companion] bias calibration per tau", flush=True)
    companions = [bias_companion(t) for t in TAUS]
    for c in companions:
        print(f"  tau={c['tau_s']:.1f}s  bias={c['bias_s']:.3f}s "
              f"(t0={c['t0_s']:.2f}s)  c_eff={c['c_eff']:.3f}  "
              f"finite={c['finite_frac']:.3f}", flush=True)

    # -- main sweep ------------------------------------------------------
    if CHK.exists():
        c = np.load(CHK)
        T_at = c["T_at"]
        done = c["done"]
        print(f"[resume] {done.sum()}/{n_tau * n_src}", flush=True)
    else:
        T_at = np.full((n_tau, n_src, n_st), np.inf, dtype=np.float32)
        done = np.zeros((n_tau, n_src), dtype=bool)

    # static fields do not depend on the pulse -> build once
    cfg0 = make_cfg(TAUS[0])
    static = build_static_fields(mask, dx, dy, dz, cfg=cfg0)

    t_start = time.time()
    for a, tau in enumerate(TAUS):
        cfg = make_cfg(tau)
        for b, s in enumerate(src_list):
            if done[a, b]:
                continue
            t1 = time.time()
            T = wave_api.solve_forward(
                mask, dx, dy, dz,
                (int(ix[s]), int(iy[s]), int(iz[s])),
                cfg=cfg, static=static)
            T_at[a, b] = T[ix, iy, iz]
            done[a, b] = True
            np.savez_compressed(CHK, T_at=T_at, done=done)
            print(f"[tau={tau:.1f} src {s:3d}] {time.time()-t1:5.1f}s  "
                  f"finite {np.isfinite(T_at[a, b]).sum()}/{n_st}  "
                  f"elapsed {(time.time()-t_start)/60:.1f}min", flush=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    np.savez_compressed(
        OUT_DIR / "exp3_data.npz",
        T_at=T_at, taus=np.array(TAUS, dtype=np.float32),
        src_idx=np.array(src_list, dtype=np.int32),
        biases=np.array([c["bias_s"] for c in companions], dtype=np.float32),
        c_effs=np.array([c["c_eff"] for c in companions], dtype=np.float32),
        st_ix=ix.astype(np.int32), st_iy=iy.astype(np.int32),
        st_iz=iz.astype(np.int32), stations_xy_m=stations,
        min_peak_amp=MIN_PEAK, git_sha=sha)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(dict(taus=TAUS, companions=companions,
                       min_peak_amp=MIN_PEAK, dt=DT, git_sha=sha), f, indent=2)
    print(f"[done] -> {OUT_DIR}/exp3_data.npz", flush=True)


if __name__ == "__main__":
    main()
