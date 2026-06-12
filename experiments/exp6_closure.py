"""E4.4 + E4.5 — closing the loops (paper Q3 appendix).

E4.5 coupled iteration on the real domain (claim C6):
    run_coupled (solve_coupled.py) from 5 different initialisations of
    (c0, cz_up, cz_down) on 10 station pairs of the 50 m domain;
    record iterations-to-convergence and the dispersion of the fixed
    point across initialisations.

E4.4 wind-inversion closure (paper §2.6.2):
    truth = the E4.3 vortex field at M=0.3. Synthesise UAV flights
    along wave-routed paths under the truth wind (per segment the
    ground velocity solves the same airspeed-ellipsoid quadratic as
    exp4's metric; logged airspeed = ground velocity - truth wind +
    sensor noise sigma=0.3 m/s). Invert per-segment wind
    (wind_inversion.estimate_wind logic: u = q_g - q_a), reconstruct a
    field by Gaussian-kernel regression (L = 600 m, shrink-to-zero
    prior far from data), re-solve the wave with the recovered field,
    and compare station arrival times against the truth-wind solve —
    versus the ignore-wind baseline.

Outputs output/runs/exp6_closure/{exp6_data.npz, results.json}.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp1_dijkstra_gt import nudge_stations  # noqa: E402
from exp4_wind import vortex_wind, build_static_wind, seg_times_wind  # noqa: E402
import wave_api  # noqa: E402
from wave_api import WaveConfig  # noqa: E402
from solve_coupled import run_coupled, trace_path  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp6_closure"

SRC_IDS = list(range(0, 118, 10))
INITS = [  # (c0, cz_up, cz_down)
    ("production", (15.0, 6.0, 4.0)),
    ("high", (18.0, 7.5, 5.5)),
    ("low", (12.0, 4.5, 3.0)),
    ("asym_up", (15.0, 8.0, 3.0)),
    ("asym_dn", (13.0, 5.0, 5.0)),
]
LO_D, HI_D = 700.0, 3000.0
NOISE_SIGMA = 0.3          # airspeed sensor noise [m/s] per component
KERNEL_L = 350.0           # regression length scale [m]; must satisfy
                           # k*L << 1 vs the flow scale (Gaussian kernel
                           # attenuates sin(kx) by exp(-k^2 L^2 / 2):
                           # L=800 on the 3 km vortex kept only 25%)
SHRINK_EPS = 0.05          # shrink-to-zero regulariser weight


# --------------------------------------------------------------- E4.5
def run_e45(mask, dx, dy, dz, stations):
    pairs = []
    st2 = stations[:, :2]
    for s in SRC_IDS[:10]:
        d = np.hypot(*(st2 - st2[s]).T)
        ok = np.where((d >= 1500.0) & (d <= 2500.0))[0]
        if len(ok) == 0:
            continue
        pairs.append((s, int(ok[np.argmin(np.abs(d[ok] - 2000.0))])))
    out = []
    for s, b in pairs:
        for name, p0 in INITS:
            t0 = time.time()
            r = run_coupled(mask, dx, dy, dz, stations[s], stations[b],
                            p0=p0, max_iter=10, alpha=0.5, verbose=False)
            h = r["history"]
            conv = (len(h) < 10
                    or (h[-1]["rel_change"] < 0.01
                        and h[-1]["path_rms"] < 2.0))
            out.append(dict(src=s, dst=b, init=name, p0=p0,
                            n_iter=len(h), converged=bool(conv),
                            p_final=[float(x) for x in r["params"]],
                            wall_s=time.time() - t0))
            print(f"[e45 {s}->{b} {name:10s}] iters={len(h)} "
                  f"conv={conv} p={r['params']}", flush=True)
    return out


# --------------------------------------------------------------- E4.4
def synth_tracks(mask, dx, dy, dz, stations, u3, v3, w3, ix, iy, iz):
    """Wave paths under truth wind -> synthetic (track, t, q_a) logs."""
    cfg = WaveConfig()
    static = build_static_wind(jnp.asarray(mask), dx, dy, dz, u3, v3, w3)
    st_xyz = np.stack([(ix + 0.5) * dx, (iy + 0.5) * dy,
                       (iz + 0.5) * dz], axis=1)
    st2 = st_xyz[:, :2]
    rng = np.random.default_rng(7)
    tracks = []
    T_true = np.full((len(SRC_IDS), len(stations)), np.inf, dtype=np.float32)
    for a, s in enumerate(SRC_IDS):
        T = wave_api._run(static, (int(ix[s]), int(iy[s]), int(iz[s])),
                          dx, dy, dz, cfg)
        T_true[a] = T[ix, iy, iz]
        d = np.hypot(*(st2 - st2[s]).T)
        cand = np.where((d >= LO_D) & (d <= HI_D)
                        & np.isfinite(T_true[a]))[0]
        ang = np.arctan2(*(st2[cand] - st2[s]).T[::-1])
        chosen = [int(cand[np.argmin(np.abs(ang - t))])
                  for t in np.linspace(-np.pi * 0.8, np.pi * 0.8, 5)]
        for b in dict.fromkeys(chosen):
            path, status = trace_path(np.asarray(T, np.float64),
                                      st_xyz[s], st_xyz[b], dx, dy, dz)
            if path.shape[0] < 5:
                continue
            # decimate to ~50 m segments (logging cadence)
            keep = [0]
            for i in range(1, len(path)):
                if np.linalg.norm(path[i] - path[keep[-1]]) >= 50.0:
                    keep.append(i)
            p = path[keep]
            seg_t = seg_times_wind(p, u3, v3, w3, dx, dy, dz)
            if not np.all(np.isfinite(seg_t)):
                continue
            t = np.concatenate([[0.0], np.cumsum(seg_t)])
            mids = 0.5 * (p[1:] + p[:-1])
            # truth wind at segment midpoints (grid sample, like exp4)
            ii = np.clip((mids[:, 0] / dx).astype(int), 0, mask.shape[0] - 1)
            jj = np.clip((mids[:, 1] / dy).astype(int), 0, mask.shape[1] - 1)
            kk = np.clip((mids[:, 2] / dz).astype(int), 0, mask.shape[2] - 1)
            q_tru = np.stack([u3[ii, jj, kk], v3[ii, jj, kk],
                              w3[ii, jj, kk]], axis=1)
            q_g = (p[1:] - p[:-1]) / seg_t[:, None]
            q_a = q_g - q_tru + rng.normal(0.0, NOISE_SIGMA, q_tru.shape)
            # per-segment wind estimate (wind_inversion: u = q_g - q_a)
            u_est = q_g - q_a
            tracks.append(dict(mids=mids, u_est=u_est))
    return tracks, T_true, static


def reconstruct_field(tracks, nx, ny, nz, dx, dy):
    pts = np.concatenate([t["mids"][:, :2] for t in tracks])
    uv = np.concatenate([t["u_est"][:, :2] for t in tracks])
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy
    GX, GY = np.meshgrid(X, Y, indexing="ij")
    g = np.stack([GX.ravel(), GY.ravel()], axis=1)
    rec = np.zeros((g.shape[0], 2))
    # chunked Gaussian-kernel regression with shrink-to-zero prior
    for lo in range(0, g.shape[0], 4096):
        gc = g[lo:lo + 4096]
        d2 = ((gc[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        w = np.exp(-0.5 * d2 / KERNEL_L ** 2)
        rec[lo:lo + 4096] = (w @ uv) / (w.sum(1) + SHRINK_EPS)[:, None]
    u2 = rec[:, 0].reshape(nx, ny)
    v2 = rec[:, 1].reshape(nx, ny)
    u = np.repeat(u2[:, :, None], nz, axis=2).astype(np.float32)
    v = np.repeat(v2[:, :, None], nz, axis=2).astype(np.float32)
    return u, v, np.zeros_like(u), len(pts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    stations = z["stations_xy_m"]
    ix, iy, iz = nudge_stations(mask, stations, dx, dy, dz)

    # ---- E4.5 -----------------------------------------------------------
    e45 = run_e45(mask, dx, dy, dz, stations)

    # ---- E4.4 -----------------------------------------------------------
    u3, v3, w3 = vortex_wind(nx, ny, nz, dx, dy, 4.5)
    tracks, T_true, _ = synth_tracks(mask, dx, dy, dz, stations,
                                     u3, v3, w3, ix, iy, iz)
    print(f"[e44] {len(tracks)} tracks", flush=True)
    ur, vr, wr, n_obs = reconstruct_field(tracks, nx, ny, nz, dx, dy)
    print(f"[e44] field from {n_obs} segment estimates; "
          f"|u_rec|max={np.hypot(ur, vr).max():.2f}", flush=True)

    # recovery quality: relative RMS error of (u,v) at z-slice 5, over
    # all columns and over columns within 1 km of any observation
    obs_xy = np.concatenate([t["mids"][:, :2] for t in tracks])
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy
    GX, GY = np.meshgrid(X, Y, indexing="ij")
    g = np.stack([GX.ravel(), GY.ravel()], axis=1)
    dmin = np.full(g.shape[0], np.inf)
    for lo in range(0, g.shape[0], 4096):
        d2 = ((g[lo:lo + 4096, None, :] - obs_xy[None, :, :]) ** 2).sum(-1)
        dmin[lo:lo + 4096] = np.sqrt(d2.min(1))
    near = (dmin < 1000.0).reshape(nx, ny)
    derr = np.hypot(ur[:, :, 5] - u3[:, :, 5], vr[:, :, 5] - v3[:, :, 5])
    dtru = np.hypot(u3[:, :, 5], v3[:, :, 5])
    rel_all = float(np.sqrt((derr ** 2).mean() / (dtru ** 2).mean()))
    rel_near = float(np.sqrt((derr[near] ** 2).mean()
                             / (dtru[near] ** 2).mean()))
    frac_near = float(near.mean())
    print(f"[e44] field rel RMS err: all={rel_all:.3f} "
          f"near-track={rel_near:.3f} (coverage {frac_near:.0%})",
          flush=True)

    cfg = WaveConfig()
    mask_j = jnp.asarray(mask)
    T_rec = np.full_like(T_true, np.inf)
    T_now = np.full_like(T_true, np.inf)
    static_rec = build_static_wind(mask_j, dx, dy, dz, ur, vr, wr)
    zero = np.zeros_like(ur)
    static_now = build_static_wind(mask_j, dx, dy, dz, zero, zero, zero)
    for a, s in enumerate(SRC_IDS):
        sidx = (int(ix[s]), int(iy[s]), int(iz[s]))
        T_rec[a] = wave_api._run(static_rec, sidx, dx, dy, dz, cfg)[ix, iy, iz]
        T_now[a] = wave_api._run(static_now, sidx, dx, dy, dz, cfg)[ix, iy, iz]
        print(f"[e44 src {s}] resolved", flush=True)

    st2 = np.stack([(ix + 0.5) * dx, (iy + 0.5) * dy], axis=1)
    errs = {"recovered": [], "ignore_wind": []}
    for a, s in enumerate(SRC_IDS):
        d = np.hypot(*(st2 - st2[s]).T)
        m = ((d >= LO_D) & (d <= HI_D) & np.isfinite(T_true[a])
             & np.isfinite(T_rec[a]) & np.isfinite(T_now[a]))
        errs["recovered"].append(
            np.abs(T_rec[a, m] - T_true[a, m]) / T_true[a, m])
        errs["ignore_wind"].append(
            np.abs(T_now[a, m] - T_true[a, m]) / T_true[a, m])
    e44 = {}
    for k, v in errs.items():
        v = np.concatenate(v)
        e44[k] = dict(n=len(v), med=float(np.median(v)),
                      p95=float(np.percentile(v, 95)))
        print(f"[e44 {k}] med={e44[k]['med']:.3%} "
              f"p95={e44[k]['p95']:.3%}", flush=True)

    # ---- save ------------------------------------------------------------
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    np.savez_compressed(
        OUT_DIR / "exp6_data.npz",
        u_true_z5=u3[:, :, 5], v_true_z5=v3[:, :, 5],
        u_rec_z5=ur[:, :, 5], v_rec_z5=vr[:, :, 5],
        track_mids=np.concatenate([t["mids"] for t in tracks]),
        T_true=T_true, T_rec=T_rec, T_now=T_now,
        src_ids=np.array(SRC_IDS, dtype=np.int32),
        e45_n_iter=np.array([r["n_iter"] for r in e45]),
        e45_converged=np.array([r["converged"] for r in e45]),
        e45_p_final=np.array([r["p_final"] for r in e45]))
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(dict(e45=e45, e44=e44, n_obs=n_obs,
                       field_rel_rms_all=rel_all,
                       field_rel_rms_near=rel_near,
                       coverage_frac=frac_near,
                       noise_sigma=NOISE_SIGMA, kernel_L=KERNEL_L,
                       git_sha=sha), f, indent=2)
    print(f"[done] -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
