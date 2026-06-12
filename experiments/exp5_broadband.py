"""E4.2 — broadband value on hard pairs (paper Q3 appendix, claim C5).

Hard pairs := the valid pairs with the most NEGATIVE optimality gap at
tau = 1.6 s in E4.1 — the places where the 150 m wavelength diffracts
through street canyons more than a point UAV could ("cuts corners"),
restricted to the 3 sources that contribute most of them.

For each band we trace the -grad T descent path on the full T field
and score it like the rest of the paper:
  success    = trace reaches the source within 2 cells
  path time  = exact continuum metric time along the traced path
               (exp1 polyline_time), compared to the E2.1 reference
The bands:
  short      tau = 0.2 s  (lambda ~ 19 m)
  long       tau = 1.6 s  (lambda ~ 151 m)
  broadband  per-cell earliest calibrated arrival min(T02-b02, T16-b16)
             — by linearity (T0.2 validation) this is what a two-band
             source yields; same 2 solves as short+long: equal cost.

Outputs output/runs/exp5_broadband/{exp5_data.npz, results.json}.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp1_dijkstra_gt import nudge_stations, polyline_time  # noqa: E402
from exp3_wavelength_gap import make_cfg  # noqa: E402
import wave_api  # noqa: E402
from wave_api import build_static_fields  # noqa: E402
from solve_coupled import trace_path  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp5_broadband"

TAU_SHORT, TAU_LONG = 0.2, 1.6
N_HARD = 20
N_SOURCES = 3
LO_D, HI_D = 700.0, 3000.0


def pick_hard_pairs():
    d3 = np.load(ROOT / "output" / "runs" / "exp3_wavelength" / "exp3_data.npz")
    gt = np.load(ROOT / "output" / "runs" / "exp1_dijkstra" / "gt_domain_5m.npz")
    taus = d3["taus"]
    il = int(np.argmin(np.abs(taus - TAU_LONG)))
    bias_l = float(d3["biases"][il])
    st = d3["stations_xy_m"]
    D = np.sqrt(((st[:, None, :2] - st[None, :, :2]) ** 2).sum(-1))
    rows = []
    for b, s in enumerate(d3["src_idx"]):
        tw = d3["T_at"][il, b] - bias_l
        ts = gt["T_str"][b]
        m = (np.isfinite(tw) & np.isfinite(ts)
             & (D[int(s)] >= LO_D) & (D[int(s)] <= HI_D))
        for j in np.where(m)[0]:
            rows.append((int(s), int(j),
                         (tw[j] - ts[j]) / ts[j], float(ts[j])))
    rows.sort(key=lambda r: r[2])                 # most negative first
    # keep the N_SOURCES sources that dominate the worst-40 tail
    from collections import Counter
    cnt = Counter(r[0] for r in rows[:2 * N_HARD])
    keep = {s for s, _ in cnt.most_common(N_SOURCES)}
    hard = [r for r in rows if r[0] in keep][:N_HARD]
    return hard


def path_collides(path, mask, dx, dy, dz, step_m=2.5):
    """True if any sample along the polyline lies inside a building."""
    if path.shape[0] < 2:
        return False
    seg = np.diff(path, axis=0)
    L = np.linalg.norm(seg, axis=1)
    for a, s, l in zip(path[:-1], seg, L):
        n = max(2, int(l / step_m) + 1)
        pts = a[None, :] + np.linspace(0, 1, n)[:, None] * s[None, :]
        ii = np.clip(np.floor(pts[:, 0] / dx).astype(int), 0, mask.shape[0] - 1)
        jj = np.clip(np.floor(pts[:, 1] / dy).astype(int), 0, mask.shape[1] - 1)
        kk = np.clip(np.floor(pts[:, 2] / dz).astype(int), 0, mask.shape[2] - 1)
        if mask[ii, jj, kk].any():
            return True
    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hard = pick_hard_pairs()
    srcs = sorted({h[0] for h in hard})
    print(f"[hard] {len(hard)} pairs from sources {srcs}; "
          f"worst gap {hard[0][2]:+.1%}", flush=True)

    d3 = np.load(ROOT / "output" / "runs" / "exp3_wavelength" / "exp3_data.npz")
    taus = list(np.round(d3["taus"], 3))
    b02 = float(d3["biases"][taus.index(TAU_SHORT)])
    b16 = float(d3["biases"][taus.index(TAU_LONG)])

    z = np.load(ROOT / "input" / "domain_5m.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    ix, iy, iz = nudge_stations(mask, z["stations_xy_m"], dx, dy, dz)
    st_xyz = np.stack([(ix + 0.5) * dx, (iy + 0.5) * dy,
                       (iz + 0.5) * dz], axis=1)
    gt = np.load(ROOT / "output" / "runs" / "exp1_dijkstra" / "gt_domain_5m.npz")
    tref_of = {int(s): gt["T_str"][b] for b, s in enumerate(gt["src_idx"])}

    cfg_s, cfg_l = make_cfg(TAU_SHORT), make_cfg(TAU_LONG)
    static = build_static_fields(mask, dx, dy, dz, cfg=cfg_s)

    results = []
    for s in srcs:
        sidx = (int(ix[s]), int(iy[s]), int(iz[s]))
        t0 = time.time()
        T_s = wave_api.solve_forward(mask, dx, dy, dz, sidx,
                                     cfg=cfg_s, static=static)
        T_l = wave_api.solve_forward(mask, dx, dy, dz, sidx,
                                     cfg=cfg_l, static=static)
        print(f"[src {s}] 2 solves in {time.time()-t0:.0f}s", flush=True)
        T_bb = np.fmin(T_s - b02, T_l - b16)      # calibrated earliest
        fields = dict(short=T_s, long=T_l, broadband=T_bb)
        for (ss, j, gap16, tref) in hard:
            if ss != s:
                continue
            for band, T in fields.items():
                for tracer in ("raw", "smooth"):
                    t1 = time.time()
                    Tt = np.asarray(T, dtype=np.float64)
                    if tracer == "smooth":
                        # normalised convolution: smooth T inside the
                        # finite region only (a smoothed 1e6 plateau
                        # bleeds ~300 s into 40 m streets and overwhelms
                        # the gradient — v2 failure mode); buildings
                        # keep their sharp repulsive step
                        fin = np.isfinite(Tt)
                        num = gaussian_filter(np.where(fin, Tt, 0.0), 2.0)
                        den = gaussian_filter(fin.astype(np.float64), 2.0)
                        Tsm = num / np.maximum(den, 1e-9)
                        Tt = np.where(fin, Tsm, Tt[fin].max() + 1e6)
                    path, status = trace_path(Tt, st_xyz[s], st_xyz[j],
                                              dx, dy, dz)
                    # collision audit: sample the path every half-cell
                    coll = path_collides(path, mask, dx, dy, dz)
                    ok = status == "reached" and not coll
                    pt = polyline_time(path.astype(np.float64), dz) \
                        if ok else np.inf
                    results.append(dict(
                        src=s, dst=j, band=band, tracer=tracer,
                        status=status, collides=bool(coll),
                        n_pts=int(len(path)),
                        path_time_s=float(pt), t_ref_s=tref,
                        path_gap=(float(pt) - tref) / tref
                        if np.isfinite(pt) else None,
                        trace_wall_s=time.time() - t1))
                    print(f"  [{s}->{j} {band:9s} {tracer:6s}] "
                          f"{status:12s} coll={coll} "
                          f"path_t={pt:8.1f}s ref={tref:7.1f}s", flush=True)
        del T_s, T_l, T_bb, fields

    summary = {}
    for tracer in ("raw", "smooth"):
        for band in ("short", "long", "broadband"):
            rs = [r for r in results
                  if r["band"] == band and r["tracer"] == tracer]
            ok = [r for r in rs
                  if r["status"] == "reached" and not r["collides"]]
            gaps = [r["path_gap"] for r in ok]
            summary[f"{band}_{tracer}"] = dict(
                n=len(rs), success=len(ok),
                n_collide=sum(r["collides"] for r in rs),
                med_path_gap=float(np.median(gaps)) if gaps else None,
                p95_path_gap=float(np.percentile(np.abs(gaps), 95))
                if gaps else None)
            print(f"[summary {band}/{tracer}] {len(ok)}/{len(rs)} ok, "
                  f"med path gap "
                  f"{summary[f'{band}_{tracer}']['med_path_gap']}",
                  flush=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(dict(hard_pairs=[(h[0], h[1], h[2]) for h in hard],
                       bias_short=b02, bias_long=b16,
                       per_trace=results, summary=summary,
                       git_sha=sha), f, indent=2)
    print(f"[done] -> {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
