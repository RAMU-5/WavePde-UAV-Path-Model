"""E3.1 + E3.2 + E3.3 — efficiency claims on the production 5 m domain
(paper Q4, claims C3 adjoint / C4 multicolor / C7 throughput).

E3.1  Adjoint accuracy: for 10 destinations, ONE backward solve
      (q -> -q, w_grav included) vs the 118-row forward column of
      pair_matrix_5m_v2.npz, on valid band pairs (700-3000 m).
E3.2  Multicolor: (a) crosstalk of a 2-source batched solve vs source
      separation (reference = the production single-source rows);
      (b) colour count vs separation threshold for the 118 stations;
      (c) full validation of the largest colour class at the chosen
      safe separation.
E3.3  Throughput: build/cold/warm wall times on 50 m, 5 m, and a 2.5 m
      refined central subdomain; steps/s and cell*steps/s.
Extra: full 5 m T field + traced path from station 50 for Fig.9
      (scale-transfer figure, multirotor panel).

All solves use the production 5 m config (min_peak_amp = 1e-9 = dx^3
scaled — solve_multicolor.solve_batched still carries the 50 m anchor
default and is NOT used here for that reason).

Outputs output/runs/exp7_efficiency/{exp7_data.npz, results.json}.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

import wave_api  # noqa: E402
from wave_api import WaveConfig, build_static_fields, flip_static_fields  # noqa: E402
from solve_multicolor import arrange_by_colour  # noqa: E402
from solve_coupled import trace_path  # noqa: E402
from exp1_dijkstra_gt import polyline_time  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp7_efficiency"

N_DST = 10                    # E3.1 destinations
SEPS_M = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000]
COLOR_SEPS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
# full-class validation at the aggressive end of the kxN tradeoff
# (2-source bins understate k-source crosstalk); 6 km = 2*band HI is
# the provably clean threshold but its largest class is only k=2,
# already covered by the separation bins above
VALIDATE_SEPS = [2000.0, 3000.0]
FIG9_SRC = 50


def production_station_cells(mask, stations_xy_m, dx, dy, dz):
    """Bit-for-bit copy of solve_all.py station snapping (round, then
    horizontal shell r<=5 at same z, then climb) so backward T is read
    at the SAME cells the production matrix used."""
    nx, ny, nz = mask.shape
    ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)
    for b in np.where(mask[ix, iy, iz])[0]:
        ix0, iy0, iz0 = ix[b], iy[b], iz[b]
        found = False
        for r in range(1, 6):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = ix0 + di, iy0 + dj
                    if not (0 < i < nx - 1 and 0 < j < ny - 1):
                        continue
                    if not mask[i, j, iz0]:
                        ix[b], iy[b] = i, j
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            k = iz0
            while k < nz - 1 and mask[ix0, iy0, k]:
                k += 1
            iz[b] = k
    return ix, iy, iz


def batched_run(static, src_cells, dx, dy, dz, cfg):
    """One solve with a sum-of-Gaussians source (linearity, §2.9.4);
    same jit entry + min_peak as the single-source production path."""
    src = sum(wave_api.gaussian_source(static.free.shape, c,
                                       sigma=cfg.source_sigma_cells)
              for c in src_cells)
    T = wave_api._solve_jit(static.c2_xy, static.c2_z,
                            static.u, static.v, static.w,
                            static.ax, static.ay, static.az,
                            static.one_minus_damp, static.free, src,
                            dx, dy, dz, cfg.n_steps,
                            cfg.dt, cfg.pulse_t0_steps,
                            cfg.pulse_tau_steps, cfg.min_peak_amp)
    jax.block_until_ready(T)
    return np.asarray(jax.device_get(T))


def timed_solves(mask, dx, dy, dz, cfg, src_idx, n_warm=3):
    """Static build + cold (first, incl. compile) + warm median."""
    t0 = time.time()
    static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
    t_static = time.time() - t0
    t0 = time.time()
    T = wave_api.solve_forward(mask, dx, dy, dz, src_idx,
                               cfg=cfg, static=static)
    t_cold = time.time() - t0
    warms = []
    for _ in range(n_warm):
        t0 = time.time()
        T = wave_api.solve_forward(mask, dx, dy, dz, src_idx,
                                   cfg=cfg, static=static)
        warms.append(time.time() - t0)
    return static, T, t_static, t_cold, float(np.median(warms))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {}

    z = np.load(ROOT / "input" / "domain_5m.npz")
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    st_xyz_m = z["stations_xy_m"]
    n_st = len(st_xyz_m)
    cfg5 = WaveConfig(min_peak_amp=1e-6 * (dx / 50.0) ** 3)
    ix, iy, iz = production_station_cells(mask, st_xyz_m, dx, dy, dz)
    cells = list(zip(ix.tolist(), iy.tolist(), iz.tolist()))

    m = np.load(ROOT / "output" / "wave" / "pair_matrix_5m_v2.npz")
    T_mat = m["T_matrix"]            # [src, dst]
    valid = m["valid_mask"]
    D = m["distances_m"]

    # ---------------- E3.3 part 1: 5 m timings (+ Fig.9 field) -------
    print("[5m] static + cold + warm ...", flush=True)
    static5, T_fig9, t_st5, t_cold5, t_warm5 = timed_solves(
        mask, dx, dy, dz, cfg5, cells[FIG9_SRC])
    res["t_5m"] = dict(static_s=t_st5, cold_s=t_cold5, warm_s=t_warm5,
                       n_cells=int(mask.size), n_steps=cfg5.n_steps,
                       dt=cfg5.dt)
    print(f"[5m] static {t_st5:.1f}s cold {t_cold5:.1f}s "
          f"warm {t_warm5:.1f}s", flush=True)

    # Fig.9 multirotor panel: slice + traced path from station 50
    band = np.where(valid[FIG9_SRC] & np.isfinite(T_mat[FIG9_SRC]))[0]
    dst9 = int(band[np.argmax(D[FIG9_SRC, band])])
    p0 = np.array([(ix[FIG9_SRC] + .5) * dx, (iy[FIG9_SRC] + .5) * dy,
                   (iz[FIG9_SRC] + .5) * dz])
    p1 = np.array([(ix[dst9] + .5) * dx, (iy[dst9] + .5) * dy,
                   (iz[dst9] + .5) * dz])
    path9, status9 = trace_path(np.asarray(T_fig9, np.float64), p0, p1,
                                dx, dy, dz)
    iz_slice = 10                                  # 52.5 m AGL
    np.savez_compressed(
        OUT_DIR / "fig9_multirotor.npz",
        T_slice=T_fig9[:, :, iz_slice], mask_slice=mask[:, :, iz_slice],
        path=path9, path_status=status9,
        path_time_s=polyline_time(path9.astype(np.float64), dz)
        if status9 == "reached" else np.inf,
        src=FIG9_SRC, dst=dst9, dx=dx, dz=dz, z_m=(iz_slice + .5) * dz,
        st_xyz=np.stack([(ix + .5) * dx, (iy + .5) * dy,
                         (iz + .5) * dz], 1))
    print(f"[fig9] src {FIG9_SRC} -> {dst9}: {status9}", flush=True)
    del T_fig9

    # ---------------- E3.1: backward vs forward column ----------------
    rng = np.random.default_rng(0)
    cand = [d for d in range(n_st)
            if (valid[:, d] & np.isfinite(T_mat[:, d])).sum() >= 20]
    dsts = sorted(rng.choice(cand, size=N_DST, replace=False).tolist())
    print(f"[e31] destinations {dsts}", flush=True)
    e31 = []
    for d in dsts:
        t0 = time.time()
        Tb = wave_api.solve_backward(mask, dx, dy, dz, cells[d],
                                     cfg=cfg5, static=static5)
        wall = time.time() - t0
        tb_st = Tb[ix, iy, iz]
        del Tb
        rows = np.where(valid[:, d] & np.isfinite(T_mat[:, d])
                        & np.isfinite(tb_st) & (np.arange(n_st) != d))[0]
        rel = np.abs(tb_st[rows] - T_mat[rows, d]) / T_mat[rows, d]
        e31.append(dict(dst=d, n=len(rows), wall_s=wall,
                        med=float(np.median(rel)),
                        p95=float(np.percentile(rel, 95)),
                        max=float(rel.max()),
                        n_missing=int(np.sum(~np.isfinite(tb_st[
                            valid[:, d] & np.isfinite(T_mat[:, d])]))),
                        rel=rel.tolist()))
        print(f"[e31 dst {d:3d}] n={len(rows)} med {np.median(rel):.4f} "
              f"max {rel.max():.4f} wall {wall:.0f}s", flush=True)
    allrel = np.concatenate([np.array(e["rel"]) for e in e31])
    res["e31"] = dict(dsts=dsts, n_pairs=int(allrel.size),
                      med=float(np.median(allrel)),
                      p95=float(np.percentile(allrel, 95)),
                      max=float(allrel.max()),
                      wall_backward_s=float(np.median(
                          [e["wall_s"] for e in e31])),
                      wall_forward_column_s=t_warm5 * n_st,
                      per_dst=[{k: v for k, v in e.items() if k != "rel"}
                               for e in e31])
    print(f"[e31] ALL med {res['e31']['med']:.4f} "
          f"p95 {res['e31']['p95']:.4f} max {res['e31']['max']:.4f}",
          flush=True)

    # ---------------- E3.2a: crosstalk vs separation ------------------
    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)
    e32a = []
    used = set()
    for sep in SEPS_M:
        # closest unused pair to the target separation
        order = np.argsort(np.abs(D2 - sep), axis=None)
        for o in order:
            i, j = np.unravel_index(o, D2.shape)
            if i < j and i not in used and j not in used:
                break
        used.update((i, j))
        i, j = int(i), int(j)
        t0 = time.time()
        Tb = batched_run(static5, [cells[i], cells[j]], dx, dy, dz, cfg5)
        wall = time.time() - t0
        tb_st = Tb[ix, iy, iz]
        del Tb
        entry = dict(sep_m=float(D[i, j]), src_a=i, src_b=j, wall_s=wall)
        for tag, s in (("a", i), ("b", j)):
            ks = np.where(valid[s] & np.isfinite(T_mat[s])
                          & np.isfinite(tb_st))[0]
            rel = (tb_st[ks] - T_mat[s, ks]) / T_mat[s, ks]
            entry[f"{tag}_n"] = len(ks)
            entry[f"{tag}_med"] = float(np.median(np.abs(rel)))
            entry[f"{tag}_max"] = float(np.abs(rel).max())
            entry[f"{tag}_frac_bad"] = float(np.mean(np.abs(rel) > .01))
        e32a.append(entry)
        print(f"[e32a sep {entry['sep_m']:5.0f}m {i}+{j}] "
              f"max |err| {max(entry['a_max'], entry['b_max']):.4f} "
              f"frac>1% {max(entry['a_frac_bad'], entry['b_frac_bad']):.2f}",
              flush=True)
    res["e32_crosstalk"] = e32a

    # ---------------- E3.2b: colours vs separation --------------------
    res["e32_colours"] = []
    for sep in COLOR_SEPS:
        classes = arrange_by_colour(st_xyz_m, sep)
        res["e32_colours"].append(dict(
            sep_m=sep, n_colours=len(classes),
            class_sizes=sorted((len(c) for c in classes), reverse=True)))
        print(f"[e32b sep {sep}] {len(classes)} colours, "
              f"largest {max(len(c) for c in classes)}", flush=True)

    # ---------------- E3.2c: largest colour class, aggressive end -----
    res["e32_bigclass"] = []
    for vsep in VALIDATE_SEPS:
        classes = arrange_by_colour(st_xyz_m, vsep)
        big = max(classes, key=len)
        print(f"[e32c sep {vsep:.0f}] batch of {len(big)}: {big}",
              flush=True)
        t0 = time.time()
        Tb = batched_run(static5, [cells[s] for s in big],
                         dx, dy, dz, cfg5)
        wall_big = time.time() - t0
        tb_st = Tb[ix, iy, iz]
        del Tb
        rels = []
        for s in big:
            ks = np.where(valid[s] & np.isfinite(T_mat[s])
                          & np.isfinite(tb_st))[0]
            rels.append((tb_st[ks] - T_mat[s, ks]) / T_mat[s, ks])
        rels = np.abs(np.concatenate(rels))
        res["e32_bigclass"].append(dict(
            sep_m=vsep, k=len(big), members=[int(s) for s in big],
            n_colours_total=len(classes), wall_s=wall_big,
            n_pairs=int(rels.size), med=float(np.median(rels)),
            p95=float(np.percentile(rels, 95)), max=float(rels.max()),
            frac_bad=float(np.mean(rels > .01))))
        print(f"[e32c sep {vsep:.0f}] k={len(big)} "
              f"med {np.median(rels):.4f} max {rels.max():.4f} "
              f"frac>1% {np.mean(rels > .01):.2f} wall {wall_big:.0f}s",
              flush=True)
    del static5

    # ---------------- E3.3 part 2: 50 m ------------------------------
    z50 = np.load(ROOT / "input" / "domain.npz")
    mask50 = z50["mask"]
    d50 = (float(z50["dx"]), float(z50["dy"]), float(z50["dz"]))
    cfg50 = WaveConfig()                          # 50 m anchor defaults
    c50 = (mask50.shape[0] // 2, mask50.shape[1] // 2,
           mask50.shape[2] // 2)
    if mask50[c50]:
        c50 = tuple(int(v) for v in
                    np.argwhere(~mask50)[len(np.argwhere(~mask50)) // 2])
    _, _, t_st50, t_cold50, t_warm50 = timed_solves(
        mask50, *d50, cfg50, c50)
    res["t_50m"] = dict(static_s=t_st50, cold_s=t_cold50, warm_s=t_warm50,
                        n_cells=int(mask50.size), n_steps=cfg50.n_steps,
                        dt=cfg50.dt)
    print(f"[50m] cold {t_cold50:.1f}s warm {t_warm50:.2f}s", flush=True)

    # ---------------- E3.3 part 3: 2.5 m refined subdomain ------------
    # central 600x600x42 cells of the 5 m mask, refined 2x -> CFL needs
    # dt = 0.05 at dx = 2.5 with c0 = 15 (0.1 is marginally unstable)
    x0, y0 = (mask.shape[0] - 600) // 2, (mask.shape[1] - 600) // 2
    sub = mask[x0:x0 + 600, y0:y0 + 600, :]
    m25 = np.repeat(np.repeat(np.repeat(sub, 2, 0), 2, 1), 2, 2)
    cfg25 = WaveConfig(dt=0.05, min_peak_amp=1e-6 * (2.5 / 50.0) ** 3)
    c25 = (600, 600, 20)
    if m25[c25]:
        free = np.argwhere(~m25[:, :, 20])
        c25 = (int(free[len(free) // 2][0]), int(free[len(free) // 2][1]), 20)
    _, _, t_st25, t_cold25, t_warm25 = timed_solves(
        m25, 2.5, 2.5, 2.5, cfg25, c25)
    res["t_2p5m"] = dict(static_s=t_st25, cold_s=t_cold25,
                         warm_s=t_warm25, n_cells=int(m25.size),
                         n_steps=cfg25.n_steps, dt=cfg25.dt)
    print(f"[2.5m] cold {t_cold25:.1f}s warm {t_warm25:.1f}s", flush=True)

    # ---------------- pipeline totals (Fig.5) -------------------------
    res["pipelines"] = dict(
        warm_solve_s=t_warm5,
        full_matrix_naive_s=n_st * t_warm5,
        full_matrix_multicolor_s={
            str(c["sep_m"]): c["n_colours"] * t_warm5
            for c in res["e32_colours"]},
        one_destination_forward_s=n_st * t_warm5,
        one_destination_backward_s=res["e31"]["wall_backward_s"])

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    res["git_sha"] = sha
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(OUT_DIR / "exp7_data.npz",
                        e31_rel=allrel,
                        dsts=np.array(dsts))
    print(f"[done] -> {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
