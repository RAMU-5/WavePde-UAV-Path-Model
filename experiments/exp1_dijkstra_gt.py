"""exp1 — E2.1: discrete ground truth for the 118-station travel-time
matrix (paper §5.3, Fig. 4).

Reference = anisotropic Dijkstra on the 26-neighbour voxel graph,
followed by string-pulling: each graph path is greedily shortcut with
line-of-sight checks and re-timed with the exact continuum metric. The
shortcutting matters: the 26-direction graph overestimates travel time
by up to ~8% in directions like (2,1,0) (it must zig-zag), so the raw
graph time is an upper bound, not a fair reference. Both numbers are
saved — the gap between them is itself a paper-worthy methodological
point.

Metric (matches the wave solver's medium exactly):
  - horizontal speed  a_h(z) = C0 * reg(z)   (regulatory altitude band)
  - vertical          ellipsoid semi-axis a_z = (cz_up+cz_down)/2 = 5
                      plus constant upward drift w = (cz_up-cz_down)/2 = 1
  - segment time for displacement d = (dx,dy,dz): the t > 0 solving
      (dx^2+dy^2)/a_h^2 + (dz - w t)^2/a_z^2 = t^2
    (pure up: h/6, pure down: h/4 — the wave w_grav encoding.)

Station cells use the EXACT nudge logic of solve_one_source.py so both
methods route between identical cells.

Usage:
  python exp1_dijkstra_gt.py --domain domain.npz                 # 50 m, all 118
  python exp1_dijkstra_gt.py --domain domain_5m.npz --sources 0,10,...  # 5 m subset

Output: output/runs/exp1_dijkstra/gt_<domain-stem>.npz
  T_dij (n_src,118) raw graph times, T_str (n_src,118) string-pulled
  times, src_idx, per-source runtimes, git SHA.
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "runs" / "exp1_dijkstra"

# Medium parameters — keep in sync with model/solve_all.py constants.
C0 = 15.0
CZ_UP, CZ_DOWN = 6.0, 4.0
A_Z = 0.5 * (CZ_UP + CZ_DOWN)        # 5.0
W_UP = 0.5 * (CZ_UP - CZ_DOWN)       # +1.0 upward drift
REG_Z_LOW, REG_Z_FULL, REG_LOW_FACTOR = 30.0, 40.0, 0.05


def reg_of_z(z):
    """Regulatory horizontal-speed factor at height z (vectorised)."""
    z = np.asarray(z, dtype=np.float64)
    frac = REG_LOW_FACTOR + (1.0 - REG_LOW_FACTOR) * (
        (z - REG_Z_LOW) / (REG_Z_FULL - REG_Z_LOW))
    return np.where(z >= REG_Z_FULL, 1.0,
                    np.where(z < REG_Z_LOW, REG_LOW_FACTOR, frac))


def metric_time(dxe, dye, dze, a_h):
    """Exact anisotropic+drift segment time (vectorised).

    Solves t^2 (1 - w^2/a_z^2) + 2 w dz t / a_z^2 - (h^2/a_h^2 + dz^2/a_z^2) = 0.
    """
    h2 = dxe * dxe + dye * dye
    A = 1.0 - (W_UP / A_Z) ** 2
    B = 2.0 * W_UP * dze / A_Z ** 2
    C = -(h2 / (a_h * a_h) + (dze * dze) / A_Z ** 2)
    disc = B * B - 4.0 * A * C
    return (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)


def nudge_stations(mask, stations_xy_m, dx, dy, dz):
    """EXACT copy of the solve_one_source.py station nudge."""
    nx, ny, nz = mask.shape
    st_ix = np.clip(np.round(stations_xy_m[:, 0] / dx).astype(int), 1, nx - 2)
    st_iy = np.clip(np.round(stations_xy_m[:, 1] / dy).astype(int), 1, ny - 2)
    st_iz = np.clip(np.round(stations_xy_m[:, 2] / dz).astype(int), 1, nz - 2)
    inside_b = mask[st_ix, st_iy, st_iz]
    for b in np.where(inside_b)[0]:
        ix0, iy0, iz0 = st_ix[b], st_iy[b], st_iz[b]
        found = False
        for r in range(1, 6):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ix = ix0 + di; iy = iy0 + dj
                    if not (0 < ix < nx - 1 and 0 < iy < ny - 1):
                        continue
                    if not mask[ix, iy, iz0]:
                        st_ix[b], st_iy[b] = ix, iy
                        found = True; break
                if found: break
            if found: break
        if not found:
            iz = iz0
            while iz < nz - 1 and mask[ix0, iy0, iz]:
                iz += 1
            st_iz[b] = iz
    return st_ix, st_iy, st_iz


# 13 half-space offsets of the 26-neighbourhood.
OFFSETS = [(di, dj, dk)
           for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
           if (di, dj, dk) > (0, 0, 0)]


def build_graph(mask, dx, dy, dz):
    """Directed CSR graph over the full flat grid; edges between free
    cells only. Asymmetric in z (climb != descend)."""
    nx, ny, nz = mask.shape
    n = mask.size
    free = ~mask
    zc = (np.arange(nz) + 0.5) * dz          # matches build_static_fields
    a_h_cell = (C0 * reg_of_z(zc)).astype(np.float64)     # (nz,)
    flat = np.arange(n, dtype=np.int32).reshape(mask.shape)

    rows, cols, vals = [], [], []
    for di, dj, dk in OFFSETS:
        sa = (slice(max(0, -di), nx - max(0, di)),
              slice(max(0, -dj), ny - max(0, dj)),
              slice(max(0, -dk), nz - max(0, dk)))
        sb = (slice(max(0, di), nx - max(0, -di)),
              slice(max(0, dj), ny - max(0, -dj)),
              slice(max(0, dk), nz - max(0, -dk)))
        ok = free[sa] & free[sb]
        ia = flat[sa][ok]
        ib = flat[sb][ok]
        # midpoint reg: average of the two cell layers
        ka = (ia % nz).astype(np.int64)      # z index from flat id
        kb = (ib % nz).astype(np.int64)
        a_h = 0.5 * (a_h_cell[ka] + a_h_cell[kb])
        dze = dk * dz
        w_ab = metric_time(di * dx, dj * dy, +dze, a_h).astype(np.float32)
        w_ba = metric_time(di * dx, dj * dy, -dze, a_h).astype(np.float32)
        rows.append(ia); cols.append(ib); vals.append(w_ab)
        rows.append(ib); cols.append(ia); vals.append(w_ba)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    print(f"[graph] {n} nodes, {rows.size} directed edges", flush=True)
    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return G


def reconstruct(pred, src_node, dst_node):
    path = [dst_node]
    u = dst_node
    while u != src_node:
        u = pred[u]
        if u < 0:
            return None
        path.append(u)
        if len(path) > 200000:
            return None
    return path[::-1]


def los_free(p0, p1, mask, dx, dy, dz, step_m):
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L < 1e-9:
        return True
    npts = max(2, int(L / step_m) + 1)
    ts = np.linspace(0.0, 1.0, npts)
    pts = p0[None, :] + ts[:, None] * seg[None, :]
    # cell i spans [i*dx, (i+1)*dx) under the (i+0.5)*dx centre convention
    ii = np.clip(np.floor(pts[:, 0] / dx).astype(int), 0, mask.shape[0] - 1)
    jj = np.clip(np.floor(pts[:, 1] / dy).astype(int), 0, mask.shape[1] - 1)
    kk = np.clip(np.floor(pts[:, 2] / dz).astype(int), 0, mask.shape[2] - 1)
    return not mask[ii, jj, kk].any()


def straighten(pts, mask, dx, dy, dz):
    """Greedy metric-aware string pulling.

    A shortcut i->j is accepted only if it is BOTH line-of-sight free
    AND faster under the continuum metric than the sub-path it replaces.
    The medium is inhomogeneous (the regulatory band makes z<30 m slow),
    so a straight segment is NOT automatically faster — a naive
    visibility-only shortcut drags the path through the slow layer and
    produced 6x over-estimates before this check was added. With the
    check, the result is a valid continuum path with time <= the graph
    time, i.e. a strictly tighter upper bound on the true optimum.
    """
    step = 0.45 * min(dx, dy, dz)
    n = len(pts)
    d = pts[1:] - pts[:-1]            # single-cell edges: no subdivision
    midz = 0.5 * (pts[1:, 2] + pts[:-1, 2])
    edge_t = metric_time(d[:, 0], d[:, 1], d[:, 2], C0 * reg_of_z(midz))
    cum = np.concatenate([[0.0], np.cumsum(edge_t)])
    out = [pts[0]]
    total = 0.0
    i = 0
    while i < n - 1:
        j = n - 1
        seg_t = None
        while j > i + 1:
            cand_t = polyline_time(np.asarray([pts[i], pts[j]]), dz)
            if cand_t <= cum[j] - cum[i] + 1e-9 and \
                    los_free(pts[i], pts[j], mask, dx, dy, dz, step):
                seg_t = cand_t
                break
            j = i + max(1, (j - i) // 2)
        if seg_t is None:
            j = i + 1
            seg_t = edge_t[i]
        out.append(pts[j])
        total += seg_t
        i = j
    return np.asarray(out), total


def polyline_time(pts, dz_cell):
    """Metric time along a polyline; segments subdivided so reg(z) is
    sampled every ~half cell of height."""
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        npieces = max(1, int(L / (0.5 * dz_cell)) + 1)
        ts = np.linspace(0.0, 1.0, npieces + 1)
        sub = a[None, :] + ts[:, None] * seg[None, :]
        mids_z = 0.5 * (sub[:-1, 2] + sub[1:, 2])
        d = sub[1:] - sub[:-1]
        a_h = C0 * reg_of_z(mids_z)
        total += float(np.sum(metric_time(d[:, 0], d[:, 1], d[:, 2], a_h)))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="domain.npz")
    ap.add_argument("--sources", default="all",
                    help="'all' or comma-separated station indices")
    args = ap.parse_args()

    z = np.load(ROOT / "input" / args.domain, allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations = z["stations_xy_m"]
    n_st = len(stations)
    print(f"[domain] {args.domain}: {mask.shape}, dx={dx}", flush=True)

    st_ix, st_iy, st_iz = nudge_stations(mask, stations, dx, dy, dz)
    nz_ = mask.shape[2]
    node_of_st = (st_ix.astype(np.int64) * mask.shape[1] + st_iy) * nz_ + st_iz
    node_of_st = node_of_st.astype(np.int64)
    st_xyz = np.stack([(st_ix + 0.5) * dx, (st_iy + 0.5) * dy,
                       (st_iz + 0.5) * dz], axis=1)

    t0 = time.time()
    G = build_graph(mask, dx, dy, dz)
    print(f"[graph] built in {time.time()-t0:.0f}s", flush=True)

    srcs = (list(range(n_st)) if args.sources == "all"
            else [int(s) for s in args.sources.split(",")])
    T_dij = np.full((len(srcs), n_st), np.inf, dtype=np.float64)
    T_str = np.full((len(srcs), n_st), np.inf, dtype=np.float64)
    walls = []
    for a, s in enumerate(srcs):
        t1 = time.time()
        dist, pred = dijkstra(G, directed=True, indices=node_of_st[s],
                              return_predecessors=True)
        t_dij = time.time() - t1
        T_dij[a, :] = dist[node_of_st]
        # string-pull finite pairs in (a slightly widened) valid band
        t2 = time.time()
        d2d = np.hypot(st_xyz[:, 0] - st_xyz[s, 0],
                       st_xyz[:, 1] - st_xyz[s, 1])
        for b in range(n_st):
            if b == s or not np.isfinite(T_dij[a, b]):
                continue
            if not (600.0 <= d2d[b] <= 3200.0):
                continue
            path = reconstruct(pred, node_of_st[s], node_of_st[b])
            if path is None:
                continue
            ids = np.asarray(path, dtype=np.int64)
            kk = ids % nz_
            jj = (ids // nz_) % mask.shape[1]
            ii = ids // (nz_ * mask.shape[1])
            pts = np.stack([(ii + 0.5) * dx, (jj + 0.5) * dy,
                            (kk + 0.5) * dz], axis=1)
            _, T_str[a, b] = straighten(pts, mask, dx, dy, dz)
        del pred, dist
        walls.append((t_dij, time.time() - t2))
        print(f"[src {s:3d}] dijkstra {t_dij:.1f}s  "
              f"string-pull {walls[-1][1]:.1f}s  "
              f"med T_dij {np.median(T_dij[a][np.isfinite(T_dij[a])]):.0f}s",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    out = OUT_DIR / f"gt_{Path(args.domain).stem}.npz"
    np.savez_compressed(
        out, T_dij=T_dij, T_str=T_str, src_idx=np.asarray(srcs),
        st_ix=st_ix, st_iy=st_iy, st_iz=st_iz, st_xyz=st_xyz,
        walls=np.asarray(walls), dx=dx,
        meta=json.dumps(dict(git_sha=sha, domain=args.domain,
                             c0=C0, cz_up=CZ_UP, cz_down=CZ_DOWN)))
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
