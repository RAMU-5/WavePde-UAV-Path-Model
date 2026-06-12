"""E4.3 — spatially varying wind: validity and the reciprocity limit
(paper Q3, Fig. 7).

(a) Divergence-free vortex wind on the real 50 m domain. The wave PDE
    takes the wind as a coefficient field; the reference is the E2.1
    anisotropic Dijkstra generalised to a full wind vector: segment
    time t solves  |d/t - q|_M = 1  on the airspeed ellipsoid
    (horizontal a_h(z), vertical A_Z, plus the W_UP gravity drift), i.e.

      A t^2 + B t + C = 0,
      A = (u^2+v^2)/a_h^2 + w^2/A_Z^2 - 1
      B = -2 [ (dx u + dy v)/a_h^2 + dz w/A_Z^2 ]
      C = (dx^2+dy^2)/a_h^2 + dz^2/A_Z^2

    smaller positive root  t = 2C / (-B + sqrt(B^2-4AC));  no positive
    root (upwind faster than airspeed, e.g. inside the 0.75 m/s
    regulatory band against a 4.5 m/s wind) -> edge dropped.

    Headline: per-pair wind-induced time change dT = T(wind) - T(0)
    predicted by wave vs by Dijkstra (bias and resolution error cancel
    in the difference), plus the absolute gap per Mach number.

(b) Reciprocity error vs ||div q||: wind = vortex + beta * grad(phi)
    (compressive blob), beta in {0, 1/3, 2/3, 1}; uniform wind as the
    near-zero-divergence reference. Forward T(i->j) vs backward
    (q -> -q) T(j->i) on 6x6 station pairs. This turns the known
    div-q caveat of the adjoint trick into a measured operating limit.

Wave and Dijkstra see the SAME cell-centred wind arrays. 50 m domain:
cheap enough to run everything (one GPU + CPU) in ~30 min.

Outputs output/runs/exp4_wind/exp4_data.npz; figure by
tools/visualize/paper/fig07_wind.py.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp1_dijkstra_gt import (  # noqa: E402
    C0, A_Z, W_UP, reg_of_z, nudge_stations, OFFSETS, reconstruct, los_free)
import wave_api  # noqa: E402
from wave_api import WaveConfig, StaticFields  # noqa: E402
import solve_all as SA  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OUT_DIR = ROOT / "output" / "runs" / "exp4_wind"

SRC_IDS = list(range(0, 118, 10))            # the 12 Q2 ground-truth sources
FWD_IDS, BWD_IDS = SRC_IDS, SRC_IDS          # reciprocity pairs (12 x 12)
VORTEX_L = 3000.0                            # vortex cell size, >> buildings
COMP_L = 1000.0                              # compressive checkerboard period
LO_D, HI_D = 700.0, 3000.0


# ------------------------------------------------------------- wind fields
def cellgrid_xy(nx, ny, dx, dy):
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    return np.meshgrid(x, y, indexing="ij")


def vortex_wind(nx, ny, nz, dx, dy, qmax):
    """Streamfunction psi = sin(2 pi x/L) sin(2 pi y/L): analytically
    divergence-free cellular vortices, z-independent, w = 0."""
    X, Y = cellgrid_xy(nx, ny, dx, dy)
    kx = 2.0 * np.pi / VORTEX_L
    u2 = np.sin(kx * X) * np.cos(kx * Y)      # = d(psi)/dy / (psi0 k)
    v2 = -np.cos(kx * X) * np.sin(kx * Y)     # = -d(psi)/dx / (psi0 k)
    s = qmax / np.sqrt(u2 ** 2 + v2 ** 2).max()
    u = np.repeat((u2 * s)[:, :, None], nz, axis=2).astype(np.float32)
    v = np.repeat((v2 * s)[:, :, None], nz, axis=2).astype(np.float32)
    return u, v, np.zeros_like(u)


def comp_wind(nx, ny, nz, dx, dy, qmax):
    """q = grad(phi), phi = checkerboard of sources/sinks (period
    COMP_L): purely compressive (curl-free), div ~ 2 pi qmax / COMP_L —
    well above the wall-masking divergence floor."""
    X, Y = cellgrid_xy(nx, ny, dx, dy)
    k = 2.0 * np.pi / COMP_L
    u2 = np.cos(k * X) * np.sin(k * Y)        # = d(phi)/dx / (phi0 k)
    v2 = np.sin(k * X) * np.cos(k * Y)        # = d(phi)/dy / (phi0 k)
    s = qmax / np.sqrt(u2 ** 2 + v2 ** 2).max()
    u = np.repeat((u2 * s)[:, :, None], nz, axis=2).astype(np.float32)
    v = np.repeat((v2 * s)[:, :, None], nz, axis=2).astype(np.float32)
    return u, v, np.zeros_like(u)


def div_stats(u, v, w, free, dx, dy, dz):
    """RMS / max of div(q) of the masked field over interior free cells
    (the masking itself creates divergence at building walls — that is
    the field the solver actually sees, so it is what we report)."""
    um, vm, wm = u * free, v * free, w * free
    d = (np.gradient(um, dx, axis=0) + np.gradient(vm, dy, axis=1)
         + np.gradient(wm, dz, axis=2))
    sel = free[1:-1, 1:-1, 1:-1].astype(bool)
    dv = d[1:-1, 1:-1, 1:-1][sel]
    return float(np.sqrt(np.mean(dv ** 2))), float(np.abs(dv).max())


# ------------------------------------------------------------- wave side
def build_static_wind(mask_j, dx, dy, dz, uF, vF, wF):
    """Production static fields with full (nx,ny,nz) wind arrays —
    mirrors solve_all.build_static_fields_param exactly (wind masked by
    free, w_grav added, (q.grad)q recomputed from the actual fields)."""
    s = SA.build_static_fields_param(mask_j, dx, dy, dz)   # production constants
    free = s["free"]
    w_grav = 0.5 * (SA.CZ_DOWN - SA.CZ_UP)
    u = jnp.asarray(uF) * free
    v = jnp.asarray(vF) * free
    w = jnp.asarray(wF) * free - w_grav
    g = SA.grad_central
    ax = u * g(u, 0, dx) + v * g(u, 1, dy) + w * g(u, 2, dz)
    ay = u * g(v, 0, dx) + v * g(v, 1, dy) + w * g(v, 2, dz)
    az = u * g(w, 0, dx) + v * g(w, 1, dy) + w * g(w, 2, dz)
    return StaticFields(
        c2_xy=s["c2_xy"], c2_z=s["c2_z"], u=u, v=v, w=w,
        ax=ax, ay=ay, az=az,
        one_minus_damp=(1.0 - s["damp"]).astype(jnp.float32),
        free=free)


# -------------------------------------------------- wind-aware Dijkstra
def metric_time_wind(dxe, dye, dze, a_h, u, v, w):
    """Smaller positive root of A t^2 + B t + C = 0 (see module doc);
    inf where the segment cannot be traversed."""
    A = (u * u + v * v) / (a_h * a_h) + (w * w) / A_Z ** 2 - 1.0
    B = -2.0 * ((dxe * u + dye * v) / (a_h * a_h) + dze * w / A_Z ** 2)
    C = (dxe * dxe + dye * dye) / (a_h * a_h) + (dze * dze) / A_Z ** 2
    disc = B * B - 4.0 * A * C
    denom = -B + np.sqrt(np.maximum(disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 2.0 * C / denom
    return np.where((disc >= 0.0) & (denom > 1e-12), t, np.inf)


def build_graph_wind(mask, dx, dy, dz, u3, v3, w3):
    """26-neighbour directed graph; edge wind = mean of endpoint cells.
    Non-traversable (supersonic upwind) edges are dropped."""
    nx, ny, nz = mask.shape
    n = mask.size
    free = ~mask
    zc = (np.arange(nz) + 0.5) * dz
    a_h_cell = (C0 * reg_of_z(zc)).astype(np.float64)
    flat = np.arange(n, dtype=np.int32).reshape(mask.shape)
    uf, vf, wf = (a.ravel() for a in (u3, v3, w3 + W_UP))

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
        ka = (ia % nz).astype(np.int64)
        kb = (ib % nz).astype(np.int64)
        a_h = 0.5 * (a_h_cell[ka] + a_h_cell[kb])
        um = 0.5 * (uf[ia] + uf[ib])
        vm = 0.5 * (vf[ia] + vf[ib])
        wm = 0.5 * (wf[ia] + wf[ib])
        dze = dk * dz
        w_ab = metric_time_wind(di * dx, dj * dy, +dze, a_h, um, vm, wm)
        w_ba = metric_time_wind(-di * dx, -dj * dy, -dze, a_h, um, vm, wm)
        for rr, cc, ww in ((ia, ib, w_ab), (ib, ia, w_ba)):
            fin = np.isfinite(ww)
            rows.append(rr[fin]); cols.append(cc[fin])
            vals.append(ww[fin].astype(np.float32))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    print(f"[graph] {n} nodes, {rows.size} directed edges", flush=True)
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def _sample_wind(pts, u3, v3, w3, dx, dy, dz):
    ii = np.clip(np.floor(pts[:, 0] / dx).astype(int), 0, u3.shape[0] - 1)
    jj = np.clip(np.floor(pts[:, 1] / dy).astype(int), 0, u3.shape[1] - 1)
    kk = np.clip(np.floor(pts[:, 2] / dz).astype(int), 0, u3.shape[2] - 1)
    return u3[ii, jj, kk], v3[ii, jj, kk], w3[ii, jj, kk] + W_UP


def seg_times_wind(sub, u3, v3, w3, dx, dy, dz):
    """Segment times along a polyline with wind and a_h taken as the
    MEAN OF THE ENDPOINT samples — the same convention as the graph
    edge weights, so re-timing a graph path reproduces the graph time
    exactly (a midpoint convention disagreed near the feasibility
    boundary of the supersonic regulatory band and produced spurious
    inf edges)."""
    ua, va, wa = _sample_wind(sub[:-1], u3, v3, w3, dx, dy, dz)
    ub, vb, wb = _sample_wind(sub[1:], u3, v3, w3, dx, dy, dz)
    a_h = 0.5 * (C0 * reg_of_z(sub[:-1, 2]) + C0 * reg_of_z(sub[1:, 2]))
    d = sub[1:] - sub[:-1]
    return metric_time_wind(d[:, 0], d[:, 1], d[:, 2], a_h,
                            0.5 * (ua + ub), 0.5 * (va + vb),
                            0.5 * (wa + wb))


def polyline_time_wind(pts, dz_cell, u3, v3, w3, dx, dy, dz):
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        npieces = max(1, int(L / (0.5 * dz_cell)) + 1)
        ts = np.linspace(0.0, 1.0, npieces + 1)
        sub = a[None, :] + ts[:, None] * seg[None, :]
        t = seg_times_wind(sub, u3, v3, w3, dx, dy, dz)
        if not np.all(np.isfinite(t)):
            return np.inf       # infeasible shortcut -> caller rejects it
        total += float(np.sum(t))
    return total


def straighten_wind(pts, mask, dx, dy, dz, u3, v3, w3):
    """Metric-aware string pulling with the wind metric (cf. exp1)."""
    step = 0.45 * min(dx, dy, dz)
    n = len(pts)
    edge_t = seg_times_wind(pts, u3, v3, w3, dx, dy, dz)
    cum = np.concatenate([[0.0], np.cumsum(edge_t)])
    total = 0.0
    i = 0
    while i < n - 1:
        j = n - 1
        seg_t = None
        while j > i + 1:
            cand_t = polyline_time_wind(np.asarray([pts[i], pts[j]]), dz,
                                        u3, v3, w3, dx, dy, dz)
            if cand_t <= cum[j] - cum[i] + 1e-9 and \
                    los_free(pts[i], pts[j], mask, dx, dy, dz, step):
                seg_t = cand_t
                break
            j = i + max(1, (j - i) // 2)
        if seg_t is None:
            j = i + 1
            seg_t = float(edge_t[i])
        total += seg_t
        i = j
    return total


def dijkstra_case(mask, dx, dy, dz, u3, v3, w3, node_of_st, st_xyz, srcs):
    n_st = len(st_xyz)
    nz_ = mask.shape[2]
    G = build_graph_wind(mask, dx, dy, dz, u3, v3, w3)
    T_dij = np.full((len(srcs), n_st), np.inf)
    T_str = np.full((len(srcs), n_st), np.inf)
    for a, s in enumerate(srcs):
        dist, pred = dijkstra(G, directed=True, indices=node_of_st[s],
                              return_predecessors=True)
        T_dij[a] = dist[node_of_st]
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
            T_str[a, b] = straighten_wind(pts, mask, dx, dy, dz, u3, v3, w3)
        print(f"  [dij src {s:3d}] med "
              f"{np.median(T_dij[a][np.isfinite(T_dij[a])]):.0f}s", flush=True)
    return T_dij, T_str


# ----------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations = z["stations_xy_m"]
    n_st = len(stations)
    nx, ny, nz = mask.shape
    ix, iy, iz = nudge_stations(mask, stations, dx, dy, dz)
    node_of_st = ((ix.astype(np.int64) * ny + iy) * nz + iz).astype(np.int64)
    st_xyz = np.stack([(ix + 0.5) * dx, (iy + 0.5) * dy,
                       (iz + 0.5) * dz], axis=1)
    print(f"[domain] {mask.shape} dx={dx} dz={dz}, {n_st} stations", flush=True)

    cfg = WaveConfig()          # production 50 m settings (min_peak 1e-6)
    mask_j = jnp.asarray(mask)
    free_np = (~mask).astype(np.float32)
    zero = np.zeros((nx, ny, nz), dtype=np.float32)

    # ---- wind cases -----------------------------------------------------
    vort15 = vortex_wind(nx, ny, nz, dx, dy, 1.5)
    vort45 = vortex_wind(nx, ny, nz, dx, dy, 4.5)
    comp45 = comp_wind(nx, ny, nz, dx, dy, 4.5)

    cases_a = [("nowind", (zero, zero, zero)),
               ("vortex_M01", vort15),
               ("vortex_M03", vort45)]
    # solenoidal <-> compressive blend at FIXED amplitude |q|max = 4.5
    # (so the divergence sweep is not confounded with wind speed)
    cases_b = [("uniform_M02", None)]            # near-zero-div reference
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        u = (1.0 - beta) * vort45[0] + beta * comp45[0]
        v = (1.0 - beta) * vort45[1] + beta * comp45[1]
        s = 4.5 / np.sqrt(u ** 2 + v ** 2).max()
        cases_b.append((f"blend_b{beta:.2f}", (u * s, v * s, zero)))

    meta = {}
    for name, fld in cases_a + cases_b:
        if fld is None:
            fld = (np.full_like(zero, 3.0), zero, zero)   # uniform (3,0,0)
        rms, mx = div_stats(fld[0], fld[1], fld[2], free_np, dx, dy, dz)
        spd = float(np.sqrt(fld[0] ** 2 + fld[1] ** 2).max())
        meta[name] = dict(div_rms=rms, div_max=mx, qmax=spd)
        print(f"[wind {name}] qmax={spd:.2f} m/s  div rms={rms:.2e} "
              f"max={mx:.2e} 1/s", flush=True)

    out = {}

    # ---- (a) wave + Dijkstra under vortex wind --------------------------
    for name, fld in cases_a:
        print(f"[case {name}] wave forward x{len(SRC_IDS)}", flush=True)
        static = build_static_wind(mask_j, dx, dy, dz, *fld)
        Tw = np.full((len(SRC_IDS), n_st), np.inf, dtype=np.float32)
        Tfields = {}
        for a, s in enumerate(SRC_IDS):
            t1 = time.time()
            T = wave_api._run(static, (int(ix[s]), int(iy[s]), int(iz[s])),
                              dx, dy, dz, cfg)
            Tw[a] = T[ix, iy, iz]
            if s == 50:
                Tfields[name] = T.astype(np.float32)
            print(f"  [wave src {s:3d}] {time.time()-t1:.1f}s", flush=True)
        out[f"Twave_{name}"] = Tw
        if name in Tfields:
            out[f"Tfield_{name}_src50"] = Tfields[name]
        print(f"[case {name}] dijkstra", flush=True)
        T_dij, T_str = dijkstra_case(mask, dx, dy, dz, *fld,
                                     node_of_st=node_of_st, st_xyz=st_xyz,
                                     srcs=SRC_IDS)
        out[f"Tdij_{name}"] = T_dij.astype(np.float32)
        out[f"Tstr_{name}"] = T_str.astype(np.float32)

    # ---- (b) reciprocity vs divergence -----------------------------------
    for name, fld in cases_b:
        if fld is None:
            static = build_static_wind(mask_j, dx, dy, dz,
                                       np.full_like(zero, 3.0), zero, zero)
        else:
            static = build_static_wind(mask_j, dx, dy, dz, *fld)
        Tf = np.full((len(FWD_IDS), n_st), np.inf, dtype=np.float32)
        Tb = np.full((len(BWD_IDS), n_st), np.inf, dtype=np.float32)
        print(f"[case {name}] fwd x{len(FWD_IDS)} + bwd x{len(BWD_IDS)}",
              flush=True)
        for a, s in enumerate(FWD_IDS):
            T = wave_api._run(static, (int(ix[s]), int(iy[s]), int(iz[s])),
                              dx, dy, dz, cfg)
            Tf[a] = T[ix, iy, iz]
        flipped = wave_api.flip_static_fields(static)
        for a, s in enumerate(BWD_IDS):
            T = wave_api._run(flipped, (int(ix[s]), int(iy[s]), int(iz[s])),
                              dx, dy, dz, cfg)
            Tb[a] = T[ix, iy, iz]
        out[f"Tfwd_{name}"] = Tf
        out[f"Tbwd_{name}"] = Tb

    # ---- save -----------------------------------------------------------
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    out.update(
        src_ids=np.array(SRC_IDS, dtype=np.int32),
        fwd_ids=np.array(FWD_IDS, dtype=np.int32),
        bwd_ids=np.array(BWD_IDS, dtype=np.int32),
        st_xyz=st_xyz.astype(np.float32),
        u_vortex45_z5=vort45[0][:, :, 5].astype(np.float32),
        v_vortex45_z5=vort45[1][:, :, 5].astype(np.float32),
        meta=json.dumps(dict(cases=meta, git_sha=sha, dx=dx, dz=dz,
                             vortex_L=VORTEX_L, comp_L=COMP_L)))
    np.savez_compressed(OUT_DIR / "exp4_data.npz", **out)
    print(f"[done] -> {OUT_DIR}/exp4_data.npz", flush=True)


if __name__ == "__main__":
    main()
