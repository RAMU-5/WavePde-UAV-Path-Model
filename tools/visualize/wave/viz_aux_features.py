"""Visualisations of the 7 auxiliary features implemented in
the wave model (paper sections 2.5, 2.6.2, 2.6.3/2.7, 2.9.4-5, 2.10,
2.11, 2.13.1).

For each feature, this script runs the underlying function on the real
Nanshan domain (or a sensible synthetic input if the function does not
need the city) and writes a single PNG + a paired NPZ into
visualization/aux/. The figures are referenced in
experiments_results.tex section "Auxiliary capabilities".

Usage (CPU JAX, ~3-4 min total on a single login-node core):
    VENV=venv/bin/python
    $VENV tools/visualize/wave/viz_aux_features.py

Each --only flag runs just one panel:
    $VENV tools/visualize/wave/viz_aux_features.py --only backward
    $VENV tools/visualize/wave/viz_aux_features.py --only multicolor coupled
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(ROOT / "_jax_cache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit

sys.path.insert(0, str(ROOT / "model"))
from solve_all import (build_static_fields, build_static_fields_param,
                       gaussian_source, solve_one,
                       N_STEPS, C0, CZ_UP, CZ_DOWN, DT)
from solve_backward import build_backward_fields
from solve_multicolor import (conflict_graph, greedy_four_color,
                              batched_source, estimate_min_separation)
from solve_coupled import (run_coupled, trace_path)
from solve_broadband import (solve_one_broadband,
                             DEF_TAU_L_STEPS, DEF_TAU_S_STEPS,
                             DEF_T0_L_STEPS, DEF_T0_S_STEPS,
                             DEF_OMEGA_S, DEF_PHI, broadband_amp)
from wind_inversion import estimate_wind
from uncertainty_diffusion import compute_uncertainty, monte_carlo_dispersion
from winged_alignment import find_alignment_point, build_two_phase_trajectory

OUT_DIR = ROOT / "output" / "figures" / "aux"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Common helpers
# --------------------------------------------------------------------------

def load_ctx():
    z = np.load(ROOT / "input" / "domain.npz", allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    stations = z["stations_xy_m"]
    pf = np.load(ROOT / "output" / "wave" / "pair_matrix.npz", allow_pickle=False)
    return dict(mask=mask, dx=dx, dy=dy, dz=dz,
                stations=stations, Tfwd=pf["T_matrix"])


def cell_idx(xyz, dx, dy, dz, shape):
    return (int(np.clip(round(xyz[0]/dx), 1, shape[0]-2)),
            int(np.clip(round(xyz[1]/dy), 1, shape[1]-2)),
            int(np.clip(round(xyz[2]/dz), 1, shape[2]-2)))


def solve_forward(ctx, src_xyz):
    mask_jnp = jnp.asarray(ctx["mask"])
    s = build_static_fields(mask_jnp, ctx["dx"], ctx["dy"], ctx["dz"])
    omd = (1.0 - s["damp"]).astype(jnp.float32)
    idx = cell_idx(src_xyz, ctx["dx"], ctx["dy"], ctx["dz"], ctx["mask"].shape)
    sp = gaussian_source(ctx["mask"].shape, idx)
    T = jit(solve_one, static_argnums=(11, 12, 13, 14))(
        s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
        s["ax"], s["ay"], s["az"], omd, s["free"], sp,
        ctx["dx"], ctx["dy"], ctx["dz"], N_STEPS)
    jax.block_until_ready(T)
    return np.asarray(jax.device_get(T))


def building_overlay(ax, mask, dx, dy, scale=1000.0):
    """Show building footprints as grey patches on a top-down ax.
    All consumer scatters and imshows in this module use km (scale=1000).
    """
    foot = mask.any(axis=2).astype(float).T  # (NY,NX)
    ax.imshow(np.ma.masked_where(foot == 0, foot),
              origin="lower", cmap="Greys",
              extent=[0, mask.shape[0]*dx/scale,
                      0, mask.shape[1]*dy/scale],
              alpha=0.55, vmin=0, vmax=1.4)


# --------------------------------------------------------------------------
# #1 backward / adjoint   (paper §2.5)
# --------------------------------------------------------------------------

def viz_backward():
    print("\n[viz #1 backward / adjoint]")
    ctx = load_ctx()
    st = ctx["stations"]; mask = ctx["mask"]
    dx, dy, dz = ctx["dx"], ctx["dy"], ctx["dz"]
    Tfwd = ctx["Tfwd"]

    # Pick dst so that the forward pair-matrix has many finite entries
    # in column dst (those are exactly the origins we can compare against).
    cols_finite = np.isfinite(Tfwd).sum(axis=0)
    dst_i = int(np.argmax(cols_finite))
    src_i = 17 if dst_i != 17 else 48      # forward T-field for the panel (arbitrary)
    src_xyz = st[src_i]; dst_xyz = st[dst_i]
    print(f"  src_i={src_i}, dst_i={dst_i} (col has {cols_finite[dst_i]} finite forward T)")

    print(f"  forward solve from src={src_i}")
    T_F = solve_forward(ctx, src_xyz)

    print(f"  backward solve to dst={dst_i}")
    mask_jnp = jnp.asarray(mask)
    sB = build_backward_fields(mask_jnp, dx, dy, dz)
    omd_B = (1.0 - sB["damp"]).astype(jnp.float32)
    idx_d = cell_idx(dst_xyz, dx, dy, dz, mask.shape)
    sp_d = gaussian_source(mask.shape, idx_d)
    T_B_jax = jit(solve_one, static_argnums=(11, 12, 13, 14))(
        sB["c2_xy"], sB["c2_z"], sB["u"], sB["v"], sB["w"],
        sB["ax"], sB["ay"], sB["az"], omd_B, sB["free"], sp_d,
        dx, dy, dz, N_STEPS)
    jax.block_until_ready(T_B_jax)
    T_B = np.asarray(jax.device_get(T_B_jax))

    # Scatter: T_forward(i -> dst_i)  vs  T_backward(dst_i -> i) at station i.
    # In the backward field that's exactly T_B at station i's index.
    pairs_x, pairs_y, dist_m = [], [], []
    col = Tfwd[:, dst_i]
    for i in range(len(st)):
        if i == dst_i: continue
        t_fwd = float(col[i])
        if not np.isfinite(t_fwd) or t_fwd < 30:
            continue
        idx_i = cell_idx(st[i], dx, dy, dz, mask.shape)
        t_bwd = float(T_B[idx_i])
        if not np.isfinite(t_bwd):
            continue
        pairs_x.append(t_fwd); pairs_y.append(t_bwd)
        dist_m.append(float(np.linalg.norm(st[i] - dst_xyz)))
    pairs_x = np.array(pairs_x); pairs_y = np.array(pairs_y)
    rel_err = np.mean(np.abs(pairs_x - pairs_y) / pairs_x) if len(pairs_x) else float("nan")
    print(f"  scatter n={len(pairs_x)}  mean rel err = {rel_err*100:.2f}%")

    # Slice index ~ 70 m
    kz = int(round(70.0 / dz))
    extent = [0, mask.shape[0]*dx/1000, 0, mask.shape[1]*dy/1000]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    a = axes[0]
    F_slice = T_F[:, :, kz].T
    im = a.imshow(np.where(np.isfinite(F_slice), F_slice, np.nan),
                  origin="lower", cmap="viridis", extent=extent)
    building_overlay(a, mask, dx, dy)
    a.scatter([src_xyz[0]/1000], [src_xyz[1]/1000], c="red", marker="*",
              s=130, edgecolors="white", label="source")
    a.set_title(f"forward T (src={src_i})  z=70 m")
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")
    plt.colorbar(im, ax=a, label="T [s]")
    a.legend(loc="lower right", fontsize=8)

    a = axes[1]
    B_slice = T_B[:, :, kz].T
    im = a.imshow(np.where(np.isfinite(B_slice), B_slice, np.nan),
                  origin="lower", cmap="viridis", extent=extent)
    building_overlay(a, mask, dx, dy)
    a.scatter([dst_xyz[0]/1000], [dst_xyz[1]/1000], c="cyan", marker="v",
              s=110, edgecolors="black", label="destination")
    a.set_title(f"backward T (dst={dst_i})  z=70 m  [q→−q]")
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")
    plt.colorbar(im, ax=a, label="T [s]")
    a.legend(loc="lower right", fontsize=8)

    a = axes[2]
    if len(pairs_x):
        sc = a.scatter(pairs_x, pairs_y, c=dist_m, cmap="plasma", s=22)
        lo, hi = min(pairs_x.min(), pairs_y.min()), max(pairs_x.max(), pairs_y.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        plt.colorbar(sc, ax=a, label="dist to dst [m]")
        a.set_xlabel("forward T(i → dst) [s]")
        a.set_ylabel("backward T(dst → i) [s]")
        a.set_title(f"adjoint symmetry, n={len(pairs_x)}  mean rel err {rel_err*100:.2f}%")
        a.legend(loc="upper left", fontsize=8)
    a.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"§2.5 Backward / adjoint wave routing — src={src_i}, dst={dst_i}",
                 y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_backward_field.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_backward_field.npz",
                        T_F_slice=T_F[:, :, kz], T_B_slice=T_B[:, :, kz],
                        forward_T=pairs_x, backward_T=pairs_y,
                        dist_m=np.array(dist_m), rel_err=rel_err,
                        src_i=src_i, dst_i=dst_i, z_slice_m=70.0)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_backward_field.{{png,npz}}")
    del T_F, T_B, T_B_jax, sB, mask_jnp; gc.collect()


# --------------------------------------------------------------------------
# #2 multicolor   (paper §2.13.1)
# --------------------------------------------------------------------------

def viz_multicolor():
    print("\n[viz #2 multicolor batched routing]")
    ctx = load_ctx()
    st = ctx["stations"]; mask = ctx["mask"]
    dx, dy, dz = ctx["dx"], ctx["dy"], ctx["dz"]

    # Conflict graph over ALL stations.  The conservative
    # estimate_min_separation = 2*c0*N_steps*dt gives ~13.5 km which is
    # larger than the domain diagonal (everyone conflicts).  For the
    # delivery-routing use case the conflict radius is the SERVICE
    # region a single drone is asked to cover, not the wave's farthest
    # reach — paper §2.13.1.  We use a service-radius separation of
    # 2 km, which is in line with Meituan's reported last-mile horizon
    # and gives a non-trivial conflict graph for the 4-colouring.
    sep = 2000.0
    print(f"  min_separation = {sep:.0f} m  (service radius, paper §2.13.1)")
    adj = conflict_graph(st, min_separation_m=sep)
    n_edges = sum(len(v) for v in adj.values()) // 2
    colours = greedy_four_color(adj, len(st))
    colour_arr = np.array(colours, dtype=int)
    counts = np.bincount(colour_arr, minlength=4)
    print(f"  edges={n_edges}, colour class sizes = {counts.tolist()}")

    # Pick 4 corner stations (one per colour ideally) for the batched run
    nx_m = (mask.shape[0] - 1) * dx
    ny_m = (mask.shape[1] - 1) * dy
    targets = [(0.18 * nx_m, 0.18 * ny_m),
               (0.82 * nx_m, 0.18 * ny_m),
               (0.18 * nx_m, 0.82 * ny_m),
               (0.82 * nx_m, 0.82 * ny_m)]
    chosen = []
    for tx, ty in targets:
        d = np.linalg.norm(st[:, :2] - np.array([tx, ty]), axis=1)
        chosen.append(int(np.argmin(d)))
    chosen = list(dict.fromkeys(chosen))[:4]
    print(f"  4 batched sources: {chosen}")

    mask_jnp = jnp.asarray(mask)
    s = build_static_fields(mask_jnp, dx, dy, dz)
    omd = (1.0 - s["damp"]).astype(jnp.float32)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))

    # 4 individual solves
    t0 = time.time()
    T_indep_at_probe = {}
    probes = []
    for src in chosen:
        idx = cell_idx(st[src], dx, dy, dz, mask.shape)
        sp = gaussian_source(mask.shape, idx)
        T = solve_jit(s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
                      s["ax"], s["ay"], s["az"], omd, s["free"], sp,
                      dx, dy, dz, N_STEPS)
        jax.block_until_ready(T)
        T_np = np.asarray(jax.device_get(T))
        col = ctx["Tfwd"][:, src]
        # probe = closest station to src with finite finite forward T
        cand = [j for j in range(len(st)) if j != src and np.isfinite(col[j])]
        cand.sort(key=lambda j: col[j])
        for j in cand[:30]:
            d_other = min(np.linalg.norm(st[j] - st[c2])
                          for c2 in chosen if c2 != src) if len(chosen) > 1 else np.inf
            d_self = np.linalg.norm(st[j] - st[src])
            if d_other > 2.0 * d_self and 80 < col[j] < 250:
                probes.append((src, j))
                d_idx = cell_idx(st[j], dx, dy, dz, mask.shape)
                T_indep_at_probe[(src, j)] = float(T_np[d_idx])
                break
        del T, T_np; gc.collect()
    t_indep = time.time() - t0
    print(f"  4 indep solves: {t_indep:.1f}s")

    # 1 batched solve
    idxs = [cell_idx(st[src], dx, dy, dz, mask.shape) for src in chosen]
    sp_b = batched_source(mask.shape, idxs)
    t0 = time.time()
    T_batch = solve_jit(s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
                        s["ax"], s["ay"], s["az"], omd, s["free"], sp_b,
                        dx, dy, dz, N_STEPS)
    jax.block_until_ready(T_batch)
    T_batch = np.asarray(jax.device_get(T_batch))
    t_batch = time.time() - t0
    print(f"  1 batched:      {t_batch:.1f}s")

    T_batch_at_probe = {}
    for src, dst in probes:
        d_idx = cell_idx(st[dst], dx, dy, dz, mask.shape)
        T_batch_at_probe[(src, dst)] = float(T_batch[d_idx])

    # 2-panel figure: (a) conflict graph + 4-colouring + 4 chosen sources;
    #                 (b) batched T-field z=70m; (c) bar timing.
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6])

    # (a)
    a = fig.add_subplot(gs[0, 0])
    building_overlay(a, mask, dx, dy)
    cmap_q = plt.get_cmap("tab10")
    for i, j_list in adj.items():
        for j in j_list:
            if j > i:
                a.plot([st[i, 0]/1000, st[j, 0]/1000],
                       [st[i, 1]/1000, st[j, 1]/1000],
                       c="0.6", lw=0.4, alpha=0.6, zorder=1)
    K = int(colour_arr.max()) + 1
    cmap_q = plt.get_cmap("tab20" if K > 10 else "tab10")
    for cidx in range(K):
        sel = np.where(colour_arr == cidx)[0]
        lbl = f"colour {cidx} (n={len(sel)})" if cidx < 4 else None
        a.scatter(st[sel, 0]/1000, st[sel, 1]/1000,
                  c=[cmap_q(cidx)], s=22,
                  label=lbl, edgecolors="black", linewidths=0.3, zorder=3)
    for src in chosen:
        a.scatter([st[src, 0]/1000], [st[src, 1]/1000],
                  s=180, marker="*", c="red",
                  edgecolors="white", linewidths=1.3, zorder=4)
    a.set_title(f"conflict graph (sep={sep:.0f} m)  "
                f"K={K} colours  |E|={n_edges}  "
                f"(speedup N/K≈{len(st)/K:.1f}×)")
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")
    a.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # (b)
    a = fig.add_subplot(gs[0, 1])
    kz = int(round(70.0 / dz))
    sl = T_batch[:, :, kz].T
    im = a.imshow(np.where(np.isfinite(sl), sl, np.nan),
                  origin="lower", cmap="viridis",
                  extent=[0, mask.shape[0]*dx/1000, 0, mask.shape[1]*dy/1000])
    building_overlay(a, mask, dx, dy)
    for src in chosen:
        a.scatter([st[src, 0]/1000], [st[src, 1]/1000], marker="*",
                  c="red", s=150, edgecolors="white", linewidths=1.0)
    for src, dst in probes:
        a.scatter([st[dst, 0]/1000], [st[dst, 1]/1000], marker="v",
                  c="cyan", s=70, edgecolors="black")
    plt.colorbar(im, ax=a, label="T [s]")
    a.set_title("batched T-field (4 superposed Gaussians) z=70 m")
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")

    # (c) timing + scatter accuracy
    a = fig.add_subplot(gs[0, 2])
    a.bar(["4× indep", "1× batched"], [t_indep, t_batch],
          color=["0.6", "C0"], edgecolor="black")
    a.set_ylabel("wall time [s]")
    speedup = t_indep / max(t_batch, 1e-9)
    a.set_title(f"timing  speedup ×{speedup:.1f}")
    for i, v in enumerate([t_indep, t_batch]):
        a.text(i, v + 0.02 * max(t_indep, t_batch), f"{v:.1f}s",
               ha="center", va="bottom", fontsize=9)
    if probes:
        text = "src→dst   T_indep   T_batch\n" + "\n".join(
            f"{src:3d}→{dst:3d}  {T_indep_at_probe[(src,dst)]:7.2f}  "
            f"{T_batch_at_probe[(src,dst)]:7.2f}"
            for src, dst in probes)
        a.text(0.5, -0.35, text, ha="center", va="top",
               transform=a.transAxes, family="monospace", fontsize=8,
               bbox=dict(facecolor="white", edgecolor="0.7"))

    fig.suptitle("§2.13.1 Four-colour batched routing", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_multicolor.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_multicolor.npz",
                        colours=colour_arr,
                        chosen=np.array(chosen),
                        probes=np.array(probes) if probes else np.zeros((0, 2), int),
                        T_indep=np.array([T_indep_at_probe[p] for p in probes]),
                        T_batch=np.array([T_batch_at_probe[p] for p in probes]),
                        t_indep=t_indep, t_batch=t_batch,
                        speedup=speedup, sep=sep, n_edges=n_edges,
                        T_batch_slice=T_batch[:, :, kz])
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_multicolor.{{png,npz}}")
    del T_batch, mask_jnp, s; gc.collect()


# --------------------------------------------------------------------------
# #3 wave / dynamics coupling  (paper §2.10)
# --------------------------------------------------------------------------

def viz_coupled():
    print("\n[viz #3 coupled iteration]")
    ctx = load_ctx()
    st = ctx["stations"]; mask = ctx["mask"]
    dx, dy, dz = ctx["dx"], ctx["dy"], ctx["dz"]

    src_i = 48
    cands = np.where(np.isfinite(ctx["Tfwd"][src_i])
                     & (ctx["Tfwd"][src_i] > 100)
                     & (ctx["Tfwd"][src_i] < 250))[0]
    dst_i = int(cands[0])
    src_xyz = st[src_i]; dst_xyz = st[dst_i]
    print(f"  src={src_i} dst={dst_i}  init cz_up=8 (off-spec) ...")

    t0 = time.time()
    res = run_coupled(mask, dx, dy, dz, src_xyz, dst_xyz,
                      p0=(C0, 8.0, 5.0), max_iter=5, alpha=0.6,
                      tol_param=0.01, tol_path_m=5.0, verbose=False)
    print(f"  {len(res['history'])} iters in {time.time()-t0:.1f}s")

    hist = res["history"]
    iters = [h["iter"] for h in hist]
    c0_arr = np.array([h["params"][0] for h in hist])
    cu_arr = np.array([h["params"][1] for h in hist])
    cd_arr = np.array([h["params"][2] for h in hist])
    rel_arr = np.array([h["rel_change"] for h in hist])
    pr_arr = np.array([h["path_rms"] if h["path_rms"] != float("inf") else np.nan
                       for h in hist])

    final_path = res["path"]

    # Need iter-0 path for overlay. Re-run a one-shot solve with the
    # initial (off-spec) parameters to get the initial trace.
    s0 = build_static_fields_param(jnp.asarray(mask), dx, dy, dz,
                                   c0=C0, cz_up=8.0, cz_down=5.0)
    omd0 = (1.0 - s0["damp"]).astype(jnp.float32)
    idx_src = cell_idx(src_xyz, dx, dy, dz, mask.shape)
    sp0 = gaussian_source(mask.shape, idx_src)
    T0_jax = jit(solve_one, static_argnums=(11, 12, 13, 14))(
        s0["c2_xy"], s0["c2_z"], s0["u"], s0["v"], s0["w"],
        s0["ax"], s0["ay"], s0["az"], omd0, s0["free"], sp0,
        dx, dy, dz, N_STEPS)
    jax.block_until_ready(T0_jax)
    T0 = np.asarray(jax.device_get(T0_jax))
    path0, _ = trace_path(T0, src_xyz, dst_xyz, dx, dy, dz)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    a = axes[0, 0]
    a.plot(iters, c0_arr, "-o", label="c₀ (m/s)")
    a.plot(iters, cu_arr, "-s", label="c_z↑ (m/s)")
    a.plot(iters, cd_arr, "-^", label="c_z↓ (m/s)")
    a.axhline(8.0, color="C1", ls=":", lw=1, alpha=0.6,
              label="init c_z↑=8 (off-spec)")
    a.set_xlabel("iteration"); a.set_ylabel("speed [m/s]")
    a.set_title("wave-equation params per iter")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = axes[0, 1]
    a.plot(iters, rel_arr, "-o", color="C3", label="rel param change")
    a.set_xlabel("iteration")
    a.set_ylabel("max rel change", color="C3")
    a.tick_params(axis='y', labelcolor="C3")
    a2 = a.twinx()
    finite_mask = np.isfinite(pr_arr)
    a2.plot(np.array(iters)[finite_mask], pr_arr[finite_mask],
            "-s", color="C0", label="path RMS Δ (m)")
    a2.set_ylabel("path RMS Δ [m]", color="C0")
    a2.tick_params(axis='y', labelcolor="C0")
    a.set_title("convergence")
    a.grid(alpha=0.3)

    a = axes[1, 0]
    building_overlay(a, mask, dx, dy)
    if path0.shape[0] > 1:
        a.plot(path0[:, 0]/1000, path0[:, 1]/1000, "-",
               color="C1", lw=2, alpha=0.85, label=f"iter 0 (c_z↑=8)")
    if final_path.shape[0] > 1:
        a.plot(final_path[:, 0]/1000, final_path[:, 1]/1000, "-",
               color="C0", lw=2, alpha=0.95, label=f"iter {len(hist)-1} (converged)")
    a.scatter([src_xyz[0]/1000], [src_xyz[1]/1000], marker="*",
              c="red", s=160, edgecolors="white", label="src")
    a.scatter([dst_xyz[0]/1000], [dst_xyz[1]/1000], marker="v",
              c="cyan", s=110, edgecolors="black", label="dst")
    a.set_xlim(0, mask.shape[0]*dx/1000); a.set_ylim(0, mask.shape[1]*dy/1000)
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")
    a.set_title("path before/after coupling (top-down)")
    a.legend(fontsize=8)

    a = axes[1, 1]
    if path0.shape[0] > 1:
        s0_arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path0, axis=0), axis=1)])
        a.plot(s0_arc, path0[:, 2], color="C1", lw=2, label="iter 0")
    if final_path.shape[0] > 1:
        s1_arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(final_path, axis=0), axis=1)])
        a.plot(s1_arc, final_path[:, 2], color="C0", lw=2, label="converged")
    a.set_xlabel("arc length [m]"); a.set_ylabel("altitude z [m]")
    a.set_title("vertical profile")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    fig.suptitle(f"§2.10 Wave / dynamics fixed-point coupling — src={src_i}, dst={dst_i}",
                 y=1.00, fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_coupled.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_coupled.npz",
                        iters=np.array(iters), c0=c0_arr, cz_up=cu_arr,
                        cz_down=cd_arr, rel_change=rel_arr, path_rms=pr_arr,
                        path0=path0, path_final=final_path,
                        src=src_xyz, dst=dst_xyz)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_coupled.{{png,npz}}")
    del T0, T0_jax, s0; gc.collect()


# --------------------------------------------------------------------------
# #4 broadband / multi-scale source  (paper §2.9.4-2.9.5)
# --------------------------------------------------------------------------

def viz_broadband():
    print("\n[viz #4 broadband multi-scale source]")
    ctx = load_ctx()
    st = ctx["stations"]; mask = ctx["mask"]
    dx, dy, dz = ctx["dx"], ctx["dy"], ctx["dz"]

    src_i = 48
    src_xyz = st[src_i]
    idx = cell_idx(src_xyz, dx, dy, dz, mask.shape)
    sp = gaussian_source(mask.shape, idx)

    mask_jnp = jnp.asarray(mask)
    s = build_static_fields(mask_jnp, dx, dy, dz)
    omd = (1.0 - s["damp"]).astype(jnp.float32)
    solve_bb = jit(solve_one_broadband, static_argnums=(11, 12, 13, 14))

    def go(A_L, A_S, label):
        t0 = time.time()
        T = solve_bb(s["c2_xy"], s["c2_z"], s["u"], s["v"], s["w"],
                     s["ax"], s["ay"], s["az"], omd, s["free"], sp,
                     dx, dy, dz, N_STEPS,
                     A_L, DEF_T0_L_STEPS*DT, DEF_TAU_L_STEPS*DT,
                     A_S, DEF_T0_S_STEPS*DT, DEF_TAU_S_STEPS*DT,
                     DEF_OMEGA_S, DEF_PHI)
        jax.block_until_ready(T)
        T_np = np.asarray(jax.device_get(T))
        print(f"  {label}: {time.time()-t0:.1f}s, finite={np.isfinite(T_np).sum()}")
        return T_np

    T_L = go(1.0, 0.0, "long")
    T_S = go(0.0, 1.0, "short")
    T_B = go(1.0, 1.0, "broadband")

    both = np.isfinite(T_L) & np.isfinite(T_S) & np.isfinite(T_B)
    diffs = np.abs(T_B[both] - np.minimum(T_L[both], T_S[both]))
    med = float(np.median(diffs)); p95 = float(np.quantile(diffs, 0.95))
    print(f"  |T_B - min(T_L,T_S)|: median={med:.3f}s p95={p95:.3f}s")

    kz = int(round(70.0 / dz))
    extent = [0, mask.shape[0]*dx/1000, 0, mask.shape[1]*dy/1000]

    fig = plt.figure(figsize=(14.5, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])

    # Top row: 3 T-field slices
    vmin = np.nanmin([np.nanmin(np.where(np.isfinite(F), F, np.nan))
                      for F in (T_L, T_S, T_B)])
    vmax = np.nanmax([np.nanmax(np.where(np.isfinite(F), F, np.nan))
                      for F in (T_L, T_S, T_B)])
    for j, (T, name) in enumerate([(T_L, "long-only (τ=24·DT)"),
                                   (T_S, "short-only (Gabor)"),
                                   (T_B, "broadband (sum)")]):
        a = fig.add_subplot(gs[0, j])
        sl = T[:, :, kz].T
        im = a.imshow(np.where(np.isfinite(sl), sl, np.nan),
                      origin="lower", cmap="viridis", extent=extent,
                      vmin=vmin, vmax=vmax)
        building_overlay(a, mask, dx, dy)
        a.scatter([src_xyz[0]/1000], [src_xyz[1]/1000], marker="*",
                  c="red", s=130, edgecolors="white")
        a.set_title(f"{name}   z=70 m")
        a.set_xlabel("x [km]")
        if j == 0: a.set_ylabel("y [km]")
        plt.colorbar(im, ax=a, label="T [s]", shrink=0.85)

    # Bottom row left: source waveform; middle: |T_B - min| hist; right: coverage
    a = fig.add_subplot(gs[1, 0])
    n_show = 80
    tt = np.arange(n_show) * DT
    amp_L = np.array([float(broadband_amp(t, 1.0, DEF_T0_L_STEPS*DT, DEF_TAU_L_STEPS*DT,
                                          0.0, 0.0, 1.0, 0.0, 0.0)) for t in tt])
    amp_S = np.array([float(broadband_amp(t, 0.0, 0.0, 1.0,
                                          1.0, DEF_T0_S_STEPS*DT, DEF_TAU_S_STEPS*DT,
                                          DEF_OMEGA_S, DEF_PHI)) for t in tt])
    amp_B = amp_L + amp_S
    a.plot(tt, amp_L, label="long Gaussian", color="C0")
    a.plot(tt, amp_S, label="short Gabor", color="C3")
    a.plot(tt, amp_B, label="broadband sum", color="black", lw=1.5)
    a.set_xlabel("time [s]"); a.set_ylabel("source amplitude")
    a.set_title("source waveforms")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 1])
    a.hist(diffs, bins=40, color="0.5", edgecolor="black")
    a.axvline(med, color="C0", ls="--", label=f"median {med:.2f} s")
    a.axvline(p95, color="C3", ls="--", label=f"p95 {p95:.2f} s")
    a.set_xlabel("|T_B − min(T_L,T_S)|  [s]"); a.set_ylabel("count")
    a.set_title("peak-time shift vs. component min")
    a.legend(fontsize=8)

    a = fig.add_subplot(gs[1, 2])
    fL = np.isfinite(T_L); fS = np.isfinite(T_S); fB = np.isfinite(T_B)
    rows = [("long",  int(fL.sum())),
            ("short", int(fS.sum())),
            ("broad", int(fB.sum())),
            ("∪ L,S",  int((fL | fS).sum())),
            ("∩ B with ∪", int(((fL | fS) & fB).sum()))]
    a.bar([r[0] for r in rows], [r[1] for r in rows],
          color=["C0", "C3", "black", "0.6", "C2"], edgecolor="black")
    a.set_ylabel("finite-T cells")
    a.set_title("coverage by component")
    for i, (_, v) in enumerate(rows):
        a.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"§2.9.4-2.9.5 Broadband (multi-scale) wave source — src={src_i}",
                 y=1.00)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_broadband.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_broadband.npz",
                        T_L_slice=T_L[:, :, kz], T_S_slice=T_S[:, :, kz],
                        T_B_slice=T_B[:, :, kz],
                        diffs=diffs, median_diff=med, p95_diff=p95,
                        src=src_xyz, z_slice_m=70.0,
                        src_amp_t=tt, src_amp_L=amp_L,
                        src_amp_S=amp_S, src_amp_B=amp_B)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_broadband.{{png,npz}}")
    del T_L, T_S, T_B, mask_jnp, s; gc.collect()


# --------------------------------------------------------------------------
# #5 wind inversion   (paper §2.6.2)
# --------------------------------------------------------------------------

def viz_wind_inversion():
    print("\n[viz #5 wind inversion]")
    ctx = load_ctx()
    st = ctx["stations"]
    src = st[48]; dst = st[92]
    direction = (dst - src); direction[2] = 0.0
    direction /= max(np.linalg.norm(direction), 1e-9)

    # Truth wind has a vertical sinusoidal swing so the figure is informative
    n = 80; dt_step = 1.0
    speed = 12.0
    t = np.arange(n) * dt_step
    u_truth = np.column_stack([
        1.5 + 0.5 * np.sin(0.2 * t),
        -0.8 + 0.3 * np.cos(0.15 * t),
        0.6 * np.sin(0.1 * t),
    ])
    q_a = np.tile(speed * direction, (n, 1))
    track = np.empty((n, 3)); track[0] = src
    for i in range(1, n):
        track[i] = track[i-1] + (q_a[i-1] + u_truth[i-1]) * dt_step

    # Predicted wind = truth + small bias for residual demo
    def u_pred_fn(pos, t_s):
        return np.array([1.5, -0.8, 0.0])  # constant-wind prior
    res = estimate_wind(track, t, q_a, u_pred_fn=u_pred_fn)
    print(f"  recovered mean u = {res.u_est.mean(axis=0)}")
    print(f"  truth mean      = {u_truth.mean(axis=0)}")

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])

    # 3D track (left, spans full height of column 0)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(track[:, 0], track[:, 1], track[:, 2], "k-", lw=1.5,
              label="track")
    sub = slice(0, n-1, 4)
    ax3d.quiver(res.pos_mid[sub, 0], res.pos_mid[sub, 1], res.pos_mid[sub, 2],
                res.u_est[sub, 0], res.u_est[sub, 1], res.u_est[sub, 2],
                length=18, color="C3", normalize=False, label="u_est")
    ax3d.scatter([src[0]], [src[1]], [src[2]], c="red", marker="*",
                 s=110, edgecolors="white", label="src")
    ax3d.scatter([track[-1, 0]], [track[-1, 1]], [track[-1, 2]],
                 c="cyan", marker="v", s=80, edgecolors="black", label="end")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("synthetic track + recovered wind quiver")
    ax3d.legend(fontsize=8, loc="upper left")

    # Component panels: u_x, u_y, u_z at (0,1), (0,2), (1,2)
    labels = ["u_x [m/s]", "u_y [m/s]", "u_z [m/s]"]
    slots = [gs[0, 1], gs[0, 2], gs[1, 2]]
    for j, (lab, slot) in enumerate(zip(labels, slots)):
        a = fig.add_subplot(slot)
        a.plot(t, u_truth[:, j], "k-", lw=2, label="truth")
        a.plot(res.t_mid, res.u_est[:, j], "C3.-", lw=1.2, label="u_est = q_g − q_a")
        if res.u_pred is not None:
            a.plot(res.t_mid, res.u_pred[:, j], "C0--", lw=1, label="u_pred (prior)")
        a.set_xlabel("t [s]"); a.set_ylabel(lab)
        a.grid(alpha=0.3)
        if j == 0: a.legend(fontsize=8)
        a.set_title(lab.split(" ")[0])

    # Residual ε panel at (1,1)
    a = fig.add_subplot(gs[1, 1])
    if res.epsilon is not None:
        eps_norm = np.linalg.norm(res.epsilon, axis=1)
        a.plot(res.t_mid, eps_norm, "C2-")
        a.set_ylabel("|ε| = |u_est − u_pred| [m/s]")
    a.set_xlabel("t [s]"); a.set_title("data-assimilation residual")
    a.grid(alpha=0.3)

    fig.suptitle("§2.6.2 Wind inversion u = q_g − q_a along observed track",
                 y=1.00)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_wind_inversion.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_wind_inversion.npz",
                        track=track, t=t, q_a=q_a, u_truth=u_truth,
                        u_est=res.u_est, u_pred=res.u_pred, epsilon=res.epsilon,
                        t_mid=res.t_mid)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_wind_inversion.{{png,npz}}")


# --------------------------------------------------------------------------
# #11 uncertainty diffusion   (paper §2.6.3 / §2.7)
# --------------------------------------------------------------------------

def viz_uncertainty():
    """Diffusion-based uncertainty visualised AS A TUBE around a real
    wave-routed delivery trajectory (not a synthetic constant-cruise
    track). The trajectory is loaded from
    database/paths_048.npz (source station 48) and the tube radius is
    σ_position(s) computed segment-by-segment along the actual path."""
    print("\n[viz #11 uncertainty diffusion — on real routed path]")
    from uav_dynamics import integrate_dynamics, UAVDynParams

    db = ROOT / "output" / "database" / "paths_048.npz"
    d = np.load(db, allow_pickle=True)
    po, pl, pc = d["path_offsets"], d["path_lengths"], d["paths_concat"]
    # Pick the longest path with > 200 points (a substantial route)
    valid = [k for k in range(len(pl)) if pl[k] > 200]
    valid.sort(key=lambda k: -pl[k])
    k_path = valid[0]
    dst_i = int(d["dst_indices"][k_path])
    path = pc[po[k_path]:po[k_path+1]].astype(np.float64)
    print(f"  routed path: src=48 -> dst={dst_i}  n_pts={path.shape[0]}")

    # Resample uniformly in arc length for cleaner kinematic curves
    seg = np.diff(path, axis=0)
    s_raw = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
    n_resample = 240
    s_uni = np.linspace(0, s_raw[-1], n_resample)
    path_u = np.stack([np.interp(s_uni, s_raw, path[:, j]) for j in range(3)],
                      axis=-1)

    # Integrate dynamics on the resampled path to get V along the route
    dyn = integrate_dynamics(path_u, UAVDynParams())
    V = np.asarray(dyn["V"])          # (n_resample,)
    t = np.asarray(dyn["t"])          # cumulative time (s) along path

    # Air-relative velocity along the path: speed V along tangent, plus
    # mild turbulent fluctuations representing urban-canopy ABL eddies.
    # Paper §2.8 default: L_eddy = 1 m (street-canyon scale).
    # σ_turb = 0.5 m/s lateral, 0.2 m/s vertical (typical mild ABL,
    # Counihan urban surface layer).
    SIG_TURB_LAT = 0.5    # m/s
    SIG_TURB_VER = 0.2    # m/s
    L_eddy = 1.0          # m  (paper §2.8 default)

    tan = np.diff(path_u, axis=0)
    tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    tan = np.vstack([tan[:1], tan])          # pad to n_resample
    rng = np.random.default_rng(7)
    nominal = V[:, None] * tan
    # Build a local frame for adding lateral fluctuations
    e_z = np.array([0.0, 0.0, 1.0])
    side = np.cross(tan, e_z); side /= np.maximum(
        np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    up   = np.cross(tan, side)
    fluct = (SIG_TURB_LAT * rng.normal(size=n_resample)[:, None] * side
             + SIG_TURB_VER * rng.normal(size=n_resample)[:, None] * up)
    q_along = nominal + fluct

    # Bypass the windowed-variance estimator: pass the known
    # turbulence variance σ_q² = 2·σ_lat² + σ_ver² directly.  The
    # estimator would otherwise pick up heading curvature of the
    # routing path as "fluctuation" and inflate τ by ~25×.
    sigma_q2_true = (2.0 * SIG_TURB_LAT**2 + SIG_TURB_VER**2)
    sigma_q2_arr = np.full(n_resample, sigma_q2_true)
    prof = compute_uncertainty(s_uni, t, q_along, L_eddy=L_eddy,
                               sigma_q2_override=sigma_q2_arr)
    sigma_3drms = np.sqrt(6.0 * np.maximum(prof.tau, 0.0))
    sigma_paper = prof.sigma
    sigma_axis  = np.sqrt(2.0 * np.maximum(prof.tau, 0.0))

    # Monte Carlo at the route's final time — use the ACTUAL D(t) along
    # the path so the MC reference is consistent with the closed-form
    # σ_3D-RMS = √(6 τ).
    D_t_arr = prof.D.copy()
    t_arr = t.copy()
    def D_of_t(tt):
        return float(np.interp(np.clip(tt, t_arr[0], t_arr[-1]),
                               t_arr, D_t_arr))
    D_mean = float(D_t_arr.mean())
    mc = monte_carlo_dispersion(D_of_t, t_max=float(t[-1]),
                                dt=0.05, n_walkers=2000, rng=rng)
    print(f"  T_total={t[-1]:.1f}s arc={s_uni[-1]:.0f}m   "
          f"V_mean={V.mean():.2f}m/s   D_mean={D_mean:.3f}m²/s")
    print(f"  σ_3D_RMS at dst = {sigma_3drms[-1]:.2f} m   "
          f"MC r_rms = {mc['r_rms']:.2f} m")

    # ----------------- figure -----------------
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.7, 1, 1])

    # Top-down with tube envelope (ribbon of width 2*sigma in the
    # lateral direction). Use sigma_3drms for the radius.
    a = fig.add_subplot(gs[0, 0])
    sigma_use = sigma_3drms
    # ribbon: trajectory ± sigma along the horizontal-normal direction
    horiz = path_u[:, :2]
    htan = np.diff(horiz, axis=0)
    htan = htan / np.maximum(np.linalg.norm(htan, axis=1, keepdims=True), 1e-9)
    htan = np.vstack([htan[:1], htan])
    hnorm = np.stack([-htan[:, 1], htan[:, 0]], axis=-1)
    left  = horiz + hnorm * sigma_use[:, None]
    right = horiz - hnorm * sigma_use[:, None]
    ctx = load_ctx()
    building_overlay(a, ctx["mask"], ctx["dx"], ctx["dy"])
    a.fill(np.r_[left[:, 0]/1000, right[::-1, 0]/1000],
           np.r_[left[:, 1]/1000, right[::-1, 1]/1000],
           color="C3", alpha=0.25,
           label=fr"$\pm\sigma_{{3D-RMS}}$ tube")
    a.plot(horiz[:, 0]/1000, horiz[:, 1]/1000, "C3-", lw=1.5,
           label="routed path")
    a.scatter([path_u[0, 0]/1000], [path_u[0, 1]/1000], marker="*",
              c="red", s=130, edgecolors="white", label="src 48")
    a.scatter([path_u[-1, 0]/1000], [path_u[-1, 1]/1000], marker="v",
              c="cyan", s=110, edgecolors="black", label=f"dst {dst_i}")
    a.set_xlim(0, ctx["mask"].shape[0]*ctx["dx"]/1000)
    a.set_ylim(0, ctx["mask"].shape[1]*ctx["dy"]/1000)
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")
    a.set_title(f"routed path + ±σ uncertainty tube  (σ_final={sigma_use[-1]:.1f} m)")
    a.legend(fontsize=8, loc="upper right")

    # 3D path with tube
    a = fig.add_subplot(gs[0, 1], projection="3d")
    a.plot(path_u[:, 0], path_u[:, 1], path_u[:, 2], "C3-", lw=2,
           label="routed path")
    # 3D tube: 12-sided cross-section (cone-of-uncertainty)
    n_ring = 12
    theta = np.linspace(0, 2*np.pi, n_ring)
    tube_x = np.zeros((n_resample, n_ring))
    tube_y = np.zeros((n_resample, n_ring))
    tube_z = np.zeros((n_resample, n_ring))
    for i in range(n_resample):
        for j, th in enumerate(theta):
            offset = sigma_use[i] * (side[i] * np.cos(th) + up[i] * np.sin(th))
            tube_x[i, j] = path_u[i, 0] + offset[0]
            tube_y[i, j] = path_u[i, 1] + offset[1]
            tube_z[i, j] = path_u[i, 2] + offset[2]
    # plot the tube as a translucent surface
    a.plot_surface(tube_x, tube_y, tube_z, alpha=0.25, color="C3",
                   linewidth=0, antialiased=True)
    a.scatter([path_u[0, 0]], [path_u[0, 1]], [path_u[0, 2]], marker="*",
              c="red", s=120, edgecolors="white")
    a.scatter([path_u[-1, 0]], [path_u[-1, 1]], [path_u[-1, 2]], marker="v",
              c="cyan", s=90, edgecolors="black")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("z [m]")
    a.set_title("3-D uncertainty tube  (radius=σ_3D-RMS(s))")

    # Altitude profile with band
    a = fig.add_subplot(gs[0, 2])
    a.plot(s_uni, path_u[:, 2], "C3-", lw=2, label="altitude z(s)")
    a.fill_between(s_uni, path_u[:, 2] - sigma_use,
                          path_u[:, 2] + sigma_use,
                   color="C3", alpha=0.25, label="±σ")
    a.set_xlabel("arc length s [m]"); a.set_ylabel("z [m]")
    a.set_title("vertical profile with ±σ band")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # Middle row: V(t), D(t)
    a = fig.add_subplot(gs[1, 0])
    a.plot(t, V, "C0-", lw=1.5)
    a.set_xlabel("t [s]"); a.set_ylabel("V [m/s]")
    a.set_title("speed V(t) (uav_dynamics)")
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 1])
    a.plot(t, prof.D, "C0-", lw=1.5)
    a.set_xlabel("t [s]"); a.set_ylabel("D(t) [m²/s]")
    a.set_title(r"diffusion coefficient  $D=\sigma_q^2\,T_L$")
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 2])
    a.plot(t, prof.tau, "C1-", lw=1.5)
    a.set_xlabel("t [s]"); a.set_ylabel(r"$\tau(t)=\int D\,\mathrm{d}s$  [m²]")
    a.set_title("cumulative integral τ(t)")
    a.grid(alpha=0.3)

    # Bottom row: sigma curves + MC, and sigma vs arc-length
    a = fig.add_subplot(gs[2, 0:2])
    a.plot(t, sigma_paper, label=r"paper:  $2\sqrt{\tau}$", color="C0")
    a.plot(t, sigma_3drms, label=r"3-D RMS:  $\sqrt{6\tau}$", color="C3")
    a.plot(t, sigma_axis,  label=r"per-axis:  $\sqrt{2\tau}$", color="C2")
    a.axhline(mc["r_rms"], color="C3", ls=":", alpha=0.8,
              label=f"MC $r_{{rms}}$ at $t_{{end}}$ = {mc['r_rms']:.1f} m (n=2000)")
    a.set_xlabel("t [s]"); a.set_ylabel(r"$\sigma_{position}$ [m]")
    a.set_title("position std along the routed path: 3 conventions + Monte Carlo")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 2])
    a.plot(s_uni, sigma_3drms, "C3-", lw=1.5)
    a.set_xlabel("arc length s [m]")
    a.set_ylabel(r"$\sigma_{3D-RMS}$ [m]")
    a.set_title("uncertainty radius along route")
    a.grid(alpha=0.3)

    fig.suptitle(
        f"§2.6.3 / §2.7 Diffusion-based positional uncertainty "
        f"along a wave-routed path  (src=48 → dst={dst_i}, "
        f"{s_uni[-1]:.0f} m / {t[-1]:.1f} s)", y=1.00)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_uncertainty.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_uncertainty.npz",
                        path=path_u, s=s_uni, t=t, V=V,
                        D=prof.D, tau=prof.tau,
                        sigma_paper=sigma_paper, sigma_3drms=sigma_3drms,
                        sigma_axis=sigma_axis,
                        mc_r_rms=mc["r_rms"], mc_r_std=mc["r_std"],
                        mc_r_mean=mc["r_mean"],
                        src_idx=48, dst_idx=dst_i, L_eddy=L_eddy)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_uncertainty.{{png,npz}}")


# --------------------------------------------------------------------------
# #16 winged-UAV alignment   (paper §2.11)
# --------------------------------------------------------------------------

def viz_winged():
    print("\n[viz #16 winged alignment]")
    ctx = load_ctx()
    st = ctx["stations"]; mask = ctx["mask"]
    dx, dy, dz = ctx["dx"], ctx["dy"], ctx["dz"]
    src_i = 48
    src_xyz = st[src_i]
    print(f"  forward solve from src={src_i} ...")
    T = solve_forward(ctx, src_xyz)

    nx_m = (mask.shape[0] - 1) * dx
    ny_m = (mask.shape[1] - 1) * dy
    far = np.array([0.5 * nx_m, 0.5 * ny_m, src_xyz[2]])
    launch = src_xyz.copy()
    heading = far - launch; heading[2] = 0.0
    heading /= max(np.linalg.norm(heading), 1e-9)

    x_star, score, score_field, grad_T = find_alignment_point(
        T, dx, dy, dz, launch, heading,
        search_radius_m=800.0, cone_half_angle_deg=40.0, min_score=0.5)
    print(f"  x_star = {x_star},  score = {score:.3f}")

    # Pick a destination for the 2-phase trajectory: a station roughly
    # along the heading from launch.
    cand = []
    for j in range(len(st)):
        if j == src_i: continue
        v = st[j] - launch; v[2] = 0
        n = np.linalg.norm(v)
        if n < 200: continue
        cos = float(np.dot(v / n, heading))
        if cos > 0.6 and np.isfinite(ctx["Tfwd"][j, src_i]):
            cand.append((cos, j, n))
    cand.sort(key=lambda x: -x[2])  # farthest aligned
    dst_i = cand[0][1] if cand else int(np.argmax(np.linalg.norm(st - launch, axis=1)))
    dst_xyz = st[dst_i]
    print(f"  dst = station {dst_i} at {dst_xyz}")

    phase1, phase2 = build_two_phase_trajectory(
        launch, heading, x_star, grad_T, dx, dy, dz, dst_xyz,
        step_frac=0.4)
    print(f"  phase1 n={len(phase1)}  phase2 n={len(phase2)}")

    # Plot top-down with score-field heatmap restricted to launch z slice
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    a = axes[0]
    kz = int(np.clip(round(launch[2] / dz), 0, mask.shape[2]-1))
    score_slice = score_field[:, :, kz].T
    # Only show cells with valid score
    im = a.imshow(np.where(np.isfinite(score_slice), score_slice, np.nan),
                  origin="lower", cmap="RdYlGn", vmin=-1, vmax=1,
                  extent=[0, mask.shape[0]*dx/1000, 0, mask.shape[1]*dy/1000])
    building_overlay(a, mask, dx, dy)
    # Draw search cone outline
    R = 800.0  # m
    theta0 = np.arctan2(heading[1], heading[0])
    half = np.deg2rad(40.0)
    a.plot([launch[0]/1000,
            (launch[0] + R * np.cos(theta0 - half))/1000],
           [launch[1]/1000,
            (launch[1] + R * np.sin(theta0 - half))/1000],
           "k--", lw=1, alpha=0.7)
    a.plot([launch[0]/1000,
            (launch[0] + R * np.cos(theta0 + half))/1000],
           [launch[1]/1000,
            (launch[1] + R * np.sin(theta0 + half))/1000],
           "k--", lw=1, alpha=0.7, label="search cone")
    a.plot(phase1[:, 0]/1000, phase1[:, 1]/1000, "-",
           color="C0", lw=2.5, label="phase 1 (heading)")
    a.plot(phase2[:, 0]/1000, phase2[:, 1]/1000, "-",
           color="C3", lw=2, label="phase 2 (−∇T)")
    a.scatter([launch[0]/1000], [launch[1]/1000], marker="*",
              c="red", s=150, edgecolors="white", label="launch")
    a.scatter([x_star[0]/1000], [x_star[1]/1000], marker="o",
              c="yellow", s=120, edgecolors="black", label=f"x*  score={score:.3f}")
    a.scatter([dst_xyz[0]/1000], [dst_xyz[1]/1000], marker="v",
              c="cyan", s=110, edgecolors="black", label=f"dst (st {dst_i})")
    plt.colorbar(im, ax=a, label="cos(−∇T, ê_uav)")
    a.set_title(f"alignment-score field  (launch heading θ={np.degrees(theta0):.0f}°)")
    a.set_xlabel("x [km]"); a.set_ylabel("y [km]")
    a.legend(loc="upper left", fontsize=7, framealpha=0.85)

    a = axes[1] = fig.add_subplot(1, 2, 2, projection="3d")
    a.plot(phase1[:, 0], phase1[:, 1], phase1[:, 2],
           color="C0", lw=2.5, label="phase 1")
    a.plot(phase2[:, 0], phase2[:, 1], phase2[:, 2],
           color="C3", lw=2.0, label="phase 2 (−∇T)")
    a.scatter([launch[0]], [launch[1]], [launch[2]], marker="*",
              c="red", s=130)
    a.scatter([x_star[0]], [x_star[1]], [x_star[2]], marker="o",
              c="yellow", s=110, edgecolors="black")
    a.scatter([dst_xyz[0]], [dst_xyz[1]], [dst_xyz[2]], marker="v",
              c="cyan", s=90, edgecolors="black")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("z [m]")
    a.set_title(f"2-phase trajectory  (x*=[{x_star[0]:.0f}, {x_star[1]:.0f}, {x_star[2]:.0f}])")
    a.legend(fontsize=8)

    fig.suptitle("§2.11 Winged-UAV alignment point + 2-phase trajectory", y=1.00)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aux_winged.png", dpi=140, bbox_inches="tight")
    np.savez_compressed(OUT_DIR / "aux_winged.npz",
                        launch=launch, heading=heading,
                        x_star=x_star, score=float(score),
                        phase1=phase1, phase2=phase2, dst=dst_xyz,
                        score_slice=score_field[:, :, kz])
    plt.close(fig)
    print(f"  saved {OUT_DIR}/aux_winged.{{png,npz}}")
    del T, grad_T, score_field; gc.collect()


# --------------------------------------------------------------------------

ALL = [
    ("backward",   viz_backward),
    ("multicolor", viz_multicolor),
    ("coupled",    viz_coupled),
    ("broadband",  viz_broadband),
    ("wind",       viz_wind_inversion),
    ("uncertainty",viz_uncertainty),
    ("winged",     viz_winged),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these features (names from %s)" % [k for k,_ in ALL])
    args = p.parse_args()
    selected = set(args.only) if args.only else None
    t_total = time.time()
    for key, fn in ALL:
        if selected is not None and key not in selected:
            continue
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"\n[ERROR in {key}] {e}")
            traceback.print_exc()
        print(f"  ... {key} took {time.time()-t0:.1f}s")
        gc.collect()
    print(f"\nALL DONE in {time.time()-t_total:.1f}s. PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
