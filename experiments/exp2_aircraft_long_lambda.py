"""Wavelength experiment for the regional aircraft scenario.
Compare against exp2_aircraft.py with: τ=20 (vs 6), σ=10 cells (vs 4),
path smoothing σ_path = 8 km (vs 5).

Output: visualization/3d_aircraft/aircraft_regional_long_lambda.png
"""
from __future__ import annotations
import os
from pathlib import Path

_THIS = Path(__file__).resolve().parent
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_THIS.parent / "_jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import jax.numpy as jnp
import exp2_aircraft as EA

# --- Long-wavelength overrides ----------------------------------------
EA.AC_PULSE_TAU = 20            # was 6  → ~3x longer temporal pulse
EA.AC_SOURCE_SIGMA_CELLS = 10.0 # was 4  → 2.5x wider spatial source (4 km)
EA.PATH_SMOOTH_SIGMA_M = 8000.0 # was 5000 → 8 km smoothing for 8-km R_min


# Replace the source generator to use peak normalisation (g/g.max);
# otherwise the per-cell amplitude shrinks with σ³ and the wave
# attenuates below the first-peak detection threshold at far cells.
def gaussian_source_3d_peak(shape, idx, sigma_cells):
    nx, ny, nz = shape
    ix = jnp.arange(nx)[:, None, None]
    iy = jnp.arange(ny)[None, :, None]
    iz = jnp.arange(nz)[None, None, :]
    r2 = ((ix - idx[0]) ** 2 + (iy - idx[1]) ** 2 + (iz - idx[2]) ** 2)
    g = jnp.exp(-0.5 * r2 / (sigma_cells ** 2))
    return (g / g.max()).astype(jnp.float32)


EA.gaussian_source_3d = gaussian_source_3d_peak


# Replace smooth_path so the Z axis is NOT pinned to endpoints — that
# pinning was causing a 2 km vertical drop in a single sample at the
# end of every path (z went from ~10 km cruise to 8 km dst in one step).
# Now z is also smoothed at the boundary so the descent ramps gradually.
def smooth_path_relaxed(path_xyz, sigma_m=None, step_size=None):
    if sigma_m is None:
        sigma_m = EA.PATH_SMOOTH_SIGMA_M
    if path_xyz.shape[0] < 4:
        return path_xyz
    seg = np.diff(path_xyz, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    if step_size is None:
        step_size = max(50.0, sigma_m / 12.0)
    s_new = np.arange(0.0, arc[-1] + 1e-6, step_size)
    if s_new.size < 4:
        return path_xyz
    rs = np.column_stack([
        np.interp(s_new, arc, path_xyz[:, 0]),
        np.interp(s_new, arc, path_xyz[:, 1]),
        np.interp(s_new, arc, path_xyz[:, 2]),
    ])
    sigma_steps = max(1.5, sigma_m / step_size)
    half = int(np.ceil(3.0 * sigma_steps))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_steps) ** 2)
    k = k / k.sum()
    out = np.empty_like(rs)
    for axis in range(3):
        v = rs[:, axis]
        if half > 0:
            head = v[half:0:-1]
            tail = v[-2:-half - 2:-1]
            padded = np.concatenate([head, v, tail])
            if padded.size != v.size + 2 * half:
                padded = np.pad(v, half, mode="edge")
        else:
            padded = v
        out[:, axis] = np.convolve(padded, k, mode="valid")[: v.size]
    # Pin only the xy endpoints (path must start at align, end at dst
    # in xy). Let z float at the boundaries so the descent is smooth.
    out[0, :2] = rs[0, :2]
    out[-1, :2] = rs[-1, :2]
    return out


EA.smooth_path = smooth_path_relaxed


# Patch trace_aircraft so we don't re-pin z at the phase-2 endpoints
# (the original sets phase2[0] = align_xyz exactly, undoing our
# relaxed z smoothing). Instead we keep xy continuity but let z float
# smoothly. The connection between phase1 (ending at z=8km) and
# phase2 (now starting at smoothed z, possibly ~9km) is handled by a
# short bridging segment.
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import map_coordinates as _map_coords


def trace_aircraft_relaxed(T_field, src_xyz_m, dst_xyz_m, dx, dy, dz,
                           runway_dir, phase1_length, cruise_alt,
                           gamma=0.3, max_steps=40000):
    src = np.array(src_xyz_m, dtype=np.float64)
    dst = np.array(dst_xyz_m, dtype=np.float64)
    n_p1 = max(2, int(phase1_length / (gamma * min(dx, dy, dz))))
    s_arr = np.linspace(0.0, phase1_length, n_p1)
    x_p1 = src[0] + runway_dir[0] * s_arr
    y_p1 = src[1] + runway_dir[1] * s_arr
    z_p1 = np.linspace(src[2], cruise_alt, n_p1)
    phase1 = np.column_stack([x_p1, y_p1, z_p1])
    align_xyz = phase1[-1]

    finite = np.isfinite(T_field)
    if not finite.any():
        empty = np.empty((0, 3))
        return phase1, "no_finite", phase1, empty, empty
    T_max = T_field[finite].max()
    inf_mask = ~finite
    dist_to_finite = distance_transform_edt(inf_mask)
    fill_alpha = T_max / 30.0
    T_safe = np.where(finite, T_field, T_max + fill_alpha * dist_to_finite)
    gx, gy, gz = np.gradient(T_safe, dx, dy, dz)

    step = gamma * min(dx, dy, dz)
    pos = dst.copy()
    path2 = [pos.copy()]
    status = "max_steps"
    align_xy = align_xyz[:2]
    best_d_align = np.inf
    best_idx = 0
    stale = 0
    floor_hits = 0
    z_max_allowed = (T_field.shape[2] - 2) * dz
    for _ in range(max_steps):
        coords = np.array([[pos[0]/dx], [pos[1]/dy], [pos[2]/dz]])
        gv = np.array([
            float(_map_coords(gx, coords, order=1, mode="nearest")[0]),
            float(_map_coords(gy, coords, order=1, mode="nearest")[0]),
            float(_map_coords(gz, coords, order=1, mode="nearest")[0]),
        ])
        on_floor = pos[2] <= cruise_alt + 0.5 * dz
        if on_floor and gv[2] > 0.0:
            gv[2] = 0.0
            floor_hits += 1
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            status = "zero_grad"; break
        pos = pos - step * gv / gn
        if pos[2] < cruise_alt:
            pos[2] = cruise_alt
        if pos[2] > z_max_allowed:
            pos[2] = z_max_allowed
        path2.append(pos.copy())
        d_a = np.linalg.norm(pos[:2] - align_xy)
        if d_a < best_d_align - 0.5 * step:
            best_d_align = d_a; best_idx = len(path2) - 1; stale = 0
        else:
            stale += 1
        if d_a < 2.0 * max(dx, dy):
            status = "reached_align"; break
        if stale > 4000:
            status = f"stale_align{best_d_align:.0f}m"; break
    full_back = np.array(path2[::-1])
    cut_back = max(0, len(full_back) - 1 - best_idx)
    phase2_raw = full_back[cut_back:]
    # XY-prepend align so phase 2 connects to phase 1 in xy. Keep raw
    # z so smoothing can decide the climb / descent profile freely.
    if phase2_raw.shape[0] > 1:
        first_xy = align_xyz.copy()
        first_xy[2] = phase2_raw[0, 2]   # keep raw z at start
        phase2_raw = np.vstack([first_xy[None, :], phase2_raw[1:]])
    phase2 = smooth_path_relaxed(phase2_raw)
    if phase2.size:
        phase2[:, 2] = np.maximum(phase2[:, 2], cruise_alt)
        # Pin only xy at the endpoints (z is free, smoothed)
        phase2[0, :2] = align_xyz[:2]
        phase2[-1, :2] = phase2_raw[-1, :2]
    full = np.vstack([phase1, phase2])
    status = (f"{status}_align{best_d_align:.0f}m_"
              f"floor{floor_hits}_zmax{phase2[:,2].max() if phase2.size else 0:.0f}m")
    return full, status, phase1, phase2, phase2_raw


EA.trace_aircraft = trace_aircraft_relaxed

# Wrap further: add a Phase-3 descent ramp at the end so the aircraft
# arrives at the dst altitude (8 km) with a smooth, ~25 km horizontal
# glide-slope rather than a step. Bridges phase 1 (8km) and phase 2
# (variable z that may be ~10km) by also adjusting phase 2 START so
# z=8km at align, then climbs along phase 2.
PHASE3_DESCENT_LENGTH_M = 25000.0   # 25 km horizontal descent ramp
PHASE2_TRANSITION_LENGTH_M = 25000.0  # 25 km phase-2 startup transition

ORIG_trace = EA.trace_aircraft


def trace_aircraft_with_descent(*args, **kwargs):
    full, status, p1, p2, p2_raw = ORIG_trace(*args, **kwargs)
    if p2 is None or p2.size == 0:
        return full, status, p1, p2, p2_raw

    align_xyz = p1[-1]
    # Use the ORIGINAL dst altitude (8 km), not p2[-1] which has been
    # pulled to the smoothed cruise altitude (~10 km) by the relaxed
    # smoother.
    dst_xyz_input = np.array(args[2], dtype=np.float64)
    dst_xyz = p2[-1].copy()
    dst_xyz[2] = dst_xyz_input[2]   # restore original dst altitude (8 km)
    cruise_alt = align_xyz[2]   # 8 km

    # Arc-length along phase 2
    seg = np.diff(p2, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]

    # Smooth descent ramp: in the last PHASE3_DESCENT_LENGTH_M of arc,
    # interpolate z from current path-z down to dst z (8km). Use a
    # cosine ramp so the rate-of-change is also smooth at both ends.
    z_orig = p2[:, 2].copy()
    descent_frac = np.clip(
        (arc - (total - PHASE3_DESCENT_LENGTH_M)) / PHASE3_DESCENT_LENGTH_M,
        0.0, 1.0)
    cos_ramp = 0.5 * (1.0 - np.cos(np.pi * descent_frac))   # 0→1 smooth
    z_descent = (1.0 - cos_ramp) * z_orig + cos_ramp * dst_xyz[2]

    # Climb-in transition: in the first PHASE2_TRANSITION_LENGTH_M of arc,
    # ramp z from cruise_alt (=8km, where phase 1 ended) up to the
    # routing-derived z. Same cosine ramp.
    climbin_frac = np.clip(
        ((total - arc) - (total - PHASE2_TRANSITION_LENGTH_M))
        / PHASE2_TRANSITION_LENGTH_M, 0.0, 1.0)
    # climbin_frac is 1 at start, 0 elsewhere — flip
    climbin_frac = 1.0 - climbin_frac    # 0 at start, 1 elsewhere
    cos_climb = 0.5 * (1.0 - np.cos(np.pi * climbin_frac))
    z_with_climb = (1.0 - cos_climb) * cruise_alt + cos_climb * z_descent

    p2_new = p2.copy()
    p2_new[:, 2] = z_with_climb
    # Re-pin endpoints
    p2_new[0] = align_xyz
    p2_new[-1] = dst_xyz

    full_new = np.vstack([p1, p2_new])
    return full_new, status, p1, p2_new, p2_raw


EA.trace_aircraft = trace_aircraft_with_descent


# Override the output filename
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
ORIG_savefig = plt.Figure.savefig

LONG_OUT_PNG = EA.OUT / "aircraft_regional_long_lambda.png"
LONG_OUT_NPZ = EA.OUT / "aircraft_regional_long_lambda.npz"


def patched_savefig(self, fname, *args, **kwargs):
    fname = str(fname)
    if fname.endswith("aircraft_regional.png"):
        fname = str(LONG_OUT_PNG)
    return ORIG_savefig(self, fname, *args, **kwargs)


plt.Figure.savefig = patched_savefig

ORIG_savez = np.savez_compressed


def patched_savez(file, *args, **kwargs):
    file = str(file)
    if file.endswith("aircraft_regional.npz"):
        file = str(LONG_OUT_NPZ)
    return ORIG_savez(file, *args, **kwargs)


np.savez_compressed = patched_savez


if __name__ == "__main__":
    print("[long-λ] τ={}  σ_xy={} cells  σ_path={} m"
          .format(EA.AC_PULSE_TAU, EA.AC_SOURCE_SIGMA_CELLS,
                  EA.PATH_SMOOTH_SIGMA_M))
    EA.main()
