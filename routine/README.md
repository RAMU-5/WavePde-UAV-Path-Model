# routine — neural-physics UAV path planner

Station-to-station flight path planning by solving a convected-wave PDE
on a 3D building voxel mask, then tracing the optimal path as the
gradient-descent flow of the arrival-time field `T(x,y,z)`.

This is a packaging of the algorithm from
`neural_physics_uav_flight_path_optimisation/main_article.tex`,
extracted from the working scripts in `wave_test_large/`
(`solve_all.py`, `solve_coupled.py`, `solve_backward.py`,
`exp_sk_wind.py`, `exp2_aircraft.py`). The PDE kernels and trace logic
are the original implementations; only the constants and helper plumbing
were reorganised into `WaveConfig` / `StaticFields` / `plan_path` so the
algorithm can be called from a single line.

## Pipeline

```
(mask, src_xyz, dst_xyz)
    │
    ├── build_static_fields ── (c²_xy, c²_z, u, v, w, ax, ay, az, damp, free)
    │
    ├── gaussian_source     ── soft point pulse at src cell
    │
    ├── solve_one (JAX)     ── leapfrog the convected wave PDE
    │                            D²ψ/Dt² − ∇·(c²∇ψ) + damping = src
    │                          → T(x,y,z) = wavefront first-peak time
    │
    └── trace_path_3d       ── Euler descent on −∇T from dst back to src
                              → (M, 3) trajectory in metres
```

The optimal travel time from `src_xyz` to `dst_xyz` is `T[dst_idx]`; the
optimal path is the integral curve of `−∇T` reaching `src` (Fermat's
principle).

## Install

```bash
pip install -e routine/                 # CPU JAX
pip install -e 'routine/[cuda]'         # CUDA JAX
pip install -e 'routine/[dev]'          # + pytest
```

JAX wheel selection is environment-dependent; on Leonardo we typically
reuse the venv in `wave_test_large/venv/` which already has JAX configured.

## Quick start

```python
import numpy as np
from routine import plan_path, WaveConfig

# (mask is a (NX,NY,NZ) bool array — True = building / no-fly cell)
mask = np.load("wave_test_large/domain.npz")["mask"]

res = plan_path(
    mask, dx=50.0, dy=50.0, dz=14.0,
    src_xyz=(1200.0, 1500.0, 5.0),
    dst_xyz=(4400.0, 4800.0, 5.0),
    cfg=WaveConfig(c0=15.0, cz_up=6.0, cz_down=4.0,
                    wind=(0.0, 0.0, 0.0), n_steps=4500),
    mode="three_phase",   # vertical climb + 2D cruise + vertical descent
    cruise_z_m=40.0,
    smooth_sigma_m=80.0,
    return_T=False,
)
print("travel time:", res.travel_time, "s")
print("path shape :", res.path.shape, "status:", res.status)
```

## Reusing T across many destinations from one source

```python
from routine import build_static_fields, solve_forward, trace_path_3d

static = build_static_fields(mask, dx, dy, dz)         # build once
T = solve_forward(mask, dx, dy, dz, src_idx, static=static)
for dst_xyz in destinations:                            # cheap trace per dst
    path, status = trace_path_3d(T, src_xyz, dst_xyz, dx, dy, dz)
```

## Reusing one T for many sources (adjoint / backward)

`routine.solve_backward(mask, dx,dy,dz, dst_idx)` solves with the source
placed at the destination and `q → −q`. The resulting T-field gives the
travel time **to** `dst` from every cell — one solve covers K candidate
origins (paper §2.5).

## Public API

| name | purpose |
|---|---|
| `WaveConfig`, `StaticFields` | PDE parameters and pre-built fields |
| `build_static_fields(mask, dx, dy, dz, cfg)` | wave-speed + advection + sponge |
| `gaussian_source(shape, idx)` | soft point pulse |
| `solve_one(static, src, dx, dy, dz, cfg)` | one leapfrog wave solve |
| `solve_forward(mask, dx, dy, dz, src_idx, cfg, static=None)` | forward T-field |
| `solve_backward(mask, dx, dy, dz, dst_idx, cfg)` | adjoint T-field |
| `trace_path_3d(T, src, dst, dx, dy, dz)` | 3D gradient descent |
| `trace_path_2d_at_z(T, iz, src_xy, dst_xy, dx, dy)` | 2D slice descent |
| `build_three_phase_path(T, src, dst, dx, dy, dz, cruise_z_m)` | climb-cruise-descent |
| `smooth_path(path, sigma_m)` | arc-length Gaussian smoother |
| `snap_to_free_cell(mask, xyz, dx, dy, dz)` | nudge station out of buildings |
| `plan_path(mask, dx, dy, dz, src, dst, ...)` | one-call wrapper |

## Notes

- All inputs are **numpy** arrays in SI units. T is returned as `np.float32`.
- `WaveConfig.n_steps × dt` must be longer than the slowest valid travel
  time across the domain, otherwise distant cells stay at `+inf`.
- The PDE uses an explicit leapfrog; CFL is satisfied for the defaults
  (`c0=15, cz=5, dt=0.1, dx=50, dz=14`). If you change `c0` or `dx,dy,dz`,
  check that `c·dt/min(dx,dy,dz) < 1/√3`.
- JIT compile cost is ~30–60s on first call; the persistent cache at
  `$JAX_COMPILATION_CACHE_DIR` (default: `$(cwd)/_jax_cache`) amortises
  this across subsequent runs.
