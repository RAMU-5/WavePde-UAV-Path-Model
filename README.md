# WavePde-UAV-Path-Model

UAV routing and fleet scheduling via the convected wave equation:
`(∂t + q·∇)²ψ − ∇·(c²∇ψ) = s` is solved with an explicit leapfrog
scheme in JAX (float32, persistent JIT cache); the first-peak arrival
time T(x) of a Gaussian pulse gives the optimal travel-time field used
for path tracing and scheduling on the Nanshan domain (118 stations,
Shenzhen). Scales from delivery multirotors (400 m buildings) to
regional aircraft (80 km domains).

## Repository layout

```
model/         the method — PDE solvers (one leapfrog kernel in solve_all.py;
               backward/broadband/multicolor/coupled variants import it),
               wave_api.py (WaveConfig parameterised API) + plan.py
               (one-call plan_path) + path_trace.py (-∇T streamlines),
               UAV dynamics, wind inversion, mean-field scheduler
input/         domain voxel masks + stations (domain.npz 50 m, domain_5m.npz 5 m,
               stations.json); rebuild with scripts/build_domain.py
output/        ALL generated artifacts (gitignored):
               wave/ (T fields, pair matrices), database/ (training paths),
               figures/, runs/, slurm_logs/
scripts/       SLURM submit scripts + pipeline steps
               (build_domain → submit_array_5m → aggregate_pair_matrix → build_database)
experiments/   paper experiments (exp1 UAV 3D, exp2 regional aircraft,
               wavelength sweeps, SCALED wind coupling) +
               validation/ — correctness tests (reciprocity, linearity,
               multicolor superposition, coupled iteration, real-domain checks)
tools/         scheduling simulators + figure generators
               (visualize/{wave,routine,scheduling}) + osm/ — OSM ingestion /
               voxelisation scripts (raw geodata lives in .cache/shenzhen-osm/,
               not in git)
aura/          FMM (Rander3 anisotropic eikonal) routing framework — baseline
docs/          paper drafts + implementation notes + experiment plan
venv/          Python env with jax[cuda12] (gitignored)
```

## Solver map (model/)

| File | What it does | Paper § |
|---|---|---|
| `solve_all.py` | Core kernel: `build_static_fields` + `solve_one` (leapfrog), sequential all-pairs with checkpointing | core |
| `solve_one_source.py` | One source per SLURM array task → `output/wave/T_source_*.npz` | core |
| `solve_backward.py` | Adjoint/reciprocity solve (source at destination, wind q→−q): one solve gives all-origins→dst times | 2.5 |
| `solve_broadband.py` | Long Gaussian + short Gabor source in one solve | 2.9.4–5 |
| `solve_multicolor.py` | 4-colouring domain decomposition, batched non-conflicting sources (~4× fewer solves) | 2.13.1 |
| `solve_coupled.py` | Wave-parameter ↔ UAV-dynamics fixed-point iteration | 2.10 |
| `uav_dynamics.py` | Point-mass UAV dynamics integration | 2.10 |
| `wind_inversion.py` | Kinematic wind estimation from flight tracks | 2.6.2 |
| `scheduler_meanfield.py` | Mean-field task assignment (vs Hungarian/greedy) | sched |
| `wave_api.py` | `WaveConfig`-parameterised API over the same kernel (`solve_forward`/`solve_backward`) | lib |
| `plan.py` + `path_trace.py` | One-call `plan_path`: solve → trace −∇T streamline → smooth | lib |

There is exactly one implementation of the PDE step; every variant and
the API layer import it from `solve_all.py`. Library quick start:

```python
import sys; sys.path.insert(0, "<root>/model")
from plan import plan_path
res = plan_path(mask, dx, dy, dz, src_xyz, dst_xyz)
print(res.travel_time, res.status)
```

## Typical workflow

```bash
# 0. (once) build domain from OSM
venv/bin/python scripts/build_domain.py --dx 5 --dy 5 --dz 5 --out domain_5m.npz
# 1. 118 forward solves on GPU (one per array task)
sbatch scripts/submit_array_5m.sh
# 2. assemble 118×118 travel-time matrix
venv/bin/python scripts/aggregate_pair_matrix.py --domain domain_5m.npz \
    --prefix T_source_5m_ --out pair_matrix_5m.npz
# 3. downstream: database / scheduling / figures
venv/bin/python scripts/build_database.py
```

Quick correctness check (pure python, ~2 s): `python3 experiments/validation/back_sym_test.py`

## Validation caveats

Reciprocity (backward solver) is exact for divergence-free wind — the
current uniform-wind configuration qualifies; re-validate before using a
spatially varying wind field. The sponge damping assumes first-arrival
paths stay clear of the outer boundary.

See `docs/experiment_plan.md` for the paper experiment programme.
