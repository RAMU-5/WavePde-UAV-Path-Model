# wave_test_large — convected wave-equation routing solvers

3D convected wave equation `(∂t + q·∇)²ψ − ∇·(c²∇ψ) = s` solved with an
explicit leapfrog scheme in JAX (float32, persistent JIT cache). The
first-peak arrival time T(x) of a Gaussian pulse gives the optimal UAV
travel-time field used for routing on the Nanshan domain (118 stations).

A parameterised library version of the same kernel lives in
`../routine/wave.py` (`WaveConfig` dataclass, `solve_forward` /
`solve_backward`); the scripts here are the batch / experiment entry
points with the paper's tuned constants hard-coded.

## Layout

```
wave_test_large/
├── domain.npz, domain_5m.npz     # INPUTS: mask, dx/dy/dz, stations (stay here)
├── stations.json
├── outputs/                      # all result npz/png land here
│   ├── T_source_5m_<i>.npz       #   per-source T fields (118, from SLURM array)
│   ├── pair_matrix*.npz          #   118×118 travel-time matrices
│   ├── pair_times.npz            #   solve_all.py checkpoint
│   └── scheduler_demo.npz, smoke_5m_T.npz, ...
├── validation/                   # 2D/minimal pure-python validations
├── jobs/, slurm_logs/, submit_*.sh
├── venv/, _jax_cache/            # environment + JIT cache (not code)
└── *.py                          # solvers & post-processing (below)
```

## Solver scripts

| Script | What it does | Paper § |
|---|---|---|
| `solve_all.py` | Core solver. `build_static_fields` + `solve_one` (leapfrog kernel). Sequential all-pairs loop with checkpointing | core |
| `solve_one_source.py` | One source per SLURM array task → `outputs/T_source_*.npz` | core |
| `solve_backward.py` | Adjoint/reciprocity solve: source at destination, wind q→−q; one solve gives all-origins→dst times | 2.5 |
| `solve_broadband.py` | Long-wave Gaussian + short-wave Gabor source in a single solve | 2.9.4–5 |
| `solve_multicolor.py` | Graph 4-colouring domain decomposition: non-conflicting sources batched per solve (~4× fewer solves) | 2.13.1 |
| `solve_coupled.py` | Fixed-point iteration between wave parameters and UAV dynamics | 2.10 |
| `smoke_5m.py` | Quick single-source smoke test on the 5 m grid | — |
| `test_wavelength.py`, `exp1_uav_3d.py`, `exp2_aircraft*.py`, `exp_sk_wind.py` | Experiments (pulse-width sweep, regional aircraft domain, wind) | exp |

All variants import the kernel from `solve_all.py` — there is exactly one
implementation of the PDE step.

## Post-processing / downstream

- `aggregate_pair_matrix.py` — assemble `outputs/T_source_*.npz` →
  `outputs/pair_matrix.npz` (use `--prefix T_source_5m_ --domain domain_5m.npz --out pair_matrix_5m.npz` for the 5 m run)
- `build_database.py` — training database from the pair matrix
- `scheduler_meanfield.py`, `viz_scheduler.py` — mean-field assignment demo
- `viz3d.py`, `viz_aux_features.py`, `post_wavelength*.py` — figures
- consumers elsewhere: `tools/scheduling/*` and
  `tools/visualize/scheduling/simulation.py` read `outputs/pair_matrix.npz`

## Validation

| Test | Checks |
|---|---|
| `validation/back_sym_test.py` | reciprocity T_bwd(B→A, −q) = T_fwd(A→B, q), pure python, q=0 and uniform q (+ no-flip control) |
| `validate_backward_minimal.py` | same, 2D numpy version |
| `validate_backward_real.py` | forward-vs-backward pairs on the real domain (`--pair-file` resolves into `outputs/`) |
| `validation/validate_broadband_minimal.py` | linearity: ψ_bb = ψ_long + ψ_short |
| `validation/validate_multicolor.py` | batched multi-source = superposition |
| `validate_all_real.py`, `validation/validate_*` | remaining paper-claim checks |

Known caveat: reciprocity is exact for divergence-free q (current uniform
wind qualifies); re-validate with a spatially varying wind field before
trusting `solve_backward.py` there. The sponge damping assumes first-arrival
paths stay clear of the outer boundary.

## Typical workflow

```bash
sbatch submit_array_5m.sh                       # 118 tasks → outputs/T_source_5m_*.npz
python aggregate_pair_matrix.py --domain domain_5m.npz \
    --prefix T_source_5m_ --out pair_matrix_5m.npz
python build_database.py                        # or scheduler/viz scripts
```
