# model/ — convected wave-equation solvers

See the repository-level `README.md` for the full project map and
workflow. This directory holds the method itself.

Physics: `(∂t + q·∇)²ψ − ∇·(c²∇ψ) + damping = s(x, t)`, with
anisotropic UAV speed (c_xy horizontal; c_z vertical with climb/descend
asymmetry encoded as a constant vertical advection w_grav), uniform wind
q, hard Dirichlet walls on buildings and an 8-layer sponge at the outer
boundary. Discretisation: variable-coefficient flux Laplacian
(face-averaged c²) + centred advection, explicit leapfrog in time,
float32 JAX with persistent JIT cache. Arrival time T = first amplitude
peak per cell.

Conventions shared by every script here:

- inputs are read from `../input/` (`domain.npz`, `domain_5m.npz`)
- results are written to `../output/wave/`
- the JIT cache lives in `../_jax_cache/` (shared across SLURM tasks)
- `solve_all.py` owns the kernel (`step_kernel`, `solve_one`,
  `build_static_fields`); all variants import from it — never duplicate
  the PDE step
- scripts outside `model/` import these modules via
  `sys.path.insert(0, <root>/model)`

The parameterised library form (WaveConfig dataclass, no hard-coded
constants) is `../routine/wave.py`.
