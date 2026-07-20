# Uncertainty qualification experiment

`experiments/exp9_uncertainty.py` supplies quantitative evidence for the
uncertainty-aware safety-margin term already defined in the manuscript. It
does not require a real wind field and does not change the paper's Methods.

## Questions answered

1. Does increasing `D_diff` change the nominal wave-guided path and its
   clearance from a controlled building?
2. How large is the positional uncertainty tube along that path?
3. What fraction of stochastic trajectories collide with the building?
4. Does the sampled terminal-error p95 agree with the analytical 3D radius?

The experiment deliberately distinguishes three quantities that are often
conflated:

- `2*sqrt(tau)`: the radius proxy written in the manuscript;
- `sqrt(2*tau)`: the exact standard deviation of each Cartesian component;
- `sqrt(2*tau*chi2.ppf(0.95, 3))`: the exact 3D 95% confidence radius.

## Protocol

- Geometry: the same 5 m single-box domain as E1.4/Fig. 3.
- Wind: zero, so diffusion is the only changed variable.
- Sweep: `D_diff = {0, 0.5, 1, 2.5, 5, 10, 20} m^2/s`.
- For each level: solve an empty companion domain to recalibrate the source
  peak bias, solve the obstacle domain, trace and smooth the path, then draw
  4000 Brownian trajectory realisations.
- Primary endpoints: nominal path time, minimum building clearance, minimum
  `clearance-r95`, collision probability with Wilson 95% CI, and Monte-Carlo
  terminal-position-error p95.
- Reproducibility: fixed random seeds, JSON metadata, NPZ arrays, and git SHA.

## RTX 5090 commands

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
python -m experiments.exp9_uncertainty --quick
python -m experiments.exp9_uncertainty --samples 4000
python -m tools.visualize.paper.fig10_uncertainty
```

The controlled domain is small, so the full sweep should take minutes rather
than hours on the RTX 5090. The same UQ functions can subsequently be applied
to selected paths from `input/domain_5m.npz` without changing the statistical
protocol.
