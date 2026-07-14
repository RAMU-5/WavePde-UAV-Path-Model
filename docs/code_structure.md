# Code structure

The repository has two routing implementations and a set of research drivers.
Keep reusable code separate from scripts that read files or submit jobs.

## Stable packages

- `model/` — WavePDE solver, path tracing, UAV dynamics, uncertainty and
  scheduling primitives. Import its public API with `from model import ...`.
- `aura/` — anisotropic fast-marching baseline and its Stage 1 simulation.
  It has a separate dependency set in `aura/requirements.txt`.

Code in these packages should not import from `experiments/`, `scripts/` or
`tools/`. The WavePDE time-step implementation remains owned by
`model/solve_all.py`; solver variants reuse it.

## Research and operational entrypoints

- `experiments/` — numbered, paper-facing experiments and `validation/` checks;
  run one as `python -m experiments.exp0_analytic`.
- `scripts/` — data preparation, aggregation and SLURM entrypoints; run a Python
  step as `python -m scripts.build_domain`.
- `tools/` — OSM ingestion, scheduling prototypes and visualisation programs.

These layers may import a stable package. They should resolve data relative to
the repository root and write generated files under `output/`.

## Data and documents

- `input/` — small, versioned inputs required to reproduce runs.
- `output/` — generated fields, databases, figures and logs; ignored by Git.
- `paper/` — paper source and selected figures.
- `docs/` — design notes, experiment plans and supporting documents.

## Development commands

Install the core package in editable mode:

```bash
python -m pip install -e '.[dev,experiments]'
```

Then use `make check`, `make validate-fast`, or `make test`. GPU/full-domain
validation remains in `experiments/validation/` and the SLURM scripts because
it is too expensive for the default test suite.
