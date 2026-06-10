# Aura — Stage 1 (Nanshan 20 km², transport-only)

City-scale UAV routine scheduling. See `../Leviston_Rountine_Scheduler.pdf` for
full architecture. Stage 1 scope and choices are pinned in
`../.claude/plans/system-leonardo-scratch-large-userexter-shimmering-wombat.md`
(or whichever plan file produced this build) and in `config/stage1.yaml`.

## Stage 1 locked decisions

| Decision | Choice |
| --- | --- |
| Wind | Constant +x at 3 m/s. SCALED swap point in `wind/`. |
| Demand | Synthetic Poisson driven by WorldPop population grid. |
| FMM | `agd` Rander3, Zermelo->Rander, CUDA backend preferred. |
| MAPF | `cbs-py`, delay-only solutions. |
| Tube MPC | `do-mpc` point-mass 3D, horizon 20 @ 50 Hz. |

## Layer map

```
ingest/  -> OSM streets, building footprints, WorldPop raster
geom/    -> rasterization, masks, CRS plumbing
wind/    -> IWindField interface (Stage1 = ConstantWind)
cost/    -> 4D Rander metric assembly
network/ -> L0 station siting + Voronoi
fmm/     -> L2 anisotropic FMM (offline + incremental)
topo/    -> L1 stub (Stage1 = all-pairs pass-through)
assignment/ -> L0.5 ALNS
mpc/     -> L3 tube MPC
cbs/     -> L4 corridor conflict resolution
sim/     -> layered scheduler + UAV plant
missions/-> delivery / passive sensing / hazards
metrics/ -> KPI + regression
viz/     -> 2D + GIF
io/      -> versioned asset and trajectory store
cli/     -> build_assets, run_stage1
tests/   -> R1-R6 regression suite
```

## Quick start (after running install_stage1.sh)

```bash
python -m aura.cli.build_assets --bbox nanshan --out runs/v1/
python -m aura.cli.run_stage1 --assets runs/v1 --demo
```

## Development conventions

- All public APIs use `numpy` arrays in SI units (m, s, kg). CRS = EPSG:32650 (UTM 50N).
- All rasters are stored row-major with shape `(nx, ny[, nz])`, with `axes=("x","y","z")` consistent with `aura.geom.grid.SimGrid`.
- No emojis in files. No multi-line docstrings unless the WHY is non-obvious.
- Reused patterns from `aura_toy_archive/` are noted via `# pattern: aura_toy_archive/<file>:<func>` comments where helpful.
