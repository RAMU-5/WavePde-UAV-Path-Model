# Experiment plan — validating the wave-PDE routing method

Goal: build the evidence chain for the paper, from cheap correctness
checks to system-scale demonstrations. Ordered so that each tier
de-risks the next; start at Tier 1 (hours, mostly CPU) before spending
GPU budget on Tiers 4–6.

Paper claims to support:

- **C1 accuracy** — first-peak arrival time ≈ optimal travel time
- **C2 generality** — anisotropy, wind, 3D obstacles handled by one PDE
- **C3 adjoint speedup** — one backward solve replaces K forward solves
- **C4 multicolor speedup** — ~4× batching via superposition
- **C5 broadband** — one solve yields both smooth and sharp guidance
- **C6 coupling** — wave ↔ dynamics fixed point converges
- **C7 performance** — 118 sources at 5 m in ~15 min on A100s
- **C8 system value** — city-scale scheduling on the T-matrix
- **C9 scale transfer** — same code from multirotor to regional aircraft

Status legend: ✅ done · 🔶 partial (code exists, needs systematic runs/figures) · ⬜ to do

---

## Tier 0 — existing validations (inventory)

| ID | Check | Status |
|----|-------|--------|
| T0.1 | Reciprocity q=0 and uniform q (`experiments/validation/back_sym_test.py`, 5/5 exact) | ✅ |
| T0.2 | Broadband linearity ψ_bb = ψ_long + ψ_short | ✅ |
| T0.3 | Multicolor batching = superposition of single solves | ✅ |
| T0.4 | Real-domain forward-vs-backward pairs (`validate_backward_real.py`) | ✅ |
| T0.5 | Coupled-iteration convergence (synthetic) | ✅ |

These belong in the paper as a "verification" table; no new compute needed.

## Tier 1 — analytic accuracy & convergence (CPU/1 GPU, ~hours) ⬜ START HERE

**E1.1 Free-space benchmark.** Empty domain, source at centre: exact
T*(x) = |x − x₀|/c. Measure relative error vs distance for
dx ∈ {20, 10, 5, 2.5} m and pulse width τ ∈ {2, 4, 8, 16} steps.
Quantifies the two intrinsic error sources: discretisation and
peak-detection bias (expected ≈ t₀ pulse offset — calibrate and subtract).
*Figure: log-log error-vs-dx convergence curve with fitted order.*

**E1.2 Uniform-wind benchmark.** Closed form for convected propagation:
arrival on an ellipse, T*(x) along/against wind = d/(c±|q|). Verify the
advection terms quantitatively, not just reciprocity.
*Table: error at Mach q/c ∈ {0.1, 0.3, 0.5}.*

**E1.3 Anisotropy benchmark.** Vertical vs horizontal propagation with
c_z ≠ c_xy and climb/descend asymmetry: compare against the 1D exact
times. Validates the w_grav encoding.

**E1.4 Single-obstacle diffraction.** One rectangular building; reference
solution = high-resolution FMM (aura) or 0.5 m wave solve. Error of the
traced path time, not just T at a point.
*Figure: paths + T-field slices, wave vs reference.*

Deliverable: "the solver is 2nd-order accurate in space; arrival-time
bias is τ-limited and calibrated out" — the foundation for C1.

## Tier 2 — baseline comparisons on the real domain (1–2 GPU-days) ⬜

**E2.1 Discrete ground truth.** Anisotropic Dijkstra on the 26-neighbour
voxel graph of `domain_5m.npz` (edge weight = segment time with c_xy,
c_z↑, c_z↓). Compute the same 118×118 matrix.
*Key figure: scatter T_wave vs T_dijkstra + optimality-gap histogram
(expect gap ≤ a few %, wave slightly smoother paths).*

**E2.2 FMM comparison.** aura's Rander3 solver on identical inputs:
accuracy vs runtime vs memory, GPU-friendliness. Honest comparison —
FMM is the natural competitor; the paper must show where the wave
formulation wins (GPU throughput, wind generality, superposition
tricks) and where it doesn't.

**E2.3 Lower bound sanity.** T / (d/c) ratio distribution (already
computed: median 1.24, p95 1.78 at 5 m) — include as a table.

## Tier 3 — efficiency claims (reuse existing data + ~1 GPU-day) 🔶

**E3.1 Adjoint speedup (C3).** For 10 random destinations: K=118 forward
column vs 1 backward solve. Report max/median |ΔT|/T and measured
wall-clock ratio. Most code exists (`solve_backward.py`).

**E3.2 Multicolor scaling (C4).** N_solves vs N_sources for the
118-station set; error vs minimum source separation; end-to-end
time for the full matrix: naive 118 / multicolor ~30 / backward-per-
destination. *Figure: one bar chart of the three pipelines.*

**E3.3 GPU throughput (C7).** Wall time vs grid size (50 m → 5 m → 2.5 m
subdomain) on one A100; steps/s and cells·steps/s; JIT-cache effect on
array jobs. *Table for the reproducibility section.*

## Tier 4 — robustness & physics (2–3 GPU-days) 🔶

**E4.1 Wavelength operating point.** Systematise `test_wavelength.py`:
path-time optimality gap (vs E2.1 ground truth) as a function of
λ/building-spacing. Identifies the operating point and motivates
broadband. *Figure: gap vs λ curve.*

**E4.2 Broadband value (C5).** Long-only vs short-only vs broadband on
N=20 hard pairs (narrow streets): success rate + path quality at equal
cost. Builds on E4.1.

**E4.3 Spatially varying wind.** Divergence-free synthetic field (vortex
street / SCALED-UrbanFlow output via `exp_sk_wind.py`): (a) forward
solve validity — paths drift downwind as expected, compare against
Dijkstra with wind-modified edge weights; (b) reciprocity error vs
∇·q — quantifies the known caveat, turning it into a measured limit
instead of a footnote.

**E4.4 Wind-inversion closure.** Synthetic flight under known wind →
`wind_inversion.py` recovery → re-solve with recovered wind → compare
routes. Closes the estimation-to-planning loop (§2.6.2).

**E4.5 Coupled iteration on the real domain (C6).** Convergence of
(c0, cz↑, cz↓) from 5 different initialisations on 10 station pairs;
iterations-to-tolerance histogram.

## Tier 5 — system level (CPU, days) 🔶

**E5.1 Scheduler optimality.** Mean-field vs Hungarian vs greedy on the
real cost matrix: cost gap and runtime vs problem size (10 → 5000
UAV-order pairs). Code exists (`scheduler_meanfield.py`, `tools/scheduling/`).

**E5.2 City-scale simulation.** `tools/visualize/scheduling/simulation.py`
(1M orders/day, 7080 UAVs, batching + predictive rebalancing): delivery
KPIs with wave-T routing vs straight-line-time routing — quantifies how
much routing accuracy matters downstream. *Figure: KPI table + timeline.*

## Tier 6 — scale transfer (exists, needs figures) 🔶

**E6.1 Regional aircraft (C9).** `experiments/exp2_aircraft.py`
(80×80×12 km, 200 m/s): same kernel, different config. Present T-field +
climb-profile paths next to the Nanshan multirotor case.

---

## Suggested execution order (small → large)

| Week | Run | Cost |
|------|-----|------|
| 1 | E1.1–E1.4 (analytic suite, new `experiments/exp0_analytic.py`) | CPU + <1 GPU-h |
| 1 | E2.3, E3.1 (reuse existing matrices) | <1 GPU-h |
| 2 | E2.1 Dijkstra ground truth + scatter figure | 1 GPU-day + CPU |
| 2 | E3.2, E3.3 (efficiency tables) | 1 GPU-day |
| 3 | E2.2 FMM comparison | 1–2 days |
| 3–4 | E4.1–E4.5 (robustness) | 2–3 GPU-days |
| 4 | E5.1–E5.2, E6.1 (system + transfer, mostly CPU/figures) | days |

Metrics, defined once and reused everywhere:

- **optimality gap** = (T_method − T_ref)/T_ref on valid pairs (700–3000 m)
- **path time** = dynamics-integrated time along the traced path (not the T value)
- **success** = traced path reaches destination within ε = 2 cells
- runtime = wall-clock on one A100, JIT-warm, median of 3

Pitfalls to control: peak-detection bias (calibrate with E1.1), station
nudging consistency between methods, sponge proximity for pairs near
the domain edge, and float32 accumulation at long propagation distances.
