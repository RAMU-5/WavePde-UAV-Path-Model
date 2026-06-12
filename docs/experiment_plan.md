# Experiment plan — validating the wave-PDE routing method

Companion: `paper_experiments_design.md` designs the Results *section*
(figure/table specs, narrative, main-vs-appendix split); this file
orders the *runs*; `paper_experiments_checklist_zh.md` is the Chinese
quick-reference progress checklist.

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
  (✗ out of the experiments chapter — routing-only scope, 2026-06-12)
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

## Tier 1 — analytic accuracy & convergence ✅ DONE 2026-06-11
(`experiments/exp0_analytic.py`, results in `paper/Q1/`; headline:
pointwise ≤0.9% at the production point, geodesic gap ≤0.52%, measured
order 0.6–0.9 — the arrival-time observable is not a smooth-field L2
norm, do not claim 2nd order; M≤0.3 wind envelope measured)

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

**E2.1 Discrete ground truth.** ✅ DONE 2026-06-11 (`exp1_dijkstra_gt.py`,
results in `paper/Q2/`). Anisotropic Dijkstra + metric-aware
string-pulling (raw 26-neighbour graph carries +1–5% angular bias).
Headline: 5 m, 555 valid pairs, med −4.1%, p95 |gap| 6.4%. Exposed and
fixed a production bug: min_peak_amp must scale with dx³ —
pair_matrix_5m_v2.npz is the corrected matrix; downstream consumers
(database/scheduling) should migrate to v2.

**E2.2 FMM comparison.** aura's Rander3 solver on identical inputs:
accuracy vs runtime vs memory, GPU-friendliness. Honest comparison —
FMM is the natural competitor; the paper must show where the wave
formulation wins (GPU throughput, wind generality, superposition
tricks) and where it doesn't.

**E2.3 Lower bound sanity.** T / (d/c) ratio distribution (already
computed: median 1.24, p95 1.78 at 5 m) — include as a table.

## Tier 3 — efficiency claims ✅ DONE 2026-06-12

(`experiments/exp7_efficiency.py`, results in `paper/Q4/`)

**E3.1 Adjoint speedup (C3).** ✅ 10 destinations, 476 band pairs vs
the production forward matrix: med 0.35%, p95 1.0%; one outlier pair
(59↔29) where the production matrix itself returns inf in reverse —
detector fragility, not reciprocity failure. One destination column:
96 min → 49 s = 118×.

**E3.2 Multicolor scaling (C4 — narrowed).** ✅ Measured crosstalk
threshold = 2×band radius = 6 km, exactly the disk-overlap criterion.
On this 7 km domain with the 3 km band: 82 colours = only 1.4×; the
"~4×" claim holds for a ≤1.5 km delivery-radius matrix (28 solves,
4.2×). Batch size does not change per-solve cost.

**E3.3 GPU throughput (C7).** ✅ Warm: 50 m 0.24 s / 5 m 48.9 s /
2.5 m subdomain 76.7 s (dt=0.05, CFL); large grids saturate at
~7e12 cell·steps/s; 50 m grid is launch-latency bound. Full 118
matrix re-measured: 96 A100-min (49 s/solve), ~12 min on an 8-GPU
array — restate C7 accordingly.

## Tier 4 — robustness & physics (2–3 GPU-days) 🔶

**E4.1 Wavelength operating point.** ✅ DONE 2026-06-12
(`exp3_wavelength_gap.py`, results in `paper/Q3/`). Gap is
λ-insensitive for λ ≤ 0.6·street width (med ≈ −4%, p95 6.5%);
long-λ limit at λ ≳ 1.2·street width (−6.4%/9.7% at 2.5×); short
end grid-saturated. Production τ=0.4 sits mid-plateau.

**E4.2 Broadband value (C5).** ✅ DONE 2026-06-12 (`exp5_broadband.py`,
results in `paper/Q3/`). On the 20 hardest long-λ pairs the bottleneck
is the tracer, not the T field: raw tracing 14/10/10 (short/long/bb),
normalised-convolution smoothing lifts all three bands to 17/20 —
broadband adds no path-level success. C5 narrowed to field-level value
+ long-λ gap repair (E4.1 item 4).

**E4.3 Spatially varying wind.** ✅ DONE 2026-06-12 (`exp4_wind.py`,
results in `paper/Q3/`). (a) per-pair wind-induced ΔT vs wind-aware
Dijkstra (full-wind quadratic metric): corr 0.86 at M=0.3, residuals
within 50 m resolution noise; (b) reciprocity error vs ‖∇·q‖ at fixed
|q|=4.5: 1.5% discrete floor → 3.7% med at div 2.8e-2 1/s (pure
compressive) — graceful adjoint degradation.

**E4.4 Wind-inversion closure.** ✅ DONE 2026-06-12 (`exp6_closure.py`,
results in `paper/Q3/`). Vortex-M0.3 truth → 1254 noisy segment
observations → kernel regression (L=350 m; L=800 attenuates the 3 km
vortex to 25%, exp(−k²L²/2)) → re-solve: med |ΔT|/T 3.8% vs 8.5%
ignoring wind — closure error halves even though pointwise field
rel RMS is only ~0.73 (tracks sample exactly the corridors that
matter).

**E4.5 Coupled iteration on the real domain (C6).** ✅ DONE 2026-06-12
(same job). 33/50 runs converge, median 4 iterations; per-pair c0*
spread across inits 1–2 m/s (clear attractor); 2/10 pairs oscillate.
Fixed point (c0*≈5.7) is set by the surrogate dynamics on unsmoothed
voxel paths, not the regulatory production params — report the
contraction property, not parameter recovery.

## Tier 5 — system level ✗ OUT OF SCOPE (2026-06-12)

The paper validates the ROUTING algorithm only; scheduler comparisons
and city-scale simulation are not part of it (user decision). Claim
C8 is dropped from the experiments chapter accordingly.

**E5.1 Scheduler optimality.** ✗ Out of scope. (An exploratory run
exists — `experiments/exp8_scheduler.py`, results in
`output/runs/exp8_scheduler/` — kept for reference, not in the paper.
Its takeaway, should it ever be needed: scipy Hungarian is optimal
and fastest at every size tested; the mean-field variants do not win
on a single node.)

**E5.2 City-scale simulation.** ✗ Out of scope, same decision.

## Tier 6 — scale transfer ✅ DONE 2026-06-12

**E6.1 Regional aircraft (C9).** ✅ Fig.9 (`fig09_transfer.py`):
Nanshan 5 m T-slice + smoothed-tracer path (exp7b — the raw tracer
stalled at 187 m, re-confirming E4.2's tracer-bottleneck finding)
side by side with the 80 km aircraft domain's 4 cloud-avoiding
trajectories; config-swap table in the caption.

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
