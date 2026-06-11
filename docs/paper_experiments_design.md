# Paper experiment design — the Results section, figure by figure

Companion to `experiment_plan.md` (which orders the *runs* small→large
and carries the per-experiment protocols). This document designs the
*paper*: the narrative of the Experiments/Results section, every figure
and table with its axes and expected message, the main-text/appendix
split, and the minimum viable set for a first submission.

The current draft (`neural_physics_uav_flight_path_optimisation.pdf`)
has Method (§2 wave–advection formulation, §3 hierarchical routing,
§4 city-scale parallelisation) and Discussion but no Results section.
The design below slots in as **§5 Numerical Experiments**, pushing
Discussion/Conclusion to §6/§7.

---

## The story in one paragraph

The Results section must answer four reviewer questions in order:
(Q1) *Is it correct?* — verified against analytic solutions and a
discrete ground truth; (Q2) *Why not just use Dijkstra/FMM?* — because
one GPU-resident PDE kernel handles wind + anisotropy + 3D obstacles
natively and admits adjoint/superposition tricks that change the
asymptotics of all-pairs problems; (Q3) *Where does it break?* — a
measured wavelength operating point and a measured ∇·q reciprocity
limit, not hand-waving; (Q4) *Does it matter downstream?* — city-scale
scheduling KPIs improve when fed wave-accurate times. Each subsection
below is one of those answers.

## §5.1 Experimental setup (½ page, no results)

Domains table (Table 1) — reused by every later experiment:

| Domain | Grid | Resolution | Purpose |
|---|---|---|---|
| `analytic` (empty / 1 obstacle) | up to 512³ | 2.5–20 m | exact-solution benchmarks |
| Nanshan (`domain_5m.npz`) | ~700×700×40 | 5 m | real city, 118 stations |
| Nanshan coarse (`domain.npz`) | ~70×70×40 | 50 m | sweeps, ablations |
| Regional (`exp2_aircraft.py`) | 80×80×12 km | 200 m | scale transfer |

Fixed method parameters (c_xy=15, cz↑=6, cz↓=4 m/s, dt=0.1 s, 8-layer
sponge, float32, A100) and the **four metrics used everywhere**:

- **optimality gap** = (T_method − T_ref)/T_ref over valid pairs (700–3000 m)
- **path time** = dynamics-integrated time along the traced path (the
  honest number — never report raw T as path cost)
- **success rate** = traced path reaches destination within ε = 2 cells
- **runtime** = wall-clock, JIT-warm, median of 3, one A100

State the peak-detection bias t₀ and its calibration here, once
(measured in E1.1), so every later number is post-calibration.

## §5.2 Verification & accuracy → claims C1, C2  *(runs: E1.1–E1.4, T0.x)*

The "is it correct" subsection. Four panels + one table.

| Item | Spec | Message |
|---|---|---|
| **Fig. 1** convergence | log-log: relative T error vs dx ∈ {20,10,5,2.5} m, one curve per pulse width τ ∈ {2,4,8,16}; fitted slope annotated | "2nd-order in space; residual bias is τ-limited and calibrated out" |
| **Fig. 2(a)** wind ellipse | T contours around a source under uniform q, overlaid on the analytic ellipse d/(c±q·d̂); inset: error vs Mach q/c ∈ {0.1,0.3,0.5} | advection terms are quantitatively right, not just reciprocal |
| **Fig. 2(b)** anisotropy | vertical-pair arrival times vs the 1D exact d/cz↑, d/cz↓ | the w_grav climb/descend encoding is exact |
| **Fig. 3** single obstacle | T-slice + traced path, wave (5 m) vs reference (0.5 m wave or aura FMM); path-time error printed | diffraction/shadowing handled; path metric introduced visually |
| **Table 2** verification suite | the five Tier-0 checks (reciprocity 5/5, broadband linearity, multicolor superposition, real-domain fwd/bwd, coupled convergence) with max errors | machine-checkable correctness, zero extra compute |

Risk to control: E1.1 must subtract source onset t₀ before fitting the
convergence order, otherwise the curve floors at the bias level and a
reviewer will call the order claim wrong.

## §5.3 Comparison with baselines → C1 on the real city  *(runs: E2.1–E2.3)*

The "why not Dijkstra/FMM" subsection. One headline figure, one table.

| Item | Spec | Message |
|---|---|---|
| **Fig. 4** (headline) | (a) scatter of 118×118 pair times, T_wave vs T_dijkstra (26-neighbour anisotropic voxel Dijkstra = discrete ground truth), identity line; (b) optimality-gap histogram with median/p95 annotated | "within X% of ground truth across ~7k city pairs" — *the* accuracy number of the paper |
| **Table 3** method comparison | rows: wave (ours), aura FMM (Rander3), Dijkstra; columns: median/p95 gap, runtime per source, memory, GPU-friendly?, native wind?, native anisotropy?, superposition tricks? | honest: FMM may win single-source CPU accuracy; we win throughput + generality + batched/all-pairs asymptotics |
| inline | T/(d/c) ratio distribution (already computed: median 1.24, p95 1.78) | detour structure of the city, sanity lower bound |

Fairness requirements (reviewers will check): identical voxel mask and
station nudging for all three methods; Dijkstra edge weights use the
same c_xy/cz↑/cz↓; FMM gets the same anisotropic metric; report both
per-source and 118-source-total runtimes (the superposition tricks only
help the latter — say so explicitly).

## §5.4 Efficiency → C3, C4, C7  *(runs: E3.1–E3.3)*

The "what the wave formulation buys you" subsection.

| Item | Spec | Message |
|---|---|---|
| **Fig. 5** pipeline costs | one bar chart, end-to-end wall time for the full 118×118 matrix: naive 118 forward / multicolor ~30 solves / 1-backward-per-destination; each bar annotated with max |ΔT|/T vs naive | superposition + reciprocity change the constant *and* the asymptotics: K destinations ⇒ K solves → 1 |
| **Table 4** throughput | grid size (50 m → 5 m → 2.5 m sub-domain) vs steps/s, cell·steps/s, JIT-warm vs cold; A100 | reproducibility + the C7 number "118 sources at 5 m in ~15 min" |
| inline | adjoint accuracy: 10 random destinations, K=118 forward column vs 1 backward, max/median |ΔT|/T | speedup is not bought with error |

## §5.5 Robustness & operating envelope → C5 + wind physics  *(runs: E4.1–E4.5)*

The "where does it break" subsection — measured limits, the paper's
credibility section. Two figures main text, two closures appendix.

| Item | Spec | Message |
|---|---|---|
| **Fig. 6** wavelength operating point | path-time optimality gap (vs Dijkstra GT) vs λ/building-spacing; vertical band = recommended operating range; broadband point overlaid | principled parameter choice; motivates broadband (C5) |
| **Fig. 7** varying-wind limit | (a) paths under a divergence-free vortex field drift downwind, vs wind-weighted Dijkstra; (b) reciprocity error vs ‖∇·q‖ for progressively non-solenoidal fields | the known caveat becomes a measured validity limit |
| App. fig | broadband vs long-only vs short-only on 20 narrow-street pairs: success rate + path quality at equal cost | C5 value quantified |
| App. fig | wind-inversion closure (synthetic flight → recovered wind → re-plan, route difference) and coupled-iteration convergence from 5 inits on 10 pairs | the estimation→planning and wave↔dynamics loops close |

## §5.6 System level → C8  *(runs: E5.1–E5.2)*

The "does accuracy matter downstream" subsection — connects to paper §3–4.

| Item | Spec | Message |
|---|---|---|
| **Table 5** scheduler | mean-field vs Hungarian vs greedy on the real 118×118 cost matrix: cost gap, runtime vs problem size 10→5000 pairs | mean-field ≈ Hungarian quality at greedy-like cost |
| **Fig. 8** city simulation | 1M orders/day, 7080 UAVs: delivery KPIs (on-time %, fleet utilisation, p95 delivery time) with wave-T routing vs straight-line-time routing | routing accuracy propagates to fleet KPIs — the system claim |

## §5.7 Scale transfer → C9  *(run: E6.1)*

Half a page. **Fig. 9**: Nanshan multirotor T-field/path next to the
80 km regional-aircraft domain (200 m/s, climb profiles), same kernel,
config table inset. Message: "the method is a configuration, not a
codebase per vehicle."

---

## Main text vs appendix

Main text (9 figures, 5 tables — trim to venue limit by demoting
Fig. 2b, Fig. 3, Table 4 first):

- Figs: 1 convergence · 2 wind+anisotropy · 3 obstacle · **4 GT scatter
  (headline)** · 5 pipeline bars · 6 wavelength · 7 wind limit ·
  8 city KPIs · 9 scale transfer
- Tables: 1 domains · 2 verification · 3 method comparison ·
  4 throughput · 5 scheduler

Appendix: broadband ablation, inversion/coupling closures, JIT-cache &
SLURM reproducibility details, station-nudging protocol, full
118×118 matrices as artifacts.

## Minimum viable set (first submission, ~1 week, ≤3 GPU-days)

1. **E1.1** convergence (Fig. 1) — cheapest, calibrates t₀ for everything
2. **E2.1** Dijkstra GT + scatter (Fig. 4) — the headline; everything
   else inherits its reference
3. **E3.1 + E3.2** adjoint + multicolor (Fig. 5) — the efficiency claim
   is the novelty; data mostly exists
4. **E3.3** throughput (Table 4) — free, instrument existing runs
5. **E5.1** scheduler (Table 5) — code exists, CPU only
6. **E6.1** aircraft figure (Fig. 9) — exists, needs replotting

Defer to revision if pressed: E2.2 FMM (needs aura tuning), E4.3–E4.5
(wind physics), E5.2 full simulation.

## Expected headline numbers (to be filled, with falsification criteria)

| Claim | Expect | Abort/rethink if |
|---|---|---|
| convergence order | ≈2.0 in dx | <1.5 → check flux Laplacian / peak detector |
| optimality gap vs Dijkstra | median ≤ ~3%, p95 ≤ ~8% | p95 > 15% → wavelength operating point wrong, run E4.1 first |
| adjoint vs forward column | |ΔT|/T ≤ 1% (uniform q) | systematic sign bias → sponge asymmetry, enlarge margin |
| multicolor | ~4× fewer solves, error ≪ gap | crosstalk > gap → increase colour separation |
| 118×5 m matrix | ~15 min wall on A100 array | — (already measured, re-confirm) |

Every experiment writes `output/runs/<exp_id>/` (npz + json metadata +
the exact git SHA) so figures are regenerable by
`tools/visualize/paper/` scripts — one script per figure, named
`fig01_convergence.py`, … This is the contract that keeps the paper
rebuildable.
