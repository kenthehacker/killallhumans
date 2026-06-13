# Research Synthesis — Multi-Track Generalization (iter-003 stall)

3 substantive research reports returned (Opus 4.7 max-thinking + GPT-5.5 extra-high + Gemini 3.1 Pro). Composer crashed at 2.7s — model error, not output.

## Unanimous consensus: add corridor constraints to the trajectory optimizer

All 3 agents independently picked the same #1 fix:

| Agent | #1 Pick | Backing paper |
|---|---|---|
| Opus 4.7 | SFC + MPCC++ prismatic tunnel in poly QP + per-track ILC reset | Krinner et al., MPCC++ RSS 2024; Liu et al., SFC IEEE RA-L 2017 |
| GPT-5.5 | Gate-order corridor constraints + dense plan validator (replay sequencer on candidate trajectory) | same family of refs |
| Gemini 3.1 Pro | Safe Flight Corridors (SFC) + ILC max_corr scaled to corridor width | same |

**The diagnosis they all converge on**: the polynomial trajectory's first
segment over-curves enough that it crosses gate-2's opening before gate-1
is credited. A corridor constraint (W(θ), H(θ)) along the centerline that
**narrows at gates and widens between them** geometrically excludes future-
gate planes from earlier segments. Linear inequalities in the existing
QP — no new solver, no new dependencies.

## Convergence on secondary candidates

| Technique | Opus | GPT-5.5 | Gemini |
|---|---|---|---|
| TOGT (gate-polygon traversal) | C2 | mentioned | C1 |
| MPCC++ (full receding-horizon NLP) | C1 (offline encoding) | C2 | C2 |
| Complementary Progress Constraints (CPC) | C3 | — | — |
| Robust / topology-aware ILC | flagged | C3 | flagged |
| ILC max_corr scaled to corridor width | flagged | — | C3 |

## What everyone says NOT to do

- **End-to-end RL** (Swift, On-Your-Own, DreamerV3): requires GPU training infra; the stall is geometric, not perceptual.
- **Soften the in-order DQ**: it correctly encodes the AIGP competition rule.
- **Tighten race_01-only hyperparameters**: per the regression matrix evidence, this pumps the other 6 tracks further into the red.

## Recommended iter-004 plan (synthesised)

### Phase 1 (this iter — small)
1. **Per-track ILC reset** (Opus): non-race_01 tracks start from neutral
   defaults, no carry-over of race_01 corrections. Already mostly in
   place via `ilc_section_overrides` being race_01-specific; verify.
2. **Plan validator (GPT-5.5's idea, brilliant)**: before accepting a
   trajectory, sample it at high resolution, replay through a fresh
   GateSequencer, reject if it would DQ or crash. Uses our existing
   honesty infrastructure as a planning gate.

### Phase 2 (next iter — medium)
3. **SFC corridor constraint in polynomial QP** (Opus + Gemini #1):
   - Build per-segment Frenet tunnel keyed off Catmull-Rom centerline.
   - Linear inequality constraints on polynomial sample points.
   - Tunnel narrows at gates (≈ gate opening half-width − safety margin),
     widens mid-segment (lateral free space).
   - Smooth W(θ), H(θ) via Hermite spline.

### Phase 3 (if SFC isn't enough)
4. **Gate-polygon traversal** (TOGT, Opus C2 + Gemini): force the
   trajectory through the gate's interior polygon, not just past the
   centerline waypoint. Fixes the 2 strut crashes (slalom gate-8,
   aigp_default gate-1).

### Phase 4 (long-term, deferred)
5. **CPC time-allocation** (Opus C3): provable in-order ordering at solve
   time. Requires NLP solver (casadi + IPOPT). Defer unless 1-4 plateau.

## Success criteria for iter-004 (Phase 1 + 2)

Per Opus's matrix-grounded acceptance:
- **≥ 4 of 7 tracks credit gate-1 before any DQ** (vs current: only race_01 does)
- **≥ 3 tracks reach is_complete=True** (currently only race_01)
- **No regression on race_01** (must stay sim_passed=True)

Verified via `scripts/benchmark_matrix.py` against the
`.loop/state/regression_baseline_2026_05_24.json` baseline I captured today.

## Papers / systems shortlist (high-confidence citations)

- **Krinner, Romero, Bauersfeld, Zeilinger, Carron, Scaramuzza. "MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints."** *RSS 2024*, paper 109. <https://www.roboticsproceedings.org/rss20/p109.pdf>
- **Liu, Watterson, Mohta, Sun, Bhattacharya, Taylor, Kumar. "Planning Dynamically Feasible Trajectories for Quadrotors Using Safe Flight Corridors."** *IEEE RA-L* 2(3):1688–1695, 2017.
- **Qin, Michet, Chen, Liu. "Time-Optimal Gate-Traversing Planner."** *ICRA 2024*. Open implementation at `FSC-Lab/TOGT-Planner`.
- **Foehn, Romero, Scaramuzza. "Time-optimal planning for quadrotor waypoint flight" (CPC).** *Science Robotics* 6(56):eabh1221, 2021. Preprint arXiv:2007.06255.
- **Bristow, Tharayil, Alleyne. "A survey of iterative learning control."** *IEEE Control Systems Magazine* 26(3):96–114, 2006. (For per-track ILC reset reasoning.)
