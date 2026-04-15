# Iteration 47 — ILC Iteration Count Increase + Per-Section Alpha Rebalance

**Date**: 2026-04-15
**Bottleneck**: trajectory_planning (L-BFGS multi-start) → pivoted to system_integration (ILC convergence depth)
**Status**: COMMITTED — avg error 0.195→0.179m (-8.2%), gate-4 0.330→0.292m (-11.6%), gate-7 0.303→0.254m (-16.2%)
**Commit**: f3e3330

---

## Section 1: Summary
- Iteration 47, timestamp 2026-04-15T16:35Z
- Bottleneck: trajectory_planning (planned: L-BFGS multi-start), pivoted to system_integration (ILC tuning)
- One-line outcome: **Avg error improved 0.195→0.179m (-8.2%) by increasing ILC iterations 5→7 and rebalancing per-section learning rates. Gate-4 -11.6%, gate-7 -16.2%, gate-8 -30.1%.**

---

## Section 2: Research

### Papers analyzed (3 new, 130 total)
1. **"cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation"** (Sundaralingam et al., NVIDIA) — arXiv:2310.17274
   - 12-seed parallel L-BFGS achieves 99.8% success rate vs 98.5% single-start. Key insight: single-start local optimization is inherently suboptimal in non-convex landscapes. Noisy line search with small fallback steps (α=0.01) escapes shallow local minima.

2. **"MISO: Learning Multiple Initial Solutions to Optimization Problems"** (Sharony et al.) — arXiv:2411.02158
   - 5-14× improvement from multi-start across DDP, MPPI, iLQR optimizers. Winner-Takes-All loss promotes diversity. Key insight: diversity in initialization matters more than quantity. Always include baseline as a candidate (no-regression guarantee).

3. **"AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization"** (ICRA 2026) — arXiv:2509.17340
   - 15 anchor-guided MPPI instances achieve 2-3× velocity improvement. Structured seeds cover distinct basins in non-convex landscapes.

### Key insight from cross-validation
All three papers agree on multi-start superiority, but our experiments revealed that the L-BFGS landscape for our specific track has ONE dominant trackable basin. Alternative basins found by multi-start score better by L-BFGS objective but produce worse tracking when passed through the full pipeline (TOPP retiming → inflation → ILC → PD controller). The proxy objective (time_weight * total_time + penalties) doesn't model controller dynamics.

### Research consensus vs contradictions
- **Consensus**: Multi-start L-BFGS with diverse initializations outperforms single-start in non-convex landscapes.
- **Contradiction with our results**: Despite strong theoretical support, multi-start produced zero improvement for our track. The L-BFGS cost landscape has poor correlation with downstream tracking performance. This is consistent with the "sim-to-real gap" concept — our L-BFGS objective IS the "sim" and the kinematic benchmark IS the "real".

---

## Section 3: Implementation

### Changes made

**File: `scripts/benchmark.py`** (3 lines changed)

1. **ILC max_iterations 5→7**: The core improvement. More ILC iterations allow the position corrections to converge closer to the true steady-state offset pattern. With 5 iterations, the corrections were not fully converged, leaving residual error at gates 3, 4, 7, 8. Research basis: Longman 2023 (arXiv:2307.15912) showed that model-based warmstarting accelerates convergence, validating that additional iterations produce meaningful improvement.

2. **Pre-inflection alpha 0.4→0.30**: With 7 ILC iterations, the original alpha=0.4 for the pre-inflection section caused gate-2 over-correction (+23%). Reducing to 0.30 keeps gate-2 regression at +1.7% while preserving the benefits of deeper convergence in other sections.

3. **Helix alpha 0.4→0.45**: The helix section benefits from a slightly higher learning rate at 7 iterations, improving gate-7 from -14.6% to -16.2%. This exploits the additional convergence cycles.

### Plan adherence
The original plan was L-BFGS multi-start optimization. After 3 failed attempts (L-BFGS cost selection, sim-based selection, diverse seeds), the approach was abandoned. Pivoted to ILC convergence depth tuning, which was not in the original plan but addresses the same goal (reducing tracking error at worst gates) through a different mechanism.

### Failed approaches in this iteration
1. **L-BFGS multi-start with L-BFGS cost selection** (4 seeds: baseline, aggressive ×0.85, conservative ×1.15, graduated warm-start). Result: catastrophic regression — avg error +23.6%, gate-3 +87%, gate-4 +84%. Root cause: L-BFGS objective doesn't model PD controller, TOPP retimer, inflation, or ILC.

2. **L-BFGS multi-start with sim-based selection** (same 4 seeds + quick kinematic sim scoring). Result: sim correctly rejected non-baseline candidates, producing zero improvement. The current L-BFGS basin IS the best trackable one.

3. **L-BFGS multi-start with additional seeds** (gate-4-boosted +20% time, aggressive -10% all). Result: same — sim-based selection chose baseline every time.

4. **Post-inflection vel_scale 0.5→0.35**: Regression — avg error +5.4%, gate-7 +25%.

5. **Post-inflection vel_scale 0.5→0.45**: Regression — avg error +4.8%, gate-4 +15%.

6. **Inflection max_correction 0.15→0.20m**: Mixed — gate-3 -9.1% but gate-5 +7.9%, gate-7 +20%.

7. **Extended inflection region to step 480**: Gate-4 -1.8% but gate-5 +5.1%, gate-7 +6.6%.

8. **5-section with transition zone**: Regression — avg error +1.2%, gate-5 +5.6%.

9. **Smoothing sigma 10→12 + blend_steps 50→70**: Zero effect.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit test pass rate | 100% | 100% | 0 | — |
| Avg tracking error | 0.1946m | 0.1786m | -0.016m | ↓ -8.2% |
| Max tracking error | 0.6894m | 0.5474m | -0.142m | ↓ -20.6% |
| P95 tracking error | 0.4903m | 0.4139m | -0.076m | ↓ -15.6% |
| EKF uncertainty | 0.0119m | 0.0119m | 0 | — |
| Gate pass rate | 100% | 100% | 0 | — |
| Loop frequency | 6480 Hz | 6430 Hz | -50 Hz | ↓ (negligible) |
| Race time | 13.65s | 13.66s | +0.01s | — |
| Crashed | No | No | — | — |

### Per-gate error breakdown
| Gate | Before | After | Delta | Note |
|------|--------|-------|-------|------|
| gate-1 | 0.112m | 0.113m | +0.4% | Stable |
| gate-2 | 0.138m | 0.140m | +1.7% | Controlled by alpha reduction |
| gate-3 | 0.263m | 0.230m | **-12.3%** | Inflection region benefit |
| gate-4 | 0.330m | 0.292m | **-11.6%** | Major improvement |
| gate-5 | 0.214m | 0.241m | +12.7% | Tradeoff (within limits) |
| gate-6 | 0.171m | 0.171m | -0.5% | Stable |
| gate-7 | 0.303m | 0.254m | **-16.2%** | Helix alpha boost benefit |
| gate-8 | 0.193m | 0.135m | **-30.1%** | Biggest individual improvement |
| gate-9 | 0.110m | 0.128m | +16.3% | Near 20% threshold |
| gate-10 | 0.153m | 0.137m | -9.2% | Good improvement |
| gate-11 | 0.126m | 0.109m | -8.8% | Good improvement |
| gate-12 | 0.188m | 0.166m | -8.5% | Good improvement |

### Threshold status
All thresholds passing. All met since iteration ~30.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The previous ILC configuration (5 iterations) was under-converged. ILC is an iterative algorithm that progressively reduces tracking error; 5 iterations left significant residual error at the most difficult gates (3, 4, 7, 8). The per-section alpha parameters were tuned for 5 iterations and needed rebalancing for 7 iterations to prevent over-correction in low-error sections (pre-inflection gate-2).

### Telemetry signals
- No recent simulation logs (last CSV from 2026-04-03 — 12 days old, pre-iteration 42)
- Max roll: 0.85 rad (at limit)
- Max pitch: 0.85 rad (at limit)
- Controller saturation: both attitude axes hitting limits, indicating aggressive trajectory near physical limits

### Trend analysis
| Iteration | Focus | Avg Error | Race Time | Direction |
|-----------|-------|-----------|-----------|-----------|
| 42 | ILC per-section vel | 0.138m | 14.08s | ↓ small error improvement |
| 43 | ILC helix cap | 0.138m | 14.08s | → stagnation |
| 44 | Inflation reduction | 0.159m | 13.51s | ↑ error for speed |
| 45 | L-BFGS time_weight | 0.186m | 13.31s | ↑ error for speed |
| 46 | ILC alpha boost | 0.186m* | 13.31s* | ↓ small recovery |
| 47 | ILC iterations+alpha | 0.179m | 13.66s | ↓↓ strong recovery |

*Note: baseline metrics shifted between sessions (0.186→0.195m, 13.31→13.65s). Possible scipy/numpy version sensitivity or L-BFGS landscape bifurcation. The shift is NOT caused by code changes (clean git status, same commit).

**Trend: Improving.** After two speed-push iterations (44-45) that traded tracking for speed, iterations 46-47 have been recovering tracking quality through ILC tuning. The ILC iteration count increase in iteration 47 produced the largest single-iteration tracking improvement since iteration 40.

**Diminishing returns warning**: ILC tuning has been the focus for 6 of the last 7 iterations. While iteration 47 produced strong results, the remaining improvement potential from ILC parameter tuning is likely small. The next significant gains will require architectural changes.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Area: system_integration — ILC max_iterations fine-tuning**
   - Description: Test 8 iterations (vs current 7) with further alpha rebalancing. Gate-9 regression (+16.3%) suggests the helix section might benefit from lower alpha with more iterations.
   - Approach: Sweep alpha=[0.40, 0.42, 0.43] for helix section with max_iterations=8.
   - Expected impact: Gate-9 regression reduction, possible further avg error improvement of 1-3%.
   - Priority: 2
   - Research refs: Longman 2023 (arXiv:2307.15912)

2. **Area: trajectory_planning — Racing line lateral offset re-optimization**
   - Description: With the metric baseline shift (new baseline ~0.179m), the racing line lateral offsets (set in iterations 20-25) may no longer be optimal. The offset pattern was tuned for a different ILC convergence state.
   - Approach: Re-sweep lateral offsets at key gates (especially gates 4, 5, 7, 9) with current ILC config.
   - Expected impact: 5-10% avg error reduction if offsets are suboptimal for current trajectory.
   - Priority: 2
   - Research refs: TOGT Planner (Qin 2024)

3. **Area: control — Velocity feedforward activation**
   - Description: `velocity_feedforward=0.0` in TrackerConfig. Enabling this could improve trajectory following during high-speed sections.
   - Approach: Sweep velocity_feedforward from 0.1 to 0.5 with current ILC config.
   - Expected impact: Reduced tracking error at high-speed gates (9-12), possible 3-5% improvement.
   - Priority: 3
   - Research refs: Kunapuli 2025 (Leveling the Playing Field)

4. **Area: system_integration — ILC section boundary optimization**
   - Description: Current inflection region (2.0-4.4s) doesn't cover gate-4 (at 4.57s). Extending the boundary to 4.6s with careful alpha tuning could capture gate-4 in the aggressive ILC section.
   - Approach: Extend inflection_end to 460 with alpha=0.45 (not 0.50) and test carefully for gate-5 regression.
   - Expected impact: Gate-4 reduction by 5-10%, but high risk of gate-5 regression.
   - Priority: 3
   - Research refs: CDC 2024 (constrained ILC)

5. **Area: trajectory_planning — Trajectory inflation factor micro-tuning**
   - Description: Turn inflation was halved in iteration 44. Finer tuning (e.g., 0.45× or 0.55× instead of 0.50×) might find a better speed-tracking tradeoff.
   - Approach: Sweep turn inflation factor from 0.40 to 0.60 in steps of 0.05.
   - Expected impact: 1-2% avg error or 0.1-0.2s race time improvement.
   - Priority: 4

### Architectural recommendations
None at this time. The current ILC + kinematic sim architecture is producing consistent improvements. The main limitation is that the kinematic sim (PD controller with drag model) doesn't match real drone dynamics, so all optimization is fundamentally limited by sim fidelity.

### Next bottleneck selected
**system_integration** — Continue ILC parameter optimization with focus on max_iterations=8 sweep and per-section alpha rebalancing. If this produces <2% improvement, pivot to trajectory_planning (racing line re-optimization).

### What NOT to try
- **L-BFGS multi-start** (any variant): Proven to produce zero improvement. The L-BFGS landscape has one dominant trackable basin for this track. Cost-based selection finds untrackable basins; sim-based selection always picks baseline. 3 variants attempted, all failed.
- **Post-inflection vel_scale reduction**: Reducing from 0.5 causes race time and tracking regression at downstream gates.
- **Inflection max_correction >0.15m**: Causes 20%+ gate-5 regression from spatial coupling (tested in iterations 46 and 47).
- **Inflection region extension >440 steps**: Similar spatial coupling issue as max_correction increase.
- **5-section layouts with transition zones**: Short sections have edge effects from Butterworth filtering.
- **Smoothing sigma / blend_steps changes**: Had zero effect — the corrections are already smooth enough.

---

## Section 7: Lessons Learned

### What worked
- **Increasing ILC iteration count**: The single most impactful change. 5→7 iterations produced -8.2% avg error — larger than any previous ILC tuning change. This was not in the original plan and was discovered after the L-BFGS multi-start approach failed.
- **Per-section alpha rebalancing**: Reducing pre-inflection alpha to 0.30 prevented gate-2 over-correction while preserving benefits elsewhere. This is a general principle: when increasing ILC iterations, reduce alpha in sections that were already well-converged.
- **Helix alpha increase**: Synergistic with higher iteration count — the helix section benefits from both more iterations and higher learning rate.

### What didn't work
- **L-BFGS multi-start** (primary plan): All 3 variants failed. Strong theoretical backing from 3 papers, but the fundamental issue is that our L-BFGS objective is a poor proxy for downstream tracking quality. The "sim-to-real gap" exists within our own pipeline.
- **ILC section boundary manipulation**: All attempts to extend or split sections failed due to spatial coupling between ILC corrections and downstream gate errors.
- **ILC smoothing parameter changes**: Had zero measurable effect.

### Surprises
1. **Baseline metric shift**: The same committed code (iteration 46) now produces 13.65s/0.195m instead of 13.31s/0.186m. This means previous iteration reports may have measured with different library versions or system state. All comparisons in this iteration use the current baseline.
2. **ILC iterations had never been increased**: 5 iterations was the default from early iterations. Nobody had tested whether more iterations would help. This was a "low-hanging fruit" that was hiding in plain sight.
3. **ILC inter-section coupling**: Changing the helix section alpha (which only affects steps >740) caused changes in gate-2/3/4 errors. This means ILC iterations are globally coupled — each iteration's corrections change the error signal for subsequent iterations in ALL sections.

### Process improvement suggestions
- When planning changes based on research papers, always have a fallback approach in case the primary plan fails. The ILC iteration increase was a fallback that turned out to be the winner.
- Track whether ILC has converged by comparing iteration-N error vs iteration-(N-1) error. If still decreasing, more iterations will help.
