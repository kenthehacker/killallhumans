# Iteration 48 — ILC 8 Iterations + Inflection/Post-Inflection Alpha Rebalance

**Date**: 2026-04-15
**Bottleneck**: system_integration (ILC convergence depth + per-section alpha optimization)
**Status**: COMMITTED — avg error 0.173→0.163m (-5.8%), gate-7 0.210→0.164m (-21.9%), gate-5 0.264→0.252m (-4.6%)
**Commit**: 731ab41

---

## Section 1: Summary
- Iteration 48, timestamp 2026-04-15T16:55Z
- Bottleneck: system_integration (ILC convergence depth)
- One-line outcome: **Avg error improved 0.173→0.163m (-5.8%) by increasing ILC iterations 7→8, lowering convergence threshold, reducing inflection alpha 0.50→0.46, and boosting post-inflection alpha 0.40→0.50. All 12 gates improved or flat. Best combined speed+accuracy ever: 13.31s / 0.163m.**

---

## Section 2: Research

### Papers analyzed (3 new, 133 total)
1. **"An ILC Algorithm with a Tuning Parameter for Fastest Convergence Speed"** (Sci China Inf Sci, 2026)
   - Derives optimal parameter using past 2 iterations' data for maximum convergence speed
   - Key insight: 2-iteration lookback could accelerate convergence, but our per-section architecture makes this complex to implement

2. **"MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI"** (arXiv 2601.15222, Jan 2026)
   - Won A2RL 2025 competition at 100 km/h, beating human world champions
   - Key for competition: offline parameter optimization, domain randomization, robust state estimation
   - Actionable: offline parameter sweep methodology confirmed as valid approach

3. **"QPGP-PILC: Quasi-Periodic Gaussian Process Predictive ILC"** (arXiv 2602.18014, Feb 2026)
   - Predicts next iteration's error profile via GP regression for faster convergence
   - Key insight: more iterations = deeper convergence (confirmed by our results)

### Key insight from cross-validation
The convergence_threshold was the gating factor preventing the 8th ILC iteration from running. With 7 iterations at the original threshold (0.002), the ILC had already converged — but convergence at 0.002 resolution left residual error. Lowering to 0.0005 allowed the 8th iteration to run, producing measurable improvements.

### Research consensus vs contradictions
- **Consensus**: More ILC iterations help when not converged. Alpha should decrease with more iterations to prevent over-correction.
- **Our finding**: Contrary to simple theory (alpha ~ 1/√N), the optimal approach is to INCREASE alpha in undersaturated sections (post-inflection) while DECREASING in oversaturated ones (inflection). This is consistent with Liu, Zheng & Chen 2023's section-specific gain theory.

---

## Section 3: Implementation

### Changes made
**File: `scripts/benchmark.py`** (4 parameter changes)

1. **max_iterations 7→8**: One more ILC iteration for deeper convergence. Required lowering convergence_threshold to actually run.

2. **convergence_threshold 0.002→0.0005**: The original threshold stopped ILC after 7 iterations because global avg error improvement was < 0.002m. At 0.0005, the 8th iteration runs and contributes.

3. **Inflection alpha 0.50→0.46**: The inflection section's ILC corrections propagate spatially to gate-5 through drone inertial dynamics. Reducing alpha by 8% reduced this spatial coupling, dramatically improving gate-5 from +10.1% regression (at alpha=0.50) to -4.6% improvement (at alpha=0.46).

4. **Post-inflection alpha 0.40→0.50**: The post-inflection section (gates 5-7) was the most under-converged. Increasing alpha by 25% allowed deeper convergence, producing -21.9% improvement at gate-7 and -10.6% at gate-6.

### Systematic sweep results
| Inflection α | Avg err | Gate-3 | Gate-4 | Gate-5 | Gate-7 |
|-------------|---------|--------|--------|--------|--------|
| 0.50 (8it) | -1.7% | -1.5% | -6.2% | +10.1% | -8.1% |
| 0.48 | -5.6% | +0.0% | -4.2% | -0.4% | -21.7% |
| **0.46** | **-5.8%** | **+0.0%** | **-2.1%** | **-4.6%** | **-21.9%** |
| 0.44 | -6.0% | +1.6% | 0.0% | -8.6% | -21.9% |
| 0.42 | -6.2% | +2.4% | +2.3% | -12.7% | -22.3% |

Selected α=0.46 as Pareto-optimal: best balance with no regressions.

### Plan adherence
Followed plan exactly — started with Config C (max_iterations=8 only), found convergence threshold was limiting, then swept per-section alphas systematically.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit test pass rate | 100% | 100% | 0 | — |
| Avg tracking error | 0.1733m | 0.1632m | -0.0101m | ↓ **-5.8%** |
| Max tracking error | 0.7261m | 0.7394m | +0.013m | ↑ +1.8% |
| P50 tracking error | 0.1381m | 0.1317m | -0.006m | ↓ -4.6% |
| P95 tracking error | 0.4240m | 0.4146m | -0.009m | ↓ -2.2% |
| EKF uncertainty | 0.0119m | 0.0119m | 0 | — |
| Gate pass rate | 100% | 100% | 0 | — |
| Loop frequency | 7508 Hz | 7632 Hz | +124 Hz | — |
| Race time | 13.31s | 13.31s | 0 | — |
| Crashed | No | No | — | — |

### Per-gate error breakdown
| Gate | Before | After | Delta | Note |
|------|--------|-------|-------|------|
| gate-1 | 0.113m | 0.113m | +0.2% | Stable |
| gate-2 | 0.208m | 0.198m | **-4.8%** | Pre-inflection convergence |
| gate-3 | 0.206m | 0.206m | +0.0% | Flat (inflection alpha reduction) |
| gate-4 | 0.241m | 0.236m | **-2.1%** | Inflection convergence |
| gate-5 | 0.264m | 0.252m | **-4.6%** | Post-inflection alpha boost |
| gate-6 | 0.117m | 0.104m | **-10.6%** | Post-inflection convergence |
| gate-7 | 0.210m | 0.164m | **-21.9%** | Biggest individual improvement |
| gate-8 | 0.237m | 0.223m | **-5.9%** | Helix convergence |
| gate-9 | 0.108m | 0.104m | -3.3% | Helix convergence |
| gate-10 | 0.111m | 0.107m | -3.9% | Helix convergence |
| gate-11 | 0.108m | 0.104m | -3.5% | Helix convergence |
| gate-12 | 0.179m | 0.172m | -3.7% | Helix convergence |

### Threshold status
All thresholds passing.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The ILC was still under-converged at 7 iterations due to two factors: (1) the convergence_threshold (0.002) stopped iteration early, preventing the 8th pass from running; (2) the post-inflection section (alpha=0.40) was the most under-converged section, requiring higher learning rate to match the other sections' convergence depth. Additionally, the inflection section's alpha=0.50 was causing excessive spatial coupling to gate-5, meaning the inflection corrections were overshooting and distorting the drone's approach to gate-5.

### Telemetry signals
- Max roll: 0.85 rad (at limit)
- Max pitch: 0.85 rad (at limit)
- Avg thrust: 0.872 (unchanged)
- Controller saturation: both attitude axes still hitting limits

### Trend analysis
**Trend: STRONG IMPROVEMENT.** Three consecutive successful iterations (46→47→48) of ILC tuning, each producing meaningful error reduction:
- Iter 46: -0.37% avg error (alpha boost)
- Iter 47: -8.2% avg error (iteration count 5→7)
- Iter 48: -5.8% avg error (iteration count 7→8, alpha rebalance)

The compound improvement over iterations 46-48 is significant: from the iter 45 error level (~0.186m at 13.31s) to 0.163m at 13.31s — a total ~12% error reduction without any speed loss.

**Diminishing returns**: The ILC is now well-converged at 8 iterations. Further iteration count increases (9, 10) are unlikely to yield >1-2% improvement. The convergence threshold is already near zero effect (0.0005 vs 0.0001 gives identical results). The per-section alphas are at their Pareto-optimal values for the current 4-section layout.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Area: trajectory_planning — Racing line lateral offset re-optimization**
   - Description: The lateral offsets were tuned in iterations 20-25 for a different ILC state (5 iterations, different alphas). With the current 8-iteration ILC, the optimal offsets may have shifted.
   - Approach: Re-sweep lateral offsets at gates 4, 5, 7 with current ILC config.
   - Expected impact: 3-8% avg error reduction if offsets are suboptimal.
   - Priority: 1
   - Research refs: TOGT Planner (Qin 2024), BO Racing Line (Heilmeier 2020)

2. **Area: system_integration — Competition robustness verification**
   - Description: The system is optimized for the deterministic benchmark. Competition will have noise, timing jitter, perception errors.
   - Approach: Add Gaussian noise to state estimates, test with varying gate positions, verify graceful degradation.
   - Expected impact: Competition readiness, identify failure modes.
   - Priority: 2
   - Research refs: MonoRace (2026), On Your Own (Romero 2025)

3. **Area: control — Velocity feedforward re-evaluation with 8-iteration ILC**
   - Description: velocity_feedforward=0.0 in TrackerConfig. Previous tests at 5 ILC iterations showed +9.1% regression. With 8 iterations, the ILC may compensate differently.
   - Approach: Test velocity_feedforward=0.1-0.3 with current config.
   - Expected impact: Possible 2-3% improvement at high-speed gates.
   - Priority: 3
   - Research refs: Kunapuli 2025

4. **Area: system_integration — ILC section boundary micro-tuning**
   - Description: Current boundaries (200, 440, 740) were set in iter 28 and haven't been revisited.
   - Approach: Test ±20 step shifts at each boundary.
   - Expected impact: 1-2% improvement if boundaries don't align with optimal correction regions.
   - Priority: 4

### Architectural recommendations
The ILC-based approach has reached diminishing returns. The next significant improvement likely requires either:
1. Racing line re-optimization (reset lateral offsets for current ILC state)
2. A fundamentally different controller (MPC/MPCC instead of PD) which would unlock new performance regimes
3. Real hardware deployment to validate sim-to-real transfer

### Next bottleneck selected
**trajectory_planning** — Racing line lateral offset re-optimization. The current offsets were tuned for a different ILC convergence state and may be suboptimal for the current 8-iteration configuration. This is the highest-impact change that doesn't require architectural changes.

### What NOT to try
- **ILC max_iterations=9+**: Convergence is essentially complete at 8 iterations. Threshold 0.0005 and 0.0001 give identical results.
- **Inflection alpha >0.50 or <0.44**: Well-characterized Pareto frontier. Alpha=0.46 is the sweet spot.
- **Post-inflection alpha >0.55**: Likely to cause gate-5 regression through the same spatial coupling mechanism.
- **Convergence threshold <0.0005**: Identical results to 0.0005 (already tested with 0.0001).
- **ILC section boundary manipulation**: Tested extensively in iters 28, 46, 47 — all boundary changes cause spatial coupling issues.

---

## Section 7: Lessons Learned

### What worked
- **Convergence threshold reduction was the key enabler**: The ILC was "done" at 7 iterations only because the threshold was too coarse. Lowering it unlocked the 8th iteration.
- **Systematic alpha sweep**: Testing 6 inflection alpha values (0.42-0.53) revealed a clear Pareto frontier and identified the optimal tradeoff point.
- **Counter-intuitive alpha direction**: Increasing post-inflection alpha (0.40→0.50) while decreasing inflection alpha (0.50→0.46) is contrary to the "reduce all alphas with more iterations" heuristic. The right approach is section-specific.

### What didn't work
- **alpha=0.53 post-inflection**: Caused convergence pattern change — the ILC stopped early because the faster-converging post-inflection section reduced global improvement below threshold.

### Surprises
1. **The convergence threshold was the real bottleneck**: Not the iteration count itself. At threshold=0.002, max_iterations=8 produced IDENTICAL results to 7 iterations.
2. **Post-inflection was heavily under-converged**: Alpha=0.40 was far from optimal. The 25% increase to 0.50 produced -21.9% improvement at gate-7 — the largest single-gate improvement in recent iterations.
3. **Inflection-gate-5 spatial coupling has a clean Pareto frontier**: The relationship between inflection alpha and gate-5 regression is smooth and monotonic, making it easy to find the optimal tradeoff.

### Process improvement suggestions
- Always test convergence threshold as a first step when adding iterations. The threshold may be the binding constraint, not the iteration count.
- When sweeping per-section parameters, test them independently rather than all-at-once. The interactions are complex but each section's effect is relatively isolated once the others are fixed.
