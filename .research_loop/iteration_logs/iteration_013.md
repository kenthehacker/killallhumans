# Iteration 13 — Racing Line Optimization + Proximity-Based Helix Inflation

**Date**: 2026-04-14
**Bottleneck**: system_integration (trajectory non-determinism) → PIVOT → trajectory_planning (helix tracking)
**Status**: COMMITTED — avg error 0.358→0.179m (-50%), race time 13.31→17.70s (+33%)
**Commit**: 14c3958

---

## Section 1: Summary
- Iteration 13, timestamp 2026-04-14T01:06Z
- Original bottleneck (trajectory non-determinism) was disproved — optimizer is deterministic (5/5 cross-process)
- Pivoted to trajectory planning: racing line optimization for helix
- Outcome: **avg tracking error halved** (0.358→0.179m), helix tracking reduced 74-82%, race time +33%

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **AROLA: Modular Architecture for Scaled Autonomous Racing** (arXiv:2602.02730, 2026)
   - Standardized 8-layer architecture with Race Monitor benchmarking framework
   - Key for us: consistency score metric, controller A/B testing methodology
   - Not directly actionable for this iteration's bottleneck

2. **Improving Drone Racing via Iterative Learning MPC** (arXiv:2508.01103, Zhao et al., 2025)
   - 6.05% improvement even on top of MPCC++ (state-of-the-art)
   - Key insight: **trajectory quality > controller complexity**
   - Spatially-varying cost near gates, arc-length parameterization
   - Confirms iteration 12's finding that controller tuning is secondary

3. **Rethinking Reference Trajectories via MPPI** (arXiv:2509.14726, Zhao et al., 2025)
   - Gate progress objective eliminates reference trajectory entirely
   - Competitive with trajectory tracking on simple tracks
   - Requires GPU (0.4ms desktop, 6.7ms embedded) — future direction

### Key insight from research
All three papers plus iteration 12 converge on the same conclusion: **trajectory quality has more impact than controller sophistication**. The ILMPC paper explicitly demonstrates this by showing 6% improvement from trajectory refinement on top of MPCC++ — a controller far more advanced than our PD tracker.

### Research consensus
- Gates are regions, not points (TOGT, ILMPC) → optimize pass-through positions
- Trajectory quality > controller tuning (ILMPC, MPPI) → invest in planning
- Spatially-varying precision near gates (ILMPC) → not all track sections are equal

### Critical finding: Non-determinism debunked
Ran 5 separate Python processes with identical inputs — all produced the same trajectory time (13.5914142702s). The non-determinism reported in iteration 12 was likely caused by incomplete code revert during that session's extensive experimentation (40+ configurations tested).

---

## Section 3: Implementation

### Changes made

**File 1: `planning/racing_line.py`**
- `max_lateral_offset`: 0.4 → 0.6
  - The racing line optimizer was hitting its maximum offset bounds at 0.339m (√(0.24² + 0.24²)) for most gates
  - Increasing to 0.6 allows 0.36m per-axis offset, leaving 0.24m margin to gate edge
  - Research: TOGT (Qin 2024) treats gates as regions; Swift (Kaufmann 2023) learned aggressive corner-cutting
- `smoothness_weight`: 0.3 → 0.40
  - With offset=0.6, smooth≥0.35 steers the L-BFGS into a qualitatively different (smoother) local minimum
  - This is the critical change — a bifurcation in the optimization landscape
  - smooth=0.40 balances helix tracking (0.172m) and S-turn accuracy (0.422m)

**File 2: `planning/trajectory_optimizer.py`**
- Added proximity-based inflation in `_inflate_sharp_turns()`:
  - For consecutive gates within 6m with turn angle > 23°, apply up to 25% inflation
  - Specifically targets the helix where gates are 3.6-4.9m apart
  - Compounds with existing angle-based and centripetal inflation

### Plan adherence
Partially followed the plan. The smoothness_weight required extensive tuning due to a discovered L-BFGS bifurcation:
- smooth=0.3 → 12.78s trajectory (fast but gate-7: 0.971m, WORSE)
- smooth=0.35 → 17.42s (gate-3: 0.884m regression)
- smooth=0.40 → 17.88s (best balance: gate-3: 0.422m, gate-7: 0.172m)
- smooth=0.50 → 17.40s (gate-3: 0.385m, gate-7: 0.158m — similar quality)

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.358m | **0.179m** | **-50.0%** | ↑↑↑ |
| Max tracking error | 1.463m | **0.697m** | **-52.4%** | ↑↑↑ |
| P50 tracking error | 0.276m | 0.132m | -52.2% | ↑↑↑ |
| P95 tracking error | 0.864m | 0.541m | -37.4% | ↑↑ |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7886 | 7939 | +0.7% | → |
| Trajectory time | 13.59s | 17.88s | +31.6% | ↓↓ |
| Race time | 13.31s | **17.70s** | **+33.0%** | ↓↓ |
| Worst gate | gate-7 (0.659m) | gate-3 (0.422m) | -36.0% | ↑ |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.119 | 0.127 | +0.008 | → same |
| gate-2 | 0.268 | 0.266 | -0.002 | → same |
| gate-3 | 0.402 | **0.422** | +0.020 | slight regression (now worst) |
| gate-4 | 0.388 | 0.319 | -0.069 | ↑ improved |
| gate-5 | 0.191 | 0.184 | -0.007 | → same |
| gate-6 | 0.196 | 0.178 | -0.018 | → same |
| gate-7 | **0.659** | **0.172** | **-0.487** | ↑↑↑ MASSIVE improvement |
| gate-8 | 0.528 | 0.096 | -0.432 | ↑↑↑ MASSIVE improvement |
| gate-9 | 0.407 | 0.093 | -0.314 | ↑↑↑ MASSIVE improvement |
| gate-10 | 0.404 | 0.098 | -0.306 | ↑↑↑ MASSIVE improvement |
| gate-11 | 0.385 | 0.098 | -0.287 | ↑↑↑ MASSIVE improvement |
| gate-12 | 0.344 | 0.117 | -0.227 | ↑↑ big improvement |

### Threshold status
| Threshold | Required | Current | Target | Status |
|-----------|----------|---------|--------|--------|
| Avg error | <0.5m | 0.179m | <0.25m | **MEETS ASPIRATIONAL** |
| Max error | <2.0m | 0.697m | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | 17.70s | <14s | PASS (misses aspirational) |
| Loop Hz | >100 | 7939 | >100 | PASS |
| No crash | required | no crash | — | PASS |

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The racing line optimizer's L-BFGS was hitting its maximum offset bounds (0.339m) at most gates, particularly constraining corner-cutting through the helix. The offset bound (max_lateral_offset=0.4 × half_width=0.6 = 0.24m per axis) was too tight for the optimizer to find smooth paths through closely-spaced helix gates (3.6-4.9m apart).

Increasing the offset range to 0.6 AND slightly increasing smoothness weight triggered a **bifurcation** in the L-BFGS landscape. The optimizer transitions from a "fast" basin (12.78s, high curvature, poor tracking) to a "smooth" basin (17.4s, low curvature, excellent tracking) at smooth≈0.35. The smooth basin produces qualitatively different racing lines that are dramatically more trackable.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (tilt limit, unchanged)
- Avg thrust: 0.747 (decreased from 0.779 — less aggressive maneuvers)
- Avg pitch: -0.074 (shallower than -0.135 — less forward lean = lower speed)
- The controller is no longer saturating as much in the helix

### Trend analysis
**Trend: IMPROVING (breakthrough)**

Iteration progression:
- Iter 8-10: trajectory planning improvements (feedforward, inflation, FOV)
- Iter 11: predictive FF — avg 0.481→0.358m (controller)
- Iter 12: gain scheduling exhausted in kinematic sim (controller ceiling)
- **Iter 13: racing line optimization — avg 0.358→0.179m (planning breakthrough)**

The shift from controller tuning (diminishing returns) to trajectory planning (breakthrough improvement) was the right strategic decision. All three new papers confirmed this direction.

### Racing line L-BFGS bifurcation
A critical discovery: the racing line optimizer has at least two distinct local minima when max_lateral_offset=0.6:
- **Fast basin** (smooth<0.35): 12.78s trajectory, high curvature, gate-7: 0.971m
- **Smooth basin** (smooth≥0.35): 17.4s trajectory, low curvature, gate-7: 0.172m

This bifurcation means small parameter changes can cause large performance swings. Future iterations should validate which basin the optimizer converges to after any racing_line.py change.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **S-turn gate-3/4 treatment** (Priority 1, trajectory_planning)
   - Gate-3 at 0.422m is now the worst gate — alternating-direction S-turn
   - Approach: detect consecutive opposite-direction turns and boost inflation for the second turn
   - Expected: gate-3 error 0.42→0.25m
   - Research: TACO (Sanghvi 2025), Alternating Peak (Foehn 2024)

2. **Race time recovery** (Priority 2, trajectory_planning)
   - 13.31→17.70s is a 33% regression — need to recover speed
   - Approach: binary search for minimum smoothness_weight maintaining smooth basin; or speed up straight segments while keeping helix smooth
   - Expected: 17.70→15.0s while maintaining tracking
   - Research: Teissing (RA-L 2024) norm constraints

3. **Norm-based acceleration constraints** (Priority 3, trajectory_planning)
   - Sphere vs box constraints give >20% faster trajectories
   - Research: Teissing (RA-L 2024)

4. **KAIST-style heading FOV control** (Priority 4, trajectory_planning)
   - Decouple yaw from position for zero-cost FOV

5. **Racing line L-BFGS basin stability** (Priority 5, system_integration)
   - Guard against accidentally flipping basins

### Architectural recommendations
- The racing line optimizer's L-BFGS sensitivity suggests it should be replaced with a more robust optimizer (e.g., CMA-ES, multi-start) for the racing line problem
- Consider caching the racing line result and only re-optimizing when gate configuration changes

### What NOT to try
- **Controller tuning in kinematic sim** — exhaustively proven infeasible in iter 12
- **smoothness_weight < 0.35 with offset=0.6** — flips to fast basin with poor tracking
- **Reducing proximity inflation** — helix needs the additional time budget
- **Velocity feedforward** — transient-dominated sections can't use it (iter 11)

---

## Section 7: Lessons Learned

### What worked
- **Pivoting from the stated bottleneck**: The state.json said "trajectory non-determinism" but empirical testing proved it was a non-issue. Pivoting to the actual bottleneck (helix tracking) yielded a breakthrough.
- **Sweeping the parameter space**: Testing 7 smoothness_weight values revealed the L-BFGS bifurcation that would have been missed with fewer tests.
- **Embracing the speed-accuracy tradeoff**: Accepting a 33% race time increase to achieve 50% tracking improvement was the right call given the competition targets.

### What didn't work
- **Racing line offset=0.6 with smooth=0.3**: The fast basin produces untrackable trajectories
- **Racing line offset=0.5**: Unexpectedly also falls into a slow basin (17.48s) — the landscape is complex
- **Proximity inflation alone** (without racing line change): Only 5% improvement vs 50% with full change

### Surprises
- **L-BFGS bifurcation in the racing line**: A tiny smoothness_weight change (0.32→0.35) causes a 4.6s jump in trajectory time — a qualitative phase transition in the optimization landscape
- **Trajectory non-determinism was a false alarm**: 5/5 cross-process runs identical. The iteration 12 observation was likely due to in-session code state.
- **The helix section became the BEST part of the track**: From worst (0.35-0.66m) to best (0.09-0.17m) — a complete reversal.

### Process suggestions
- When state.json identifies a bottleneck, empirically verify it before investing research time
- Parameter sweeps across optimizer settings are essential — the landscape has discrete basins
- The aspirational targets (<0.25m avg, <1.0m max) are now achievable through trajectory planning alone — controller improvements may not be needed
