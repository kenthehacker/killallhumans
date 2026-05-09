# Iteration 30 — Inflation Reduction Round 2 (1% per parameter)

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (inflation factors, compression floors)
**Status**: COMMITTED — race time 13.80→13.68s (-0.9%), avg error 0.185→0.191m (+3.2%)
**Commit**: 85a081a

---

## Section 1: Summary
- Iteration 30, timestamp 2026-04-14T17:58Z
- Bottleneck: trajectory_planning — continued inflation reduction from iter 29
- One-line outcome: **Reduced all inflation factors by 1% each (S-turn: 1.09/1.07, TOPP: 0.65/0.59). Race time 13.80→13.68s (-0.9%). The 2% reduction attempted first FAILED due to racing line basin switching; 1% is the maximum safe step size.**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Track-centric Iterative Learning for Global Trajectory Optimization** (arXiv:2601.21027, 2026)
   - Wavelet transform + Bayesian optimization for global trajectory optimization
   - Achieved 20.7% lap time improvement through iterative learning
   - Confirms iterative approach to trajectory optimization

2. **IteraOptiRacing: Unified Planning-Control Framework** (arXiv:2507.09714, 2025)
   - Iterative optimization using historical lap data
   - Unified planner-controller framework for iterative improvement
   - Relevant: historical tracking data guides parameter reduction

3. **Time-Optimal Flight with Safety Constraints** (arXiv:2403.17551, 2024)
   - MPCC with prismatic tunnel safety constraints
   - 100% success rate while pushing to physical limits (80+ km/h)
   - Relevant: safety margin design for time-optimal racing

### Key insight from cross-validation
The planned 2% per-parameter reduction (extrapolated from iter 29's success with mixed 1-3%) FAILED. The racing line selection sits on a basin boundary where ANY 2% TOPP floor change triggers basin switching. The tipping point wasn't about individual parameter magnitude — it's about the cumulative effect on the trajectory optimizer output that feeds into `_select_by_sim()`.

### Research consensus
- **Strong**: Iterative, incremental speed improvement is the right paradigm (Track-centric ILC, SPIRAL, ILMPC)
- **Strong**: Safety margins should be reduced proportionally to tracking improvement (Spatial ILC, COP)
- **New insight**: The basin switching boundary imposes a hard limit on how fast we can reduce inflation (~1% per iteration)

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`**

In `_inflate_sharp_turns()`:
- S-turn junction inflation: 1.10→1.09 (-1%)
- S-turn standard second-gate: 1.08→1.07 (-1%)
- S-turn approach deceleration: 1.02→1.01 (-1%)
- S-turn first-gate departure: 1.03→1.02 (-1%)
- S-turn junction departure: 1.01→1.005 (-0.5%)

In `_topp_retime()`:
- max_compression_protected: 0.66→0.65 (-1%)
- max_compression_easy: 0.60→0.59 (-1%)

### Plan adherence
Partially followed. The plan proposed 2% per parameter. Three failed attempts with 2% revealed:
1. All 8 changes at 2% each → basin switching (26.81s)
2. TOPP 2% + end speed only → basin switching (26.93s)
3. End speed alone (0.65→0.70) → no effect (end speed doesn't bind)

Final successful approach: 1% per parameter, tested incrementally (S-turn first, TOPP second).

### Failed attempts
1. **All changes at 2% per parameter**: S-turn 1.08/1.06 + TOPP 0.64/0.58 + end speed 0.70. Race time 26.81s. Same basin switching as iter 29's aggressive attempt.
2. **TOPP 2% + end speed only**: TOPP 0.64/0.58 + end speed 0.70. Race time 26.93s. Even without S-turn changes, 2% TOPP alone triggers basin switching.
3. **Frozen inflation decoupling**: Added `frozen_inflation=True` flag to decouple racing line evaluation. Still failed — the trajectory ITSELF was untrackable with 2% changes, not just the racing line selection.
4. **End speed 0.65→0.70**: Zero effect. The curvature-based speed limits at the track end are below 0.70*max_v, so the backward pass terminal condition doesn't bind.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before (iter 29) | After (iter 30) | Delta | Direction |
|--------|-------------------|-----------------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.18477m | **0.19087m** | **+0.006m (+3.2%)** | ↑ expected |
| Max tracking error | 0.7248m | 0.7422m | +0.017m (+2.3%) | ↑ mild |
| P50 tracking error | 0.1827m | 0.1900m | +0.007m (+4.0%) | ↑ |
| P95 tracking error | 0.3847m | 0.3865m | +0.002m (+0.5%) | → |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7851 | 7792 | -0.8% | → (still >100) |
| **Race time** | **13.80s** | **13.68s** | **-0.12s (-0.9%)** | **✓ improved** |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.111 | 0.111 | +0.0% | → |
| gate-2 | 0.178 | 0.179 | +0.4% | → |
| gate-3 | 0.236 | 0.247 | +4.7% | ↑ mild |
| gate-4 | 0.186 | 0.202 | +8.4% | ↑ notable |
| gate-5 | 0.167 | 0.180 | +7.6% | ↑ notable |
| gate-6 | 0.158 | 0.162 | +2.7% | → |
| gate-7 | 0.282 | 0.284 | +0.6% | → |
| gate-8 | 0.224 | 0.239 | +6.5% | ↑ |
| gate-9 | 0.194 | 0.206 | +6.0% | ↑ |
| gate-10 | 0.178 | 0.170 | -4.5% | ✓ improved |
| gate-11 | 0.173 | 0.175 | +1.5% | → |
| gate-12 | 0.134 | 0.138 | +2.9% | → |

### Gate pass times
| Gate | Before | After | Savings |
|------|--------|-------|---------|
| gate-2 | 1.89s | 1.88s | -0.01s |
| gate-3 | 2.93s | 2.91s | -0.02s |
| gate-4 | 4.32s | 4.26s | -0.06s |
| gate-5 | 5.50s | 5.42s | -0.08s |
| gate-12 | 13.79s | 13.67s | -0.12s |

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
Inflation reduction is approaching a hard limit. The racing line optimizer's `_select_by_sim` creates an optimization landscape where ANY 2% change to TOPP compression floors causes basin switching (catastrophic regression from ~14s to ~27s race time). Only 1% per parameter per iteration is safe. This means future speed improvements from inflation reduction will be smaller (0.1-0.15s per iteration).

### 5b.2 — Telemetry signals
- max_abs_roll: 0.85 rad (saturation, unchanged)
- max_abs_pitch: 0.85 rad (saturation, unchanged)
- avg_thrust: 0.800 (slight increase from 0.798 — faster trajectory)
- avg_pitch: -0.111 (unchanged)
- No CSV telemetry (kinematic sim only)

### 5b.3 — Code quality
- Parameter-only changes, no algorithmic modifications
- Clean incremental testing (S-turn first, then TOPP)
- Failed `frozen_inflation` approach left as knowledge but reverted

### 5b.4 — Failed approaches analysis
Three key lessons from this iteration's failed attempts:
1. **2% per-parameter is the hard limit**: Even isolating TOPP changes to 2% causes basin switching. The tipping point is absolute parameter values, not individual deltas.
2. **End speed optimization is a dead end**: 0.65→0.70 had zero effect because curvature limits dominate at track end. Would need higher values (0.85+) to have any effect, which is likely unsafe.
3. **Frozen inflation doesn't help**: The issue isn't racing line selection sensitivity — it's the trajectory itself being untrackable at 2% more aggressive parameters. The controller/ILC can't handle the speed.

### 5b.5 — Trend analysis
- **Diminishing returns on inflation reduction**: Iter 29 gained 0.23s (-1.6%), iter 30 gained 0.12s (-0.9%). Next iteration would likely gain <0.08s.
- **Inflation reduction near exhaustion**: 1% per iteration is the safe limit, and remaining headroom is small.
- **Error accumulation**: Avg error trending up: 0.175→0.185→0.191m. Three more iterations at this rate would hit the 0.25m threshold.
- **Speed improvement phase**: Race time improved 14.03→13.80→13.68s over 2 iterations. Total gain: 0.35s.

### 5b.6 — Cumulative inflation since initial calibration
| Parameter | Initial | Current | Total reduction |
|-----------|---------|---------|----------------|
| S-turn junction | ~1.15 | 1.09 | -6% |
| S-turn standard | ~1.12 | 1.07 | -5% |
| approach | ~1.04 | 1.01 | -3% |
| departure pure | ~1.05 | 1.02 | -3% |
| departure junc | ~1.03 | 1.005 | -2.5% |
| protected floor | ~0.72 | 0.65 | +7% compression |
| easy floor | ~0.68 | 0.59 | +9% compression |

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **[trajectory_planning]** Gate-7 helix entry optimization — persistent worst at 0.284m
   - Helix-specific speed profiling or racing line tightening
   - Expected: gate-7 0.284→0.250m, possible race time improvement
   - Priority: 1
   - Research: TOPPQuad (Mao 2024), FBGA (Piazza 2025)

2. **[trajectory_planning]** Further inflation reduction — round 3 (VERY diminishing returns)
   - Another 1% per parameter would gain ~0.08s
   - Expected: race time 13.68→13.60s, avg error 0.191→0.198m
   - Priority: 2 (but approaching exhaustion)
   - Research: ILMPC (arXiv:2508.01103)

3. **[trajectory_planning]** Racing line re-optimization for current parameters
   - Current racing line was optimized with higher inflation. Re-running with current parameters might find better local optima.
   - Expected: 0.05-0.15s race time
   - Priority: 3
   - Research: F1-Init (Shehadeh 2026), Spatially-Aware CMA-ES (Wachter 2026)

4. **[trajectory_planning]** Decouple racing line selection from trajectory optimizer
   - Use simplified kinematic evaluation (no inflation/TOPP) for `_select_by_sim`
   - Would unlock larger parameter changes without basin switching
   - Priority: 4 (architectural fix, high effort)
   - Research: Track-centric ILC (2026), TACO (Sanghvi 2025)

5. **[control]** MPCC controller for contouring/progress decomposition
   - Orthogonal to inflation reduction
   - Expected: tracking improvement across all gates
   - Priority: 5
   - Research: MPCC++ (Krinner 2024)

### Architectural recommendations
- **Racing line basin switching is the hard limit on inflation reduction speed.** Decoupling `_select_by_sim` from the trajectory optimizer would be the most impactful architectural change, but it's a complex refactor.
- The ILC→inflation→speed pipeline is working but approaching natural limits.
- Consider pivoting to gate-7 helix optimization (persistent worst gate for 10+ iterations) for the next iteration.

### Next bottleneck
**trajectory_planning** — gate-7 helix entry optimization. Gate-7 has been the worst gate (0.276→0.282→0.284m) for many iterations and is unaffected by inflation reduction. Addressing it would reduce avg error AND potentially enable further speed improvements.

### What NOT to try
- 2% per-parameter inflation reduction — causes basin switching at current absolute values
- End speed optimization (0.65→0.70) — curvature limits dominate, no effect
- `frozen_inflation` decoupling — doesn't help because the trajectory itself is untrackable, not just the racing line selection
- Aggressive combined reductions (>7 parameters at >1% each)

---

## Section 7: Lessons Learned

### What worked
- **1% per parameter is the safe maximum for this iteration**: Smaller than iter 29's 1-3% because we're deeper into the optimization landscape near a basin boundary.
- **Incremental testing was essential**: Testing S-turn first, then adding TOPP caught the exact point of failure.
- **Per-gate monitoring**: Gate-4 and gate-5 continue as the canary gates for S-turn inflation.

### What didn't work
- **2% per parameter (TOPP) caused basin switching**: Even 2% TOPP floor reduction alone triggered catastrophic 26.8s race time. The basin boundary is sharper than expected at these absolute values.
- **frozen_inflation decoupling**: The issue wasn't racing line selection but trajectory trackability. Fixing selection didn't help.
- **End speed optimization**: max_v*0.70 doesn't bind at track end. Curvature limits at helix exit are the real constraint.

### Surprises
- **2% TOPP alone triggers basin switching**: In iter 29, 3% TOPP (0.63→0.60) was fine. Now 2% TOPP (0.60→0.58) fails. The absolute value matters more than the delta — we're at a cliff edge.
- **Gate-10 improved again (-4.5%)**: Two consecutive iterations where faster trajectory helps gate-10. May be due to better dynamics alignment in the helix exit region.
- **The optimization landscape has a sharp basin boundary around TOPP floor ≈0.58-0.60**: This is the critical observation for future iterations.

### Process improvements
- Test EACH parameter change independently before combining, not just parameter groups
- The 2% threshold from iter 29 was misleading — it depends on absolute parameter values, not just delta percentage
- Consider S-turn and TOPP changes as a single "speed budget" rather than independent axes
