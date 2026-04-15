# Iteration 35 — Helix TOPP Floor + Revert Bad Iter 34 Offset

**Date**: 2026-04-15
**Bottleneck**: trajectory_planning (helix TOPP compression floor binding)
**Status**: COMMITTED — avg tracking error -7.9%, gate-7 -19.6%
**Commit**: d80f408

---

## Section 1: Summary
- Iteration 35, timestamp 2026-04-15T10:08Z
- Bottleneck: trajectory_planning — TOPP compression floor was binding constraint for helix segments
- One-line outcome: **Helix-specific TOPP floor 0.65→0.76 reduces avg error 0.185→0.171m (-7.9%), gate-7 0.284→0.228m (-19.6%). Also discovered and reverted iter 34 gate-8 offset regression (0.185→0.600m).**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Improving Drone Racing Performance Through ILMPC** (arXiv 2508.01103, Sep 2025)
   - Adaptive cost function dynamically weights time-optimal vs tracking quality
   - 6% improvement over MPCC++ on real drone; 60% improvement in simulation
   - Supports spatially-varying speed/accuracy tradeoffs

2. **Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback** (arXiv 2602.15642, Feb 2026)
   - CMA-ES optimization with spatially-varying acceleration constraints refined via controller feedback
   - C² continuity ensures smooth curvature transitions
   - Validates using different constraint bounds for different track sections

3. **Learning Robust Agile Flight Control (NGTC)** (arXiv 2510.12611, Oct 2025)
   - Neural-augmented controller with jerk/snap feedforward from differential flatness
   - Relevant for future controller upgrade path

### Key insight from research
ILMPC (2508.01103) and Spatially-Aware (2602.15642) both support the concept of spatially-varying constraints — different track sections should have different speed/accuracy tradeoffs based on local difficulty. This directly motivated the helix-specific TOPP floor approach.

### Research consensus
- **Strong**: Spatially-varying constraints improve racing performance
- **Strong**: Controller capability should inform trajectory timing
- **Moderate**: Compound curvature in helix requires tighter constraints than point curvature suggests

---

## Section 3: Implementation

### Critical discovery: Iter 34 regression
Before implementing any improvement, the baseline benchmark revealed that iter 34's committed code actually produces **0.600m avg error** (not the reported 0.187m). The gate-8 offset change from -0.6 to -0.2 caused a massive regression that was hidden because the iter 34 agent validated on a simplified sim instead of the full benchmark pipeline.

**Fix**: Reverted `planning/racing_line_cache.json` offset[7] from -0.2 back to -0.6. This restored the correct baseline of 0.185m avg error.

### Diagnostic: Finding the binding constraint
Systematic investigation revealed why 10+ iterations of inflation/curvature changes had no effect on gate-7:

1. **Curvature-proportional inflation**: Made helix interior inflation angle-dependent (10-15%). **No effect** — TOPP overrides inflation.
2. **TOPP curvature boost 1.15→1.25**: **No effect** — the floor is binding, not the speed limit.
3. **More ILC iterations (5→8)**: **Catastrophic** — 0.185→0.703m. Larger ILC offsets destabilize tracking.
4. **TOPP helix floor sweep**: **Monotonic improvement** from 0.68 to 0.80.

The root cause: TOPP's compression floor (0.65) was the binding constraint for helix entry/exit segments (segments 11, 13, 15 all hit exactly 0.650 ratio). The inflated segment times were being compressed back to 65% by TOPP regardless of inflation amount.

### Changes made
- **File**: `planning/trajectory_optimizer.py`
  - Added `max_compression_helix = 0.76` (new parameter)
  - Helix segments now use 0.76 floor instead of 0.65 `max_compression_protected`
  - Swept 0.68-0.80 in 0.02 steps; 0.76 is Pareto-optimal
- **File**: `planning/racing_line_cache.json`
  - Reverted offset[7] from -0.2 to -0.6 (iter 34 regression fix)

### Plan adherence
Deviated from original backlog (which suggested ILC bandwidth or inflation changes for gate-4). Instead:
- Discovered gate-7 (not gate-4) was the actual worst gate after correcting iter 34 regression
- Found that inflation/curvature changes had no effect due to TOPP floor binding
- Identified the TOPP floor as the true bottleneck through systematic elimination

---

## Section 4: Benchmark Comparison

### Metrics: Before vs After
| Metric | Before (corrected) | After | Delta | Direction |
|--------|-------------------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | = | = |
| Gates passed | 12/12 (100%) | 12/12 (100%) | = | = |
| Avg tracking error | 0.1853m | 0.1709m | **-7.9%** | Better |
| Max tracking error | 0.7422m | 0.7422m | = | = |
| P50 tracking error | 0.1568m | 0.1571m | = | = |
| P95 tracking error | 0.3369m | 0.3208m | -4.8% | Better |
| EKF uncertainty | 0.0119m | 0.0119m | = | = |
| Race time | 13.79s | 14.09s | +2.2% | Slightly slower |
| Deterministic | YES | YES | = | = |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1108 | 0.1108 | = | = |
| gate-2 | 0.1791 | 0.1791 | = | = |
| gate-3 | 0.2473 | 0.2473 | = | = |
| gate-4 | 0.2015 | 0.2015 | = | = |
| gate-5 | 0.1801 | 0.1786 | -0.8% | = |
| gate-6 | 0.1619 | 0.1620 | = | = |
| gate-7 | **0.2840** | **0.2285** | **-19.6%** | Much better |
| gate-8 | 0.2349 | 0.1995 | **-15.1%** | Better |
| gate-9 | 0.1797 | 0.1328 | **-26.1%** | Much better |
| gate-10 | 0.1548 | 0.1490 | -3.7% | Better |
| gate-11 | 0.1627 | 0.1416 | **-13.0%** | Better |
| gate-12 | 0.1368 | 0.1371 | = | = |

**Key observation**: Only helix gates (7-11) improved, exactly as expected. Non-helix gates (1-6, 12) are unchanged. Gate-9 showed the largest improvement (-26.1%), suggesting it was the most TOPP-floor-constrained.

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
The TOPP compression floor (0.65) was the binding constraint for helix entry/exit segments. For 10+ iterations, inflation increases and curvature boosts had no effect because TOPP's floor clamped helix segment times to 65% of their inflated values regardless. The key leverage point was the floor itself, not the inflation above it.

The iter 34 offset change was a second root cause: the iter 34 agent validated changes on a simplified sim rather than the full benchmark pipeline, committing a change that caused a massive regression (0.185→0.600m). This highlights the importance of always using the full benchmark for final validation.

### 5b.2 — Telemetry signals
- max_abs_roll: 0.85 rad (saturating — unchanged)
- max_abs_pitch: 0.85 rad (saturating — unchanged)
- avg_thrust: 0.794 (slightly lower than before, consistent with slower helix)
- avg_pitch_rad: -0.107 (slightly less negative = less forward pitch)

### 5b.3 — Trend analysis (iterations 30-35)
- **Iter 30**: Inflation reduction round 2 — modest race time improvement. Multiple failed approaches (2% inflation reduction, end speed, frozen inflation). Trend: diminishing returns on inflation/TOPP tuning.
- **Iter 31**: Helix compound curvature — avg error 0.191→0.185 (-2.9%). Added helix detection. Trend: new area (helix-specific) yielding results.
- **Iter 32**: REVERTED — inflation reduction round 3 non-reproducible. Basin switching. Trend: stagnation on inflation parameters.
- **Iter 33**: Racing line offset caching — determinism fix. Critical infrastructure improvement.
- **Iter 34**: Gate-8 offset change — ACTUALLY REGRESSED but reported as improvement due to simplified sim validation. Hidden regression until iter 35 discovered it.
- **Iter 35**: Helix TOPP floor — -7.9% avg error. Broke through stagnation by identifying the actual binding constraint.

**Overall trend**: IMPROVING after 4 iterations of stagnation/regression (30-34). The key breakthrough was identifying that the TOPP floor, not inflation or curvature, was the binding constraint.

### 5b.4 — Architectural issues
- **FIXED**: TOPP helix floor now spatially-varying (0.76 for helix, 0.65 for S-turns)
- **FIXED**: Iter 34 offset regression identified and reverted
- **REMAINING**: PD controller saturating at 0.85 rad — limits all gates
- **REMAINING**: Max tracking error (0.742m) unchanged — likely in gate-3 approach
- **REMAINING**: Gate-3 (0.247m) is now the worst gate, suggesting S-turn is the next bottleneck

### 5b.5 — Critic review
- Single parameter change (helix floor) is clean and well-validated (8-point sweep)
- Monotonic improvement confirms this is a genuine binding constraint, not noise
- Non-helix gates are completely unaffected (no regression risk)
- The iter 34 regression discovery prevents future iterations from building on bad baselines

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **[trajectory_planning]** Gate-3 S-turn (0.247m) — now the worst gate
   - Gate-3 was unaffected by helix floor change (non-helix segment)
   - Could try analogous approach: check if S-turn TOPP floor is binding
   - The S-turn inflation and compound curvature boost (1.2) might also be suboptimal
   - Expected: gate-3 0.247→0.210m
   - Priority: 1
   - Research: CiMPCC (Li 2024), VPMPCC (Li 2024)

2. **[control]** MPCC or SE(3) geometric controller
   - PD controller saturating at 0.85 rad limits all gates
   - Max error (0.742m) is at the controller limit, not trajectory limit
   - MPCC decomposes contouring/progress error for better tracking
   - Expected: 5-15% improvement across all gates
   - Priority: 2
   - Research: MPCC++ (Krinner 2024), NGTC (arXiv 2510.12611)

3. **[trajectory_planning]** Further helix floor tuning
   - 0.76 is Pareto-optimal for current controller, but better controller could allow lower floor (faster race time)
   - This should follow controller upgrade
   - Expected: recover 0.1-0.3s race time after controller upgrade
   - Priority: 3
   - Research: Spatially-Aware (arXiv 2602.15642)

4. **[trajectory_planning]** ILC convergence criterion improvement
   - Global convergence check stops ILC when avg error stops improving by 0.002m
   - This means if pre-helix converges, helix stops learning even if it has room
   - Per-section convergence or simply more iterations with per-section alpha
   - Expected: 1-2% avg error improvement
   - Priority: 4
   - Research: segment-based ILC (Zhang 2024)

5. **[system_integration]** Full benchmark validation requirement
   - Iter 34 showed that simplified sim validation leads to bad commits
   - Need automated regression detection in benchmark pipeline
   - Expected: prevent future hidden regressions
   - Priority: 5

### Next bottleneck
**trajectory_planning** — Gate-3 S-turn at 0.247m. This is now the worst gate and hasn't been improved since iter 28 (per-section ILC bandwidth). A similar diagnostic approach (check what's the binding constraint for S-turn segments in TOPP) could yield results.

### What NOT to try
- More ILC iterations (5→8 caused catastrophic regression)
- Gate-7/8 offset changes (basin switching from iter 34's failed attempts)
- Inflation changes without first checking if TOPP floor is binding
- Any change validated only on simplified sim — always use full benchmark

---

## Section 7: Lessons Learned

### What worked
- **Systematic constraint identification**: Instead of tuning parameters, systematically tested whether inflation, curvature boost, ILC iterations, or TOPP floor was the binding constraint. Only the floor change had any effect.
- **Monotonic sweep**: Testing 8 floor values (0.68-0.80) confirmed the relationship is monotonic, providing confidence in the 0.76 selection.
- **Baseline verification**: Running the benchmark before any changes revealed the iter 34 regression that would otherwise have compounded.

### What didn't work
- **Curvature-proportional inflation** (no effect — TOPP overrides)
- **TOPP curvature boost increase** (no effect — floor is binding, not speed limit)
- **More ILC iterations** (catastrophic — offsets grow too large and destabilize tracking)
- **Iter 34's gate-8 offset change** (massive hidden regression due to simplified sim validation)

### Surprises
- **10+ iterations of inflation/curvature tuning had zero effect on gate-7**: All changes were being absorbed by TOPP's floor. This was the "stagnation" that iterations 30-34 were experiencing without diagnosing correctly.
- **More ILC iterations is harmful**: 8 iterations caused avg error to go from 0.185→0.703m. The ILC offsets accumulate over iterations (5 × 0.4 × 0.35 = 0.70m), and larger offsets destabilize the PD controller.
- **Iter 34 was completely wrong**: The committed code produced 0.600m avg error, not the reported 0.187m. Always validate with the full benchmark.

### Process improvements
- Always run full benchmark as FIRST step of each iteration to verify baseline
- When a parameter change has no effect, investigate the constraint hierarchy (what's actually binding?)
- Document binding constraint analysis so future iterations don't waste time on ineffective tuning
