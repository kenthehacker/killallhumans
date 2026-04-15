# Iteration 43 — Helix ILC Correction Cap Increase

**Date**: 2026-04-15
**Bottleneck**: system_integration (acceleration correction → helix correction cap)
**Status**: COMMITTED — avg error 0.1384→0.1381m (-0.26%), gate-7 -2.6%
**Commit**: e7c09f3

---

## Section 1: Summary
- Iteration 43, timestamp 2026-04-15T15:10Z
- Bottleneck: system_integration (ILC acceleration feedforward correction)
- One-line outcome: **Accel correction FAILED across 6 configs; pivoted to helix correction cap increase (0.35→0.45m) for -0.26% avg error, gate-7 -2.6%**

---

## Section 2: Research

### Papers analyzed (3 new, 115→118 total)
1. **Learning Robust Agile Flight Control with Stability Guarantees** (arXiv:2510.12611, 2025)
   - Neural-augmented controller with jerk feedforward
   - Key finding: jerk feedforward important but snap (4th derivative) doesn't improve tracking in practice
   - Relevant: confirms that higher-order derivative feedforward has diminishing returns
2. **Optimizing Control-Friendly Trajectories with Self-Supervised Residual Learning** (arXiv:2601.02738, 2026)
   - Self-supervised residual learning for trajectory optimization
   - Key finding: trajectory corrections should maintain smoothness; noise in corrections degrades tracking
3. **ILC with DOB for Mismatched Dynamics** (arXiv:2404.10231, 2024)
   - Framework for ILC under dynamics mismatch
   - Key finding: disturbance observer compensates for mismatch but requires careful filter design

### Key insight from research
The "Robust Agile Flight Control" paper found that snap feedforward (position's 4th derivative) does NOT improve tracking beyond jerk feedforward, even in real hardware. This aligns with our finding: the second derivative of ILC position offsets (analogous to acceleration/jerk feedforward) adds noise without meaningful improvement because the ILC position offsets already compensate for the acceleration mismatch through cross-track corrections.

### Research consensus vs contradictions
- **Consensus**: Higher-order derivative feedforward has diminishing returns (Tal 2018 showed jerk helps but snap is marginal; our experiment confirms accel correction is unhelpful)
- **Contradiction with Schoellig 2012**: Schoellig's full feedforward correction assumes synchronized dynamics. Our system has an intentional gain mismatch (kp=6 ILC vs kp=7 benchmark) that makes the ILC offsets implicitly optimal for the mismatched system.

---

## Section 3: Implementation

### Changes made
1. **Primary attempt (FAILED)**: ILC acceleration feedforward correction
   - Added `cumulative_accel_offset` computation (Butterworth-filtered second derivative of position offsets)
   - Added per-section acceleration scaling (7th element of section_boundaries)
   - Tested 6 configurations:
     - Config A: All sections (0.0/0.2/0.3/0.5), ILC inner sim + benchmark: +1.2% regression
     - Config B: Helix-only 0.3, ILC inner sim + benchmark, 1.0 Hz cutoff: +0.77%
     - Config C: Helix-only 0.3, ILC inner sim + benchmark, 0.5 Hz cutoff: +0.72%
     - Config D: Helix-only 0.3, execution-time only: +0.48%
     - Config E: Helix-only 0.1, ff_ratio=0.8: +0.13%
     - Config F: Helix-only 0.1, post-tracker direct: +0.13%
   - ALL configurations regressed. Reverted.

2. **Pivot: Helix max_correction cap increase**
   - `scripts/benchmark.py`: Changed helix section max_correction_m from 0.35 to 0.45
   - Swept: 0.35 (baseline), 0.40, 0.45, 0.55
   - Also tested increasing inflection (0.15→0.20) and post-inflection (0.15→0.20) caps — both regressed
   - **0.45m is optimal**: gate-7 -2.6%, avg -0.26%

### Plan adherence
- Deviated from original plan (acceleration correction) because it failed across all configurations
- Pivoted to a simpler approach (correction cap increase) that succeeded

---

## Section 4: Benchmark Comparison

### Full metrics table

| Metric | Before (iter 42) | After | Delta | Direction |
|--------|-------------------|-------|-------|-----------|
| Avg tracking error | 0.1384m | 0.1381m | **-0.26%** | improved |
| Max tracking error | 0.7269m | 0.7269m | 0% | same |
| P50 tracking error | 0.1134m | 0.1135m | +0.1% | same |
| P95 tracking error | 0.3323m | 0.3309m | -0.4% | improved |
| EKF uncertainty | 0.0119m | 0.0119m | 0% | same |
| Race time | 14.08s | 14.08s | 0% | same |
| Gate pass rate | 100% | 100% | 0% | same |
| Loop Hz | 7509 | 7426 | -1.1% | same |
| Crashed | false | false | — | same |
| Unit tests | 9/9 | 9/9 | — | same |

### Per-gate error breakdown

| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1129 | 0.1129 | 0.0% | same |
| gate-2 | 0.2137 | 0.2137 | 0.0% | same |
| gate-3 | 0.1907 | 0.1907 | 0.0% | same |
| gate-4 | 0.1422 | 0.1422 | 0.0% | same |
| gate-5 | 0.1392 | 0.1392 | 0.0% | same |
| gate-6 | 0.0736 | 0.0736 | 0.0% | same |
| gate-7 | 0.1687 | **0.1643** | **-2.6%** | improved |
| gate-8 | 0.1289 | **0.1279** | **-0.8%** | improved |
| gate-9 | 0.1075 | 0.1089 | +1.3% | mild regr |
| gate-10 | 0.1442 | 0.1441 | -0.1% | same |
| gate-11 | 0.1160 | 0.1158 | -0.2% | same |
| gate-12 | 0.1410 | 0.1411 | +0.1% | same |

### Threshold status
ALL thresholds passing. No threshold failures.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
Gate-7's ILC offsets were saturating at the 0.35m cap, limiting the correction. The helix approach (gates 7-8) requires larger position offsets because the PD controller consistently undercuts the turns. Increasing the cap to 0.45m allows the ILC to apply its full computed correction at gate-7, reducing error by 2.6%. Other sections (pre-inflection, inflection, post-inflection) are not cap-limited and increasing their caps hurts downstream gates.

### Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_abs_roll_rad | 0.85 | 0.85 |
| max_abs_pitch_rad | 0.85 | 0.85 |
| avg_pitch_rad | -0.096 | -0.096 |
| avg_thrust | 0.837 | 0.838 |

Controller outputs unchanged — the correction cap change only affects ILC offset magnitude, not control authority.

### Why acceleration correction failed
The ILC position offsets already compensate for acceleration mismatch through cross-track corrections. Adding d²Δp/dt² to the feedforward is redundant because:
1. The ILC learns offsets that account for the full tracking dynamics (including acceleration feedforward)
2. The gain mismatch (kp=6 ILC, kp=7 benchmark) creates an implicit "beneficial mismatch" where ILC corrections are calibrated for the mismatched system
3. The second derivative amplifies noise even with Butterworth filtering
4. Any change to the feedforward signal invalidates the ILC's calibration

This is consistent with the "Robust Agile Flight Control" paper (2510.12611) which found snap feedforward doesn't improve tracking beyond jerk feedforward.

### Trend analysis
- **Diminishing returns**: Avg error improvements across last 6 iterations: -13.4%, FAIL, -1.0%, -6.7%, -1.2%, -0.26%. The system is approaching a local optimum.
- **Correction cap was the last "free" parameter**: Most other ILC parameters (alpha, cutoff, section boundaries, velocity scaling) have been exhaustively swept.
- **Architectural ceiling**: The kinematic sim PD controller has fundamental limitations (no attitude dynamics, simplified drag model). Further improvements likely require either:
  a. PyBullet validation + full-physics tuning
  b. Speed recovery (pushing race time from 14.08s toward 13s)
  c. Fundamentally different trajectory optimization approach

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Speed recovery via segment-selective compression** (priority 1)
   - Area: trajectory_planning
   - Problem: Race time stuck at 14.08s. Current headroom: avg error 0.138m vs 0.25m threshold = 0.112m headroom
   - Approach: Identify which segments have the most tracking headroom and selectively compress their timing. Unlike uniform compression (failed iter 14), this targets only segments where error is well below threshold.
   - Expected impact: 0.3-0.5s race time reduction (14.08→13.6-13.8s) with avg error staying under 0.20m
   - Research refs: CPC (Foehn 2021), TOPP speed profiling (Mao 2024)

2. **PyBullet full-physics validation** (priority 2)
   - Area: system_integration
   - Problem: 43 iterations of kinematic sim tuning unvalidated in full physics
   - Approach: Run benchmark with PyBullet to discover sim-to-real transfer gap
   - Expected impact: Discover whether kinematic sim improvements transfer to full physics
   - Research refs: NGTC (Pries 2025), LoL-NMPC (Gupta 2025)

3. **Pre-inflection section ILC alpha increase** (priority 3)
   - Area: system_integration
   - Problem: Pre-inflection (gate-1/2) uses alpha=0.4 with vel_scale=0.0. Gate-2 is the worst gate at 0.214m
   - Approach: Try alpha=0.5 for pre-inflection only (more aggressive position correction without velocity correction may reduce gate-2 error)
   - Expected impact: Gate-2 reduction ~5-10%
   - Research refs: Schoellig 2012 (alpha tuning)

4. **Section boundary fine-tuning** (priority 4)
   - Area: system_integration
   - Problem: Section boundaries at fixed time values (2.0/4.4/7.4s) may not be optimal
   - Approach: Shift helix boundary earlier (7.0s) to give gate-7 approach more aggressive correction
   - Expected impact: Small gate-7 improvement
   - Research refs: Zhang 2024 (segment-wise ILC)

### Architectural recommendations
- The kinematic sim + PD controller is at its optimization ceiling. Future major improvements require:
  1. Full-physics (PyBullet) validation to understand the transfer gap
  2. Speed/accuracy trade-off exploration (the system has 0.112m of headroom to the 0.25m threshold)
  3. Consider switching to time-competitive mode rather than accuracy-competitive mode

### Next bottleneck selected
**trajectory_planning** — speed recovery via segment-selective compression. The system has significant headroom between current avg error (0.138m) and the threshold (0.25m), suggesting we can trade some accuracy for speed.

### What NOT to try
- ILC acceleration feedforward correction — exhaustively tested in 6 configurations, all regressed
- ILC gain synchronization — failed in iter 39, the "beneficial mismatch" is load-bearing
- Inflation/TOPP floor changes > 1% — triggers racing line basin switching (iters 29-32)
- Uniform time compression — binding constraint on turn segments (iter 14)
- Post-inflection/inflection correction cap increase — hurts gate-5 (tested this iteration)

---

## Section 7: Lessons Learned

### What worked
- **Helix correction cap increase**: Simple, targeted, effective. The ILC offsets were cap-limited at gate-7.
- **Systematic sweep of configurations**: Testing 6 acceleration correction configs before reverting provided clear evidence of failure
- **Quick pivot**: After detecting failure of primary approach, pivoting to correction cap optimization was efficient

### What didn't work
- **ILC acceleration feedforward correction**: All 6 configurations regressed. The ILC position offsets already compensate for acceleration mismatch. Adding acceleration correction is redundant and harmful.
- **Non-helix correction cap increases**: Inflection and post-inflection sections at 0.20m regressed gate-5 (7.8% regression with inflection at 0.20m)

### Surprises
- **Acceleration correction is universally harmful**: Even at 0.1 scaling with ff_ratio compensation, the correction makes things worse. This contradicts the theoretical expectation from Schoellig 2012.
- **The "beneficial mismatch" is real**: The ILC inner sim's lower gains (kp=6, kd=4, ff=0.4) vs benchmark's higher gains (kp=7, kd=5.5, ff=0.50) creates a calibration that works precisely BECAUSE of the mismatch. Any attempt to add consistency (acceleration correction, gain sync) disrupts this calibration.
- **Cap limitation was hidden**: The 0.35m helix cap had been unchanged since early iterations and was quietly limiting improvement. A simple diagnostic (checking offset magnitudes) would have caught this earlier.

### Suggestions for improving the iteration process
- Add a diagnostic that checks whether ILC offsets are hitting their per-section caps
- When an approach fails across multiple configs, document the failure thoroughly and pivot quickly
- The system may benefit from a "headroom analysis" tool that identifies which gates have room for speed increase vs which are at their accuracy limits
