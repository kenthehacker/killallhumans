# Iteration 41 — Velocity-Corrected ILC

**Date**: 2026-04-15
**Bottleneck**: system_integration (local optimum breakout)
**Status**: COMMITTED — avg error 0.150→0.140m (-6.7%), max error 0.727→0.701m (-3.6%)
**Commit**: 4386900

---

## Section 1: Summary
- Iteration 41, timestamp 2026-04-15T14:35Z
- Bottleneck: system_integration (system at local optimum after 40 iterations)
- One-line outcome: **Velocity-corrected ILC breaks 5-iteration plateau — avg error -6.7%, helix gates -10 to -26%**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Improving Drone Racing Performance Through Iterative Learning MPC** (arXiv:2508.01103, 2025)
   - ILMPC with adaptive cost design for safe high-speed flight
   - Augments learning MPC with iterative data across trials
   - Relevant: demonstrates that velocity profile co-optimization with position is key

2. **ILC with Mismatch Compensation for Residual Vibration Suppression** (arXiv:2411.07862, 2024)
   - Adaptive mismatch-compensated ILC for Delta robots
   - Convergence proof via Barrier Composite Energy Function
   - Relevant: model mismatch between ILC and execution creates residual error

3. **Quasi-Periodic Gaussian Process Predictive ILC** (arXiv:2602.18014, 2026)
   - GP-based predictive ILC for handling drift across iterations
   - Faster convergence than standard ILC
   - Relevant: predictive component addresses systematic errors

### Existing papers re-examined
- **Track-centric ILC (Nam 2026)**: Co-optimizes position AND velocity profiles → 20.7% improvement
- **Segment-based AILC (Zhang 2026)**: Per-section bandwidth selection, recommends 1.8 Hz for S-turn
- **Schoellig 2012**: ILC should correct feedforward inputs, not just positions
- **Kunapuli 2025**: Feedforward is the single most important fix for geometric controllers

### Key insight from research
**Consensus across 3+ papers**: Position-only ILC leaves velocity/acceleration feedforward uncorrected, creating a systematic mismatch that limits achievable accuracy. The correct approach is to also correct the velocity reference with the time derivative of the position correction.

### Research consensus vs contradictions
- **Consensus**: Feedforward (velocity) correction improves tracking (Schoellig, Kunapuli, Nam)
- **No contradiction found**: All papers agree on the importance of velocity/acceleration feedforward consistency
- **Key gap**: No paper addresses the optimal SCALING of velocity corrections (we found 0.5x empirically)

---

## Section 3: Implementation

### Changes made
1. **`planning/trajectory_optimizer.py`** — `compute_ilc_offset_table()`:
   - Added `cumulative_vel_offset` computation at each ILC iteration via `np.gradient(cumulative_offset, dt, axis=0)`
   - Applied 0.5x-scaled velocity offset in ILC inner sim: `target_vel = ref.velocity + 0.5 * vel_offset[step]`
   - Changed return type from `np.ndarray` to `Tuple[np.ndarray, np.ndarray]` (position + velocity offsets)
   - Final velocity offset computed from converged position offset before return

2. **`scripts/benchmark.py`**:
   - Unpacked both position and velocity offsets from ILC result
   - Applied 0.5x-scaled velocity offset in simulation loop: `target_vel += 0.5 * ilc_vel_offsets[step]`

### Plan adherence
- Followed plan exactly for the position offset + velocity derivative approach
- Deviated from plan by adding 0.5x scaling factor (discovered via sweep: 0.3x, 0.5x, 1.0x tested)
- Original plan didn't specify scaling; empirical testing showed 1.0x causes gate-2 regression >20%

### Scaling sweep results
| Scale | Avg Error | Delta | Gate-2 Delta |
|-------|-----------|-------|-------------|
| 0.0 (baseline) | 0.1501 | 0% | 0% |
| 0.3 | 0.1420 | -5.4% | +7.9% |
| **0.5** | **0.1401** | **-6.7%** | +13.4% |
| 1.0 | 0.1426 | -5.0% | +26.8% |

0.5x was selected as the best overall (maximum avg error reduction with gate-2 within 20% threshold).

---

## Section 4: Benchmark Comparison

### Full metrics table

| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Avg tracking error | 0.1501m | 0.1401m | **-6.7%** | improved |
| Max tracking error | 0.7269m | 0.7011m | **-3.6%** | improved |
| P50 tracking error | 0.1279m | 0.1167m | **-8.7%** | improved |
| P95 tracking error | 0.3430m | 0.3422m | -0.2% | same |
| EKF uncertainty | 0.0119m | 0.0119m | 0% | same |
| Race time | 14.07s | 14.08s | +0.07% | same |
| Gate pass rate | 100% | 100% | 0% | same |
| Loop Hz | 7670 | 7752 | +1.1% | same |
| Crashed | false | false | — | same |
| Unit tests | 9/9 | 9/9 | — | same |

### Per-gate error breakdown

| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1129 | 0.1147 | +1.6% | ~ |
| gate-2 | 0.2137 | 0.2422 | +13.4% | **regressed** |
| gate-3 | 0.1880 | 0.1885 | +0.2% | ~ |
| gate-4 | 0.1699 | 0.1342 | **-21.0%** | improved |
| gate-5 | 0.1354 | 0.1395 | +3.0% | ~ |
| gate-6 | 0.0855 | 0.0734 | **-14.2%** | improved |
| gate-7 | 0.1899 | 0.1695 | **-10.8%** | improved |
| gate-8 | 0.1541 | 0.1318 | **-14.4%** | improved |
| gate-9 | 0.1437 | 0.1072 | **-25.5%** | improved |
| gate-10 | 0.1621 | 0.1491 | **-8.0%** | improved |
| gate-11 | 0.1117 | 0.1153 | +3.3% | ~ |
| gate-12 | 0.1432 | 0.1408 | -1.7% | ~ |

### Threshold status
ALL thresholds passing. No threshold failures.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The ILC was computing position-only corrections while leaving velocity references unchanged from the original trajectory. This created a systematic position-velocity mismatch: the controller was told to be at a SHIFTED position but move at the ORIGINAL velocity. The PD controller's velocity error term (`kd * (ref.velocity - actual_vel)`) was partially fighting the ILC position correction. Adding the smooth time derivative of the position offset as a velocity correction resolves this mismatch.

### Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_abs_roll_rad | 0.85 | 0.85 |
| max_abs_pitch_rad | 0.85 | 0.85 |
| avg_pitch_rad | -0.096 | -0.096 |
| avg_thrust | 0.837 | 0.837 |

Controller outputs unchanged — the velocity correction doesn't change the control authority, only the reference tracking consistency.

### Gate-2 regression analysis
Gate-2 regression (+13.4%) is caused by the velocity correction interacting with the ILC gain mismatch. The ILC inner sim uses kp=6, kd=4, ff=0.4 while the benchmark uses kp=7, kd=5.5, ff=0.50. The velocity correction amplifies this mismatch in the first 2 seconds (gate-2 at 1.92s). The 0.5x scaling mitigates but doesn't eliminate this. A potential fix: use 0.3x or 0.0x scaling specifically in the pre-inflection section (steps 0-200).

### Trend analysis
- **Plateau broken**: After 5 iterations (36-40) of zero improvement, this iteration achieved -6.7% avg error
- **Architectural change was key**: Parameter tuning was exhausted; the velocity correction is a structural change
- **The velocity correction is orthogonal to previous tuning**: It doesn't change gains, racing line, ILC parameters, or section boundaries. It adds a new dimension of correction
- **Diminishing returns risk**: Further velocity correction scaling optimization has limited upside (~1-2%)

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Per-section velocity correction scaling** (priority 1)
   - Area: system_integration
   - Problem: Gate-2 regresses 13.4% due to uniform 0.5x velocity scaling
   - Approach: Use 0.0-0.3x scaling for pre-inflection section (0-200 steps) and 0.5-0.7x for helix
   - Expected impact: Recover gate-2 error while maintaining helix gains
   - Research: Bristow & Alleyne 2007 (time-varying gains), Zhang 2026 (segment-specific parameters)

2. **Acceleration correction from ILC offsets** (priority 2)
   - Area: system_integration
   - Problem: Feedforward acceleration is still from original trajectory, not corrected path
   - Approach: Compute second derivative of ILC offset, add to feedforward acceleration
   - Expected impact: Further 3-5% avg error reduction; higher risk of noise
   - Research: Schoellig 2012 (full feedforward correction), Nam 2026 (velocity+accel profiles)

3. **Full ILC re-derivation with velocity corrections** (priority 3)
   - Area: system_integration
   - Problem: ILC inner sim still uses mismatched gains (kp=6 vs kp=7). With velocity corrections, the mismatch has different effects than before
   - Approach: Synchronize ILC inner sim gains to match benchmark AND use velocity corrections. Previous sync attempt (iter 39) failed without velocity corrections — may succeed with them
   - Expected impact: Could further improve convergence if mismatch was masking residual errors
   - Research: Wu 2024 (mismatch compensation), Schoellig 2012

4. **PyBullet full-physics validation** (priority 4)
   - Area: system_integration
   - Problem: All optimizations are in kinematic sim; unvalidated in full physics
   - Approach: Run benchmark with PyBullet sim to validate accumulated improvements
   - Expected impact: Discover real-world viability of 41 iterations of kinematic-sim tuning
   - Research: NGTC (Pries 2025), LoL-NMPC (Gupta 2025)

### Architectural recommendations
- The velocity correction approach opens a new optimization dimension. Future iterations should explore:
  - Per-section velocity scaling (different for S-turn vs helix)
  - Acceleration corrections (second derivative of offset)
  - Full ILC gain re-synchronization (now feasible with velocity corrections)
- The ILC inner sim gain mismatch (kp=6 vs kp=7) should be re-examined — velocity corrections may change whether the mismatch helps or hurts

### Next bottleneck selected
**system_integration** — per-section velocity correction scaling to recover gate-2

### What NOT to try
- Full velocity scaling (1.0x) — causes gate-2 regression >26%
- Execution-time-only velocity corrections (no change to ILC inner sim) — causes +4.8% avg error regression
- Further parameter tuning of racing line, TOPP floor, or controller gains — exhausted in iterations 36-40

---

## Section 7: Lessons Learned

### What worked
- **Identifying the position-velocity mismatch**: A fundamental flaw that was invisible during parameter tuning
- **Sweeping the scaling factor**: 0.3x, 0.5x, 1.0x revealed the trade-off between avg error and gate-2 regression
- **Applying velocity corrections in BOTH ILC inner sim and benchmark**: Execution-time-only corrections were ineffective because ILC learned corrections assuming no velocity correction

### What didn't work
- **Execution-time-only velocity correction**: The ILC corrections are calibrated for position-only execution. Adding velocity corrections only at runtime creates a NEW mismatch
- **Full (1.0x) velocity scaling**: Too aggressive for gate-2; the pre-inflection section is sensitive to velocity perturbations

### Surprises
- The velocity correction gave a **6.7% improvement** on a system declared at a local optimum — architectural changes still work even when parameter tuning is exhausted
- Gate-9 improved by **25.5%** — the largest single-gate improvement in recent memory
- The 0.5x scaling was optimal, not 1.0x — partial correction is better than full correction due to the ILC gain mismatch amplification

### Suggestions for improving the iteration process
- When stuck at a local optimum, focus on structural mismatches in the algorithm rather than parameter tuning
- The position-velocity mismatch was present since iteration 1 but never identified because it was masked by larger errors. As the system improves, previously-invisible issues become the dominant error source
- Per-section scaling should be the DEFAULT for future changes — uniform scaling creates trade-offs that section-specific scaling can eliminate
