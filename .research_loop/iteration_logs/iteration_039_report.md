# Iteration 39 — S-turn TOPP Floor Re-tuning for New Controller

**Date**: 2026-04-15
**Bottleneck**: control (gate-2 straight→turn overshoot from iter 38 controller upgrade)
**Status**: COMMITTED — avg error 0.151→0.150m (-0.4%), gate-2 -2.1%
**Commit**: b29722e

---

## Section 1: Summary
- Iteration 39, timestamp 2026-04-15T13:37Z
- Bottleneck: control → gate-2 regression from iter 38's gain increase
- One-line outcome: **S-turn TOPP floor 0.67→0.70, avg error -0.4%, gate-2 -2.1%, max error -2.0%**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Newton-Raphson Flow for Aggressive Quadrotor Tracking Control** (arXiv:2408.11197, Aug 2024)
   - Closed-form predictor for lookahead in nonlinear tracking
   - Addresses integrator wind-up causing transient overshoots
   - Key insight: lookahead duration should match system response time

2. **Accurate Tracking via INDI and Differential Flatness** (Tal & Karaman, arXiv:1809.04048, 2018)
   - Jerk/snap feedforward reduces overshoot at turns
   - 50ms lookahead approximates jerk feedforward
   - Less overshoot when higher-order derivatives tracked

### Key findings from research
- Predictive feedforward lookahead timing is less important than controller gain matching
- Our 50ms lookahead has effectively zero impact because min-snap polynomial acceleration changes smoothly over 0-50ms windows
- The critical insight was a CODE finding, not a PAPER finding: the ILC uses stale controller gains

### Research consensus vs contradictions
- Literature: full feedforward (ff=1.0) is standard → our sim constrains to ff=0.50
- Literature: jerk feedforward reduces overshoot → our lookahead provides this but has negligible effect due to smooth trajectories
- NEW finding: ILC-controller gain mismatch creates extreme coupling that masks all controller changes

---

## Section 3: Implementation

### Changes made
1. **File**: `planning/trajectory_optimizer.py` (line 891)
   - `max_compression_sturn`: 0.67 → 0.70
   - Slows S-turn segments to reduce gate-2 tracking error

### Plan adherence
Deviated significantly from initial plan. Original plan targeted ILC gain synchronization (fixing stale gains in ILC inner sim). This was tried first but REGRESSED by +15.1%. Multiple alternative approaches were tested:
- ILC gain sync: +15.1% regression (ILC corrections optimized for old dynamics)
- Lookahead reduction (0.05→0.00-0.04s): zero effect (smooth trajectory acceleration)
- Velocity feedforward (vff=0.1-0.5): zero effect (masked by ILC coupling)
- ILC section parameter changes: basin switching (race time 27.5s)
- TOPP S-turn floor increase: small but consistent improvement

### What actually worked
TOPP floor raise (0.67→0.70) was the only safe lever. It changes the trajectory itself (which the ILC adapts to), avoiding the ILC-controller coupling trap.

---

## Section 4: Benchmark Comparison

### Metrics: Before vs After
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | = | = |
| Gates passed | 12/12 (100%) | 12/12 (100%) | = | = |
| Avg tracking error | 0.1507m | **0.1501m** | **-0.4%** | **Better (ALL-TIME BEST)** |
| Max tracking error | 0.7421m | **0.7269m** | **-2.0%** | **Better** |
| P50 tracking error | 0.1286m | 0.1279m | -0.5% | Better |
| P95 tracking error | 0.3530m | **0.3430m** | **-2.8%** | **Better** |
| EKF uncertainty | 0.0119m | 0.0119m | = | = |
| Race time | 14.01s | 14.07s | +0.4% | Slightly slower |
| Deterministic | YES | YES | = | = |
| Worst gate | gate-2 (0.218m) | gate-2 (0.214m) | -2.1% | Better |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1128 | 0.1129 | +0.0% | = |
| **gate-2** | 0.2183 | **0.2137** | **-2.1%** | **Improved** |
| gate-3 | 0.1860 | 0.1880 | +1.1% | Slightly worse |
| gate-4 | 0.1655 | 0.1699 | +2.7% | Slightly worse |
| gate-5 | 0.1334 | 0.1354 | +1.5% | Slightly worse |
| **gate-6** | 0.0877 | **0.0855** | **-2.5%** | **Improved** |
| gate-7 | 0.1884 | 0.1899 | +0.8% | ≈ |
| **gate-8** | 0.1636 | **0.1541** | **-5.9%** | **Improved** |
| gate-9 | 0.1432 | 0.1437 | +0.4% | ≈ |
| gate-10 | 0.1602 | 0.1621 | +1.2% | Slightly worse |
| **gate-11** | 0.1139 | **0.1117** | **-2.0%** | **Improved** |
| **gate-12** | 0.1471 | **0.1432** | **-2.6%** | **Improved** |

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
The gate-2 regression from iter 38 is fundamentally caused by the TOPP speed profile: the S-turn floor (0.67) was tuned for the old controller (kp=6, kd=4, ff=0.4) but the new controller (kp=7, kd=5.5, ff=0.50) overshoots at the same speed due to higher feedforward amplifying straight→turn transitions. Raising the floor to 0.70 slows the S-turn approach, giving the stronger controller time to settle.

### 5b.2 — Critical discovery: ILC-Controller coupling
The ILC inner sim uses OLD gains (kp=6, kd=4, ff=0.4) while the benchmark uses NEW gains (kp=7, kd=5.5, ff=0.50). Attempts to synchronize them REGRESSED by +15.1%. The ILC corrections are calibrated to work with the mismatched setup. This creates a trap:
- Controller changes that don't propagate into ILC are masked (vff, lookahead)
- ILC changes that don't match the benchmark controller cause regression
- Only trajectory-level changes (TOPP floors, racing line) are safe because ILC adapts to whatever trajectory is given

This ILC-controller coupling is the dominant architectural constraint limiting further improvement.

### 5b.3 — Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_roll | 0.85 rad | 0.85 rad (saturating) |
| max_pitch | 0.85 rad | 0.85 rad (saturating) |
| avg_thrust | 0.836 | 0.837 |
| avg_pitch | -0.097 | -0.096 |

### 5b.4 — Trend analysis
| Iter | Approach | Avg Error | Delta | Trend |
|------|----------|-----------|-------|-------|
| 37 | S-turn TOPP 0.65→0.67 | 0.174m | -1.1% | Diminishing returns |
| 38 | PD gain sweep | 0.151m | -13.5% | BREAKTHROUGH |
| **39** | **S-turn TOPP 0.67→0.70** | **0.150m** | **-0.4%** | **Diminishing returns** |

**Trend: DIMINISHING RETURNS.** After 39 iterations, the system is heavily optimized. TOPP floor, controller gains, and ILC are all near their limits. Incremental tuning yields <1% improvement.

### 5b.5 — Architectural issues
1. **CRITICAL**: ILC-controller gain mismatch creates coupling trap — prevents independent controller optimization
2. **REMAINING**: Controller still saturating at max_tilt=0.85 rad
3. **REMAINING**: Kinematic sim drag model prevents ff>0.52
4. **REMAINING**: No PyBullet validation of accumulated gains
5. **NEW**: ILC convergence code has a bug when max_iterations ≤ 1 (prev_avg_err unset)

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **[system_integration] Fix ILC-controller coupling**
   - The ILC inner sim gains must match the benchmark for future controller changes to have effect
   - But naive synchronization regresses 15.1%
   - Need: re-tune ILC section parameters (alpha, max_correction, cutoff) WITH synchronized gains
   - Expected: unlock controller tuning space, potentially 5-10% improvement
   - Priority: 1
   - Research: Segment-based AILC (Zhang 2026), Dual ILC (Ewering 2025)

2. **[trajectory_planning] Racing line re-optimization with new controller**
   - The racing line cache was generated with old controller. New controller may track tighter lines.
   - Clear the cache and re-run optimization
   - Expected: potentially different basin selection with new dynamics
   - Priority: 2
   - Research: Spatially-Aware CMA-ES (Wachter 2026)

3. **[control] PyBullet validation of accumulated gains**
   - All 39 iterations tuned on kinematic sim. Need to validate on full physics.
   - Priority: 3

4. **[trajectory_planning] Gate-2 specific waypoint offset**
   - Direct approach: slightly move gate-2 entry/exit waypoints to reduce approach speed
   - Risky: gate offset changes caused basin switching in iters 32, 34
   - Priority: 4

5. **[control] Adaptive feedforward blending**
   - Reduce ff near gate-2 transition, increase elsewhere
   - Would require spatially-varying ff parameter
   - Priority: 5
   - Research: TACO (Sanghvi 2025), Spatially-Aware Controller (Wachter 2026)

### Next bottleneck
**system_integration** — The ILC-controller coupling is the dominant constraint. Fixing it would unlock the controller tuning space for future iterations. Without fixing it, all controller improvements are masked.

### What NOT to try
- ILC gain sync alone (regresses +15.1%)
- Lookahead changes (zero effect — smooth trajectory)
- Velocity feedforward alone (masked by ILC coupling)
- ILC section boundary changes (basin switching)
- TOPP floor reduction (basin switching at any decrease)

---

## Section 7: Lessons Learned

### What worked
- TOPP floor adjustment: The one remaining safe lever for improvement
- Systematic sweep: Tested 15+ configurations before finding the best
- Quick pivoting: Abandoned ILC sync approach immediately after regression

### What didn't work
- ILC gain synchronization: Counter-intuitive — "fixing" the mismatch made things worse
- Lookahead changes: Zero effect due to smooth trajectory acceleration
- Velocity feedforward: Completely masked by ILC coupling
- ILC section parameter changes: All caused basin switching

### Surprises
- **ILC coupling is absolute**: The ILC corrections are calibrated to a specific controller configuration. Even small controller changes are masked or cause regression.
- **Lookahead has zero effect**: All values from 0.00 to 0.05 give identical results. The trajectory is too smooth for lookahead timing to matter.
- **max_iterations=0 crashes**: The ILC code has a latent bug where prev_avg_err is unset when no iterations run.

### Process improvements
- The sweep scripts need better file restoration (one left max_iterations=0 in benchmark.py)
- Future iterations should test without ILC first to understand raw controller behavior
- ILC-controller coupling should be documented as a top-level architectural constraint
