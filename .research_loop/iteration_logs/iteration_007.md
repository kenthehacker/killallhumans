# Iteration 7 — Centripetal Acceleration Feasibility Check for S-Turn Gates

**Date**: 2026-04-13
**Bottleneck**: trajectory_planning (gate-3/gate-4 S-turn tracking error 0.661m/0.598m)
**Status**: COMMITTED — gate-3 error 0.661→0.462m (-30%), avg error 0.336→0.285m (-15%)
**Commit**: 050205f

---

## Section 1: Summary
- Iteration 7, timestamp 2026-04-13T17:00
- Bottleneck: trajectory_planning — gate-3/gate-4 S-turn had 0.661m/0.598m tracking error despite being moderate turns (48°/38°), because long approach distances (11.7m/10.5m) allowed high speed into the turns
- Outcome: Centripetal acceleration feasibility check (v²×κ) identifies high-speed moderate turns and inflates segment times. Avg error -15%, max error -23%, gate-3 -30%. Race time regresses +10% (15.28→16.74s).

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **"TOPPQuad: Dynamically-Feasible Time-Optimal Path Parametrization for Quadrotors"** (Mao et al., IROS 2024)
   - URL: https://arxiv.org/abs/2309.11637
   - Time-optimal path parametrization with per-motor thrust constraints
   - Key insight: dynamic feasibility requires checking centripetal acceleration (v²κ) against thrust limits, not just turn geometry
   - 40-50% faster than α-scaled min-snap, hardware validated on CrazyFlie

2. **"Aggressiveness-Aware Learning-based Control of Quadrotor UAVs with Safety Guarantees"** (Colombo et al., 2026)
   - URL: https://arxiv.org/abs/2602.21936
   - GP-based gain scheduling minimizes controller aggressiveness while maintaining tracking
   - Key insight: better disturbance model → lower feedback gains needed → less actuator stress
   - Less directly applicable since our kinematic sim doesn't model gain-dependent behavior

3. **"An Alternating Peak-Optimization Method"** (de Vries et al., ECC 2024) — previously analyzed
   - Confirmed: peak constraint violation ratio (kappa) is the correct per-segment feasibility metric

### Previously analyzed (used)
4. **TACO** (Sanghvi 2025) — section-adaptive trajectory parameters
5. **LMPC** (Zhao et al., IROS 2025) — adaptive cost for per-section aggressiveness
6. **Teissing** (RA-L 2024) — boundary velocity optimization at waypoints
7. **Leveling the Playing Field** (2025) — feedforward is critical for geometric controllers

### Key insight
**Turn angle alone is insufficient for dynamic feasibility — the centripetal acceleration a_c = v²κ captures the combined effect of speed AND curvature.** Gate-3 has only a 48° turn but 8.1 m/s approach speed, giving a_c = 4.69 m/s². Gate-7 (previously fixed) had 69° but only 4.5 m/s, giving a_c = 5.17 m/s². Both exceed the PD controller's tracking bandwidth, but only gate-7 was caught by the angle threshold.

### Critical code finding
**The kinematic sim bypasses the controller's tilt limit.** The benchmark uses `tracker.last_desired_acceleration` (raw PD+feedforward, unclamped by tilt) and only caps total acceleration at 15 m/s². The 0.85 rad "saturation" in controller traces is cosmetic — it doesn't affect drone motion. This means `max_tilt_rad` adjustments have zero effect on benchmark results.

### Research consensus
- Speed×curvature product is the correct metric for trajectory feasibility (TOPPQuad, Teissing, Alternating Peak)
- Post-optimization feasibility correction is preferable to distorting the optimizer objective (confirmed by iter 6 failures)
- Inflation should be proportional to violation magnitude (Alternating Peak: kappa ratio)

---

## Section 3: Implementation

### Changes made
1. **`planning/trajectory_optimizer.py`** — Extended `_inflate_sharp_turns()`:
   - **New centripetal acceleration check**: For gates with turn angles 17°-60° (below the sharp-turn threshold), estimates approach speed from L-BFGS segment times, computes curvature from gate-center geometry, and calculates centripetal acceleration a_c = v²κ
   - **Threshold**: a_centripetal = 3.5 m/s² — tuned through 4 parameter sweeps (5.5, 4.0, 3.5+0.15, 3.5+0.25)
   - **Inflation factor**: 1.0 + 0.25 × severity, where severity = min((a_c - threshold) / threshold, 1.0)
   - **Affected gates**: 2 (a_c=4.78), 3 (a_c=4.69), 4 (a_c=3.64), 5 (a_c=3.69), 12 (a_c=3.98)

### Parameter sweep results
| Threshold | Coeff | Avg Error | Max Error | Gate-3 | Race Time |
|-----------|-------|-----------|-----------|--------|-----------|
| 5.5 | 0.25 | 0.322 | 0.990 | 0.661 | 15.67 |
| 4.0 | 0.25 | 0.296 | 0.822 | 0.523 | 16.45 |
| 3.5 | 0.15 | 0.302 | 0.820 | 0.530 | 16.16 |
| **3.5** | **0.25** | **0.285** | **0.759** | **0.462** | **16.74** |

Selected a=3.5, c=0.25 for strongest tracking improvement while staying under 1.5s race time threshold.

### Plan adherence
Followed the plan closely. The main discovery was that the initial threshold (5.5 m/s²) was too high and missed gates 2-3 entirely. Iterative tuning was needed to find the right threshold. The centripetal acceleration metric worked as expected from the research.

---

## Section 4: Benchmark Comparison

### Full metrics (before → after)
| Metric | Before | After | Delta | Direction | Threshold | Status |
|--------|--------|-------|-------|-----------|-----------|--------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | 0 | = | 100% | PASS |
| Gates passed | 12/12 (100%) | 12/12 (100%) | 0 | = | 100% | PASS |
| Avg tracking error | 0.336m | **0.285m** | -0.051m | ↓↓ | 0.5m | PASS |
| Max tracking error | 0.990m | **0.759m** | -0.231m | ↓↓↓ | 2.0m | PASS |
| P50 tracking error | 0.311m | **0.251m** | -0.060m | ↓↓ | — | — |
| P95 tracking error | 0.819m | **0.682m** | -0.137m | ↓↓ | — | — |
| EKF uncertainty | 0.012m | 0.012m | ~0 | = | 0.5m | PASS |
| Loop Hz | 7753 | 7722 | -31 | = | 100 | PASS |
| Trajectory time | 15.46s | **16.96s** | +1.50s | ↑ | — | — |
| Race time | 15.28s | **16.74s** | +1.46s | ↑↑ | 30s | PASS |

### Per-gate error breakdown
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.108 | 0.113 | +0.005 | Negligible |
| gate-2 | 0.272 | **0.250** | -0.022 | Centripetal inflated (a_c=4.78) |
| gate-3 | **0.661** | **0.462** | **-0.199** | **BEST: S-turn entry, -30%** |
| gate-4 | **0.598** | **0.509** | **-0.089** | Centripetal inflated (a_c=3.64) |
| gate-5 | 0.317 | **0.274** | -0.043 | Centripetal inflated (a_c=3.69) |
| gate-6 | 0.253 | 0.257 | +0.004 | Unchanged (angle-based) |
| gate-7 | 0.447 | 0.447 | 0 | Unchanged (angle-based) |
| gate-8 | 0.285 | 0.285 | 0 | Unchanged |
| gate-9 | 0.211 | **0.175** | -0.036 | Downstream benefit |
| gate-10 | 0.283 | **0.247** | -0.036 | Downstream benefit |
| gate-11 | 0.338 | **0.231** | **-0.107** | Strong downstream benefit |
| gate-12 | 0.209 | **0.167** | -0.042 | Centripetal inflated (a_c=3.98) |

### Threshold status
All thresholds PASS. Avg tracking error has 0.215m headroom to 0.5m threshold. Max tracking error well below 2.0m.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
Gate-3/gate-4 high tracking errors were caused by insufficient time allocation at moderate turns with long approach distances. The PD controller (kp_xy=6, kd_xy=4, max_accel=15 m/s²) cannot redirect the drone fast enough when speed is high. The centripetal acceleration a_c = v²κ correctly captures this: a_c at gate-3 was 4.69 m/s² — exceeding the ~3.5 m/s² threshold where the PD controller begins to overshoot significantly.

### Telemetry signals
- Max abs roll: 0.85 rad (at controller trace limit — cosmetic, doesn't affect sim)
- Max abs pitch: 0.85 rad (cosmetic)
- Avg pitch: -0.106 rad (less aggressive than before due to slower trajectory)
- Avg thrust: 0.739 (lower — less aggressive)
- Controller saturation gates: still 3, 4, 7 in traces (cosmetic, doesn't affect kinematic sim)

### Critical architecture finding
**The kinematic sim's `last_desired_acceleration` bypasses the tilt limit.** Roll/pitch clamping at max_tilt_rad=0.85 only affects the recorded trace, NOT the acceleration applied to the drone. The real constraint is `max_accel=15.0 m/s²` plus `drag=0.5` in the kinematic sim. This means:
- Raising `max_tilt_rad` has ZERO effect on benchmark results
- The "controller saturation" reported in traces is cosmetic
- The actual tracking limit comes from PD bandwidth + kinematic sim acceleration cap

### Trend analysis
**Speed-accuracy tradeoff intensifying**: Each iteration since iter 5 trades race time for tracking quality.

| Iteration | Race Time | Avg Error | Approach |
|-----------|----------|-----------|----------|
| 5 (speed) | 14.73s | 0.398m | Speed optimization |
| 6 (angle) | 15.28s (+3.7%) | 0.336m (-16%) | Sharp turn inflation |
| 7 (centripetal) | 16.74s (+10%) | 0.285m (-15%) | Speed-curvature inflation |

Race time has regressed 14% since iter 5 while accuracy improved 28%. The system is approaching a Pareto frontier where further inflation gains diminish and speed costs grow. Next iteration should focus on recovering race time.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **Recover race time toward 14s** (Priority 1, trajectory_planning)
   - Currently 16.74s, was 14.73s (iter 5). Aspirational target: 14s.
   - Proposed approach: Increase time_weight from 2.0 to 2.5 in L-BFGS. The inflation will still compensate at turns but the optimizer will push harder for speed on straight segments. Also consider reducing kinematic sim drag from 0.5 to 0.3.
   - Expected impact: race time 16.74→15.5s, avg error may increase slightly
   - Research refs: [togt_planner_qin_2024, realtime_mintime_teissing_2024]

2. **S-turn cumulative effect** (Priority 2, trajectory_planning)
   - Gate-4 still at 0.509m — the S-turn cumulative effect (alternating turn directions) isn't captured by single-gate curvature
   - Proposed approach: detect consecutive turns in opposite directions and boost inflation for the second turn. Or add an angular velocity reset term.
   - Expected impact: gate-4 0.509→0.40m
   - Research refs: [taco_sanghvi_2025]

3. **Controller gains optimization** (Priority 3, control)
   - PD gains kp_xy=6, kd_xy=4 limit tracking bandwidth. Higher gains improve turn tracking at cost of potential oscillation.
   - Proposed approach: Bayesian optimization of kp_xy, kd_xy, kp_z, kd_z against benchmark reward. Per Leveling the Playing Field paper.
   - Expected impact: avg error -10% without race time cost
   - Research refs: [leveling_playing_field_2025]

4. **PyBullet physics validation** (Priority 4, system_integration)
   - Kinematic sim doesn't model tilt limits, motor dynamics, or aerodynamics. Results may not transfer.
   - Proposed approach: Install gym-pybullet-drones, enable PyBullet benchmark
   - Expected impact: more realistic benchmarking

5. **Perception pipeline integration** (Priority 5, perception_estimation)
   - Runner still uses raw detections, bypasses EKF/GateTracker/PnP
   - Proposed approach: Integrate estimation modules into sim runner
   - Expected impact: eliminate target jumps in PyBullet sim

### Architectural recommendations
- **The kinematic sim's tilt bypass is a design concern.** Results optimized in this sim may not transfer to hardware where tilt limits are physical constraints. Consider either (a) enforcing tilt limits in the kinematic sim by converting back from attitude to acceleration, or (b) prioritizing PyBullet sim which handles physics correctly.
- **The speed-accuracy Pareto frontier suggests a fundamentally different controller may be needed.** MPC (which can look ahead and preemptively slow before turns) would naturally solve the turn overshoot problem without needing post-hoc time inflation. The infrastructure for this exists in the codebase but isn't used by the benchmark.

### Next bottleneck
`trajectory_planning` — race time recovery (16.74s → 15s target) via increased time_weight.

### What NOT to try
- Don't modify entry/exit offsets (strong coupling with L-BFGS — iter 6 lesson)
- Don't lower centripetal threshold below 3.5 (catches too many gates, diminishing returns)
- Don't raise max_tilt_rad (has no effect on kinematic sim — confirmed this iteration)
- Don't modify transition_time acceleration estimate (iter 5 lesson)

---

## Section 7: Lessons Learned

### What worked
- **Centripetal acceleration as a feasibility metric**: v²κ correctly identifies gates where high speed + moderate curvature exceeds controller capability. This is a direct application of TOPPQuad's core insight.
- **Parameter sweep for threshold tuning**: Testing 4 threshold values (5.5, 4.0, 3.5+0.15, 3.5+0.25) found the best balance. The initial threshold was too conservative and missed the target gates entirely.
- **Post-optimization approach**: Consistent with iter 6's lesson — corrections applied AFTER L-BFGS are more targeted than distorting the optimizer's objective.

### What didn't work
- **Initial threshold (5.5 m/s²) was too high**: Missed gates 2-3 completely. The PD controller's effective bandwidth is lower than expected — it starts overshooting at a_c > 3.5 m/s².
- **Tilt limit analysis was wrong in previous iterations**: The "controller saturation at 0.85 rad" reported in diagnostics was misleading. It's a cosmetic artifact in the kinematic sim, not the real constraint.

### Surprises
- **Tilt limit bypass in kinematic sim**: The sim uses raw desired acceleration from the controller, not the tilt-clamped attitude command. This means all previous analysis of "controller saturation at 0.85 rad tilt" was incorrect as a causal diagnosis — the tilt clamp doesn't affect the simulation.
- **Downstream gate improvement**: Gates 9-12 improved significantly (up to -0.107m at gate-11) even though they weren't directly inflated. The slower overall trajectory gives the PD controller more time everywhere.
- **Gate-4 improvement was modest (-15%)**: Despite centripetal inflation, gate-4 remains at 0.509m. The S-turn cumulative effect (oscillating turn directions across gates 2-3-4) isn't captured by per-gate curvature estimates.

### Process suggestions
- Always verify that reported "saturation" actually affects the simulation dynamics, not just the logged trace
- When tuning a continuous parameter, test at least 3-4 values spanning the expected range
- Check which gates actually trigger the new logic (debug output) before committing
