# Iteration 8 — Activate Feedforward Acceleration (Fix Wiring Bug)

**Date**: 2026-04-13
**Bottleneck**: trajectory_planning (speed-accuracy Pareto tradeoff, race time 16.74s / avg error 0.285m)
**Status**: COMMITTED — avg error 0.285→0.227m (-20%), max error 0.759→0.644m (-15%), race time maintained
**Commit**: a3ddf5b

---

## Section 1: Summary
- Iteration 8, timestamp 2026-04-13T17:12
- Bottleneck: trajectory_planning — the speed-accuracy Pareto tradeoff that dominated iterations 5-7, where improving tracking accuracy required slowing the trajectory
- Outcome: Found and fixed a critical wiring bug where the benchmark zeroed out trajectory acceleration before passing it to the controller, completely nullifying the feedforward capability. With feedforward activated at weight 0.4, avg error improved 20% while race time was maintained — breaking the Pareto tradeoff for the first time.

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **"Accurate Tracking of Aggressive Quadrotor Trajectories Using INDI and Differential Flatness"** (Tal & Karaman, IEEE TCST 2021)
   - URL: https://arxiv.org/abs/1809.04048
   - Tracks at 12.9 m/s with 2.1g and 6.6cm RMS error using jerk/snap feedforward
   - Key: higher-order feedforward (angular velocity, angular acceleration from trajectory derivatives) significantly improves aggressive tracking

2. **"DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control"** (Huang et al., CoRL 2023)
   - URL: https://arxiv.org/abs/2310.09053
   - Feedforward-feedback-adaptive control structure trained via RL
   - 34-36% error reduction over adaptive MPC baselines
   - Confirms feedforward is the primary contributor to steady-state accuracy

3. **"Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag"** (Faessler et al., IEEE RA-L 2018)
   - URL: https://arxiv.org/abs/1705.02480
   - Drag-aware differential flatness for accurate feedforward at high speed
   - Key: unmodeled drag creates systematic tracking error that scales with velocity
   - Directly explains why our full feedforward (1.0) overshoots — drag mismatch

### Previously analyzed (key references)
4. **"Leveling the Playing Field"** (Kunapuli et al., 2025) — "feedforward is the most important single fix for geometric controllers"
5. **"TACO"** (Sanghvi et al., 2025) — trajectory-aware controller optimization

### Key insight
The benchmark had a wiring bug: `trajectory.sample()` returns a full TrajectoryPoint with acceleration from min-snap polynomials, but the benchmark constructed a NEW TrajectoryPoint with acceleration=(0,0,0) before passing it to the controller. This completely nullified the controller's feedforward capability that has existed since iteration 3.

### Research consensus
All papers unanimously agree: feedforward acceleration from the reference trajectory is the foundational requirement for accurate trajectory tracking. Without it, any feedback controller must build position error before generating corrective acceleration.

---

## Section 3: Implementation

### Changes made
1. **`scripts/benchmark.py`** — Fixed the TrajectoryPoint construction:
   - Pass `ref.acceleration`, `ref.jerk`, `ref.yaw_rate` from `trajectory.sample()` to the controller
   - Gate-seeking fallback (after trajectory ends) still uses zero acceleration
   - Set `feedforward_accel=0.4` in TrackerConfig (optimal from sweep)

2. **`control/mpc_tracker.py`** — Updated TrackerConfig default:
   - Changed `feedforward_accel` default from 1.0 to 0.4
   - Added documentation explaining the drag mismatch reason for partial feedforward

### Parameter sweep (feedforward weight)
| ff_weight | Avg Error | Max Error | Race Time | Notes |
|-----------|-----------|-----------|-----------|-------|
| 0.0 (baseline) | 0.285 | 0.759 | 16.74 | Pure PD, no feedforward |
| 0.3 | 0.230 | 0.628 | 16.70 | Good max error |
| **0.4** | **0.227** | **0.644** | **16.69** | **BEST avg error, balanced** |
| 0.5 | 0.229 | 0.732 | 16.67 | Good turn tracking |
| 0.7 | 0.245 | 0.925 | 16.64 | Gate-8 regression |
| 1.0 | 0.299 | 1.225 | 16.60 | Full regression at gates 8-12 |

Full feedforward (1.0) fails because kinematic sim drag (0.5) is unmodeled by controller. At high speed, drag force (0.5 × 10 m/s = 5 m/s²) is significant. The controller's feedforward doesn't compensate for drag, so the PD terms must handle both error correction AND drag compensation. When feedforward is too strong, PD authority is reduced and drag causes overshoot.

### Plan adherence
Followed the plan for the primary fix. Attempted but abandoned time_weight increase (2.0→2.5) because L-BFGS converged to a worse local minimum (trajectory time 21.6s instead of 17s). This is a known sensitivity issue with the optimizer.

---

## Section 4: Benchmark Comparison

### Full metrics (before → after)
| Metric | Before | After | Delta | Direction | Threshold | Status |
|--------|--------|-------|-------|-----------|-----------|--------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | 0 | = | 100% | PASS |
| Gates passed | 12/12 (100%) | 12/12 (100%) | 0 | = | 100% | PASS |
| Avg tracking error | 0.285m | **0.227m** | **-0.058m** | ↓↓↓ | 0.5m | PASS |
| Max tracking error | 0.759m | **0.644m** | **-0.115m** | ↓↓ | 2.0m | PASS |
| P50 tracking error | 0.251m | **0.192m** | -0.059m | ↓↓ | — | — |
| P95 tracking error | 0.682m | **0.543m** | -0.139m | ↓↓ | — | — |
| EKF uncertainty | 0.012m | 0.012m | ~0 | = | 0.5m | PASS |
| Loop Hz | 7536 | 7564 | +28 | = | 100 | PASS |
| Trajectory time | 16.96s | 16.96s | 0 | = | — | — |
| Race time | 16.74s | **16.69s** | -0.05s | = | 30s | PASS |

### Per-gate error breakdown
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.113 | 0.118 | +0.005 | Negligible (start segment) |
| gate-2 | 0.250 | 0.256 | +0.006 | Negligible |
| gate-3 | **0.462** | **0.348** | **-0.114** | S-turn entry — feedforward anticipates centripetal |
| gate-4 | **0.509** | **0.386** | **-0.123** | S-turn exit — cumulative error reduced |
| gate-5 | 0.274 | **0.192** | **-0.082** | Moderate turn benefit |
| gate-6 | 0.257 | **0.164** | **-0.093** | |
| gate-7 | **0.447** | **0.362** | **-0.085** | Helix entry — feedforward at sharp turn |
| gate-8 | 0.285 | **0.263** | -0.022 | Slight improvement |
| gate-9 | 0.175 | **0.155** | -0.020 | |
| gate-10 | 0.247 | **0.202** | -0.045 | |
| gate-11 | 0.231 | **0.194** | -0.037 | |
| gate-12 | 0.167 | **0.112** | **-0.055** | |

### Threshold status
All thresholds PASS. Avg tracking error now has 0.273m headroom to 0.5m threshold. Max error well under aspirational 1.0m target.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The benchmark's sim loop was constructing a controller reference with zero acceleration, completely nullifying the GeometricTracker's feedforward capability. The trajectory optimizer produces correct min-snap acceleration values. The controller is designed to use them. The wiring between them was broken — the benchmark extracted position/velocity/yaw from the trajectory sample but hard-coded acceleration, jerk, and yaw_rate to zero.

This is why the speed-accuracy Pareto tradeoff was so severe in iterations 5-7: the controller could ONLY react to errors (PD feedback), never anticipate them (feedforward). Every increase in trajectory speed immediately translated to larger tracking errors at turns.

### Telemetry signals
- Max abs roll: 0.85 rad (cosmetic tilt limit — doesn't affect kinematic sim)
- Max abs pitch: 0.85 rad (cosmetic)
- Avg pitch: -0.103 rad (similar to before)
- Avg thrust: 0.739 (similar)
- Controller saturation: gates 3, 4, 7 in traces (cosmetic)

### Full feedforward failure analysis
Full feedforward (weight=1.0) caused gate-8 error to spike from 0.285→0.569m because:
1. The kinematic sim has `drag = 0.5 * vel` which acts as a velocity-dependent damping force
2. The controller's feedforward doesn't model drag — it assumes `actual_accel = commanded_accel`
3. At high speed (~10 m/s), drag ≈ 5 m/s² — a significant unmodeled force
4. Strong feedforward reduces the PD controller's effective authority to compensate for drag
5. This creates overshoot at high-speed segments (gates 8-12)

Per Faessler et al. (2018), the correct fix is drag-compensated feedforward: `accel_ff = ref_acc + drag_estimate * vel`. This is a future improvement path.

### Trend analysis
**Pareto frontier broken**: For the first time since iteration 5, accuracy improved without sacrificing race time. The trend shifted from "diminishing returns" to "new improvement axis."

| Iteration | Race Time | Avg Error | Approach | Tradeoff |
|-----------|----------|-----------|----------|----------|
| 5 | 14.73s | 0.398m | Speed optimization | Speed ↑, accuracy ↓ |
| 6 | 15.28s (+4%) | 0.336m (-16%) | Angle inflation | Speed ↓, accuracy ↑ |
| 7 | 16.74s (+10%) | 0.285m (-15%) | Centripetal inflation | Speed ↓, accuracy ↑ |
| **8** | **16.69s (=)** | **0.227m (-20%)** | **Feedforward activation** | **Accuracy ↑, speed =** |

The next improvement axis is clear: recover race time while maintaining the feedforward accuracy gains. With feedforward active, the controller can handle faster trajectories at turns, so inflation can be reduced.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **Recover race time toward 14s target** (Priority 1, trajectory_planning)
   - Currently 16.69s, was 14.73s at iter 5. Target: <15s.
   - Proposed approach: Reduce centripetal inflation threshold from 3.5 to 4.5 (feedforward now handles moderate turns). Or reduce inflation coefficient from 0.25 to 0.15.
   - Expected impact: race time 16.69→15.5s, avg error may increase to ~0.25m
   - Research refs: [togt_planner_qin_2024, realtime_mintime_teissing_2024]

2. **Drag-compensated feedforward** (Priority 2, control)
   - Full feedforward (1.0) failed due to unmodeled drag. Adding drag compensation (`accel_ff = ref_acc + drag*vel`) should enable higher feedforward weight.
   - Expected impact: avg error -10-15% further, enabling more aggressive trajectories
   - Research refs: [flatness_rotor_drag_faessler_2018]

3. **S-turn cumulative effect** (Priority 3, trajectory_planning)
   - Gate-4 still at 0.386m (was 0.509m, improved by feedforward but still highest)
   - Alternating turn directions across gates 2-3-4 create cumulative tracking error
   - Proposed approach: Detect consecutive opposite-direction turns, boost inflation for second turn
   - Research refs: [taco_sanghvi_2025]

4. **Controller gains re-optimization** (Priority 4, control)
   - With feedforward now active, the optimal PD gains may be different
   - Bayesian optimization of kp_xy, kd_xy against benchmark reward
   - Expected impact: avg error -5-10%
   - Research refs: [leveling_playing_field_2025]

5. **PyBullet physics validation** (Priority 5, system_integration)
   - Results optimized for kinematic sim may not transfer to physics sim
   - Install gym-pybullet-drones, run comparison

### Architectural recommendations
- **Drag-compensated feedforward is the cleanest path to full feedforward.** The kinematic sim's drag model is simple (linear, coefficient 0.5), so the compensation is straightforward. This would let us increase feedforward_accel from 0.4 to potentially 0.8-1.0.
- **The trajectory inflation approach (iters 6-7) may become unnecessary** once feedforward + drag compensation are fully active. The controller would be able to handle the aggressive trajectory without needing post-optimization time inflation.

### Next bottleneck
`trajectory_planning` — race time recovery via reduced inflation, leveraging feedforward's improved turn tracking.

### What NOT to try
- Don't increase L-BFGS time_weight beyond 2.0 — causes optimizer instability (tested, trajectory time jumped to 21.6s)
- Don't set feedforward_accel > 0.5 without drag compensation — regression at gates 8-12
- Don't modify entry/exit offsets (iter 6 lesson, still holds)
- Don't modify transition_time acceleration estimate (iter 5 lesson, still holds)

---

## Section 7: Lessons Learned

### What worked
- **Bug hunting in the sim loop**: The wiring bug was hiding in plain sight for 7 iterations. Reading the benchmark code line-by-line — not just the modules — revealed it.
- **Parameter sweep discipline**: Testing 5 feedforward weights (0.3-1.0) found the optimal value and revealed the drag mismatch mechanism.
- **Research-first approach**: The "Leveling the Playing Field" paper's emphasis on feedforward motivated a closer look at whether feedforward was actually active. It wasn't.

### What didn't work
- **Full feedforward (1.0)**: Caused regression due to kinematic sim drag mismatch
- **time_weight increase (2.0→2.5)**: L-BFGS optimizer is sensitive to this parameter and converged to a worse local minimum

### Surprises
- **The feedforward bug was 7 iterations old**: The controller was designed for feedforward from the start, but the benchmark never supplied the acceleration data. All prior iterations of trajectory planning improvements (inflation, centripetal checks) were compensating for a controller that was operating at half capability.
- **Partial feedforward (0.4) works better than full (1.0)**: This is explained by the drag mismatch, but was not obvious a priori. The "Leveling the Playing Field" paper recommends full feedforward, but assumes the dynamics model is accurate. Our kinematic sim's drag creates a systematic model mismatch.
- **Race time slightly improved (16.74→16.69s)**: Feedforward allows the drone to track the trajectory more tightly, so it reaches gates slightly faster despite the same trajectory.

### Process suggestions
- When analyzing a module (like the controller), trace the data flow end-to-end through the sim loop. The module itself may be correct but its inputs may be wrong.
- Always verify that configured features (like `feedforward_accel=1.0`) are actually being exercised by checking the data that flows into them.
