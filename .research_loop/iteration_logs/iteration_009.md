# Iteration 9 — Reduce Post-Optimization Inflation (Feedforward-Enabled Speed Recovery)

**Date**: 2026-04-13
**Bottleneck**: trajectory_planning (recover race time leveraging feedforward activation from iter 8)
**Status**: COMMITTED — race time 21.56→21.05s (-2.4%), inflation reduced, drag compensation tested and rejected
**Commit**: 2f18ce1

---

## Section 1: Summary
- Iteration 9, timestamp 2026-04-13T17:30
- Bottleneck: trajectory_planning — with feedforward active (iter 8), the controller handles turns better, so the conservative inflation added in iters 6-7 can be reduced
- Outcome: Three inflation parameters reduced. Race time improves modestly (-0.51s). Drag compensation tested extensively but rejected — sim drag provides beneficial velocity damping.
- **Metrics discrepancy note**: The commit message reports 16.69→15.56s based on measurements taken during the implementation session. However, re-running the benchmark now consistently produces ~21s race times due to the FOV relaxation step (`_relax_for_fov`) adding ~6s. This suggests the original measurements were taken in a transient code state. All metrics in this report use verified, reproducible benchmark results.

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **"Why Change Your Controller When You Can Change Your Planner: Drag-Aware Trajectory Generation"** (Zhang et al., L4DC 2024)
   - URL: https://arxiv.org/abs/2401.04960
   - Core insight: plan trajectories that account for drag instead of modifying the controller
   - 83% position tracking error reduction via learned tracking penalty
   - Our system: drag is known (linear, coefficient 0.5), so analytical approaches preferred over learned penalty
   - Key takeaway: drag-aware planning enables faster trajectories that stay within controller's tracking envelope

2. **"Real-Time Minimum-Time Trajectory Planning with Norm Constraints"** (Teissing et al., IEEE RA-L 2024)
   - Norm-based acceleration constraint (sphere) vs per-axis (box) gives >20% faster trajectories
   - LTD algorithm runs in real-time on embedded hardware
   - Our optimizer uses per-axis velocity/acceleration constraints — switching to norm-based is a future improvement
   - Expected impact: 10-20% faster trajectories without accuracy loss

3. **"ℒ₁Quad: ℒ₁ Adaptive Augmentation of Geometric Control for Agile Quadrotors"** (Wu et al., IEEE TCST 2025)
   - ℒ₁ adaptive augmentation of SE(3) geometric controller for drag compensation
   - 5x smaller tracking error than baseline geometric controller
   - Over-engineering for our known drag model — ℒ₁ designed for UNKNOWN disturbances
   - Confirmed that drag compensation should be exact for known models

### Previously analyzed (key references)
4. **"Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag"** (Faessler et al., IEEE RA-L 2018) — drag-compensated feedforward
5. **"Leveling the Playing Field"** (2025) — feedforward is critical
6. **TOPPQuad** (Mao et al., IROS 2024) — centripetal acceleration feasibility

### Key insight
**Drag compensation in the controller regresses tracking because the sim's drag provides beneficial velocity damping.** The effective error damping is `kd + drag_coefficient`. With drag compensation active (cancelling the 0.5 drag), damping drops from `kd + 0.5 = 4.5` to `kd = 4.0`, reducing the damping ratio from ~0.92 to ~0.82. Even increasing kd to 5.5 doesn't recover the same behavior because drag also acts as a natural speed limiter into turns.

---

## Section 3: Implementation

### Changes made
1. **`planning/trajectory_optimizer.py`** — Three inflation parameter reductions:
   - `a_centripetal_threshold`: 3.5 → 4.5 m/s² (feedforward handles moderate turns)
   - Angle-based inflation coefficient: 0.35 → 0.25 (60°→1.25x, 90°→1.50x instead of 1.60x)
   - Centripetal inflation coefficient: 0.25 → 0.15 (range: 1.0x to 1.15x instead of 1.25x)

2. **`control/mpc_tracker.py`** — Added drag compensation infrastructure (currently disabled):
   - Added `drag_coefficient: float = 0.0` to TrackerConfig
   - Drag compensation term `+ c.drag_coefficient * vel[i]` in `track()` method
   - Set to 0.0 because drag compensation regresses tracking (see failed approaches)

3. **`scripts/benchmark.py`** — Comment update for feedforward_accel

### Failed experiments within this iteration
1. **Drag compensation (Faessler 2018)**: 8 configurations tested (drag_coeff × ff_weight × kd_xy):
   - drag=0.5, ff=0.8, kd=4.0: avg error 0.358m (+79% vs baseline)
   - drag=0.5, ff=0.8, kd=5.0: avg error 0.345m (+73%)
   - drag=0.5, ff=0.8, kd=5.5: avg error 0.372m (+86%)
   - drag=0.3, ff=0.6, kd=4.0: avg error 0.353m (+77%)
   - All configurations regressed. Root cause: sim drag = beneficial velocity damping.

---

## Section 4: Benchmark Comparison

### Full metrics (before → after, verified)
| Metric | Before | After | Delta | Direction | Threshold | Status |
|--------|--------|-------|-------|-----------|-----------|--------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | 0 | = | 100% | PASS |
| Gates passed | 12/12 (100%) | 12/12 (100%) | 0 | = | 100% | PASS |
| Avg tracking error | 0.200m | 0.209m | +0.009m | ↑ | 0.5m | PASS |
| Max tracking error | 1.342m | 1.284m | -0.058m | ↓ | 2.0m | PASS |
| P50 tracking error | 0.081m | 0.087m | +0.006m | ↑ | — | — |
| P95 tracking error | 0.875m | 0.948m | +0.073m | ↑ | — | — |
| EKF uncertainty | 0.012m | 0.012m | ~0 | = | 0.5m | PASS |
| Loop Hz | ~6866 | ~6866 | ~0 | = | 100 | PASS |
| Trajectory time | 21.70s | 21.18s | -0.52s | ↓ | — | — |
| Race time | 21.56s | 21.05s | -0.51s | ↓ | 30s | PASS |

### Per-gate error breakdown
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.124 | 0.130 | +0.006 | Negligible |
| gate-2 | 0.131 | 0.135 | +0.004 | Negligible |
| gate-3 | 0.721 | **0.771** | +0.050 | S-turn entry — reduced inflation |
| gate-4 | 0.404 | **0.432** | +0.028 | S-turn exit — reduced centripetal inflation |
| gate-5 | 0.208 | **0.230** | +0.022 | Moderate turn |
| gate-6 | 0.228 | **0.248** | +0.020 | Reduced angle inflation |
| gate-7 | 0.106 | **0.141** | +0.035 | Helix entry — reduced angle inflation |
| gate-8 | 0.069 | 0.070 | +0.001 | Negligible |
| gate-9 | 0.055 | 0.055 | 0 | Unchanged |
| gate-10 | 0.070 | 0.071 | +0.001 | Negligible |
| gate-11 | 0.072 | 0.072 | 0 | Unchanged |
| gate-12 | 0.076 | 0.078 | +0.002 | Negligible |

### Key observation
Gates 7-12 have very low errors (<0.15m) because the FOV relaxation step adds substantial time to later segments. The primary tracking challenge is in gates 3-5 (the S-turn section) where errors are 0.23-0.77m.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The iteration 9 inflation reduction produces a modest 2.4% race time improvement. The effect is small because the FOV relaxation step (`_relax_for_fov`) is the dominant time adder — it adds ~6 seconds to the trajectory, overwhelming the ~0.5s savings from reduced inflation.

The FOV relaxation iterates 5 times, each time multiplying by 1.1 for every segment with a turn angle > 30°. With 12 gates and entry/exit waypoints, many segments have turns > 30°, causing compound inflation.

### Telemetry signals
- Max abs roll: 0.85 rad (cosmetic tilt limit)
- Max abs pitch: 0.85 rad (cosmetic)
- Avg pitch: -0.055 rad
- Avg thrust: 0.715
- Controller saturation: cosmetic (doesn't affect kinematic sim)

### FOV relaxation analysis (NEW FINDING)
The `_relax_for_fov` method adds ~6 seconds to every trajectory:
- L-BFGS produces ~14.8s base trajectory
- Inflation adds ~0.5s → ~15.3s
- FOV relaxation adds ~5.9s → ~21.2s

The FOV penalty is 14,727 (threshold for relaxation is 1.0). This is overwhelmingly triggered. The relaxation uses waypoint-level turn angles (>30° threshold), catching most segments. Each of 5 iterations applies 1.1× to all turning segments.

**This is the primary speed bottleneck.** Reducing inflation parameters has diminishing returns when FOV relaxation dominates.

### Trend analysis
| Iteration | Race Time* | Avg Error* | Approach | Tradeoff |
|-----------|-----------|-----------|----------|----------|
| 5 | 20.51s | 0.333m | Speed optimization | Baseline (with FOV relax) |
| 6 | 20.90s | 0.312m | Angle inflation | Speed ↓, accuracy ↑ |
| 7 | 21.56s | 0.306m | Centripetal inflation | Speed ↓, accuracy ↑ |
| 8 | 21.56s | 0.200m | Feedforward activation | Accuracy ↑, speed = |
| **9** | **21.05s** | **0.209m** | **Reduced inflation** | **Speed ↑, accuracy ≈** |

*Verified metrics from current environment (differ from originally reported due to FOV relaxation impact)

The feedforward fix (iter 8) dramatically improved accuracy without speed cost. Iter 9 recovers some speed but the gains are limited by FOV relaxation.

### Drag compensation failure analysis
Tested extensively: adding `drag_coefficient * vel` to controller's accel_des. 8 configurations tested with drag coefficients 0.3-0.5, feedforward weights 0.4-0.8, and kd_xy 4.0-5.5. All regressed. Root cause: sim drag (`-0.5*vel`) provides:
1. **Extra damping**: effective kd = kd_xy + 0.5 = 4.5. Removing drag drops damping ratio from 0.92 to 0.82.
2. **Speed limiting into turns**: drag naturally slows the drone before sharp turns, giving the PD controller more time.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **Fix FOV relaxation — primary speed bottleneck** (Priority 1, trajectory_planning)
   - FOV relaxation adds ~6s to trajectory. The penalty threshold (1.0) is too aggressive and the relaxation algorithm (5 iterations × 1.1× per turning segment) compounds excessively.
   - Proposed approaches: (a) raise the FOV penalty threshold to 100 or 1000, (b) cap maximum relaxation to 1.5× per segment, (c) only relax segments with extreme FOV violations (>1 radian outside FOV), (d) disable FOV relaxation entirely for kinematic sim benchmarks.
   - Expected impact: race time 21.05→15-16s
   - Research refs: [perception_aware_planning_eth_2026]

2. **Norm-based acceleration constraints** (Priority 2, trajectory_planning)
   - Per Teissing RA-L 2024: norm constraint vs per-axis gives >20% faster trajectories
   - Currently optimizer uses per-axis velocity and acceleration constraints
   - Expected impact: 10-20% faster base trajectory
   - Research refs: [realtime_mintime_teissing_2024]

3. **S-turn cumulative effect on gates 3-4** (Priority 3, trajectory_planning)
   - Gate-3 is worst at 0.771m, gate-4 at 0.432m
   - S-turn creates alternating turn directions that accumulate tracking error
   - Proposed approach: detect consecutive opposite-direction turns and boost inflation for second turn
   - Research refs: [taco_sanghvi_2025]

4. **Controller gains re-optimization** (Priority 4, control)
   - With feedforward active, optimal PD gains may differ
   - Bayesian optimization of kp_xy, kd_xy against benchmark reward
   - Expected impact: avg error -5-10%
   - Research refs: [leveling_playing_field_2025]

5. **PyBullet physics validation** (Priority 5, system_integration)
   - Kinematic sim results may not transfer to physics sim
   - Expected impact: more realistic benchmarking

### Architectural recommendations
- **FOV relaxation is the critical path for speed.** The current implementation is too aggressive — it adds 40% to trajectory time. Fixing this should be the next iteration's primary focus.
- **Drag compensation is a dead end for this kinematic sim.** The sim's drag model provides beneficial damping that shouldn't be cancelled. Document this clearly so future iterations don't re-attempt it.
- **The inflation parameters are now well-tuned for the current controller capability.** Further inflation reduction will trade accuracy for marginal speed gains. The big speed gains must come from fixing FOV relaxation or changing the trajectory optimization approach.

### Next bottleneck
`trajectory_planning` — FOV relaxation is adding ~6s to the trajectory. This is the primary speed bottleneck.

### What NOT to try
- Don't increase time_weight beyond 2.0 — L-BFGS converges to worse local minimum (iter 8 lesson)
- Don't add drag compensation to controller — regresses tracking (iter 9 lesson, extensively tested)
- Don't modify entry/exit offsets (iter 6 lesson)
- Don't modify transition_time acceleration estimate (iter 5 lesson)
- Don't reduce inflation parameters further — diminishing returns when FOV relaxation dominates

---

## Section 7: Lessons Learned

### What worked
- **Inflation parameter reduction**: Modest but real speed improvement (-2.4%) by leveraging feedforward's improved turn handling
- **Systematic drag compensation sweep**: Testing 8 configurations definitively ruled out drag compensation for this sim

### What didn't work
- **Drag compensation (Faessler 2018 style)**: All configurations regressed tracking. The sim's linear drag provides beneficial velocity damping that acts as extra kd.
- **Large inflation reductions**: The FOV relaxation step dominates trajectory time, limiting the impact of inflation parameter changes.

### Surprises
- **FOV relaxation adds ~6 seconds to every trajectory**: This was discovered when verifying benchmark reproducibility. The `_relax_for_fov` method is the dominant time adder, not the L-BFGS optimizer or the inflation parameters.
- **Metrics discrepancy**: The benchmark produces different results now than in the previous session. The most likely cause is that intermediate code states (with FOV relaxation disabled or modified) produced the originally reported metrics. This highlights the importance of verifying benchmark reproducibility.
- **Gates 7-12 have excellent tracking (<0.15m)**: The FOV relaxation preferentially slows turn-heavy later segments, giving the controller plenty of time for the helix section.

### Process suggestions
- Always re-run the benchmark at the start of a new session to verify baseline metrics before making changes
- When benchmarks produce unexpected results, trace the trajectory generation pipeline step by step to identify where time is being added
- Document environmental dependencies (scipy/numpy versions) that could affect L-BFGS convergence
