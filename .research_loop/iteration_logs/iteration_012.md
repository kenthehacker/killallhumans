# Iteration 12 — Trajectory-Aware Gain Scheduling / Dual-Timescale Feedforward

**Date**: 2026-04-14
**Bottleneck**: control — helix tracking error (gate-7: 0.659m worst gate)
**Status**: REVERTED — all code changes reverted; no net improvement
**Commits**: d253f6d (dual-timescale FF), 3e07fc9 (revert)

---

## Section 1: Summary
- Iteration 12, timestamp 2026-04-14
- Bottleneck: control — PD controller helix tracking, specifically gate-7 (0.659m)
- Outcome: **REVERTED**. Investigated trajectory-aware gain scheduling (6 approaches) and dual-timescale feedforward. Gain scheduling fundamentally doesn't work in kinematic sim without attitude dynamics. Dual-timescale FF improved on one trajectory realization but regressed on another, revealing trajectory optimizer non-determinism as a systemic issue.

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Deep Q-Learning-Based Gain Scheduling** (arXiv:2603.03127, 2026)
   - URL: https://arxiv.org/abs/2603.03127
   - DQN selects from 625 pre-certified gains; phase variable enables trajectory-aware anticipation; 14-dim state input
   - Key insight: gain scheduling needs attitude dynamics to be effective

2. **Task-Parameter Nexus** (arXiv:2412.12448, Shen et al., 2024)
   - URL: https://arxiv.org/abs/2412.12448
   - Speed×curvature grid determines optimal gains; D-gain should be REDUCED for aggressive trajectories
   - Key contradiction resolved: P/D ratio should shift toward P during turns

3. **Adaptive Gain Scheduling using RL** (arXiv:2403.07216, 2024)
   - URL: https://arxiv.org/abs/2403.07216
   - PPO reactive gain scheduling; 43-49% ISE reduction; observes current error only (no trajectory lookahead)
   - Demonstrates that reactive gain scheduling is inferior to predictive approaches

### Previously analyzed (key references)
4. **TACO** (Sanghvi 2025) — trajectory-aware gain optimization at 2Hz
5. **Tal & Karaman** (2018/2021) — jerk/snap feedforward via differential flatness
6. **Aggressiveness-Aware Control** (Colombo 2026) — formal gain scheduling framework

### Research consensus
All 6 papers agree gain scheduling provides 30-50% improvement in real drone systems with attitude dynamics. However, this improvement relies on the gains affecting HOW the drone achieves desired acceleration (through attitude changes), which doesn't exist in a kinematic sim where acceleration is applied directly.

---

## Section 3: Implementation

### Approaches tested (all failed or reverted)

#### Approach 1: Curvature-based gain scheduling (EMA of acceleration magnitude)
- 10 configurations tested with varying kp_boost (0.3-1.0), kd_boost (0-0.5), acc_scale (20-50), ema_alpha (0.05-0.2)
- **Result**: All configurations regressed avg error by 5-20%
- **Root cause**: Trajectory acceleration spikes at ALL polynomial segment boundaries (gate waypoints), not just at hard turns. The curvature signal doesn't discriminate between geometrically hard turns and routine segment transitions.

#### Approach 2: Error-based gain scheduling
- 6 configurations tested with varying error thresholds and boost magnitudes
- **Result**: All configurations regressed avg error
- **Root cause**: In kinematic sim, boosting kp when error is large redirects the clamped acceleration AWAY from the optimal feedforward direction. With direct acceleration control, the feedforward direction IS the optimal direction; PD correction only helps when the error is small enough not to dominate.

#### Approach 3: Feedforward attenuation during saturation
- 4 configurations with ff_budget limiting total FF acceleration
- **Result**: Catastrophic regression — race timeouts or severe error increase
- **Root cause**: The feedforward IS the primary acceleration source; attenuating it removes the controller's ability to follow the trajectory.

#### Approach 4: Predictive position targeting
- 14 configurations shifting position target ahead in time
- **Result**: All regressed; shifting position target causes PD terms to overshoot
- **Root cause**: The position target shift makes the drone aim ahead of where it should be, compounding error at turns.

#### Approach 5: Integral term (PID)
- 5 configurations with varying ki gains
- **Result**: All regressed; integral causes overshoot on helix transitions
- **Root cause**: The helix is pure transients; integral accumulates during one turn and causes overshoot into the next.

#### Approach 6: Dual-timescale feedforward (COMMITTED then REVERTED)
- Split feedforward between current-time (weight 0.10) and lookahead (weight 0.25) at 70ms
- Optimized on 13.59s trajectory realization: avg error 0.358→0.300m (-16.1%)
- **Reverted** because trajectory optimizer produces different results across sessions (17.46s now). On the current trajectory, dual FF is 5.5% worse than single FF.
- **Root cause**: The dual FF's benefit was specific to the fast trajectory's dynamics; the slower trajectory doesn't need the directional blending.

### Plan adherence
The original plan (curvature-based gain scheduling) failed systematically. Five alternative approaches were tried. The only one that showed improvement (dual-timescale FF) was not robust to trajectory variation and was reverted.

---

## Section 4: Benchmark Comparison

### Current metrics (after revert — same as iteration 11 baseline)
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Unit tests | 9/9 (100%) | 100% | PASS |
| Gates passed | 12/12 (100%) | 100% | PASS |
| Avg tracking error | 0.252m | 0.5m | PASS |
| Max tracking error | 1.504m | 2.0m | PASS |
| EKF uncertainty | 0.012m | 0.5m | PASS |
| Loop Hz | ~5723 | 100 | PASS |
| Trajectory time | 17.46s | — | — |
| Race time | 17.32s | 30s | PASS |

**NOTE**: The trajectory optimizer now produces 17.46s trajectories instead of the 13.59s from the previous session. This changes ALL per-gate error profiles. The metrics above are from the current trajectory realization.

### Per-gate error (current trajectory, 17.46s)
| Gate | Error | Notes |
|------|-------|-------|
| gate-1 | 0.116 | Good |
| gate-2 | 0.150 | Good |
| gate-3 | **0.905** | **WORST — S-turn** |
| gate-4 | 0.435 | S-turn |
| gate-5 | 0.298 | Moderate |
| gate-6 | 0.165 | Good |
| gate-7 | 0.161 | Good (was 0.659 on fast trajectory) |
| gate-8 | 0.104 | Good |
| gate-9 | 0.099 | Good |
| gate-10 | 0.107 | Good |
| gate-11 | 0.106 | Good |
| gate-12 | 0.134 | Good |

**Key observation**: On the slower trajectory, the helix is easy (0.10-0.16m) but gate-3 (S-turn) is terrible (0.905m). This is the opposite of the fast trajectory where the helix was hardest.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis: Why gain scheduling fails in kinematic sim
In a kinematic sim with direct acceleration control (`accel = accel_des - drag * vel`, clamped to max_accel), the controller's desired acceleration is directly applied to the drone. There are no attitude dynamics — no roll/pitch response time, no angular rate limits, no inertia. The feedforward acceleration direction is already optimal; the PD terms provide small corrections.

Gain scheduling works in REAL drones because:
1. Higher kp/kd → more aggressive attitude commands → faster rotation → quicker direction change
2. The gains affect HOW the drone achieves the desired acceleration through attitude changes

In kinematic sim:
1. Higher kp/kd → larger PD contribution → changes the DIRECTION of the clamped acceleration
2. But the feedforward direction was already optimal → PD redirects AWAY from optimal

This is a fundamental limitation of kinematic sim for controller tuning.

### Trajectory optimizer non-determinism
The L-BFGS optimizer with Apple Accelerate BLAS produces different convergence paths across sessions:
- Previous session: 14.83s → inflate → FOV relax → 13.59s total
- Current session: 14.83s → inflate → FOV relax → 17.46s total

The FOV relaxation step (`_relax_for_fov`) is the primary source of variation. The FOV penalty computation involves many floating-point operations whose order/precision can vary with Apple Accelerate.

### Telemetry signals (current trajectory, 17.46s)
- Max abs roll: 0.85 rad (tilt limit, unchanged)
- Max abs pitch: 0.85 rad (tilt limit)
- Avg thrust: 0.762
- Gate-3 error 0.905m is the dominant issue — S-turn on slower trajectory

### Trend analysis
**Trend: Plateau (with systemic risk)**. The controller improvements from iteration 11 (predictive FF) are solid, but the trajectory optimizer's non-determinism means benchmark-to-benchmark comparisons may not be valid. Any controller tuning done on one trajectory realization may regress on another.

Two critical implications:
1. Controller tuning should be robust across trajectory variations
2. The trajectory optimizer needs deterministic output (set random seeds, fix BLAS precision, or add a trajectory caching mechanism)

---

## Section 6: Forward-Looking

### Improvements backlog (re-prioritized based on findings)
1. **Fix trajectory optimizer determinism** (Priority 1, system_integration)
   - Root cause: L-BFGS convergence sensitivity + FOV relaxation amplification
   - Proposed: Cache trajectory after first generation, or set deterministic BLAS mode, or seed the optimizer
   - Expected: Reproducible benchmarks, valid iteration-to-iteration comparisons
   - Risk: May lock in a suboptimal trajectory; need to validate cached trajectory quality

2. **S-turn trajectory inflation** (Priority 2, trajectory_planning)
   - Gate-3 at 0.905m is now the worst gate by far (on 17.46s trajectory)
   - Detect consecutive opposite-direction turns and boost inflation
   - Expected: gate-3 error 0.90→0.40m

3. **Norm-based acceleration constraints** (Priority 3, trajectory_planning)
   - Teissing RA-L 2024: >20% faster with sphere vs box constraints
   - Expected: faster, smoother trajectory

4. **KAIST-style heading FOV control** (Priority 4, trajectory_planning)
   - Decouple yaw from position; handle FOV via heading
   - Expected: reduce FOV relaxation overhead (17.46→~15s trajectory)

5. **PyBullet physics validation** (Priority 5, system_integration)
   - Kinematic sim doesn't support gain scheduling — real physics needed
   - Expected: enable controller tuning approaches that require attitude dynamics

### What NOT to try
- **Any form of gain scheduling in kinematic sim** — fundamentally limited by lack of attitude dynamics
- **Integral terms (PID)** — wind-up on transient-dominated helix
- **Higher kp/kd globally** — redirects clamped acceleration away from optimal FF direction
- **Feedforward attenuation** — catastrophic; FF is essential for race completion
- **Controller tuning optimized on a single trajectory** — not robust to trajectory variation

---

## Section 7: Lessons Learned

### What worked
- **Systematic failure analysis**: Testing 6 distinct approaches with 40+ configurations revealed a fundamental limitation of kinematic sim for gain scheduling
- **A/B comparison across trajectory realizations**: Discovering the dual-timescale FF regression on the second trajectory prevented committing a non-robust change
- **Quick revert**: Reverted within the same iteration rather than shipping a fragile improvement

### What didn't work
- **Curvature-based gain scheduling**: Acceleration spikes at all segment boundaries, not just hard turns
- **Error-based gain scheduling**: Redirects acceleration away from optimal FF direction under clamp
- **Dual-timescale feedforward**: Not robust to trajectory variation
- **All gain-based controller improvements in kinematic sim**: The simulation doesn't model attitude dynamics

### Surprises
- **Trajectory optimizer non-determinism**: The same code, same inputs, same libraries produce different trajectories across sessions. This invalidates all previous iteration-to-iteration comparisons that assumed the trajectory was constant.
- **Gate-3 vs gate-7 reversal**: On the fast trajectory (13.59s), gate-7 (helix) was worst. On the slow trajectory (17.46s), gate-3 (S-turn) is worst by 5.6x. The bottleneck gate is trajectory-speed-dependent.
- **Kinematic sim limitation**: Gain scheduling is one of the most well-studied and effective techniques in quadrotor control, but it fundamentally cannot help in a sim without attitude dynamics.

### Process suggestions
- Before committing controller tuning, verify robustness by running the benchmark multiple times (to catch trajectory variation) or cache the trajectory
- The trajectory optimizer non-determinism should be the #1 priority to fix — it undermines all empirical evaluation
- Consider moving to PyBullet physics for any further controller improvement work
