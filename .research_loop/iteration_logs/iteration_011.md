# Iteration 11 — Predictive Feedforward Acceleration Lookahead

**Date**: 2026-04-13
**Bottleneck**: control — PD controller can't track helix turns at speed (gates 7-12 avg 0.641m)
**Status**: COMMITTED — avg error 0.481→0.358m (-25.6%), race time 13.34→13.31s
**Commit**: 7306716

---

## Section 1: Summary
- Iteration 11, timestamp 2026-04-13T18:17
- Bottleneck: control — helix tracking error dominated by PD controller's reactive nature
- Outcome: 50ms predictive feedforward lookahead reduces avg error from 0.481m to 0.358m (-25.6%) while maintaining 13.31s race time. All thresholds pass with margin.

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **TACO: Trajectory-Aware Controller Optimization** (Sanghvi et al., 2025)
   - URL: https://arxiv.org/abs/2511.02060
   - Adapts controller gains in real-time based on upcoming trajectory and state
   - 2Hz gain adaptation, 0.5s horizon, significantly outperforms static tuning

2. **MPCC++: Model Predictive Contouring Control** (Krinner et al., RSS 2024)
   - URL: https://arxiv.org/abs/2403.17551
   - Safety constraints + learned residual dynamics + TuRBO hyperparameter tuning
   - Matches RL lap times with 100% gate safety; speeds > 80 km/h

3. **Neural-Augmented INDI** (Cobo-Briesewitz et al., 2025)
   - URL: https://arxiv.org/abs/2503.09441
   - Learning-based prediction of INDI residuals for smoother disturbance rejection

### Previously analyzed (key references)
4. **Tal & Karaman 2018** — INDI + differential flatness for jerk/snap feedforward (6.6cm RMS)
5. **L1Quad 2025** — L1 adaptive augmentation (5x RMSE reduction)
6. **DATT 2023** — Learned feedforward + L1 adaptive (0.049m on smooth trajectories)

### Research consensus
**All papers agree**: anticipatory feedforward (whether via differential flatness derivatives, learned trajectory embeddings, or predictive gain adaptation) is essential for aggressive tracking. Pure feedback control cannot keep up with rapid trajectory changes.

### Key insight from cross-validation
The mathematical analysis of reference-velocity drag feedforward was correct (cancels drag-induced steady-state forcing) but failed in practice because the helix is transient-dominated. The successful approach — predictive feedforward lookahead — came from Tal & Karaman's insight that jerk feedforward anticipates angular rate changes, adapted to the time domain.

---

## Section 3: Implementation

### Changes made
1. **`control/mpc_tracker.py`** — Added `velocity_feedforward` parameter to TrackerConfig
   - New parameter for reference-velocity drag feedforward (not active, kept for future experiments)

2. **`scripts/benchmark.py`** — Added 50ms predictive feedforward lookahead
   - Samples trajectory acceleration from `sim_time + 0.05s` instead of `sim_time`
   - Position/velocity targets remain at current time (PD tracks current error)
   - Only the feedforward acceleration is shifted forward

### Failed experiments within iteration
1. **Reference-velocity feedforward (vff=0.5, ff=0.7)**: Race time 18.36s, avg error 0.929m — catastrophic overshoot. Velocity feedforward in ref_vel direction makes drone faster into turns.
2. **Higher ff alone (ff=0.6, no vff)**: Avg error 0.552m, race time 14.62s — worse than baseline. Higher feedforward amplifies wrong-direction acceleration under tracking error.
3. **Higher gains + small vff (kp=8, kd=5, vff=0.15)**: Avg error 0.545m — higher gains don't help; they amplify oscillation at turns.
4. **Small vff alone (vff=0.1)**: Avg error 0.516m — even tiny velocity feedforward worsens helix transients.
5. **ff=0.5 with 50ms lookahead**: Avg error 0.381m — slightly worse than ff=0.4 with lookahead.

### Plan adherence
The original plan (reference-velocity drag feedforward) failed systematically. The successful approach (predictive feedforward lookahead) was a research-backed pivot inspired by the same underlying insight from Tal & Karaman 2018 — anticipate trajectory changes rather than react to them.

---

## Section 4: Benchmark Comparison

### Full metrics (before → after)
| Metric | Before | After | Delta | Direction | Threshold | Status |
|--------|--------|-------|-------|-----------|-----------|--------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | 0 | = | 100% | PASS |
| Gates passed | 12/12 (100%) | 12/12 (100%) | 0 | = | 100% | PASS |
| Avg tracking error | 0.481m | **0.358m** | **-0.123m** | ↓ | 0.5m | PASS |
| Max tracking error | 1.724m | **1.463m** | **-0.261m** | ↓ | 2.0m | PASS |
| P50 tracking error | 0.390m | 0.276m | -0.114m | ↓ | — | — |
| P95 tracking error | 1.148m | 0.864m | -0.284m | ↓ | — | — |
| EKF uncertainty | 0.012m | 0.012m | ~0 | = | 0.5m | PASS |
| Loop Hz | ~7541 | ~7916 | +375 | ↑ | 100 | PASS |
| Trajectory time | 13.59s | 13.59s | 0 | = | — | — |
| Race time | 13.34s | **13.31s** | **-0.03s** | ↓ | 30s | PASS |

### Per-gate error breakdown
| Gate | Before | After | Delta | % Change | Notes |
|------|--------|-------|-------|----------|-------|
| gate-1 | 0.111 | 0.119 | +0.008 | +7.2% | Slight regression (minor) |
| gate-2 | 0.262 | 0.268 | +0.006 | +2.3% | Negligible |
| gate-3 | 0.533 | **0.402** | **-0.131** | **-24.6%** | S-turn improved |
| gate-4 | 0.511 | **0.388** | **-0.123** | **-24.1%** | S-turn improved |
| gate-5 | 0.236 | 0.191 | -0.045 | -19.1% | Improved |
| gate-6 | 0.208 | 0.196 | -0.012 | -5.8% | Minor |
| gate-7 | 0.832 | **0.659** | **-0.173** | **-20.8%** | Helix entry — significant |
| gate-8 | 0.668 | **0.528** | **-0.140** | **-21.0%** | Helix |
| gate-9 | 0.624 | **0.407** | **-0.217** | **-34.8%** | Helix — best improvement |
| gate-10 | 0.593 | **0.404** | **-0.189** | **-31.9%** | Helix |
| gate-11 | 0.609 | **0.385** | **-0.224** | **-36.8%** | Helix — best % improvement |
| gate-12 | 0.522 | **0.344** | **-0.178** | **-34.1%** | Helix exit |

**Helix gates (7-12) avg: 0.641m → 0.455m (-29.0%)**
**Non-helix gates (1-6) avg: 0.310m → 0.261m (-15.8%)**

### Lookahead sweep results
| Lookahead | Avg Error | Max Error | gate-7 |
|-----------|-----------|-----------|--------|
| 0ms (base) | 0.481m | 1.724m | 0.832 |
| 30ms | 0.382m | 1.552m | 0.724 |
| 40ms | 0.364m | 1.490m | 0.680 |
| **50ms** | **0.358m** | **1.463m** | **0.659** |
| 60ms | 0.362m | 1.440m | 0.649 |
| 70ms | 0.369m | 1.457m | 0.659 |
| 100ms | 0.394m | 1.548m | 0.743 |

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The PD controller computes feedforward acceleration from the trajectory at the CURRENT time. But during turns, the needed acceleration changes rapidly (jerk is high). By the time the position/velocity error builds and the PD terms respond, the optimal acceleration direction has already shifted. The 50ms lookahead gives the feedforward term a head start — it begins commanding the turn acceleration 50ms before the current-time feedforward would, reducing the error peak during the turn.

This is physically equivalent to Tal & Karaman 2018's jerk feedforward via differential flatness, but implemented as a simple time shift rather than computing angular rate references from jerk derivatives. Both approaches achieve the same goal: the controller anticipates the trajectory's curvature rather than reacting to the resulting tracking error.

### Telemetry signals
- Max abs roll: 0.85 rad (unchanged — still hitting tilt limit)
- Max abs pitch: 0.85 rad (unchanged)
- Avg thrust: 0.779 (decreased from 0.804 — less aggressive correction needed)
- Controller still saturating at tilt limit during helix

### Why velocity feedforward failed (key finding)
Mathematical analysis showed `vff * ref_vel` should cancel drag-induced steady-state forcing. This was correct analytically, but the helix is not at steady state — it's a sequence of transient direction changes. Adding velocity-direction acceleration makes the drone FASTER into each turn, but turn tracking requires DECELERATION in the old direction and ACCELERATION in the new direction. The velocity feedforward adds energy in the wrong mode.

The predictive feedforward succeeds because it shifts the ACCELERATION direction (which captures turning requirements), not the velocity direction (which captures along-track speed).

### Trend analysis
**Trend: Improving.** Two consecutive iterations of progress. The speed-accuracy Pareto has shifted favorably: iteration 10 sacrificed accuracy for speed, iteration 11 recovered accuracy without sacrificing speed. The system is now at (13.31s, 0.358m) vs the original (15.56s, 0.274m) from iteration 9 — 14% faster and only 30% worse tracking.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **Trajectory-aware gain scheduling for helix** (Priority 1, control)
   - Gate-7 (0.659m) and gate-8 (0.528m) remain the worst gates
   - Detect high-curvature sections from trajectory jerk and temporarily increase kp/kd
   - TACO paper shows 2Hz gain adaptation outperforms static tuning
   - Expected: gate-7 0.659→0.45m, helix avg 0.455→0.35m

2. **S-turn trajectory inflation** (Priority 2, trajectory_planning)
   - Gates 3-4 at 0.40/0.39m — second largest error source
   - Detect consecutive opposite-direction turns and boost inflation for second turn
   - Expected: gates 3-4 error 0.40→0.25m

3. **Norm-based acceleration constraints** (Priority 3, trajectory_planning)
   - Teissing RA-L 2024: >20% faster with sphere vs box constraints
   - Expected: 10-20% faster base trajectory

4. **KAIST-style heading FOV control** (Priority 4, trajectory_planning)
   - Decouple yaw from position; handle FOV via heading with zero speed penalty
   - Expected: further FOV relaxation reduction or elimination

5. **PyBullet physics validation** (Priority 5, system_integration)

### Architectural recommendations
- The predictive feedforward lookahead is a simple, effective hack, but it assumes the trajectory is smooth and the drone can follow it with delay. For a real competition, the lookahead should be integrated into the controller architecture (e.g., as a trajectory-aware feedforward encoder per TACO/DATT).
- The 50ms lookahead corresponds to ~5 control cycles at 100Hz. On real hardware with latency, this would need to be adjusted to account for command-to-actuation delay.

### What NOT to try
- Don't increase feedforward_accel beyond 0.4 — regresses helix tracking due to wrong-direction amplification
- Don't add velocity_feedforward — regresses transient tracking in helix
- Don't increase kp/kd globally — doesn't help (tested kp=8, kd=5)
- Don't increase lookahead beyond 60ms — over-anticipation worsens tracking
- MPCC is too complex for one iteration — would require full controller redesign

---

## Section 7: Lessons Learned

### What worked
- **Predictive feedforward via time shift**: Simple 50ms lookahead on trajectory acceleration achieves ~25% error reduction. This is the time-domain analog of Tal & Karaman's jerk feedforward.
- **Systematic parameter sweep**: Testing 6 lookahead values (30-100ms) found the optimal at 50ms.
- **Quick pivoting from failed approach**: The original velocity feedforward plan failed, but the underlying insight (anticipate trajectory changes) led to the successful time-shift approach.

### What didn't work
- **Reference-velocity drag feedforward**: Mathematically elegant but practically useless for transient-dominated dynamics.
- **Higher feedforward weight (ff=0.5, 0.6)**: Amplifies wrong-direction acceleration under tracking error.
- **Higher PD gains (kp=8, kd=5)**: Doesn't help — the issue was anticipation, not gain.

### Surprises
- **The velocity feedforward failure**: The mathematical analysis was rigorous and correct at steady state, but the helix is pure transients. This is a cautionary tale about linear analysis of highly dynamic systems.
- **50ms is 5 timesteps**: The optimal lookahead is exactly 5 simulation timesteps (dt=0.01s). This suggests the improvement comes from "seeing one turn ahead" in the trajectory.
- **Gate-9 through gate-12 improved more than gate-7**: The consecutive turn anticipation compounds — later helix gates benefit more because the controller is already in a better state from anticipating the previous turn.

### Process suggestions
- When a mathematically motivated approach fails, check whether the failure mode is transient-dominated vs steady-state-dominated. The helix is all transients.
- Predictive/lookahead approaches can be tested with a simple time shift before implementing complex derivative-based feedforward.
- The per-gate breakdown is essential for understanding WHY an approach helps or hurts at specific locations.
