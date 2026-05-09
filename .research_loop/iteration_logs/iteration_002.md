# Iteration 2 — Tighten Thresholds to Aspirational Targets

**Date**: 2026-04-13
**Bottleneck**: threshold_tightening
**Status**: COMMITTED — thresholds tightened, all aspirational targets pass
**Commit**: 1185e1b

---

## Section 1: Summary
- Iteration 2, timestamp 2026-04-13T15:28
- Bottleneck: threshold_tightening (all metrics well within original thresholds)
- Outcome: Tightened benchmark thresholds to aspirational targets (avg 1.0→0.5m, max 4.0→2.0m, EKF 1.0→0.5m). Added curvature-aware time allocation and full feedforward. All thresholds pass.

---

## Section 2: Research
### Papers analyzed
1. **TOGT Planner** (Qin et al., ICRA 2024) — arxiv:2309.06837
   - Time-optimal gate-traversing with gates as regions, not points
   - Curvature-aware time allocation via joint optimization
2. **Leveling the Playing Field** (Kunapuli et al., 2025) — arxiv:2506.17832
   - Geometric controllers with full feedforward are competitive with RL
   - "Feedforward is the most important single fix for geometric controllers"
3. **Perception-Aware Planning** (ETH 2026) — arxiv:2603.04305 (analysis in progress)
4. **Fast Minimum-Snap** (Burke 2020) — arxiv:2008.00595 (analysis in progress)

### Key insight
The TOGT planner shows that per-segment time optimization is critical, but our L-BFGS optimizer already handles this — the initial allocation heuristic matters less than expected. The feedforward paper confirms our geometric controller approach is sound.

### Consensus
- Curvature drives time allocation (TOGT)
- Full feedforward essential for GC (Leveling the Playing Field)
- Gates should be treated as regions (TOGT) — future work

---

## Section 3: Implementation
### Changes made
1. `scripts/benchmark.py`: Tightened thresholds — avg error 1.0→0.5m, max error 4.0→2.0m, EKF 1.0→0.5m
2. `scripts/benchmark.py`: Added `feedforward_accel=1.0` to tracker config
3. `control/mpc_tracker.py`: Changed default feedforward weight from 0.8 to 1.0
4. `planning/trajectory_optimizer.py`: Replaced uniform 0.6 speed factor with curvature-aware allocation (0.45-0.65 depending on turn angle)

### Plan adherence
Followed plan exactly. Three changes as planned: threshold tightening, curvature-aware allocation, feedforward increase.

---

## Section 4: Benchmark Comparison

### Full metrics (before → after)
| Metric | Before | After | Delta | Threshold | Status |
|--------|--------|-------|-------|-----------|--------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | 0 | 100% | PASS |
| Gates passed | 12/12 (100%) | 12/12 (100%) | 0 | 100% | PASS |
| Avg tracking error | 0.333m | 0.333m | ~0 | **0.5m** | PASS |
| Max tracking error | 0.822m | 0.827m | +0.005m | **2.0m** | PASS |
| P95 tracking error | 0.626m | 0.625m | -0.001m | — | — |
| EKF uncertainty | 0.012m | 0.012m | ~0 | **0.5m** | PASS |
| Loop Hz | 7556 | 7551 | ~0 | 100 | PASS |
| Trajectory time | 16.02s | 16.02s | ~0 | — | — |
| Race time | 16.39s | 16.41s | +0.02s | 30s | PASS |

### Per-gate error breakdown
| Gate | Before | After | Notes |
|------|--------|-------|-------|
| gate-1 | 0.182 | 0.182 | Good |
| gate-2 | 0.292 | 0.292 | OK |
| gate-3 | 0.424 | 0.424 | S-turn, elevated |
| gate-4 | 0.408 | 0.408 | Turn, elevated |
| gate-5 | 0.345 | 0.345 | OK |
| gate-6 | 0.206 | 0.207 | Good |
| gate-7 | 0.457 | 0.457 | Helix entry, elevated |
| gate-8 | 0.277 | 0.277 | Helix |
| gate-9 | 0.289 | 0.288 | Helix |
| gate-10 | 0.296 | 0.296 | Helix |
| gate-11 | 0.295 | 0.296 | Helix |
| gate-12 | **0.693** | **0.694** | Worst — gate-seeking fallback |

### Threshold status
All thresholds pass with the new aspirational targets.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The system's tracking error is dominated by two factors: (1) gate-12's gate-seeking fallback after the trajectory ends at t=16.02s, accounting for the worst error at 0.694m; and (2) turn-section PD controller lag at gates 3, 4, 7 where curvature is high. The curvature-aware initialization didn't change metrics because L-BFGS already optimizes segment times; the feedforward increase (0.8→1.0) is correct but too small to shift kinematic sim metrics.

### Telemetry signals
- Target jumps > 1m: N/A (kinematic sim, no detection)
- Detection frame %: N/A
- Recovery count: N/A
- Max abs roll: 0.7 rad (at controller limit)
- Max abs pitch: 0.7 rad (at controller limit)

### Code quality findings
- The L-BFGS optimizer in `_optimize_time_allocation` effectively overrides the initial time allocation, making the curvature-aware heuristic less impactful than expected. However, it provides a better starting point for convergence.
- The gate-seeking fallback (benchmark.py:363-373) creates TrajectoryPoints with zero acceleration/jerk, causing tracking quality to degrade when the trajectory ends before the drone passes through gate-12.
- The max tilt limit of 0.7 rad is being hit regularly, suggesting the controller wants more aggressive attitude commands at turns.

### Trend analysis
- **Improving**: First iteration fixed duration bug (10→12 gates). This iteration locked in aspirational thresholds.
- **Stagnating metric**: Tracking error hasn't changed — it requires either faster trajectory (more aggressive) or better controller tuning.
- **Pattern**: gate-12 is consistently the worst by ~2x, always due to the same gate-seeking fallback.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **Fix gate-12 trajectory extension** (Priority 1, trajectory_planning)
   - Add a virtual waypoint ~2m past gate-12 so the trajectory covers the full gate approach
   - Expected: gate-12 error 0.69m → ~0.3m
2. **Increase controller gains** (Priority 2, control)
   - kp_xy 4.0→5.0, kd_xy 3.0→3.5 to reduce turn tracking lag
   - Expected: gates 3,4,7 error reduction ~0.05m each
3. **Reduce race time** (Priority 3, trajectory_planning)
   - After controller gains increase, try higher speed factors (0.70)
   - Target: <14s race completion
4. **Install gym-pybullet-drones** (Priority 4, system_integration)
   - Enable PyBullet benchmark for realistic testing
5. **Integrate perception filtering** (Priority 5, perception_estimation)
   - GateTracker/GatePnP into runner per PLAN.md

### Architectural recommendations
- The gate-seeking fallback in benchmark.py should be replaced with trajectory extension past the last gate. This is a design issue, not a tuning issue.

### Next bottleneck
`trajectory_planning` — fix gate-12 trajectory extension and begin race time reduction.

### What NOT to try
- Don't increase max_velocity without matching controller gains (failed in iteration 1)
- Don't change multiple velocity/timing parameters simultaneously

---

## Section 7: Lessons Learned

### What worked
- Tightening thresholds was safe — ample headroom between current metrics and aspirational targets
- Full feedforward is the correct default for geometric controllers (research-backed)
- Curvature-aware allocation is better code even if L-BFGS masks the impact

### What didn't work
- Curvature-aware time allocation didn't measurably change metrics because L-BFGS optimizes over the initial guess anyway
- The kinematic sim's direct acceleration path means feedforward weight changes don't propagate through the attitude computation

### Surprises
- The L-BFGS optimizer is more robust than expected — bad initial guesses converge to the same solution
- Gate-12 error (0.694m) is 2x the average, consistently the worst gate across iterations

### Process suggestions
- Consider running the benchmark 3x and averaging to reduce noise
- The gate-seeking fallback should be addressed as a structural fix, not a tuning target
