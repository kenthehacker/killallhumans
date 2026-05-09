# Iteration 1 — Fix Benchmark Duration Mismatch

**Date**: 2026-04-13
**Bottleneck**: trajectory_planning (gate pass rate)
**Status**: RESOLVED — all thresholds now pass

---

## Baseline Metrics (before)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Unit tests | 9/9 (100%) | 100% | PASS |
| Gates passed | 10/12 (83%) | 100% | **FAIL** |
| Avg tracking error | 0.320m | < 1.0m | PASS |
| Max tracking error | 0.683m | < 4.0m | PASS |
| EKF uncertainty | 0.012m | < 1.0m | PASS |
| Loop Hz | 7513 | > 100 | PASS |
| Crash | No | No | PASS |
| Trajectory time | 16.02s | — | — |
| Sim duration | 15.0s | — | — |

**Root cause**: The CLI default `--duration=15.0` was shorter than the trajectory time (16.02s). The sim terminated before the drone could reach gates 11 and 12. The `max_total_time_s` threshold is 30s, so the sim should run long enough for the trajectory to complete.

---

## Post-Fix Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Unit tests | 9/9 (100%) | 100% | PASS |
| Gates passed | 12/12 (100%) | 100% | **PASS** |
| Avg tracking error | 0.333m | < 1.0m | PASS |
| Max tracking error | 0.822m | < 4.0m | PASS |
| EKF uncertainty | 0.012m | < 1.0m | PASS |
| Loop Hz | 7710 | > 100 | PASS |
| Crash | No | No | PASS |
| Trajectory time | 16.02s | — | — |
| Race complete time | 16.39s | < 30s | PASS |

**All thresholds passed.**

---

## Changes Made

1. `scripts/benchmark.py` line 722: Changed `--duration` default from `15.0` to `30.0` to match `max_total_time_s` threshold.

---

## Failed Approaches (this iteration)

- Tried increasing `max_velocity` from 10.0 to 12.0 AND making initial time allocation more aggressive (0.6→0.7 average speed factor). This made the trajectory too fast for the kinematic sim to track, causing 6.4m max tracking error and only 8/12 gates. The kinematic sim's PD controller can't follow overly aggressive references.

---

## Research Notes

- Searched for TOGT planner and time-optimal trajectory papers (arxiv 2309.06837, 2402.18021, 2305.02772, Science Robotics CPC paper, 2508.01103, 2512.09571)
- Key insight from TOGT: optimal trajectories achieve 65-80% of max velocity on average. Our current 60% factor in `_initial_time_allocation` is conservative but stable.
- Increasing trajectory aggressiveness requires matching controller capability — the kinematic sim's PD controller with max_accel=12 m/s² and max_speed=12 m/s can't track faster references without significant tracking error.

---

## Deep Diagnostic

### Per-gate analysis
- Gate-12 has the highest tracking error (0.69m) — it's reached via gate-seeking fallback after the trajectory ends
- Gates 3-4 have elevated error (~0.4m) — these are in the turn section
- Gates 1, 6, 8 have low error (<0.28m) — straight-line approaches

### Trend analysis
- Tracking error is well within thresholds (0.33m avg vs 1.0m limit)
- There's headroom to tighten thresholds toward aspirational targets
- The trajectory could be made faster IF the controller is upgraded

### Architectural observations
- The kinematic sim is a useful fast proxy but doesn't exercise perception
- PyBullet sim is unavailable (missing gym-pybullet-drones dependency)
- The real bottleneck per PLAN.md is perception/estimation in the full PyBullet sim

---

## Forward-Looking: Next Iteration Priorities

### 1. Tighten thresholds toward aspirational targets
Current performance is well within thresholds. Consider tightening:
- avg_tracking_error: 1.0 → 0.5m
- max_tracking_error: 4.0 → 2.0m

### 2. Speed up the trajectory (with controller improvements)
The trajectory takes 16s. To go faster:
- Upgrade controller (geometric tracker gains, or switch to SE(3))
- Then increase trajectory aggressiveness
- Target: <12s race completion

### 3. Address the real bottleneck: perception/estimation
PLAN.md identifies the core problem as raw detections commanding the drone in the PyBullet sim. This requires:
- Installing gym-pybullet-drones to enable the full sim
- Integrating GateTracker/GatePnP/EKF into the runner
- Filtering detections before they reach control

### Improvements Backlog (ordered)
1. Tighten benchmark thresholds (low risk, immediate)
2. Install gym-pybullet-drones and enable PyBullet benchmark
3. Integrate perception filtering (GateTracker, outlier rejection)
4. Speed up trajectory with matched controller upgrade
5. Perception-aware trajectory planning (FOV constraints)

---

## Lessons Learned

1. **Don't change multiple parameters at once** — increasing both max_velocity and time allocation factor simultaneously caused a regression that was hard to diagnose.
2. **Match trajectory aggressiveness to controller capability** — the kinematic sim's PD controller can't track references faster than its max_accel/max_speed limits.
3. **Check for configuration mismatches first** — the 15s vs 30s duration default was a simple oversight, not a deep algorithmic problem.
