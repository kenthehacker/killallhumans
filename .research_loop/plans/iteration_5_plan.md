# Iteration 5 Plan — Speed Optimization

## Objective
Reduce race time from 23.0s toward 14s target while maintaining:
- Gate pass rate = 100%
- Avg tracking error < 0.5m
- No crash

## Research Basis
- TOGT Planner (Qin 2024): L-BFGS time optimization with relaxed penalties
- AOS (Shao/Scaramuzza 2024): segment times as free variables, joint optimization
- "On Your Own" (Romero 2025): competition speeds well above 10 m/s
- "Leveling the Playing Field" (2025): feedforward enables aggressive tracking

## Root Cause
The benchmark creates `TrajectoryOptimizer(constraints=DroneConstraints(max_velocity=10.0))` — an artificial 10 m/s ceiling. The L-BFGS acceleration penalty weight (50) is also too high for the 25-segment trajectory.

## Files to Modify

### 1. `scripts/benchmark.py` (lines ~291-313)
- Change `max_velocity=10.0` → `max_velocity=15.0` in TrajectoryOptimizer creation
- Change `max_speed = 12.0` → `max_speed = 15.0` in kinematic sim
- Change `max_accel = 12.0` → `max_accel = 15.0` in kinematic sim

### 2. `planning/trajectory_optimizer.py`
- `DroneConstraints.max_acceleration`: 12.0 → 15.0
- `DroneConstraints.max_tilt_angle`: 0.7 → 0.85
- `_initial_time_allocation` speed factors: 0.65/0.55/0.45 → 0.80/0.70/0.55
- `_optimize_time_allocation` acceleration penalty weight: 50 → 25

### 3. `control/mpc_tracker.py`
- `TrackerConfig.max_tilt_rad`: 0.7 → 0.85

## Algorithm Changes (Pseudocode)

### Trajectory Optimizer — `_optimize_time_allocation`:
```
# Before: penalty += (accel - max_acceleration)^2 * 50
# After:  penalty += (accel - max_acceleration)^2 * 25
```
Rationale: TOGT uses cubic penalties; reducing our quadratic weight makes the optimizer more willing to push speed limits at segment boundaries.

### Initial Time Allocation:
```
# Before: straight=0.65, moderate=0.55, sharp=0.45 of max_velocity
# After:  straight=0.80, moderate=0.70, sharp=0.55 of max_velocity
```
With max_velocity=15, this gives initial speeds of 12/10.5/8.25 m/s instead of 6.5/5.5/4.5 m/s.

## Risk Assessment
- **Tracking error increase**: Expected 0.186m → 0.25-0.35m. Safe within 0.5m threshold.
- **Controller saturation**: Max tilt increase 0.7→0.85 provides headroom.
- **Gate miss**: Unlikely — trajectory still passes through entry/exit waypoints.
- **Crash**: Unlikely — kinematic sim has crash detection at z<0.05.

## Rollback Criteria
- If avg tracking error > 0.45m: revert (too close to threshold)
- If gate pass rate < 100%: revert immediately
- If crash: revert
- If race time doesn't improve by at least 3s: revert (not worth the risk)

## Test Plan
1. Run unit tests after each file change
2. Run full benchmark and compare all metrics
3. Check per-gate error breakdown for regressions
