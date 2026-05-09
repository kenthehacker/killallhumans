# Iteration 2 Implementation Plan

## Objective
Tighten benchmark thresholds to aspirational targets and improve trajectory quality through curvature-aware time allocation and full feedforward tracking.

**Target metrics:**
- max_avg_tracking_error_m: 1.0 → 0.5 (aspirational)
- max_max_tracking_error_m: 4.0 → 2.0 (aspirational)
- Maintain: 12/12 gates, no crash, >100 Hz loop

## Research Basis
1. **TOGT Planner (Qin 2024)**: curvature-aware segment time allocation — allocate more time to turning segments, less to straight segments
2. **Leveling the Playing Field (2025)**: feedforward acceleration is the most important fix for geometric controllers — increase from 0.8 to 1.0

## Files to Modify

### 1. `scripts/benchmark.py` — Tighten thresholds
**Change**: Update THRESHOLDS dict at line 46-55
- `max_avg_tracking_error_m`: 1.0 → 0.5
- `max_max_tracking_error_m`: 4.0 → 2.0
- Also increase `feedforward_accel` in the tracker config (line 298-301) from default 0.8 to 1.0

### 2. `planning/trajectory_optimizer.py` — Curvature-aware time allocation
**Change**: Replace `_initial_time_allocation()` method (lines 265-276)
- Instead of uniform `avg_speed = max_velocity * 0.6` for all segments:
  - Compute turn angle at each waypoint (angle between incoming/outgoing vectors)
  - Straight segments (turn < 0.3 rad): use factor 0.65 (slightly faster)
  - Moderate turns (0.3-1.0 rad): use factor 0.55 (moderate)
  - Sharp turns (> 1.0 rad): use factor 0.45 (slower, more time for tracking)
- This is directly inspired by TOGT's insight that "high-curvature segments naturally require more time"

### 3. `control/mpc_tracker.py` — Increase default feedforward
**Change**: Update `TrackerConfig.feedforward_accel` default (line 47)
- From 0.8 to 1.0 (full feedforward)
- Research backing: "feedforward terms are the most important single fix" (Leveling the Playing Field, 2025)

## Algorithm Changes (pseudocode)

### Curvature-aware time allocation:
```python
def _initial_time_allocation(self, waypoints):
    times = []
    for i in range(len(waypoints) - 1):
        dist = norm(waypoints[i+1] - waypoints[i])

        # Compute turn angle at the endpoint
        if i + 1 < len(waypoints) - 1:
            v_in = waypoints[i+1] - waypoints[i]
            v_out = waypoints[i+2] - waypoints[i+1]
            turn_angle = angle_between(v_in, v_out)
        else:
            turn_angle = 0.0  # last segment, no turn

        # Curvature-aware speed factor
        if turn_angle < 0.3:      # < ~17 degrees
            speed_factor = 0.65
        elif turn_angle < 1.0:    # < ~57 degrees
            speed_factor = 0.55
        else:                     # sharp turn
            speed_factor = 0.45

        avg_speed = self.constraints.max_velocity * speed_factor
        t = max(dist / avg_speed, 0.3)
        times.append(t)
    return times
```

## Risk Assessment
- **Low risk**: Threshold tightening — current metrics already meet aspirational targets (0.333m < 0.5m, 0.822m < 2.0m)
- **Low risk**: Feedforward increase — from 0.8 to 1.0 is a small change, reduces tracking lag
- **Medium risk**: Curvature-aware allocation — changes trajectory timing, could affect gate passing. But the L-BFGS optimizer runs after this, so it will fine-tune.

## Rollback Criteria
- Revert ALL changes if avg_tracking_error > 0.5m OR max_tracking_error > 2.0m
- Revert trajectory changes only if gate_pass_rate < 1.0

## Test Plan
1. Run unit tests after each file change
2. Run `--mode sim` benchmark to verify trajectory quality
3. Run `--mode full` for final comparison
