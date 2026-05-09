# Iteration 2 Plan — Closest-Point Trajectory Tracking + Velocity Clamp

## Target
Replace time-based `trajectory.sample(sim_time)` with position-based `trajectory.find_closest(pos)` + 0.3s lookahead, and clamp commanded velocity to 5 m/s.

## File: `scripts/visual_demo.py` lines 387-415

### Current code (lines 388-396):
```python
ref = self.trajectory.sample(sim_time)
closest = self.trajectory.find_closest(pos)
trk_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, closest.position)))
self._tracking_errors.append(trk_err)

# 6. Compute target for GPDDrone.step()
target_pos = ref.position
target_vel = ref.velocity
target_yaw = ref.yaw
```

### New code:
```python
closest = self.trajectory.find_closest(pos)
trk_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, closest.position)))
self._tracking_errors.append(trk_err)

lookahead_time = min(closest.time + 0.3, self.trajectory.total_time)
ref = self.trajectory.sample(lookahead_time)

# 6. Compute target for GPDDrone.step()
target_pos = ref.position
target_yaw = ref.yaw
ref_speed = math.sqrt(sum(v * v for v in ref.velocity))
MAX_CMD_SPEED = 5.0
if ref_speed > MAX_CMD_SPEED:
    scale = MAX_CMD_SPEED / ref_speed
    target_vel = tuple(v * scale for v in ref.velocity)
else:
    target_vel = ref.velocity
```

### Also update CSV telemetry (line 432-433) to log ref from lookahead, not removed raw ref.

## Expected behavior
- Trajectory reference stays close to the drone's actual position
- 0.3s lookahead provides gentle "pull" toward next waypoint
- 5 m/s velocity clamp prevents tilt saturation
- Drone should follow trajectory smoothly, reaching gates

## Success criterion
- At least 1 gate passed (vs iter 1's 0)
- avg_tracking_error < 1.0m
- OR survives past t=15s without crashing
- Peak roll < 60°

## Rollback criterion
- 0 gates AND crash before t=5s → revert
- avg_tracking_error > 3.0m → revert

## Risks
- 0.3s lookahead may be too short for fast segments (drone may stall)
- 5 m/s clamp may be too aggressive (race time increases)
- find_closest is O(n) per call — at 48 Hz with ~700 points, ~0.1ms overhead (acceptable)
