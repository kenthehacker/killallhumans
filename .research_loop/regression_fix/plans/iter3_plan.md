# Iteration 3 Plan: Polynomial Velocity Clamp + Fallback Time Extension

## Summary
Two changes to enable the drone to follow the full trajectory at Crazyflie-trackable speeds:
1. Clamp polynomial velocities in `_generate_trajectory` to ≤ max_velocity
2. Extend fallback trigger from `total_time` to `total_time * 3.0` so the drone has time to follow the clamped trajectory through all 12 gates

## Exact Changes

### File 1: `planning/trajectory_optimizer.py` — `_generate_trajectory` (line ~1544)
After the `for axis in range(3)` loop that computes positions/velocities/accelerations/jerks,
add velocity clamping before TrajectoryPoint creation:

```python
# Clamp polynomial velocity magnitudes to max_velocity
max_vel = self.constraints.max_velocity
for j in range(n_samples):
    speed = float(np.linalg.norm(velocities[j]))
    if speed > max_vel:
        scale = max_vel / speed
        velocities[j] *= scale
        accelerations[j] *= scale
        jerks[j] *= scale
```

### File 2: `scripts/visual_demo.py` — fallback trigger (line 410)
Change:
```python
if sim_time > self.trajectory.total_time and not self.sequencer.is_complete:
```
To:
```python
# Drone flies at MAX_CMD_SPEED (~5 m/s) while trajectory planned for ~10 m/s.
# Give 3x total_time for the slower drone to complete the trajectory.
if sim_time > self.trajectory.total_time * 3.0 and not self.sequencer.is_complete:
```

### File 2: `scripts/visual_demo.py` — fallback velocity (line 419)
Change:
```python
target_vel = tuple(direction / dist * min(dist * 2, 5.0))
```
To:
```python
target_vel = tuple(direction / dist * min(dist * 2, MAX_CMD_SPEED))
```

## Expected New Behavior
- Trajectory polynomial velocities capped at 10 m/s (was 16.84 m/s)
- Drone stays in trajectory mode until ~42.9s (was 14.3s)
- More gates passed via smooth polynomial tracking instead of jerky fallback
- Fallback only activates for gates remaining after trajectory completion

## Success Criteria
- ≥ 6/12 gates passed (currently 4)
- Survival past 40s (currently 35.2s crash)
- Peak ref_speed ≤ 10 m/s (currently 16.84)

## Rollback Criteria
- Gates passed decreased (< 4)
- Crash earlier than 35s
- Tracking error increased >30% (currently 1.467m avg)

## Risks
1. **Velocity-position inconsistency**: Clamped velocity is no longer exact derivative of position. Acceptable — controller uses position as primary reference.
2. **Extended trajectory mode at trajectory end**: When find_closest returns points near trajectory end, lookahead returns endpoint. Drone might oscillate near endpoint. Mitigated by fallback kicking in at 3x total_time.
3. **Benchmark regression**: Polynomial velocity clamp affects ALL trajectory consumers. benchmark.py uses max_velocity=10.0 for some trajectories and 15.0 for others. Both should still work (clamp enforces their respective limits).
