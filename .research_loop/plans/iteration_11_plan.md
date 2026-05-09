# Iteration 11 — Implementation Plan: Reference-Velocity Drag Feedforward

## Objective
Reduce helix tracking error (gates 7-12 avg 0.641m) by adding reference-velocity drag feedforward to the geometric controller. Target: avg error 0.481m → 0.35m while maintaining 13.34s race time.

## Research Basis
- Tal & Karaman 2018: feedforward is the most important controller fix
- L1Quad 2025: drag compensation achieves 5x RMSE reduction
- Mathematical analysis: drag forcing dominates error budget (5.0 vs 3.0 from ff deficiency)
- Key differentiation from iter 9: feedforward on ref_vel (preserves damping) vs current vel (kills damping)

## Files to Modify

### 1. `control/mpc_tracker.py` — TrackerConfig + GeometricTracker
Changes:
- Add `velocity_feedforward: float = 0.0` to TrackerConfig dataclass
- In GeometricTracker.track(), add `c.velocity_feedforward * ref_vel[i]` to each axis of accel_des
- Update docstring to explain the approach

### 2. `scripts/benchmark.py` — Synthetic sim TrackerConfig
Changes:
- Update the TrackerConfig instantiation in run_synthetic_benchmark() to pass the new parameters
- Parameters: velocity_feedforward=0.5, feedforward_accel=0.7

## Algorithm Changes (Pseudocode)

Current:
```python
accel_des = [
    kp_xy * ep[0] + kd_xy * ev[0] + ff * ref_acc[0],
    kp_xy * ep[1] + kd_xy * ev[1] + ff * ref_acc[1],
    kp_z * ep[2] + kd_z * ev[2] + ff * ref_acc[2],
]
```

New:
```python
accel_des = [
    kp_xy * ep[0] + kd_xy * ev[0] + ff * ref_acc[0] + vff * ref_vel[0],
    kp_xy * ep[1] + kd_xy * ev[1] + ff * ref_acc[1] + vff * ref_vel[1],
    kp_z * ep[2] + kd_z * ev[2] + ff * ref_acc[2] + vff * ref_vel[2],
]
```

Where vff = velocity_feedforward (new parameter, set to 0.5 = sim drag coeff).

## Risk Assessment
- **Oscillation near trajectory boundaries**: If ref_vel has discontinuities at segment junctions, the velocity feedforward could cause spikes. Mitigated by: min-snap trajectories are C4 continuous → ref_vel is smooth.
- **Overshoot at end of trajectory**: When trajectory ends and drone switches to gate-seeking, ref_vel drops to zero → velocity feedforward drops to zero. This is correct behavior (no drag compensation needed when not following trajectory).
- **Gate pass rate regression**: Unlikely since this only affects tracking accuracy, not trajectory geometry. Monitor carefully.

## Rollback Criteria
- If avg tracking error > 0.5m (threshold): revert
- If gate pass rate < 100%: revert immediately
- If race time increases > 1s: investigate (shouldn't change since trajectory is unchanged)

## Test Plan
1. Run unit tests first (`--mode unit`) — should pass unchanged
2. Run synthetic benchmark — compare per-gate errors
3. If any regression, try intermediate values (vff=0.3, ff=0.5)
4. Try multiple parameter combinations if first attempt doesn't work
