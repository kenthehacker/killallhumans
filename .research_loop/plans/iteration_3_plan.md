# Iteration 3 Plan: Gate-12 Trajectory Extension + Controller Gains

## Objective
- Reduce gate-12 tracking error from 0.694m to ~0.3m by eliminating the gate-seeking fallback
- Reduce turn gate (3, 4, 7) tracking error by ~0.05m each via controller gains
- Target: avg tracking error 0.333m → ~0.28m

## Research Basis
- "On Your Own" (Romero 2025, arxiv:2510.13644): dual waypoints per gate at ±0.4m along gate normal
- TOGT Planner (Qin 2024, arxiv:2309.06837): gates as regions, not points
- "Leveling the Playing Field" (Kunapuli 2025): geometric controller gain tuning

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — TrajectoryOptimizer.optimize()
**Change**: After building waypoints from gates, append a virtual finish waypoint 2.0m past the last gate along its normal direction.

**Pseudocode**:
```python
# In optimize(), after building waypoints list:
if gates:
    last_gate = gates[-1]
    normal = np.array(last_gate.normal)
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 0.1:
        normal = normal / norm_mag
    else:
        # Fallback: use direction from second-to-last to last gate
        normal = waypoints[-1] - waypoints[-2]
        normal = normal / np.linalg.norm(normal)
    finish_wp = waypoints[-1] + normal * 2.0
    waypoints.append(finish_wp)
```

Also need to ensure _generate_trajectory handles the extra segment correctly — the last segment should end with low velocity (deceleration into finish).

### 2. `scripts/benchmark.py` — TrackerConfig gains
**Change**: Increase kp_xy from 4.0 to 5.0, kd_xy from 3.0 to 3.5.

```python
tracker = GeometricTracker(TrackerConfig(
    kp_xy=5.0, kd_xy=3.5,  # was 4.0, 3.0
    kp_z=6.0, kd_z=4.0,
    ...
))
```

## Risk Assessment
- **Low risk**: Virtual waypoint is past the course, so trajectory still passes through all real gates
- **Low risk**: L-BFGS optimizer will allocate time for the extra segment automatically
- **Low risk**: Controller gains increase is conservative (25% kp, 17% kd)
- **Monitoring**: Watch for oscillation or max tracking error increase

## Rollback Criteria
- If avg tracking error increases by >0.05m: revert
- If max tracking error increases by >0.2m: revert
- If any gate pass rate drops below 100%: revert

## Test Plan
1. Run unit tests after trajectory_optimizer.py change
2. Run full benchmark and compare all metrics
