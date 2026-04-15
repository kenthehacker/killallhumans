# Iteration 40 Plan: Increase max_tilt_rad to Reduce ILC-Controller Saturation Mismatch

## Objective
Reduce average tracking error by increasing the controller's maximum tilt angle from 0.85 rad (49°) toward ~0.98 rad (56°). This reduces tilt saturation at tight gates (2, 3, 7) and better aligns the benchmark controller's dynamics with the ILC inner sim's unconstrained model.

**Target metric**: avg_tracking_error improvement ≥ 1% (from 0.1501m baseline)
**Secondary target**: gate-2 error reduction (currently worst at 0.214m)

## Research Basis
- **NGTC (Pries 2025)**: Standard max_tilt β=56° (0.977 rad) for aggressive flight. DFBC degrades 10x at saturation.
- **LoL-NMPC (Gupta 2025)**: 22-29% improvement from modeling actuator saturation. 3x prediction error reduction.
- **ILC Mismatch (Wu 2024)**: Plant-model mismatch in ILC causes suboptimal convergence.

## Files to Modify
1. **`control/mpc_tracker.py` line 72**: Change `max_tilt_rad: float = 0.85` default
2. **`planning/trajectory_optimizer.py` line 36**: Update `max_tilt_angle` documentation parameter
3. **`scripts/benchmark.py`**: The TrackerConfig is created without explicit max_tilt_rad, so it uses the default from TrackerConfig. Only need to change the default.

## Algorithm Changes
No algorithmic changes. Single parameter change:
```python
# control/mpc_tracker.py line 72
max_tilt_rad: float = 0.85  # BEFORE
max_tilt_rad: float = X.XX  # AFTER (best from sweep)
```

## Implementation Plan
1. Read current TrackerConfig to confirm max_tilt_rad default location
2. Sweep max_tilt_rad values: 0.90, 0.95, 1.00, 1.05
3. For each value, modify TrackerConfig default and run benchmark
4. Select the value with best avg_tracking_error (no regression on any gate > 20%)
5. Commit the best value

## Risk Assessment
- **Basin switching**: LOW — this changes the controller, not the trajectory. ILC and racing line are unchanged.
- **Velocity instability**: MODERATE — higher tilt → higher velocities → more drag. Monitor max velocity.
- **ILC coupling**: LOW — ILC corrections are position offsets, not acceleration commands. Higher tilt makes the controller track them better.
- **Oscillation**: LOW — damping ratio ζ=1.13 is well overdamped.

## Rollback Criteria
- Revert if avg_tracking_error increases by > 5%
- Revert if any single gate error increases by > 20%
- Revert if crash occurs
- Revert if race time increases by > 5%

## Test Plan
1. Run unit tests after parameter change (should pass — TrackerConfig default change)
2. Run full benchmark for each sweep value
3. Compare per-gate errors to identify which gates improve/worsen
4. Select best value based on avg error with no gate regression > 20%
