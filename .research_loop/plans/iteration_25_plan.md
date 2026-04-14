# Iteration 25 Plan — Offline ILC Trajectory Pre-Compensation

## Objective
Reduce helix section tracking error floor (gates 7-10: 0.24-0.33m) by 30-50% using offline Iterative Learning Control. Target: avg error 0.211→0.17m, gate-7 error 0.327→0.22m. Race time should be unchanged (ILC corrects position, not timing).

## Research Basis
- **Schoellig et al. 2012**: P-type ILC with feedforward correction achieves 87% error reduction in 3-5 iterations on real quadrotors. The systematic tracking error in our sim is the exact target.
- **Spatial ILC (Lv 2023)**: Validates ILC convergence for racing in 7-20 iterations.
- **ILMPC (Zhao 2025)**: Confirms iterative trajectory improvement converges from any reasonable initialization.

## Files to Modify

### 1. `planning/trajectory_optimizer.py`
Add a new method `apply_ilc_correction(trajectory, sim_fn, ...)` to the `TrajectoryOptimizer` class (or as a standalone function) that:
1. Takes a `RaceTrajectory` and a kinematic sim function
2. Runs the sim to get actual positions
3. Computes position error at each timestep
4. Smooths the error with a Gaussian kernel
5. Applies the correction: `new_pos = old_pos + alpha * smoothed_error`
6. Recomputes velocity and acceleration via finite differences
7. Returns a corrected `RaceTrajectory`
8. Iterates until convergence or max iterations

### 2. `scripts/benchmark.py`
After trajectory generation (line ~294), add the ILC correction loop before running the main sim. The flow becomes:
1. Generate trajectory (existing)
2. Apply ILC correction (new — uses a mini kinematic sim identical to `_kinematic_eval`)
3. Run benchmark sim with corrected trajectory (existing)

## Algorithm Details

### ILC Update Rule (P-type, position domain)
```python
for iteration in range(max_ilc_iterations):
    actual_positions = run_kinematic_sim(trajectory)
    for k in range(len(trajectory.points)):
        ref_pos = trajectory.points[k].position
        act_pos = actual_positions[k]
        error = (ref_pos[i] - act_pos[i] for i in range(3))
        # Accumulate error for smoothing

    # Smooth error with Gaussian kernel (sigma = 5 timesteps)
    smoothed_error = gaussian_filter1d(error, sigma=5)

    # Clip correction magnitude
    smoothed_error = clip(smoothed_error, max=0.3)

    # Apply correction
    new_positions = old_positions + alpha * smoothed_error

    # Recompute velocity and acceleration
    new_velocity = finite_diff(new_positions, dt)
    new_acceleration = finite_diff(new_velocity, dt)

    # Build new trajectory
    trajectory = RaceTrajectory(new_points, ...)

    # Check convergence
    if improvement < 0.01: break
```

### Parameters
- alpha = 0.5 (learning rate — aggressive for deterministic sim)
- max_iterations = 5
- smoothing sigma = 5 timesteps (at dt=0.02, this is 100ms)
- max_correction = 0.3m per axis

## Risk Assessment
- **LOW RISK**: ILC is purely additive post-processing
- **Rollback**: If any metric regresses >5%, revert. The original trajectory is untouched.
- **Gate passage**: Correction magnitude cap prevents trajectory from leaving gate apertures
- **Race time**: Should be unchanged since we correct position, not timing

## Test Plan
1. Run unit tests (verify no breakage)
2. Run full benchmark with ILC
3. Compare per-gate errors before/after
4. Verify gate pass rate = 100%
5. Verify race time within ±0.5s of baseline
