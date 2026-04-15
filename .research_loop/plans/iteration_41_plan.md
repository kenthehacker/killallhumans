# Iteration 41 Plan: Velocity-Corrected ILC

## Objective
Add velocity corrections to the ILC system to eliminate the position-velocity mismatch. Target: reduce avg tracking error from 0.150m toward 0.130-0.140m while maintaining race time ≤14.1s and 100% gate pass.

## Research Basis
- Schoellig 2012: ILC should correct feedforward inputs, not just positions
- Kunapuli 2025: feedforward is the most important single fix for geometric controllers
- Nam 2026: co-optimizing position and velocity profiles yields 20.7% improvement
- Wu 2024: model mismatch between ILC learning and execution causes residual error

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — `compute_ilc_offset_table()`
- After each ILC iteration, compute velocity offset from position offset: `vel_offset = np.gradient(cumulative_offset, dt, axis=0)`
- Apply velocity offset in the inner sim: `target_vel = ref.velocity + vel_offset[step]`
- Return BOTH position and velocity offset arrays (change return type)
- Also apply velocity offset inside the ILC inner sim so learning is consistent with execution

### 2. `scripts/benchmark.py` — simulation loop
- Unpack both position and velocity offsets from ILC
- Apply velocity offset: `target_vel = np.array(ref.velocity) + ilc_vel_offsets[step]`
- This ensures the GeometricTracker receives consistent position+velocity references

## Algorithm Changes

### In `compute_ilc_offset_table`:
```
After computing cumulative_offset at each ILC iteration:
  1. vel_offset = np.gradient(cumulative_offset[:actual_steps], dt, axis=0)
  2. In the inner sim, use: target_vel = ref.velocity + vel_offset[step]
  3. This feeds into the PD velocity error term: vel_err = target_vel - vel

Return: (cumulative_offset, vel_offset) tuple
```

### In `benchmark.py`:
```
ilc_result = compute_ilc_offset_table(...)
if ilc_result is not None:
    ilc_offsets, ilc_vel_offsets = ilc_result
else:
    ilc_offsets, ilc_vel_offsets = None, None

# In sim loop:
if ilc_vel_offsets is not None and step < len(ilc_vel_offsets):
    target_vel = target_vel + ilc_vel_offsets[step]
```

## Risk Assessment
- **Regression risk**: Moderate. The velocity correction changes the controller's velocity error term, which could destabilize tracking if corrections are too large.
- **Mitigation**: The velocity offsets are derivatives of Butterworth-filtered position offsets, so they're inherently smooth and bounded. At 0.35 Hz cutoff, max velocity offset ≈ 2π*0.35*0.15 ≈ 0.33 m/s (small relative to flight velocity ~5-10 m/s).
- **Basin switching risk**: NONE — this doesn't change the racing line or trajectory optimization.

## Rollback Criteria
- If avg error increases by >5% (0.158m), revert
- If race time increases by >0.3s (14.37s), revert
- If any gate error increases by >20%, revert

## Test Plan
1. Run unit tests after code changes
2. Run full benchmark
3. Compare per-gate errors before/after
4. Verify race time unchanged (velocity corrections should not affect timing)
