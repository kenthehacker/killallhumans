# Iteration 17 Plan — Remove FOV Relaxation Post-Processing

## Objective
Reduce race time by removing the redundant `_relax_for_fov` post-processing stage from the trajectory optimizer. Target: race time 13.95s → ~13.5s with no accuracy regression.

## Research Basis
- **ETH 2026** (arXiv:2603.04305): FOV soft constraints add +8.1% when integrated in optimizer. Post-processing is redundant when optimizer already has FOV penalty.
- **MonoRace 2026** (arXiv:2601.15222): A2RL competition winner uses NO post-processing FOV relaxation. Perception handled via adaptive cropping + EKF.
- **Drift-Corrected VIO 2025** (arXiv:2512.20475): Heading-based FOV control adds +0% race time. Position trajectory doesn't need slowing for FOV.
- **FOV CBF 2025** (arXiv:2502.01009): Advocates embedding FOV in optimizer, eliminating post-processing.
- **PA-MPPI 2025** (arXiv:2509.14978): Perception cost integrated directly in sampling loop.
- **Mastering Diverse Tracks 2025** (arXiv:2512.09571): Perception reward baked into RL reward function.

## Files to Modify
1. `planning/trajectory_optimizer.py` — **ONLY FILE MODIFIED**

## Algorithm Changes

### Change: Remove FOV relaxation call from generate()

Current pipeline (lines ~267-294):
```
segment_times = self._inflate_sharp_turns(waypoints, segment_times, gates)
points = self._generate_trajectory(waypoints, segment_times, start_velocity, gates)
fov_penalty = self.add_fov_constraints(points, gates)     # ← REMOVE
if fov_penalty > 1.0:                                      # ← REMOVE
    segment_times = self._relax_for_fov(...)               # ← REMOVE
    points = self._generate_trajectory(...)                 # ← REMOVE
segment_times = self._topp_retime(waypoints, segment_times, start_velocity, gates)
points = self._generate_trajectory(waypoints, segment_times, start_velocity, gates)
```

New pipeline:
```
segment_times = self._inflate_sharp_turns(waypoints, segment_times, gates)
points = self._generate_trajectory(waypoints, segment_times, start_velocity, gates)
# FOV awareness handled by L-BFGS penalty (weight=10) — no post-processing needed
segment_times = self._topp_retime(waypoints, segment_times, start_velocity, gates)
points = self._generate_trajectory(waypoints, segment_times, start_velocity, gates)
```

Note: We also remove the intermediate `_generate_trajectory` call that was only needed to evaluate FOV penalty. This saves one trajectory generation.

### What stays
- `_relax_for_fov` method definition stays in code (not deleted)
- `add_fov_constraints` method stays in code (not deleted)
- L-BFGS FOV penalty (weight=10) stays in optimizer objective
- `FOVConfig` class stays

## Risk Assessment
- **Risk**: Some turn segments might have high FOV violations without post-processing
- **Mitigation**: L-BFGS penalty (weight=10) already accounts for FOV. The post-processing was documented as "safety net, not primary mechanism" (iter 14 comments).
- **Risk**: TOPP retimer might compress segments too aggressively without FOV inflation
- **Mitigation**: TOPP has velocity limits from curvature that bound compression
- **Worst case**: If metrics regress, revert with `git checkout -- .`

## Rollback Criteria
- If avg tracking error increases by > 0.02m (from 0.232m baseline)
- If any gate error increases by > 0.05m
- If race time does NOT improve (the whole point is speed recovery)
- If gate pass rate drops below 100%

## Test Plan
1. Run unit tests first (`--mode unit`)
2. Run full benchmark (`--mode full`)
3. Compare all metrics vs baseline
4. Check per-gate breakdown for regressions
