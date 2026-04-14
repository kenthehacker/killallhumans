# Iteration 22 Plan — Sim-Based Racing Line Selection

## Objective
Replace proxy-objective selection (path_length + curvature²) with kinematic-sim-based selection for the racing line multi-start optimizer. Target: recover gate-3 tracking error from 0.374m toward ~0.34m.

## Research Basis
- **AERO-MPPI (Chen 2026)**: Ensemble optimizers with re-rollout selection under common cost
- **T-MPC (de Groot 2024)**: Parallel planners with fallback guarantee (Theorem 2)
- **BO Racing Line (Jain 2020)**: Black-box sim evaluation as oracle for trajectory selection
- **TACO (Sanghvi 2025)**: Trajectory-aware optimization reduces error 32%
- **Multi-obj PID (Vaiuso 2025)**: Closed-loop sim evaluation beats proxy by 42.7%

## Files to Modify
1. **`planning/racing_line.py`** — Main changes

## Algorithm Changes

### Current flow (in `RacingLineOptimizer.optimize()`):
```
10 L-BFGS starts → 10 results → select by result.fun (proxy objective)
```

### New flow:
```
10 L-BFGS starts → 10 results → for each:
  1. Build gate waypoints from offsets
  2. Build full trajectory (TrajectoryOptimizer.optimize())
  3. Run lightweight kinematic sim (inline PD controller, ~20 lines)
  4. Measure avg_tracking_error and per-gate errors
→ select by composite score: 0.7 * avg_error + 0.3 * worst_gate_error
→ tie-break by race time (faster wins if errors within 1%)
→ fallback to L-BFGS selection if sim fails
```

### Kinematic sim evaluator (inline in racing_line.py):
- PD controller: kp_xy=6, kd_xy=4, kp_z=8, kd_z=5, ff_accel=0.4
- Physics: drag=0.5, max_accel=15, max_speed=15
- dt=0.02 (coarser than benchmark 0.01 for speed, sufficient for ranking)
- Measures: avg error, per-gate error (using gate pass detection)
- No external dependencies (self-contained, avoids control/ import)

### Key design decisions:
1. dt=0.02 (not 0.01): 2x faster evaluation, sufficient for ranking candidates
2. Inline PD controller: avoids cross-package dependency (planning→control)
3. Composite score: weighted avg + worst gate prevents "globally mediocre" selection
4. Always include zero-init as candidate: T-MPC Theorem 2 fallback guarantee

## Risk Assessment
- **Regression risk**: LOW — T-MPC Theorem 2 guarantees no regression (zero-init fallback)
- **Speed regression**: Possible small race time increase if sim selects a smoother racing line. Mitigated by composite score (doesn't purely minimize error; considers speed).
- **Computational cost**: ~5-7s for 10 evaluations. Acceptable for offline planning.
- **Unit test regression**: None expected — racing_line unit test checks that optimization runs without error, not specific values.

## Rollback Criteria
- If avg tracking error increases by >5%: revert
- If race time increases by >3%: revert
- If any gate error increases by >20%: revert

## Test Plan
1. Run unit tests after code change
2. Run full benchmark
3. Compare per-gate errors, avg error, race time vs baseline
