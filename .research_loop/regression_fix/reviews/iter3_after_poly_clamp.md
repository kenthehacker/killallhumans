# Critic Review (After Edit) — Iteration 3

## Diff Summary
1. `planning/trajectory_optimizer.py` (+13 lines): Post-hoc velocity clamp in `_generate_trajectory` — caps polynomial velocity magnitude to `max_velocity`, scales accel/jerk by same ratio.
2. `scripts/visual_demo.py` (+3 lines, -2 lines): Extend fallback trigger from `total_time` to `total_time * 3.0`. Use `MAX_CMD_SPEED` constant in fallback velocity.

## Quality Assessment

### Change 1: Polynomial velocity clamp
- **Correct?** YES. The clamp runs after all 3 axes are computed for each sample point, so the norm is computed correctly across all axes. The scale factor preserves velocity direction.
- **Does scaling vel AND accel preserve consistency?** Approximately. The velocity is no longer the exact derivative of position, but for feedforward-based tracking this is acceptable. The acceleration scaling by `scale` (not `scale²`) is conservative — it slightly over-estimates feedforward, which is safer than under-estimating.
- **Discontinuities?** NO. Boundary velocities are already constrained by `max_velocity` at line 1509. The clamp only affects mid-segment overshoots, which transition smoothly back to boundary values. The `np.linalg.norm` + scaling is a smooth operation.
- **Performance?** Negligible. Adds one norm computation per sample point per segment. The trajectory has ~1500 total points; the added computation is microseconds.
- **Type safety?** YES. The clamp modifies numpy arrays in-place before they're converted to tuples via `tuple(velocities[j])` at line 1569. TrajectoryPoint types are preserved.

### Change 2: Fallback trigger extension
- **Correct?** YES. `total_time * 3.0` gives 42.9s for a 14.3s trajectory. The drone at ~3.5 m/s average needs ~40s to traverse the trajectory. The 3x multiplier provides margin.
- **Edge cases?** If the drone reaches the trajectory endpoint before 42.9s, `find_closest` returns the endpoint, and lookahead returns the endpoint too. The drone would hover near the endpoint until 42.9s, then fallback activates. This is acceptable — it's better than crashing at 35.2s in fallback mode.
- **Impact on non-visual_demo consumers?** None. This change is only in visual_demo.py.

### Change 3: MAX_CMD_SPEED constant
- **Correct?** YES. `MAX_CMD_SPEED = 5.0` is defined at line 401. The fallback previously hardcoded `5.0`; now it uses the same constant as trajectory-mode velocity clamp.

## Risks Checked
- [x] No physics simulation files modified
- [x] No safety checks disabled
- [x] Types preserved (numpy arrays → tuples)
- [x] Boundary conditions unaffected (polynomial boundary velocities ≤ max_velocity)
- [x] No new dependencies added
- [x] Changes are minimal and targeted

## Severity Assessment
**No HIGH severity issues.** All changes are LOW severity.

## Verdict
**APPROVE** — proceed to benchmark and visual_demo testing.
