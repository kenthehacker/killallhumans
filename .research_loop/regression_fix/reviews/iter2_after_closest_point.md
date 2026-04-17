# Post-Edit Review — Iteration 2: Closest-Point Tracking + Velocity Clamp

## Diff Analysis

### Changes (scripts/visual_demo.py)
1. **Removed**: `ref = self.trajectory.sample(sim_time)` — time-based sampling (root cause)
2. **Added**: `lookahead_time = min(closest.time + 0.3, self.trajectory.total_time)` — position-anchored lookahead
3. **Added**: `ref = self.trajectory.sample(lookahead_time)` — sample at lookahead time
4. **Added**: Velocity clamp block — caps `ref_speed` at 5.0 m/s with direction-preserving scaling
5. **Preserved**: `closest = self.trajectory.find_closest(pos)` for tracking error computation (was already there)
6. **Preserved**: Gate fallback logic, slowdown logic, CSV telemetry unchanged

### Correctness Check
- `closest.time` is a float (from `TrajectoryPoint.time`) — confirmed in dataclass definition
- `self.trajectory.total_time` is a float — confirmed in `RaceTrajectory` dataclass
- `min()` correctly clamps lookahead to trajectory end
- Velocity clamp preserves direction (proportional scaling) — correct
- `target_vel` is assigned in all code paths (clamp if/else covers both) — no undefined variable risk
- CSV telemetry logs `ref.position` and `ref.velocity` which now reflect the lookahead reference — consistent
- CSV also logs `target_pos`, `target_vel` which may differ from `ref` after gate_fallback or slowdown — correct

### Risk Assessment
- **No HIGH severity issues found**
- MEDIUM: When drone is significantly behind trajectory, find_closest may snap to a point far along the curve if trajectory has a hairpin turn. For race_01 (simple oval-ish), this is unlikely. Monitor in telemetry.
- LOW: 0.3s lookahead is conservative. May need tuning in future iters for faster segments.

### Verdict
Proceed to benchmark.
