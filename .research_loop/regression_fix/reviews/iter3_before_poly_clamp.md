# Red-Team Review (Before Edit) — Iteration 3

## Plan Summary
1. Add post-hoc velocity clamp in `_generate_trajectory` (cap to max_velocity, scale accel/jerk)
2. Extend fallback trigger from 1x to 3x total_time
3. Use MAX_CMD_SPEED constant in fallback velocity

## Failure Mode Analysis

### FM1: Velocity-position inconsistency (severity: LOW)
**Issue**: After clamping, velocity is no longer the exact time-derivative of position.
**Impact**: Controller uses position as primary target. Velocity is feedforward only. Slightly reduced feedforward won't cause instability — it's equivalent to a slower reference.
**Verdict**: Acceptable. No fix needed.

### FM2: Acceleration scaling may be too conservative (severity: LOW)
**Issue**: Scaling accel by `scale = max_vel/speed` instead of `scale²` means the acceleration is over-reported at clamped points. The true time-rescaled acceleration would be smaller.
**Impact**: Controller gets slightly larger acceleration feedforward than the clamped velocity implies. This makes the controller slightly more aggressive (anticipating faster changes). For a PD controller with feedforward, this is a minor effect — the position error dominates.
**Verdict**: Acceptable. Using `scale` instead of `scale²` is conservative in the right direction (more feedforward, not less).

### FM3: Extended trajectory mode endpoint behavior (severity: MEDIUM)
**Issue**: When drone reaches the end of the trajectory (find_closest returns the last few points), the 0.3s lookahead returns the endpoint. Drone would try to reach the endpoint and hover there, ignoring remaining gates.
**Impact**: If the drone reaches the trajectory endpoint before all gates are passed, it would stop navigating to gates. However, the trajectory goes through ALL 12 gates plus a virtual finish, so this should not happen before all gates.
**Mitigation**: If the drone gets stuck near endpoint, the 3x fallback timer (42.9s) will eventually trigger fallback mode for remaining gates.
**Verdict**: Low risk. Monitor in telemetry for "stuck at endpoint" behavior.

### FM4: Fallback at 3x total_time may be too late (severity: LOW)
**Issue**: 3x total_time = 42.9s. If the drone tumbles before this, we lose gates.
**Impact**: The current crash at 35.2s is in fallback mode. In the new code, at 35.2s the drone would be in trajectory mode instead. Trajectory mode is smoother, so less likely to tumble.
**Verdict**: Net improvement expected. The 3x multiplier is a safety margin, not the critical path.

### FM5: Does clamping introduce discontinuities between segments? (severity: LOW)
**Issue**: Adjacent segments are C3 continuous at boundaries. After clamping, velocity at segment boundaries might have a jump if one side is clamped and the other isn't.
**Impact**: At segment boundaries, both sides share the same velocity value (continuity constraint in min-snap). If the boundary velocity is already ≤ max_velocity (it is, because boundary velocities are computed with max_velocity cap at line 1509), then clamping doesn't change boundary values. Discontinuities only happen at mid-segment peaks, which are ALREADY discontinuous in the higher derivatives (jerk/snap). The clamping smooths these peaks.
**Verdict**: No additional discontinuity risk.

### FM6: benchmark.py regression (severity: MEDIUM)
**Issue**: Polynomial velocity clamp affects ALL trajectories, including those used by benchmark.py.
**Impact**: benchmark.py uses max_velocity=10.0 for unit-test trajectories (line 121) and max_velocity=15.0 for simulation trajectories (lines 296, 667). Both will have their polynomial velocities clamped to their respective limits. This may change ILC convergence (ILC uses trajectory.sample() which returns clamped velocities).
**Mitigation**: Run smoke_test.py to verify unit tests pass. The clamp makes trajectories more realistic, not less.
**Verdict**: Acceptable risk. Run benchmark to verify.

## Overall Assessment
**No HIGH severity issues found.** Proceed with implementation.
The main risk (FM3) is mitigated by the 3x fallback timer and the trajectory covering all gates.
