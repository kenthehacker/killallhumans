# Iteration 2 Report — Closest-Point Tracking + Velocity Clamp

## 1. Summary
Replaced time-based `trajectory.sample(sim_time)` with position-based `trajectory.find_closest(pos)` + 0.3s lookahead, and clamped commanded velocity to 5 m/s. This is the core fix for the root cause identified in iter 0: the min-snap polynomial's 16.8 m/s peaks caused immediate tilt saturation at t=0.15s. With this fix, the drone follows the trajectory through gate-1 at t=1.83s (previously unreachable), survives 35.2s (vs 8.8s), and passes 4 gates total via the gate_fallback mode after trajectory time expires.

## 2. What Changed
- `scripts/visual_demo.py:387-407` — Replaced `trajectory.sample(sim_time)` with `find_closest(pos)` + 0.3s lookahead + 5 m/s velocity clamp
- Commit: `0819bc8`

### Approach tried and reverted within this iteration
- Changed gate_fallback condition from `sim_time > total_time` to `closest.time >= total_time - 0.05` (position-based). This reduced gates from 4→1 because the drone never exited trajectory mode — the clamped speed means it can't complete the trajectory in real-time. Reverted; sim_time-based fallback is correct for now.

## 3. Metrics Before/After

| Metric | Iter 1 (baseline) | Iter 2 | Change |
|--------|--------------------|--------|--------|
| gates_passed | 0/12 | 4/12 | +4 gates |
| sim_time | 8.83s | 35.17s | +26.3s (+298%) |
| avg_tracking_error | 0.680m | 1.467m | +0.787m (note: iter1 crashed early) |
| max_tracking_error | 2.917m | 4.602m | +1.685m |
| p95_tracking_error | 2.347m | 3.524m | +1.177m |
| avg_loop_hz | 3398 | 3557 | +159 |
| crashed | yes (alt=0.04m) | yes (alt=0.06m) | still crashes |
| peak_ref_speed | 16.83 m/s | 16.84 m/s | unchanged (polynomial) |
| **peak_target_speed** | **16.83 m/s** | **5.00 m/s** | **-11.83 m/s (CLAMP WORKS)** |
| peak_roll | 178.37° | 174.79° | still tumbles (during fallback) |
| peak_pitch | 57.36° | 83.03° | worse (longer flight) |
| first_tilt_>30° | 0.40s | 0.27s | earlier (but drone recovers) |
| target_jumps_>1m | 0 | 9 | new (fallback transitions) |
| target_source | 100% trajectory | 41% traj, 59% fallback | split |

Note: iter 1's lower avg_tracking_error (0.680m) is an artifact of crashing at 8.8s before reaching high-error trajectory segments.

## 4. Root Cause of Remaining Error

### 34m target jump at trajectory→fallback transition (t=14.27s)
When `sim_time` exceeds the trajectory's 14.3s duration, the fallback targets gate-2 at (18,4,2.2) while the drone is at (52.7,3.2,3.8). This 34m jump causes violent maneuvering. The drone recovers and reaches gates 2-4, but the accumulated instability leads to a tumble and crash at t=35.2s during approach to gate-5.

### Why trajectory tracking alone isn't enough
The trajectory is planned for 14.3s at 10 m/s max speed, but we clamp to 5 m/s. The drone only traverses ~40% of the trajectory by t=14.3s. The fallback then guides it through gates 2-4 using direct-to-gate navigation, which works but introduces large target jumps.

### Crash mechanism
During gate_fallback toward gate-5 (x≈46), the drone enters a tumble (roll reaching -159°) at t=34.8s and loses altitude progressively to 0.06m.

## 5. Next Iteration's Recommended Bottleneck

**`polynomial_velocity_clamp_in_trajectory_optimizer`** (priority 2 in backlog)

Clamp the min-snap polynomial velocities to ≤5 m/s inside `trajectory_optimizer.py::_generate_trajectory`. This would allow reverting to `trajectory.sample(sim_time)` because the reference velocities would be Crazyflie-trackable. The trajectory's total_time would increase from 14.3s to ~28s+, giving the drone enough time to reach all 12 gates via the polynomial path without needing fallback.

**Alternative**: Increase the 0.3s lookahead to adaptive (proportional to closest-point velocity magnitude) and smooth the fallback transition to avoid 34m jumps.

## 6. Risks / Open Questions
- The velocity clamp in trajectory_optimizer would change the trajectory for ALL consumers (benchmark, competition). Need to verify no regressions.
- The fallback gate_fallback mode has no velocity clamp — the `min(dist * 2, 5.0)` at line 419 helps but the deceleration profile is aggressive.
- The 0.3s lookahead is a tuning parameter. For slow segments it may overshoot; for fast segments it may be too conservative. Iter 3+ may need adaptive lookahead.
- Reproducibility: runs with this code produce identical results (gate timings and metrics match across v1 and v3 runs).

## 7. Failed Approaches Added to State
- `position_based_fallback_condition`: Changing `sim_time > total_time` to `closest.time >= total_time - 0.05` caused the drone to stay in trajectory mode indefinitely (never reaches end because of speed clamp). Reduced gates from 4→1. Reverted.
