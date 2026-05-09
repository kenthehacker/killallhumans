# Pre-Edit Review — Iteration 2: Closest-Point Tracking + Velocity Clamp

## Review (manual — Codex MCP tools not available)

### Plan Summary
Replace time-based `trajectory.sample(sim_time)` with `trajectory.find_closest(pos)` + 0.3s lookahead + 5 m/s velocity clamp.

### Failure Mode Analysis

1. **find_closest at trajectory start** — LOW severity
   - At t=0, drone is at start position. find_closest returns the first point (t=0). Lookahead = 0.3s. This is fine — gives a gentle initial target.

2. **find_closest near trajectory end** — LOW severity
   - When drone approaches final waypoint, closest.time + 0.3 may exceed total_time. Code uses `min(..., total_time)` — safely clamps to last point.

3. **find_closest when drone is far off trajectory** — MEDIUM severity
   - If drone drifts significantly, find_closest returns nearest point which may be behind the drone. Combined with 0.3s lookahead, this could cause back-tracking. Mitigation: the existing gate_fallback logic (line 400-411) handles post-trajectory scenarios. During trajectory, 0.3s lookahead should pull forward.

4. **Velocity clamp direction preservation** — LOW severity
   - Clamping scales all velocity components proportionally, preserving direction. No issue.

5. **CSV telemetry consistency** — LOW severity
   - `ref` variable now points to the lookahead sample, not the time-based sample. CSV columns ref_pos/ref_vel will reflect the lookahead reference. This is actually correct — we want to log what the controller is actually targeting.

6. **Performance of find_closest at 48 Hz** — LOW severity
   - ~700 trajectory points, vectorized numpy. Estimated <0.1ms per call. Negligible at 48 Hz.

7. **Interaction with should_slow_down()** — LOW severity
   - Slowdown multiplier (0.3x) still applies on top of clamped velocity. 5.0 * 0.3 = 1.5 m/s near gates. This is fine for approach.

### Verdict
No HIGH severity issues. Proceed with implementation.
