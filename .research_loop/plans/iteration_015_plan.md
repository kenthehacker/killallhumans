# Iteration 15 Plan — TOPP-RA Speed Retiming

## Objective
Replace the heuristic `_compress_times` with a TOPP-RA-style forward-backward speed retimer that uses actual polynomial curvature. Target: race time 14.62→13.5-14.0s while maintaining tracking error ≤0.30m.

## Research Basis
- **TOPPQuad** (Mao, IROS 2024): fix geometry, optimize speed profile → 40-50% faster
- **FBGA** (Piazza, RA-L 2025): forward-backward matches optimal control within 0.36%
- **TOPP-RA** (Pham & Pham, 2017): forward-backward via reachability analysis is robust and fast

## Files to Modify
- `planning/trajectory_optimizer.py` — replace `_compress_times` with `_topp_retime`

## Algorithm

### `_topp_retime(waypoints, segment_times, start_velocity, gates)`

```
1. Generate trajectory with current segment_times to get polynomial evaluation
2. For each segment i (0..N-1):
   a. Evaluate polynomial at 20 evenly-spaced time points
   b. Compute position derivatives: velocity v(t), acceleration a(t)
   c. Compute curvature at each point:
      κ(t) = |v × a| / |v|³
   d. Record: segment_length (arc-length), max_curvature, avg_curvature
3. For each segment, compute speed limit:
   v_max[i] = min(max_velocity, sqrt(a_centripetal_budget / max(max_curvature[i], 1e-6)))
4. Forward pass (from start to end):
   v_fwd[0] = min(start_speed, v_max[0])
   for i = 1..N-1:
     v_accel = sqrt(v_fwd[i-1]² + 2 * a_lon_max * segment_length[i-1])
     v_fwd[i] = min(v_max[i], v_accel)
5. Backward pass (from end to start):
   v_bwd[N-1] = v_max[N-1] (or end speed)
   for i = N-2..0:
     v_decel = sqrt(v_bwd[i+1]² + 2 * a_lon_max * segment_length[i])
     v_bwd[i] = min(v_bwd[i], v_decel)
6. Combine: v_opt[i] = min(v_fwd[i], v_bwd[i])
7. Compute new segment times: new_time[i] = segment_length[i] / max(v_opt[i], 0.5)
8. Apply safety margin: new_time[i] = max(new_time[i], old_time[i] * 0.70)
   (don't compress more than 30% per segment to avoid aggressive polynomial reshaping)
9. Return new_times
```

### Key Parameters
- `a_centripetal_budget`: 10.0 m/s² (conservative: max tilt 0.85 rad → g*tan(0.85) ≈ 11.4)
- `a_lon_max`: 8.0 m/s² (longitudinal acceleration/deceleration budget)
- `max_compression_ratio`: 0.70 (don't compress any segment below 70% of original)
- `curvature_samples`: 20 per segment
- `min_speed`: 2.0 m/s

## Risk Assessment
- **Tracking regression**: Changing segment times changes polynomial coefficients. Mitigated by 70% compression floor.
- **Gate miss**: If speed through turns is too high. Mitigated by using actual polynomial curvature.
- **Crash**: Unlikely — all thresholds currently met with margin.

## Rollback Criteria
- If avg tracking error > 0.35m (38% regression), revert
- If any gate is missed, revert
- If crash occurs, revert

## Test Plan
1. Run unit tests after implementation
2. Run full benchmark
3. Compare metrics: race time (must improve), avg error (must stay <0.35m), gate pass (must be 100%)
