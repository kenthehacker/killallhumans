# Iteration 7 Plan — Centripetal Acceleration Feasibility Check

## Objective
Reduce gate-3/gate-4 S-turn tracking error (0.661m/0.598m → target 0.45m/0.40m) by extending the post-optimization time inflation to use centripetal acceleration estimates instead of turn angle thresholds alone. Expected: avg error 0.336→0.30m.

## Research Basis
- **TOPPQuad** (Mao, IROS 2024): Dynamic feasibility requires checking per-motor thrust against speed AND curvature. Centripetal acceleration a_c = v²κ is the binding constraint at moderate turns with high speed.
- **Alternating Peak** (de Vries, ECC 2024): Peak constraint violation ratio (kappa) per segment is the correct metric for time inflation — inflate proportional to the violation.
- **TACO** (Sanghvi, 2025): Trajectory parameters should adapt to local characteristics.
- **Teissing** (RA-L 2024): Boundary velocity optimization shows speed at turn entry is the key variable.

## Files to Modify
1. **`planning/trajectory_optimizer.py`** — Modify `_inflate_sharp_turns()` to add centripetal acceleration check

## Algorithm Changes

### Current `_inflate_sharp_turns()` logic:
```
For each gate gi (1 to n-2):
  Compute turn_angle from gate_centers[gi-1], gate_centers[gi], gate_centers[gi+1]
  If turn_angle > 1.05 rad (60°):
    severity = (turn_angle - 1.05) / (π/2 - 1.05)
    inflate = 1.0 + 0.35 * severity  (range: 1.25x to 1.60x)
    Inflate seg_entry and seg_through by inflate factor
```

### New logic — add centripetal acceleration check:
```
For each gate gi (1 to n-2):
  Compute turn_angle from gate_centers
  Compute approach_distance = norm(gate_centers[gi] - gate_centers[gi-1])

  # Estimate average speed on approach segment (from L-BFGS times)
  # seg_entry = 2*gi is the segment ending at gate entry waypoint
  # The approach segments are seg_entry-1 and seg_entry
  approach_dist_wp = sum of distances for approach waypoint segments
  approach_time = sum of times for approach waypoint segments
  avg_speed = approach_dist_wp / approach_time

  # Centripetal acceleration estimate
  curvature = turn_angle / approach_distance  # 1/radius approximation
  a_centripetal = avg_speed² * curvature

  # Check BOTH conditions:
  # 1. Original angle-based inflation (>60°) — unchanged
  # 2. NEW: centripetal acceleration inflation (a_c > threshold)

  a_threshold = 8.0  # m/s² — ~80% of controller tracking bandwidth

  If turn_angle > 1.05:
    # Original sharp-turn inflation
    severity = (turn_angle - 1.05) / (π/2 - 1.05)
    inflate = 1.0 + 0.35 * severity
  elif a_centripetal > a_threshold:
    # NEW: speed-curvature inflation for moderate turns at high speed
    severity = (a_centripetal - a_threshold) / a_threshold
    severity = min(severity, 1.0)
    inflate = 1.0 + 0.25 * severity  (range: 1.0x to 1.25x)
  else:
    continue  # no inflation needed

  Inflate seg_entry and seg_through
```

### Parameter rationale:
- `a_threshold = 8.0 m/s²`: The sim's max_accel is 15 m/s², and the PD controller with gains kp=6, kd=4 has a ~0.5s settling time. At gate-3: speed ≈ 10 m/s, curvature ≈ 0.84/11.7 = 0.072 rad/m → a_c ≈ 7.2 m/s². At gate-4: speed ≈ 9 m/s, curvature ≈ 0.66/10.5 = 0.063 → a_c ≈ 5.1 m/s². So 8.0 might be too high. Try 5.0-6.0 initially.
- Max inflation 1.25x: more conservative than sharp-turn inflation (1.60x) since these are moderate turns

## Risk Assessment
- **Race time regression**: Inflating at gates 3-4 will slow the race by ~0.3-0.8s. Acceptable if tracking improves.
- **Other gates affected**: Need to check if other gates also have high a_c. Gates 2, 5 have moderate turns too.
- **No structural risk**: This is an additive condition to existing code, not a replacement.

## Rollback Criteria
- If avg tracking error doesn't improve by at least 0.02m, revert
- If race time increases by more than 1.5s, revert
- If gate pass rate drops below 100%, revert immediately

## Test Plan
1. Run unit tests after edit
2. Run full benchmark
3. Compare per-gate errors — gates 3, 4 should improve
4. Verify gates 6, 7, 8, 10 (already inflated) don't regress
5. Check race time stays under 17s
