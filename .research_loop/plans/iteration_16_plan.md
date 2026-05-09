# Iteration 16 Plan — S-Turn Compound Inflation

## Objective
Reduce tracking error at gates 3-4 (the S-turn) from 0.452/0.465m to ~0.35m while maintaining race time ≤13.60s. Target avg error improvement: 0.251→0.23m.

## Research Basis
- **CiMPCC** (Li, ITSC 2024): Smoothed compound curvature captures S-turn effect; curvature doesn't drop between consecutive turns
- **VPMPCC** (Li, 2024): Learned profiles show early deceleration before S-turns; approach segments need slowing
- **TACO** (Sanghvi 2025): Trajectory parameters should adapt to local curvature-speed characteristics
- **Alternating Peak** (de Vries, ECC 2024): Per-segment peak utilization reveals under-inflated S-turn segments

## Files to Modify
1. **`planning/trajectory_optimizer.py`** — `_inflate_sharp_turns()` and `_topp_retime()`

## Algorithm Changes

### Change 1: S-Turn Detection in `_inflate_sharp_turns`
After computing turn_angle at each gate, detect S-turn pairs:

```python
# S-turn detection: consecutive turns with opposite lateral direction
# Cross product sign change indicates direction reversal
if gi >= 2:
    v_prev_in = gate_centers[gi-1] - gate_centers[gi-2]
    v_prev_out = gate_centers[gi] - gate_centers[gi-1]
    cross_prev = np.cross(v_prev_in, v_prev_out)[2]  # Z-component (yaw plane)

    v_curr_in = gate_centers[gi] - gate_centers[gi-1]
    v_curr_out = gate_centers[gi+1] - gate_centers[gi]
    cross_curr = np.cross(v_curr_in, v_curr_out)[2]

    # Opposite signs = S-turn (direction reversal in yaw plane)
    is_s_turn = (cross_prev * cross_curr < 0) and (turn_angle > 0.25)
```

### Change 2: S-Turn Compound Inflation
For the second gate of an S-turn pair, apply additional inflation:

```python
if is_s_turn:
    # The second turn of an S-turn requires extra time because:
    # 1. Drone arrives with lateral velocity in the wrong direction
    # 2. Must reverse lateral velocity (takes ~2x the time of a single turn)
    # Research: CiMPCC compound curvature, VPMPCC sustained low speed
    s_turn_factor = 1.15  # 15% extra inflation for S-turn compound effect
    inflate = max(inflate, s_turn_factor)

    # Also inflate the APPROACH segments (gate_prev exit → this gate entry)
    # Research: VPMPCC shows early deceleration is critical
    approach_seg = seg_entry - 1  # segment from previous gate exit to this entry
    if 0 <= approach_seg < len(times):
        times[approach_seg] *= 1.10  # 10% approach inflation
```

### Change 3: Compound Curvature in TOPP Retimer
In `_topp_retime`, detect S-turn segments and boost their effective curvature:

```python
# In Step 2 (speed limits from curvature):
# For segments between S-turn gates, use compound curvature
# This prevents TOPP from speeding up the S-turn transition
if is_s_turn_region[i]:
    k_max *= 1.3  # 30% curvature boost for S-turn compound effect
```

## Risk Assessment
- **Regression risk**: The inflation only targets gates 3-4 (and possibly 4-5). Other gates should be unaffected.
- **Race time risk**: Inflation at gates 3-4 will add ~0.2-0.4s. Current race time is 13.50s with 0.5s margin to the 14s aspirational target.
- **Overshoot risk**: If inflation is too aggressive, gate-3/4 will slow down excessively. The TOPP retimer's compression floor (65%) limits this.

## Rollback Criteria
- If avg tracking error > 0.26m (regression from 0.251m), revert
- If race time > 14.0s (exceeds aspirational target), revert
- If any gate fails to pass, revert

## Test Plan
1. Run unit tests first (should be unaffected since changes are in post-optimization)
2. Run full benchmark
3. Compare per-gate errors: gate-3 and gate-4 should improve, other gates should be stable
4. Verify race time stays below 14.0s
