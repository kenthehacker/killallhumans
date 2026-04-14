# Iteration 21 Plan: Segment-Selective Compression Floor

## Objective
Recover ~0.10-0.15s race time (14.10→<14.0s) by lowering the TOPP retimer's compression floor selectively on low-error segments, without regressing tracking accuracy.

## Research Basis
- **FBGA (Piazza 2025)**: Forward-backward velocity profiling yields segment-specific speeds — easy segments get compressed more than hard ones. Our uniform floor prevents this natural differentiation.
- **STORM (Zhang 2025)**: Spatial-temporal decoupling with LP for per-segment time optimization — segment times should be individually optimized.
- **TOPPQuad (Mao 2024)**: Fix geometry, optimize timing. Straight segments have more speed margin than turns.
- **Sequence Modeling (Mao 2025)**: Per-segment speed depends causally on preceding segments; independent segments can be compressed independently.

## Files to Modify
- **`planning/trajectory_optimizer.py`** — `_topp_retime()` method only

## Algorithm Changes

### Current State
```python
max_compression = 0.68  # uniform floor for all segments
```
All segments can only be compressed to 68% of their L-BFGS time, regardless of difficulty.

### Proposed Change
Make `max_compression` per-segment:
```python
# Per-segment compression floor
floor = []
for i in range(n):
    if i in s_turn_segments:
        floor.append(0.68)  # S-turn: keep current protection
    elif seg_curv[i] > 0.3 or (i > 0 and seg_curv[i-1] > 0.3):
        floor.append(0.68)  # High curvature: keep current protection
    else:
        floor.append(0.60)  # Straight/easy: allow more compression
```

Then in Step 5, replace:
```python
new_time = max(new_time, times[i] * max_compression)
```
with:
```python
new_time = max(new_time, times[i] * floor[i])
```

### Parameter Choice Rationale
- `0.68` for protected segments: matches current proven floor (iter 17)
- `0.60` for easy segments: 12% more compression than current. Conservative relative to iter 15's original 0.65 floor, but targeted only at segments that have tracking error well below threshold.
- Curvature threshold `0.3 rad/m`: segments with Menger curvature >0.3 are turns. This catches helix segments and moderate turns while allowing straights through.

## Risk Assessment
- **Tracking regression on compressed segments**: Low risk. Gate-1 (0.112m), gate-5 (0.150m), gate-6 (0.158m) have 50-70% margin below the 0.25m aspirational target.
- **Spillover to adjacent segments**: The TOPP forward-backward propagation handles sequential coupling — if a compressed straight feeds into a turn, the backward pass will slow the straight appropriately.
- **Over-compression**: The TOPP forward-backward speed limits from curvature (`v_limit = sqrt(a_centripetal / k_max)`) still apply. The floor only matters when the L-BFGS time is more conservative than the TOPP optimal time.

## Rollback Criteria
- If avg tracking error increases > 5% (above 0.229m): revert
- If any single gate error increases > 20%: revert
- If race time doesn't improve at all: revert (change had no effect)

## Test Plan
1. Run unit tests after edit
2. Run full benchmark
3. Compare per-gate errors — verify S-turn/helix gates are unchanged
4. Verify race time decreased
