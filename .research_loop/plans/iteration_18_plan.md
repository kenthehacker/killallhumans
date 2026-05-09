# Iteration 18 Plan — Bidirectional Proximity Inflation for Helix Recovery

## Objective
Recover helix gate accuracy (gates 7-12) that regressed in iteration 17 due to FOV
relaxation removal. Target: avg error 0.248→0.235m, gate-11 error 0.200→0.175m,
while maintaining race time <14s.

## Research Basis
- **CiMPCC** (Li et al., ITSC 2024): Compound curvature for sequential turns —
  each gate's difficulty depends on ALL neighbors, not just the next gate.
- **TACO** (Sanghvi et al., 2025): Controller-aware trajectory adaptation confirms
  reshaping the reference trajectory is sufficient.
- **Quad-LCD** (Srikanthan et al., 2025): Per-segment feasibility depends on approach
  context, supporting bidirectional neighborhood analysis.

## Root Cause
The `_inflate_sharp_turns` proximity check only considers distance to the NEXT gate.
Gate 11 (49.9° turn angle, below 60° threshold) gets only 0.7% proximity inflation
because dist to gate-12 is 5.66m. But dist to gate-10 is only 3.64m — the helix
context is asymmetric. FOV relaxation was compensating for this gap.

## Files to Modify
1. **`planning/trajectory_optimizer.py`** — `_inflate_sharp_turns` method only

## Algorithm Changes

### Change 1: Bidirectional proximity check (lines ~445-450)
Replace the forward-only proximity check with a bidirectional one:

```python
# BEFORE (forward-only):
if gi + 1 < n_gates:
    dist_between = float(np.linalg.norm(gate_centers[gi + 1] - gate_centers[gi]))
    if dist_between < 6.0 and turn_angle > 0.4:
        proximity_factor = 1.0 + 0.12 * (1.0 - dist_between / 6.0)
        inflate = max(inflate, proximity_factor)

# AFTER (bidirectional):
dist_next = float(np.linalg.norm(gate_centers[gi + 1] - gate_centers[gi])) if gi + 1 < n_gates else 999.0
dist_prev = float(np.linalg.norm(gate_centers[gi] - gate_centers[gi - 1])) if gi > 0 else 999.0
dist_closest = min(dist_next, dist_prev)
if dist_closest < 6.0 and turn_angle > 0.4:
    proximity_factor = 1.0 + 0.18 * (1.0 - dist_closest / 6.0)
    inflate = max(inflate, proximity_factor)
```

### Change 2: Increase proximity multiplier from 0.12 to 0.18
This compensates for the 3-8% inflation that FOV relaxation was providing.

## Expected Impact
| Gate | Before (inflation) | After (inflation) | Expected Error Change |
|------|-------------------|-------------------|----------------------|
| gate-9 | 2.3% | 3.4% | 0.227→0.215m |
| gate-11 | 0.7% | 7.1% | 0.200→0.175m |
| gate-8 | 2.3% | 7.1% | 0.267→0.245m |
| gates 1-6 | 0% | 0% | unchanged |

## Risk Assessment
- **Race time**: Max ~0.1s increase (inflation on ~10% of segments)
- **Gate-4**: No change expected (dist 8.55-10.46m, no proximity trigger)
- **Over-inflation**: Proximity uses `max()` with angle-based inflation, preventing stacking
- **Regression risk**: LOW — only helix segments affected, change is conservative

## Rollback Criteria
Revert if:
- Race time increases >0.3s (>13.92s)
- Avg tracking error increases >0.01m (>0.258m)
- Any gate error increases >0.05m

## Test Plan
1. Run unit tests (should pass unchanged — no API changes)
2. Run full benchmark
3. Compare per-gate errors, especially gates 9, 11
4. If helix recovers but avg regresses slightly, try reducing multiplier to 0.15
