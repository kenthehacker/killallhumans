# Iteration 13 Plan — Racing Line Optimization + Helix Inflation

## Objective
Reduce helix tracking error (gate-7: 0.659m → <0.45m, gate-8: 0.528m → <0.40m) by improving trajectory shape through more aggressive racing line optimization and better helix inflation. Target: avg error 0.358m → <0.30m.

## Research Basis
- **TOGT Planner (Qin 2024)**: Gates are regions, not points. Optimal path doesn't pass through centers.
- **Swift (Kaufmann 2023)**: RL agent learned aggressive corner cutting → faster AND smoother.
- **ILMPC (Zhao 2025)**: Trajectory quality has more impact than controller tuning. Spatially-varying precision matters.
- **Iteration 12 finding**: Controller tuning exhausted in kinematic sim. Trajectory planning is the remaining lever.

## Key Diagnostic
The RacingLineOptimizer is hitting its maximum offset bounds (0.339m) at most gates. It wants to cut corners more but is constrained by `max_lateral_offset=0.4`. This means the racing line through the helix is suboptimal — the optimizer could find a smoother path with more room.

Additionally, the helix inflation factor is only 7% for gate-7 (68.5° turn). Given that helix gates are very close together (3.6-4.9m), the short segments create high curvature that the PD controller can't follow.

## Files to Modify

### 1. `planning/racing_line.py` — RacingLineConfig defaults
- Increase `max_lateral_offset`: 0.4 → 0.6 (allows 0.36m offset per axis, leaves 0.24m margin to gate edge)
- Increase `smoothness_weight`: 0.3 → 0.5 (prioritize curvature reduction over path length for closer gate sequences)
- Keep `corner_cut_aggressiveness` at 0.7 (the optimizer handles this via the objective)

### 2. `planning/trajectory_optimizer.py` — _inflate_sharp_turns
- Add a **proximity-based inflation factor**: when consecutive gates are within 5m, apply additional 10% inflation on top of angle/centripetal inflation
- This specifically targets the helix where gates are 3.6-4.9m apart

## Algorithm Changes

### Racing Line (racing_line.py)
```
max_lateral_offset: 0.4 → 0.6
smoothness_weight: 0.3 → 0.5
```
This allows the optimizer to find paths with up to 50% larger offsets from gate centers, reducing curvature through the helix. The increased smoothness weight penalizes sharp turns more heavily.

### Helix Inflation (trajectory_optimizer.py)
In `_inflate_sharp_turns`, after the angle/centripetal checks, add:
```python
# Proximity-based inflation for closely-spaced gates
dist_between_gates = float(np.linalg.norm(gate_centers[gi+1] - gate_centers[gi]))
if dist_between_gates < 5.0 and turn_angle > 0.5:  # close gates with meaningful turns
    proximity_factor = 1.0 + 0.10 * (1.0 - dist_between_gates / 5.0)
    inflate = max(inflate, proximity_factor)
```
This provides up to 10% additional inflation for gates within 5m of each other.

## Risk Assessment
- Race time may increase 13.31s → ~13.5-14.0s (acceptable, well under 30s)
- Larger offsets risk cutting too close to gate edges (mitigated: 0.24m margin remaining)
- Increased smoothness weight may slow down straight sections (offset by better corner cutting)

## Rollback Criteria
- If avg tracking error increases > 5%: revert
- If any gate fails to pass: revert
- If race time > 16s: revert

## Test Plan
1. Run unit tests after code changes
2. Run full benchmark
3. Compare per-gate errors, especially gate-7 and gate-8
4. Verify all 12 gates pass
