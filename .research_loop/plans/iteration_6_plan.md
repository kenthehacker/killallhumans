# Iteration 6 Plan — Adaptive Entry/Exit Offsets Based on Turn Angle

## Objective
Reduce gate-7 tracking error from 0.932m to <0.5m and avg error from 0.398m to <0.35m, without significantly increasing race time (currently 14.73s).

## Research Basis
- "On Your Own" (Romero 2025): uses ±0.4m for normal gates, -0.4/+1.25m for Split-S → adaptive offsets in practice
- TOGT Planner (Qin 2024): gates are regions, not points → optimal path uses the full gate opening
- TACO (Sanghvi 2025): adapting trajectory parameters based on local characteristics reduces tracking error

## Files to Modify
1. **`planning/trajectory_optimizer.py`** — the `optimize()` method

## Algorithm Changes

### Current (line 194):
```python
ENTRY_EXIT_OFFSET = 0.4  # fixed for all gates
```

### Proposed:
Replace fixed offset with turn-angle-dependent function:

```python
def _compute_adaptive_offset(self, turn_angle_rad: float) -> float:
    """Scale entry/exit offset based on turn sharpness.

    Research basis:
    - "On Your Own" (Romero 2025): 0.4m normal, 1.25m for Split-S
    - TOGT Planner (Qin 2024): gates are regions, not points
    """
    MIN_OFFSET = 0.25  # gentle turns: tighter path
    MAX_OFFSET = 1.0   # sharp turns: more polynomial room
    ANGLE_MIN = 0.3    # ~17° — below this, use minimum
    ANGLE_MAX = 1.6    # ~92° — above this, use maximum

    t = np.clip((turn_angle_rad - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN), 0, 1)
    return MIN_OFFSET + t * (MAX_OFFSET - MIN_OFFSET)
```

In the `optimize()` method, compute turn angle at each gate and use adaptive offset for entry/exit waypoints:

```python
for i, g in enumerate(gates):
    # Compute turn angle from previous→this→next gate centers
    prev_pos = waypoints[-1] if i == 0 else np.array(gates[i-1].position)
    curr_pos = np.array(g.position)
    next_pos = np.array(gates[i+1].position) if i+1 < len(gates) else curr_pos + normal * 2

    v_in = curr_pos - prev_pos
    v_out = next_pos - curr_pos
    # ... compute turn angle ...

    offset = self._compute_adaptive_offset(turn_angle)
    entry = pos - normal * offset
    exit_wp = pos + normal * offset
```

## Risk Assessment
- **Regression risk**: Changing offsets for ALL gates could regress gentle gates
  - Mitigation: minimum offset 0.25m is close to 0.4m; gentle gates had <0.3m error, plenty of headroom
- **Race time risk**: Longer offsets for sharp turns add path length
  - Mitigation: shorter offsets for gentle turns compensate; L-BFGS reoptimizes times anyway
- **Controller saturation**: Already at 0.85 rad limit; smoother trajectory should REDUCE saturation

## Rollback Criteria
Revert if:
- Avg tracking error increases > 0.45m
- Any gate fails (gate pass rate < 100%)
- Race time increases > 16s
- Drone crashes

## Test Plan
1. Run unit tests (should pass since trajectory_generation test uses 3 gates with moderate turns)
2. Run full benchmark
3. Compare per-gate errors, especially gate-3, gate-4, gate-7, gate-8
