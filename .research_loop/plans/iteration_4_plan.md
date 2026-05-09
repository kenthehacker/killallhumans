# Iteration 4 Plan: Double-Waypoint Gate Traversal

## Objective
Reduce tracking error at sharp-turn gates (gate-7: 0.394m, gate-3: 0.378m, gate-4: 0.362m) by implementing dual entry/exit waypoints for ALL gates. Target: avg tracking error 0.292→0.25m, gate-7 0.394→<0.32m.

## Research Basis
- **"On Your Own" (Romero 2025, arxiv:2510.13644)**: Each gate gets two waypoints at ±0.4m along gate normal. Deployed at IROS 2024 + Abu Dhabi F1 GP.
- **TOGT Planner (Qin 2024, arxiv:2309.06837)**: Gates are regions, not points. Single waypoint loses optimality.
- **Euclidean/Non-Euclidean (Fork 2024, arxiv:2309.07262)**: Without entry/exit constraints, trajectories may not pass correctly through gates.

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — TrajectoryOptimizer.optimize()
**Current**: Builds waypoints as `[start, gate1_center, gate2_center, ..., virtual_finish]`
**New**: Expand each gate into entry/exit waypoints:
```
[start, gate1_entry, gate1_exit, gate2_entry, gate2_exit, ..., gateN_entry, gateN_exit, virtual_finish]
```

Algorithm:
```python
ENTRY_EXIT_OFFSET = 0.4  # meters, per "On Your Own" paper

waypoints = [np.array(start_position)]
for g in gates:
    pos = np.array(g.position)
    normal = np.array(g.normal, dtype=float)
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 0.1:
        normal = normal / norm_mag
    else:
        # Fallback: use direction from previous waypoint
        normal = pos - waypoints[-1]
        n = np.linalg.norm(normal)
        if n > 0.1:
            normal = normal / n
        else:
            normal = np.array([1.0, 0.0, 0.0])

    entry = pos - normal * ENTRY_EXIT_OFFSET
    exit_wp = pos + normal * ENTRY_EXIT_OFFSET
    waypoints.append(entry)
    waypoints.append(exit_wp)

# Virtual finish: 2m past last gate (existing logic, but from exit waypoint)
```

### 2. `planning/trajectory_optimizer.py` — _optimize_time_allocation()
**Change**: Lower minimum segment time from 0.2s to 0.1s to accommodate short (0.8m) entry-exit segments.

### 3. `scripts/benchmark.py` — Unit test assertion
**Change**: Update trajectory_generation test to expect correct segment count. With 3 gates + virtual finish = 3 gates × 2 waypoints + 1 virtual finish = 8 segments (start → g1_entry → g1_exit → g2_entry → g2_exit → g3_entry → g3_exit → virtual_finish).

## Risk Assessment
- **Doubling waypoints doubles segments**: More polynomial segments means more computation, but our loop runs at 7000+ Hz so this is negligible.
- **Short segments may cause oscillation**: The 0.8m entry→exit segments are short. If time allocation is too tight, polynomials may overshoot. Mitigation: min segment time of 0.1s ensures reasonable speed through gates.
- **Race time may increase**: More segments = more conservative time allocation initially. L-BFGS should optimize this, but total time may increase slightly (~0.5-1s).
- **Possible gate-11 regression**: Previous iteration saw gate-11 regress from time redistribution. Same risk here.

## Rollback Criteria
Revert if any of:
- Avg tracking error increases by >0.02m (currently 0.292m)
- Any gate error increases by >0.05m compared to baseline
- Gate pass rate drops below 100%
- Crash occurs

## Test Plan
1. Run unit tests after code changes
2. Run full benchmark
3. Compare per-gate errors against baseline
