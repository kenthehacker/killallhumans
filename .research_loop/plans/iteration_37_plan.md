# Iteration 37 Plan — S-turn TOPP Floor Optimization

## Objective
Reduce gate-3 tracking error (currently 0.247m, worst gate) by raising the S-turn TOPP compression floor. Target: gate-3 ≤ 0.220m while maintaining race time ≤ 13.98s via joint helix floor compensation.

## Research Basis
- **GripMap (Werner 2025)**: Spatially-varying constraints (scaling factors θ_ij) validated with 5.2% improvement
- **Energy-Limited MinLap (van den Eshof 2026)**: Curvature-peak decomposition supports independent section optimization
- **CiMPCC (Li 2024)**: Curvature-integrated speed mapping for S-turn regions
- **Iter 35-36 precedent**: Helix floor sweep showed perfect non-helix isolation — S-turn changes expected similarly independent

## Files to Modify
1. `planning/trajectory_optimizer.py` (line ~891-1030): Split S-turn floor from protected floor, sweep values

## Algorithm Changes

### Step 1: Create separate S-turn floor variable
Currently `max_compression_protected = 0.65` is used for S-turn, high-curvature, and pre-turn segments. Create a new `max_compression_sturn` variable for S-turn segments specifically, keeping `max_compression_protected` for the other cases.

```python
max_compression_sturn = 0.65  # S-turn floor (NEW — iteration 37)
max_compression_protected = 0.65  # high-curvature, pre-turn (unchanged)
```

Then in the floor assignment:
```python
if i in s_turn_segments:
    seg_floor.append(max_compression_sturn)  # was max_compression_protected
```

### Step 2: Sweep S-turn floor [0.67, 0.69, 0.71, 0.73]
Run full benchmark at each value. Record:
- gate-3 avg error (target: improvement)
- race time (check regression)
- all non-S-turn gates (check isolation)

### Step 3: Joint rebalancing with helix floor (if needed)
If S-turn floor increase adds race time beyond 13.98s:
- Use helix Pareto relationship: -0.02s per -0.01 helix floor (from iter 36)
- Lower helix floor to compensate
- Example: if S-turn floor 0.71 adds 0.06s → lower helix floor by 0.03 (0.72→0.69) to compensate

### Step 4: Select optimal combination
Criteria:
1. Race time ≤ 13.98s
2. Minimum avg tracking error
3. Gate-3 error minimized

## Risk Assessment
- **Basin switching**: LOW — TOPP floor changes have been safe in iters 35-36 (floor changes don't affect racing line geometry)
- **Cross-section coupling**: LOW — helix floor changes showed zero non-helix coupling
- **Regression**: MEDIUM — unknown whether S-turn floor change affects gate-2/gate-4 (adjacent gates)

## Rollback Criteria
- Revert if race time > 20s (basin switching indicator)
- Revert if avg error increases > 5% from baseline
- Revert if any gate error increases > 20%

## Test Plan
1. Unit tests: verify no test failures after code change
2. Sweep S-turn floor [0.67, 0.69, 0.71, 0.73] with full benchmark
3. If time budget allows, do joint S-turn × helix 2D sweep
4. Select best, commit
