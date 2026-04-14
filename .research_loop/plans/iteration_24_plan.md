# Iteration 24 Plan — Basin-Bridging Interpolation for Racing Line Candidates

## Objective
Break the bipartite candidate pool by generating interpolated racing lines between Basin A (fast, 13.99s, gate-3=0.374m) and Basin B (slow, 14.15s, gate-3=0.209m). Find a Pareto-intermediate candidate that offers better avg tracking error than Basin A while maintaining competitive race time.

**Target metrics:**
- Race time: ≤ 14.10s (modest regression from 13.99s is acceptable if tracking improves)
- Avg tracking error: ≤ 0.215m (improvement from 0.223m)
- Gate-3 error: ≤ 0.32m (improvement from 0.374m)
- All other metrics: no regression

## Research Basis
- **QuayPoints (2025)**: λ-interpolation `λ_interp = α·λ_A + (1-α)·λ_B` creates valid intermediate racing lines (§4.4)
- **COP (Bohm 2022)**: normalized multi-objective scoring already in place (iter 23)
- **Spatially-Aware CMA-ES (2026)**: population-based search explores across basins — interpolation is the lightweight equivalent

## Files to Modify
- `planning/racing_line.py` — `_select_by_sim()` method only

## Algorithm Changes

### In `_select_by_sim()`, after collecting all L-BFGS results:

1. **Identify the two basins** from the raw_metrics collected in Pass 1:
   - Basin A = candidate with lowest race_time among valid candidates
   - Basin B = candidate with highest race_time among valid candidates
   - (If race_time range < 0.05s, skip interpolation — all candidates are in one basin)

2. **Generate interpolated offset vectors**:
   ```python
   offsets_A = all_results[basin_a_idx].x
   offsets_B = all_results[basin_b_idx].x
   for alpha in [0.25, 0.50, 0.75]:
       offsets_interp = alpha * offsets_A + (1 - alpha) * offsets_B
   ```

3. **Evaluate interpolated candidates** through the same kinematic sim pipeline:
   - Build GateWaypoints from interpolated offsets
   - Generate trajectory via TrajectoryOptimizer
   - Run kinematic eval
   - Append (avg_err, worst_gate_err, race_time, interp_idx) to raw_metrics

4. **Score ALL candidates** (10 original + 3 interpolated = up to 13) using the existing normalized composite score

## Risk Assessment
- **Low risk**: The 10 original candidates remain in the pool. The worst case is that no interpolated candidate scores better, and the existing Basin A is still selected (no regression).
- **Medium concern**: If interpolated offsets produce trajectories with higher error than both basins (concave Pareto front), the interpolation approach is fundamentally limited. But this would be a valuable diagnostic finding.

## Rollback Criteria
- If benchmark avg_tracking_error > 0.250m, revert (aspirational target violation)
- If race_time > 14.20s, revert (significant regression)
- If any gate error > 0.500m, revert
- If unit tests fail, revert

## Test Plan
1. Run unit tests after code change
2. Run full benchmark
3. Compare per-gate errors before/after — gate-3 should improve if interpolation works
4. Check that race_time didn't regress beyond 14.10s
