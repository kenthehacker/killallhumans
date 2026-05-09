# Iteration 36 Plan: Helix TOPP Floor Pareto Rebalancing

## Objective
Recover race time from 14.09s to sub-14.0s by shifting along the validated Pareto frontier between helix tracking accuracy and race time.

## Research Basis
- **Iter 35 validation**: Monotonic sweep 0.68-0.80 showed predictable behavior
- **TOPPQuad (Mao 2024)**: Spatially-varying time dilations are optimal
- **Spatially-Aware (arXiv 2602.15642)**: Controller-guided spatially-varying constraints
- **ILMPC (arXiv 2508.01103)**: Adaptive cost dynamically weights time vs tracking

## Files to Modify
- `planning/trajectory_optimizer.py` line 892: Change `max_compression_helix = 0.76` to optimal value found by sweep

## Algorithm
1. Sweep `max_compression_helix` values: [0.70, 0.71, 0.72, 0.73, 0.74]
2. For each value, run full benchmark
3. Select the value that achieves: race_time < 14.0s AND avg_error < 0.180m
4. If multiple values qualify, pick the one with lowest avg_error
5. Apply the winning value

## Risk Assessment
- **Low risk**: Single parameter, validated range, no basin switching
- **Regression bound**: avg_error won't exceed 0.185m (pre-iter-35 baseline)
- **Non-helix gates**: Completely unaffected (confirmed in iter 35)

## Rollback Criteria
- If no sweep value achieves race_time < 14.0s → keep 0.76, report failure
- If all values regress avg_error > 0.185m → keep 0.76, report failure
- If any non-helix gate changes → stop, investigate

## Test Plan
1. Run full benchmark for each sweep value
2. Verify non-helix gates unchanged
3. Verify deterministic results
4. Compare per-gate errors against iter 35 baseline
