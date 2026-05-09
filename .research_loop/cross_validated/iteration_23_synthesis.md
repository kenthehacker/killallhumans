# Iteration 23 — Research Synthesis: Speed-Aware Composite Score

## Papers Analyzed
1. **CiMPCC** (Li et al., 2025, arXiv:2502.03695) — Curvature-integrated MPCC for lap time reduction
2. **ILMPC for Drone Racing** (Zhao et al., 2025, arXiv:2508.01103) — Adaptive cost function for iterative drone racing
3. **COP** (Bohm et al., ICRA 2022, arXiv:2203.06982) — Pareto-optimal multi-objective trajectory planning

## Consensus Across Papers

### 1. Multi-objective scoring requires normalization (3/3 papers)
All three papers normalize objectives before combining them. CiMPCC normalizes curvature to [0,1] before mapping to velocity targets. ILMPC normalizes lateral deviation by corridor radius. COP normalizes each objective by its (nadir - utopia) range across the Pareto front. The consensus is clear: **raw race_time (in seconds) cannot be combined with tracking error (in meters) without normalization**.

### 2. Linear scalarization is fragile for competing objectives (2/3 papers)
COP explicitly warns that linear weighted sums only work on convex Pareto fronts. For concave fronts, they snap to extrema. ILMPC addresses this implicitly through spatially varying weights. For our 10-candidate discrete selection, the risk is that a linear score either fully prefers fast-but-inaccurate or slow-but-accurate, with no middle ground.

### 3. Curvature/geometry should modulate the speed-accuracy trade-off (2/3 papers)
CiMPCC uses curvature to set speed targets. ILMPC uses gate proximity to modulate tracking vs time weights. Both agree that the trade-off should vary spatially — demand accuracy near critical sections, allow speed elsewhere.

### 4. Time penalty alone is insufficient — curvature context matters (2/3 papers)
CiMPCC doesn't just minimize time — it penalizes deviation from curvature-appropriate velocity targets. ILMPC's ablation shows that pure time optimization causes gate misses. Simply adding raw race_time to our score risks selecting fast-but-crashy racing lines.

## Contradictions
- **Complexity of scoring**: COP advocates for Tchebycheff (min-max) scoring which is mathematically rigorous but complex. CiMPCC and ILMPC use simpler weighted sums with normalization. For 10-candidate discrete selection, the pragmatic choice is simpler.
- **Spatial weighting**: ILMPC uses per-step spatially varying weights, while our evaluation already produces per-candidate aggregate metrics. Adding per-step spatial weights would require re-architecting the evaluator.

## Recommended Approach

**Min-max normalized linear scoring (simplified AWT)**. This takes the key insight from COP (normalize by range) without the full Tchebycheff machinery, and adds the time penalty motivated by CiMPCC and ILMPC.

### Algorithm:
1. Evaluate all 10 candidates, collect: (avg_err_i, worst_gate_i, race_time_i)
2. Compute range for each metric: range_j = max_j - min_j + eps
3. Normalize each: norm_j = (val_j - min_j) / range_j → [0, 1]
4. Composite: score = w_avg * norm_avg + w_worst * norm_worst + w_time * norm_time
5. Weights: [0.5, 0.2, 0.3] — tracking-heavy but with meaningful time incentive

### Why these weights:
- **0.5 for avg_error**: Primary metric, must not regress from 0.206m
- **0.2 for worst_gate**: Secondary, prevents gate-specific regressions
- **0.3 for race_time**: Enough to break the current bias toward slow-and-safe, recover the 0.15s regression

### Fallback:
- If all candidates have similar race_time (range < 0.05s), the time term effectively vanishes, and selection degenerates to the current error-only approach. This is safe.
- Zero-init candidate (T-MPC Theorem 2 guarantee) remains in the pool, ensuring no regression.

## Expected Outcome
Race time recovery from 14.15s toward ~14.0s while maintaining avg_error ≤ 0.21m. The normalized scoring will select candidates that are near-Pareto-optimal on the speed-accuracy frontier.
