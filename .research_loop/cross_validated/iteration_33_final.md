# Iteration 33 — Cross-Validated Research: Racing Line Offset Caching

## Cross-Validation Assessment

### Synthesis Strengths
1. Correctly identifies that competition systems pre-compute trajectories offline
2. Caching is architecturally sound — it's the standard pattern in robotics
3. The hash-based cache invalidation prevents staleness

### Synthesis Weaknesses / Challenges
1. Must ensure the FIRST run (that populates the cache) is itself deterministic
2. Should verify that scipy L-BFGS-B is deterministic on this platform (it should be, given fixed seeds and inputs)
3. The cache file format must be human-readable for debugging

### Additional Considerations
1. **Verification**: After implementing caching, run the benchmark 3x to confirm identical results
2. **Cache location**: Store alongside the racing line module, not in a temporary directory
3. **Logging**: Log whether cached offsets were loaded vs. fresh optimization was run

## Final Recommendation

**Implement racing line offset caching in `RacingLineOptimizer`.**

The change is:
- Low risk (doesn't change the optimization algorithm itself)
- High impact (eliminates the #1 process issue blocking all further improvements)
- Well-supported by research (standard practice in drone racing)
- Easily reversible (delete cache file to restore original behavior)
- Enables all future parameter tuning to be reliable

## Implementation Specification

### 1. Cache File Format
JSON file at `planning/racing_line_cache.json`:
```json
{
  "version": 1,
  "cache_key": "<sha256 of gate positions + config>",
  "offsets": [<list of float>],
  "metrics": {
    "avg_error": <float>,
    "worst_gate_error": <float>,
    "race_time": <float>
  },
  "timestamp": "<ISO 8601>",
  "n_candidates_evaluated": <int>,
  "selected_candidate_idx": <int>
}
```

### 2. Cache Key Computation
Hash of:
- Gate positions (rounded to 6 decimal places for stability)
- Gate normals and yaw
- RacingLineConfig fields (max_lateral_offset, smoothness_weight, etc.)

### 3. Cache Load Logic
In `optimize()`:
1. Compute cache key from inputs
2. Check if cache file exists and key matches
3. If match: load offsets, apply to gates, return
4. If no match: run full optimization, save cache, return

### 4. Force Re-optimization
Delete `planning/racing_line_cache.json` to force fresh optimization.
Or set `RacingLineConfig.use_cache = False`.
