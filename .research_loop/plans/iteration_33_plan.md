# Iteration 33 Plan — Racing Line Offset Caching for Determinism

## Objective
Eliminate non-deterministic racing line basin switching by caching optimized offsets.
- **Target metric**: 100% reproducibility across benchmark runs (same results every time)
- **Secondary**: No regression in any metric (exact same racing line as current best)
- **Tertiary**: ~10x faster racing line computation on cached runs

## Research Basis
- Pre-computed racing lines are standard: Romero 2025, Qin 2024, Foehn 2021
- Offline computation + caching: QuayPoints 2025, Heilmeier 2020
- Initialization sensitivity motivation: F1-Init (Shehadeh 2026)
- Sim-based oracle evaluation: BO Racing Line (Jain 2020), TACO (Sanghvi 2025)

## Files to Modify

### 1. `planning/racing_line.py` (primary)
- Add `use_cache: bool = True` field to `RacingLineConfig`
- Add `_compute_cache_key()` method to hash gate positions + config
- Add `_load_cache()` / `_save_cache()` methods for JSON serialization
- Modify `optimize()` to check cache before running optimization
- After `_select_by_sim()` succeeds, save results to cache

### 2. No other files need modification
The change is entirely within the racing line optimizer. The rest of the pipeline
calls `optimizer.optimize(gates, start_position)` and gets back optimized gates —
the interface doesn't change.

## Algorithm Changes

### Before (current)
```python
def optimize(self, gates, start_position):
    # Always runs multi-start L-BFGS + _select_by_sim
    candidates = generate_candidates()
    for x0 in candidates:
        result = minimize(objective, x0, ...)
    best_idx = self._select_by_sim(gates, all_results, start_position)
    return apply_offsets(gates, all_results[best_idx].x)
```

### After (with caching)
```python
def optimize(self, gates, start_position):
    cache_key = self._compute_cache_key(gates, start_position)

    if self.config.use_cache:
        cached = self._load_cache(cache_key)
        if cached is not None:
            return apply_offsets(gates, cached)

    # Full optimization (only on cache miss)
    candidates = generate_candidates()
    for x0 in candidates:
        result = minimize(objective, x0, ...)
    best_idx = self._select_by_sim(gates, all_results, start_position)
    best_offsets = all_results[best_idx].x

    # Save to cache
    self._save_cache(cache_key, best_offsets, metrics)

    return apply_offsets(gates, best_offsets)
```

## Risk Assessment
- **Regression risk**: ZERO — the cached offsets are exactly what the optimizer produced
- **Staleness risk**: LOW — cache is invalidated by gate position or config changes
- **Process risk**: LOW — delete cache file to force re-optimization at any time

## Rollback Criteria
- If benchmark metrics differ from baseline by >1% in ANY metric, revert
- If cache mechanism causes any error, revert

## Test Plan
1. Delete any existing cache file
2. Run benchmark (first run — populates cache)
3. Verify metrics match baseline exactly
4. Run benchmark again (second run — uses cache)
5. Verify metrics are IDENTICAL to first run
6. Run benchmark a third time to triple-confirm determinism
