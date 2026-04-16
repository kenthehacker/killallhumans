# Iteration 49 Plan — Heavy-Ball Momentum ILC

## Objective
Reduce avg tracking error below 0.163m by adding heavy-ball momentum to the per-section ILC update rule. Target: 3-5% improvement (0.163→0.155m).

## Research Basis
- Wang 2023 (arXiv:2312.14326): Nesterov-accelerated data-driven ILC with hybrid switching
- Gu et al. 2019 (referenced in Wang): accelerated gradient ILC with demonstrated faster convergence
- Heavy-ball method (Polyak 1964): u_{k+1} = u_k + alpha * grad + gamma * (u_k - u_{k-1})

## Files to Modify
1. **`planning/trajectory_optimizer.py`** — `compute_ilc_offset_table()` function
   - Add `momentum_gamma` parameter (default 0.0 for backward compatibility)
   - Store previous iteration's section offsets
   - Apply momentum term after alpha * smoothed_error update

2. **`scripts/benchmark.py`** — ILC configuration
   - Add `momentum_gamma` parameter to `compute_ilc_offset_table()` call
   - Sweep values: 0.1, 0.2, 0.3

## Algorithm Changes

### Current (P-type ILC, line 410):
```python
section_offsets[sec_idx][sec_start_c:sec_end_c] += sec_alpha * sec_smoothed
```

### Proposed (Heavy-ball momentum ILC):
```python
# Compute momentum (difference from previous iteration's offset for this section)
if ilc_iter > 0:
    momentum = section_offsets[sec_idx][sec_start_c:sec_end_c] - prev_section_offsets[sec_idx][sec_start_c:sec_end_c]
else:
    momentum = 0.0

# Store current offset before update
prev_section_offsets[sec_idx][sec_start_c:sec_end_c] = section_offsets[sec_idx][sec_start_c:sec_end_c].copy()

# Heavy-ball update: P-type + momentum
section_offsets[sec_idx][sec_start_c:sec_end_c] += sec_alpha * sec_smoothed + momentum_gamma * momentum
```

### Clipping Safety
The existing `max_correction_m` clipping (line 400-407) already bounds the correction magnitude. Momentum cannot cause unbounded growth because:
1. Each section's offset is clipped per-iteration
2. Convergence threshold stops iteration if error stops improving
3. momentum_gamma < 0.5 prevents oscillation per heavy-ball stability analysis

## Sweep Plan
Test globally (same gamma for all sections):
1. gamma = 0.0 (baseline, verify no regression from code change)
2. gamma = 0.1 (conservative momentum)
3. gamma = 0.2 (moderate)
4. gamma = 0.3 (aggressive)

If global gamma shows improvement, optionally test per-section gamma:
- Higher gamma for under-converged sections (post-inflection, helix)
- Lower gamma for sensitive sections (pre-inflection)

## Risk Assessment
- **Basin switching**: NONE — offsets are capped, racing line geometry unchanged
- **Regression**: POSSIBLE — momentum could cause oscillation in well-converged sections
- **Mitigation**: gamma < 0.5, existing max_correction_m clipping, convergence threshold
- **Revert cost**: trivial — single parameter change

## Rollback Criteria
- If ANY sweep value produces avg error > 0.163m (baseline), skip that value
- If BEST gamma still produces avg error > 0.163m, revert ALL changes
- If any individual gate regresses > 20%, skip that gamma value

## Test Plan
1. Run unit tests after code change (should pass — ILC is only used in benchmark)
2. Run full benchmark with gamma=0.0 (verify no regression from code change)
3. Sweep gamma=0.1, 0.2, 0.3 (select best)
4. Run full benchmark with best gamma (verify all metrics)
