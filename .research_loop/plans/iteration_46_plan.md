# Iteration 46 Plan — ILC Section Tuning for Gate-4 and Gate-7

## Objective
Reduce gate-4 error from 0.302m to ≤0.25m and gate-7 error from 0.252m to ≤0.23m without regressing other gates beyond 20% or increasing race time.

## Research Basis
- CDC 2024 (Constrained ILC Alternating Projection): hard-clipping ILC corrections degrades steady-state error 15-25%. Our inflection section max_correction=0.15m is likely binding at gate-4 (0.302m error).
- Longman 2023 (Speed Up ILC Convergence): model-based ILC converges to error floor; correction caps are the binding constraint, not iteration count.
- Schoellig 2012 (Optimization-based ILC): per-section parameters validated for different track regions.
- Track-centric ILC 2026: spatially varying correction limits improve lap performance.

## Files to Modify
1. `scripts/benchmark.py` — ILC section boundary parameters (lines 309-319)

## Algorithm Changes
Current configuration:
```python
inflection_end = int(4.4 / dt)     # step 440
section_boundaries = [
    (0, inflection_start, 0.4, 0.15, 0.35, 0.0),              # Pre-inflection
    (inflection_start, inflection_end, 0.45, 0.15, 0.40, 0.4), # Inflection
    (inflection_end, helix_start, 0.4, 0.15, 0.35, 0.5),      # Post-inflection
    (helix_start, n_total_steps, 0.4, 0.45, 0.35, 0.7),       # Helix
]
```

### Change 1: Increase inflection max_correction 0.15→0.20m
Gate-4 (step 420) is in the inflection section. The 0.15m correction cap limits ILC's ability to compensate for the faster trajectory. Increase to 0.20m.

### Change 2: Extend inflection_end 440→460
This gives the inflection section's higher bandwidth filter (0.40 Hz) 0.2s more coverage over the gate-4 approach region. Combined with higher correction cap, this targets gate-4 directly.

### Change 3: Increase helix max_correction 0.45→0.50m
Gate-7 (step ~753) has 0.252m error. The helix section max_correction is 0.45m (well above the error), so this change targets marginal improvement. The higher cap gives ILC more room to correct for the faster time_weight=2.3 trajectory.

Proposed configuration:
```python
inflection_end = int(4.6 / dt)     # step 460 (was 440)
section_boundaries = [
    (0, inflection_start, 0.4, 0.15, 0.35, 0.0),              # Pre-inflection (unchanged)
    (inflection_start, inflection_end, 0.45, 0.20, 0.40, 0.4), # Inflection: max_corr 0.15→0.20m
    (inflection_end, helix_start, 0.4, 0.15, 0.35, 0.5),      # Post-inflection (unchanged)
    (helix_start, n_total_steps, 0.4, 0.50, 0.35, 0.7),       # Helix: max_corr 0.45→0.50m
]
```

## Risk Assessment
- **Gate-5 regression**: Main risk. Iter 43 showed inflection+post-inflection cap increase to 0.20m regressed gate-5. Mitigation: we only change inflection section, NOT post-inflection. The inflection section now ends at step 460, which is still 67 steps before gate-5 (step ~527).
- **Gate-3 regression**: Unlikely — inflection section alpha=0.45 unchanged, and gate-3 (step ~293) is well within the section.
- **Race time regression**: Very unlikely — ILC corrections are position offsets, not timing changes.

## Rollback Criteria
- If any gate regresses >20%, revert all changes.
- If avg error increases >10%, revert.
- If race time increases >0.5s, revert.

## Test Plan
1. Apply all 3 changes
2. Run full benchmark
3. Compare per-gate errors
4. If gate-5 regresses >20%, revert change 2 (inflection_end extension) and re-test
5. If gate-5 still regresses, revert change 1 (inflection cap) and try only change 3
