# Iteration 36 — Cross-Validated Research: Helix TOPP Floor Pareto Rebalancing

## Validation Assessment
The proposed approach (lower helix TOPP floor from 0.76 to ~0.72) is well-supported:

1. **Empirical validation**: Iter 35 swept 0.68-0.80 and found monotonic behavior. Moving within this validated range is predictable.
2. **Theoretical backing**: Spatially-varying TOPP constraints are supported by TOPPQuad (Mao 2024), FBGA (Piazza 2025), Spatially-Aware (arXiv 2602.15642), and ILMPC (arXiv 2508.01103).
3. **No basin switching risk**: TOPP floor changes only affect post-optimization timing, not the racing line geometry or L-BFGS optimization. This was confirmed in iter 35 (floor change had zero effect on non-helix gates).
4. **Quantitative prediction**: 5 helix exit segments × avg 0.54s pre-TOPP time × 0.04 floor reduction = 0.108s saved. Brings race time from 14.09s to ~13.98s.

## Risk Assessment
- **Low risk**: Single parameter change with known monotonic behavior
- **Regression bound**: At worst, helix gates return to ~0.180m (baseline before iter 35)
- **Time recovery confidence**: HIGH — the math is straightforward (linear in floor value)

## Recommended Implementation
Sweep helix floor values [0.70, 0.71, 0.72, 0.73, 0.74] via benchmark. Select the lowest value that achieves race time < 14s while maintaining avg error < 0.180m.

## What Cross-Validation Would Flag
- If avg error exceeds 0.185m (the pre-iter-35 baseline), the floor is too low
- If race time doesn't decrease as predicted, there's a non-linearity we missed
- If non-helix gates change at all, something is wrong (they should be completely independent)
