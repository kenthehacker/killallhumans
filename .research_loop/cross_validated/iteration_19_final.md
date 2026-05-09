# Iteration 19 — Cross-Validated Research (Self-Review)

## Synthesis Summary
Multi-start L-BFGS-B optimization for racing_line.py, backed by 3 papers showing that ensemble/parallel optimization from diverse initializations escapes local minima that single-start gradient methods cannot.

## Critical Challenges to the Synthesis

### Challenge 1: Is gate-4's 0.413m error actually caused by a local minimum?
**Status: Likely yes.** Evidence: (a) In iteration 13, changing smoothness_weight from 0.3→0.40 caused a qualitative shift in the racing line shape, dropping helix error by 73% but increasing S-turn error — classic sign of two distinct local minima. (b) The optimizer consistently converges to the same gate-4 offset regardless of other parameter changes across iterations 14-18. (c) F1-Init paper explicitly identifies that geometric proximity ≠ objective landscape proximity — our zero-initialization may land in a poor basin for the S-turn.

### Challenge 2: Will random initialization actually find a better basin?
**Risk: Moderate.** With only 24 optimization variables and bounds of ±0.6, the search space is compact. 8-10 random starts should cover the major basins. However, if the better basin requires coordinated offsets across gates 2-4 simultaneously (not just gate-4 in isolation), random per-gate sampling may not hit the right combination. Mitigation: include at least 2 structured initializations (late-apex patterns for S-turn gates).

### Challenge 3: Could multi-start regress race time?
**Risk: Zero.** By including the current zero-initialization as one candidate and selecting by minimum objective value, the worst case is identical to current performance. T-MPC's Theorem 2 formally guarantees this.

### Challenge 4: Is the computation cost acceptable?
**Cost: Negligible.** Current racing line optimization takes <5ms. 10x = <50ms. Benchmark wall time is 0.19s total. Even 100 starts would be fine.

### Challenge 5: Should we change the selection cost?
**Decision: Use L-BFGS objective for now.** The research suggests re-evaluating with a different cost (tracking error), but that requires running the full kinematic sim per candidate (1390 steps × 10 candidates). Too expensive for this iteration. Use objective value for selection; if gate-4 doesn't improve, switch to sim-based selection in the next iteration.

## Final Recommendation
Proceed with multi-start L-BFGS-B in `racing_line.py`:
- N_starts = 10 (8 random + 1 zero baseline + 1 geometric S-turn prior)
- Selection: minimum L-BFGS objective value
- Increase maxiter from 100 to 300
- Single file change, zero regression risk

## What Could Go Wrong
1. All 10 starts converge to the same basin → no improvement (most likely failure mode)
2. A different basin wins but produces worse tracking error despite lower objective → selection cost mismatch
3. Gate-4's problem isn't the racing line offset at all — it's the segment time allocation or the TOPP retiming

If #1 occurs, the next iteration should try sim-based selection. If #3, the bottleneck is misdiagnosed and should shift to trajectory_planning.
