# Iteration 34 — Research Synthesis: Helix Gate Offset Optimization

## Papers Analyzed (New in This Iteration)

1. **CdBO — Learning to Race through Coordinate Descent Bayesian Optimisation** (Cully 2018, arXiv:1802.06179)
   - Coordinate descent structure decomposes high-dimensional racing optimization into per-section 1D BO problems
   - Key insight: cycling through track sections one at a time with GP+EI achieves sample-efficient optimization

2. **Gradient-free Multi-domain Optimization for Autonomous Systems** (Zheng 2022, arXiv:2202.13525)
   - CMA-ES is the best gradient-free optimizer for racing parameter tuning (outperforms DE, PSO, random search)
   - 100-300 evaluations sufficient for 4D parameter spaces

3. **A Data-Driven Aggressive Autonomous Racing Framework with VPMPCC** (Li 2024, arXiv:2410.11570)
   - VPMPCC velocity prediction: sustained low velocity across compound curves reduces tracking error
   - BO for parameter tuning with sim-in-the-loop evaluation

## Consensus Across Papers

**Strong consensus:**
- Sim-in-the-loop evaluation is the gold standard for trajectory parameter optimization (all 3 papers)
- Coordinate descent / per-parameter optimization is effective for racing parameter tuning (CdBO, CMA-ES)
- Gate offset optimization directly controls path shape and tracking error (TOGT, VPMPCC)

**Key insight for our system:**
The racing line cache provides deterministic sim evaluation (iter 33 achievement). Combined with coordinate descent search over gate offsets, we can systematically find better helix pass-through points. The search space is only 4D (gate-7/gate-8 × lat/vert), well within the efficient range for CdBO/CMA-ES.

## Experimental Results (This Iteration)

A coordinate descent search over gate-7/8 offsets with fast kinematic sim evaluation (no ILC) found:

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| Gate-7 lateral | 0.555 | 0.600 | +0.045 (push to boundary) |
| Gate-8 lateral | -0.600 | -0.200 | +0.400 (less aggressive cut) |
| Gate-8 vertical | -0.067 | 0.333 | +0.400 (vertical repositioning) |

**Avg tracking error (no ILC): 0.2250m → 0.2066m (-8.2%)**
- Gate-7: 0.353 → 0.328 (-7.1%)
- Gate-8: 0.169 → 0.158 (-6.5%)

## Implementation Direction

Update the racing line cache with the optimized offsets. The changes make physical sense:
- Gate-8 was at max lateral boundary (-0.6), suggesting the optimizer was constrained
- Moving gate-8 closer to center reduces path curvature in the helix
- Vertical repositioning at gate-8 smooths the helical ascent profile
