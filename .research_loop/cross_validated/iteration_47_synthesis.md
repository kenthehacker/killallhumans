# Iteration 47 — Research Synthesis: Multi-Start L-BFGS for Time-Optimal Trajectory Optimization

## Papers Analyzed
1. MISO: Learning Multiple Initial Solutions (arXiv:2411.02158) — Sharony et al. 2024
2. cuRobo: Parallelized Multi-Start L-BFGS (arXiv:2310.17274) — Sundaralingam et al. (NVIDIA)
3. AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization (arXiv:2509.17340) — ICRA 2026

## Consensus Across Papers
All three papers agree on one fundamental insight: **single-start local optimization in non-convex landscapes is inherently suboptimal, and multi-start with diverse initializations dramatically improves solution quality.**

- MISO: 5-14× cost improvement over single warm-start across DDP, MPPI, and iLQR optimizers
- cuRobo: 99.8% success rate (vs 98.5%) with 12 parallel L-BFGS seeds, 60× speedup
- AERO-MPPI: 2-3× velocity improvement via 15 anchor-seeded MPPI instances

The improvements are consistent across fundamentally different optimization methods (gradient-based L-BFGS, sampling-based MPPI, model-based DDP/iLQR), suggesting the benefit is general and applies to our L-BFGS trajectory optimizer.

## Key Technical Recommendations
1. **Structured diversity > random perturbation** (MISO, AERO-MPPI): Seeds should be geometrically meaningful, not just Gaussian noise. AERO-MPPI's LiDAR-derived anchors and MISO's diversity losses both enforce that seeds cover distinct basins.

2. **Always include the baseline** (MISO guarantee): Include the current default initialization as one of the K seeds, ensuring multi-start cannot regress vs. the baseline.

3. **Modest K is sufficient** (cuRobo): 12 seeds for trajectory optimization, with diminishing returns beyond that. For sequential execution on CPU, K=4-6 is practical.

4. **Mixed-scale perturbation** (cuRobo): Use small (σ=0.2), medium (σ=0.4), and large (σ=0.8) perturbations to cover basins at multiple scales.

5. **Warm restart from best** (AERO-MPPI, MISO): Cache the best result for future re-planning.

## Proposed Implementation for Our Optimizer

Replace the single `minimize()` call in `_optimize_time_allocation()` with a multi-start loop:

### Seed Strategy (K=4, keeping compute manageable):
1. **Seed 0 (baseline)**: Current heuristic initialization — guarantees no regression
2. **Seed 1 (graduated)**: Optimize at time_weight=1.5 first, then warm-restart at 2.3 — continuation method to reach a different basin
3. **Seed 2 (aggressive)**: Scale initial times by 0.7x — starts from faster trajectory
4. **Seed 3 (conservative)**: Scale initial times by 1.3x — starts from slower, more trackable trajectory

### Selection: Use the L-BFGS objective value (time_weight * total_time + penalties) for selection among feasible solutions.

### Determinism: Use fixed seed (np.random.default_rng(42)) for reproducibility.

## Risk Assessment
- **Compute cost**: 4× L-BFGS runs ≈ 4× planning time. Currently ~150ms → ~600ms. Acceptable since it's offline pre-computation.
- **Basin switching risk**: Multi-start could find a DIFFERENT basin that has lower L-BFGS cost but worse post-ILC tracking. Mitigation: select based on L-BFGS cost, then verify with full benchmark.
- **Determinism**: Fixed random seeds ensure reproducibility across runs.

## What Could Go Wrong
- The graduated warm-restart (seed 1) could converge to a trajectory that the TOPP retimer and inflation stages transform into something worse than the current trajectory
- The selection by L-BFGS objective may not correlate with sim tracking performance (since L-BFGS doesn't model the PD controller)
- Multi-start could produce only marginally different results if the landscape has one dominant basin

## Cross-Validation Assessment
The approach is well-supported by 3 independent papers across different optimization domains. The risk is low because:
1. Seed 0 (baseline) guarantees no regression
2. The change is isolated to `_optimize_time_allocation()` — doesn't affect ILC, controller, or racing line
3. Fixed random seeds maintain determinism

**Recommendation: Proceed with K=4 multi-start implementation.**
