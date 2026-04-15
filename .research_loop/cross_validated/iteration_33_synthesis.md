# Iteration 33 — Research Synthesis: Racing Line Determinism

## Problem Statement

The racing line optimizer's `_select_by_sim()` method creates tight coupling between
racing line selection and trajectory optimization. At current parameter values, the
optimizer landscape has near-equal-energy basins that produce non-deterministic results
across runs. This blocks all further parameter tuning (inflation, offsets, ILC).

## Research Consensus (from 89 analyzed papers)

### 1. Initialization Sensitivity is a Known Problem in Racing Optimization

**F1-Init (Shehadeh 2026)**: Demonstrated that trajectory optimization is "highly
sensitive to initialization" and converges to different local minima depending on the
starting point. Their neural network warm-start placed the optimizer in better basins
but didn't eliminate the multi-modal landscape itself.

**Spatially-Aware CMA-ES (Wachter 2026)**: Uses population-based search (CMA-ES)
precisely because gradient-based methods get trapped in local minima. Their iterative
approach adaptively tightens constraints from tracking error feedback.

**Topology-driven Parallel Optimization (de Groot 2024)**: Explicitly models multiple
topologically distinct trajectory solutions, selecting among them at runtime. Theorem 2
provides a fallback guarantee — the baseline solution is always available.

**Consensus**: Multi-start optimization helps explore basins but doesn't guarantee
deterministic selection among near-equal candidates. The fundamental issue is that
when basins have similar costs, the selection becomes noise-sensitive.

### 2. Pre-Computed Racing Lines are Standard Practice

**On Your Own (Romero 2025)**: Uses a pre-computed time-optimal reference trajectory.
The trajectory is computed offline and used as-is during flight. No online re-optimization.

**TOGT Planner (Qin 2024)**: Pre-computes gate-traversing trajectories offline.
The planner runs once before the race.

**CPC (Foehn 2021)**: Complementary progress constraints operate on a pre-computed
reference trajectory.

**Consensus**: Competition-winning drone racing systems pre-compute their trajectories
offline and use them as fixed references. Online re-optimization is only for reacting
to unexpected obstacles, not for re-choosing the racing line.

### 3. Caching/Freezing is the Industry Standard

**QuayPoints (2025)**: Pre-computes 55 constrained racing line variants offline
(2-6 hours per track), then uses the results to identify critical track regions.
The racing lines are computed once and cached.

**BO Racing Line (Heilmeier 2020)**: Uses Bayesian optimization with a sim oracle —
the expensive evaluation is done offline, and the optimal racing line is cached.

**Consensus**: The correct architectural pattern is: expensive optimization offline →
cache result → use cached result online. This is exactly what we need.

### 4. Geometric Quality Metrics Exist but are Approximate

**MIT Racing Line Optimization (Xiong thesis)**: Shows that ∫√κ ds (integral of square
root of curvature) correlates better with lap time than ∫κ² ds.

**CiMPCC (Li 2025)**: Uses curvature-integrated speed targets — curvature is the
primary geometric predictor of lap time.

**Consensus**: Geometric metrics (path length + curvature) are reasonable proxies but
imperfect. The sim-based evaluation in `_select_by_sim()` was added precisely because
geometric proxies were insufficient (iter 22). The right approach is to use the sim
evaluation ONCE and cache the result, not to revert to geometric-only selection.

## Proposed Direction

### Cache the Racing Line Offsets

1. After `_select_by_sim()` evaluates all candidates, save the winning offsets to a file
2. On subsequent runs, load cached offsets instead of re-optimizing
3. Use a hash of gate positions + config as cache key (invalidate when track changes)
4. This completely eliminates non-deterministic basin switching

### Benefits
- **Determinism**: Same offsets every run → same racing line → consistent metrics
- **Speed**: Skip expensive multi-start optimization + 13 sim evaluations (~10x faster)
- **Decoupling**: Future parameter changes to trajectory optimizer don't affect racing line
- **Foundation**: Enables reliable parameter tuning in future iterations

### Research Backing
- Pre-computed racing lines: Romero 2025, Qin 2024, Foehn 2021
- Offline optimization + cache pattern: QuayPoints 2025, Heilmeier 2020
- Initialization sensitivity: Shehadeh 2026, Wachter 2026
- Sim-based evaluation as oracle: Jain/Heilmeier 2020, TACO 2025

## Contradictions / Risks

1. **Staleness**: Cached offsets may become suboptimal if trajectory optimizer params change significantly. Mitigation: cache includes a config hash; cache is invalidated on track or config changes.

2. **Overfitting to one basin**: By freezing the racing line, we commit to one basin permanently. Mitigation: the current basin is well-tested and produces competitive metrics (0.185m avg, 13.79s).

3. **Loss of adaptability**: Can't discover better racing lines through parameter exploration. Mitigation: cache can be deleted to force re-optimization. Also, future iterations could implement smarter offline search (CMA-ES, multiple evaluations with voting).
