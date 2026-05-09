# Iteration 29 Research Synthesis — Trading Accuracy for Speed via Inflation Reduction

## Context
After 4 ILC iterations (25-28), avg tracking error reduced from 0.211→0.175m (-17%), providing 0.075m headroom to the 0.25m threshold. Race time stuck at 14.03s. ILC diminishing returns (improvement: -5.7%, -5.8%, -3.2%, -2.3%). Time to cash in accuracy headroom for speed.

## Research Basis (from 78 analyzed papers)

### Core Principle: ILC compensates for what inflation was protecting against

The post-optimization inflation system was designed across iterations 7-21 to compensate for systematic tracking errors caused by:
1. **S-turn lateral velocity reversal** (CiMPCC Li 2024, VPMPCC Li 2024)
2. **Helix centripetal demand** (TOPPQuad Mao 2024)
3. **Proximity-based high curvature** (Quad-LCD Srikanthan 2025)

ILC (Schoellig 2012, spatial ILC Lv 2023, our per-section implementation) now provides an independent compensation mechanism. Cross-track ILC offsets directly push the drone toward the reference, reducing the error that inflation was trying to prevent by slowing down.

### Consensus from literature

1. **Spatial ILC Virtual Tube (Lv 2023)**: Starts with a slow trajectory and progressively speeds up as ILC converges. Race time reduced 50.35→20.19s over 4 iterations. **Direct precedent: once ILC converges, trajectory can be sped up.**

2. **ILMPC (arXiv:2508.01103, 2025)**: Iterative learning MPC achieves 60.85% lap time improvement. Key mechanism: adaptive cost dynamically weights time-optimal vs centerline adherence. **As iterations reduce error, weight shifts toward speed.**

3. **TACO (Sanghvi 2025)**: Trajectory-aware controller optimization reduces tracking error 32% by adapting trajectory to controller capability. **Corollary: when controller capability improves (via ILC), trajectory constraints can be relaxed.**

4. **FBGA (Piazza 2025)**: Forward-backward speed profiling matches optimal control within 0.36%. **Per-segment compression floors are the binding constraint for race time.**

5. **TOPPQuad (Mao 2024)**: Geometry-timing decoupling: fix geometry, optimize speed → 40-50% faster. **Post-hoc inflation breaks this decoupling — reducing it recovers the benefit.**

6. **COP (Tzoumanikas 2022)**: Pareto-aware multi-objective: explicit accuracy-speed tradeoff. **We are currently on the conservative side of the Pareto front and can move toward faster solutions.**

7. **Perception-Aware Planning (ETH 2026)**: FOV constraints add only +8.1% time. Our FOV relaxation was already reduced to 8% cap in iter 14, then removed entirely in iter 17. **FOV is no longer an inflation concern.**

### Key quantitative targets

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| S-turn second-gate inflate | 1.10 (10%) | 1.06 (6%) | ILC compensates 4% of this error |
| S-turn junction inflate | 1.12 (12%) | 1.08 (8%) | ILC compensates ~4% |
| S-turn approach decel | 1.03 (3%) | 1.01 (1%) | ILC handles approach errors |
| S-turn first departure | 1.04 (4%) | 1.02 (2%) | Less conservative exit |
| S-turn junction departure | 1.02 (2%) | 1.01 (1%) | Marginal |
| TOPP protected floor | 0.68 | 0.64 | Allow 4% more compression at turns |
| TOPP easy floor | 0.63 | 0.58 | Allow 5% more compression on straights |

### Expected outcome
- Race time: 14.03→~13.3-13.6s (0.4-0.7s improvement)
- Avg error: 0.175→~0.20-0.22m (tracking regression absorbed by ILC headroom)
- Gate pass rate: 100% maintained (inflation reduction is gradual)

### Risk assessment
- **Failed approach iter 21**: compression floor 0.50 regressed gate-2 by 24%. We propose 0.58, much more conservative.
- **Failed approach iter 14**: uniform time compression didn't work. We're doing selective inflation reduction, not uniform.
- **Key safety**: ILC will re-converge on the new faster trajectory, providing fresh compensation.

### Contradictions
- None significant. All papers agree that conservative margins should be reduced as error compensation improves.
- Minor: ILMPC (Zhao 2025) warns that pure time optimization → gate misses. But we're not removing inflation, just reducing it by ~30-40%.

## Proposed Implementation Direction
**Reduce S-turn and TOPP inflation parameters in `trajectory_optimizer.py`, then re-run ILC on the faster trajectory.** This is the lowest-risk approach because:
1. Each parameter change is small (4-6% reduction)
2. ILC runs fresh on the new trajectory
3. Multiple rollback points if individual parameter changes regress
4. No new algorithms or code paths — just parameter tuning

## New Papers (iteration 29)
1. SPIRAL (Akgün 2025) — progressive speed increase through self-play. Confirms progressive speedup paradigm.
2. Robust Constrained ILC (2022) — constraint management in ILC. Relevant to systematic constraint relaxation.
3. Strategizing at Speed (2026) — learned predictive game for racing. Speed optimization strategies.
