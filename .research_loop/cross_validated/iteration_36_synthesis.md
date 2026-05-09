# Iteration 36 — Research Synthesis: Helix Floor Pareto Rebalancing

## Papers Analyzed (New)
1. **Jerk-Constrained TOPP** (arXiv:2404.07889, 2024) — Adds jerk limits to TOPP-RA, reduces peak power 25% and RMS torque 50% with 3-5% time overhead. Warm-starts from unconstrained TOPP.
2. **Primitive-Planner** (arXiv:2502.16882, Feb 2025) — TOPP-RA for quadrotors with full rigid-body dynamics and per-motor thrust constraints. Validates LP-based reachability for speed profiling.
3. **CiMPCC Lap Time Reduction** (arXiv:2502.03695, Feb 2025) — Curvature-integrated MPCC with exponential speed mapping g(K) = exp(-α·K²). Pushes speed on straights, slows on curves.

## Consensus
- **Strong**: Spatially-varying TOPP constraints are the right paradigm — different track sections need different speed/accuracy tradeoffs (Primitive-Planner, CiMPCC, iter 35 ILMPC/Spatially-Aware papers)
- **Strong**: The TOPP compression floor is the correct control lever for trading time vs accuracy in our pipeline (confirmed by iter 35 helix floor sweep)
- **Moderate**: Jerk constraints could improve S-turn tracking by smoothing velocity transitions, but the implementation complexity is high for uncertain gain

## Key Diagnostic Finding
The TOPP floor binding analysis shows:
- 5 helix exit segments ALL hit exactly the 0.76 floor (segments 11,13,15,17,19)
- 5 S-turn exit segments ALL hit exactly the 0.65 floor (segments 3,5,7,9,21)
- Entry segments are speed-limited, not floor-limited (ratio=1.0)

The iter 35 helix floor change (0.65→0.76) improved accuracy by -7.9% but added +0.30s race time (13.79→14.09s). The floor value 0.76 was selected as Pareto-optimal for accuracy, but it overshoots the 14s race time target.

## Proposed Direction
**Shift along the Pareto frontier**: lower helix floor from 0.76 to ~0.72 to recover ~0.1s race time while maintaining most of the accuracy improvement. This is the same lever validated in iter 35 (monotonic sweep showed predictable behavior).

Expected outcome at 0.72:
- Race time: ~13.98s (under 14s target)
- Avg error: ~0.175m (between 0.171m at 0.76 and 0.185m at 0.65)
- Still 2nd-best avg error ever

## What NOT to Try
- Full TOPP-RA reimplementation (Primitive-Planner) — too complex for one iteration
- CiMPCC exponential mapping — architectural change to scoring function
- Jerk constraints in TOPP — moderate complexity, uncertain gain for race time goal
- S-turn floor changes — adds time, opposite of what we need this iteration
