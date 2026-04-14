# Iteration 18 Research Synthesis — Helix-Specific Turn Inflation Recovery

## Papers Analyzed
### New in this iteration
1. **Quad-LCD** (Srikanthan et al., UPenn 2025) — Controller-aware trajectory feasibility prediction
2. **Improving Drone Racing via ILMPC** (arXiv:2508.01103, 2025) — Iterative learning MPC for drone racing
3. **Spatial ILC in Virtual Tube** (arXiv:2306.15992, 2023) — Iterative speed profile optimization

### Key existing papers used
4. **TACO** (Sanghvi et al., 2025) — Trajectory-aware controller optimization
5. **CiMPCC** (Li et al., ITSC 2024) — Curvature-integrated speed profiling with neighborhood awareness
6. **FBGA** (Piazza et al., RA-L 2025) — Forward-backward speed profiling with non-convex constraints

## Root Cause Analysis

The helix gates 7-12 regressed 0.025-0.052m in iteration 17 because FOV relaxation removal
exposed an **inflation coverage gap** in `_inflate_sharp_turns`:

### Critical finding: Helix gates have heterogeneous turn angles
| Gate | Turn Angle | >60° Threshold? | Current Proximity Inflation | Gap |
|------|-----------|-----------------|---------------------------|-----|
| gate-7 | 68.5° | YES | 4.7% | Small |
| gate-8 | 63.2° | YES (barely) | 2.3% | Moderate |
| gate-9 | **48.5°** | **NO** | 2.3% | **HIGH** |
| gate-10 | 63.2° | YES (barely) | 4.7% | Moderate |
| gate-11 | **49.9°** | **NO** | **0.7%** | **CRITICAL** |
| gate-12 | (exit) | N/A | — | — |

Gates 9 and 11 fall below the 60° angle threshold and don't get angle-based inflation.
They rely solely on proximity inflation, which is inadequate because:

1. **Proximity only checks distance to NEXT gate** — Gate 11 checks dist to gate-12 (5.66m),
   missing that dist to gate-10 is only 3.64m
2. **The 0.12 multiplier is too small** — At dist=5.66m, factor = 1.0 + 0.12 * (1-5.66/6) = 1.007

The FOV relaxation was inadvertently providing 3-8% extra inflation for these gates.
Removing it exposed the gap.

## Research Consensus

### What multiple papers agree on:
1. **Neighborhood curvature awareness** (CiMPCC, FBGA, Spatial ILC): Speed profiling
   should consider the curvature context around a point, not just the point itself.
   CiMPCC's curvature smoothing and look-ahead are specifically designed for sequences
   of turns where the drone arrives with velocity in the wrong direction.

2. **Controller-aware trajectory adaptation** (TACO, Quad-LCD): The trajectory planner
   should account for controller tracking capability. Quad-LCD's key insight is that
   per-segment feasibility depends on the polynomial coefficients AND approach velocity,
   not just geometry.

3. **Iterative speed refinement** (Spatial ILC, ILMPC): Speed profiles should be
   refined iteratively using per-segment tracking error feedback. The update rule
   `v_new = v_old - k_p * error` with a deadzone threshold is well-established.

4. **Bidirectional context matters** (CiMPCC, FBGA): Forward-backward propagation in
   FBGA and CiMPCC's compound curvature both emphasize that a segment's speed limit
   depends on both what came before AND what comes after.

## Proposed Implementation

### Primary: Bidirectional proximity inflation
Make `_inflate_sharp_turns` check distance to BOTH previous and next gate (currently
only checks next). Use the minimum distance to either neighbor.

**Research backing**: CiMPCC's compound curvature principle — in a tight sequence like
the helix, each gate's handling difficulty depends on its relationship to ALL neighbors,
not just the next gate.

### Secondary: Increase proximity multiplier
Raise the proximity inflation multiplier from 0.12 to 0.18 to compensate for removed
FOV relaxation.

**Research backing**: TACO shows that static parameters are always a compromise;
the removed FOV relaxation was an implicit adaptation mechanism. Quad-LCD validates
that reshaping the reference trajectory (via inflation) is sufficient without
changing controller parameters.

### Expected impact
- Gate 11: inflation 0.7% → 7.1% (using min(dist_to_gate10, dist_to_gate12) = 3.64m)
- Gate 9: inflation 2.3% → 3.4% (both neighbors at 4.87m)
- Gate 8: no change (already has angle-based inflation ~7%)
- Gates 1-6: no change (all >6m apart, proximity doesn't trigger)

## Contradictions / Risks
- **Spatial ILC suggests iterative refinement**, not one-shot parameter adjustment.
  However, our benchmark loop IS an iterative process (18 iterations and counting),
  so this is compatible.
- **Quad-LCD suggests learned feasibility**, which would be better than heuristic
  inflation. However, that requires data collection infrastructure — too complex for
  a single iteration. The heuristic fix is the pragmatic choice.
- **Risk**: Increasing proximity multiplier could over-inflate gate-8 (where proximity
  gives 8.8% with 0.18 multiplier, overriding its current 7.5% angle inflation).
  This is a minor risk since `max()` ensures only the highest inflation applies.
