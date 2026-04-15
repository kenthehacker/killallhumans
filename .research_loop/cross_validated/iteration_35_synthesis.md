# Iteration 35 — Research Synthesis

## Focus: Helix TOPP Floor as Binding Constraint

### Problem
Gate-7 (0.284m) has been the worst gate for 10+ iterations. Previous approaches
(inflation increases, curvature boosts, ILC parameter tuning) had no effect because
the TOPP speed profiler's compression floor (0.65) was the binding constraint for
helix entry/exit segments.

### Key Insight (from diagnostic, not from papers)
The _topp_retime function applies `max(new_time, floor * old_time)` per segment.
For helix segments, the curvature-based speed limit produces times BELOW the floor,
so the floor is always hit. Increasing inflation or curvature boosts has no effect
because TOPP's floor clamps the final time.

### Papers Analyzed (3 new, 95 existing)

1. **Improving Drone Racing Performance Through ILMPC** (arXiv 2508.01103)
   - Adaptive cost function that dynamically weights time-optimal tracking vs
     centerline adherence. 6% improvement over MPCC++ on real drone.
   - Relevant: the concept of spatially-varying tradeoff between speed and accuracy
     directly supports our helix-specific floor approach.

2. **Spatially-Aware Adaptive Trajectory Optimization** (arXiv 2602.15642)
   - Controller-guided feedback iteratively refines spatially-varying constraints.
   - C² continuity ensures smooth curvature transitions.
   - Relevant: validates our approach of using different constraint bounds (floors)
     for different track sections based on controller capability.

3. **Learning Robust Agile Flight Control (NGTC)** (arXiv 2510.12611)
   - Neural-augmented controller with differential flatness feedforward from jerk/snap.
   - Relevant for future controller upgrade (priority 2 in backlog).

### Consensus
- **Strong**: Spatially-varying speed/accuracy tradeoffs improve racing performance
  (ILMPC, Spatially-Aware, CiMPCC)
- **Strong**: Controller capability limits should inform trajectory timing, not just
  dynamics (Spatially-Aware, TACO)
- **Moderate**: Helix/compound curvature requires tighter constraints than point
  curvature suggests (CiMPCC, TOPPQuad)

### Implementation Direction
Raise the TOPP compression floor for helix segments from 0.65 to 0.76. This is a
spatially-varying constraint that retains more inflated time at the helix, where
the PD controller needs more margin due to compound curvature and tight spacing.

Sweep confirmed: monotonic improvement from 0.68 to 0.80, with 0.76 as the
Pareto-optimal point (best tracking improvement per race time cost).
