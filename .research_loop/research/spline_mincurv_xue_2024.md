# Spline-Based Minimum-Curvature Trajectory Optimization for Autonomous Racing

- **URL**: https://arxiv.org/abs/2309.09186
- **Authors**: Haoru Xue, Tianwei Yue, John M. Dolan
- **Year**: 2023 (ICRA 2024)
- **Venue**: ICRA 2024

## Key Contribution

This paper presents a B-spline-based minimum-curvature trajectory optimization that achieves comparable optimality to state-of-the-art methods with 90% fewer decision variables, reducing computation from seconds to milliseconds. The approach is particularly relevant for chicanes and S-turn sections where curvature minimization directly enables higher speeds through complex turn sequences.

The key insight for our S-turn optimization: minimum-curvature paths through S-turns/chicanes allow higher speeds because curvature is the binding constraint for centripetal acceleration. By smoothing the racing line through the S-turn (reducing peak curvature), we can either increase speed or reduce tracking error at the same speed.

## Technical Approach

The trajectory is represented as a C2-continuous B-spline curve with control points as decision variables. The optimization minimizes:

min ∫ κ²(s) ds (integrated squared curvature)

subject to track boundary constraints at each control point. The B-spline representation guarantees C2 continuity (smooth curvature transitions) while keeping the optimization problem low-dimensional.

Key advantages over waypoint-based methods:
1. **Smooth curvature**: B-spline guarantees no curvature discontinuities at waypoints
2. **Fewer variables**: 90% reduction in decision variables vs fine-grained discretization
3. **Millisecond solve time**: Enables real-time adaptation and iterative refinement
4. **Direct manipulation**: Control points can be dragged to modify the trajectory

The approach explicitly handles chicanes/S-bends where the racing line must cross the track centerline, requiring careful curvature management to avoid oscillation.

## Results

- 90% reduction in decision variables vs state-of-the-art
- Computation time: milliseconds (vs seconds)
- Similar optimality level to fine-grained methods
- Demonstrated on multiple track configurations including chicanes

## Relevance to Our System

**Moderately relevant.** Our racing line is computed using L-BFGS optimization with lateral offsets and entry/exit points per gate. The curvature through the S-turn (gates 2-4) is determined by the lateral offset geometry. The S-turn's high curvature is what makes the TOPP floor binding — the speed must be reduced to maintain trackability at the curvature peak.

Key insights for our S-turn floor optimization:
1. The S-turn curvature is fundamentally geometric — it's determined by the gate positions and lateral offsets, not the speed profile
2. To reduce gate-3 error, we can either: (a) slow down through the S-turn (raise TOPP floor), or (b) reduce curvature (modify racing line offsets). Option (b) failed in iter 32 due to basin switching
3. Therefore option (a) — raising the S-turn TOPP floor — is the correct approach, consistent with this paper's finding that curvature is the binding constraint

## Actionable Takeaways

1. **Curvature is king**: In S-turns, peak curvature determines minimum safe speed. Our TOPP floor should be set based on the curvature at gate-3, not a uniform value
2. **C2 continuity matters**: Smooth speed transitions reduce tracking error at S-turn entry/exit. Our TOPP retimer should ensure smooth velocity transitions between sections
3. **Racing line curvature optimization is separate from speed optimization**: Since lateral offset changes cause basin switching (failed in iter 32), we must work with the fixed racing line geometry and optimize speed only
4. **S-turn specific analysis**: The gate-3 S-turn is analogous to a chicane — the trajectory must reverse lateral direction, creating peak curvature. This peak curvature determines the binding speed constraint

## Limitations & Caveats

- Designed for car racing (minimum-curvature = maximum speed), not quadrotor racing
- Our trajectory is polynomial (min-snap), not B-spline, so direct adoption of the formulation isn't possible
- The paper doesn't address tracking error explicitly — only trajectory smoothness

## Key Parameters / Constants

- 90% decision variable reduction from B-spline parameterization
- C2 continuity guaranteed by B-spline formulation
- Curvature-speed relationship: v_max ∝ 1/√κ (from centripetal acceleration limit)
- Chicane/S-turn: curvature peak at direction reversal point
