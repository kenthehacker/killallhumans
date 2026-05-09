# Iteration 4 Research Synthesis: Double-Waypoint Gate Traversal

## Research Sources
1. **"On Your Own" (Romero et al., 2025)** — arxiv:2510.13644
   - KEY: Each gate assigned TWO waypoints at ±0.4m along gate x-axis (normal direction)
   - Waypoints centered on y/z axes
   - Forces trajectory through gate correctly, prevents cutting through banners
   - Used with time-optimal trajectory generator (full rigid-body dynamics + actuator constraints)
   - Deployed at IROS 2024 + Abu Dhabi F1 GP — outperformed professional pilot

2. **TOGT Planner (Qin et al., 2024)** — arxiv:2309.06837
   - Gates as regions, not points
   - Single waypoint per gate is an approximation that misses time-optimal solutions
   - Polynomial trajectory with differential flatness

3. **Euclidean/Non-Euclidean Trajectory Optimization (Fork & Borrelli, 2024)** — arxiv:2309.07262
   - Without entry/exit constraints, optimizer may find trajectories that enter gate then backtrack
   - High-fidelity quadrotor dynamics matter for feasibility
   - 100x faster than comparable methods

4. **Perception-Aware Planning (ETH, 2026)** — arxiv:2603.04305
   - Time-optimal with FOV constraints + convex gate representations
   - 0.07m average tracking error at 9.8m/s
   - 55% → 100% success rate improvement with perception awareness

## Consensus
- **Strong consensus**: All papers agree that treating gates as single points loses optimality
- **Strong consensus**: Entry/exit waypoints or gate region constraints improve trajectory quality
- **"On Your Own" is most directly applicable**: Same problem setting (racing, min-snap trajectory, gates with normals)
- **±0.4m offset is the standard**: Used in deployed competition system

## Key Insight for Our System
Our current trajectory goes: start → gate1_center → gate2_center → ... → gate12_center → virtual_finish

The min-snap solver connects these with polynomials, but at sharp turns (gates 3, 4, 7), the polynomial must make abrupt direction changes AT the gate. With dual waypoints:
- start → gate1_entry → gate1_exit → gate2_entry → gate2_exit → ... → virtual_finish
- The turn happens BETWEEN exit and next entry (where there's more distance)
- The trajectory smoothly passes THROUGH the gate along its normal
- This should directly reduce errors at high-turn gates (3, 4, 7)

## Implementation Direction
Apply ±0.4m entry/exit waypoints for ALL gates in `TrajectoryOptimizer.optimize()`.
Also lower minimum segment time to accommodate short (0.8m) entry-exit segments.
