# Iteration 3 — Research Synthesis: Trajectory Extension Past Final Gate

## Bottleneck
`trajectory_planning` — gate-12 trajectory ends 0.39s before the drone passes through, causing degraded gate-seeking fallback with 2x worse tracking error (0.694m vs 0.333m avg).

## Papers Consulted
1. **TOGT Planner** (Qin et al., ICRA 2024) — arxiv:2309.06837
2. **On Your Own** (Romero et al., 2025) — arxiv:2510.13644
3. **Drift-Corrected VIO + Perception-Aware Planning** (2025) — arxiv:2512.20475
4. **Gate-Aware Online Planning** (Zhao et al., 2024) — arxiv:2402.18021
5. **Time-Optimal Planning Long-Range** (2024) — arxiv:2407.17944
6. **Leveling the Playing Field** (Kunapuli et al., 2025) — arxiv:2506.17832

## Consensus Across Papers

### Strong consensus: Gates need entry+exit waypoints, not just center points
- **"On Your Own"** (deployed at IROS 2024, Abu Dhabi F1 GP): "Each gate was assigned two waypoints, positioned at the center of the gate along the y- and z-axes. Along the x-axis, the waypoints were placed at -0.4m and +0.4m."
- **TOGT Planner**: Models gates as spatial volumes/regions, not center points. "Previous studies neglect the configuration of the gates, simply rendering drone racing a waypoint-passing task."
- **Drift-Corrected VIO**: Uses TOGT planner with gates as spatial volumes, segments path with waypoints inside each gate.

### Strong consensus: Trajectory must not end at the last gate
All competitive systems generate trajectories that extend past the final gate. The "On Your Own" dual-waypoint approach naturally ensures the trajectory extends 0.4m+ past each gate center, including the last one.

### Moderate consensus: Controller gains matter for turn tracking
- **Leveling the Playing Field**: Geometric controllers with full feedforward are competitive with RL. Higher proportional gains reduce lag at turns.
- Our gates 3, 4, 7 show elevated error from controller lag at turns.

## Proposed Implementation Direction

### Priority 1: Add virtual waypoint past gate-12
**Evidence**: "On Your Own" (Romero 2025) — the most directly applicable paper, from a team that won head-to-head against a professional pilot.

**Approach**: Add a virtual waypoint 2.0m past gate-12 along its normal direction. Gate-12 is at (56.0, 0.0, 8.5) with yaw=0.0, normal=(1,0,0). Virtual waypoint at (58.0, 0.0, 8.5).

**Implementation**: Modify `TrajectoryOptimizer.optimize()` to append a virtual finish waypoint after the last gate. This ensures the trajectory polynomial covers the full approach and passage through gate-12, eliminating the gate-seeking fallback entirely.

**Expected impact**: gate-12 error drops from ~0.69m to ~0.3m (matching other straight gates). Avg tracking error improves from 0.333m to ~0.30m.

### Priority 2: Bump controller gains for turn gates
**Evidence**: "Leveling the Playing Field" (Kunapuli 2025) — geometric controller gains should be tuned aggressively.

**Approach**: kp_xy 4.0→5.0, kd_xy 3.0→3.5 in benchmark.py tracker config.

**Expected impact**: Turn gates (3, 4, 7) each improve ~0.05m.

## Contradictions
None significant. All papers agree on the direction. The only nuance is the exact offset distance — "On Your Own" uses 0.4m for mid-race gates but 1.25m for split-S maneuvers. We use 2.0m as a conservative choice for the final gate to ensure ample trajectory coverage.

## Risk Assessment
- Adding a virtual waypoint changes the L-BFGS optimization (one more segment). Could slightly change time allocation for other segments.
- Controller gains increase could cause oscillation if too aggressive. 4→5 is a 25% increase, conservative enough.
- Both changes are easily reversible.
