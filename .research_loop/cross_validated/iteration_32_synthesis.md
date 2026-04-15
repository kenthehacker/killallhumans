# Iteration 32 — Research Synthesis: Gate-7 Helix Entry Offset Expansion

## Papers Analyzed (New)
1. **Efficient Generation of Smooth Paths with Curvature Guarantees by Mollification** (González-Calvin et al., Madrid, 2025) — curvature bounds for waypoint-interpolating paths
2. **Gate-Aware Online Planning for Two-Player Drone Racing** (PMPC, 2024) — magnetic induction line curves for gate traversal with perpendicularity constraints

## Existing Papers (Referenced from Previous Iterations)
- TOGT (Qin 2024): gates are regions, optimize pass-through point within gate opening
- On Your Own (Romero 2025): 0.4m entry/exit offset standard, **1.25m for Split-S maneuvers**
- CiMPCC (Li 2024): compound curvature for sequential same-direction turns
- Richter et al. (MIT 2013/2016): min-snap polynomial curvature scales inversely with segment length
- SCP-TOW (Shen 2024): state-triggered constraints for time-optimal waypoint flight
- Perception-Aware Planning (Qin et al., 2026): convex gate constraints + Split-S with larger offsets

## Root Cause Analysis

**Gate-7 has been the worst gate (0.284m) for 10+ iterations, and iteration 31 proved this is trajectory-shape-driven, not speed-driven.** Increasing inflation from 8.7% to 10% had <0.05% effect on gate-7, while helix interior gates (9-11) improved 7-13%.

### The geometry problem
Gate-7 is the helix entry transition: a 68.5° turn with:
- Approach distance from gate-6: 4.69m
- Entry offset: 0.4m (current ENTRY_EXIT_OFFSET)
- Through-gate segment: 0.8m (2 × 0.4m)
- Departure to gate-8: 3.64m

The min-snap polynomial must execute a 68.5° turn within 0.8m (the through-gate segment between entry and exit waypoints). This creates inherently high polynomial curvature that no amount of speed reduction can fix — the PD controller can't track the sharp geometric shape regardless of speed.

### Why larger offsets help (mathematical basis)

**Richter et al. (MIT)**: Min-snap polynomial curvature at a waypoint is proportional to turn_angle / segment_length². Doubling the offset from 0.4m to 0.8m approximately quadruples the effective area over which curvature is distributed, reducing peak curvature by ~75%.

**TOGT (Qin 2024)**: Gates are regions, not points. The optimal trajectory doesn't need to pass through the exact gate center — it can use the full gate opening width (~1.5m) to create smoother paths. Our system currently forces passage through the center.

**On Your Own (Romero 2025)**: Uses 1.25m offsets (3.125× our 0.4m) for Split-S maneuvers. The Split-S has a similar geometric challenge to our helix entry: a large heading change in a short distance. If 1.25m is appropriate for Split-S (which involves a 180° flip), then 0.6-0.8m should be appropriate for our 68.5° helix entry turn.

**PMPC (2024)**: Uses magnetic induction line curves that are perpendicular to gates at entry/exit. The curve shape between gates naturally adapts to the inter-gate distance and angular change. Larger angular changes produce longer curves, analogous to larger offsets.

### Why we can't use too-large offsets

**Iteration 6 failed approach**: Wide offset ranges (0.25-1.0m) distorted waypoint geometry. Asymmetric offsets also failed because short segments near gate centers caused L-BFGS to over-slow.

**Key constraint**: Adjacent gates' exit and entry waypoints must not overlap. Gate-6→gate-7 distance is 4.69m. If both use 0.8m offsets, we'd have: gate-6-exit at 0.8m past gate-6 + gate-7-entry at 0.8m before gate-7 = leaving 4.69 - 0.8 - 0.8 = 3.09m between them. This is fine. But if we went to 1.25m: 4.69 - 1.25 - 1.25 = 2.19m, which is getting tight.

## Proposed Approach

**Adaptive entry/exit offset based on local curvature at each gate.**

The algorithm:
1. Compute turn angle at each gate center (already done in `_inflate_sharp_turns`)
2. For gates with turn_angle > threshold (e.g., 60°), use larger offset
3. Scale: 0.4m baseline → up to 0.8m for the sharpest turns
4. Ensure adjacent waypoints don't overlap (maintain minimum 2.0m between exit[i] and entry[i+1])

Specifically for the helix entry (gate-7):
- Current: 0.4m offset → 0.8m through-gate segment
- Proposed: 0.7m offset → 1.4m through-gate segment
- Expected curvature reduction: ~(0.4/0.7)² ≈ 67% of original peak curvature
- Expected gate-7 error: 0.284m → ~0.255-0.265m

### Risk Assessment
- Only the waypoint placement changes — all downstream optimization (L-BFGS, inflation, TOPP) runs unchanged
- The iteration 6 failure was with ADAPTIVE offsets in L-BFGS, not with a fixed per-gate offset computation
- S-turn section (gates 1-6) should be minimally affected since their turn angles are smaller
- If gate-3 (48.2° turn) also gets a slight offset increase, it could improve too

## Consensus
Strong agreement across papers that:
1. Larger waypoint spacing reduces polynomial curvature at turns (Richter, TOGT, On Your Own)
2. Variable offsets per gate are the norm in competitive systems (On Your Own uses 0.4m/1.25m)
3. The through-gate segment length directly controls the curvature envelope at each gate
4. Safe range is 0.4-0.8m for normal racing turns, up to 1.25m for extreme maneuvers
