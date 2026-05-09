# Iteration 6 — Research Synthesis: Adaptive Entry/Exit Offsets

## Problem Statement
Gate-7 (helix entry) has 0.932m tracking error — 2.3x the average. The 94° turn at gate-6 and 69° turn at gate-7 are both executed with a fixed 0.4m entry/exit offset, identical to gentle 22° turns like gate-1. This forces minimum-snap polynomials to create sharp bends in short distances.

## Research Consensus

### 1. Variable Offsets Are Used in Practice
"On Your Own" (Romero 2025) uses:
- Normal gates: ±0.4m entry/exit along gate normal
- Split-S maneuver: -0.4m entry / +1.25m exit
This is the only competitive system to publish its waypoint placement strategy. The Split-S uses a 3x larger exit offset — clear evidence that aggressive maneuvers need longer offsets.

### 2. Gates Are Regions, Not Points
TOGT Planner (Qin 2024) explicitly models gates as traversable regions. The optimal trajectory doesn't pass through gate centers — it passes through strategically chosen points within the gate opening. Our entry/exit waypoints are a discrete approximation of this concept.

### 3. Trajectory Adaptation Reduces Tracking Error
TACO (Sanghvi 2025) demonstrates that adapting trajectory parameters based on upcoming trajectory characteristics "significantly reduces tracking error." While TACO adapts controller gains, the same principle applies to trajectory shape.

### 4. Per-Section Aggressiveness Tuning Works
LMPC (Zhao 2025) uses an adaptive cost function that varies aggressiveness per track section. Sections with sharp turns get more conservative treatment. Result: 60.85% lap time improvement.

## Proposed Approach
Scale entry/exit offsets based on turn angle at each gate:
- Turn angle < 30°: offset = 0.25m (gentle, direct path)
- Turn angle 30-60°: offset = 0.4m (moderate, matches "On Your Own" baseline)
- Turn angle 60-90°: offset = 0.7m (sharp, needs more polynomial room)
- Turn angle > 90°: offset = 1.0m (very sharp, like Split-S scaling)

Expected impact: Gate-7 error 0.932m → ~0.5m, avg error 0.398m → ~0.35m, minimal race time impact.

## Contradictions / Risks
- Longer offsets add more waypoints further apart, potentially increasing total path length and race time
- The offset scaling must not regress gentle gates (currently gate-1 at 0.108m error)
- The L-BFGS optimizer may partially compensate for poor offsets, masking the benefit
