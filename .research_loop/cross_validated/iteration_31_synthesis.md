# Iteration 31 — Research Synthesis: Gate-7 Helix Entry Optimization

## Papers Analyzed (New)
1. **Online Velocity Profile Generation and Tracking** (Langmann/Ogretmen, TUM, 2025) — apex-based forward-backward velocity profiling
2. **Nonlinear Receding-Horizon Differential Game for Drone Racing** (Sung et al., Kyoto, 2025) — path-following with projection-point dynamics
3. **Real-time Planning of Minimum-time Trajectories for Agile UAV Flight** (Teissing et al., CTU Prague, 2024) — LTD thrust decomposition, gradient-based waypoint velocity optimization

## Existing Papers (Referenced from Previous Iterations)
- TOPPQuad (Mao 2024): dynamic feasibility via speed profile optimization
- FBGA (Piazza 2025): forward-backward within 0.36% of optimal control
- CiMPCC (Li 2024): compound curvature for sequential turns
- VPMPCC (Li 2024): early deceleration before S-turns

## Root Cause Analysis

**Gate-7 has been the worst gate (0.276→0.284m) for 10+ iterations.** Geometric analysis reveals:

### The asymmetry
- Gate-6 (93.9° turn, curvature 0.216): gets **25% inflation** (angle-based: severity=1.0 → 1.25x)
- Gate-7 (68.5° turn, curvature 0.269): gets only **8.7% inflation** (proximity-based: 1.087x)

Gate-7 has **higher curvature** than gate-6 but gets **3x less inflation** because the 60° angle threshold creates a cliff — gate-6's 94° turn jumps to full severity, while gate-7's 68.5° only reaches severity=0.278.

### The helix blind spot
The existing compound curvature boost (1.2x) applies ONLY to S-turn segments (opposite-direction consecutive turns). The helix (gates 6-12) has ALL same-direction turns — it's NOT detected as needing compound treatment. Yet helix sections have sustained high curvature without recovery straights, making them harder than isolated turns.

### Supporting evidence from research
- **CiMPCC (Li 2024)**: Compound curvature doesn't drop between consecutive same-direction turns. The current S-turn-only detection misses this.
- **Online Velocity Profile (Ogretmen 2025)**: Apex-based velocity profiling identifies local curvature maxima and iteratively computes the maximum feasible velocity at each apex. Gate-7 IS a curvature apex in the helix.
- **FBGA (Piazza 2025)**: Forward-backward propagation naturally creates speed dips at high-curvature points, but only if the curvature signal is correct.
- **TOPPQuad (Mao 2024)**: Dynamic feasibility requires considering sustained curvature, not just point curvature.

## Proposed Approach

**Add helix detection and compound curvature treatment**, analogous to the existing S-turn mechanism:

1. **In `_inflate_sharp_turns`**: Detect helix sections (3+ consecutive same-direction turns with inter-gate distances < 7m). Apply helix compound inflation:
   - Helix entry gate (first helix gate): extra ~3-4% inflation
   - Interior helix gates: extra ~2% inflation (sustain the speed reduction)

2. **In `_topp_retime`**: Add helix segments to the curvature boost set (like `s_turn_segments`). Apply 1.15x curvature boost for helix segments (analogous to 1.2x for S-turns, but smaller since helix curvature is already high).

### Risk Assessment
- Race time may increase by ~0.05-0.15s (helix section slower)
- S-turn parameters are UNCHANGED — no basin switching risk
- Only the helix section (gates 6-12) is affected
- ILC can still optimize within the slower helix

### Expected Impact
- Gate-7: 0.284m → ~0.255-0.265m
- Race time: 13.68s → ~13.73-13.80s
- Avg error: 0.191m → ~0.186-0.189m (net improvement)

## Consensus
Strong agreement across papers that:
1. Sustained high-curvature sections need compound treatment (CiMPCC, TOPPQuad)
2. Forward-backward speed profiling naturally handles this if curvature signal is correct (FBGA, Online VP)
3. The curvature boost should be proportional to section severity, not a binary flag
