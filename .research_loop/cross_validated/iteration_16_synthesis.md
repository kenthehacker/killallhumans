# Iteration 16 — Research Synthesis: S-Turn Time Inflation

## Bottleneck
Gates 3-4 form an S-turn (consecutive opposite-direction turns) with avg tracking errors of 0.452m and 0.465m — the worst gates on the track. The TOPP retimer from iter 15 speeds up straight segments but exacerbates the S-turn by increasing approach speed.

## Track Geometry Analysis
- Gate-2→3: direction (10, -6, -0.7), heading south — LEFT turn
- Gate-3→4: direction (10, 3, 0.7), heading north — RIGHT turn (opposite direction)
- Gate-4→5: direction (8, -3, -0.4), heading south — LEFT turn (another reversal)

Turn angles: gate-3 ≈ 47.6°, gate-4 ≈ 37.3°. Both below the 60° angle-based inflation threshold. The centripetal check (a_c = v²κ) operates independently at each gate, missing the compound S-turn effect.

## Papers Analyzed (new: 2, previously analyzed: 4 directly relevant)

### New Papers
1. **CiMPCC** (Li et al., ITSC 2024): Curvature-integrated MPCC maps smoothed track curvature to velocity reference. Key finding: smoothed curvature naturally captures S-turn compound effect because curvature doesn't fully drop between consecutive turns. 11.4-12.5% lap time improvement. **α=1.0 (linear mapping) is optimal.**

2. **VPMPCC** (Li et al., 2024): Data-driven velocity prediction via Bayesian Optimization learns optimal speed profiles. Key finding: learned profiles show "decelerate early, stay slow through compound turns" pattern. The APPROACH segment to the second S-turn needs slowing, not just the turn segments.

### Previously Analyzed (directly relevant)
3. **TACO** (Sanghvi 2025): Trajectory-aware controller optimization. Confirms that trajectory parameters (timing, shape) should adapt to local curvature-speed characteristics. Relevant for understanding why static inflation factors fail at S-turns.

4. **Alternating Peak Optimization** (de Vries/Foehn, ECC 2024): Peak-normalized time allocation ensures each segment saturates its hardest constraint. The S-turn segments are NOT saturating their curvature constraint because inflation is calibrated for individual turns, not compound turns.

5. **TOPPQuad** (Mao, IROS 2024): Geometry-timing decoupling. TOPP speed limits should handle curvature-aware slowdown — but our implementation uses point curvature at each waypoint, missing the compound S-turn effect.

6. **FBGA** (Piazza, RA-L 2025): Forward-backward propagation is near-optimal for speed profiling. Validates our TOPP retimer approach but highlights that the speed LIMITS (not the propagation) are the source of the S-turn issue.

## Research Consensus

All papers converge on three principles for handling S-turns / chicanes:

1. **Compound curvature**: Treat consecutive opposite-direction turns as a single compound maneuver with higher effective curvature than either individual turn.

2. **Early deceleration**: The approach segment to an S-turn needs extra time — the drone must begin lateral velocity reversal before reaching the turn apex.

3. **Sustained low speed**: Through the compound turn, speed should remain low (not spike between the two turn apices). The inter-turn section is NOT a straight — it's a transition.

## Contradictions
None significant. All papers agree that individual-turn treatment of S-turns is insufficient. The disagreement is in approach: CiMPCC uses curvature smoothing, VPMPCC uses learned profiles, TOPP uses speed limits. For our offline planning, any approach works.

## Proposed Implementation Direction

**Approach: S-turn detection + compound inflation + approach segment inflation**

Based on the strongest evidence (CiMPCC smoothing principle + VPMPCC early deceleration):

1. **Detect S-turns**: Consecutive gates where cross-product of turn vectors changes sign (opposite lateral direction)
2. **Compound curvature**: For the second turn of an S-turn, multiply effective curvature by 1.3-1.5x
3. **Approach inflation**: Inflate the segments between the two S-turn gates (gate-3 exit → gate-4 entry) by 10-15%
4. **TOPP speed limit**: In the TOPP retimer, use compound curvature for segments within S-turn regions

This is a targeted change to `_inflate_sharp_turns` with a supporting change to `_topp_retime`, affecting only the S-turn gates while preserving the speed improvements on straights.
