# Iteration 20 — Research Synthesis: S-Turn First-Gate Approach Inflation

## Bottleneck
Gate-3 tracking error = 0.463m (worst gate). Gate-3 sits at the JUNCTION of two S-turn pairs: it is both the second gate of pair (2,3) AND the first gate of pair (3,4). Current code only applies compound inflation when a gate is the second of a pair.

## Gate Geometry Analysis
- gate-2: 53.2° turn, cross_z=-100, S-turn first (pair 2→3)
- gate-3: 48.2° turn, cross_z=+90, S-turn JUNCTION (second of 2→3, first of 3→4)
- gate-4: 37.8° turn, cross_z=-54, S-turn JUNCTION (second of 3→4, first of 4→5)
- gate-5: 35.4° turn, cross_z=+40, S-turn second (pair 4→5)

The region gates 2-5 is a cascading S-turn sequence — the drone must reverse lateral direction at every gate.

## Papers Analyzed (New in This Iteration)
1. **Mastering Diverse Tracks** (Yu et al., RA-L 2025) — zigzag track primitives for S-turns
2. **Imitation Learning Time-Optimal Control** (Zhou et al., 2024) — transition phase velocity management
3. **Learning Agile Gate Traversal** (Sun et al., NUS 2025) — adaptive MPC cost scheduling

## Key Research Insights

### Consensus: The First Gate of an S-Turn Needs Approach/Departure Management
1. **VPMPCC (Li 2024)**: "Early deceleration before S-turns" — the approach segment to the SECOND turn of an S-turn needs slowing. But when gate-3 is the first of pair (3,4), the DEPARTURE from gate-3 is the approach to gate-4. Current code doesn't inflate gate-3's departure.
2. **CiMPCC (Li 2024)**: Compound curvature for consecutive opposite turns. At a junction gate like gate-3, compound curvature is even higher because there's no straight recovery between S-turn pairs.
3. **Mastering Diverse Tracks (Yu 2025)**: The RL policy receives N→N+1 gate vector as input, allowing it to shape the approach through gate N considering gate N+1. This is a multi-step lookahead that our single-gate inflation doesn't have.
4. **Imitation Learning (Zhou 2024)**: For opposite-direction flights (S-turns), the transition uses acceleration-based entry thresholds (not velocity). The drone detects when it's decelerating and bridges with a polynomial to the next segment.

### Consensus: Junction Gates Need Extra Inflation
A gate that is simultaneously the second of one S-turn pair AND the first of another has:
- Residual lateral velocity from the previous S-turn
- No recovery distance before the next reversal
- Higher effective curvature than a single S-turn

### Proposed Approach: First-Gate S-Turn Inflation
For gates identified as the FIRST gate of an S-turn pair:
1. Inflate the EXIT segments (departure from gate-3 toward gate-4 entry) by 1.08-1.12
2. For JUNCTION gates (both first AND second), apply a multiplicative boost to the existing compound inflation: 1.10 → 1.15

This gives the controller more time to:
- Settle at the gate-3 position after the (2→3) turn
- Reverse lateral velocity before accelerating toward gate-4

### Ranking of Actionable Changes
1. **First-gate S-turn departure inflation** — direct fix for gate-3 error (Priority 1)
2. **Junction gate compound inflation boost** — multiplicative boost for cascading S-turns (Priority 1, same change)
3. **TOPP retimer S-turn region expansion** — extend compound curvature boost to first-gate regions too (Priority 2)

## Risk Assessment
- Gate-4 may regress slightly if departure inflation from gate-3 slows the approach to gate-4
- But gate-4 is currently at 0.310m (well under target), so some regression is acceptable
- Gate-2 and gate-5 are far from limits (0.215m, 0.155m) — unlikely to be affected
