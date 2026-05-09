# Iteration 20 — Cross-Validated Research: S-Turn First-Gate Approach Inflation

## Research Synthesis (see iteration_20_synthesis.md)
Gate-3 is at the junction of cascading S-turns. It gets S-turn compound inflation (1.10) as the second gate of pair (2,3), but NO special handling as the first gate of pair (3,4). Papers (CiMPCC, VPMPCC, Mastering Diverse Tracks, Imitation Learning) all support that S-turn entry management requires lookahead to the next gate.

## Cross-Validation Challenges

### Challenge 1: Will departure inflation at gate-3 just shift the problem to gate-4?
**Assessment**: Possible but manageable. Gate-4 is at 0.310m (well under the 0.5m threshold). Even a 10-15% regression to ~0.35m would be acceptable. The S-turn coupling means some trade-off is inevitable, but the NET effect should be positive because gate-3 (0.463m) has more room to improve than gate-4 has to regress.

### Challenge 2: Won't the TOPP retimer undo the inflation?
**Assessment**: The TOPP retimer has `max_compression = 0.68` — it can compress segments by up to 32%. However, the S-turn compound curvature boost (1.2x) already limits compression in S-turn regions. We need to ensure the new first-gate S-turn segments are also marked in the `s_turn_segments` set so the TOPP retimer respects them.

### Challenge 3: Is 1.08-1.12 departure inflation enough? Or too much?
**Assessment**: The current second-gate compound inflation is 1.10 with 1.05 approach. For first-gate departure, we should start conservative at 1.08 and test. The existing failed approaches show that aggressive inflation (>15%) leads to race time regressions.

### Challenge 4: Are there other gates affected by this change?
**Assessment**: Yes — gate-4 is also a junction gate (second of 3,4 AND first of 4,5). Gate-2 is the first of pair (2,3) but NOT a junction. Gate-5 is only the second of pair (4,5). The change will affect gates 2-5 in the cascading S-turn sequence.

## Validated Implementation Plan

### Change 1: Add first-gate S-turn detection
In `_inflate_sharp_turns`, after the existing S-turn second detection, add detection for when the current gate is the FIRST of an S-turn pair (next gate has opposite cross_z direction).

### Change 2: Apply departure-side inflation for S-turn first gates
For the first gate of an S-turn pair, inflate the EXIT segments (from gate exit toward next gate entry) by 1.08.

### Change 3: Boost junction gate inflation
For gates that are BOTH first and second of S-turn pairs, increase compound inflation from 1.10 to 1.15.

### Change 4: Expand TOPP retimer S-turn regions
In the TOPP retimer's S-turn segment detection, also mark segments around first-gate S-turn gates for compound curvature boost.

## Expected Impact
- Gate-3 error: 0.463m → ~0.38-0.42m (targeted 15-20% improvement)
- Gate-4 error: 0.310m → 0.31-0.35m (acceptable small regression)
- Avg error: 0.230m → ~0.225m
- Race time: 13.88s → ~13.95-14.00s (slight increase from inflation)

## Rollback Criteria
- If avg tracking error increases > 5% (above 0.241m), revert
- If race time exceeds 14.1s, revert
- If any gate error exceeds 0.7m, revert
