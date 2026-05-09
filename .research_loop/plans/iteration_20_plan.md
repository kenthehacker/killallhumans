# Iteration 20 Plan: First-Gate S-Turn Departure Inflation + Junction Boost

## Objective
Reduce gate-3 tracking error from 0.463m to ~0.38m by adding departure-side inflation for S-turn first gates and boosting junction gate inflation. Target: 15-20% gate-3 improvement with ≤5% gate-4 regression.

## Research Basis
- CiMPCC (Li, ITSC 2024): Compound curvature at consecutive opposite turns — curvature doesn't drop between S-turn pairs
- VPMPCC (Li, 2024): Early deceleration before S-turns; approach segments to second turn need slowing
- Mastering Diverse Tracks (Yu, RA-L 2025): N→N+1 gate lookahead shapes trajectory through gate N for gate N+1
- Imitation Learning (Zhou, 2024): Opposite-direction flights need acceleration-based transition management

## Files to Modify
**1. `planning/trajectory_optimizer.py` — `_inflate_sharp_turns()` method**

### Change A: Add first-gate S-turn detection and departure inflation
After existing S-turn second detection, add:
```python
# First-gate S-turn detection: this gate starts an S-turn pair
# (next gate has opposite turn direction)
is_s_turn_first = False
if gi + 1 < n_gates - 1 and turn_angle > 0.25:
    if cross_z[gi] != 0 and gi + 1 in cross_z_forward:
        is_s_turn_first = (cross_z[gi] * cross_z_forward[gi + 1] < 0)
```

For first-gate S-turns, inflate departure segments:
```python
if is_s_turn_first:
    # Inflate EXIT segment (gate exit → next gate entry)
    depart_seg = seg_through + 1  # segment from this gate exit to next entry
    if 0 <= depart_seg < len(times):
        times[depart_seg] *= 1.08  # 8% departure inflation
```

### Change B: Junction gate compound inflation boost
For gates that are BOTH first AND second of S-turn pairs:
```python
if is_s_turn and is_s_turn_first:
    # Junction gate — cascading S-turn, extra inflation
    s_turn_inflate = 1.15  # boosted from 1.10
```

**2. `planning/trajectory_optimizer.py` — `_topp_retime()` method**

### Change C: Extend S-turn region detection to first-gate S-turns
Currently only marks segments around second-gate S-turns. Add detection for first-gate S-turns so the compound curvature boost applies to those segments too.

## Algorithm Changes
1. Pre-compute `cross_z` for gate gi+1 (forward-looking) in addition to gi-1 (backward-looking)
2. Detect is_s_turn_first by checking if cross_z[gi] and cross_z[gi+1] have opposite signs
3. Apply departure inflation (1.08) to exit segments of first-gate S-turns
4. Boost junction gate compound inflation to 1.15 (from 1.10)
5. Extend TOPP S-turn segments to include first-gate regions

## Risk Assessment
- Gate-4 may regress 5-10% (0.310→0.34m) — acceptable
- Race time may increase ~0.1s (13.88→13.98s) — acceptable under 14.0s target
- TOPP retimer may partially undo inflation — mitigated by extending S-turn regions

## Rollback Criteria
- avg error > 0.241m (>5% regression from 0.230m)
- race time > 14.1s
- any gate error > 0.7m
- gate-3 error doesn't improve

## Test Plan
1. Run unit tests after implementation
2. Run full benchmark
3. Compare per-gate errors, especially gates 2-5
