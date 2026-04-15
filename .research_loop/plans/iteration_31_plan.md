# Iteration 31 Plan — Helix Compound Curvature Treatment

## Objective
Reduce gate-7 tracking error from 0.284m toward 0.255m by adding helix detection and compound curvature treatment to both `_inflate_sharp_turns` and `_topp_retime`.

## Research Basis
- CiMPCC (Li, ITSC 2024): Compound curvature for sequential same-direction turns
- TOPPQuad (Mao, IROS 2024): Dynamic feasibility requires sustained curvature awareness
- FBGA (Piazza, RA-L 2025): Forward-backward speed profiling at curvature apexes
- Online Velocity Profile (Ogretmen/Langmann, TUM 2025): Apex-based velocity refinement for high-curvature sections

## Files to Modify
1. `planning/trajectory_optimizer.py` — `_inflate_sharp_turns()` and `_topp_retime()`

## Algorithm Changes

### Change 1: Helix detection in `_inflate_sharp_turns`
After the existing S-turn detection, add helix detection:
```
helix_gates = set()  # gate indices in helix sections
helix_entry_gates = set()  # first gate of each helix section
For each consecutive pair of gates (gi, gi+1):
    If cross_z[gi] and cross_z[gi+1] have SAME sign:
        AND min(dist_to_prev, dist_to_next) < 7.0m:
        Mark gi+1 as helix gate
Track consecutive helix gates. If 3+ consecutive → confirmed helix.
First gate = helix_entry.
```

Apply helix compound inflation:
- Helix entry gate: `inflate = max(inflate, 1.12)` — 12% inflation (was only getting 8.7%)
- Helix interior gates: no extra (existing proximity handles them)

### Change 2: Helix detection in `_topp_retime`
Add `helix_segments` set analogous to `s_turn_segments`:
```
Detect helix gates (same logic as above).
For each helix gate gi, mark segments 2*gi-1 to 2*gi+2 as helix_segments.
```

Apply curvature boost:
```
if i in helix_segments and k > 1e-4:
    k *= 1.15  # 15% compound curvature boost for helix
```
Note: S-turn still gets 1.2x (unchanged), helix gets 1.15x.

## Risk Assessment
- S-turn parameters completely UNCHANGED → no basin switching risk
- Only helix section (gates 6-12) affected
- Expected race time increase: +0.05-0.15s (acceptable if gate-7 improves)
- ILC continues to optimize within the slower helix

## Rollback Criteria
- If avg tracking error > 0.200m (5% worse), revert
- If race time > 14.0s, revert
- If any gate error exceeds 0.35m, revert

## Test Plan
1. Run unit tests first (should pass unchanged)
2. Run full benchmark
3. Compare gate-7 error, avg error, race time
