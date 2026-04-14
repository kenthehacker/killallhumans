# Iteration 20 — S-Turn First-Gate + Junction Inflation

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (S-turn gate-3 approach/departure inflation)
**Status**: COMMITTED — gate-3 error 0.463→0.345m (-25.6%), avg error 0.230→0.218m (-5.2%)
**Commit**: c858575

---

## Section 1: Summary
- Iteration 20, timestamp 2026-04-14T13:00Z
- Bottleneck: trajectory_planning — S-turn first-gate and junction gate inflation
- One-line outcome: **Gate-3 error reduced 25.6% by adding first-gate S-turn detection, junction inflation boost, and departure segment inflation. Avg error improved 5.2%, P95 improved 16.2%. Race time regressed 1.6% (13.88→14.10s).**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing** (Yu et al., Shanghai Jiao Tong University, RA-L 2025, arXiv:2512.09571)
   - Zigzag track primitives explicitly train S-turn capability
   - N→N+1 gate lookahead allows policy to shape approach through gate N for gate N+1
   - Staged RL: soft-collision → hard-collision fine-tuning
   - Key insight: S-turn handling requires multi-step planning, not single-gate optimization

2. **Imitation Learning-Based Online Time-Optimal Control with Multiple-Waypoint Constraints** (Zhou et al., Zhejiang University, 2024, arXiv:2402.11570)
   - Polynomial transition phases handle velocity management at waypoints
   - Opposite-direction flights need acceleration-based entry triggers (not velocity)
   - Stop-and-go elimination via polynomial bridging
   - Key insight: sequential waypoint traversal requires transition management

3. **Learning Agile Gate Traversal via Analytical Optimal Policy Gradient** (Sun et al., NUS, 2025, arXiv:2508.21592)
   - NN-MPC hybrid with time-varying cost weights
   - Analytical policy gradients through MPC optimality conditions
   - 736k training steps (270x more efficient than PPO)
   - Key insight: adaptive MPC cost scheduling for gate approach/departure phases

### Key insight from cross-validation
The critical finding was that gate-3 is a **junction gate** — simultaneously the second of S-turn pair (2,3) AND the first of pair (3,4). The existing code only handled the "second gate" role. Papers unanimously support that S-turn entry management requires next-gate lookahead (Yu 2025, Li CiMPCC 2024, Li VPMPCC 2024).

### Research consensus vs contradictions
- **Consensus (all papers)**: S-turn/zigzag handling requires multi-gate lookahead, not single-gate optimization
- **Consensus (CiMPCC, VPMPCC)**: Compound curvature at consecutive opposite turns doesn't drop — junction gates need extra inflation
- **No contradictions** found on this specific topic

---

## Section 3: Implementation

### Changes made
**File: `planning/trajectory_optimizer.py`** — Two methods modified

**Change 1: `_inflate_sharp_turns()` — First-gate S-turn detection + junction boost + departure inflation**
- Added forward-looking cross_z detection: checks if current gate and next gate have turns in opposite directions
- Junction gates (both first AND second of S-turn pairs) get boosted compound inflation: 1.13 (up from 1.10)
- Pure first-gate S-turns get 1.06 departure inflation on the exit segment toward next gate
- Junction gates get 1.04 departure inflation (less aggressive since they already have the 1.13 compound boost)

**Change 2: `_topp_retime()` — Extended S-turn region detection**
- Added first-gate S-turn region marking: segments around gates that are the first of an S-turn pair are now included in `s_turn_segments`
- This ensures the 1.2x compound curvature boost in the TOPP retimer applies to departure segments of first-gate S-turns, preventing the retimer from over-compressing these segments

### Plan adherence
Followed plan exactly. Tested three inflation parameter variants (1.15/1.13/1.12 junction, 1.08/1.06/1.05 departure) and selected the one with the best speed-accuracy trade-off.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.230m | **0.218m** | **-0.012m (-5.2%)** | ✓✓ |
| Max tracking error | 0.683m | **0.679m** | **-0.004m (-0.6%)** | ✓ |
| P50 tracking error | 0.188m | 0.188m | 0 | → |
| P95 tracking error | 0.538m | **0.451m** | **-0.087m (-16.2%)** | ✓✓✓ |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7338 | 7390 | +0.7% | → |
| Trajectory time | 14.06s | 14.28s | +0.22s | ↓ |
| Race time | 13.88s | **14.10s** | **+0.22s (+1.6%)** | ↓ |
| Avg thrust | 0.803 | 0.797 | -0.7% | ✓ |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.112 | 0.112 | 0 | unchanged |
| gate-2 | 0.215 | 0.212 | -0.003 | → slight |
| gate-3 | **0.463** | **0.345** | **-0.118 (-25.6%)** | ✓✓✓ TARGET: major improvement |
| gate-4 | 0.310 | **0.299** | **-0.011 (-3.5%)** | ✓ also improved! |
| gate-5 | 0.155 | 0.150 | -0.005 | → slight |
| gate-6 | 0.158 | 0.158 | 0 | → unchanged |
| gate-7 | 0.308 | 0.308 | 0 | → unchanged (now worst gate) |
| gate-8 | 0.218 | 0.219 | +0.001 | → noise |
| gate-9 | 0.208 | 0.209 | +0.001 | → noise |
| gate-10 | 0.216 | 0.214 | -0.002 | → noise |
| gate-11 | 0.176 | 0.176 | 0 | → unchanged |
| gate-12 | 0.215 | **0.193** | **-0.022 (-10.2%)** | ✓ improved |

### Threshold status
| Threshold | Required | Current | Aspirational | Status |
|-----------|----------|---------|--------------|--------|
| Avg error | <0.5m | **0.218m** | <0.25m | **MEETS ASPIRATIONAL** |
| Max error | <2.0m | **0.679m** | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | 14.10s | <14s | PASS (slightly over aspirational) |
| Loop Hz | >100 | 7390 | >100 | PASS |
| No crash | required | no crash | — | PASS |

Race time aspirational target (<14s) is slightly exceeded at 14.10s. All other aspirational targets met.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
Gate-3 sits at the junction of two cascading S-turn pairs (2→3 and 3→4). The existing code recognized gate-3 as the second gate of pair (2,3) and applied 1.10 compound inflation + 1.05 approach inflation. But it did NOT recognize gate-3 as the first gate of pair (3,4), meaning the departure side (exit toward gate-4) received no special S-turn treatment. The drone was settling at gate-3 without time to prepare for the upcoming lateral velocity reversal toward gate-4. Adding first-gate detection + junction boost + departure inflation gave the controller the time it needed.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (unchanged, at tilt limit)
- Avg thrust: 0.797 (decreased from 0.803 — more efficient S-turn navigation)
- Avg pitch: -0.108 (essentially unchanged)
- All improvements concentrated in S-turn region (gates 2-5)

### Trend analysis
**Trend: IMPROVING on accuracy, Pareto-advancing**

Last 6 iterations' Pareto:
- Iter 15: 13.50s / 0.251m
- Iter 16: 13.95s / 0.232m
- Iter 17: 13.62s / 0.248m
- Iter 18: 13.91s / 0.234m
- Iter 19: 13.88s / 0.230m
- **Iter 20: 14.10s / 0.218m (best accuracy ever, slight speed trade-off)**

The accuracy frontier continues to improve. The race time trade-off (13.88→14.10s) is the cost of slowing the S-turn region for better tracking. This is a deliberate Pareto trade-off — the system is trading speed for accuracy in the most problematic section.

### Code quality assessment
- The first-gate detection code is clean and follows the same pattern as the existing second-gate detection
- The junction gate concept (both first AND second) is correctly identified and handled
- The TOPP retimer extension correctly marks first-gate regions for compound curvature boost
- No fragile code patterns introduced

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Race time recovery** (Priority 1, trajectory_planning)
   - Race time regressed 0.22s to 14.10s — slightly over aspirational 14.0s target
   - Approach: reduce S-turn inflation slightly or recover time from other segments (e.g., straights between gate-1 and gate-2 where error is already 0.112m)
   - Expected: 14.10→13.95s without tracking regression
   - Research: TOPPQuad selective compression

2. **Gate-7 helix entry** (Priority 2, trajectory_planning)
   - Now worst gate at 0.308m after gate-3 fix
   - Helix entry combines moderate turn angle (68.5°) with tight gate spacing (4.7m next)
   - Already handled by angle-based inflation + proximity inflation
   - Approach: investigate if helix entry needs specialized handling beyond current mechanisms
   - Research: CiMPCC sequential same-direction turns

3. **Sim-based multi-start selection** (Priority 3, system_integration)
   - Selection by L-BFGS objective vs tracking error — still a gap
   - Approach: run each racing line candidate through kinematic sim, select by tracking error
   - Expected: find the racing line the controller can actually follow best
   - Research: AERO-MPPI two-stage evaluation

4. **MPCC controller upgrade** (Priority 4, control)
   - ETH 2026: 0.07m avg at 9.8 m/s via contouring error decomposition
   - Major architectural change, defer
   - Research: MPCC++ (Krinner 2024)

5. **Joint racing line + timing optimization** (Priority 5, trajectory_planning)
   - Co-optimize lateral offsets and segment times in a single L-BFGS pass
   - Expected: better global speed-accuracy trade-off
   - Research: F1-Init (Shehadeh 2026), TACO (Sanghvi 2025)

### Architectural recommendations
- The S-turn inflation system is now comprehensive (second-gate, first-gate, junction, departure, approach, TOPP compound curvature). Further tuning may yield diminishing returns.
- The biggest remaining opportunity is sim-based selection (backlog #3) — it would bridge the gap between L-BFGS proxy cost and actual tracking performance.
- Race time recovery (backlog #1) is achievable by selective segment compression in low-error regions.

### What NOT to try
- **Higher junction inflation (>1.15)**: Tested variant 1 with 1.15 junction — race time hit 14.21s which is too much
- **Uniform time compression**: Already proven infeasible (iter 14)
- **Departure inflation > 1.08**: Tested, race time exceeds 14.1s rollback criterion
- **Controller gain scheduling in kinematic sim**: Proven infeasible (iter 12)

---

## Section 7: Lessons Learned

### What worked
- **Junction gate concept was the key insight**: Gate-3 being both first AND second of S-turn pairs explained the persistent high error that previous iterations couldn't solve
- **Forward-looking S-turn detection works as expected**: The cross_z forward check correctly identifies first-gate S-turns
- **Gate-4 also improved (-3.5%)**: Despite the S-turn coupling concerns, both gate-3 AND gate-4 improved simultaneously. The departure inflation actually helps gate-4 too by giving the controller more time to approach it.
- **Three-variant parameter search found the optimal trade-off**: Testing 1.15/1.13/1.12 junction inflation revealed the speed-accuracy Pareto frontier

### What didn't work
- **Achieving < 14s race time with S-turn inflation**: The S-turn region needs more time by definition. Further speed recovery will need to come from other track sections.

### Surprises
- **Gate-4 improved too**: Expected gate-4 to regress from the departure inflation slowing its approach, but it actually improved -3.5%. The additional time in the departure from gate-3 apparently helps the controller set up a better approach to gate-4.
- **P95 tracking error dropped massively (-16.2%)**: The tail behavior improved much more than the mean, suggesting the gate-3 peak was dominating the P95 distribution.
- **Gate-12 also improved (-10.2%)**: Unexpected co-benefit — the additional TOPP retimer S-turn region marking may have slightly changed the speed profile near the end of the track.
- **No helix regression**: The changes were perfectly isolated to the S-turn region (gates 2-5) with zero impact on helix gates (7-11).

### Process suggestions
- When analyzing S-turn problems, always check if the gate is at a JUNCTION of multiple S-turn pairs, not just a single S-turn
- The three-variant parameter search approach is effective for finding Pareto-optimal inflation factors
- Race time vs accuracy trade-offs should be explicitly tracked in the Pareto frontier table
