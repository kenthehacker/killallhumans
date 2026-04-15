# Iteration 44 — Speed Recovery via Turn Inflation Reduction

**Date**: 2026-04-15
**Bottleneck**: trajectory_planning (speed recovery)
**Status**: COMMITTED — race time 14.08→13.51s (-4.0%), avg error 0.138→0.159m (+15.2%)
**Commit**: 2f84473

---

## Section 1: Summary
- Iteration 44, timestamp 2026-04-15T15:25Z
- Bottleneck: trajectory_planning (speed recovery via inflation reduction)
- One-line outcome: **Race time reduced 14.08→13.51s (-4.0%) by halving turn inflation factors; avg error 0.138→0.159m, still well under 0.25m threshold**

---

## Section 2: Research

### Papers analyzed (3 new, 118→121 total)
1. **RAPTOR: A Foundation Policy for Quadrotor Control** (arXiv:2509.11481, 2025)
   - Foundation policy for quadrotor control across diverse tasks
   - Relevant: demonstrates that modern controllers can handle more aggressive trajectories than traditional safety margins assume
2. **Online Velocity Profile Generation for Autonomous Racing** (arXiv:2505.05157, 2025)
   - Online velocity profile adjustment for varying constraints
   - Relevant: shows that velocity profiles can be made more aggressive with real-time adjustment
3. **Curriculum Reinforcement Learning for Quadrotor Racing** (arXiv:2602.24030, 2026)
   - Staged training: soft→hard collision constraints
   - Relevant: progressive difficulty increase parallels our approach of gradually reducing inflation

### Key insight from research
The research consensus across MonoRace (2026), CPC (Foehn 2021), and TACO (Sanghvi 2025) is clear: post-optimization inflation is a legacy safety mechanism from weaker controllers. Modern systems with feedforward control and ILC compensation can handle significantly more aggressive trajectories.

### Research consensus vs contradictions
- **Consensus**: Over-conservative trajectory timing wastes race time; modern controllers (with feedforward and ILC) can track faster trajectories than initially assumed.
- **Contradiction**: None — all research supports reducing inflation when tracking margin exists.

---

## Section 3: Implementation

### Changes made
File: `planning/trajectory_optimizer.py` — `_inflate_sharp_turns()` method

| Parameter | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Sharp turn severity (>60°) | 0.25 | 0.12 | -52% |
| Centripetal severity | 0.15 | 0.08 | -47% |
| S-turn junction | 1.09 | 1.04 | -56% |
| S-turn second-gate | 1.07 | 1.03 | -57% |
| S-turn approach | 1.01 | 1.005 | -50% |
| S-turn first departure | 1.02 | 1.01 | -50% |
| S-turn junction departure | 1.005 | 1.002 | -60% |
| Proximity factor | 0.22 | 0.12 | -45% |
| Helix entry | 1.12 | 1.06 | -50% |
| Helix interior | 1.10 | 1.05 | -50% |

### Plan adherence
Followed plan exactly. No deviations needed — the first-try implementation succeeded.

---

## Section 4: Benchmark Comparison

### Full metrics table

| Metric | Before (iter 43) | After | Delta | Direction |
|--------|-------------------|-------|-------|-----------|
| Race time | 14.08s | **13.51s** | **-4.0%** | **IMPROVED** |
| Trajectory time | 14.27s | 13.69s | -4.1% | improved |
| Avg tracking error | 0.138m | 0.159m | +15.2% | regressed (within budget) |
| Max tracking error | 0.727m | 0.699m | -3.9% | improved |
| P50 tracking error | 0.113m | 0.136m | +20.4% | regressed (within budget) |
| P95 tracking error | 0.331m | 0.366m | +10.6% | regressed (within budget) |
| EKF uncertainty | 0.0119m | 0.0119m | 0% | same |
| Gate pass rate | 100% | 100% | 0% | same |
| Loop Hz | 7011 | 7659 | +9.2% | improved (fewer steps) |
| Crashed | false | false | — | same |
| Unit tests | 9/9 | 9/9 | — | same |

### Per-gate error breakdown (before vs after)

| Gate | Before | After | Delta | Headroom to 0.25m |
|------|--------|-------|-------|--------------------|
| gate-1 | 0.113m | 0.113m | 0.0% | 0.137m |
| gate-2 | 0.214m | 0.216m | +0.9% | **0.034m** |
| gate-3 | 0.191m | **0.226m** | **+18.3%** | **0.024m** |
| gate-4 | 0.142m | 0.190m | +33.8% | 0.060m |
| gate-5 | 0.139m | 0.161m | +15.9% | 0.089m |
| gate-6 | 0.074m | 0.080m | +9.1% | 0.170m |
| gate-7 | 0.164m | 0.186m | +13.4% | 0.064m |
| gate-8 | 0.128m | 0.192m | +50.0% | 0.058m |
| gate-9 | 0.109m | 0.117m | +7.7% | 0.133m |
| gate-10 | 0.144m | 0.151m | +4.8% | 0.099m |
| gate-11 | 0.116m | 0.126m | +8.6% | 0.124m |
| gate-12 | 0.141m | 0.170m | +20.2% | 0.080m |

### Threshold status
ALL thresholds passing. No threshold failures.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The post-optimization turn inflation factors (introduced in iters 7-31) were calibrated for a weaker controller (kp=4, no feedforward, no ILC). After 43 iterations of controller improvements (kp=7, kd=5.5, ff=0.50, per-section ILC with velocity correction), the inflation factors were over-protective by ~50%. Halving them recovers 0.57s of race time while keeping avg error well under the 0.25m threshold (0.091m remaining headroom).

### Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_abs_roll_rad | 0.85 | 0.85 |
| max_abs_pitch_rad | 0.85 | 0.85 |
| avg_pitch_rad | -0.096 | -0.100 |
| avg_thrust | 0.838 | 0.852 |

Controller effort increased (thrust +1.7%) consistent with faster trajectory. Roll/pitch saturation unchanged — actuator limits are not the binding constraint yet.

### Trend analysis
- **BREAKTHROUGH**: After 5 iterations of diminishing accuracy returns (iters 38-43, total -21.4% avg error), this iteration pivots to SPEED, recovering 0.57s in a single iteration.
- **Speed improvement history**: Race time has been stuck at 14.08s since approximately iter 36. This is the first significant race time reduction since the TOPP retimer was introduced.
- **Accuracy-speed tradeoff**: The Pareto frontier shifted — we traded 15.2% avg error increase for 4.0% race time decrease. The error budget still has 0.091m headroom for further speed recovery.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Further speed recovery via TOPP floor reduction** (priority 1)
   - Area: trajectory_planning
   - Problem: Race time at 13.51s with 0.091m avg error headroom. TOPP floors are the second speed-limiting layer.
   - Approach: Carefully reduce helix floor (0.72→0.69), easy floor (0.59→0.56), protected floor (0.65→0.62). WARNING: S-turn floor (0.70) decrease caused basin switching in iter 39 — leave it alone or try 0.69 very carefully.
   - Expected impact: Race time 13.51→13.0-13.3s
   - Research refs: FBGA (Piazza 2025), TOPP-RA (Pham 2017)

2. **Gate-3 ILC improvement** (priority 2)
   - Area: system_integration
   - Problem: Gate-3 at 0.226m with only 0.024m headroom — tightest gate. Limits further speed recovery.
   - Approach: Increase S-turn inflection section ILC alpha from 0.4 to 0.45 to improve correction at gate-3. Or try wider bandwidth (0.40→0.45 Hz) for the inflection section.
   - Expected impact: Gate-3 error reduction 5-10%, freeing headroom for more speed
   - Research refs: Schoellig 2012, Bristow & Alleyne 2007

3. **L-BFGS time_weight increase** (priority 3)
   - Area: trajectory_planning
   - Problem: L-BFGS time_weight=2.0 hasn't been changed since early iterations. With ILC, more aggressive base times may be tolerable.
   - Approach: Try time_weight=2.5. This failed in iter 8 but pre-ILC.
   - Expected impact: Potentially 0.1-0.3s race time reduction
   - Research refs: TOGT (Qin 2024)

4. **Centripetal threshold increase** (priority 4)
   - Area: trajectory_planning
   - Problem: a_centripetal_threshold=4.5 m/s² is conservative with ILC.
   - Approach: Increase to 5.5 m/s² — allows faster approach to moderate turns.
   - Expected impact: Small race time improvement at moderate-curvature segments
   - Research refs: TOPPQuad (Mao 2024)

### Architectural recommendations
- The inflation reduction approach can be pushed further, but gate-3 (0.226m) is the binding constraint. Any further speed gain requires either (a) improving gate-3 ILC accuracy or (b) accepting that gate-3 will approach the 0.25m threshold.
- The system is now at a fundamentally different operating point — trading accuracy for speed rather than optimizing accuracy at fixed speed. Future iterations should monitor the Pareto frontier (race_time vs avg_error).

### Next bottleneck selected
**trajectory_planning** — continue speed recovery, specifically via TOPP floor reduction. The inflation has been halved; now target the TOPP floors as the next speed-limiting layer.

### What NOT to try
- S-turn TOPP floor reduction to 0.66 or below — basin switching (iter 39)
- Uniform time compression — still fails for turn segments (iter 14)
- Inflation reduction much further — gate-3 at 0.226m with 0.024m headroom limits this
- Gate offset changes — basin switching (iters 32, 34)

---

## Section 7: Lessons Learned

### What worked
- **Halving all inflation factors simultaneously**: Instead of careful 1-2% changes (iters 29-32 approach), a bold 50% across-the-board reduction succeeded because the racing line cache prevented basin switching and the ILC compensated.
- **Pivoting from accuracy to speed**: After 5 iterations of diminishing accuracy returns, switching objective unlocked the biggest single-iteration improvement since iter 38.
- **Simple implementation**: Only modifying inflation constants (10 parameters) in one function, with no algorithmic changes. Simple changes are easier to debug and revert.

### What didn't work
- Nothing failed this iteration — the first-try implementation succeeded.

### Surprises
- **50% inflation reduction only caused 15.2% avg error increase**: The ILC absorbed much of the increased tracking difficulty, confirming that the old inflation was over-protective.
- **Gate-1 was completely unaffected** (0.113m → 0.113m): The first gate has no turn inflation, so reduction had zero impact.
- **Max tracking error actually improved** (0.727→0.699m, -3.9%): Faster trajectory means the drone spends less time at high-error points, reducing the worst-case error.

### Suggestions for improving the iteration process
- When a system has significant headroom on a metric (>40% unused budget), prioritize trading that headroom for improvement on the primary competition objective (race time).
- Bold parameter changes (50%) are safe when: (1) racing line is cached, (2) ILC compensates, (3) there's sufficient headroom.
- The Pareto frontier (race_time vs avg_error) should be tracked explicitly in state.json to guide tradeoff decisions.
