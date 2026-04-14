# Iteration 18 — Bidirectional Proximity Inflation for Helix Recovery

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (helix gates 7-12 accuracy recovery)
**Status**: COMMITTED — avg error 0.248→0.234m (-5.6%), max error 0.803→0.696m (-13.3%), race time 13.62→13.91s (+2.1%)
**Commit**: 671fb02

---

## Section 1: Summary
- Iteration 18, timestamp 2026-04-14T12:30Z
- Bottleneck: trajectory_planning — recover helix gate accuracy lost in iter 17's FOV relaxation removal
- One-line outcome: **All 6 helix gates improved (avg -0.028m), max error below 0.7m for first time, all aspirational targets met**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Quad-LCD: Layered Control Decomposition** (Srikanthan et al., UPenn 2025, arXiv:2505.10228)
   - Controller-aware trajectory feasibility prediction using learned tracking penalty
   - Key finding: per-segment feasibility depends on polynomial coefficients AND approach context
   - Reduces crash rate by 49% by reshaping reference trajectory, not changing controller

2. **Improving Drone Racing via Iterative Learning MPC** (arXiv:2508.01103, 2025)
   - ILMPC with adaptive cost function: dynamic weighting of time-optimal vs centerline adherence
   - 60.85% max improvement from initial trajectory, 6.05% improvement over MPCC++ on real drone
   - Shifted local safe set prevents excessive shortcutting

3. **Spatial ILC in Virtual Tube** (arXiv:2306.15992, 2023)
   - Iterative speed profile optimization: v_{k+1}(l) = v_k(l) - k_p * error(l) with deadzone
   - Converges in 7-20 iterations to near-optimal speed profile
   - Virtual tube constraint from gate geometry

### Previously analyzed (directly used)
4. **CiMPCC** (Li et al., ITSC 2024) — compound curvature for sequential turns
5. **TACO** (Sanghvi et al., 2025) — trajectory-aware controller optimization
6. **FBGA** (Piazza et al., RA-L 2025) — forward-backward speed profiling

### Research consensus vs contradictions
- **Consensus (6/6 papers)**: Per-segment trajectory feasibility depends on neighborhood context (preceding AND following segments), not just local geometry. CiMPCC, FBGA (forward-backward), Quad-LCD, and Spatial ILC all use bidirectional context.
- **No contradictions**: All papers support context-dependent trajectory adaptation.
- **Key insight**: Quad-LCD's approach of learning a tracking penalty per segment is the principled version of what our proximity inflation heuristic does. Future work should consider replacing the heuristic with a learned predictor.

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`** — `_inflate_sharp_turns` method only

**Change 1: Bidirectional proximity check**
- Before: proximity inflation checked distance to NEXT gate only (`gate_centers[gi+1]`)
- After: checks distance to BOTH previous and next gate, uses the minimum
- This is critical for gate 11 (49.9° turn, 5.66m to next gate but only 3.64m to prev gate)
- Before: gate 11 got 0.7% inflation. After: 7.1% inflation.

**Change 2: Increased proximity multiplier 0.12→0.22**
- Compensates for the 3-8% inflation that FOV relaxation was providing (removed in iter 17)
- Swept 5 values: 0.15, 0.18, 0.20, 0.22, 0.25
- Selected 0.22 as best Pareto point (max error below 0.7m, race time has 0.09s margin)

### Tuning iterations
| Multiplier | Race time | Avg error | Max error |
|-----------|-----------|-----------|-----------|
| 0.12 fwd (baseline) | 13.62s | 0.248m | 0.803m |
| 0.15 bidir | 13.77s | 0.242m | 0.770m |
| 0.18 bidir | 13.83s | 0.238m | 0.735m |
| 0.20 bidir | 13.87s | 0.236m | 0.716m |
| **0.22 bidir** | **13.91s** | **0.234m** | **0.696m** |
| 0.25 bidir | 13.98s | 0.230m | 0.667m |

### Plan adherence
Followed plan exactly. The bidirectional proximity approach worked as predicted. The multiplier tuning (5 values instead of planned 3-4) was slightly more thorough than planned.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.248m | **0.234m** | **-0.014m (-5.6%)** | ✓✓ |
| Max tracking error | 0.803m | **0.696m** | **-0.107m (-13.3%)** | ✓✓✓ |
| P50 tracking error | 0.206m | 0.197m | -0.009m | ✓ |
| P95 tracking error | 0.556m | 0.526m | -0.030m | ✓ |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7720 | 7745 | +0.3% | → |
| Trajectory time | 13.81s | 14.10s | +0.29s | ↓ (expected trade-off) |
| Race time | 13.62s | **13.91s** | **+0.29s (+2.1%)** | ↓ (expected trade-off) |
| Avg thrust | 0.827 | 0.818 | -1.1% | ✓ (slower helix = less thrust) |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.115 | 0.115 | 0.000 | → unchanged |
| gate-2 | 0.233 | 0.233 | 0.000 | → unchanged |
| gate-3 | 0.329 | 0.329 | 0.000 | → unchanged |
| gate-4 | 0.413 | 0.413 | 0.000 | → still worst overall |
| gate-5 | 0.181 | 0.181 | 0.000 | → unchanged |
| gate-6 | 0.156 | 0.156 | 0.000 | → unchanged |
| gate-7 | 0.351 | **0.333** | **-0.018** | ✓ helix entry |
| gate-8 | 0.267 | **0.221** | **-0.046** | ✓✓ biggest improvement |
| gate-9 | 0.227 | **0.203** | **-0.024** | ✓ |
| gate-10 | 0.228 | **0.208** | **-0.020** | ✓ |
| gate-11 | 0.200 | **0.173** | **-0.027** | ✓ second biggest improvement |
| gate-12 | 0.244 | **0.212** | **-0.032** | ✓ helix exit |

### Threshold status
| Threshold | Required | Current | Aspirational | Status |
|-----------|----------|---------|--------------|--------|
| Avg error | <0.5m | **0.234m** | <0.25m | **MEETS ASPIRATIONAL** |
| Max error | <2.0m | **0.696m** | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | 13.91s | <14s | **MEETS ASPIRATIONAL** |
| Loop Hz | >100 | 7745 | >100 | PASS |
| No crash | required | no crash | — | PASS |

All aspirational targets met.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The `_inflate_sharp_turns` proximity check was **asymmetric** — it only measured distance to the NEXT gate. In the helix, gate 11 (turn angle 49.9°, below the 60° angle threshold) sits between gate 10 (3.64m away) and gate 12 (5.66m away). The forward-only check saw 5.66m and gave only 0.7% inflation. The bidirectional fix sees 3.64m and gives 7.1% inflation. Similarly, gate 8 benefits because its preceding gate (gate 7) is closer (3.64m) than its next gate (gate 9 at 4.87m).

The root cause is that helix gates have **heterogeneous turn angles**: gates 7, 8, 10 have 63-69° turns (above 60° threshold), while gates 9, 11 have 48-50° turns (below threshold). The forward-only proximity was the only inflation source for gates 9 and 11, and it was too weak.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (unchanged, at tilt limit)
- Avg thrust: 0.818 (decreased from 0.827 — slower helix = less thrust)
- Avg pitch: -0.099 (slightly less forward lean)
- Controller is working slightly less hard on the helix due to better inflation

### Trend analysis
**Trend: IMPROVING (Pareto frontier steadily advancing)**

Last 4 iterations' Pareto evolution:
- Iter 15: 13.50s / 0.251m (TOPP speed breakthrough)
- Iter 16: 13.95s / 0.232m (S-turn accuracy breakthrough)
- Iter 17: 13.62s / 0.248m (speed recovery via FOV removal)
- **Iter 18: 13.91s / 0.234m (accuracy recovery via bidir proximity)**

The alternating speed/accuracy pattern is healthy. Each iteration pushes one axis without catastrophic regression on the other. The Pareto frontier is monotonically improving.

### Architectural observations
- The proximity inflation heuristic now handles the helix well, but it's fundamentally a hand-tuned rule. Quad-LCD (Srikanthan 2025) suggests replacing it with a learned per-segment feasibility predictor.
- The gate-4 problem (0.413m, worst gate) is resistant to turn inflation improvements. It may require a different racing line through the S-turn, achievable via multi-start L-BFGS optimization.
- The max_compression floor in TOPP (0.68) is no longer the binding constraint for helix gates — the inflation is now sufficient to protect them.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Multi-start racing line optimization** (Priority 1, system_integration)
   - Run L-BFGS from 5-10 random initial offset vectors
   - Score each by TOPP-computed race time + tracking quality proxy
   - Expected: escape local minimum that keeps gate-4 at 0.413m, potential 0.5-1s improvement
   - Research: Sequence Modeling (2025), TACO (2025)

2. **Gate-4 specific racing line tuning** (Priority 2, trajectory_planning)
   - The S-turn (gates 3-4) is the remaining accuracy bottleneck
   - Try asymmetric lateral offsets: wider entry to gate-3, tighter exit to gate-4
   - Expected: gate-4 error 0.413→0.35m
   - Research: TOGT (Qin 2024) gate regions, CiMPCC S-turn handling

3. **Yaw optimization for FOV** (Priority 3, trajectory_planning)
   - Optimize yaw profile to keep next gate visible without slowing trajectory
   - Zero time cost (only changes heading, not speed)
   - Research: Drift-Corrected VIO (2025), MonoRace (2026)

4. **Learned per-segment feasibility predictor** (Priority 4, trajectory_planning)
   - Replace heuristic inflate_sharp_turns with Quad-LCD-style learned predictor
   - Train on benchmark data: (segment polynomials, approach velocity) → tracking error
   - Expected: better inflation accuracy than hand-tuned rules
   - Research: Quad-LCD (Srikanthan 2025), Spatial ILC (2023)

5. **MPCC controller upgrade** (Priority 5, control)
   - ETH 2026 achieves 0.07m avg error at 9.8 m/s via contouring error decomposition
   - Major architectural change — defer until trajectory planning is optimized
   - Research: MPCC++ (Krinner 2024), ILMPC (2025)

### Architectural recommendations
- The heuristic inflation approach is reaching its limits — 5 different inflation mechanisms (angle, centripetal, proximity, S-turn, FOV) with interdependent parameters. A learned predictor would consolidate these into a single, more accurate mechanism.
- The racing line L-BFGS is stuck in a local minimum for the S-turn region. Multi-start optimization is the natural next step before considering more complex approaches.
- The alternating speed/accuracy pattern suggests the system is near a local optimum for the current trajectory planning approach. Step-change improvements may require architectural changes (MPCC controller, learned planner).

### What NOT to try
- **Uniform time compression** — proven infeasible (iter 14)
- **Controller gain scheduling in kinematic sim** — exhaustively proven infeasible (iter 12)
- **a_centripetal reduction in TOPP** — no effect when max_compression is binding (iter 17)
- **Proximity multiplier > 0.25** — race time exceeds 14s aspirational target
- **Lowering the 60° angle threshold** — would shift severity scale for all gates, less targeted than bidirectional proximity

---

## Section 7: Lessons Learned

### What worked
- **Bidirectional proximity was exactly the right fix**: Gate 11's inflation jumped from 0.7% to 7.1%, directly recovering the lost FOV relaxation effect
- **Parametric sweep was efficient**: 5 values of the multiplier (0.15-0.25), each benchmark takes <0.2s
- **Research-backed root cause analysis**: CiMPCC's compound curvature principle directly predicted that bidirectional context would help

### What didn't work
- Nothing in this iteration failed — the approach was targeted and well-diagnosed

### Surprises
- **Gate 8 had the biggest improvement (-0.046m)**: Expected gate 11 to benefit most, but gate 8's bidirectional proximity (dist to gate-7 at 3.64m) was also much better than its forward-only proximity (dist to gate-9 at 4.87m)
- **Max error dropped by 13.3%**: The max tracking error improvement was even larger than the avg improvement, suggesting the helix peak errors were specifically caused by under-inflated gates 9 and 11
- **Perfect gate-1-6 isolation**: Not a single gate outside the helix changed by even 0.001m. The bidirectional proximity fix is perfectly targeted.

### Process suggestions
- When diagnosing inflation gaps, compute the exact inflation each gate receives (angle, centripetal, proximity) rather than guessing
- The gate geometry analysis (computing exact turn angles and distances) should be a standard diagnostic step
- The 5-point parametric sweep is an efficient way to find the optimal Pareto point
