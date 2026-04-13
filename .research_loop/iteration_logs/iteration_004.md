# Iteration 4 — Dual Entry/Exit Waypoints Per Gate

**Date**: 2026-04-13
**Bottleneck**: trajectory_planning (turn gate tracking error, particularly gate-7)
**Status**: COMMITTED — avg error 0.292→0.186m (-36.5%), gate-7 0.394→0.256m (-35%)
**Commit**: d92acef

---

## Section 1: Summary
- Iteration 4, timestamp 2026-04-13T15:58
- Bottleneck: trajectory_planning — sharp turn gates (3, 4, 7) had high tracking error due to abrupt direction changes at single gate-center waypoints
- Outcome: Implemented dual entry/exit waypoints at ±0.4m along each gate's normal direction. Average tracking error dropped 36.5%. Race time increased 16.97→23.0s due to more conservative time allocation across 25 segments.

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **"On Your Own: Pro-level Autonomous Drone Racing"** (Romero et al., 2025) — arxiv:2510.13644
   - KEY PAPER: Dual waypoints at ±0.4m along gate normal, deployed at IROS 2024 + Abu Dhabi F1 GP
   - Outperformed professional pilot
2. **"Euclidean and Non-Euclidean Trajectory Optimization for Quadrotor Racing"** (Fork & Borrelli, 2024) — arxiv:2309.07262
   - Gates as regions, not points; high-fidelity dynamics; 100x faster than comparable methods
3. **"Precise Aggressive Aerial Maneuvers with Sensorimotor Policies"** (2026) — arxiv:2604.05828
   - Addresses aggressive tilt angles up to 90° for gap traversal

### Previously analyzed (used)
4. **TOGT Planner** (Qin 2024) — gates as regions
5. **"Leveling the Playing Field"** (2025) — geometric controller tuning

### Key insight
"On Your Own" provides the exact implementation recipe: ±0.4m entry/exit waypoints per gate, centered on y/z axes. This ensures the trajectory optimizer generates polynomials that fly THROUGH gates along their normal direction rather than making sharp direction changes at gate centers.

### Consensus
- **Strong**: All competitive systems treat gates as regions (entry/exit), not point waypoints
- **Strong**: Dual waypoints improve trajectory smoothness through turns
- **No contradictions** found

---

## Section 3: Implementation

### Changes made
1. **`planning/trajectory_optimizer.py`** — `TrajectoryOptimizer.optimize()`:
   - Each gate now generates TWO waypoints: entry at `position - normal * 0.4m` and exit at `position + normal * 0.4m`
   - Gate normals are normalized; fallback uses direction from previous waypoint
   - Virtual finish waypoint (2m past last gate) remains unchanged
   - Total waypoints: 25 (start + 12 gates × 2 + virtual finish) vs 14 before

2. **`planning/trajectory_optimizer.py`** — `_initial_time_allocation()` and `_optimize_time_allocation()`:
   - Lowered minimum segment time from 0.2s/0.3s to 0.1s
   - Accommodates short (0.8m) entry→exit segments without artificial slowdown

3. **`scripts/benchmark.py`** — Unit test:
   - Updated segment count assertion: 4 → 7 (3 gates × 2 + finish)

### Plan adherence
Followed plan exactly. Three targeted changes as planned.

---

## Section 4: Benchmark Comparison

### Full metrics (before → after)
| Metric | Before | After | Delta | Direction | Threshold | Status |
|--------|--------|-------|-------|-----------|-----------|--------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | 0 | = | 100% | PASS |
| Gates passed | 12/12 (100%) | 12/12 (100%) | 0 | = | 100% | PASS |
| Avg tracking error | 0.292m | **0.186m** | **-0.107m** | ↓↓ | 0.5m | PASS |
| Max tracking error | 0.679m | **0.500m** | **-0.180m** | ↓↓ | 2.0m | PASS |
| P50 tracking error | 0.293m | **0.165m** | **-0.128m** | ↓↓ | — | — |
| P95 tracking error | 0.520m | **0.413m** | **-0.107m** | ↓↓ | — | — |
| EKF uncertainty | 0.012m | 0.012m | ~0 | = | 0.5m | PASS |
| Loop Hz | 7571 | 7197 | -374 | ↓ (negligible) | 100 | PASS |
| Trajectory time | 16.94s | 23.17s | +6.23s | ↑ | — | — |
| Race time | 16.97s | **23.0s** | **+6.03s** | ↑↑ | 30s | PASS |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Direction | Notes |
|------|--------|-------|-------|-----------|-------|
| gate-1 | 0.175 | **0.072** | **-0.104** | ↓↓ | 59% improvement |
| gate-2 | 0.296 | **0.155** | **-0.141** | ↓↓ | 48% improvement |
| gate-3 | 0.378 | **0.280** | -0.098 | ↓ | Now worst gate, sharp S-turn |
| gate-4 | 0.362 | **0.253** | -0.109 | ↓ | 30% improvement |
| gate-5 | 0.310 | **0.215** | -0.095 | ↓ | |
| gate-6 | 0.198 | 0.233 | **+0.034** | ↑ | Only regression — helix entry |
| gate-7 | 0.394 | **0.256** | **-0.139** | ↓↓ | 35% improvement, was worst |
| gate-8 | 0.250 | **0.152** | -0.098 | ↓ | |
| gate-9 | 0.253 | **0.120** | -0.134 | ↓↓ | 53% improvement |
| gate-10 | 0.262 | **0.156** | -0.106 | ↓ | |
| gate-11 | 0.320 | **0.153** | **-0.167** | ↓↓ | 52% improvement |
| gate-12 | 0.310 | **0.146** | -0.164 | ↓↓ | 53% improvement |

### Threshold status
All thresholds pass. Tracking error metrics comfortably within aspirational targets.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The dual-waypoint approach solved the turn-gate tracking problem comprehensively — average error dropped 36.5%. However, race time regressed 35% (16.97→23.0s) because doubling the number of waypoints (14→25 segments) causes the L-BFGS optimizer to allocate time too conservatively. The optimizer spreads time evenly across segments rather than recognizing that short entry→exit segments (0.8m) should be traversed quickly. Average pitch dropped from -0.138 to -0.072 rad, confirming the drone is flying more level (= slower).

### Telemetry signals
- Target jumps > 1m: N/A (kinematic sim)
- Detection frame %: N/A
- Recovery count: N/A
- Max abs roll: 0.7 rad (still at controller limit)
- Max abs pitch: 0.7 rad (still at controller limit)
- Avg pitch: -0.072 rad (previously -0.138 — drone is flying more level/slower)

### Trend analysis
- **STRONGLY IMPROVING**: Iterations 1-4 show consistent improvement:
  - Iter 1: Fixed benchmark bug
  - Iter 2: No code impact (curvature allocation masked by L-BFGS)
  - Iter 3: Virtual finish + gains → -12% avg error
  - Iter 4: Dual waypoints → -36% avg error
- **NEW CONCERN**: Race time has worsened (16.97→23.0s), now a critical priority
- **No stagnation**: Each iteration addressed a different aspect successfully
- **Accumulated knowledge**: "On Your Own" paper technique proven highly effective

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **Speed optimization** (Priority 1, trajectory_planning)
   - Race time 23.0s → target 14s
   - L-BFGS is too conservative with 25 segments
   - Approach: Increase speed factors (0.65→0.80 straights, 0.55→0.70 moderate, 0.45→0.55 sharp). Also consider adding velocity penalty to L-BFGS objective to prevent excessive slowdown.
   - Expected: 23→16s first pass, then tighten to 14s
   - Research: TOGT Planner speed optimization

2. **Increase max tilt limit** (Priority 2, control)
   - 0.7 → 0.8 rad
   - Enables higher speed through turns
   - Should be combined with speed optimization

3. **Adaptive entry/exit offset** (Priority 3, trajectory_planning)
   - Scale 0.4m offset by approach angle
   - May fix gate-6 regression and reduce total trajectory length

4. **Install gym-pybullet-drones** (Priority 4, system_integration)
5. **Perception pipeline integration** (Priority 5, perception_estimation)

### Architectural recommendations
- The speed optimization should target the L-BFGS objective function rather than just the initial allocation (lesson from iteration 2: L-BFGS overrides initial guesses)
- Consider adding a minimum average speed constraint to the L-BFGS objective
- The tilt limit increase should be paired with speed optimization — no point going faster if controller can't handle the turns

### Next bottleneck
`trajectory_planning` — speed optimization to reduce race time from 23.0s toward 14s target

### What NOT to try
- Don't retry curvature-aware initial allocation alone — L-BFGS masks it (failed iter 2)
- Don't increase max_velocity without paired speed factor changes (failed iter 1)
- Don't remove dual waypoints to recover speed — the 36% tracking improvement is too valuable

---

## Section 7: Lessons Learned

### What worked
- **Dual waypoint approach**: Massive 36.5% average error reduction. This is the single most impactful change so far. "On Your Own" paper technique works exactly as described.
- **Research-driven**: Paper cited exact parameters (±0.4m) that we used directly
- **Clean implementation**: Only 3 files changed, all unit tests pass first try

### What didn't work
- Nothing failed outright, but race time regression was larger than expected (+35%)

### Surprises
- Gate-6 was the only regression, suggesting the straight-to-helix transition has a unique geometry that dual waypoints may slightly mishandle
- The helix gates (7-12) benefited even more than expected — improvements of 35-53% at each gate
- Loop Hz dropped slightly (7571→7197) due to more trajectory points, but still >7000x faster than real-time

### Process suggestions
- Speed optimization is now critical and should be the sole focus of iteration 5
- Consider modifying the L-BFGS objective function to include a speed term (penalty for total time exceeding a target) rather than just adjusting initial allocations
