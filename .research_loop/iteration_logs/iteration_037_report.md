# Iteration 37 — S-turn TOPP Floor Split and Raise (0.65→0.67)

**Date**: 2026-04-15
**Bottleneck**: trajectory_planning (gate-3 S-turn TOPP floor binding)
**Status**: COMMITTED — avg error 0.176→0.174m (-1.1%), max error 0.742→0.716m (-3.5%)
**Commit**: ba00d0d

---

## Section 1: Summary
- Iteration 37, timestamp 2026-04-15T10:55Z
- Bottleneck: trajectory_planning — gate-3 at 0.247m (worst gate), S-turn TOPP floor (0.65) identified as binding
- One-line outcome: **S-turn TOPP floor split from protected + raise to 0.67 → avg error -1.1%, max error -3.5%, race time +0.04s (13.98→14.02s)**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **GripMap: Spatially Resolved Constraint Framework** (Werner et al., IEEE IV 2025, arXiv:2504.12115)
   - Framework for spatially-varying dynamic constraints in Frenet frame
   - Each cell stores scaling factor θ_ij modulating acceleration limits
   - 5.2% lap time improvement from spatial constraint resolution
   - **Key insight**: Our TOPP floors are a 1D GripMap — validated approach

2. **Computationally Efficient Minimum-lap-time Control** (van den Eshof et al., ITSC 2026, arXiv:2603.02339)
   - Decomposes speed optimization at curvature peaks (corner apexes) into boundary-value problems
   - Bang-bang optimal structure: full throttle / coast / brake
   - **Key insight**: Curvature-peak decomposition supports independent section optimization

3. **Spline-Based Minimum-Curvature Trajectory Optimization** (Xue et al., ICRA 2024, arXiv:2309.09186)
   - B-spline curvature minimization for chicanes/S-turns
   - 90% decision variable reduction with millisecond solve times
   - **Key insight**: In S-turns, peak curvature is the binding speed constraint; we must work within fixed racing line geometry (offset changes cause basin switching)

### Key insight from cross-validation
The research consensus strongly supports spatially-varying TOPP constraints. However, the diagnostic finding that **joint S-turn/helix floor optimization is unfavorable** was not predicted by any paper — it's specific to our system's asymmetric curvature sensitivity. Gate-7 (helix entry) has ~2.3x the sensitivity per floor unit vs gate-3 (S-turn), making time-neutral rebalancing always net-negative.

### Research consensus vs contradictions
- **Strong consensus**: Spatially-varying speed constraints are correct (GripMap, CiMPCC, FBGA, Spatially-Aware CMA-ES)
- **Strong consensus**: Curvature peaks are binding constraints (Energy-Limited MinLap, iter 36 data)
- **New finding**: Joint optimization of S-turn+helix floors is unfavorable due to asymmetric sensitivity — not contradicted by papers but not addressed either

---

## Section 3: Implementation

### Changes made
1. **File**: `planning/trajectory_optimizer.py`
   - **Line 891**: Split `max_compression_protected` into two variables:
     - `max_compression_sturn = 0.67` (new, for S-turn segments)
     - `max_compression_protected = 0.65` (unchanged, for high-curvature/pre-turn)
   - **Line 1013**: Changed floor assignment for S-turn segments to use `max_compression_sturn` instead of `max_compression_protected`

### Sweep methodology
- **Phase 1**: Pure S-turn sweep [0.66, 0.67, 0.68, 0.69, 0.71, 0.73] at helix=0.72
  - All values improve avg error; monotonic gate-3 improvement
  - Race time increases ~0.02s per 0.01 floor
  - Helix gates completely isolated (gate-7 unchanged ±0.001m)
- **Phase 2**: Joint S-turn×helix sweep (6 combinations)
  - Large helix reductions: (0.67,0.70), (0.69,0.68), (0.71,0.66), (0.73,0.63)
  - All WORSE avg error due to gate-7 sensitivity
- **Phase 3**: Fine-grid joint sweep (6 combinations)
  - (0.66,0.71), (0.67,0.71), (0.68,0.71): marginal improvements, worst gate worsens
  - (0.67,0.715), (0.67,0.718): intermediate, still worse than pure S-turn
- **Selection**: Pure S-turn increase s=0.67 at h=0.72 — best avg error improvement with minimal worst-gate change

### Plan adherence
Followed the plan closely. The plan proposed joint rebalancing; the sweep revealed this is unfavorable, so selected pure S-turn increase instead.

---

## Section 4: Benchmark Comparison

### Metrics: Before vs After
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | = | = |
| Gates passed | 12/12 (100%) | 12/12 (100%) | = | = |
| Avg tracking error | 0.1760m | **0.1741m** | **-1.1%** | **Better** |
| Max tracking error | 0.7422m | **0.7159m** | **-3.5%** | **Better** |
| P50 tracking error | 0.1654m | 0.1636m | -1.1% | Better |
| P95 tracking error | 0.3462m | 0.3409m | -1.5% | Better |
| EKF uncertainty | 0.0119m | 0.0119m | = | = |
| Race time | **13.98s** | **14.02s** | **+0.3%** | Slightly worse |
| Deterministic | YES | YES | = | = |
| Worst gate | gate-3 (0.247m) | gate-7 (0.247m) | ≈ | = |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1109 | 0.1109 | 0.0% | = |
| gate-2 | 0.1791 | 0.1787 | -0.2% | ≈ |
| gate-3 | **0.2473** | **0.2446** | **-1.1%** | **Better** |
| gate-4 | 0.2015 | 0.2021 | +0.3% | ≈ |
| gate-5 | 0.1799 | 0.1720 | **-4.4%** | **Better** |
| gate-6 | 0.1613 | 0.1583 | -1.9% | Better |
| gate-7 | 0.2472 | 0.2474 | +0.1% | = |
| gate-8 | 0.2165 | 0.2099 | **-3.0%** | **Better** |
| gate-9 | 0.1469 | 0.1448 | -1.4% | Better |
| gate-10 | 0.1531 | 0.1535 | +0.3% | ≈ |
| gate-11 | 0.1480 | 0.1480 | 0.0% | = |
| gate-12 | 0.1367 | 0.1343 | -1.8% | Better |

**Key observation**: S-turn floor change improved gates 3, 5, 6, 8, 9, 12 — more widespread than expected. The S-turn segment detection marks segments around multiple gates (not just gate-3). Gates 5 and 8 showed the largest unexpected improvements (-4.4% and -3.0%).

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
The S-turn TOPP floor (0.65) was binding for S-turn exit segments, limiting how much time the trajectory retains through S-turn regions. Raising to 0.67 gives the PD controller 2% more time through these segments, reducing tracking error across all S-turn-classified gates. The max error improvement (0.742→0.716m) suggests the peak tracking error occurred in an S-turn region, not the helix.

### 5b.2 — Telemetry signals
- max_roll: 0.85 rad (saturating — unchanged from every recent iteration)
- max_pitch: 0.85 rad (saturating)
- avg_thrust: 0.796 (unchanged)
- avg_pitch: -0.107 rad (unchanged)
- **max tracking error location**: shifted from ~0.742m to ~0.716m — S-turn region

### 5b.3 — Joint optimization finding
The most important diagnostic finding: **gate-7 sensitivity to helix floor is 2.3x greater than gate-3 sensitivity to S-turn floor** (0.003m/0.01 vs 0.0014m/0.01). This makes time-neutral joint rebalancing always net-negative. The two floors are effectively independent optimization dimensions, but with asymmetric curvature sensitivity.

### 5b.4 — Trend analysis (iterations 35-37)
- **Iter 35**: Helix TOPP floor 0.65→0.76 → avg error -7.9% (BREAKTHROUGH)
- **Iter 36**: Helix TOPP floor 0.76→0.72 → race time -0.8% (Pareto rebalance)
- **Iter 37**: S-turn TOPP floor 0.65→0.67 → avg error -1.1% (diminishing returns)

**Trend: DIMINISHING RETURNS on TOPP floor tuning.** Three consecutive iterations on the same lever family (spatial TOPP floors) with gains: 7.9% → rebalance → 1.1%. The spatial floor approach is fully characterized: helix (0.72), S-turn (0.67), protected (0.65), easy (0.59). Further floor tuning would yield <1% gains.

### 5b.5 — Architectural issues
- **REMAINING**: PD controller saturating at 0.85 rad limits all gates
- **REMAINING**: Max tracking error 0.716m at controller limit
- **REMAINING**: TOPP floor tuning exhausted — diminishing returns
- **NEW**: Race time marginally over aspirational 14s target (14.02s)
- **NEW**: S-turn/helix floors proven independent — no beneficial coupling

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **[control]** MPCC or SE(3) geometric controller to replace PD
   - PD controller saturating at 0.85 rad limits all gates
   - Max error (0.716m) is at controller limit, not trajectory limit
   - Would unlock ability to track faster trajectories
   - Expected: 5-15% tracking improvement, unlock sub-13.5s race time
   - Priority: 1
   - Research: MPCC++ (Krinner 2024), NGTC (arXiv 2510.12611), TACO (arXiv 2511.02060)

2. **[trajectory_planning]** Proper TOPP-RA with LP-based reachability
   - Replace heuristic forward-backward + manual floors with LP-based reachability analysis
   - Eliminates floor tuning entirely; automatically optimal speed profile
   - Significant implementation effort but removes the main tuning bottleneck
   - Priority: 2
   - Research: Primitive-Planner (arXiv 2502.16882), TOPPQuad (Mao 2024)

3. **[trajectory_planning]** Race time recovery to sub-14s
   - Currently 14.02s — 0.02s over aspirational target
   - Options: lower easy floor (0.59→0.58), fine-tune helix floor (0.72→0.715)
   - Expected: 0.02-0.04s improvement
   - Priority: 3
   - Research: iter 36 Pareto analysis

4. **[system_integration]** PyBullet full-physics simulation
   - All recent iterations on synthetic kinematic sim
   - Need to validate gains transfer to PyBullet physics
   - Critical for competition readiness
   - Priority: 4

5. **[perception_estimation]** Integrate EKF + gate PnP into runner
   - PLAN.md identifies this as the key architectural gap
   - Runner still uses raw detection → target, bypassing estimation pipeline
   - Priority: 5 (blocked on PyBullet validation)

### Next bottleneck
**control** — The PD controller is the dominant performance limiter. It saturates at 0.85 rad roll/pitch, which means:
1. The controller cannot track faster trajectories even if we generate them
2. Max tracking error (0.716m) is controller-limited, not trajectory-limited
3. TOPP floor tuning is exhausted — further gains require controller improvement

The MPCC or SE(3) geometric controller would address the root cause, not just symptoms.

### What NOT to try
- Further TOPP floor tuning (diminishing returns — 3 iterations exhausted the space)
- Joint S-turn/helix floor optimization (proven unfavorable — asymmetric sensitivity)
- More ILC iterations (catastrophic in iter 35 — cumulative offset saturation)
- Gate offset changes (basin switching — failed in iters 32, 34)
- Inflation parameter changes (TOPP floor is binding, not inflation — confirmed iter 35)

---

## Section 7: Lessons Learned

### What worked
- **S-turn floor split**: Separating the S-turn floor from the general protected floor is clean, testable, and independent
- **Systematic sweep**: 16 total configurations tested (4 pure S-turn + 4 large joint + 6 fine joint + 2 precision)
- **Independent verification**: Gate-7 isolation confirmed — S-turn and helix floors are truly independent

### What didn't work
- **Joint S-turn/helix optimization**: The asymmetric sensitivity makes this unfavorable. This was the main hypothesis going in, but the data clearly refutes it.
- **Aspirational race time**: The 0.04s regression means we slightly exceeded the <14s aspirational target

### Surprises
- **Wider S-turn impact**: Gates 5, 8, 12 improved 1.8-4.4% — the S-turn segment detection covers more of the trajectory than expected
- **Max error improvement**: 0.742→0.716m (-3.5%) was unexpected — the peak error was in an S-turn region, not helix
- **Gate-7 sensitivity asymmetry**: 2.3x more sensitive per floor unit than gate-3 — important constraint for future optimization

### Process improvements
- The "sweep + joint optimization" two-phase approach is systematic and trustworthy
- 16 benchmarks × ~10s each = ~3 minutes of sweep time — very efficient
- The floor-splitting pattern (create new variable, assign to specific segment type) is reusable for further spatial refinement
- TOPP floor tuning is now exhausted; future iterations should shift to controller or architecture improvements
