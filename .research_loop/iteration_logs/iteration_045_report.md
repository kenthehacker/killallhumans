# Iteration 45 — L-BFGS Time Weight Increase + Gate-3 ILC Alpha

**Date**: 2026-04-15
**Bottleneck**: trajectory_planning (speed recovery via L-BFGS time incentive)
**Status**: COMMITTED — race time 13.51→13.31s (-1.5%), avg error 0.159→0.186m (+17.0%)
**Commit**: (pending)

---

## Section 1: Summary
- Iteration 45, timestamp 2026-04-15T15:45Z
- Bottleneck: trajectory_planning (speed recovery)
- One-line outcome: **Race time reduced 13.51→13.31s (-1.5%) by increasing L-BFGS time_weight 2.0→2.3; gate-3 improved via ILC alpha 0.4→0.45; avg error 0.159→0.186m, under 0.25m threshold**

---

## Section 2: Research

### Papers analyzed (3 new, 121→124 total)
1. **Stochastic Control of UAVs: Optimal Tradeoff** (arXiv:2409.10369, 2024)
   - Optimal Covariance Steering for quadrotors in stochastic environments
   - Relevant: demonstrates speed-accuracy tradeoff as a fundamental control property
2. **DiffLab: Two-Phase Framework for Quadrotor Racing** (arXiv:2512.09571, 2025)
   - Progressive difficulty increase for RL-based racing
   - Relevant: staged approach to increasing aggressiveness parallels our progressive inflation reduction
3. **Dynamics-Invariant Quadrotor Control via Scale-Aware DRL** (arXiv:2503.09622, 2025)
   - Scale-aware dynamics randomization for cross-platform control
   - Relevant: demonstrates that controllers can adapt to varied dynamics with appropriate training

### Key insight from research
The existing 121-paper research base already strongly supports this iteration's approach. TOGT (Qin 2024) demonstrates that time minimization should dominate the L-BFGS objective. The key insight: with ILC compensation, the L-BFGS can be pushed to produce more aggressive base times than what was safe at iter 8 (when time_weight=2.5 failed with no ILC).

### Research consensus vs contradictions
- **Consensus**: L-BFGS time_weight controls the base trajectory aggressiveness; ILC can compensate for increased tracking difficulty.
- **Contradiction**: Iter 8 showed time_weight=2.5 caused "convergence to worse local minimum," but that was pre-ILC. At 2.3, L-BFGS finds a good minimum for the current system.

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`**
1. L-BFGS time_weight: 2.0 → 2.3 (line 1256)
   - Increases the weight of total traversal time in the L-BFGS objective function
   - Pushes the optimizer to find shorter segment times, trading penalty violations for speed

**File: `scripts/benchmark.py`**
2. ILC inflection section alpha: 0.4 → 0.45 (line 317)
   - Increases learning rate for the S-turn inflection section (steps 200-440)
   - Improves gate-3 correction by 12.5% per ILC iteration

### Failed attempts during this iteration
1. **TOPP acceleration budget increase** (a_centripetal 10→11, a_longitudinal 8→9.5): Catastrophic — race time exploded to 26.8s. The higher speed limits caused TOPP to produce trajectory timing that destabilized ILC. Even with original floors, still catastrophic.
2. **TOPP floor reduction only** (helix 0.72→0.70, easy 0.59→0.57, protected 0.65→0.63): Marginal — only 0.05s improvement. Confirms iter 21 lesson: floors rarely bind.
3. **Centripetal inflation threshold increase** (4.5→6.0): Zero effect — no turns had centripetal acceleration in the 4.5-6.0 range after iter 44's inflation halving.
4. **time_weight=2.2**: Converged to a WORSE local minimum than 2.3 — gate-3 at 0.390m, gate-4 at 0.498m. L-BFGS landscape is highly non-convex.
5. **time_weight=2.3 + floor reductions**: Slightly faster (13.25s) but worse per-gate tracking. Floor reductions added noise without benefit.

### Plan adherence
Deviated significantly from original plan. The plan called for TOPP acceleration budgets + floor reductions, but both failed. Pivoted to L-BFGS time_weight increase as the primary speed lever.

---

## Section 4: Benchmark Comparison

### Full metrics table

| Metric | Before (iter 44) | After | Delta | Direction |
|--------|-------------------|-------|-------|-----------|
| Race time | 13.51s | **13.31s** | **-1.5%** | **IMPROVED** |
| Trajectory time | 13.69s | 13.47s | -1.6% | improved |
| Avg tracking error | 0.159m | 0.186m | +17.0% | regressed (within budget) |
| Max tracking error | 0.699m | 0.746m | +6.7% | regressed (within budget) |
| P50 tracking error | 0.136m | 0.150m | +10.3% | regressed |
| P95 tracking error | 0.366m | 0.480m | +31.1% | regressed |
| EKF uncertainty | 0.0119m | 0.0119m | 0% | same |
| Gate pass rate | 100% | 100% | 0% | same |
| Loop Hz | 7879 | 7751 | -1.6% | slightly worse |
| Crashed | false | false | — | same |
| Unit tests | 9/9 | 9/9 | — | same |

### Per-gate error breakdown

| Gate | Before | After | Delta | Headroom to 0.25m |
|------|--------|-------|-------|-------------------|
| gate-1 | 0.113m | 0.113m | 0.0% | 0.137m |
| gate-2 | 0.216m | 0.211m | **-2.3%** | 0.039m |
| gate-3 | 0.226m | 0.221m | **-2.2%** | 0.029m |
| gate-4 | 0.190m | **0.302m** | **+58.9%** | **-0.052m (OVER)** |
| gate-5 | 0.161m | 0.204m | +26.7% | 0.046m |
| gate-6 | 0.080m | 0.118m | +47.5% | 0.132m |
| gate-7 | 0.186m | **0.252m** | **+35.5%** | **-0.002m (OVER)** |
| gate-8 | 0.192m | 0.240m | +25.0% | 0.010m |
| gate-9 | 0.117m | 0.117m | 0.0% | 0.133m |
| gate-10 | 0.151m | 0.136m | **-9.9%** | 0.114m |
| gate-11 | 0.126m | 0.124m | -1.6% | 0.126m |
| gate-12 | 0.170m | 0.201m | +18.2% | 0.049m |

### Threshold status
ALL official thresholds passing. Per-gate aspirational 0.25m exceeded at gate-4 (0.302m) and gate-7 (0.252m).

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The L-BFGS time_weight=2.3 pushes the optimizer to find shorter segment times for the mid-course sections (gates 4-8), where the trajectory has the most room for compression. However, these are also the most challenging tracking sections (S-turn exit + helix entry). The ILC corrections are calibrated for time_weight=2.0's trajectory and partially compensate but can't fully absorb the 58% error increase at gate-4.

### Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_abs_roll_rad | 0.85 | 0.85 |
| max_abs_pitch_rad | 0.85 | 0.85 |
| avg_pitch_rad | -0.100 | -0.093 |
| avg_thrust | 0.852 | 0.870 |

Controller effort increased significantly (thrust +2.1%) consistent with faster trajectory. Attitude at saturation — actuator limits are near the binding constraint.

### Trend analysis
- **Speed recovery trajectory**: iter 43 (14.08s) → iter 44 (13.51s, -0.57s) → iter 45 (13.31s, -0.20s)
- **Diminishing returns on speed**: Each iteration yields less speed for more error increase
- **Error budget utilization**: 0.186m / 0.25m = 74% utilized (was 64% after iter 44)
- **Pareto frontier**: Race time vs avg error: (14.08, 0.138) → (13.51, 0.159) → (13.31, 0.186). The frontier is steepening — each 0.1s of race time costs more error.

### Key finding: L-BFGS non-convexity
The L-BFGS landscape is highly non-convex around time_weight=2.0-2.5. Different weights find completely different local minima:
- 2.0: 13.69s trajectory, good tracking
- 2.2: Bad minimum — gate-3 at 0.39m, gate-4 at 0.50m
- 2.3: 13.47s trajectory, acceptable tracking
- 2.5: Failed in iter 8 (pre-ILC)

This means time_weight tuning is unreliable for further speed recovery.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Gate-4 and gate-7 ILC improvement** (priority 1)
   - Area: system_integration
   - Problem: Gate-4 at 0.302m and gate-7 at 0.252m are over aspirational threshold. These gates limit further speed recovery.
   - Approach: (a) Extend inflection section to cover gate-4 (inflection_end from step 440 to 460); (b) Increase helix section max_correction from 0.45→0.50m for gate-7.
   - Expected impact: Gate-4 and gate-7 error reduction 10-20%
   - Research refs: Schoellig 2012, Track-centric ILC 2026

2. **Further inflation reduction (targeted)** (priority 2)
   - Area: trajectory_planning
   - Problem: S-turn junction inflation still at 1.04, helix entry at 1.06. Some gates (1, 9, 10, 11) have >0.1m headroom.
   - Approach: Reduce helix entry 1.06→1.03, helix interior 1.05→1.03. Target segments serving gates with headroom.
   - Expected impact: Race time reduction 0.05-0.10s
   - Research refs: MonoRace (2026), CPC (Foehn 2021)

3. **L-BFGS optimization warm-starting** (priority 3)
   - Area: trajectory_planning
   - Problem: L-BFGS landscape is non-convex — different time_weights find different local minima. No control over which minimum is found.
   - Approach: Try multiple random restarts and select the best. Or use graduated time_weight: start at 2.0, optimize, then restart at 2.5 from previous solution.
   - Expected impact: Could unlock time_weight=2.5+ safely
   - Research refs: F1 Data-Driven Init (2026), TOGT (Qin 2024)

4. **Velocity-corrected ILC scaling for new trajectory** (priority 4)
   - Area: system_integration
   - Problem: The time_weight=2.3 trajectory has different velocity profiles. The velocity correction scaling (0.0/0.4/0.5/0.7 per section) was tuned for the old trajectory.
   - Approach: Re-tune velocity correction scaling for the new trajectory speed profile.
   - Expected impact: Reduce per-gate errors by 5-10%
   - Research refs: Spatial ILC (Lv 2023)

### Architectural recommendations
- The system is approaching actuator saturation (roll/pitch at 0.85 rad cap). Further speed increases will require either (a) raising the tilt cap or (b) accepting that the kinematic sim imposes a speed ceiling.
- The L-BFGS non-convexity is a fundamental limitation. Multi-start optimization or continuation methods would be more robust than single-shot L-BFGS.
- The Pareto frontier is steepening — remaining speed gains require disproportionate accuracy sacrifice. Future iterations may need to shift focus to robustness rather than pure speed.

### Next bottleneck selected
**system_integration** — improve ILC at gate-4 and gate-7 to recover tracking quality lost in this iteration, freeing headroom for further speed recovery in iteration 47.

### What NOT to try
- TOPP acceleration budget increase (a_centripetal > 10.0, a_longitudinal > 8.0) — catastrophic ILC destabilization
- TOPP floor reduction alone — marginal (0.05s max, iter 21 confirmed, iter 45 re-confirmed)
- Centripetal inflation threshold increase — zero effect with current gate geometry
- time_weight=2.2 — finds bad L-BFGS minimum (gate-3=0.39m, gate-4=0.50m)
- time_weight > 2.3 — likely to find even worse minima or fail like iter 8

---

## Section 7: Lessons Learned

### What worked
- **L-BFGS time_weight increase 2.0→2.3**: The first successful time_weight change since iter 8. ILC compensates for the more aggressive base times. Gate-3 and gate-2 actually improved (ILC alpha helps), while mid-course gates regressed.
- **ILC inflection alpha 0.4→0.45**: Targeted improvement at gate-3 (0.226→0.221m) without affecting gate-2 (pre-inflection section with vel_scale=0.0 is protected).
- **Reverting failed approaches quickly**: Tested 5 configurations before finding the right one. Fast iteration within a single session.

### What didn't work
- **TOPP acceleration budgets**: Catastrophic failure. Changing the TOPP speed profile changes the trajectory timing enough to completely destabilize ILC. The ILC is calibrated for a specific trajectory shape — any significant change to the TOPP retimer breaks it.
- **TOPP floor reduction**: Marginal gain (0.05s). Confirms floors are not the binding constraint. The L-BFGS base times + inflation define the trajectory speed, not the TOPP floor.
- **L-BFGS time_weight=2.2**: Found a worse minimum than 2.3. The L-BFGS landscape is non-monotonic — higher weight doesn't always mean worse minimum.

### Surprises
- **time_weight=2.3 found a GOOD minimum while 2.2 found a BAD one**: Non-convex optimization is unpredictable. The 2.3 minimum has 661 trajectory points (similar to baseline 672), while 2.2 had 661 points but radically different segment time distribution.
- **TOPP acceleration budget increase was catastrophic**: Expected modest speed increase, got 26.8s race time. The ILC system is far more sensitive to TOPP changes than to inflation changes.
- **Gate-10 improved (-9.9%)**: Unexpected benefit — the faster trajectory through the helix means less time at high-error points near gate-10.
- **Centripetal threshold had zero effect**: After iter 44's halving, no turns had centripetal between 4.5-6.0 m/s². The centripetal inflation was already negligible.

### Suggestions for improving the iteration process
- When TOPP changes fail catastrophically, it means the ILC is tightly coupled to the trajectory shape. Future speed recovery should focus on changes that preserve the trajectory SHAPE while reducing time (inflation reduction, time_weight).
- Multi-start L-BFGS would avoid the 2.2 vs 2.3 gamble.
- Track the Pareto frontier explicitly: plot (race_time, avg_error) across iterations to visualize diminishing returns.
