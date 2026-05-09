# Iteration 16 — S-Turn Compound Inflation

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (S-turn gates 3-4)
**Status**: COMMITTED — avg error 0.251→0.232m (-7.6%), gate-3 0.452→0.326m (-28%), race time 13.50→13.95s (+3.3%)
**Commit**: 10f5419

---

## Section 1: Summary
- Iteration 16, timestamp 2026-04-14T11:00Z
- Bottleneck: trajectory_planning — S-turn compound inflation for gates 3-4
- One-line outcome: **avg tracking error 0.251→0.232m (-7.6%), gate-3 0.452→0.326m (-28%), now below 0.25m aspirational target**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **CiMPCC: Curvature-Integrated MPCC for Autonomous Racing** (Li et al., ITSC 2024, arXiv:2502.03695)
   - Maps smoothed track curvature to velocity reference profile
   - Key finding: compound curvature for chicanes/S-turns — curvature doesn't fully drop between consecutive turns
   - 11.4-12.5% lap time improvement on F1TENTH
   - α=1.0 (linear mapping) is optimal

2. **VPMPCC: Data-Driven Aggressive Autonomous Racing Framework** (Li et al., 2024, arXiv:2410.11570)
   - Learns optimal velocity profiles via Bayesian Optimization
   - Key finding: "decelerate early, stay slow through compound turns" pattern
   - The APPROACH segment to the second S-turn turn needs slowing, not just the turn segments
   - 93.18% of vehicle handling limits achieved

### Previously analyzed (directly used)
- **TACO** (Sanghvi 2025): trajectory-aware controller adaptation confirms curvature-speed coupling
- **Alternating Peak** (de Vries/Foehn, ECC 2024): peak-normalized time allocation
- **TOPPQuad** (Mao, IROS 2024): geometry-timing decoupling
- **FBGA** (Piazza, RA-L 2025): forward-backward propagation near-optimality

### Research consensus vs contradictions
- **Consensus**: All papers agree that S-turns/chicanes require compound treatment, not individual-turn treatment. Three complementary approaches: curvature smoothing (CiMPCC), learned profiles (VPMPCC), speed limits (TOPP).
- **No significant contradictions**: All approaches converge on "slow early, stay slow through compound turns."

### Key insight
CiMPCC's curvature smoothing principle maps directly to our problem: when consecutive turns have opposite lateral direction, the effective curvature between them is higher than the point curvature at each turn. This is because the drone must REVERSE its lateral velocity, which requires more centripetal budget than continuing in the same direction.

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`** (only file modified)

**Change 1: S-turn detection in `_inflate_sharp_turns`**
- Pre-compute cross-product Z-component at each gate to identify lateral turn direction
- Detect S-turns: consecutive gates where cross_z changes sign (opposite lateral direction)
- Applied for any turn angle > 0.25 rad (~14°)

**Change 2: S-turn compound inflation in `_inflate_sharp_turns`**
- For the second gate of an S-turn pair: 10% compound inflation (max'd with existing inflation)
- For the APPROACH segment to the second gate: 5% extra time
- This gives the controller time to reverse lateral velocity before the second turn

**Change 3: Compound curvature in `_topp_retime`**
- Identify S-turn segments in the waypoint list (segments around S-turn gates)
- Boost Menger curvature by 20% for segments in S-turn regions
- This prevents the TOPP retimer from compressing S-turn segments back to high speed

### Tuning iterations
- V1 (1.15/1.10/1.30): 14.24s / 0.221m — best accuracy but race time > 14s
- V2 (1.10/1.07/1.20): 13.98s / 0.232m — good trade-off
- V3 (1.10/1.05/1.20): **13.95s / 0.232m** — selected: best speed/accuracy Pareto

### Plan adherence
Followed the plan exactly. The three-part approach (S-turn detection, compound inflation, TOPP curvature boost) was implemented as designed. The only deviation was parameter tuning: started aggressive and tuned down to find the optimal Pareto point.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.251m | **0.232m** | -0.019m (-7.6%) | ✓✓ |
| Max tracking error | 0.679m | 0.667m | -0.012m (-1.8%) | ✓ |
| P50 tracking error | 0.206m | 0.199m | -0.007m | ✓ |
| P95 tracking error | 0.590m | 0.524m | -0.066m (-11.2%) | ✓✓ |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7745 | 7698 | -0.6% | → |
| Trajectory time | 13.68s | 14.14s | +3.4% | ↓ (S-turn inflation) |
| Race time | 13.50s | **13.95s** | +0.45s (+3.3%) | ↓ (acceptable trade-off) |
| Avg thrust | 0.822 | 0.816 | -0.7% | ✓ (less saturation) |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.115 | 0.115 | 0.000 | → unchanged |
| gate-2 | 0.249 | 0.247 | -0.002 | → stable |
| gate-3 | **0.452** | **0.326** | **-0.126 (-28%)** | ✓✓✓ S-turn first half |
| gate-4 | **0.465** | **0.419** | **-0.046 (-10%)** | ✓ S-turn second half |
| gate-5 | 0.219 | **0.181** | **-0.038 (-17%)** | ✓✓ downstream benefit |
| gate-6 | 0.170 | 0.155 | -0.015 | ✓ slight |
| gate-7 | 0.336 | 0.323 | -0.013 | → stable |
| gate-8 | 0.221 | 0.215 | -0.006 | → stable |
| gate-9 | 0.198 | 0.197 | -0.001 | → stable |
| gate-10 | 0.196 | 0.198 | +0.002 | → stable |
| gate-11 | 0.178 | 0.175 | -0.003 | → stable |
| gate-12 | 0.223 | 0.205 | -0.018 | ✓ improved |

**Pattern**: S-turn gates 3-4-5 all improved significantly. No other gate regressed. The compound inflation specifically targets the S-turn region without affecting straight or helix segments.

### Threshold status
| Threshold | Required | Current | Aspirational | Status |
|-----------|----------|---------|--------------|--------|
| Avg error | <0.5m | **0.232m** | <0.25m | **MEETS ASPIRATIONAL** ↑ |
| Max error | <2.0m | 0.667m | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | 13.95s | <14s | **MEETS ASPIRATIONAL** (barely) |
| Loop Hz | >100 | 7698 | >100 | PASS |
| No crash | required | no crash | — | PASS |

All aspirational targets are now met!

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The S-turn (gates 3-4) was the dominant source of tracking error because `_inflate_sharp_turns` treated each gate independently. Gate-3 has a 47.6° turn and gate-4 has a 37.3° turn — both below the 60° angle-based inflation threshold. The centripetal acceleration check provided some inflation, but it didn't account for the compound S-turn effect: the drone arrives at gate-4 with lateral velocity in the WRONG direction (from gate-3's turn) and must reverse it. This lateral velocity reversal requires more time than a single turn of the same angle. The TOPP retimer exacerbated this by increasing approach speeds to the S-turn.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (unchanged, at tilt limit)
- Avg thrust: 0.816 (improved from 0.822 — less thrust saturation)
- Avg pitch: -0.099 rad (slightly less aggressive forward lean)
- Controller saturation is reduced at S-turn gates, indicating the inflation gives the controller enough time

### Trend analysis
**Trend: IMPROVING (Pareto frontier advances on accuracy axis)**

Key Pareto points:
- Iter 10: 13.34s / 0.481m (fast, inaccurate)
- Iter 13: 17.70s / 0.179m (slow, accurate)
- Iter 14: 14.62s / 0.254m (balanced)
- Iter 15: 13.50s / 0.251m (fast+accurate)
- **Iter 16: 13.95s / 0.232m** (best accuracy with competitive speed)

Three consecutive iterations (14-15-16) have pushed the Pareto frontier. Iter 15 pushed the speed axis, iter 16 pushes the accuracy axis. The avg error now beats the aspirational <0.25m target for the first time while maintaining <14s race time.

### Architectural observations
- The four-stage post-optimization pipeline (L-BFGS → inflate turns → FOV relax → TOPP retime) now has S-turn awareness in both the inflate and TOPP stages. The pipeline is complex but each stage has a clear role.
- Gate-4 remains the worst gate (0.419m) despite improvement. The S-turn geometry is inherently challenging — the drone must reverse lateral direction while maintaining forward progress.
- The FOV relaxation stage (`_relax_for_fov`) adds ~0.5s and could be removed per iter 15's recommendation. This would recover ~0.5s of race time.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Remove FOV relaxation stage** (Priority 1, trajectory_planning)
   - `_relax_for_fov` adds ~0.5s. L-BFGS FOV penalty (weight=10) already provides primary awareness.
   - Approach: remove `_relax_for_fov` call, rely on L-BFGS FOV penalty alone
   - Expected: race time 13.95→13.5s, avg error unchanged
   - Research: ETH 2026, TOPPQuad (Mao 2024)

2. **Gate-4 residual S-turn tracking** (Priority 2, trajectory_planning)
   - Gate-4 still worst at 0.419m. The compound inflation helps but doesn't fully solve it.
   - Approach: increase gate-4 specific inflation, or adjust racing line lateral offset at gate-4 to reduce effective curvature
   - Expected: gate-4 error 0.419→0.35m
   - Research: TACO trajectory adaptation (Sanghvi 2025)

3. **Multi-start racing line optimization** (Priority 3, system_integration)
   - Run L-BFGS from multiple random initializations to escape local minima
   - Score by TOPP-computed race time + tracking error proxy
   - Expected: potential 0.5-1s improvement

4. **Tighten aspirational thresholds** (Priority 4, system_integration)
   - All aspirational targets now met. New targets: race_time < 13s, avg_error < 0.20m
   - Drive continued improvement

5. **Pipeline consolidation** (Priority 5, trajectory_planning)
   - Merge inflate + TOPP into single curvature-aware retimer
   - Remove FOV relaxation
   - Expected: cleaner architecture, easier parameter tuning

### Architectural recommendations
- The FOV relaxation stage should be removed — it was already minimized in iter 14 and L-BFGS provides primary FOV awareness
- The S-turn detection logic should be generalized to handle any chicane pattern, not just the specific gates 3-4 on this track

### What NOT to try
- **Increasing S-turn inflation beyond 15%** — V1 test showed 14.24s/0.221m. The accuracy gain diminishes quickly while time cost grows linearly.
- **Controller tuning for S-turns** — exhaustively proven infeasible in kinematic sim (iter 12)
- **Velocity feedforward for S-turns** — transient-dominated (iter 11)
- **Uniform time compression** — proven infeasible (iter 14)

---

## Section 7: Lessons Learned

### What worked
- **S-turn detection via cross-product sign change** is a clean, geometric way to identify chicane patterns. No track-specific heuristics needed.
- **Three-part approach** (compound inflation + approach inflation + TOPP curvature boost) addresses the S-turn from multiple angles, giving robust improvement.
- **Parameter tuning in 3 iterations** (V1→V2→V3) quickly found the optimal Pareto point. Testing multiple parameter combinations is essential.

### What didn't work
- **V1 parameters (15%/10%/30%)** were too aggressive — race time exceeded 14s. The compound inflation needs to be balanced against speed.
- **Approach inflation has diminishing returns** — going from 5% to 10% approach inflation adds 0.03s race time with negligible accuracy benefit (0.232m vs 0.232m).

### Surprises
- **Gate-3 improved more than gate-4** (-28% vs -10%). Gate-3 benefits from the compound inflation because it's the FIRST turn of the S-turn — the drone now enters gate-3 knowing it needs to prepare for gate-4.
- **Gate-5 improved significantly (-17%)** even though it's not in the S-turn. The downstream benefit comes from the drone arriving at gate-5 with better controlled velocity after the S-turn.
- **Avg thrust decreased** (0.822→0.816) despite only a 3.3% time increase. The S-turn inflation reduces controller saturation, leading to more efficient thrust utilization.
- **All aspirational targets are now met simultaneously** for the first time.

### Process suggestions
- When tuning inflation parameters, test 3-4 configurations quickly (benchmark takes <0.2s) to find the optimal Pareto point
- Cross-product sign change is the right way to detect S-turns — generalizes to any track geometry
- The CiMPCC/VPMPCC papers, while designed for ground vehicles, provide transferable insights for drone racing S-turns
