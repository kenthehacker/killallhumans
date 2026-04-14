# Iteration 17 — Remove FOV Relaxation Post-Processing

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (FOV relaxation removal)
**Status**: COMMITTED — race time 13.95→13.62s (-2.4%), avg error 0.232→0.248m (+6.9%)
**Commit**: f15087f

---

## Section 1: Summary
- Iteration 17, timestamp 2026-04-14T12:12Z
- Bottleneck: trajectory_planning — remove `_relax_for_fov` post-processing stage
- One-line outcome: **race time 13.95→13.62s (-2.4%), pipeline simplified from 4 to 3 stages, all aspirational targets still met**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **MonoRace: Winning Champion-Level Drone Racing** (Bahnam et al., 2026, arXiv:2601.15222)
   - A2RL competition winner using monocular pipeline
   - NO post-processing FOV relaxation — perception handled via adaptive cropping + EKF
   - 16.56s lap, 28.23 m/s peak, 100% reliability

2. **PA-MPPI: Perception-Aware MPPI** (Zhai et al., UZH 2025, arXiv:2509.14978)
   - Perception cost embedded directly in MPPI sampling loop (not post-processed)
   - 50 Hz real-time on commodity hardware
   - Demonstrates integrated perception-planning is superior to staged approaches

3. **Mastering Diverse Tracks** (Yu et al., 2025, arXiv:2512.09571)
   - RL-based vision racing with perception reward baked into training
   - Perception reward weight = 0.1, keeps gate in FOV during turns
   - Two-phase training: soft collision → hard collision

### Previously analyzed (directly used)
4. **Perception-Aware Planning** (ETH/UZH 2026, arXiv:2603.04305)
5. **FOV CBF Quadrotor** (Pan et al., 2025, arXiv:2502.01009)
6. **Drift-Corrected VIO** (2025, arXiv:2512.20475)

### Research consensus vs contradictions
- **Consensus (6/6 papers)**: FOV awareness should be INTEGRATED into the planning loop, not post-processed. All papers embed perception constraints directly in their optimizer/policy/controller.
- **No contradictions**: Not a single paper advocates for post-processing FOV relaxation.
- **Key insight**: The A2RL competition winner (MonoRace) uses ZERO post-processing FOV relaxation and achieves 100% race completion at champion-level speeds.

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`** (only file modified)

**Change 1: Remove FOV relaxation call from generate()**
- Removed the `add_fov_constraints()` → `_relax_for_fov()` → `_generate_trajectory()` block
- This eliminates one intermediate trajectory generation + the FOV relaxation loop
- The L-BFGS optimizer's FOV penalty (weight=10) remains as the integrated mechanism
- Pipeline simplified: L-BFGS → inflate_turns → TOPP retime (was 4 stages)

**Change 2: Raise TOPP max_compression from 0.65 to 0.68**
- Without FOV relaxation, the TOPP retimer compresses helix segments too aggressively
- Raising the floor from 65% to 68% protects helix turns from over-compression
- Tested 0.65, 0.68, 0.70 — 0.68 gives optimal speed/accuracy trade-off

### Tuning iterations
- FOV removal only (max_compression=0.65): 13.47s / 0.254m — too aggressive, helix regressed
- max_compression=0.68: **13.62s / 0.248m** — selected: best Pareto
- max_compression=0.70: 13.75s / 0.246m — too conservative, diminishing speed gain
- a_centripetal=9.0 (with 0.65): identical to 0.65 — not the binding constraint

### Plan adherence
Followed the plan (remove FOV relaxation) but added a compensating change (max_compression raise) that wasn't in the original plan. The compensation was needed because the FOV relaxation was providing beneficial turn inflation for the helix, not just FOV awareness.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.232m | **0.248m** | +0.016m (+6.9%) | ↓ trade-off |
| Max tracking error | 0.667m | 0.803m | +0.136m (+20.4%) | ↓ trade-off |
| P50 tracking error | 0.199m | 0.206m | +0.007m | → |
| P95 tracking error | 0.524m | 0.556m | +0.032m | ↓ slight |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7698 | 7720 | +0.3% | → |
| Trajectory time | 14.14s | 13.81s | -0.33s (-2.3%) | ✓ |
| Race time | 13.95s | **13.62s** | **-0.33s (-2.4%)** | ✓✓ |
| Avg thrust | 0.816 | 0.827 | +1.3% | ↓ (faster flight) |
| Traj gen time | 64ms | 47ms | -27% | ✓ (removed intermediate generation) |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.115 | 0.115 | 0.000 | → unchanged |
| gate-2 | 0.247 | 0.233 | **-0.014** | ✓ improved |
| gate-3 | 0.326 | 0.329 | +0.003 | → stable |
| gate-4 | 0.419 | 0.413 | -0.006 | → stable |
| gate-5 | 0.181 | 0.181 | 0.000 | → unchanged |
| gate-6 | 0.155 | 0.156 | +0.001 | → stable |
| gate-7 | 0.323 | **0.351** | **+0.028** | ↓ helix regression |
| gate-8 | 0.215 | **0.267** | **+0.052** | ↓ worst regression |
| gate-9 | 0.197 | **0.227** | **+0.030** | ↓ helix regression |
| gate-10 | 0.198 | **0.228** | **+0.030** | ↓ helix regression |
| gate-11 | 0.175 | **0.200** | **+0.025** | ↓ helix regression |
| gate-12 | 0.205 | **0.243** | **+0.038** | ↓ helix regression |

**Pattern**: Gates 1-6 unchanged or improved. Gates 7-12 (helix) all regressed by 0.025-0.052m. The FOV relaxation was providing ~3-5% time inflation for the helix. The max_compression increase (0.65→0.68) partially compensated but didn't fully replace the lost inflation.

### Threshold status
| Threshold | Required | Current | Aspirational | Status |
|-----------|----------|---------|--------------|--------|
| Avg error | <0.5m | **0.248m** | <0.25m | **MEETS ASPIRATIONAL** (barely) |
| Max error | <2.0m | 0.803m | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | **13.62s** | <14s | **MEETS ASPIRATIONAL** ✓ |
| Loop Hz | >100 | 7720 | >100 | PASS |
| No crash | required | no crash | — | PASS |

All aspirational targets still met.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The `_relax_for_fov` post-processing stage was doing DOUBLE DUTY: (1) FOV awareness (its documented purpose) and (2) beneficial time inflation for helix turns (an undocumented side effect). The FOV relaxation identified helix turn segments as having high FOV penalty and inflated them by ~3-8%, which happened to give the PD controller enough time to track the helix. Removing it exposed the helix to TOPP's aggressive compression, causing 0.025-0.052m regression on gates 7-12.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (unchanged, at tilt limit)
- Avg thrust: 0.827 (increased from 0.816 — faster flight needs more thrust)
- Avg pitch: -0.101 (slightly more forward lean than 0.099)
- Controller is working harder on the helix but not saturating

### Trend analysis
**Trend: IMPROVING (Pareto frontier advancing on speed axis)**

Key Pareto points:
- Iter 14: 14.62s / 0.254m
- Iter 15: 13.50s / 0.251m
- Iter 16: 13.95s / 0.232m (accuracy push)
- **Iter 17: 13.62s / 0.248m (speed push)**

Three iterations (15-16-17) have alternated between speed and accuracy optimization. This is a healthy Pareto improvement pattern. The next iteration should focus on accuracy recovery.

### Architectural observations
- The pipeline is now cleaner: L-BFGS → inflate_turns → TOPP retime (3 stages vs 4)
- `_relax_for_fov` method still exists in code as a potential future tool
- The `add_fov_constraints` method is no longer called during generation
- Trajectory generation is ~27% faster (47ms vs 64ms) due to removing one intermediate generation

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Gate-4 + helix accuracy recovery** (Priority 1, trajectory_planning)
   - Gate-4 still worst at 0.413m. Helix gates 7-12 regressed 0.025-0.052m each.
   - Approach: Increase _inflate_sharp_turns inflation for helix-specific turns (gates with moderate angles 20-40°). Currently only gates with >60° turns get angle-based inflation.
   - Expected: Helix gates recover to iter 16 levels, gate-4 drops to 0.38m
   - Research: TACO (Sanghvi 2025), CiMPCC (Li 2024)

2. **Multi-start racing line optimization** (Priority 2, system_integration)
   - Run L-BFGS from 5-10 random initializations, score by TOPP-computed race time
   - Expected: potential 0.5-1s improvement from escaping local minima
   - Research: Sequence Modeling (2025)

3. **Yaw optimization for FOV** (Priority 3, trajectory_planning)
   - Instead of slowing trajectory for FOV, optimize yaw profile to keep gates visible
   - Drift-Corrected VIO: heading-based FOV adds +0% race time
   - Expected: better FOV awareness with zero time cost
   - Research: Drift-Corrected VIO (2025), MonoRace (2026)

4. **Tighten aspirational thresholds** (Priority 4, system_integration)
   - New targets: race_time < 13s, avg_error < 0.20m
   - Current: 13.62s / 0.248m — still room to improve on both axes

5. **MPCC controller upgrade** (Priority 5, control)
   - ETH 2026 demonstrates MPCC achieves 0.07m avg error at 9.8 m/s
   - Contouring/progress error decomposition prevents corner-cutting
   - Major architectural change — defer until trajectory planning is optimized
   - Research: ETH 2026, MPCC++ (Krinner 2024)

### Architectural recommendations
- The helix needs dedicated treatment beyond generic turn inflation — its consistent curvature with elevation changes creates a unique tracking challenge
- The max_compression parameter in TOPP should eventually be replaced by a per-segment controller-capability model that predicts tracking error
- MonoRace's approach of camera forward-pitch (43-50°) to keep gates in view is an alternative to trajectory-based FOV handling

### What NOT to try
- **a_centripetal reduction in TOPP** — tested, has no effect because max_compression is the binding constraint
- **max_compression > 0.70** — tested, speed gain becomes marginal (<0.20s) while accuracy loss persists
- **Uniform time compression** — proven infeasible in iter 14
- **Controller tuning for helix** — exhaustively proven infeasible in kinematic sim (iter 12)

---

## Section 7: Lessons Learned

### What worked
- **FOV relaxation removal**: Successfully simplified the pipeline from 4 to 3 stages and recovered 0.33s of race time
- **max_compression tuning**: Quick parametric sweep (0.65, 0.68, 0.70) found the optimal balance
- **Research consensus was clear**: 6/6 papers agreed on integrated over post-processed FOV

### What didn't work
- **Pure removal (max_compression=0.65)**: Helix regressed too much — avg error exceeded 0.25m aspirational
- **a_centripetal=9.0**: No effect because max_compression was the binding constraint, not curvature-based speed limits. Useful negative finding.

### Surprises
- **FOV relaxation was doing double duty**: Documented purpose was FOV awareness, but the real value was helix turn inflation. This is a common anti-pattern — a module has undocumented side effects that other modules depend on.
- **Gate-8 was the most affected**: The entry to the helix (gate-7→gate-8 transition) is the most sensitive to timing changes. Gate-8 error jumped from 0.215 to 0.267m.
- **Gates 1-6 were unaffected or improved**: The FOV relaxation only touched turn segments (>30° turns), so approach/S-turn gates weren't affected.
- **Trajectory generation got 27% faster**: Removing the intermediate trajectory generation (needed only for FOV evaluation) was a significant compute savings.

### Process suggestions
- Before removing a "redundant" stage, check if it has undocumented side effects by analyzing per-gate breakdowns
- The parametric sweep (test 3-4 values of max_compression) was efficient — each benchmark takes <0.2s
- When a change trades speed for accuracy, commit only if ALL aspirational targets remain met
