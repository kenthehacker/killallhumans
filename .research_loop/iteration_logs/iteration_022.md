# Iteration 22 — Sim-Based Racing Line Selection

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (proxy objective ≠ tracking error)
**Status**: COMMITTED — gate-3 error 0.374→0.209m (-44%), avg error 0.223→0.206m (-7.6%)
**Commit**: 6fb6eea

---

## Section 1: Summary
- Iteration 22, timestamp 2026-04-14T14:24Z
- Bottleneck: trajectory_planning — L-BFGS proxy objective (path_length + curvature²) doesn't correlate with actual tracking error
- One-line outcome: **Replaced proxy-based racing line selection with kinematic-sim-based evaluation. Gate-3 S-turn error dropped 44% (0.374→0.209m), avg tracking error improved 7.6% (0.223→0.206m). Race time regressed 1.1% (13.99→14.15s).**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Computing the Racing Line using Bayesian Optimization** (Jain & Morari, CDC 2020, arXiv:2002.04794)
   - Frames racing line selection as black-box BO problem with sim oracle
   - Key insight: evaluate by actual lap time simulation, not geometric proxy
   - Path-velocity decoupling: fix path, solve 1D speed profile
   - Converges in <3 minutes for 20D parameter space

2. **Methods for Multi-objective Optimization PID Controller for Quadrotor UAVs** (Vaiuso et al., 2025, arXiv:2509.17423)
   - Compares metaheuristics, BO, DRL for PID tuning via closed-loop sim
   - Grey Wolf Optimizer achieves 42.7% cost reduction vs manual baseline
   - Key insight: closed-loop sim evaluation with composite cost beats proxy-based tuning
   - Multi-objective cost function: 8 weighted sub-terms including time, error, oscillation

3. **Sampling-Based Motion Planning with Online Racing Line Generation** (Ögretmen et al., IV 2024, arXiv:2403.18643)
   - Generates N candidate trajectories, evaluates by multi-term cost
   - Online racing line generation: re-optimize from current state
   - Key insight: evaluate candidates by actual dynamic feasibility, not just geometry
   - 800 candidates at 100Hz using Frenetix C++ library

### Key insight from cross-validation
The architectural insight was confirmed across all 6 papers analyzed: proxy objectives are systematically inferior to closed-loop evaluation. The specific insight unique to our system was that our existing 10-candidate multi-start L-BFGS already generates diverse racing lines — the missing step was evaluating them by simulation instead of by the L-BFGS objective value.

### Research consensus vs contradictions
- **Strong consensus (5+ papers)**: Multi-candidate evaluation with sim-based selection is the right paradigm
- **No contradictions** on the core approach
- **Nuance**: BO Racing Line suggests iterative refinement (100+ evaluations). We found simple enumeration of 10 candidates sufficient since each is a fully optimized L-BFGS solution

---

## Section 3: Implementation

### Changes made
**File: `planning/racing_line.py`**

**Change 1: Collect all L-BFGS results (not just best)**
- Previously: `if result.fun < best_result.fun: best_result = result`
- Now: `all_results.append(result)` — keeps all 10 candidate solutions

**Change 2: Sim-based selection method `_select_by_sim()`**
- For each of 10 candidates: build gate waypoints → build full trajectory (TrajectoryOptimizer) → run kinematic sim → measure error
- Selection by composite score: `0.7 * avg_tracking_error + 0.3 * worst_gate_error`
- Fallback to L-BFGS objective if sim evaluation fails (T-MPC Theorem 2 guarantee)

**Change 3: Lightweight kinematic evaluator `_kinematic_eval()`**
- Self-contained PD controller: kp_xy=6, kd_xy=4, kp_z=8, kd_z=5, ff_accel=0.4
- Physics: drag=0.5, max_accel=15, max_speed=15, dt=0.02
- Same physics as benchmark but coarser timestep (2x faster evaluation)
- No external dependencies (avoids planning→control package dependency)

### Plan adherence
Followed the plan exactly. No deviations needed — the implementation was straightforward because the multi-start infrastructure already existed.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.223m | **0.206m** | **-0.017m (-7.6%)** | ✓✓ |
| Max tracking error | 0.678m | 0.682m | +0.004m | → |
| P50 tracking error | 0.190m | **0.176m** | **-0.014m (-7.4%)** | ✓ |
| P95 tracking error | 0.478m | **0.449m** | **-0.029m (-6.1%)** | ✓ |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7797 | 8085 | +3.7% | → |
| Trajectory time | 14.17s | 14.35s | +0.18s | ↓ mild |
| Race time | 13.99s | **14.15s** | **+0.16s (+1.1%)** | ↓ mild |
| Avg thrust | 0.799 | 0.796 | -0.4% | → |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.115 | 0.115 | 0 (+0%) | unchanged |
| gate-2 | 0.234 | 0.246 | +0.012 (+5.1%) | mild regression |
| gate-3 | 0.374 | **0.209** | **-0.165 (-44%)** | **MAJOR IMPROVEMENT** |
| gate-4 | 0.310 | **0.281** | **-0.029 (-9.4%)** | good improvement |
| gate-5 | 0.155 | **0.139** | **-0.016 (-10.3%)** | good improvement |
| gate-6 | 0.158 | **0.148** | **-0.010 (-6.3%)** | improvement |
| gate-7 | 0.308 | 0.311 | +0.003 (+1.0%) | unchanged (now worst) |
| gate-8 | 0.219 | 0.218 | -0.001 | unchanged |
| gate-9 | 0.209 | 0.206 | -0.003 | unchanged |
| gate-10 | 0.214 | 0.210 | -0.004 | unchanged |
| gate-11 | 0.176 | 0.174 | -0.002 | unchanged |
| gate-12 | 0.201 | 0.199 | -0.002 | unchanged |

### Threshold status
| Threshold | Required | Current | Aspirational | Status |
|-----------|----------|---------|--------------|--------|
| Avg error | <0.5m | **0.206m** | <0.25m | **MEETS ASPIRATIONAL** |
| Max error | <2.0m | **0.682m** | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | **14.15s** | <14s | PASS (misses aspirational by 0.15s) |
| Loop Hz | >100 | 8085 | >100 | PASS |
| No crash | required | no crash | — | PASS |

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The L-BFGS proxy objective (path_length + smoothness*curvature²) was systematically selecting racing lines that are geometrically efficient but dynamically poor for the PD controller at the S-turn (gate-3). The sim-based evaluator found a different candidate from the same 10-start pool that the controller tracks 44% better. This confirms the iter 21 architectural diagnosis: the proxy objective and tracking error are decoupled. The new selection criterion directly optimizes what matters — actual sim tracking error.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (at tilt limit, unchanged)
- Avg thrust: 0.796 (slightly lower — less aggressive flight)
- Avg pitch: -0.110 (essentially unchanged)
- Gate-3 improvement was concentrated in the S-turn section — the selected racing line has a smoother approach angle that the PD controller can follow more accurately
- Helix gates (7-12): virtually zero change

### Trend analysis
**Trend: IMPROVING — Architectural change broke through plateau**

The sim-based selection was the first genuine architectural improvement in 6+ iterations (vs parameter tuning). It delivered the largest single-iteration improvement since iteration 13 (which introduced smoothness_weight for helix gates). The diminishing returns trend from iterations 17-21 was broken by changing the selection criterion rather than tuning parameters.

Pareto frontier evolution:
- Iterations 15-21: parameter tuning within L-BFGS proxy paradigm → 0.218-0.251m avg error
- Iteration 22: architectural change to sim-based selection → 0.206m avg error (new best)

### Architectural issues remaining
1. **Race time regression**: Sim-based selection has no speed term → selects smoother racing line at expense of speed. Need composite score with time penalty.
2. **Gate-7 stuck at ~0.31m for 6+ iterations**: Completely unaffected by any racing line change. The helix geometry is fixed regardless of S-turn racing line. Needs different approach.
3. **Kinematic sim fidelity**: The sim evaluator uses dt=0.02 (coarser than benchmark's dt=0.01). Results match well but there's inherent model mismatch.
4. **Evaluation cost**: 10 full trajectory builds + 10 kinematic sims adds ~5s to planning. Acceptable for offline but would need optimization for online replanning.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Race time recovery via speed-aware composite score** (Priority 1, trajectory_planning)
   - Race time 14.15s → target <14.0s
   - Add time penalty to sim-based composite: `score = 0.6*avg_err + 0.2*worst_gate + 0.2*race_time_normalized`
   - Expected: recover to ~14.0s keeping most tracking improvement
   - Research: BO Racing Line (Jain 2020) — multi-objective time + tracking

2. **Gate-7 helix entry** (Priority 2, trajectory_planning)
   - At 0.311m for 6+ iterations — isolated from all racing line changes
   - May need per-section racing line or helix-specific inflation
   - Research: CiMPCC (Li 2024), Mastering Diverse Tracks (Yu 2025)

3. **Gate-2 approach** (Priority 3, trajectory_planning)
   - At 0.246m (+5.1% regression from 0.234m)
   - The sim-based selector traded gate-2 for gate-3; could add gate-2 weighting

4. **MPCC controller upgrade** (Priority 4, control)
   - Contouring/progress error decomposition
   - Would provide tracking improvements across ALL gates
   - Research: MPCC++ (Krinner 2024)

5. **PyBullet integration** (Priority 5, system_integration)
   - Real physics validation for competition readiness
   - Would make sim-based racing line selection even more valuable

### Architectural recommendations
- The sim-based selection paradigm is a keeper. Future iterations should refine the composite score (add speed term) rather than revert to proxy selection.
- Consider extending sim-based evaluation to other optimization decisions (inflation parameters, compression floors) — same principle applies.
- The per-gate error decomposition in the evaluator could drive per-section racing line optimization (different smoothness weights for S-turn vs helix).

### What NOT to try
- **Reverting to L-BFGS proxy selection**: The 44% gate-3 improvement proves the proxy is fundamentally flawed for tracking error optimization
- **Increasing N_STARTS beyond 10**: Diminishing returns — 10 diverse starts already cover the key basins
- **Gain scheduling in kinematic sim**: Still fundamentally doesn't work (iter 12)
- **Drag compensation**: Still removes beneficial damping (iter 9)

---

## Section 7: Lessons Learned

### What worked
- **The core idea was validated immediately**: Sim-based selection found a dramatically better racing line on the first try. The L-BFGS proxy was clearly selecting a locally optimal but dynamically poor candidate.
- **T-MPC Theorem 2 approach**: Including zero-init as a candidate guaranteed no regression risk. The sim evaluator correctly ranked the zero-init candidate lower than the selected one.
- **Self-contained kinematic evaluator**: Inline PD controller avoided cross-package dependencies while replicating benchmark physics accurately.
- **Coarser dt=0.02 was sufficient**: Results match dt=0.01 benchmark closely, suggesting the ranking is robust to timestep choice.

### What didn't work
- **No speed term in composite score**: The selection purely minimizes tracking error, which selects a smoother but slower racing line. Race time regressed 1.1%.

### Surprises
- **The magnitude of improvement was unexpected**: 44% reduction at gate-3 in a single iteration, after 5+ iterations of incremental tuning yielded only single-digit percent changes. This strongly validates the "proxy vs actual" hypothesis.
- **The selected racing line was from an existing candidate**: The improvement came not from a new optimization approach but from better evaluation of existing candidates. The 10-start pool already contained the good solution — it was just being filtered out by the wrong criterion.
- **Helix gates were completely unaffected**: The S-turn racing line and helix racing line are independently optimized regions. Changes to one don't propagate to the other. This means gate-7 requires a targeted approach.
- **Gate-2 → gate-3 trade-off**: The new racing line slightly worsens gate-2 to dramatically improve gate-3. This suggests the two gates are geometrically coupled — a smoother gate-3 approach requires a different entry angle through gate-2.

### Process suggestions
- When architectural diagnostics identify a "proxy vs actual metric" mismatch, the fix should be evaluated before any further parameter tuning. This iteration's 44% improvement dwarfs all parameter tuning gains from iterations 17-21.
- The sim-based evaluation paradigm could be extended to other optimization decisions (time allocation, inflation parameters, compression floors).
- Racing_line unit test now takes 150ms (vs 10ms before). This is acceptable but should be monitored — if N_STARTS or sim fidelity increase, test time could grow.
