# Iteration 29 — Speed Recovery via Post-Optimization Inflation Reduction

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (inflation factors, compression floors)
**Status**: COMMITTED — race time 14.03→13.80s (-1.6%), avg error 0.175→0.185m (+5.5%)
**Commit**: cd3f4f8

---

## Section 1: Summary
- Iteration 29, timestamp 2026-04-14T17:34Z
- Bottleneck: trajectory_planning — race time stuck at 14.03s for 4 iterations while ILC reduced tracking error to 0.175m, creating 0.075m headroom to 0.25m threshold
- One-line outcome: **Reduced post-optimization inflation factors (S-turn: 1.10/1.12→1.08/1.10, TOPP floors: 0.68/0.63→0.66/0.60) to trade accuracy headroom for speed. Race time 14.03→13.80s (-1.6%), first race time improvement since iter 17.**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **SPIRAL: Self-Play Incremental Racing** (Akgün, IEEE ASYU 2025)
   - Progressive speed improvement through self-play: start conservative, incrementally push speed
   - Confirms paradigm: as tracking improves, trajectory can be sped up

2. **On Robustness in Optimization-Based Constrained ILC** (2022)
   - Constraint management in ILC: as convergence improves, constraints can be relaxed
   - Forward-backward splitting for robust constraint satisfaction

3. **Strategizing at Speed: Learned Model Predictive Game** (2026)
   - Multi-agent drone racing with speed optimization strategies
   - Confirms time-accuracy tradeoff is fundamental in racing

### Key insight from cross-validation
The initial plan proposed aggressive reductions (S-turn 1.06/1.08, TOPP 0.58/0.64). Cross-validation identified that racing line selection uses the full trajectory optimizer internally, so parameter changes affect which racing line basin is selected. This predicted the catastrophic failure of the first attempt (26.82s race time from basin switching). The moderated reductions (2% instead of 4%) avoided this trap.

### Research consensus
- **Strong**: Once ILC reduces systematic error, post-optimization inflation (which compensated for that error) can be reduced (Spatial ILC Lv 2023, ILMPC 2508.01103)
- **Strong**: Progressive speedup is safer than aggressive one-shot speedup (SPIRAL 2025)
- **Strong**: Compression floors are the binding constraint for race time (FBGA Piazza 2025)

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`**

In `_inflate_sharp_turns()`:
- S-turn junction inflation: 1.12→1.10 (-2%)
- S-turn standard second-gate: 1.10→1.08 (-2%)
- S-turn approach deceleration: 1.03→1.02 (-1%)
- S-turn first-gate departure: 1.04→1.03 (-1%)
- S-turn junction departure: 1.02→1.01 (-1%)

In `_topp_retime()`:
- max_compression_protected: 0.68→0.66 (-2%)
- max_compression_easy: 0.63→0.60 (-3%)

### Plan adherence
Partially followed. The plan proposed 4% S-turn reductions and 4-5% TOPP reductions. The first attempt at those levels caused catastrophic failure (racing line basin switching). Moderated to 1-2% S-turn and 2-3% TOPP reductions. The lesson: changes that affect racing line selection must be incremental.

### Failed attempt
First attempt: S-turn 1.06/1.08 + TOPP 0.58/0.64. Race time exploded to 26.82s (from 14.03s). Root cause: the kinematic sim inside racing line `_select_by_sim` also uses the trajectory optimizer, so parameter changes caused a different (untrackable) racing line to be selected. The fix was more conservative parameter changes that don't shift the racing line selection.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before (iter 28) | After (iter 29) | Delta | Direction |
|--------|-------------------|-----------------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.17458m | **0.18477m** | **+0.010m (+5.5%)** | ↑ expected |
| Max tracking error | 0.7045m | 0.7248m | +0.020m (+2.9%) | ↑ mild |
| P50 tracking error | 0.1731m | 0.1827m | +0.010m (+5.5%) | ↑ |
| P95 tracking error | 0.3637m | 0.3847m | +0.021m (+5.8%) | ↑ |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 6914 | 7704 | +11.4% | → (still >100) |
| **Race time** | **14.03s** | **13.80s** | **-0.23s (-1.6%)** | **✓ improved!** |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.111 | 0.111 | +0.0% | → |
| gate-2 | 0.170 | 0.178 | +5.0% | ↑ mild |
| gate-3 | 0.222 | 0.236 | +6.3% | ↑ mild |
| gate-4 | 0.167 | 0.186 | +11.4% | ↑ |
| gate-5 | 0.144 | 0.167 | +16.0% | ↑ notable |
| gate-6 | 0.150 | 0.158 | +5.0% | ↑ mild |
| gate-7 | 0.276 | 0.282 | +2.2% | → |
| gate-8 | 0.188 | 0.224 | +19.4% | ↑ notable |
| gate-9 | 0.184 | 0.194 | +5.4% | ↑ mild |
| gate-10 | 0.184 | 0.178 | -3.3% | ✓ improved |
| gate-11 | 0.171 | 0.173 | +0.9% | → |
| gate-12 | 0.129 | 0.134 | +3.8% | ↑ mild |

### Gate pass times
| Gate | Before | After | Savings |
|------|--------|-------|---------|
| gate-4 | 4.41s | 4.32s | -0.09s |
| gate-5 | 5.63s | 5.50s | -0.13s |
| gate-6 | 6.81s | 6.66s | -0.15s |
| gate-7 | 8.06s | 7.88s | -0.18s |
| gate-12 | 14.02s | 13.79s | -0.23s |

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
Race time was stuck at 14.03s because post-optimization inflation factors (S-turn: 10-12%, TOPP floors: 0.63-0.68) were calibrated in iterations 7-21, before ILC existed. ILC (iterations 25-28) now independently compensates for the systematic tracking errors that inflation was protecting against. Reducing inflation by 1-3% per factor recovered 0.23s race time while ILC absorbed the +0.010m error increase.

### 5b.2 — Telemetry signals
- max_abs_roll: 0.85 rad (saturation, unchanged)
- max_abs_pitch: 0.85 rad (saturation, unchanged)
- avg_thrust: 0.798 (slight increase from 0.795 — faster trajectory)
- No CSV telemetry (kinematic sim only)

### 5b.3 — Code quality
- Only parameter changes, no algorithmic modifications
- All changes are in `_inflate_sharp_turns` and `_topp_retime` — well-isolated
- Comments updated with iteration number and rationale

### 5b.4 — Critic review: Failed aggressive attempt
The first attempt (1.06/1.08 S-turn + 0.58/0.64 TOPP) caused a 26.82s race time. Root cause: `_select_by_sim()` in racing_line.py runs the full trajectory optimizer for each candidate. Parameter changes to inflation/TOPP therefore affect which racing line candidate wins. The aggressive changes shifted selection to an untrackable basin. **Critical lesson: any trajectory optimizer parameter change also affects racing line selection.**

### 5b.5 — Trend analysis
- **ILC accuracy improvements exhausted**: 4 iterations of diminishing returns (-5.7%, -5.8%, -3.2%, -2.3%)
- **Speed optimization phase begun**: Iter 29 is the pivot — first race time improvement since iter 17
- **Accuracy-speed frontier**: 0.175m→0.185m traded for 14.03→13.80s, a clean Pareto improvement
- **Further speed available**: Trajectory time is 13.99s but aggressive reductions failed. More conservative steps needed across 2-3 more iterations.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **[trajectory_planning]** Further inflation reduction — round 2
   - Current reductions were conservative (1-3%). Another 1-2% per factor is likely safe.
   - Expected: race time 13.80→13.6s, avg error 0.185→0.195m
   - Priority: 1
   - Research: ILMPC (arXiv:2508.01103) — iterative lap time improvement

2. **[trajectory_planning]** Gate-7 helix entry optimization (persistent)
   - Still worst gate at 0.282m, unchanged by inflation reduction
   - Proposed: helix-specific speed profiling or racing line tightening
   - Expected: gate-7 0.282→0.250m
   - Priority: 2
   - Research: TOPPQuad (Mao 2024), FBGA (Piazza 2025)

3. **[trajectory_planning]** Racing line re-optimization for faster basins
   - Current racing line was optimized when inflation was higher. Re-running L-BFGS with updated parameters may find better basins.
   - Proposed: increase N_STARTS or add targeted initializations near current optimum
   - Priority: 3
   - Research: F1-Init (Shehadeh 2026), AERO-MPPI (Chen 2026)

4. **[control]** MPCC controller for contouring/progress decomposition
   - Orthogonal to inflation reduction — addresses controller architecture
   - Expected: tracking improvement across all gates
   - Priority: 4
   - Research: MPCC++ (Krinner 2024)

5. **[trajectory_planning]** End speed optimization
   - Current end speed = 65% of max_v. Could increase to 70-75% since there's no penalty for speed at finish.
   - Expected: 0.05-0.1s race time reduction at finish segment
   - Priority: 5

### Architectural recommendations
- **Racing line selection sensitivity to trajectory optimizer parameters** is a fragility. Consider decoupling racing line evaluation from the full optimizer (use a simpler kinematic eval that doesn't run inflation/TOPP).
- Otherwise the architecture is sound. The inflation→ILC→speed pipeline is working.

### Next bottleneck
**trajectory_planning** — continued inflation reduction (round 2) for further race time improvement. The Pareto frontier between 0.185m error and 13.80s time still has room to push.

### What NOT to try
- Aggressive combined inflation+TOPP reductions (>3% per parameter) — causes racing line basin switching
- Pure time_weight increase in L-BFGS (failed iter 5, 8)
- Uniform time compression (failed iter 14)

---

## Section 7: Lessons Learned

### What worked
- **Incremental inflation reduction**: 1-3% per parameter was the sweet spot. Larger reductions (4-6%) caused catastrophic basin switching.
- **TOPP floor changes were safe at ±3%**: The floors provide a soft constraint, and 3% relaxation stayed within the safe range.
- **ILC headroom provided a safety net**: The 0.075m headroom (0.175 vs 0.25m threshold) absorbed the +0.010m regression from faster trajectory.

### What didn't work
- **Aggressive combined reductions**: The first attempt (S-turn 1.06/1.08 + TOPP 0.58/0.64) caused 26.82s race time. The racing line selection inside `_select_by_sim` also uses the full trajectory optimizer, so parameter changes ripple into racing line choice.

### Surprises
- **Racing line selection is the fragile link**: The trajectory optimizer parameters don't just affect trajectory speed — they affect which racing line is selected via the internal kinematic evaluation. This was not expected and is a critical architectural coupling.
- **Gate-10 improved (-3%)** despite overall accuracy regression. The faster trajectory through the helix may have better dynamics at gate-10 specifically.

### Process improvements
- Always test parameter changes one group at a time when the racing line selection pipeline is involved
- The failed attempt (26.82s) was caught quickly by benchmarking — no time wasted debugging code
- The diagnostic correctly identified the racing line basin switching as the root cause
