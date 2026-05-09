# Iteration 32 — Gate-7 Offset Expansion + Inflation Reduction Round 3

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (gate-7 helix entry offset expansion)
**Status**: REVERTED — both approaches caused basin switching
**Commit**: e9f9f66 (revert)

---

## Section 1: Summary
- Iteration 32, timestamp 2026-04-14T18:50Z
- Bottleneck: trajectory_planning — gate-7 helix entry offset expansion (priority-1 from iter 31)
- One-line outcome: **Gate-7 offset expansion failed (4 variants, all basin-switched). Pivoted to inflation reduction round 3 — appeared to work in one benchmark run (13.72s, 0.189m) but was NOT reproducible. On re-verification, inflation reduction also caused basin switching (0.73m avg error). Both approaches REVERTED.**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Mollification-based Curvature Bounds for Trajectory Optimization** (2025)
   - Provides smooth curvature approximation for min-snap trajectories
   - Relevant: theoretical basis for offset-curvature relationship

2. **PMPC: Gate-Aware Racing with Perception-Driven MPC** (2024)
   - Perception-motivated path constraints through gates
   - Relevant: gate pass-through geometry analysis

### Key insight from cross-validation
**ENTRY_EXIT_OFFSET is a global geometry parameter that the racing line optimizer is exquisitely sensitive to.** Even +0.02m (from 0.40m to 0.42m) triggers racing line basin switching because `_select_by_sim()` evaluates the full trajectory optimizer internally. The racing line optimizer's L-BFGS finds different local minima when the waypoint geometry changes at ANY gate.

### Research consensus
- **Strong**: Offset expansion is theoretically sound for spreading curvature (TOGT, On Your Own, Richter 2016)
- **Strong**: The implementation approach is correct but blocked by optimizer coupling
- **New insight**: The racing line and trajectory optimizers are too tightly coupled for isolated geometry changes

---

## Section 3: Implementation

### Approach A: Adaptive Gate Offsets (FAILED — 4 variants)

**Variant 1**: All gates adaptive (0.4-0.8m based on turn angle)
- Result: Race time 27.65s — catastrophic basin switching
- Cause: Racing line `_select_by_sim()` evaluated with new geometry, found different local minimum

**Variant 2**: Helix-only offsets (gate-7 at 0.6m, others unchanged)
- Result: Race time 27.58s — same basin switching
- Cause: Even changing one gate's offset triggers racing line re-evaluation

**Variant 3**: Decoupled (racing line at 0.4m, trajectory at 0.6m)
- Result: 16.23s, 0.766m avg error — geometry mismatch
- Cause: Racing line optimized for 0.4m geometry but trajectory used 0.6m waypoints

**Variant 4**: Universal 0.42m (just +5% offset increase)
- Result: 15.84s — still basin switching
- Cause: Confirms that ANY offset change triggers racing line basin switching

### Approach B: Inflation Reduction Round 3 (FAILED — non-reproducible)

6 parameters reduced by ~1% each:
- S-turn junction: 1.09→1.08, standard: 1.07→1.06
- Approach decel: 1.01→1.005, departure: 1.02→1.01, junction depart: 1.005→1.003
- TOPP easy floor: 0.59→0.58 (protected 0.65 unchanged — at basin cliff)

Initial benchmark showed: 13.72s race time, 0.189m avg error — appeared to work.
**On re-verification after context restart: 0.73m avg error — catastrophic basin switching.**
The "good" result was a non-reproducible racing line realization.

### Changes committed: NONE (all reverted)

### Plan adherence
Attempted priority-1 (offset expansion) first, then pivoted to priority-2 (inflation reduction). Both failed.

### Failed attempts
See above — 5 total failed approaches (4 offset + 1 inflation reduction).

---

## Section 4: Benchmark Comparison

### Stable baseline (iter 31 code, verified reproducible)
| Metric | Value |
|--------|-------|
| Unit tests | 9/9 (100%) |
| Gates passed | 12/12 (100%) |
| Avg tracking error | 0.1854m |
| Max tracking error | 0.6939m |
| P50 tracking error | 0.1725m |
| P95 tracking error | 0.3894m |
| EKF uncertainty | 0.012m |
| Loop Hz | 6802 |
| Race time | 14.03s |

### Per-gate error (current reproducible baseline)
| Gate | Error | Note |
|------|-------|------|
| gate-1 | 0.111 | |
| gate-2 | 0.184 | |
| gate-3 | 0.270 | worst — persistent |
| gate-4 | 0.264 | 2nd worst |
| gate-5 | 0.162 | |
| gate-6 | 0.157 | |
| gate-7 | 0.259 | improved from report's 0.284! |
| gate-8 | 0.227 | |
| gate-9 | 0.154 | |
| gate-10 | 0.148 | |
| gate-11 | 0.153 | |
| gate-12 | 0.146 | |

**Important**: These numbers differ from the iter 31 report (gate-7: 0.259 vs 0.284, gate-3: 0.270 vs 0.247). The benchmark is deterministic across runs but the racing line optimizer settles into slightly different basins between code changes. The current basin has better gate-7 but worse gate-3/gate-4.

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
**Inflation reduction round 3 is at the absolute limit of the optimizer landscape.** The racing line `_select_by_sim()` creates a basin boundary that is extremely sensitive to:
1. Any geometry change (offset expansion) — triggers immediately
2. Cumulative small parameter changes — triggers non-deterministically (worked once, then didn't)

The non-reproducibility of the "good" result suggests the racing line optimizer has multiple near-equal basins at the current parameter values. Small perturbations (e.g., floating-point differences between runs, or numpy random seed state) can tip selection between basins.

### 5b.2 — Telemetry signals
- max_abs_roll: 0.85 rad (saturation, unchanged)
- max_abs_pitch: 0.85 rad (saturation, unchanged)
- avg_thrust: 0.7958
- avg_pitch: -0.1083

### 5b.3 — Trend analysis
- **Stagnation confirmed**: Inflation reduction (the reliable improvement lever from iters 29-31) has hit diminishing returns
- **Gate-7 improved to 0.259m**: Current racing line basin treats gate-7 better than previous reports
- **Gate-3/gate-4 are now worst**: 0.270m and 0.264m respectively
- **Race time regressed**: 14.03s vs iter 31's reported 13.79s (different racing line basin)
- **The optimizer landscape is becoming increasingly fragile**: small changes cause non-deterministic outcomes

### 5b.4 — Architectural issues
1. **Racing line / trajectory optimizer coupling**: `_select_by_sim()` makes the racing line extremely sensitive to any trajectory parameter change. This blocks many improvement paths.
2. **Non-determinism in racing line optimization**: Different code paths produce different L-BFGS starting conditions, leading to different basins. Need to either: (a) seed the optimizer, or (b) run multiple seeds and pick best.
3. **Gate-3 and gate-4 now worst gates**: The current racing line basin undertreats the S-turn approach section.
4. **Inflation parameters at physical limits**: All S-turn inflation values are within ~1% of basin boundaries.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **[trajectory_planning]** Racing line robustness — run optimizer with multiple seeds, pick deterministic best
   - Current non-determinism makes parameter tuning unreliable
   - Expected: consistent baseline for future iterations
   - Priority: 1
   - Research: F1-Init (Shehadeh 2026), Spatially-Aware CMA-ES (2026)

2. **[trajectory_planning]** Gate-3/gate-4 optimization — now worst at 0.270/0.264m
   - Gate-3: 48.2° turn after 11.7m straight (high approach speed)
   - Gate-4: consecutive turn, benefits from approach deceleration
   - Expected: 0.270→0.240m
   - Priority: 2
   - Research: CiMPCC (Li 2024), VPMPCC (Li 2024)

3. **[control]** MPCC controller for contouring/progress decomposition
   - Orthogonal to trajectory changes — won't trigger basin switching
   - Expected: tracking improvement across all gates
   - Priority: 3
   - Research: MPCC++ (Krinner 2024), NRHDG (Sung 2025)

4. **[trajectory_planning]** Gate-7 offset expansion via racing line refactoring
   - Decouple `_select_by_sim()` from trajectory optimizer parameters
   - Then safely increase offsets for helix entry
   - Priority: 4 (requires significant refactoring)

5. **[trajectory_planning]** ILC improvements for gate-3/gate-4 sections
   - Per-section bandwidth tuning for approach segments
   - Priority: 5
   - Research: segment-based ILC (Zhang 2024)

### Next bottleneck
**trajectory_planning** — Racing line optimizer robustness. The non-deterministic basin selection blocks reliable parameter tuning. Must fix this before further inflation/geometry changes.

### What NOT to try
- More inflation reduction — at basin cliff, non-deterministic results
- Any offset changes — racing line basin switching (confirmed with 5 variants)
- TOPP protected floor reduction (0.65→0.64) — at basin cliff (confirmed iter 30 + 32)
- Gain scheduling — failed in iter 12, no attitude dynamics in kinematic sim

---

## Section 7: Lessons Learned

### What worked
- Nothing — both approaches reverted

### What didn't work
- **Adaptive gate offsets (4 variants)**: ALL trigger racing line basin switching. The `_select_by_sim()` evaluates the full trajectory optimizer, making it sensitive to any geometry change.
- **Inflation reduction round 3**: Non-reproducible results. Appeared to work in one benchmark run but failed on re-verification. The optimizer landscape is fragile at current parameter values.
- **Decoupled approach**: Freezing racing line eval at 0.4m while using 0.6m for trajectory creates geometry mismatch — the racing line was optimized for different waypoint positions.

### Surprises
- **Non-reproducible benchmark**: The same code can produce dramatically different results between conversation sessions. This suggests the racing line optimizer has near-equal-energy basins at current parameters.
- **Per-gate numbers differ from iter 31 report**: gate-7 is actually 0.259m (better than 0.284m) in the current basin. The racing line has multiple near-optimal solutions with different per-gate distributions but similar averages.
- **+0.02m offset triggers basin switching**: The racing line is far more sensitive to geometry than to speed parameters.

### Process improvements
- **Always verify benchmark reproducibility** before committing. Run the benchmark at least twice from a clean state.
- **Context restarts can reveal non-determinism**: The "good" result from the previous conversation was not reproducible after the context switch, revealing latent non-determinism.
- **Consider seeding the racing line optimizer** for reproducible results across sessions.
