# Iteration 33 — Racing Line Offset Caching for Determinism

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (racing line optimizer non-determinism)
**Status**: COMMITTED — determinism achieved, no metric changes
**Commit**: b477cf9

---

## Section 1: Summary
- Iteration 33, timestamp 2026-04-14T23:43Z
- Bottleneck: trajectory_planning — racing line optimizer non-determinism (basin switching)
- One-line outcome: **Added racing line offset caching. Optimized offsets saved to JSON on first run, loaded on subsequent runs. Triple-verified bit-exact determinism. No metric regression. Unblocks all future parameter tuning.**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Rethinking Reference Trajectories in Agile Drone Racing: MPPI** (2025)
   - URL: https://arxiv.org/abs/2509.14726
   - Proposes reference-free MPPI controller, removing dependency on pre-computed trajectories
   - Relevant: demonstrates offline vs online trajectory tradeoffs

2. **Efficient Trajectory Optimization via F1 Data-Driven Initialization** (Shehadeh 2026)
   - URL: https://arxiv.org/abs/2603.07126
   - Neural network predicts racing line from track geometry as warm-start
   - Relevant: initialization sensitivity in multi-modal landscapes

3. **Sampling-Based Motion Planning with Online Racing Line Generation** (Ogretmen 2024)
   - URL: https://arxiv.org/abs/2403.18643
   - Online racing line generation for multi-vehicle racing
   - Relevant: offline vs online racing line computation tradeoffs

### Key insight from research synthesis
Competition-winning drone racing systems universally pre-compute their trajectories offline and use them as fixed references (Romero 2025, Qin 2024, Foehn 2021). Online re-optimization is only for reacting to unexpected obstacles. Our `_select_by_sim()` re-computing the racing line on every benchmark run was architecturally wrong.

### Research consensus
- **Strong**: Pre-computed racing lines are standard practice in competition systems
- **Strong**: Initialization sensitivity causes convergence to different basins (Shehadeh 2026, Wachter 2026)
- **Strong**: Caching/freezing optimization results is the industry standard pattern

---

## Section 3: Implementation

### Changes made
- **File**: `planning/racing_line.py`
  - Added `use_cache: bool = True` to `RacingLineConfig`
  - Added `_compute_cache_key()`: SHA-256 hash of gate positions + config for cache invalidation
  - Added `_load_cache()` / `_save_cache()`: JSON serialization of winning offsets
  - Modified `optimize()`: checks cache before running optimization; saves cache after optimization
  - Cache file: `planning/racing_line_cache.json`

### Algorithm change
```
Before: Every benchmark run → multi-start L-BFGS (10 starts) → _select_by_sim (13 candidates) → may select different basin
After:  First run → same optimization → save offsets to cache
        Subsequent runs → load cached offsets (skip optimization entirely) → deterministic
```

### Plan adherence
Followed plan exactly. No deviations.

---

## Section 4: Benchmark Comparison

### Metrics: Before vs After
| Metric | Before (baseline) | After (with cache) | Delta |
|--------|-------------------|-------------------|-------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | = |
| Gates passed | 12/12 (100%) | 12/12 (100%) | = |
| Avg tracking error | 0.18528m | 0.18528m | = |
| Max tracking error | 0.7422m | 0.7422m | = |
| P50 tracking error | 0.1775m | 0.1775m | = |
| P95 tracking error | 0.3823m | 0.3823m | = |
| EKF uncertainty | 0.0119m | 0.0119m | = |
| Race time | 13.79s | 13.79s | = |
| Deterministic | NO | **YES** | **Fixed** |

### Per-gate error (identical across 3 runs)
| Gate | Error (m) |
|------|-----------|
| gate-1 | 0.111 |
| gate-2 | 0.179 |
| gate-3 | 0.247 |
| gate-4 | 0.202 |
| gate-5 | 0.180 |
| gate-6 | 0.162 |
| gate-7 | **0.284** (worst) |
| gate-8 | 0.235 |
| gate-9 | 0.180 |
| gate-10 | 0.155 |
| gate-11 | 0.163 |
| gate-12 | 0.137 |

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
The racing line optimizer's `_select_by_sim()` was evaluating the full trajectory optimizer for each of 13 candidates on every run. The trajectory optimizer's L-BFGS had near-equal-energy basins at current parameters, causing non-deterministic selection. This blocked ALL further parameter tuning since iterations 29-32. The fix is architectural: cache the winning offsets and reuse them deterministically.

### 5b.2 — Telemetry signals
- max_abs_roll: 0.85 rad (saturating — PD controller at physical limits)
- max_abs_pitch: 0.85 rad (saturating)
- avg_thrust: 0.797
- avg_pitch: -0.110 rad (slight forward lean, normal for racing)

### 5b.3 — Trend analysis
- **Stagnation BROKEN**: Iters 29-32 were stuck on racing line non-determinism. This iteration resolves the architectural issue.
- **Diminishing returns on trajectory_planning parameter tuning**: The easy wins (inflation reduction, per-section ILC, helix curvature treatment) have been exhausted.
- **Controller saturation**: Roll/pitch at 0.85 rad limit suggests the PD controller is at its physical limits. A better controller (MPCC, geometric SE(3)) could unlock more performance.
- **Next improvement frontier**: Either improve the racing line offsets directly (now possible with frozen cache) or improve the controller.

### 5b.4 — Architectural issues (resolved and remaining)
- **RESOLVED**: Racing line non-determinism — fixed by caching
- **REMAINING**: PD controller saturating at 0.85 rad — needs geometric or MPCC controller
- **REMAINING**: Helix gates (7-8) have trajectory-shape-driven error resistant to inflation
- **REMAINING**: S-turn gate-3 has high approach speed

### 5b.5 — Critic review
The caching implementation is clean:
- Cache key includes all relevant inputs (gate positions, config params)
- Auto-invalidation on track/config changes
- Non-fatal cache write failures (graceful degradation)
- Human-readable JSON format for debugging
- `use_cache=False` bypass for forced re-optimization

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **[trajectory_planning]** Manual racing line offset tuning for helix gates
   - With determinism achieved, manually edit `racing_line_cache.json` to test different offsets for gate-7/gate-8
   - Expected: gate-7 0.284→0.250m, gate-8 0.235→0.210m
   - Priority: 1
   - Research: TOGT (Qin 2024) — gate regions, not points

2. **[control]** MPCC or SE(3) geometric controller
   - PD controller saturating at 0.85 rad limits performance
   - MPCC decomposes contouring/progress, SE(3) handles attitude dynamics
   - Expected: 5-15% tracking improvement across all gates
   - Priority: 2
   - Research: MPCC++ (Krinner 2024), Lee SE(3) controller

3. **[trajectory_planning]** Gate-3 S-turn approach deceleration
   - Gate-3 at 0.247m — high approach speed into 48.2° turn
   - Targeted inflation or ILC adjustment for S-turn section
   - Expected: gate-3 0.247→0.220m
   - Priority: 3
   - Research: CiMPCC (Li 2025), segment-based ILC (Zhang 2024)

4. **[trajectory_planning]** Time-optimal segment compression with verified determinism
   - Now that racing line is frozen, can safely reduce TOPP compression floors
   - Expected: 0.1-0.2s race time improvement
   - Priority: 4
   - Research: TOPPQuad (Mao 2024), FBGA (Piazza 2025)

5. **[system_integration]** Port to PyBullet sim for realistic physics validation
   - Kinematic sim lacks attitude dynamics, may not reflect real performance
   - Expected: better correlation with competition performance
   - Priority: 5

### Next bottleneck
**trajectory_planning** — Manual racing line offset tuning for helix gates (gate-7: 0.284m, gate-8: 0.235m). Now that caching provides determinism, we can directly edit offsets without triggering basin switching.

### What NOT to try
- More inflation parameter tuning without changing the cache — still at basin boundaries
- Any racing line optimizer changes that bypass the cache — would reintroduce non-determinism
- Gain scheduling in kinematic sim — no attitude dynamics (failed iter 12)

---

## Section 7: Lessons Learned

### What worked
- **Racing line caching** completely eliminates the non-determinism problem that blocked iters 29-32
- **Simple architectural fix** over complex algorithmic solution — caching is simpler and more robust than multi-seed voting or geometric proxy improvements
- **Triple verification** confirms bit-exact determinism

### What didn't work
- N/A — this iteration succeeded

### Surprises
- **Cache file is only 24 floats** — the entire racing line optimization result is tiny (< 1KB JSON)
- **The optimization pipeline was deterministic** within a single process invocation (same results every time when cache is used). The non-determinism was likely from scipy's L-BFGS-B internal state across different module load orders or BLAS threading.
- **13 candidates evaluated** (10 L-BFGS + 3 basin-bridging interpolations) — significant computation saved on cached runs

### Process improvements
- Future iterations should verify determinism by running benchmark 2-3x before committing
- Cache-based approach makes it possible to do controlled experiments: change one parameter, verify cache invalidation, run benchmark
- Consider adding cache warmup script for new tracks
