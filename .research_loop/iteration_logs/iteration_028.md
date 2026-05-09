# Iteration 28 — Per-Section Butterworth Bandwidth for Gate-3 Recovery

**Date**: 2026-04-14
**Bottleneck**: trajectory_planning (per-section ILC Q-filter bandwidth)
**Status**: COMMITTED — avg error 0.179→0.175m (-2.3%), gate-3 0.292→0.222m (-24.1%)
**Commit**: 3055e9d

---

## Section 1: Summary
- Iteration 28, timestamp 2026-04-14T17:13Z
- Bottleneck: trajectory_planning — gate-3 S-turn inflection regressed 37% in iter 27 from global Butterworth cutoff
- One-line outcome: **Time-varying Q-filter bandwidth (Bristow & Alleyne 2007): S-turn inflection section uses 0.40 Hz Butterworth cutoff while smooth sections keep 0.35 Hz. Gate-3 recovered from 0.292→0.222m (-24.1%), avg error improved 0.179→0.175m (-2.3%), new all-time best.**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Bristow & Alleyne 2007/2008** — "A Time-Varying Q-Filter Design for ILC" (ACC 2007), "Monotonic Convergence of ILC for Uncertain Systems Using a Time-Varying Filter" (IEEE TAC 2008), "Optimizing Learning Convergence Speed and Converged Error" (ASME JDSMC 2008)
   - **THE foundational paper for our approach**. Proves time-varying Q-filter bandwidth strictly dominates any single global bandwidth when error frequency content is heterogeneous.
   - STFT-based bandwidth selection: set cutoff at each section to the max frequency with error power above noise floor.
   - Optimal LTV filter converges 60% faster and achieves 40% lower error at direction reversals vs optimal LTI filter.

2. **Zhang, Meng & Tan 2026** — "Segment-Based Two-Loop Adaptive ILC for Spacecraft Position and Attitude Tracking" (arXiv:2602.14660)
   - Extends segment-wise ILC to simultaneous position + attitude tracking.
   - Bounded control inputs via segment-wise projection.
   - Validates segment-wise approach on a more complex dynamical system (spacecraft 6-DOF).

3. **Ewering et al. 2025** — "Dual ILC for MIMO Dynamics with Robotic Validation" (arXiv:2509.18723)
   - Data-driven dual ILC: simultaneous tracking + model learning, no prior model knowledge.
   - Monotonic convergence conditions for both reference tracking and model error.
   - Solves tasks in 10-20 trials on real-world MIMO systems.

### Research consensus
- **Strong**: Time-varying (section-varying) Q-filter bandwidth strictly dominates any single global bandwidth (Bristow 2007, proven theoretically).
- **Strong**: Per-section convergence guaranteed independently when each section's iteration operator contracts.
- **Strong**: Butterworth zero-phase is the correct filter family for ILC Q-filters.

### Key insight
Gate-3 improvement is nearly identical across cutoffs 0.40-0.60 Hz (all ~0.222m). The main benefit comes from having the per-section structure (not from the exact cutoff). Higher cutoffs just add noise that hurts neighboring gates (gate-4, gate-5). The optimal per-section cutoff (0.40 Hz) is only 14% above the global optimum (0.35 Hz) — a small but precisely targeted bandwidth increase.

---

## Section 3: Implementation

### Changes made

**File: `planning/trajectory_optimizer.py`**
- Extended `section_boundaries` tuple format to accept 5th element: `section_cutoff_hz`
- In per-section loop, designs a section-specific Butterworth filter when 5th element is present
- Falls back to global `filter_cutoff_hz` when not specified
- Backward compatible: existing 4-element tuples still work

**File: `scripts/benchmark.py`**
- Split 2-section layout into 4-section: pre-inflection, inflection, post-inflection, helix
- Inflection section (steps 200-440, ~2.0-4.4s) uses 0.40 Hz cutoff
- All other sections use 0.35 Hz cutoff (unchanged)

### Sweep results
| Cutoff | Avg Error | Gate-3 | Gate-4 | Gate-5 |
|--------|-----------|--------|--------|--------|
| baseline (global 0.35) | 0.17865 | 0.2919 | 0.1586 | 0.1316 |
| **0.40 Hz** | **0.17458** | **0.2216** | 0.1669 | 0.1442 |
| 0.42 Hz | 0.17500 | 0.2217 | 0.1693 | 0.1459 |
| 0.45 Hz | 0.17559 | 0.2219 | 0.1731 | 0.1479 |
| 0.50 Hz | 0.17640 | 0.2234 | 0.1791 | 0.1490 |
| 0.60 Hz | 0.17804 | 0.2242 | 0.1969 | 0.1460 |

### Plan adherence
Followed the plan exactly. The sweeping identified 0.40 Hz as optimal (plan predicted 0.50-0.75 Hz). The matched-bandwidth rule from Bristow & Alleyne applies: just enough bandwidth to capture the inflection error signal, no more.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before (iter 27) | After (iter 28) | Delta | Direction |
|--------|-------------------|-----------------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.17865m | **0.17458m** | **-0.004m (-2.3%)** | ✓ |
| Max tracking error | 0.7017m | 0.7045m | +0.003m (+0.4%) | → |
| P50 tracking error | 0.1766m | **0.1731m** | **-0.004m (-2.0%)** | ✓ |
| P95 tracking error | 0.3637m | 0.3637m | 0% | → |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7218 | 6914 | -4.2% | → (still >100) |
| Race time | 14.03s | 14.03s | 0% | → |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.111 | 0.111 | 0.0% | → |
| gate-2 | 0.171 | 0.170 | -0.8% | → |
| gate-3 | 0.292 | **0.222** | **-24.1%** | ✓✓ recovered! |
| gate-4 | 0.159 | 0.167 | +5.2% | ↑ mild |
| gate-5 | 0.132 | 0.144 | +9.6% | ↑ mild |
| gate-6 | 0.165 | 0.150 | -8.8% | ✓ |
| gate-7 | 0.273 | 0.276 | +1.1% | → |
| gate-8 | 0.187 | 0.188 | +0.4% | → |
| gate-9 | 0.183 | 0.184 | +0.4% | → |
| gate-10 | 0.184 | 0.184 | -0.0% | → |
| gate-11 | 0.171 | 0.171 | +0.0% | → |
| gate-12 | 0.129 | 0.129 | -0.0% | → |

---

## Section 5: Deep Diagnostic

### 5b.1 — Benchmark Decomposition
- **Best improvement**: gate-3 (-24.1%) — the targeted gate. Per-section bandwidth directly addressed the problem.
- **Mild regressions**: gate-4 (+5.2%), gate-5 (+9.6%). These are at the boundary of the inflection section (step 440 = 4.4s, gate-4 at 4.42s). The higher bandwidth section slightly disrupts corrections at the boundary.
- **Gate-6 improved**: -8.8%, likely because the post-inflection section at 0.35 Hz is now shorter and more focused.
- **Helix unchanged**: gates 7-12 are completely unchanged — their section uses the same 0.35 Hz as before.
- **New worst gate**: gate-7 (0.276m) replaces gate-3 (0.222m) as worst.

### 5b.2 — Telemetry Analysis
- max_abs_roll: 0.85 rad (saturation, unchanged)
- max_abs_pitch: 0.85 rad (saturation, unchanged)
- avg_thrust: 0.795 (unchanged)
- No telemetry CSV available (kinematic sim only)

### 5b.3 — Code Quality
- Per-section cutoff is backward-compatible (existing 4-element tuples still work)
- Butterworth filter is redesigned per-section in the loop — minor computational cost (~200μs per section)
- `scipy.signal.butter` import is inside the loop but Python caches module imports
- Clean separation: section_boundaries format extended, not changed

### 5b.5 — Trend Analysis
- **Improving but diminishing**: avg error 0.211→0.199→0.187→0.179→0.175 (deltas: -5.7%, -5.8%, -3.2%, -2.3%)
- **ILC approaching limit**: 4 consecutive ILC iterations with diminishing returns. The easy ILC gains are exhausted.
- **Gate-7 (helix) is persistent**: 0.259→0.273→0.276m across last 3 iterations. ILC changes do not affect this gate. Root cause is likely trajectory/controller limitation, not ILC filter design.
- **Roll/pitch at saturation (0.85 rad)**: PD controller physically limited. No ILC tuning can push past this.

### 5b.6 — Improvement Synthesis

**Root cause**: Gate-7 (helix entry, 0.276m) is the new bottleneck. It's persistent across ILC iterations because the error is NOT from filter design — it's from the helix trajectory itself requiring more centripetal acceleration than the PD controller can provide at saturation. Possible fix vectors:
1. **Slow down helix approach** (reduce speed at gates 6-7 transition)
2. **Helix-specific ILC bandwidth** (currently 0.35 Hz, may need adjustment)
3. **Racing line: tighter helix radius** (reduce centripetal demand)

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)
1. **[trajectory_planning]** Gate-7 helix entry optimization
   - Gate-7 (0.276m) is persistent worst gate, unchanged by ILC filter changes
   - Proposed: speed profiling adjustment near helix entry (gates 6-7 transition)
   - Expected: gate-7 0.276→0.240m from reduced approach speed
   - Priority: 1
   - Research: TOPP (Mao 2024), FBGA (Piazza 2025)

2. **[trajectory_planning]** Racing line fine-tuning near Pareto frontier
   - Current 0.175m avg error gives 0.075m headroom to 0.25m threshold
   - Trade some tracking accuracy for faster race time (14.03→13.9s target)
   - Priority: 2
   - Research: Quaypoints 2025, Heilmeier 2020

3. **[control]** MPCC controller for contouring/progress decomposition
   - Orthogonal to ILC — addresses controller architecture
   - Expected: tracking improvement across all gates, especially helix
   - Priority: 3
   - Research: MPCC++ (Krinner 2024)

4. **[trajectory_planning]** STFT-based adaptive bandwidth per ILC iteration
   - Currently fixed bandwidth per section across all 5 iterations
   - Bristow 2007 suggests updating bandwidth profile each iteration from residual STFT
   - Expected: marginal improvement, ~1% avg error reduction
   - Priority: 4

5. **[trajectory_planning]** GP-predictive ILC momentum term
   - Add e_hat = omega * e_i to ILC update for faster convergence
   - Expected: converge in 3 iterations instead of 5
   - Priority: 5
   - Research: Nigam 2026 (QPGP-ILC)

### Architectural recommendations
- None. ILC + per-section Butterworth is sound. Shift focus from ILC tuning to racing line / speed optimization for race time improvement.

### Next bottleneck
**trajectory_planning** — focus on gate-7 helix entry and racing line speed optimization. ILC filter design is largely solved.

### What NOT to try
- Global Butterworth cutoff changes (exhaustively swept in iter 27)
- Per-section convergence tracking (fails for S-turn, documented in iter 27)
- Extended inflection boundaries (>5s causes catastrophic regression)
- Inflection cutoff > 0.60 Hz (diminishing gate-3 returns, increasing gate-4 regression)

---

## Section 7: Lessons Learned

### What worked
- **Per-section bandwidth is the right abstraction**: The Bristow & Alleyne 2007 framework directly solved the problem. Gate-3 improved 24% while other gates were largely unchanged.
- **Systematic sweep with narrow range**: Testing {0.40, 0.42, 0.45, 0.50, 0.60} Hz found 0.40 Hz optimal. The initial plan predicted 0.50-0.75 Hz — the actual optimum was lower.
- **Minimal code change**: Only 2 files modified. The section_boundaries tuple extension is clean and backward-compatible.

### What didn't work
- **Extended inflection boundaries** (1.5-5.0s): Caused catastrophic regression (avg error 0.70m). The boundary locations matter more than the cutoff value.
- **Tight boundaries** (2.0-3.8s): Worse than wide (2.0-4.4s). Gate-4 benefits from being inside the higher-bandwidth section.

### Surprises
- **Gate-3 improvement is flat across cutoffs 0.40-0.60 Hz** (all ~0.222m). The benefit comes from having the per-section structure, not from the exact cutoff value. This means even a small bandwidth increase (0.35→0.40 Hz, just 14%) is enough to capture the inflection error content.
- **0.40 Hz is optimal, not 0.75 Hz as predicted**: The matched-bandwidth rule says "just enough bandwidth" — and the gate-3 inflection error content is apparently concentrated at 0.35-0.40 Hz, not 0.4-0.8 Hz as hypothesized.

### Process improvements
- Running the boundary sensitivity test (tight vs wide) early prevented committing a worse configuration
- The 5-point cutoff sweep was efficient and sufficient for this parameter
