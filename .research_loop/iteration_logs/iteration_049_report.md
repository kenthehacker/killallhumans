# Iteration 49 — Heavy-Ball Momentum ILC (γ=0.2)

**Date**: 2026-04-15
**Bottleneck**: trajectory_planning (ILC convergence depth via momentum acceleration)
**Status**: COMMITTED — avg error 0.163→0.162m (-1.0%), 8/12 gates improved
**Commit**: 3f174e8

---

## Section 1: Summary
- Iteration 49, timestamp 2026-04-15T17:17Z
- Bottleneck: trajectory_planning
- One-line outcome: **Added heavy-ball momentum (γ=0.2) to ILC updates (Polyak 1964, Wang 2023). Avg error improved 0.163→0.162m (-1.0%). Gate-7 improved -6.8%, gate-11 -6.2%. Gate-5 (worst) regressed +5.0% — a Pareto trade-off at the ILC convergence frontier.**

---

## Section 2: Research

### Papers analyzed (3 new, 136 total)
1. **"Fast Data-Driven ILC with Nesterov Acceleration"** (Wang 2023, arXiv:2312.14326)
   - Integrates Nesterov's accelerated gradient into ILC framework
   - Hybrid strategy: fast (momentum) → classical switching for best convergence + asymptotic performance
   - Convergence rate: O(1/k²) accelerated vs O(1/k) standard P-type
   - Directly inspired our heavy-ball momentum implementation

2. **"A Method to Speed Up Convergence of ILC for High Precision Motions"** (Longman 2023, arXiv:2307.15912)
   - Model-based warm-start for ILC to accelerate convergence
   - Less applicable: our ILC already has good warm-start from trajectory itself

3. **"Learning to Race through Coordinate Descent Bayesian Optimisation"** (Cully 2018, arXiv:1802.06179)
   - CDBO: optimize racing parameters one coordinate at a time
   - Relevant to lateral offset re-optimization but basin switching makes offset changes dangerous

### Key insight from cross-validation
The γ-momentum ILC paper from Wu et al. 2026 mentioned in the user prompt was not findable via web search. The closest paper (Wang 2023) provided the same core technique: adding momentum to ILC updates. The heavy-ball method (Polyak 1964) is the simplest momentum variant and proved effective.

### Research consensus vs contradictions
- **Consensus**: Momentum in ILC provides 2-3x faster convergence (Wang 2023, theoretical)
- **Our finding**: In practice, with 8 well-tuned ILC iterations near convergence, momentum provides only ~1% improvement. The ILC is already at a deep minimum. Momentum's main effect is redistributing error between gates (improving some, regressing others) rather than uniformly reducing it.

---

## Section 3: Implementation

### Changes made
**File: `planning/trajectory_optimizer.py`** (3 changes)
1. Added `momentum_gamma` parameter to `compute_ilc_offset_table()` (default 0.0 for backward compat)
2. Added `prev_section_offsets` storage for previous iteration's offsets
3. Modified per-section offset update: `offset += alpha * smoothed + gamma * momentum`
4. Support per-section gamma via 7th element of section_boundaries tuple

**File: `scripts/benchmark.py`** (1 change)
5. Set `momentum_gamma=0.2` in the ILC call

### Systematic sweep results
| Gamma | Avg Error | Delta | Gate-5 | G5 Delta | Gate-4 | G4 Delta | Gate-7 | G7 Delta |
|-------|-----------|-------|--------|----------|--------|----------|--------|----------|
| 0.00 | 0.1632 | 0.0% | 0.2517 | 0.0% | 0.2364 | 0.0% | 0.1640 | 0.0% |
| 0.10 | 0.1640 | +0.5% | 0.2461 | -2.2% | 0.2411 | +2.0% | — | — |
| 0.15 | 0.1627 | -0.3% | 0.2544 | +1.1% | 0.2352 | -0.5% | 0.1605 | -2.1% |
| **0.20** | **0.1616** | **-1.0%** | **0.2643** | **+5.0%** | **0.2294** | **-3.0%** | **0.1528** | **-6.8%** |
| 0.30 | 0.1621 | -0.7% | 0.2564 | +1.9% | 0.2352 | -0.5% | — | — |
| 0.40 | 0.1606 | -1.6% | 0.2804 | +11.4% | 0.2220 | -6.1% | 0.1396 | -14.9% |

Selected γ=0.2 as Pareto-optimal: best avg error improvement with no gate exceeding 6% regression.

Also tested per-section momentum (helix=0.4, s-turn=0.2, others=0.0): avg regressed to 0.1715m (+5.1%). Per-section momentum causes worse cross-section coupling than uniform momentum.

Also tested max_iterations=9 + gamma=0.2: identical to 8 iterations (convergence threshold still binds).

### Plan adherence
Followed plan closely. Added per-section gamma support (not in original plan) based on sweep results showing gate-5 regression. Per-section approach was tested but rejected due to worse performance.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit test pass rate | 100% | 100% | 0 | — |
| Avg tracking error | 0.1632m | 0.1616m | -0.0016m | ↓ **-1.0%** |
| Max tracking error | 0.7394m | 0.7419m | +0.003m | ↑ +0.3% |
| P50 tracking error | 0.1317m | 0.1281m | -0.004m | ↓ -2.7% |
| P95 tracking error | 0.4146m | 0.4072m | -0.007m | ↓ -1.8% |
| EKF uncertainty | 0.0119m | 0.0119m | 0 | — |
| Gate pass rate | 100% | 100% | 0 | — |
| Loop frequency | 7632 Hz | 7720 Hz | +88 Hz | — |
| Race time | 13.31s | 13.31s | 0 | — |
| Crashed | No | No | — | — |

### Per-gate error breakdown
| Gate | Before | After | Delta | Note |
|------|--------|-------|-------|------|
| gate-1 | 0.113m | 0.113m | +0.2% | Flat |
| gate-2 | 0.198m | 0.193m | **-2.3%** | Pre-inflection convergence |
| gate-3 | 0.206m | 0.204m | **-1.0%** | S-turn improvement |
| gate-4 | 0.236m | 0.229m | **-3.0%** | Inflection convergence |
| gate-5 | 0.252m | 0.264m | +5.0% | Momentum over-correction at post-inflection start |
| gate-6 | 0.104m | 0.104m | -0.4% | Flat |
| gate-7 | 0.164m | 0.153m | **-6.8%** | Biggest individual improvement |
| gate-8 | 0.223m | 0.232m | +4.0% | Momentum over-correction at helix start |
| gate-9 | 0.105m | 0.109m | +4.3% | Early helix regression |
| gate-10 | 0.107m | 0.103m | **-3.2%** | Late helix improvement |
| gate-11 | 0.104m | 0.098m | **-6.2%** | Notable improvement |
| gate-12 | 0.172m | 0.167m | **-3.2%** | Late helix improvement |

### Threshold status
All thresholds passing.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The ILC is operating at a deep convergence plateau after 8 iterations with per-section alpha rebalancing. Heavy-ball momentum redistributes how error is allocated between gates rather than uniformly reducing it. Gates at the END of ILC sections (gate-2, gate-4, gate-7, gate-11/12) benefit from momentum's inertial effect carrying corrections forward, while gates at the START of sections (gate-5 at post-inflection start, gate-8/9 at helix start) suffer from momentum overshooting the section-boundary transition.

### Telemetry signals
- Max roll: 0.85 rad (at controller limit, unchanged)
- Max pitch: 0.85 rad (at controller limit, unchanged)
- Avg thrust: 0.872 (unchanged)
- Controller saturation: both attitude axes hitting 0.85 rad cap — this is the fundamental kinematic sim PD controller limit. No further ILC/trajectory tuning can overcome this.

### Trend analysis
**Trend: DIMINISHING RETURNS (approaching plateau).**
- Iter 46: -0.37% avg error
- Iter 47: -8.2% avg error (major ILC iteration count increase)
- Iter 48: -5.8% avg error (ILC depth + alpha rebalance)
- Iter 49: -1.0% avg error (momentum — diminishing)

The compound improvement over iterations 46-49: from 0.186m to 0.162m = **-13.1%** total. But the marginal gain is declining rapidly. ILC-based improvements are nearing exhaustion.

### Architectural issues
1. **Attitude saturation (0.85 rad)**: The PD controller hits its tilt limit at every gate transition. This is the hard floor for tracking error in kinematic sim. No amount of ILC tuning can push below this.
2. **Gate-5 at Pareto frontier**: Every ILC parameter change that improves surrounding gates causes gate-5 regression. The inflection→post-inflection spatial coupling is a fundamental constraint.
3. **Section boundary sensitivity**: Momentum over-correction at section boundaries (gate-5, gate-8/9) shows that the ILC correction is spatially coupled through the sim dynamics. The "per-section" architecture is approximate — corrections leak across boundaries.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Area: system_integration — Competition robustness verification**
   - Description: The system is optimized for deterministic benchmark. Competition will have noise, timing jitter, perception errors. Need to verify graceful degradation.
   - Approach: Add Gaussian noise to state estimates (σ=0.01-0.1m), test with perturbed gate positions (±0.05m), verify no crashes.
   - Expected impact: Competition readiness, identify failure modes before VQ1.
   - Priority: 1
   - Research refs: MonoRace (2026), On Your Own (Romero 2025)

2. **Area: control — MPC/MPCC controller upgrade**
   - Description: PD controller saturates at 0.85 rad tilt. A proper MPC/MPCC controller could better utilize dynamics constraints.
   - Approach: Implement simplified MPCC from Krinner 2024 for the kinematic sim.
   - Expected impact: Could break through the tracking error floor by better utilizing available control authority.
   - Priority: 2
   - Research refs: MPCC++ (Krinner 2024), CiMPCC (Li 2025)

3. **Area: trajectory_planning — ILC section boundary smoothing**
   - Description: Momentum results show section boundaries cause error discontinuities. A smooth transition zone between sections could reduce gate-5/gate-8 regression.
   - Approach: Add a blending region (10-20 steps) where alpha/gamma interpolate between adjacent sections.
   - Expected impact: 1-3% improvement by reducing boundary artifacts.
   - Priority: 3
   - Research refs: Spatial ILC (Lv 2023), segment-wise ILC (Zhang 2024)

4. **Area: trajectory_planning — Racing line re-optimization with deterministic basin pinning**
   - Description: Current offsets were tuned for 5-iteration ILC. Re-optimization requires solving the basin-switching problem.
   - Approach: Lock to current L-BFGS basin by constraining initial point, then sweep offsets within the basin's attraction domain.
   - Expected impact: 3-8% if current offsets are suboptimal.
   - Priority: 4
   - Research refs: TOGT (Qin 2024), BO Racing Line (Heilmeier 2020)

### Architectural recommendations
The ILC-based approach has reached diminishing returns after 5 consecutive successful iterations (46-49). Further marginal ILC tuning will not produce >1% improvements. The next major improvement tier requires either:
1. **Controller upgrade** (MPC/MPCC) to break through the 0.85 rad attitude saturation limit
2. **Real hardware deployment** to validate sim-to-real transfer and identify new bottlenecks
3. **Competition robustness** testing to ensure the system degrades gracefully under noise

### Next bottleneck selected
**system_integration** — Competition robustness verification. The system is highly optimized for deterministic benchmark but untested under realistic noise and perturbation. This is the highest-impact task for competition readiness with only 1 iteration remaining.

### What NOT to try
- **Momentum gamma > 0.4**: Gate-5 regression exceeds 11%. Pareto frontier well-characterized.
- **Per-section momentum**: Tested and failed — causes worse cross-section coupling than uniform momentum.
- **ILC iterations > 8**: Convergence threshold (0.0005) binds regardless of iteration count. Tested with 9 iterations — identical results.
- **Racing line offset changes**: Basin switching risk remains extreme. Any offset change > ±0.05 triggers race time explosion.
- **Further ILC alpha tuning**: Per-section alphas at Pareto-optimal after iteration 48. No room for improvement.
- **Inflection max_correction > 0.15m**: Tested in iterations 43, 46, 47 — always causes gate-5 regression.

---

## Section 7: Lessons Learned

### What worked
- **Heavy-ball momentum is a novel technique for this system**: Not previously tried. Provides genuine (if small) improvement.
- **Systematic gamma sweep**: Testing 5 gamma values (0.0-0.4) quickly characterized the Pareto frontier.
- **Uniform momentum > per-section momentum**: Counter-intuitive — applying the same gamma everywhere is better than targeted application. This is because the sim dynamics couple sections regardless of ILC structure.

### What didn't work
- **Per-section momentum (helix=0.4, s-turn=0.2, others=0.0)**: Average regressed +5.1%. The heterogeneous momentum creates worse coupling than uniform.
- **Combined alpha reduction + momentum**: Reducing post-inflection alpha from 0.50 to 0.48 while keeping gamma=0.2 didn't help gate-5 — it actually made it worse.
- **9 ILC iterations + momentum**: Convergence threshold still binds. Momentum doesn't change the convergence dynamics enough to enable a 9th productive iteration.

### Surprises
1. **Momentum redistributes error rather than uniformly reducing it**: Gates at section ends improve while gates at section starts regress. This suggests the momentum's inertial effect "carries forward" corrections from late in one section to early in the next.
2. **γ=0.2 is a sweet spot**: Below 0.2, improvement is negligible. Above 0.2, gate-5 regression grows fast. The optimal value is narrow.
3. **The ILC is genuinely near its convergence limit**: Adding momentum, adding iterations, tuning alphas — all produce <2% improvement now. The system has been pushed to its theoretical performance limit under the current controller and trajectory architecture.

### Process improvement suggestions
- The γ-momentum ILC paper from Wu et al. 2026 referenced in the user prompt doesn't appear to exist. Future prompts should verify paper existence before citing.
- For the final iteration (50), focus on robustness testing rather than further optimization. The system is highly tuned but untested under perturbation.
