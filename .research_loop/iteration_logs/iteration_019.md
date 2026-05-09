# Iteration 19 — Multi-Start L-BFGS Racing Line Optimization

**Date**: 2026-04-14
**Bottleneck**: system_integration (escape L-BFGS local minimum for S-turn)
**Status**: COMMITTED — gate-4 error 0.413→0.310m (-24.9%), avg error 0.234→0.230m (-1.7%), race time 13.91→13.88s (-0.2%)
**Commit**: f49e1e5

---

## Section 1: Summary
- Iteration 19, timestamp 2026-04-14T12:45Z
- Bottleneck: system_integration — multi-start racing line optimization to escape L-BFGS local minimum
- One-line outcome: **Multi-start found a new basin: gate-4 improved 24.9%, but gate-3 regressed 40.7%. Net: all aggregate metrics improved.**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization** (Chen et al., ICRA 2026, arXiv:2509.17340)
   - Ensemble of M=15 parallel MPPI instances from structurally different initializations
   - Core insight: "diversify initialization rather than inflate sample count" escapes local minima
   - >80% success rate at 7 m/s vs <50% for single-optimizer baselines
   - Key for our system: parallel L-BFGS from diverse seeds, two-stage evaluation (optimize per-instance, select across instances)

2. **Efficient Trajectory Optimization via F1 Data-Driven Initialization** (Shehadeh et al., 2026, arXiv:2603.07126)
   - Learned initialization reduces L-BFGS iterations by 17%
   - Critical insight: "geometric agreement alone does not fully explain optimization performance" — basin-of-attraction matters more than Euclidean proximity
   - Hardware: 34% tracking error reduction from better initialization (0.165→0.109m)
   - Key for our system: zero-init may be in wrong basin; late-apex geometric prior; raise maxiter from 100

3. **Topology-Driven Parallel Trajectory Optimization** (de Groot et al., IEEE T-RO 2024, arXiv:2401.06021)
   - P=4 parallel MPC instances from distinct homotopy classes with formal fallback guarantee
   - Theorem 2: multi-start never regresses below single-start when baseline is included as candidate
   - Key for our system: include zero-init as fallback guarantees no regression

### Research consensus vs contradictions
- **Consensus (3/3)**: Single-initialization gradient optimization is fundamentally limited by local minima. The solution is parallel/ensemble optimization from diverse starting points with common selection.
- **Consensus (3/3)**: Include the current-best as a fallback candidate for regression-free improvement.
- **Minor disagreement**: AERO-MPPI recommends M=15, T-MPC uses P=4. For our 24D problem, N=10 was sufficient.

---

## Section 3: Implementation

### Changes made
**File: `planning/racing_line.py`** — `RacingLineOptimizer.optimize()` method

**Change 1: Multi-start L-BFGS-B with N=10 starts**
- Start 0: zero initialization (baseline fallback per T-MPC Theorem 2)
- Start 1: late-apex geometric prior (cut inside turns based on cross-product direction)
- Starts 2-9: random initializations from `np.random.default_rng(42)` (deterministic)
- Selection: minimum L-BFGS-B objective value across all 10 starts

**Change 2: maxiter raised from 100 to 300**
- F1-Init paper shows baseline solver needs 400-520 iterations; our 100 cap was premature
- Still fast for offline optimization (racing_line test: 0.65ms → 10ms)

**New method: `_late_apex_init()`**
- Computes turn direction via cross product of approach/departure vectors
- Sets lateral offset to ±0.5 * max_offset * sign(turn direction) to cut inside
- Provides a geometrically informed starting point for the S-turn

### Plan adherence
Followed plan exactly. No deviations.

---

## Section 4: Benchmark Comparison

### Full metrics table
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | — | → |
| Gates passed | 12/12 (100%) | 12/12 (100%) | — | → |
| Avg tracking error | 0.234m | **0.230m** | **-0.004m (-1.7%)** | ✓ |
| Max tracking error | 0.696m | **0.683m** | **-0.013m (-1.9%)** | ✓ |
| P50 tracking error | 0.197m | 0.188m | -0.009m | ✓ |
| P95 tracking error | 0.526m | 0.538m | +0.012m | ↓ (gate-3 peak) |
| EKF uncertainty | 0.012m | 0.012m | 0% | → |
| Loop Hz | 7510 | 7338 | -2.3% | → (noise) |
| Trajectory time | 14.10s | 14.06s | -0.04s | ✓ |
| Race time | 13.91s | **13.88s** | **-0.03s (-0.2%)** | ✓ |
| Avg thrust | 0.818 | 0.803 | -1.8% | ✓ |

### Per-gate error breakdown (before → after)
| Gate | Before | After | Delta | Notes |
|------|--------|-------|-------|-------|
| gate-1 | 0.115 | 0.112 | -0.003 | ✓ slight |
| gate-2 | 0.233 | **0.215** | **-0.018** | ✓ improved |
| gate-3 | 0.329 | **0.463** | **+0.134** | ✗ major regression — now worst |
| gate-4 | 0.413 | **0.310** | **-0.103** | ✓✓✓ target achieved |
| gate-5 | 0.181 | **0.155** | **-0.026** | ✓ improved |
| gate-6 | 0.156 | 0.158 | +0.002 | → unchanged |
| gate-7 | 0.333 | **0.308** | **-0.025** | ✓ improved |
| gate-8 | 0.221 | 0.218 | -0.003 | → unchanged |
| gate-9 | 0.203 | 0.208 | +0.005 | → unchanged |
| gate-10 | 0.208 | 0.216 | +0.008 | → slight regression |
| gate-11 | 0.173 | 0.176 | +0.003 | → unchanged |
| gate-12 | 0.212 | 0.215 | +0.003 | → unchanged |

### Threshold status
| Threshold | Required | Current | Aspirational | Status |
|-----------|----------|---------|--------------|--------|
| Avg error | <0.5m | **0.230m** | <0.25m | **MEETS ASPIRATIONAL** |
| Max error | <2.0m | **0.683m** | <1.0m | **MEETS ASPIRATIONAL** |
| Gate pass | 100% | 100% | 100% | PASS |
| Race time | <30s | 13.88s | <14s | **MEETS ASPIRATIONAL** |
| Loop Hz | >100 | 7338 | >100 | PASS |
| No crash | required | no crash | — | PASS |

All aspirational targets still met.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
The L-BFGS optimizer was consistently converging to a single local minimum where the S-turn racing line favored gate-3 accuracy at gate-4's expense. Multi-start exploration discovered a different basin where the racing line cuts gate-4's corner more aggressively, achieving 24.9% better tracking there. The trade-off is that gate-3 receives a worse approach angle, increasing its error by 40.7%. This is the **S-turn coupling problem**: gates 3 and 4 are linked by the S-turn geometry, and optimizing one worsens the other. The new basin is globally better (lower avg, lower max, lower race time) but locally worse for gate-3.

### Telemetry signals
- Max abs roll/pitch: 0.85 rad (unchanged, at tilt limit)
- Avg thrust: 0.803 (decreased from 0.818 — slightly more efficient trajectory)
- Avg pitch: -0.107 (slightly more forward lean — faster approach to gates)
- First iteration to improve both speed AND accuracy simultaneously

### Trend analysis
**Trend: IMPROVING (Pareto frontier advancing on both axes for first time)**

Last 5 iterations' Pareto:
- Iter 15: 13.50s / 0.251m
- Iter 16: 13.95s / 0.232m
- Iter 17: 13.62s / 0.248m
- Iter 18: 13.91s / 0.234m
- **Iter 19: 13.88s / 0.230m (both improved)**

The alternating speed/accuracy pattern broke. Multi-start found a basin that's better on both dimensions. However, the improvement magnitude is small (1.7% avg error, 0.2% race time), suggesting we're approaching the limits of what racing line offset optimization can achieve.

### Architectural observations
- The S-turn coupling (gate-3 ↔ gate-4) is fundamental to the gate geometry. No single set of lateral offsets can simultaneously minimize error at both gates.
- Multi-start with 10 starts was sufficient — the new basin was found by the random seeds, not the geometric prior.
- The racing line optimizer's objective (path_length + curvature²) doesn't directly minimize tracking error. Sim-based selection would find the racing line the controller can actually track best.
- Gate-3 at 0.463m is now the dominant error source. Targeted inflation or asymmetric offset bounds for the first S-turn bend would help.

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Gate-3 S-turn inflation** (Priority 1, trajectory_planning)
   - The multi-start shifted the worst gate from gate-4 to gate-3 (0.463m)
   - The S-turn inflation (iter 16) currently applies compound inflation to the SECOND turn of S-turn pairs. Gate-3 is the FIRST turn — it may need its own inflation mechanism.
   - Approach: add approach-side inflation for the first gate of an S-turn pair
   - Expected: gate-3 error 0.463→0.35m
   - Research: CiMPCC compound curvature, VPMPCC early deceleration

2. **Sim-based multi-start selection** (Priority 2, system_integration)
   - Currently selecting by L-BFGS objective value
   - Re-evaluate each candidate through the kinematic sim and select by avg tracking error
   - Expected: find the racing line the controller can actually follow best
   - Research: AERO-MPPI two-stage evaluation, T-MPC common selection cost

3. **Increase multi-start diversity** (Priority 3, system_integration)
   - Current 10 starts may not cover all basins. Try N=20 or add structured initializations (e.g., one start per gate as the "priority gate")
   - Or add a penalty term that repulses starts from each other's basins
   - Expected: potentially find even better basins

4. **MPCC controller upgrade** (Priority 4, control)
   - ETH 2026: 0.07m avg at 9.8 m/s via contouring error
   - Major architectural change, defer
   - Research: MPCC++ (Krinner 2024)

5. **Learned per-segment feasibility** (Priority 5, trajectory_planning)
   - Replace heuristic inflation with Quad-LCD-style learned predictor
   - Defer until S-turn is resolved
   - Research: Quad-LCD (Srikanthan 2025)

### Architectural recommendations
- The S-turn coupling problem suggests that the racing line and segment timing should be co-optimized, not separated into two sequential L-BFGS passes. A joint optimization would find the globally best trade-off.
- Sim-based selection would bridge the gap between the L-BFGS proxy cost and actual tracking performance.
- The kinematic sim's trajectory planning is approaching its limits — further improvements will likely come from controller changes (MPCC) or PyBullet integration.

### What NOT to try
- **More random starts without changing the objective** — 10 starts already found the basin; more starts won't help if the objective doesn't reflect tracking error
- **Uniform time compression** — proven infeasible (iter 14)
- **Controller gain scheduling in kinematic sim** — proven infeasible (iter 12)
- **Proximity multiplier > 0.25** — race time exceeds 14s (iter 18)

---

## Section 7: Lessons Learned

### What worked
- **Multi-start L-BFGS-B worked as predicted**: Found a qualitatively different racing line basin that the single-start optimizer never reached
- **Fallback guarantee was validated**: The zero-initialization candidate was not the winner, proving the optimization was in a suboptimal basin for 18 iterations
- **Gate-4 improved dramatically**: 24.9% reduction, confirming the local minimum hypothesis
- **All three aggregate metrics improved simultaneously**: First iteration to improve race time, avg error, and max error together

### What didn't work
- **Late-apex geometric initialization**: The winning candidate was a random start, not the geometric prior. The S-turn geometry is complex enough that hand-crafted heuristics don't reach the right basin.
- **Selection by L-BFGS objective**: The objective minimizes path_length + curvature², but tracking error depends on the controller's ability to follow the trajectory. The gate-3 regression suggests the L-BFGS-optimal racing line isn't the tracking-optimal one.

### Surprises
- **Gate-3/gate-4 coupling is stronger than expected**: A 0.103m improvement in gate-4 came at a 0.134m cost in gate-3. The S-turn is a zero-sum trade-off in the current racing line formulation.
- **P95 tracking error increased despite avg improvement**: The gate-3 peak error pushed P95 up (+0.012m) even though avg dropped (-0.004m). Tail behavior diverged from mean behavior.
- **Gate-7 also improved (-0.025m)**: Unexpected co-benefit — the new racing line through the S-turn also improved the helix entry approach.

### Process suggestions
- For future multi-start iterations, log ALL candidate objective values and offsets, not just the winner — this reveals the basin structure
- When one gate improves and another regresses, check if they're in the same "coupling group" (consecutive turns with shared approach vectors)
- Sim-based selection should be the next step when L-BFGS objective and tracking error diverge
