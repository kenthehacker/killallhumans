# Iteration 24 — Research Synthesis: Breaking the Bipartite Candidate Pool

**Bottleneck**: trajectory_planning — bipartite L-BFGS candidate pool (2 basins, no intermediates)
**Research focus**: Methods for generating intermediate racing line candidates

---

## Papers Analyzed (New)

1. **QuayPoints: A Reasoning Framework for Autonomous Racing** (2025, arXiv:2510.10886)
   - λ-parameterization of track width for continuous racing line interpolation
   - 55 constrained optimizations to identify pinch points vs. free zones
   - Key: `λ_interp = α·λ_A + (1-α)·λ_B` creates valid intermediate racing lines

2. **Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback** (2026, arXiv:2602.15642)
   - CMA-ES + NURBS for derivative-free population-based racing line search
   - Spatial constraint map learns from controller tracking errors
   - CMA-ES explores across basins; L-BFGS cannot

## Papers Referenced (Previously Analyzed)

3. **BO Racing Line** (Heilmeier 2020) — sim oracle for trajectory selection
4. **COP** (Bohm, ICRA 2022) — normalized multi-objective Pareto scoring
5. **F1-Init** (Shehadeh 2026) — initialization determines which basin optimizer finds

---

## Consensus Across Papers

### Strong consensus (3/3 new + referenced):
- **Interpolation between known solutions is valid and produces feasible trajectories.**
  QuayPoints demonstrates that convex combination of λ profiles yields physically valid racing lines.
  CMA-ES implicitly does this via population crossover.
  F1-Init shows initialization from a "blend" of known good solutions can bridge basins.

### Strong consensus (2/3):
- **Population-based search is superior to multi-start gradient for racing lines.**
  CMA-ES (Spatially-Aware) and QuayPoints (55-variant sweep) both emphasize exploring the full solution space rather than running gradient descent from random seeds.

### Practical insight:
- **For our specific bipartite problem, interpolation is simpler and more targeted than CMA-ES.**
  CMA-ES is the principled general solution, but it requires adding a dependency and 4000+ iterations.
  Offset-vector interpolation between our 2 known basins directly creates 3-5 new candidates with zero extra optimization cost.

---

## Contradictions / Nuances

- QuayPoints uses constrained sub-band optimization (restrict λ range per gate) while CMA-ES uses unconstrained population search. Both work but have different computational profiles. For our problem (only 2 basins, known), interpolation is more efficient than a blind population search.

- CMA-ES paper advocates replacing L-BFGS entirely. However, our L-BFGS with sim-based selection (iter 22-23) already works well. The issue is not the optimizer but the candidate diversity. Adding interpolated candidates to the existing pool is a minimal-risk enhancement.

---

## Recommended Implementation

**Approach: Basin-bridging interpolation (QuayPoints §4.4)**

After the existing 10-start L-BFGS pool is evaluated via kinematic sim:
1. Identify the two distinct basin solutions (lowest and highest race_time among valid candidates)
2. Generate 3 interpolated candidates: `offsets = α·offsets_A + (1-α)·offsets_B` for α ∈ {0.25, 0.5, 0.75}
3. Build trajectories for interpolated candidates and evaluate via the same kinematic sim
4. Add interpolated results to the candidate pool before normalized composite scoring
5. The existing scoring function (0.5·avg_err + 0.2·worst_gate + 0.3·race_time) selects the best

**Why this approach:**
- Directly addresses the bipartite pool: creates 3 new candidates in the unexplored gap
- Zero computational overhead for optimization (only 3 extra sim evaluations, ~0.5s each)
- Preserves all existing code; only modifies `_select_by_sim` to insert interpolated candidates
- Research-backed: QuayPoints Eq. (4.4), COP normalized scoring (already in place)
- No regression possible: the existing 10 candidates remain in the pool, so the worst case is that an interpolated candidate is selected and it's worse than Basin A — but the composite score would not select it if it's worse

**Expected impact:**
- Find a racing line with ~14.05s time AND ~0.25m avg error (Pareto-intermediate)
- Gate-3 error could drop from 0.374m to ~0.30m (intermediate approach angle)
- Race time may increase slightly from 13.99s to ~14.05s (but still <14.1s)
- Net improvement in composite score if a balanced trade-off exists

---

## Cross-Validation Challenges

**Challenge 1: Do intermediate offset vectors produce feasible trajectories?**
- Yes — our offsets are within [-0.6, 0.6] and linear interpolation preserves this range.
  The min-snap polynomial optimizer handles any gate positions within the opening.

**Challenge 2: Could intermediate candidates be worse than both basins?**
- Possible but unlikely for smooth metrics. The gate-3 approach angle varies continuously
  with offset, so intermediate offsets produce intermediate approach angles.
  The kinematic sim will reveal if any intermediate is actually worse.

**Challenge 3: Is 3 interpolation points enough?**
- For this iteration, yes. If 3 points show a clear gradient, the next iteration can
  refine further (e.g., 7 points). Start simple.

**Challenge 4: Basin identification robustness?**
- We identify basins by race_time: fastest = Basin A, slowest = Basin B.
  With 10 candidates clustering into 2 groups, this is unambiguous.
