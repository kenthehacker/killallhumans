# Iteration 49 Research Synthesis

## Context
- Iteration 49, current bottleneck: trajectory_planning
- Baseline: 13.31s race time, 0.163m avg error, 100% gate pass, deterministic
- Worst gates: gate-5 (0.252m), gate-4 (0.236m), gate-8 (0.223m)
- ILC is at 8 iterations, per-section alphas at Pareto-optimal values
- The backlog recommended "Racing line lateral offset re-optimization" but code analysis reveals this is extremely risky (basin switching)

## Research Papers Analyzed (3 new, 136 total)

### 1. Fast Data-Driven ILC with Nesterov Acceleration (Wang 2023, arXiv:2312.14326)
- Integrates Nesterov's accelerated gradient method into ILC
- Key insight: a **hybrid approach** switching between fast (momentum-based) and classical ILC achieves both fast convergence speed AND good asymptotic performance
- Convergence rate: O(1/k²) for accelerated vs O(1/k) for standard P-type
- The momentum term: u_{k+1} = u_k + alpha * L * e_k + gamma * (u_k - u_{k-1})
- Switching condition: when accelerated ILC stops improving (oscillating), switch back to classical
- **Directly applicable**: our per-section ILC uses standard P-type updates. Adding momentum could squeeze more convergence from 8 iterations.

### 2. Speed Up Convergence of ILC for High Precision Motions (Longman 2023, arXiv:2307.15912)
- Uses model-based optimization to pre-compute an approximate correction, then uses ILC to refine
- Key insight: learning the world model from ILC data accelerates convergence by providing a warm-start
- **Less directly applicable**: our ILC already has a good warm start (the trajectory itself). The benefit is marginal when ILC is already near convergence.

### 3. CDBO: Coordinate Descent Bayesian Optimization for Racing (Cully 2018, arXiv:1802.06179)
- Uses coordinate descent within BO to optimize racing parameters one-at-a-time
- Key insight: for high-dimensional racing parameter spaces, optimizing one coordinate at a time avoids combinatorial explosion while still finding good solutions
- **Relevant to lateral offset re-optimization**: suggests sweeping one gate's offset at a time rather than jointly
- **BUT**: our basin switching analysis shows even single-gate offset changes are catastrophic

## Consensus Across Papers
1. **Momentum/acceleration in ILC provides 2-3x faster convergence** — both Wang 2023 and the tuning parameter paper (already analyzed) agree
2. **Hybrid strategies (fast + classical) outperform either alone** — Wang 2023 specifically
3. **Coordinate descent is effective for high-dimensional racing optimization** — Cully 2018, Heilmeier 2020 (already analyzed)

## Contradictions
1. **Momentum helps convergence** (Wang 2023) vs **our ILC is "essentially converged"** (iteration 48 diagnostic). Question: is the ILC truly converged, or stuck in a shallow basin that momentum could escape?
2. **Lateral offset re-optimization** (backlog priority #1) vs **basin switching risk** (dozens of failed approaches). Resolution: do NOT change offsets this iteration.

## Recommended Implementation Direction
**Heavy-ball momentum ILC** — the strongest evidence-backed change that:
1. Has NOT been tried before (genuinely new approach)
2. Does not touch the racing line geometry (no basin switching risk)
3. Can be implemented in 15 lines of code
4. Can be reverted instantly if it regresses
5. Has theoretical backing from Wang 2023 (Nesterov acceleration in ILC)
6. May help the ILC escape shallow convergence plateaus at gates 4, 5, 8

### Algorithm
Standard P-type ILC:
```
u_{k+1} = u_k + alpha * Q * e_k
```

Heavy-ball momentum ILC (proposed):
```
u_{k+1} = u_k + alpha * Q * e_k + gamma * (u_k - u_{k-1})
```

Where gamma ∈ [0.1, 0.4] controls momentum strength.

### Risk Assessment
- LOW risk: only modifies the offline ILC loop, not the runtime controller
- Momentum could cause over-correction → mitigated by max_correction_m clipping
- If avg error regresses, revert (takes <1 minute)
- Sweep gamma = 0.1, 0.2, 0.3, 0.4 to find optimal value per-section

### Expected Impact
- 2-5% avg error reduction if momentum helps deeper convergence
- Primarily at gates 4-5 (inflection/post-inflection) where error is highest
- No change to race time (ILC doesn't affect trajectory timing)
