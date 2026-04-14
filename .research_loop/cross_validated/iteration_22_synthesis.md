# Iteration 22 Research Synthesis — Sim-Based Racing Line Selection

## Current Bottleneck
**trajectory_planning** — Gate-3 S-turn tracking at 0.374m (regressed from 0.345m in iter 20).
Root cause: L-BFGS proxy objective (path_length + curvature²) doesn't correlate with actual tracking error.

## Papers Analyzed (New)
1. **BO Racing Line** (Jain & Morari, CDC 2020) — Bayesian Optimization for racing line selection
2. **Multi-objective PID Optimization** (Vaiuso et al., 2025) — Black-box optimization via closed-loop sim evaluation
3. **Sampling-Based Racing Line** (Ögretmen et al., IV 2024) — Sample multiple trajectories, select by cost

## Papers Referenced (Previously Analyzed)
4. **AERO-MPPI** (Chen et al., ICRA 2026) — Ensemble of M=15 parallel optimizers, select best by rollout
5. **T-MPC** (de Groot et al., T-RO 2024) — P parallel planners, non-guided fallback (Theorem 2)
6. **TACO** (Sanghvi et al., 2025) — Trajectory-aware controller optimization, adapts trajectory to controller
7. **F1-Init** (Shehadeh, 2026) — Initialization determines basin; multi-start needed

## Research Consensus

### Strong Agreement (5+ papers)
1. **Multi-candidate evaluation with selection is the right paradigm**: AERO-MPPI (15 parallel), T-MPC (P parallel), BO Racing Line (sequential eval), Sampling Racing Line (N=800 candidates). All agree: generate diverse candidates, evaluate each, select best.

2. **The evaluation metric must match the actual objective, not a proxy**: BO Racing Line evaluates by actual lap time simulation, not geometric smoothness. TACO evaluates by predicted tracking error, not path curvature. Multi-obj PID evaluates by closed-loop sim, not analytical stability margins. AERO-MPPI re-evaluates all candidates with a common cost after optimization.

3. **The selection step is cheap relative to generation**: Our 10-candidate L-BFGS already generates diverse racing lines. The missing step is evaluating them by kinematic sim (which runs in ~0.17s). 10 × 0.17s = 1.7s total — negligible vs offline planning.

### Actionable Takeaway
**Replace the L-BFGS objective-based selection with kinematic-sim-based selection.**

Current flow:
```
10 L-BFGS starts → 10 racing lines → select by L-BFGS objective (proxy)
                                       ↓
                                  Build full trajectory → Run kinematic sim
```

Proposed flow:
```
10 L-BFGS starts → 10 racing lines → Build trajectory for each
                                       ↓
                                  Run kinematic sim for each → select by tracking error
```

### Multi-Objective Selection
The BO Racing Line paper shows that lap time (speed) and tracking accuracy are often anti-correlated. Our backlog notes the same: faster racing lines have higher tracking error. The selection criterion should be:

**Primary**: avg tracking error (lower is better)
**Constraint**: race time must not regress by more than 2% from best candidate

This is a constrained selection, not a pure minimization. TACO uses a similar approach: minimize tracking error subject to trajectory constraints.

## Contradictions / Caveats
- BO Racing Line suggests iterative refinement (100+ evaluations). We don't need that — we already have 10 diverse candidates from multi-start L-BFGS. Simple enumeration is sufficient.
- Sampling Racing Line uses 800 candidates. Our 10 is much fewer, but each is a fully optimized L-BFGS solution (not a random polynomial).

## Implementation Direction
1. After the existing multi-start L-BFGS racing line optimization, build full trajectories for each of the 10 racing line candidates
2. Run each through the kinematic sim (same physics as benchmark)
3. Select the candidate with lowest avg tracking error, subject to race time within 2% of fastest
4. This is zero-risk: T-MPC Theorem 2 guarantees we can't regress vs the current approach (zero-init is included as a candidate)

## Expected Impact
- Gate-3: recover from 0.374m toward ~0.34m if a better racing line exists in the multi-start pool
- Avg tracking error: potential 5-10% improvement if current L-BFGS selection is suboptimal
- Race time: neutral or slight regression (constrained to <2%)
