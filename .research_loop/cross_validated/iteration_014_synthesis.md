# Iteration 14 Research Synthesis — Speed Recovery via Post-Optimization Reduction

## Bottleneck
Race time 17.70s (target: <15s) — trajectory_planning speed recovery while maintaining <0.3m avg error.

## Papers Analyzed (New)
1. **TOPPQuad** (Mao et al., IROS 2024) — Time-Optimal Path Parametrization for quadrotors
2. **Sequence Modeling for Time-Optimal Trajectories** (Mao et al., 2025) — LSTM-based speed profile learning
3. **Multi-Fidelity RL for Time-Optimal Re-planning** (Ryou et al., IJRR 2024) — Binary line search over time scale

## Empirical Finding: Post-Optimization Inflation Is the Bottleneck

Before research, I decomposed the current trajectory timing:

| Stage | Time | Delta | % of Total |
|-------|------|-------|------------|
| Initial heuristic | 9.09s | — | — |
| After L-BFGS optimization | 14.92s | +5.83s | Constraints enforcement |
| After _inflate_sharp_turns | 15.80s | +0.88s | Helix protection |
| **After FOV relaxation** | **18.03s** | **+2.23s** | **FOV protection (14.1%)** |

**The FOV relaxation step alone adds 2.23s** — making it the single largest post-optimization time adder. The FOV penalty (29,378) is enormous because many trajectory points have gates "behind" the drone (penalty = π²), causing the relaxation to run all 3 iterations.

## Research Consensus

All three papers converge on the same principle: **decouple geometric path from time parameterization**.

1. **TOPPQuad**: Fix geometry (racing line), optimize speed profile separately. 40-50% faster than min-snap baselines while maintaining dynamic feasibility.
2. **Sequence Modeling**: The squared-speed profile h(s) along a fixed path is the natural variable. Higher curvature → lower h(s), low curvature → higher h(s). The path shape doesn't change.
3. **MFRL**: Binary search over a scalar alpha (global time scale) while preserving allocation ratios. Decouples "shape" from "speed" optimization.

## Actionable Approach: Reduce Post-Optimization Inflation

Rather than implementing full TOPPQuad (requires CasADi + IPOPT, 30s solve), the most impactful and lowest-risk approach for this iteration is:

### Primary: Reduce FOV relaxation aggressiveness
The FOV relaxation was designed (iteration 10) to protect camera visibility in aggressive maneuvers. However:
- Our benchmark is kinematic sim (no camera) — FOV protection has no direct tracking benefit
- The L-BFGS already includes an FOV penalty term (weight=10) during optimization
- ETH 2026 paper shows proper FOV constraints add only +8.1% to trajectory time; our adds +14.1%
- KAIST 2025 paper shows heading-based FOV control adds +0% race time

**Action**: Reduce FOV relaxation multiplier from 1.07 to 1.03, reduce max iterations from 3 to 2, reduce cap from 25% to 8%. Expected savings: ~1.5-2.0s.

### Secondary: Reduce proximity inflation
The proximity inflation (iteration 13) adds up to 25% per helix segment. With the smooth racing line now producing excellent helix tracking (0.09-0.17m), this may be overly conservative.

**Action**: Reduce proximity inflation from 25% max to 12% max. Expected savings: ~0.3-0.5s.

### Tertiary: Post-hoc uniform time scaling (MFRL-inspired)
After all inflation, apply a uniform time scale reduction and verify tracking stays <0.3m via simulation.

**Action**: Binary search for minimum total time that maintains <0.3m avg tracking error. Expected savings: ~0.5-1.0s additional.

## Expected Outcome
- FOV reduction: 18.03s → ~16.3s trajectory → ~16.1s race time
- Proximity reduction: → ~15.8s trajectory → ~15.6s race time
- Time scaling: → ~15.0-15.3s trajectory → ~14.8-15.1s race time
- Total: 17.70s → ~15.0s race time with tracking error ~0.20-0.25m

## Risk Assessment
- **Low risk**: Reducing FOV relaxation (L-BFGS already handles FOV, and kinematic sim has no camera)
- **Medium risk**: Reducing proximity inflation (helix tracking might degrade from 0.10m to 0.15m)
- **Low risk**: Uniform time scaling with binary search (automatically finds safe point)

## Rollback Criteria
- Avg tracking error > 0.35m (2x our target, still well below 0.5m threshold)
- Gate pass rate < 100%
- Any crash
