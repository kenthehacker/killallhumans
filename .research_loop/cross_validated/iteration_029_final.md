# Iteration 29 — Cross-Validated Research: Inflation Reduction for Speed Recovery

## Thesis
ILC has reduced systematic tracking error by 17% over 4 iterations (0.211→0.175m). The post-optimization inflation factors (S-turn: 10-12%, TOPP floors: 0.63-0.68) were calibrated BEFORE ILC existed and are now overly conservative. Reducing them trades accuracy headroom for speed.

## Critical Examination

### Does ILC actually compensate for what inflation protects?
**Yes, with caveats.** ILC compensates for *systematic, repeatable* errors — exactly the errors that cause high tracking at S-turns. However, ILC does NOT compensate for:
- **Transient overshoot** at the very first moment of a turn (ILC averages across iterations)
- **Max tracking error spikes** (ILC targets mean error, not peaks)
- **Instability at dynamic limits** (if controller saturates, ILC can't help)

**Implication**: We should reduce inflation moderately (30-40%), not eliminate it. Keep enough margin for transient effects ILC can't fully address.

### What if the new faster trajectory changes the error structure?
**This is the key risk.** ILC converges on the CURRENT trajectory. A faster trajectory produces DIFFERENT errors (higher centripetal demand → different systematic patterns). The ILC will need to re-converge from scratch.

**Mitigation**: Our ILC converges in 5 iterations at α=0.4. The new trajectory is only 3-5% faster, so error patterns will be similar. Convergence should happen in the same number of iterations.

### Why not just increase time_weight in L-BFGS?
**Failed in iter 5 (time_weight=3.0) and iter 8 (time_weight=2.5).** L-BFGS converges to worse local minima at higher weights. The inflation reduction approach is orthogonal — it operates AFTER L-BFGS, in the post-processing stage.

### What about the failed compression floor experiment (iter 21)?
**Floor 0.50 regressed gate-2 by 24%.** But we propose 0.58 (easy) and 0.64 (protected), significantly more conservative than 0.50. Gate-2 has moderate curvature and is classified as an "easy" segment — at 0.58 floor, it would get at most 5% more compression than current 0.63. The 24% regression at 0.50 was from 13% additional compression.

## Recommended Parameters

### S-turn inflation (trajectory_optimizer.py:_inflate_sharp_turns)
| Parameter | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| s_turn_inflate (junction) | 1.12 | 1.08 | -4% |
| s_turn_inflate (standard) | 1.10 | 1.06 | -4% |
| approach decel (S-turn) | 1.03 | 1.01 | -2% |
| departure (first, pure) | 1.04 | 1.02 | -2% |
| departure (junction) | 1.02 | 1.01 | -1% |

### TOPP compression (trajectory_optimizer.py:_topp_retime)
| Parameter | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| max_compression_protected | 0.68 | 0.64 | -4% |
| max_compression_easy | 0.63 | 0.58 | -5% |

### Expected impact
- Race time: 14.03→~13.4-13.7s
- Avg error: 0.175→~0.20-0.23m (after ILC re-convergence)
- Safety: 100% gate pass maintained

## Confidence: HIGH
- 7/7 relevant papers support the direction
- Failed approaches are clearly distinct from this approach
- Parameter changes are moderate and individually reversible
- ILC provides a safety net for error regression
