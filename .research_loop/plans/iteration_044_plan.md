# Iteration 44 Plan — Speed Recovery via Turn Inflation Reduction

## Objective
Reduce race time from 14.08s toward 13.0-13.5s by reducing post-optimization turn inflation factors in `_inflate_sharp_turns()`. Target: avg tracking error stays under 0.22m (giving 0.03m safety margin below 0.25m threshold). No gate pass regressions.

## Research Basis
- CPC (Foehn 2021): Speed can increase when tracking error is below threshold
- TACO (Sanghvi 2025): Controller capability should dictate trajectory aggressiveness
- Schoellig 2012: ILC compensates 45-87% of systematic error — will re-converge on faster trajectory
- Spatial ILC (Lv 2023): Progressive speed increase within virtual tube
- MonoRace 2026: A2RL winner uses no post-processing inflation

## Files to Modify
1. `planning/trajectory_optimizer.py` — `_inflate_sharp_turns()` method (lines 631-895)

## Algorithm Changes

### _inflate_sharp_turns modifications:

| Parameter | Current | New | Reduction |
|-----------|---------|-----|-----------|
| Sharp turn severity (>60°) | 0.25 | 0.12 | -52% |
| Centripetal severity | 0.15 | 0.08 | -47% |
| S-turn junction | 1.09 | 1.04 | -56% |
| S-turn second-gate | 1.07 | 1.03 | -57% |
| S-turn approach | 1.01 | 1.005 | -50% |
| S-turn first departure | 1.02 | 1.01 | -50% |
| S-turn junction departure | 1.005 | 1.002 | -60% |
| Proximity factor | 0.22 | 0.12 | -45% |
| Helix entry | 1.12 | 1.06 | -50% |
| Helix interior | 1.10 | 1.05 | -50% |

### Why this is NOT uniform compression (which failed in iter 14):
- Iter 14 compressed ALL segments uniformly, including straight segments already at their limit
- This approach only reduces the SAFETY MARGIN added to turn segments
- Straight segments are untouched
- The L-BFGS-optimized base times remain unchanged
- TOPP compression floors remain unchanged

## Risk Assessment
- **Gate-2 risk**: Highest current error (0.214m) with only 0.036m headroom. S-turn inflation reduction may push it over 0.25m. Mitigation: S-turn parameters get more conservative reduction.
- **Gate-3 risk**: Second highest error (0.191m). S-turn junction inflation directly affects gate-3.
- **Helix risk**: Gate-7 at 0.164m is manageable — helix has 0.086m headroom.
- **Basin switching risk**: LOW because racing line is cached and TOPP floors unchanged.

## Rollback Criteria
- Revert if avg tracking error > 0.25m (threshold)
- Revert if any gate loses pass-through (gate_pass_rate < 1.0)
- Revert if race time increases (opposite of intended effect)
- Revert if drone crashes

## Test Plan
1. Run full benchmark after changes
2. Check per-gate errors — gate-2/3 are the critical gates
3. If gate-2 > 0.24m, increase S-turn factors slightly and re-test
