# Iteration 45 Plan — TOPP Speed Recovery + Gate-3 ILC Fix

## Objective
Reduce race time from 13.51s toward 13.0s by increasing TOPP acceleration budgets and reducing compression floors. Simultaneously improve gate-3 ILC to keep avg error under 0.22m. Target: all per-gate errors under 0.24m.

## Research Basis
- TOPPQuad (Mao 2024): Full-dynamic TOPP shows 15-20% speed gain vs simplified budgets
- FBGA (Piazza 2025): Forward-backward acceleration budgets should match physical limits
- Schoellig 2012: ILC alpha directly controls convergence rate
- Bristow & Alleyne 2007: Q-filter + alpha tuning for section-specific ILC
- Track-centric ILC 2026: Higher alpha in difficult sections is intended use

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — `_topp_retime()` method

#### Change 1: Increase acceleration budgets
| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| a_centripetal | 10.0 | 11.0 | Physical limit ~11.4 (g*tan(0.85)). Using 96% vs 88% of capacity. |
| a_longitudinal | 8.0 | 9.5 | Kinematic sim supports faster transitions with ILC compensation. |

#### Change 2: Reduce compression floors
| Floor | Current | New | Reduction |
|-------|---------|-----|-----------|
| max_compression_helix | 0.72 | 0.68 | -5.6% |
| max_compression_easy | 0.59 | 0.55 | -6.8% |
| max_compression_protected | 0.65 | 0.61 | -6.2% |
| max_compression_sturn | 0.70 | 0.70 | NO CHANGE (basin switching risk) |

### 2. `scripts/benchmark.py` — ILC section parameters

#### Change 3: Increase gate-3 inflection ILC alpha
| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| Inflection section alpha | 0.4 | 0.45 | 12.5% faster convergence at gate-3. Schoellig 2012. |

Keep all other ILC parameters unchanged (alpha=0.4 for other sections, cutoffs, vel_scale).

## Why this is NOT a repeat of failed approaches
- **Not iter 17** (a_centripetal reduction): We're INCREASING, not decreasing. And floors are also changing.
- **Not iter 21** (easy floor only): We're changing floors + acceleration budgets + ILC simultaneously.
- **Not iter 26** (ILC alpha 0.5/0.6 all sections): Only inflection section, only to 0.45.
- **Not iter 29-32** (inflation+TOPP combined): We're not touching inflation. Racing line is cached.
- **Not iter 37** (joint S-turn/helix floor): We're leaving S-turn floor alone.
- **Not iter 39** (S-turn floor decrease): Keeping S-turn at 0.70.

## Risk Assessment
- **Gate-3 risk (HIGH)**: Currently 0.226m with 0.024m headroom. ILC alpha increase should reduce this, but speed increase will push it back up. Net effect uncertain. If gate-3 > 0.25m, back off on helix/protected floors first.
- **Gate-2 risk (MEDIUM)**: Currently 0.216m with 0.034m headroom. S-turn floor unchanged protects gate-2. Pre-inflection ILC unchanged (vel_scale=0.0).
- **Basin switching risk (LOW)**: Racing line is cached (iter 33). Only changing TOPP parameters, not inflation or racing line.
- **Gate-8 risk (MEDIUM)**: Currently 0.192m with 0.058m headroom. Helix floor reduction may increase gate-8 error.

## Rollback Criteria
- Revert ALL if avg tracking error > 0.25m
- Revert ALL if any gate error > 0.25m
- Revert ALL if gate pass rate < 1.0
- Revert ALL if race time increases
- Revert ALL if drone crashes

## Fallback Plan
If full changes cause threshold failures:
1. First try: revert floor reductions only (keep acceleration budget + ILC alpha)
2. Second try: revert acceleration budget too (keep only ILC alpha)
3. Third try: revert everything

## Test Plan
1. Run full benchmark after all changes
2. Check per-gate errors — gate-3, gate-2, gate-8 are critical
3. If gate-3 > 0.24m, back off on protected floor (0.61→0.63)
4. If gate-8 > 0.24m, back off on helix floor (0.68→0.70)
