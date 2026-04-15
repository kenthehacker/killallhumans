# Iteration 46 — Research Synthesis: ILC Tuning for Gate-4 and Gate-7

## Papers Analyzed (3 new, 127 total)
1. Longman et al. 2023 — Speed Up ILC Convergence via Model Warmstarting
2. Brooks & Greve 2024 — ILC for Fast Nonlinear Oscillatory Dynamics
3. CDC 2024 — Constrained ILC with Alternating Projection for Actuator Saturation

## Current Problem
- Gate-4: 0.302m error (over 0.25m aspirational threshold)
- Gate-7: 0.252m error (just over threshold)
- Gate-8: 0.240m (near threshold)
- Race time: 13.31s (all-time best)
- ILC configured with per-section parameters; gate-4 in inflection section (max_corr=0.15m), gate-7 in helix section (max_corr=0.45m)

## Research Consensus
1. **Correction magnitude limits matter**: The CDC 2024 paper (alternating projection) shows that hard clipping ILC corrections can degrade steady-state error by 15-25% compared to constraint-aware approaches. Our gate-4 section has max_correction_m=0.15m, which is likely binding.
2. **More iterations ≠ better**: Longman 2023 confirms that ILC converges to a model-specific error floor. Beyond that floor, additional iterations overfit to model errors. Our 5-iteration limit is appropriate — the issue is per-iteration correction caps, not iteration count.
3. **Per-section approach is validated**: Both the constrained ILC paper and the Brooks 2024 paper support spatially varying ILC parameters. Different track regions have different dynamics and need different correction limits.

## Contradictions
- Longman suggests more model-based iterations could help, but our iter 35 showed 8 iterations caused saturation. These aren't contradictory — Longman advocates for convergence-based stopping, not blind iteration count increase.

## Actionable Direction
The strongest evidence points to **increasing the inflection section max_correction_m from 0.15m to 0.20m** specifically for gate-4. Iter 43 showed that increasing BOTH inflection AND post-inflection caps to 0.20m regressed gate-5. The key is to increase ONLY the inflection section cap while keeping post-inflection at 0.15m. This targets gate-4 without contaminating gate-5.

Additionally, extending the inflection_end boundary from 440→460 gives the inflection section's higher-bandwidth filter (0.40 Hz vs 0.35 Hz) access to more of the gate-4 approach trajectory. Combined with the higher correction cap, this should reduce gate-4 error by 10-20%.

For gate-7, the helix section already has max_correction_m=0.45m and alpha=0.4. Increasing to 0.50m is conservative and supported by the CDC paper's finding that constrained ILC benefits from relaxed limits when actuator headroom exists. Our actuator utilization (0.87 average thrust, 0.85 max tilt) shows moderate headroom.

Velocity correction re-tuning: the vel_scale was calibrated for the old trajectory (time_weight=2.0). With time_weight=2.3, velocities are higher. Testing a moderate increase in helix vel_scale from 0.7→0.8 could help, but this is risky given iter 41's sensitivity to velocity correction changes.

## Recommended Changes (prioritized)
1. **Inflection max_correction 0.15→0.20m** — targeted gate-4 fix
2. **inflection_end 440→460** — extend inflection coverage
3. **Helix max_correction 0.45→0.50m** — gate-7 improvement
4. **Helix vel_scale 0.7→0.75** (conservative) — probe velocity sensitivity

## Risk Assessment
- Gate-5 regression from inflection cap increase (main risk, mitigated by not touching post-inflection)
- Gate-8/9 regression from helix changes (low risk, helix correction already working well)
- ILC convergence instability from higher caps (monitor via per-iteration error tracking)
