# Iteration 48 Research Synthesis — ILC Convergence Depth & Competition Readiness

## Papers Discovered / Analyzed

### New Papers (3)
1. **"An ILC Algorithm with a Tuning Parameter for Fastest Convergence Speed"** (Sci China Inf Sci, 2026)
   - Proposes using past TWO iterations' data (not just one) to compute an optimal tuning parameter
   - Derives explicit expression for the optimal parameter maximizing convergence speed
   - Key insight: 2-iteration lookback accelerates convergence vs standard P-type ILC

2. **"MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI"** (arXiv 2601.15222, 2026)
   - Won A2RL 2025 competition, beat human world champions at 100 km/h
   - Uses offline optimization to tune state estimation parameters using known gate geometry
   - Key for competition: domain randomization for sim-to-real, robust to camera interference/IMU saturation

3. **"Quasi-Periodic Gaussian Process Predictive ILC (QPGP-PILC)"** (arXiv 2602.18014, Feb 2026)
   - Predicts NEXT iteration's error profile using GP regression across iteration history
   - Element-wise QPGP converges fastest; block-based is second
   - Key insight: predicting future error (not just correcting past error) accelerates convergence

### Already-Analyzed Papers Referenced
- Longman 2023 (arXiv:2307.15912): Model-based warmstarting for ILC convergence speedup
- Schoellig 2012: Optimization-based ILC for quadrocopter trajectory tracking
- Bristow & Alleyne 2007: Time-varying Q-filter bandwidth for per-section ILC
- Liu, Zheng & Chen 2023: Section-specific gains for monotone convergence

## Consensus Across Papers

1. **More ILC iterations help when not converged**: All ILC papers agree that if the error is still decreasing between iterations, more iterations produce better results. The convergence threshold (0.002m in our case) is the right guard.

2. **Per-section learning rates prevent over-correction**: Zhang 2024, Liu 2023, and our own empirical results confirm that sections with different dynamics need different alpha values. Our 4-section layout is well-supported.

3. **Alpha reduction with more iterations**: When increasing iteration count, the effective total correction ≈ alpha × N_iterations × average_correction_per_iteration. To prevent saturation, alpha should scale roughly as 1/sqrt(N) for stability. Going from 7→8 iterations (~14% more), alphas should decrease ~7%.

4. **Competition robustness requires parameter margin**: MonoRace's success came from extensive domain randomization and offline parameter optimization. Our deterministic benchmark is good for development, but competition will have noise.

## Contradictions / Tensions

1. **Predictive ILC (QPGP) vs our simple P-type**: QPGP would be more efficient but requires significant implementation complexity. With only 3 iterations remaining, implementing a GP-based ILC is too risky. Our simple P-type with more iterations achieves similar convergence.

2. **Optimal tuning parameter (2026 Sci China)**: Using 2-iteration lookback could accelerate convergence, but our section_boundaries architecture makes this complex to implement per-section. The potential gain (saving 1-2 iterations) is less valuable than the simplicity of just running 1 more iteration.

## Actionable Takeaways (Ranked by Relevance)

1. **ILC max_iterations 7→8 with alpha rebalancing** (HIGH PRIORITY)
   - Research basis: Longman 2023 (convergence speedup), QPGP-PILC 2026 (more iterations = deeper convergence)
   - Expected alpha adjustments: reduce all sections by ~7% to compensate for accumulated corrections
   - Pre-inflection: 0.30→0.28, Inflection: 0.50→0.47, Post-inflection: 0.40→0.37, Helix: 0.45→0.42
   - Risk: gate-2 or gate-9 regression from over-correction. Monitor carefully.

2. **Post-inflection alpha increase for gate-5** (MEDIUM PRIORITY)
   - Gate-5 (0.264m) is now the worst gate. It's in the post-inflection section (440-740).
   - Counter-intuitive: while overall alphas decrease for 8 iterations, the post-inflection section might benefit from a RELATIVE increase vs other sections.
   - Try: post-inflection alpha 0.40→0.42 (while others decrease)

3. **Convergence monitoring** (LOW PRIORITY, diagnostic)
   - Add logging to ILC to track per-iteration avg error. If error is still decreasing at iteration 7, iteration 8 will help. If it plateaued, 8 iterations won't add value.

## Proposed Implementation Direction

**Primary approach**: Increase ILC max_iterations from 7 to 8 with per-section alpha rebalancing. Test multiple alpha configurations:

- Config A (conservative): all alphas reduced by 7% → pre=0.28, infl=0.47, post=0.37, helix=0.42
- Config B (gate-5 targeted): same as A but post-inflection alpha=0.42 (kept higher)
- Config C (minimal change): max_iterations=8, all alphas unchanged (test convergence threshold)

If 8 iterations early-terminates (convergence_threshold=0.002 triggers), then reduce threshold to 0.001.

**Rollback criteria**: Revert if avg error increases >2% OR any gate regresses >20% OR race time increases >0.5s.
