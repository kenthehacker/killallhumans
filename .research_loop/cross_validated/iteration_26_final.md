# Iteration 26 — Cross-Validated Research: Per-Section ILC

## Final Direction
**Per-section ILC with blended boundaries and section-specific learning rates.**

## Changes from Synthesis
1. **Dropped Q-filter replacement** — Keep Gaussian smoothing to isolate the per-section effect. Butterworth is a future optimization.
2. **Reduced alpha ambition** — Don't increase alpha beyond 0.5-0.6. The main gain is section independence, not faster convergence.
3. **Added boundary blending** — Critical for avoiding offset discontinuity at the section boundary.

## Algorithm
1. Segment trajectory at the midpoint between gate-6 and gate-7 (approximately t ≈ 7.4s, step ≈ 740)
2. Run ILC for each section independently:
   - Section A (steps 0 to boundary+overlap): S-turn, alpha_a = 0.5
   - Section B (steps boundary-overlap to end): Helix, alpha_b = 0.6
3. In the overlap zone (±50 steps around boundary):
   - Blend offsets: offset = (1-w) * offset_A + w * offset_B, where w ramps linearly from 0 to 1
4. Each section has its own convergence check
5. Max correction per section: 0.15m (unchanged)

## Research Backing
- Zhang 2024: Segment-wise learning prevents cross-contamination between trajectory sections
- Liu 2023: Section-specific gains are theoretically motivated (time-varying gains for monotone convergence)
- Schoellig 2012: P-type ILC with conservative alpha converges in 3-5 iterations

## Risk Assessment
- **Low risk**: Only changes ILC offset computation, not the trajectory or controller
- **Rollback**: If per-gate regressions exceed 20%, revert to global ILC
- **Expected outcome**: Gate-4 regression eliminated, helix gates maintained or improved
