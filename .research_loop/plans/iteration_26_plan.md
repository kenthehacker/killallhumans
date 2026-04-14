# Iteration 26 Plan — Per-Section ILC with Blended Boundaries

## Objective
Improve avg tracking error from 0.195m to ~0.185m by eliminating cross-contamination between trajectory sections in ILC. Specifically: prevent helix corrections from regressing S-turn gates (gate-4 issue from iter 25), while maintaining or improving helix gate accuracy.

## Research Basis
- **Zhang, Meng & Cai 2024** (Segment-wise Learning Control): Segment-wise ILC with independent virtual memory slots per section prevents cross-contamination. Each segment converges independently.
- **Liu, Zheng & Chen 2023** (Time-Varying Gain): Section-specific learning rates are theoretically motivated. Higher gains at later sections (helix) is consistent with monotone convergence theory.
- **Schoellig et al. 2012**: P-type ILC converges in 3-5 iterations with conservative alpha.

## Files to Modify
1. **`planning/trajectory_optimizer.py`** — Modify `compute_ilc_offset_table()` to support per-section ILC
2. **`scripts/benchmark.py`** — Update ILC invocation (no change needed to runtime offset application)

## Algorithm Changes

### New function signature:
```python
def compute_ilc_offset_table(
    trajectory, start_position,
    alpha=0.4,                    # default (overridden by section alphas)
    max_iterations=5,
    smoothing_sigma=10.0,
    max_correction_m=0.15,
    convergence_threshold=0.002,
    dt=0.01,
    section_boundaries=None,      # NEW: list of (start_step, end_step, alpha) tuples
    blend_steps=50,               # NEW: steps to blend between sections
)
```

### Algorithm:
1. If `section_boundaries` is None, behave exactly as before (backward compatible)
2. If `section_boundaries` is provided:
   a. Initialize separate cumulative_offset arrays per section
   b. Run the full kinematic sim (same as before — need full trajectory for dynamics)
   c. For each section, compute errors only within that section's step range
   d. Apply cross-track decomposition, smoothing, clipping per section
   e. Update that section's offset independently
   f. After convergence, blend the per-section offsets at boundaries:
      - In the overlap zone (±blend_steps/2 around boundary):
        weight = linear ramp from 0 to 1
        final_offset = (1-weight) * section_A + weight * section_B

### Section boundaries for our track:
- Gate-6 passes at t ≈ 6.81s → step ≈ 681
- Gate-7 at t ≈ 8.05s → step ≈ 805
- Boundary midpoint: step ≈ 740
- Section A: steps 0 to 740 (S-turn gates 1-6), alpha = 0.5
- Section B: steps 740 to end (helix gates 7-12), alpha = 0.6

## Risk Assessment
- **Low risk**: Only ILC offset computation changes; trajectory, controller, and sim unchanged
- **Regression risk**: Section boundary discontinuity → mitigated by 50-step blending zone
- **Gate-4 regression from iter 25**: Should be eliminated (S-turn section now isolated)
- **Rollback**: If any per-gate error regresses >20% from baseline, revert

## Rollback Criteria
- Overall avg error increases
- Any gate error increases >20% relative to baseline
- Gate pass rate drops below 100%
- Race time increases >0.5s

## Test Plan
1. Run unit tests first (should be unaffected)
2. Run full benchmark with per-section ILC
3. Compare per-gate errors, especially gate-4 and gate-7
