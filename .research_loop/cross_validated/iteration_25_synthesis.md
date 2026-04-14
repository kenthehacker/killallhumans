# Iteration 25 Research Synthesis — Offline ILC Trajectory Pre-Compensation

## Problem Statement
The helix section (gates 7-10) has a persistent tracking error floor of 0.24-0.33m that has survived 8+ iterations of racing line optimization, time inflation tuning, and speed profiling changes. The PD controller consistently lags at the same locations in the same way. This is a SYSTEMATIC error — the exact kind that Iterative Learning Control was designed to eliminate.

## Papers Analyzed (new + existing)

### New papers (iteration 25)
1. **Schoellig et al. 2012** — "Optimization-Based Iterative Learning for Precise Quadrocopter Trajectory Tracking" (Autonomous Robots)
   - Foundational ILC work for quadrotors. Feedforward correction via Kalman filter + convex optimization.
   - Sub-centimeter tracking achieved in 3-5 iterations on real hardware (87% error reduction).
   - Key insight: decompose tracking error into systematic (learnable) and random (noise floor) components.

2. **QPGP-ILC 2026** — "Quasi-Periodic Gaussian Process Predictive ILC" (arXiv:2602.18014)
   - Extends ILC with GP prediction for drifting disturbances. O(p³) complexity.
   - Not directly needed for our deterministic sim, but validates ILC is still an active field.

### Existing papers (from prior iterations)
3. **Spatial ILC (Lv et al. 2023)** — "Time-Optimal Spatial ILC within a Virtual Tube" (arXiv:2306.15992)
   - Spatial-domain ILC that adjusts speed profile based on tracking error. 17-20% faster than baselines.
   - Convergence in ~20 iterations. Model-free.
   - Key contribution: spatial (arc-length) parameterization decouples speed and path optimization.

4. **ILMPC (Zhao et al. 2025)** — "Improving Drone Racing Through Iterative Learning MPC" (arXiv:2508.01103)
   - Adaptive cost + shifted safe set for iterative trajectory improvement.
   - 6-60% lap time improvement. Converges in 3-15 iterations.
   - Key insight: spatially varying cost (tight near gates, loose between gates).

## Research Consensus

**Strong consensus (4/4 papers):** Iterative correction based on previous trial errors converges rapidly (3-20 iterations) and eliminates systematic tracking errors. The key requirements are:
1. The trajectory must be repeatable (ours is — deterministic kinematic sim)
2. The learning rate must not overshoot (alpha < 1.0)
3. The correction should be smoothed to avoid high-frequency artifacts

**Approach selection:**
- Full ILMPC (Zhao 2025): requires MPC solver infrastructure (Acados, CasADi) — too complex for one iteration
- Spatial ILC (Lv 2023): optimizes SPEED, not POSITION — our speed profile is already near-optimal via TOPP
- Schoellig ILC (2012): optimizes POSITION feedforward — directly targets our problem (position tracking error)
- QPGP-ILC (2026): overkill for deterministic sim

**Selected approach: P-type position ILC (simplified Schoellig 2012)**

## Proposed Implementation

### Algorithm: Offline Position-ILC

After the trajectory is generated, apply iterative position correction:

```
for j in 1..max_iterations:
    actual_positions = kinematic_sim(trajectory_j)
    for each timestep k:
        error_k = trajectory_j.position[k] - actual_positions[k]
        correction_k = alpha * error_k
        correction_k = smooth(correction_k, sigma=5)
        correction_k = clip(correction_k, max_magnitude=0.3)
        trajectory_{j+1}.position[k] = trajectory_j.position[k] + correction_k
    trajectory_{j+1}.velocity = finite_diff(trajectory_{j+1}.position)
    trajectory_{j+1}.acceleration = finite_diff(trajectory_{j+1}.velocity)

    if improvement < 1%: break
```

### Why This Is Fundamentally Different From Previous Approaches

All 24 previous iterations modified the trajectory BEFORE seeing how the controller tracks it:
- Racing line offsets → change waypoint positions
- Time allocation / inflation → change segment timing
- TOPP retiming → change speed profile
- Basin interpolation → blend between known solutions

This approach modifies the trajectory AFTER observing the tracking error, directly compensating for the controller's systematic lag. It's the first closed-loop optimization of the trajectory-controller pair.

### Expected Impact
- Schoellig 2012 achieved 87% error reduction on real quadrotors in 5 iterations
- Our sim is MORE favorable (deterministic, no random disturbances)
- Conservative estimate: 30-50% reduction in helix error floor
- gate-7: 0.327m → 0.20-0.23m
- avg error: 0.211m → 0.15-0.18m
- Race time: unchanged (ILC corrects position, not timing)

### Risk Assessment
- LOW RISK: ILC is a post-processing step; if it regresses, we simply skip it
- Gate passage: correction magnitude cap (0.3m) prevents trajectory from moving outside gates
- Smoothing prevents high-frequency artifacts in the corrected trajectory
- Velocity/acceleration recomputation maintains dynamic feasibility
