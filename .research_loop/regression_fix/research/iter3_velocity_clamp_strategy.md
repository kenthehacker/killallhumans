# Iter 3 Research: Polynomial Velocity Clamping Strategy

## Problem
Min-snap polynomial trajectories can exceed `max_velocity` at mid-segment points because the velocity constraint only applies to boundary conditions (start/end of each segment). For our race_01 config with `max_velocity=10.0`, the polynomial peaks at **16.84 m/s** — 68% over the limit.

## Research Queries
1. "How do minsnap polynomial trajectories enforce velocity constraints without losing smoothness"
2. "TOPP-RA time-optimal path parameterization with velocity and acceleration constraints quadrotor"
3. "Quadrotor trajectory reference scaling when physical limits are exceeded — feedforward vs feedback approach"

## Key Findings

### Approach 1: Post-hoc Velocity Clamping (Selected)
- Clamp velocity magnitude per sample point; scale acceleration and jerk by same ratio
- **Preserves position path** (smooth polynomial), only modifies derivatives
- Standard practice in ETH ASL `mav_trajectory_generation` and MATLAB UAV Toolbox
- Limitation: velocity is no longer exact derivative of position. Acceptable when controller uses position as primary reference and velocity/accel as feedforward
- Reference: Mellinger & Kumar (ICRA 2011), Richter et al. (ISRR 2013)

### Approach 2: Time Re-allocation (TOPP-RA style)
- Already partially implemented in `_topp_retime()` which adjusts segment times
- TOPPQuad (Mao, IROS 2024): forward-backward propagation with motor thrust constraints
- Our TOPP retimer uses curvature-based speed limits but doesn't enforce mid-segment polynomial overshoots
- A full TOPP-RA re-parameterization would be more correct but is a larger change
- Reference: Pham & Pham (2017), TOPPQuad (Mao et al., IROS 2024)

### Approach 3: Penalty-based Soft Constraints (MINCO / STORM)
- MINCO (ZJU FAST Lab, 2021): unconstrained optimization with differentiable penalty terms
- STORM (Zhang et al., 2025): spatial-temporal iterative optimization with LP constraints
- Too invasive for a single-iteration fix — would require rewriting the trajectory representation
- Reference: MINCO (Wang & Gao, 2021), STORM (Zhang et al., 2025)

### Approach 4: Planner-side Trajectory Modification (Drag-Aware)
- "Why Change Your Controller When You Can Change Your Planner" (Zhang et al., 2024)
- On CrazyFlie 2.0: modifying planner to account for drag reduces tracking error by 83%
- Quad-LCD (Srikanthan et al., 2025): learned cost function to prevent motor saturation
- Relevant to future iterations for CrazyFlie-specific tuning

## Decision
**Approach 1 (post-hoc clamp)** is selected for iter 3 because:
1. Minimal code change (add 6 lines to `_generate_trajectory`)
2. No change to trajectory timing or structure
3. Consistent with existing visual_demo velocity clamp (5 m/s)
4. Position path (the polynomial itself) is preserved — only feedforward derivatives change
5. The controller already uses position as primary reference; velocity/accel are feedforward aids

## Scaling Strategy
When clamping velocity from speed `s` to `max_v`:
- `scale = max_v / s`
- `velocity *= scale` (direction preserved, magnitude limited)
- `acceleration *= scale` (conservative: true time-rescaling would be scale², but scale is safer for feedforward)
- `jerk *= scale` (same rationale)

This is conservative — the controller gets slightly reduced feedforward, which is always safe. The alternative (scale² for accel, scale³ for jerk) would be more physically correct but risks under-exciting the feedforward at transition points.

## Sources
- [Mellinger & Kumar, ICRA 2011](https://ieeexplore.ieee.org/document/5980409/)
- [ETH ASL mav_trajectory_generation](https://github.com/ethz-asl/mav_trajectory_generation)
- [TOPPQuad (Mao et al., IROS 2024)](https://arxiv.org/abs/2309.11637)
- [Drag-Aware Trajectory Generation (Zhang et al., 2024)](https://arxiv.org/html/2401.04960v1)
- [Quad-LCD (Srikanthan et al., 2025)](https://arxiv.org/html/2505.10228)
- [STORM (Zhang et al., 2025)](https://arxiv.org/html/2503.03252v1)
- [MINCO (Wang & Gao, 2021)](https://github.com/ZJU-FAST-Lab/GCOPTER)
