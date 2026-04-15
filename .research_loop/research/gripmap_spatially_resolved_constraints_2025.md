# GripMap: An Efficient, Spatially Resolved Constraint Framework for Offline and Online Trajectory Planning in Autonomous Racing

- **URL**: https://arxiv.org/abs/2504.12115
- **Authors**: Frederik Werner, Ann-Kathrin Schwehn, Markus Lienkamp, Johannes Betz
- **Year**: 2025
- **Venue**: IEEE IV 2025

## Key Contribution

GripMap introduces a systematic framework for spatially-varying dynamic constraints in autonomous racing trajectory planning. The fundamental insight is that vehicle dynamic limits are not spatially invariant — grip conditions vary with location on the track. By encoding these variations into a Frenet-frame-indexed constraint map, trajectory optimizers can exploit location-specific knowledge to generate faster and safer trajectories. The framework achieves a 5.2% lap time improvement over spatially-invariant models with only 0.77% runtime overhead.

This is directly relevant to our drone racing system where we manually define segment-specific TOPP compression floors (helix: 0.72, S-turn: 0.65, easy: 0.59). GripMap provides the theoretical framework for what we're doing heuristically — spatially resolving dynamic constraints along the trajectory.

## Technical Approach

The racetrack is discretized in the Frenet frame (s, n) where s is arc length and n is lateral offset. Each cell (s_i, n_j) stores a scaling factor θ_ij that modulates the baseline vehicle dynamic constraints:

- a_x,max(s_i, n_j) = θ_ij · a_x,max,base(v, a_z)
- a_y,max(s_i, n_j) = θ_ij · a_y,max,base(v, a_z)

The scaling factors are initialized from surface grip measurements and refined through controller feedback. Access is O(1) via perfect hashing: i = floor(s/Δs), j = floor((n + n_offset)/Δn).

The framework integrates into both offline minimum-lap-time optimization (using optimal control with state constraints on velocity, lateral deviation, orientation, and accelerations) and online MPC-based planning (high-frequency sampling with feasibility evaluation).

Key design parameters:
- Longitudinal step size: Δs = s_max / s_dim
- Lateral step size: Δn = W_max / n_dim
- Scaling factors θ_ij ∈ (0, 1] where 1.0 = full grip, lower = reduced grip

## Results

- 5.2% lap time improvement in real-world racing scenarios vs spatially-uniform models
- 0.77% runtime overhead from constraint lookups (negligible)
- Successfully prevented spin-outs during overtaking maneuvers on varying grip surfaces
- Demonstrated on both offline and online planning pipelines

## Relevance to Our System

**Highly relevant.** Our TOPP retimer uses manually-defined compression floors for different track sections:
- Helix segments: 0.72 floor (identified binding in iter 35-36)
- S-turn segments: 0.65 floor (identified binding for gate-3)
- Easy segments: 0.59 floor

This is essentially a 1D GripMap along the trajectory arc length, where our "scaling factors" are the TOPP compression floors. The GripMap framework validates our approach of spatially-varying constraints and suggests:

1. **Joint optimization**: Instead of tuning floors independently per section, optimize the entire spatial constraint profile together
2. **Controller feedback**: Use tracking error from previous runs to refine constraint values (similar to our ILC approach but applied to constraints instead of trajectory)
3. **Continuous constraints**: Move from 3 discrete floor levels to a smooth constraint profile along the trajectory

## Actionable Takeaways

1. Our section-specific TOPP floors are a valid approximation of spatially-varying constraints — confirmed by GripMap's theoretical framework
2. Joint optimization of multiple floor parameters (S-turn + helix) is theoretically well-grounded
3. Future iteration: replace discrete floors with continuous spatial constraint profile using controller feedback (akin to GripMap's θ_ij refinement)
4. The 5.2% improvement from spatially-resolved constraints validates that significant gains are available from spatial constraint tuning

## Limitations & Caveats

- Designed for car racing (friction-limited), not quadrotor racing (thrust-limited)
- The scaling factor framework assumes a point-mass vehicle model — our system uses polynomial trajectory optimization
- Real-time constraint updates require persistent state between laps, which we approximate with ILC

## Key Parameters / Constants

- Scaling factors θ_ij ∈ (0, 1]: maps to our TOPP compression floors
- O(1) constraint lookup via perfect hashing
- Runtime overhead: 0.77% (negligible for our 7600 Hz loop)
- 5.2% lap time improvement from spatial constraint resolution
