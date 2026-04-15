# A Computationally Efficient and Human Implementable Minimum-lap-time Control Policy for Energy-limited Race Cars

- **URL**: https://arxiv.org/abs/2603.02339
- **Authors**: Erik van den Eshof, Wytze de Vries, Mauro Salazar
- **Year**: 2026
- **Venue**: IEEE ITSC 2026

## Key Contribution

This paper derives analytically optimal lap-time policies for energy-limited racing by decomposing the optimal control problem into a sequence of boundary-value problems. The key insight is that the optimal speed policy has a bang-bang structure: at any point on the track, the driver should be either at full throttle, coasting, or maximum braking. This structure enables decomposition of the continuous optimization into discrete segments separated by corner apexes, where curvature-dependent grip limits trigger costate jumps.

The computational approach reduces solve time from seconds to milliseconds while maintaining global optimality — a critical property for iterative parameter sweeps like our TOPP floor tuning.

## Technical Approach

The problem minimizes lap time: ∫(1/v(s))ds with two states — kinetic energy E_kin and cumulative battery energy E_b — and controls for motor force F_m and braking force F_brk.

The optimal policy exhibits three phases based on costate ratio λ_kin/λ_b:
1. **Full throttle** when λ_kin/λ_b < -1/η⁺
2. **Coasting** when -1/η⁺ < λ_kin/λ_b < -η⁻
3. **Maximum regenerative braking** when λ_kin/λ_b > -η⁻

**Curvature-dependent speed constraints** use friction ellipse: F_grip± = ±μ_x√[F_z² - (F_y/μ_y)²], where lateral force F_y = 2κE_kin depends on track curvature κ. At corner apexes, all tire grip is consumed by lateral forces, creating instantaneous costate jumps.

The algorithm:
1. Identifies costate jump points at corner apexes (curvature maxima)
2. Uses bisection to determine optimal costate magnitudes between jumps
3. Nests algorithms to find battery costate satisfying energy constraints

## Results

- Zolder circuit: only 7 instantaneous coasting cues per lap
- Computation time: milliseconds vs seconds for direct NLP
- Maintains global optimality (proven mathematically)
- Optimal energy deployment avoids cornering phases entirely

## Relevance to Our System

**Moderately relevant.** While this paper addresses car racing (energy-limited, friction-limited), the decomposition principle applies to our system:

1. **Segment-wise optimization**: The idea of decomposing the continuous speed profile problem at "corner apexes" (curvature maxima) into independent boundary-value problems mirrors our approach of identifying binding constraints at section boundaries and optimizing each section's TOPP floor
2. **Curvature-dependent constraints**: The friction ellipse formulation (F_y = 2κE_kin) establishes that speed limits at high-curvature sections are the primary binding constraints — analogous to our finding that S-turn and helix TOPP floors are binding
3. **Bang-bang structure**: The optimal speed policy switches between full speed and grip-limited speed, similar to our TOPP retimer's forward-backward pass where segments are either floor-limited or curvature-limited

## Actionable Takeaways

1. **Corner apex decomposition**: Treat gate-3 S-turn as an "apex" where curvature is maximal, and optimize the approach/exit speed profile independently
2. **Co-state ratio insight**: The trade-off between time (kinetic energy costate) and constraint satisfaction (accuracy costate) can be formalized — our Pareto sweep is an empirical version of this
3. **Bisection for optimal parameters**: The bisection method for finding optimal costate values suggests using binary search rather than grid sweep for floor optimization — potentially faster convergence
4. **Section independence**: If apex-based decomposition holds, S-turn floor changes should be independent of helix floor changes (to verify)

## Limitations & Caveats

- Energy-limited formulation doesn't directly apply to quadrotors (thrust-limited, not energy-limited)
- Friction ellipse model differs from quadrotor thrust/attitude dynamics
- The analytically tractable structure relies on bang-bang optimality which may not hold for our polynomial trajectory formulation

## Key Parameters / Constants

- Curvature-dependent speed limit: v_max(s) = √(μ·g / κ(s)) (friction-limited)
- Bang-bang structure: optimal policy has at most 3 distinct phases per segment
- Bisection convergence: O(log(1/ε)) iterations for ε precision
- Zolder circuit: 7 switching points per lap (one per significant curvature peak)
