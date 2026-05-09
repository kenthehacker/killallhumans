# Nonlinear Receding-Horizon Differential Game for Drone Racing

- **URL**: https://arxiv.org/abs/2502.01044
- **Authors**: Kijin Sung, Kenta Hoshino, Akihiko Honda, Takeya Shima, Toshiyuki Ohtsuka (Graduate School of Informatics, Kyoto University; Advanced Technology R&D Center, Mitsubishi Electric Corporation, Japan)
- **Year**: 2025

---

## Key Contribution

This paper introduces NRHDG (Nonlinear Receding-Horizon Differential Game), a real-time game-theoretic control framework for competitive drone racing along arbitrary smooth 3D paths. The two primary technical novelties are:

1. **Embedded projection-point dynamics**: A closed-form ODE for continuously tracking the drone's orthogonal projection onto a smooth 3D path, eliminating the need for iterative distance minimization at each control step. This makes path-following constraints tractable inside a real-time MPC solver.

2. **Zero-sum saddle-point game formulation**: Instead of fixing an opponent model (as standard NMPC does), NRHDG models both the ego drone and the opponent with identical full nonlinear dynamics and solves for the saddle-point — the strategy that is optimal assuming the adversary plays optimally. This directly handles worst-case opponent behavior without assumptions about opponent speed or trajectory.

The paper also contributes a unified **competitive potential function** that naturally encodes both overtaking and obstructing objectives and switches between them based on relative track position, with no explicit mode logic.

---

## Technical Approach

### Drone Dynamics

The quadrotor model uses a 13-state vector: 3D position, 3D velocity, 3D angular velocity, and quaternion attitude (4 components). Newton-Euler equations describe translational and rotational motion under thrust and torque inputs. The system is standard for aggressive flight control literature.

### Path-Following via Projection-Point Dynamics (Theorem 2.1)

The central technical result is the derivation of an ODE governing the evolution of the path parameter θ_d, which indexes the drone's orthogonal projection onto the reference curve r(θ). Rather than solving a nonlinear minimization at each timestep, the projection satisfies:

```
θ̇_d = [ṗ_d^T · (dr/dθ)] / [‖dr/dθ‖² + (r(θ_d) - p_d)^T · (d²r/dθ²)]
```

This derivation holds as long as the singularity condition is satisfied:

```
‖r(θ_d) - p_d‖ · ‖d²r/dθ²‖ < ‖dr/dθ‖²
```

In plain language: the product of (deviation distance) × (path curvature magnitude) must be less than the square of the path tangent magnitude. For paths parameterized by arc length this simplifies to: deviation × curvature < 1, i.e., the drone cannot be more than one radius-of-curvature away from the path.

The augmented state for path-following is 15-dimensional: standard 13 states plus path parameter θ_d and arc-length progress σ. Arc-length progress σ is accumulated by integrating ‖dr/dθ‖ · θ̇_d, providing a scalar measure of how far the drone has advanced.

### Stage Cost Design

The path-following stage cost L_PF combines:
- **Position deviation penalty**: Weighted squared distance from projection point to path (penalizes cross-track error)
- **Angular velocity damping**: Penalizes high angular rates (stabilization and energy)
- **Forward progress reward**: Negative term on σ̇ (maximizes speed along path)
- **Control regularization**: Penalizes deviation from hover thrust u_ref = mg/4

### Competitive Potential Function

The overtaking/obstructing objective is encoded in a single smooth potential G_O:

```
G_O(x̄_d, p_op, θ_op) = exp(-(θ_Δ - δ₁/α)²) · tanh(θ_Δ - δ₂) · β / (1 + γR²)
```

Where:
- `θ_Δ = θ_op - θ_d` (opponent's path parameter minus ego's)
- `R` = difference in path-deviation distances between the two drones
- `α, β, γ, δ₁, δ₂` are tunable shape parameters

The function peaks when the ego is directly behind the opponent (positive θ_Δ) and the opponent is close to the path (R ≈ 0), rewarding overtaking. It reverses when the ego is ahead, creating an obstructing pressure. This single function seamlessly covers both competitive roles without a discrete mode switch.

### NMPC Baseline

The NMPC controller uses a simplified opponent model: the opponent is assumed to travel at constant speed λ parallel to the reference path. This produces a 19-state system (13 ego + projected opponent position/velocity). While computationally lighter (19 vs. 30 states), it is blind to reactive opponent strategies.

### NRHDG Game Formulation

NRHDG treats the racing scenario as a zero-sum differential game:
- Ego minimizes combined cost: `L_D = [L_PF(ego) + G_O(ego leads)] - [L_PF(opponent) + G_O(opponent leads)]`
- Opponent simultaneously maximizes L_D (adversarial)
- Saddle-point condition: `min_{u_d} max_{u_op} L_D`

Both drones share identical full dynamics, making the 30-state joint system symmetric and principled. The receding-horizon formulation solves this game online at each control step.

---

## Results

### Simulation Scenarios

Two scenarios are evaluated:
1. **Overtaking**: Ego starts behind opponent on a 3D racing path; performance measured by how much path progress the ego achieves relative to the opponent.
2. **Obstructing**: Ego starts ahead; performance measured by the follower's progress under opposition.

### Key Findings

- NRHDG outperforms NMPC on both overtaking efficiency and obstructing capability
- Role transitions (leading → following and vice versa) occur naturally from the potential function without explicit mode logic
- Both controllers successfully navigate the full 3D path without crashing
- NRHDG's advantage is most pronounced when opponents react aggressively; NMPC's fixed-speed assumption fails to model reactive behavior

### Computational Considerations

The 30-state NRHDG system is more expensive than the 19-state NMPC. The paper does not report explicit wall-clock computation times or loop frequencies. Practical deployment would require an efficient saddle-point solver (e.g., continuation/GMRES or structured SQP). This is a notable gap in the paper's empirical validation.

---

## Relevance to Our System

Our system uses min-snap polynomial trajectory with TOPP retiming and a geometric SE(3) tracker (Lee et al.). The current pain point is gate-7, a high-curvature helix section with 0.284m tracking error.

### Direct Relevance

**Projection-point dynamics are directly applicable.** Our min-snap trajectory is a smooth 3D polynomial curve. The NRHDG projection ODE (Theorem 2.1) could replace our current trajectory sampling approach. Instead of evaluating the polynomial at fixed time steps, we could integrate the projection ODE alongside the drone state and always reference the nearest point on the curve. This gives:
- True cross-track error as a control input (not a time-lagged positional error)
- Automatic adaptation of "where we are on the path" under disturbances — critical for helical sections where time-indexed references diverge from actual position under tracking lag
- Progress-maximizing behavior (the σ reward term) rather than time-tracking behavior

**At high-curvature helix sections**, the standard time-indexed reference can drift badly: if the drone falls behind by 0.1s, the reference point races ahead along the helix, creating a large cross-track error that the controller chases. The projection-based formulation inherently avoids this by always computing error to the nearest path point.

### Indirect Relevance

The competitive game formulation (NRHDG) is not relevant to our current problem (single-drone time trials). However, if the AI Grand Prix evolves into head-to-head racing, the potential function design and saddle-point MPC structure would be directly relevant.

### Singularity Condition Check for Gate-7

The singularity condition `deviation × curvature < 1` must be checked for our helix geometry. If the helix has radius r_h and the drone's tracking error is ε, then curvature κ = 1/r_h, and the condition requires ε < r_h. For a 2m radius helix, this is satisfied as long as tracking error stays below 2m — well within our operating regime.

---

## Actionable Takeaways

1. **Implement projection-based path following in the MPC tracker.** Replace time-indexed reference lookups with the projection ODE (Theorem 2.1). Integrate θ_d as an additional state in the control loop. This is the highest-value takeaway.

2. **Add arc-length progress σ as a reward signal.** Rather than tracking a time schedule, maximize σ̇ directly. This is particularly effective for helical sections where maintaining speed along the curve is more important than hitting time waypoints.

3. **Replace cross-track error computation in the geometric tracker.** Current `mpc_tracker.py` likely computes position error to a time-indexed point. Switching to nearest-point projection would reduce gate-7 error.

4. **Parameterize the helix segment of the racing line separately.** The projection ODE requires a smooth curve r(θ) with accessible first and second derivatives. If using polynomial segments, store dr/dθ and d²r/dθ² analytically.

5. **Check singularity margins.** Evaluate ‖r(θ_d) - p_d‖ · ‖d²r/dθ²‖ along the helix segment in simulation. If this approaches ‖dr/dθ‖², add a curvature-aware speed reduction in the TOPP retiming pass.

6. **Competitive game formulation is future work.** Defer NRHDG multi-agent aspects until single-agent performance is maximized.

---

## Limitations & Caveats

1. **No real-world hardware experiments.** All results are simulation-only. Real-world aerodynamics, sensor noise, communication delay, and rotor dynamics are not validated.

2. **No computation time reported.** The 30-state saddle-point problem is significantly harder than standard NMPC. Real-time feasibility at 100+ Hz loop rates is unproven. Our system requires >100 Hz.

3. **Specific weight values not disclosed.** The stage cost weights (a₁...a₇, b) and potential function shape parameters (α, β, γ, δ₁, δ₂) are not given numerically. Re-tuning from scratch would be required for any implementation.

4. **Singularity condition is path-curvature dependent.** For very tight helices (small radius), the singularity bound tightens. The paper does not evaluate extreme curvature scenarios typical of competitive drone racing.

5. **Opponent model symmetry assumption.** Both drones use identical dynamics in NRHDG. Asymmetric opponent models (e.g., a slower opponent with different constraints) are not addressed.

6. **No comparison to RL-based baselines.** Swift and similar RL controllers achieve superhuman lap times; the paper benchmarks only against NMPC, leaving the competitive landscape relative to state-of-the-art unclear.

7. **Path parameterization requirements.** The projection ODE requires analytically differentiable r(θ) with d²r/dθ² available. Paths stored as lookup tables or splines would need smooth interpolation.

---

## Key Parameters / Constants

| Parameter | Description | Value |
|-----------|-------------|-------|
| State dim (ego) | Full quadrotor state | 13 |
| Augmented state dim | With path projection | 15 |
| NMPC joint state dim | 13 ego + simplified opponent | 19 |
| NRHDG joint state dim | 15 ego + 15 opponent | 30 |
| u_ref | Hover thrust per rotor | mg/4 |
| θ_Δ | Path parameter difference (opponent − ego) | continuous |
| Singularity condition | dev × curvature < tangent² | ‖r−p‖·‖r''‖ < ‖r'‖² |
| Projection ODE | θ̇_d formula | [ṗ_d·r'] / [‖r'‖² + (r−p)·r''] |
| Potential function G_O | Parameters | α, β, γ, δ₁, δ₂ (not numerically disclosed) |
| Constant opponent speed | NMPC assumption | λ (not numerically disclosed) |

### Key Equations for Implementation

**Projection ODE (embed as auxiliary state):**
```python
theta_dot = np.dot(p_dot, r_prime) / (
    np.dot(r_prime, r_prime) + np.dot(r - p, r_double_prime)
)
```

**Singularity check (add as safety monitor):**
```python
assert np.linalg.norm(r - p) * np.linalg.norm(r_double_prime) < np.dot(r_prime, r_prime)
```

**Arc-length progress reward (add to stage cost):**
```python
sigma_dot = np.linalg.norm(r_prime) * theta_dot  # reward: -a7 * sigma_dot
```
