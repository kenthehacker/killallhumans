# Robust Trajectory Generation and Control for Quadrotor with FOV Control Barrier Certification

- **URL**: https://arxiv.org/abs/2502.01009
- **Authors**: Lishuo Pan, Mattia Catellani, Lorenzo Sabattini, Nora Ayanian
- **Year**: 2025
- **Venue**: IEEE Robotics and Automation Letters (RA-L), accepted November 2025; arXiv preprint February 2025

---

## Key Contribution

Pan et al. present a unified real-time framework that jointly handles trajectory generation and safety-certified control for quadrotors (and more generally, multi-robot systems) operating under field-of-view (FOV) constraints. The central contribution is an MPC-CBF optimizer: a model predictive controller whose feasibility is formally certified by High-Order Control Barrier Functions (HOCBFs) that encode both the geometric FOV cone and inter-agent safety distances as hard constraints. Unlike prior work that either plans FOV-safe trajectories offline or enforces safety reactively at the controller level, this paper shows how to bake the FOV constraint directly into the Bézier-curve trajectory generator in a receding-horizon loop, with forward-invariance guarantees (Theorem 1 of the paper) that hold even under sensing noise and actuation delays.

The practical consequence is that the system never needs an iterative post-processing step to recover FOV feasibility: the HOCBF constraint is imposed during trajectory generation itself. This is directly relevant to our architecture, where `_relax_for_fov()` is a separate post-optimization pass that currently adds ~6 seconds to a 14-second base trajectory. The paper's approach would eliminate that pass by encoding the FOV cone as a constraint inside the trajectory optimizer rather than as a penalty outside it.

---

## Technical Approach

### System Model

The quadrotor (and each robot in the multi-agent case) is modeled as a double integrator in position and yaw:

```
x = [r; φ; ṙ; φ̇] ∈ ℝ⁸
ẋ = Ax + Bu,   A = [0, I; 0, 0],   B = [0; I]
u = [u_r; u_φ] ∈ ℝ⁴  (translational + yaw acceleration)
```

This is appropriate for waypoint-level planning in differential-flat quadrotor frameworks; the paper is agnostic to the low-level attitude controller.

### FOV and Safety as Control Barrier Functions

The sensing region is modeled as a truncated spherical sector parameterized by:
- Maximum perception range R_s
- Minimum safety distance D_s
- Horizontal FOV half-angle β_H/2

The safety-and-range CBF is:

```
b_sr(r_ij) = [x_ij, y_ij; -x_ij, -y_ij]^T [x_ij; y_ij] + [-D_s²; R_s²]
```

The FOV CBF for a camera with horizontal angle β_H is:

```
b_fov = [tan(β_H/2),  1; tan(β_H/2), -1] [x_ij; y_ij]    (for β_H ∈ [0, π))
b_fov = [tan(π−β_H/2), sign(y_ij)]^T [x_ij; y_ij]         (for β_H ∈ (π, 2π))
```

These are second-relative-degree constraints (relative degree q=2 because control input appears only in the second derivative of the barrier function). The paper therefore lifts them into High-Order CBFs (HOCBFs), using extended class-K functions α₁(b) = γ₁ b^(2μ+1) and α₂(ψ₁) = γ₂ ψ₁^(2μ+1). The resulting HOCBF constraint is:

```
L_f² b(·) + L_g L_f b(·) u
  + (2μ+1) γ₁ b^(2μ)(·) L_f b(·)
  + γ₂ (L_f b(·) + γ₁ b^(2μ+1)(·))^(2μ+1) ≥ 0
```

The parameters γ₁, γ₂ ≥ 0 and integer μ ≥ 0 are tunable. A key finding: the HOCBF with odd-degree class-K functions enables safe recovery even after FOV loss (the barrier value goes negative), because the odd-power terms retain their sign and continue to push the system back toward the safe set. This is the formal basis of Theorem 1 (forward invariance + recovery).

### Trajectory Representation: Bézier Curves

Trajectories are piecewise Bézier curves of degree h=3 with P=3 pieces, each of duration τ_i=0.5 s. A single piece is:

```
f_i(t) = Σ_{v=0}^{h} u_{i,v} B_{i,v}^h,
B_v^h(t) = C(h,v) (t/τ)^v (1−t/τ)^{h−v}
```

The control points u_{i,v} are the decision variables. This representation has two important properties:
1. Closed-form derivatives at any order (needed for evaluating CBF Lie derivatives analytically).
2. The trajectory is bounded within the convex hull of control points (useful for collision checking).

Continuity constraints enforce C=3 smoothness (position, velocity, acceleration, jerk) across piece boundaries.

### MPC-CBF Optimization

The full continuous-time problem is:

```
min_{U} J_cost
s.t.
  ẋ(t) = Ax(t) + Bu(t)                                  (dynamics)
  d^j f(0)/dt^j = d^j r(t₀)/dt^j,  j=0,...,C          (initial conditions)
  f continuous up to derivative C                         (smoothness)
  A^cbf u(t) + b^cbf(r̂_ij(t|t₀)) ≥ 0  ∀t, ∀j         (HOCBF)
  a_min ≤ u(t) ≤ a_max                                   (actuator limits)
  v_min ≤ v(t) ≤ v_max                                   (speed limits)
```

The continuous HOCBF constraints are discretized at intervals δ=0.1 s (K=round(τ/δ) points per piece). This converts the infinite-dimensional problem into a finite QP. The cost splits into three components:

- **J_goal**: penalize endpoint deviation from desired waypoint, weight ω_k=10 for circle paths, 300 for tight formations.
- **J_effort**: integral of squared derivatives up to order C, θ_j=1 for all j (unweighted min-snap analog).
- **J_prior**: slack cost for distant agents that temporarily exit FOV, ξ_j = Ω · γ_s^j with Ω=1000 and γ_s ∈ [0.1, 0.2].

The nonlinearity in HOCBF constraints (relative to the quadratic Bézier cost) is handled by Sequential Quadratic Programming (SQP) with V=2 iterations. Neighbor positions in future time steps are predicted from the previous SQP iterate, producing linear surrogate constraints that make each inner iteration a standard QP.

### Key Algorithmic Detail: Constraint Horizon Truncation

The HOCBF constraints are enforced only over the first K_r=2 time steps of the prediction horizon per SQP iteration (not over all K steps). This dramatically reduces problem size while maintaining the safety guarantee, because the receding horizon re-solves the problem every δ seconds. This is the key to the sub-36 ms solve times reported for 10-robot scenarios.

### Slack Variables for Infeasibility Recovery

When the FOV constraint would make the problem infeasible (e.g., target temporarily behind the drone), the optimizer introduces slack variables ε_j ≥ 0 that relax the HOCBF constraints with a large penalty Ω=1000. This ensures the problem always has a solution. The HOCBF theory then guarantees the system steers itself back toward the safe set as the trajectory replans.

---

## Results

### Simulation (multi-robot circle and formation scenarios)

All experiments run in 2D (x-y plane) with yaw control, using camera β_H = 2π/3 (120°, matching typical racing cameras), 4π/3, and 2π (omnidirectional).

**Circle formation (5–10 robots, β_H = 2π/3):**
- MPC-CBF success rate: ~65–80%, vs. ~30–50% for the baseline (greedy controller without CBFs).
- Percentage of neighbors within FOV: consistently >90%.
- Makespan: 20–40 s depending on configuration.

**Formation (2–10 robots):**
- Success rate: 100% at 2 robots, 85–90% at 10 robots.
- Neighbors in FOV: >60% maintained at all robot counts.

**Computational performance (200-iteration average):**

| Robots | MPC-CBF (ms) | Baseline (ms) |
|--------|-------------|---------------|
| 2      | 5.87        | 1.06          |
| 5      | ~18         | ~5            |
| 10     | 35.98       | 11.16         |

Single-robot scenarios (relevant to our case) solve in approximately 5–6 ms.

**Sensitivity to sample interval δ (Table II, 2-robot circle):**

| δ (s)  | Neighbors in FOV | Runtime (ms) |
|--------|-----------------|--------------|
| 0.05   | 97.86%          | 118.17       |
| 0.10   | 98.01%          | 68.98        |
| 0.20   | 97.98%          | 57.57        |

This shows that coarser sampling (δ=0.1 s) achieves nearly identical FOV maintenance while cutting runtime by ~40%.

**Robustness to delay and noise (Table III):**

| Condition         | FOV (%) | Success |
|-------------------|---------|---------|
| Delay 0.05 s      | 91.52%  | 100%    |
| Delay 0.10 s      | 84.85%  | 100%    |
| Delay 0.20 s      | 74.23%  | 100%    |
| Noise σ=0.1       | 98.17%  | 100%    |
| Noise σ=0.2       | 97.29%  | 100%    |

Robustness to noise is very high (success 100% at noise levels well above typical racing sensors). Robustness to delay degrades more noticeably above 0.1 s.

### Physical Experiments

Two-robot UAV experiment on PX4/Jetson Xavier hardware:
- Camera: RealSense D435 with β_H ≈ π/4 (45°, much narrower than simulation).
- Motion capture: Vicon (ground-truth state).
- Result: "Maintained visual contact constraints throughout flight" (qualitative, no detailed numbers in the abstract-accessible portion).

---

## Relevance to Our System

Our system's primary bottleneck is `_relax_for_fov()` in `/Users/kenichi.matsuo/Personal/killallhumans/planning/trajectory_optimizer.py`. This method currently:

1. Runs the full min-snap optimizer to completion.
2. Evaluates the geometric FOV penalty via `add_fov_constraints()`.
3. If penalty > 1.0, identifies high-curvature segments and multiplies their times by 1.1.
4. Repeats up to 5 times.

This adds ~6 seconds to a 14-second base trajectory (from `state.json` diagnostic: `fov_penalty_raw: 14727`, root cause confirmed as FOV relaxation dominating race time). The penalty threshold (0.5 to break early) and the 1.1× multiplier compound across iterations, causing the trajectory to become conservative even when only a few segments are problematic.

**The paper's HOCBF approach directly addresses this architectural flaw.** Instead of FOV being an after-the-fact penalty that triggers a separate relaxation loop, the FOV constraint would be embedded inside the trajectory generator as a hard constraint during optimization. The trajectory that emerges from the optimizer is already FOV-feasible by construction, and no post-processing pass is needed.

Specific modules affected:

- **`planning/trajectory_optimizer.py`**: The `_relax_for_fov()` method and `add_fov_constraints()` penalty would be replaced by HOCBF constraints embedded in the QP/SQP problem. The `FOVConfig` class parameters (horizontal_fov_rad, vertical_fov_rad, margin_fraction, penalty_weight) would map directly to β_H, β_V, and slack cost Ω.
- **`planning/trajectory_optimizer.py` (optimize method)**: The current post-optimization check (`if fov_penalty > 1.0: segment_times = self._relax_for_fov(...)`) would be removed entirely.
- **`control/mpc_tracker.py`**: If the tracker is upgraded to receding-horizon MPC, HOCBF constraints can also be added at the tracker level for real-time safety during execution, complementing the offline plan.

Our case is simpler than the paper's: we have a single drone tracking a pre-known gate sequence (not estimating neighbor positions). This means:
- No particle filter is needed.
- No slack variables for distant agents — only for the single upcoming gate.
- The "target" positions (gates) are known exactly at planning time, not estimated.
- The HOCBF constraints reduce to purely geometric angular constraints on the camera bearing to the next gate center.

The single-robot solve time (~5–6 ms) is well within our 10 ms budget at 100 Hz loop frequency.

---

## Actionable Takeaways

1. **Replace `_relax_for_fov()` with inline HOCBF constraints in the trajectory optimizer.** Instead of a post-processing loop, formulate the FOV half-angle bound as a second-relative-degree safety constraint and impose it at discrete time steps (δ=0.1 s recommended) within the existing L-BFGS or QP solve. This eliminates the 6-second overhead entirely.

2. **Use the HOCBF formulation with odd-degree class-K functions (μ=1 suggested).** The odd-power nonlinearity is what enables recovery after FOV loss (gate temporarily behind camera during a sharp turn). With even-degree functions the barrier is not recoverable once violated; with odd degrees, the CBF continues to push back toward the safe set. Use γ₁=γ₂=1.0 as initial values (paper does not state exact values, but the sensitivity analysis shows the algorithm is not highly sensitive).

3. **Add slack variables with penalty Ω=1000 for infeasibility.** During very sharp turns, FOV feasibility may be transiently unachievable. Slack variables allow the optimizer to always find a solution and gracefully recover. This is safer than the current approach of inflating segment times, which changes the trajectory shape globally.

4. **Switch to Bézier curve trajectory representation (degree h=3, P segments).** The Bézier basis provides closed-form Lie derivatives needed for the HOCBF constraints. Our current polynomial min-snap representation also supports closed-form derivatives, but verifying compatibility with the HOCBF Lie derivative computation is required. The paper's formulation assumes the Bernstein polynomial structure for this reason.

5. **Enforce HOCBF constraints at only K_r=2 time steps per SQP iteration, not over the full horizon.** This matches the paper's key computational trick for keeping solve times under 36 ms even for 10 robots. For our single-drone case, this will keep overhead well under 6 ms.

6. **Use SQP with V=2 iterations.** The paper demonstrates that 2 SQP iterations is sufficient for convergence in practice. More iterations give diminishing returns. Initialize with the unconstrained trajectory and iterate twice.

7. **Decouple horizontal and vertical FOV constraints.** Our current `add_fov_constraints()` checks both azimuth and elevation separately with `half_h` and `half_v`. The HOCBF formulation in the paper covers horizontal FOV explicitly; a symmetric vertical constraint can be added as a second HOCBF with β_V substituted for β_H. Apply both independently.

8. **Tune the sample interval δ=0.1 s (not 0.05 s).** Table II shows negligible difference in FOV maintenance between δ=0.05 and δ=0.10, while runtime drops by ~40%. Use δ=0.10 s for our optimizer.

9. **For real-time execution (MPC tracker), use a short receding horizon of K=10 steps at δ=0.1 s (1 second lookahead).** This is consistent with the paper's recommendation and our 100 Hz control loop requirement.

10. **Pre-compute CBF constraint matrices for each gate.** Since gate positions are known at planning time, the matrices A^cbf and b^cbf can be computed once per gate during trajectory initialization rather than at each optimization iteration, reducing per-iteration overhead further.

---

## Limitations & Caveats

**Multi-robot vs. single-robot setting.** The paper is primarily motivated by multi-robot coordination (maintaining visibility of neighbors), not single-drone gate racing. The FOV constraint formulation is the same geometrically, but the "target" in our case is a fixed gate position, not a moving neighbor whose position must be estimated. This actually makes our problem easier (no particle filter, no prediction uncertainty), but we should verify the HOCBF formulation still applies when the target is stationary relative to the world frame rather than another vehicle.

**2D vs. 3D.** The experiments are conducted in the 2D horizontal plane with yaw control only. The physical experiment uses 3D flight but with limited data reported. Our races involve aggressive 3D maneuvers with significant pitch and roll; the HOCBF barrier function for the 3D camera frustum will be more complex than the 2D wedge constraints shown in the paper. The elevation angle constraint is analogous but requires separate treatment.

**Differential-flatness coupling.** Our system estimates camera body-frame orientation via differential flatness (thrust direction from acceleration). The HOCBF requires the camera bearing to the gate expressed in the body frame, which depends on attitude. For aggressive maneuvers, the differential flatness approximation may introduce errors in the barrier function evaluation, potentially causing the constraint to be slightly violated even when nominally satisfied.

**Physical experiment is qualitative.** The two-robot hardware experiment does not provide quantitative tracking error or FOV violation rates. The reported "maintained visual contact" result is qualitative. The computational results are simulation-only.

**SQP convergence with 2 iterations.** The paper reports V=2 is sufficient empirically but does not prove convergence guarantees for the outer SQP loop (only for the inner CBF constraint). In practice, with tightly coupled FOV and speed constraints in a racing trajectory, more iterations may occasionally be needed.

**Bézier degree h=3.** The paper uses cubic Bézier curves with P=3 pieces. Our min-snap optimizer uses higher-degree polynomials (typically degree 5–7) for smoother jerk/snap profiles. Adapting the HOCBF to higher-degree polynomials is straightforward (same Bernstein structure) but requires re-deriving the Lie derivative expressions for the higher-degree basis.

**Racing-specific constraints not covered.** The paper does not address minimum gate-passage speed, maximum lateral g-load through gates, or gate sequencing logic. These remain our responsibility via `gate_sequencing/sequencer.py` and `planning/racing_line.py`.

**Slack penalty may be insufficient for hard racing constraints.** Using Ω=1000 for slack cost works in navigation contexts where slight FOV loss is acceptable. In drone racing, losing sight of a gate may cause gate-miss events. If we use slack variables, we should also keep the geometric FOV penalty as a signal to the gate sequencer that an alternate trajectory segment should be tried.

---

## Key Parameters / Constants

These are specific numerical values from the paper that can be used directly or as initialization points:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Bézier degree h | 3 | Cubic curves per piece |
| Number of pieces P | 3 | Pieces per planning horizon |
| Piece duration τ_i | 0.5 s | Time per Bézier piece |
| Continuity order C | 3 | C³ smoothness across pieces |
| Discrete sample interval δ | 0.1 s | HOCBF constraint evaluation rate |
| HOCBF constraint horizon K_r | 2 | Steps with enforced CBF per SQP iter |
| SQP iterations V | 2 | Outer nonlinear iterations |
| Slack penalty weight Ω | 1000 | Cost for FOV constraint violations |
| Slack decay factor γ_s | 0.1–0.2 | Per-agent priority weighting |
| Goal cost weight ω_k | 10 (circle), 300 (formation) | Waypoint tracking weight |
| Effort cost weights θ_j | 1 (all orders) | Min-snap analog weight |
| HOCBF exponent μ | 0 (implied, standard HOCBF) | Odd-power class-K parameter |
| CBF class-K gains γ₁, γ₂ | Not stated explicitly; start with 1.0 | Recovery speed tuning |
| Acceleration limits | ±10 m/s² (xy) | Control input bounds |
| Velocity limits (circle) | ±3 m/s | Speed bounds during navigation |
| Velocity limits (formation) | ±0.5 m/s | Speed bounds during tight formation |
| Particle filter particles N_p | 100 | (Not needed for single-drone case) |
| Process covariance | 0.25 I | (Neighbor estimation; not needed) |
| Measurement covariance R_m | 0.05 I | (Neighbor estimation; not needed) |
| Physical experiment FOV β_H | ~π/4 (45°) | RealSense D435 horizontal FOV |
| Simulation FOV β_H | 2π/3 (120°) | Default simulation camera |
| Safety distance D_s | Not stated (scenario-dependent) | Minimum inter-agent distance |
| Sensing range R_s | Not stated (scenario-dependent) | Maximum detection range |

**Key insight on sample interval:** δ=0.1 s gives ~98% FOV maintenance at ~69 ms total solve time for a 2-robot case. For a single drone, expect approximately 5–6 ms solve time — well within our 10 ms per-loop budget.

**Key insight on HOCBF gains:** The paper does not provide explicit γ₁, γ₂ values; they are tuned per scenario. Start with γ₁=γ₂=1.0 and increase if FOV recovery is too slow or decrease if the constraint is too aggressive and causes infeasibility.
