# On the Performance of Jerk-Constrained Time-Optimal Trajectory Planning
- **URL**: https://arxiv.org/abs/2404.07889
- **Year**: 2024
- **Venue**: IEEE International Conference on Robotics and Automation (ICRA) 2024
- **Authors**: Not extracted from abstract page, but published under CC BY 4.0

---

## Key Contribution

This paper presents TOTP3, a time-optimal trajectory planning algorithm that extends classical Time-Optimal Path Parameterization (TOPP) to enforce third-order (jerk) constraints, i.e., limits on the rate of change of acceleration. Classical TOPP and TOPP-RA handle velocity (first-order) and acceleration/torque (second-order) constraints but produce trajectories with discontinuous accelerations — abrupt "bang-bang" transitions between maximum-acceleration and maximum-deceleration phases. These discontinuities are trackable in theory but cause significant overshoot, peak power spikes, and actuator wear in practice.

The core technical challenge is that jerk constraints introduce a non-convex term in the standard path-parameterization formulation. The authors resolve this by proving that the denominator of the jerk expression is a convex function of the optimization variables, then applying a conservative first-order (linear) approximation at the current iterate. The resulting Sequential Linear Program (SLP) is solved iteratively until convergence. This guarantees constraint satisfaction (the linearization is conservative, so the true jerk constraint is never violated) while remaining computationally tractable — the authors report 7.5 ms average solve time on a 7-DOF industrial arm, which falls within their 10 ms real-time budget.

The practical outcome is that jerk-limited trajectories are demonstrably more trackable and energy-efficient on real hardware, with only a 3–5% increase in motion duration compared to unconstrained TOPP.

---

## Technical Approach

### Path Parameterization Foundation

TOTP3 follows the standard path-parameterization approach: given a geometric path q(s) ∈ ℝⁿ parameterized by arc length s ∈ [0, 1], find a time-scaling function s(t) that minimizes total traversal time T subject to kinematic/dynamic constraints. The optimization variable is discretized as xₖ = ṡₖ² at N path samples.

Joint velocities and accelerations are expressed in terms of path derivatives and the scaling:

    q̇(s) = q'(s) ṡ
    q̈(s) = q''(s) ṡ² + q'(s) s̈

where s̈ is approximated by finite differences of ṡ² (the optimization variable).

### Constraint Hierarchy

Three levels of constraints are formulated as linear inequalities in xₖ:

**First-order (velocity):**

    ωₖ xₖ ≤ νₖ

**Second-order (acceleration/torque):**

    αₖ⁰ xₖ + αₖ¹ xₖ₊₁ ≤ βₖ

**Third-order (jerk) — the novel contribution:**

    γₖ⁰ xₖ + γₖ¹ xₖ₊₁ + γₖ² xₖ₊₂ ≤ ηₖ

The jerk at path sample k is:

    q̈̇ₖ = (jₖ² xₖ₊₂ + jₖ¹ xₖ₊₁ + jₖ⁰ xₖ) / hₖ(x)

where the denominator hₖ is a nonlinear function of three adjacent optimization variables:

    hₖ(x) = Δₖ₊₁ / (√xₖ₊₂ + √xₖ₊₁) + Δₖ / (√xₖ₊₁ + √xₖ)

(Δₖ = sₖ₊₁ − sₖ is the fixed path-discretization step.)

### Convexity Proof and Linearization

The key result (proved in their Appendix) is that hₖ(x) is a convex function of x. This means:

    hₖ(x) ≥ h̄ₖ + ∂hₖ/∂x|_{x=x̄} · (x − x̄)

i.e., the first-order Taylor expansion at any nominal point x̄ provides a global lower bound. Substituting this conservative approximation into the jerk constraint gives a linear inequality that is guaranteed to be satisfied whenever the true (nonlinear) jerk constraint is satisfied. The conservatism ensures no constraint violations during the SLP iterations.

### SLP Algorithm (Algorithm 1)

The iterative procedure is:

1. Initialize x̄ with a feasible solution (e.g., from a second-order TOPP solve).
2. Compute cost gradient ∇f(x̄) and constraint matrices A³(x̄), b³(x̄) from the linearization.
3. Solve the LP: min cᵀx subject to A¹x ≤ b¹, A²x ≤ b², A³(x̄)x ≤ b³(x̄), x ≥ 0.
4. Check convergence: if ‖x − x̄‖ < ε, stop.
5. Set x̄ ← x and go to step 2.

**Objective function** (same as TOPP-RA):

    f(x) = Σₖ (sₖ₊₁ − sₖ) / (√xₖ + √xₖ₊₁)

which approximates total traversal time T.

### Matrix Structure

- A¹: diagonal, N−1 constraints
- A²: block tridiagonal, 2(N−1) constraints (forward/backward acceleration bounds)
- A³: three-diagonal bands, 2(N−2) constraints (forward/backward jerk bounds)

The LP is therefore sparse and solved efficiently with a standard LP solver. Warm-starting from the previous SLP iterate is essential for reaching the 7.5 ms average solve time.

---

## Results

### Hardware Platform

- 7-DOF manipulator: Kawasaki RS020N 6-DOF arm + 1 additional revolute joint
- Multiple representative industrial motions tested ("Front Place Motion," "Side Place Motion," etc.)

### Jerk Limit Range Tested

- 100–1000 rad/s³ across different test configurations

### Quantitative Performance (Front Place Motion, representative results)

| Metric | Without Jerk Constraints | With Jerk Constraints | Reduction |
|--------|--------------------------|----------------------|-----------|
| Peak power consumption | Baseline | ~25% lower | −25% |
| RMS torque (primary axis) | 3624 Nm | 1504 Nm | −59% |
| RMS torque (secondary axis) | 795 Nm | 267 Nm | −66% |
| Motion duration increase | — | +3–5% | +3–5% |

### Computational Timing

| Algorithm | Mean solve time | Std dev |
|-----------|----------------|---------|
| TOTP3 (with jerk, SLP) | 7.533 ms | ±5.848 ms |
| TOPP-RA (no jerk, baseline) | 0.273 ms | ±0.089 ms |

TOTP3 is ~28× slower than TOPP-RA but still meets the 10 ms real-time threshold for industrial applications. Warm-starting from the second-order TOPP-RA solution is essential for this.

### Tracking Quality

The paper provides visual evidence (figure of joint trajectories) that without jerk constraints, robots exhibit substantial overshoot at acceleration reversal points. With jerk constraints, the trajectory remains within ±0.1° of the reference throughout the motion. The paper states jerk constraints "significantly smoothed the motions, resulting in differences in generated torque and power," and that without them the robot "deviated significantly from desired paths during rapid acceleration changes."

### Time Penalty Sensitivity

Across multiple task variations with 1000 rad/s³ jerk limits, the motion time increase was consistently 3–5%. At tighter limits (100 rad/s³), the time penalty increased (not quantified precisely for all cases), but the authors emphasize that the 1000 rad/s³ case is representative of typical industrial settings.

---

## Relevance to Our System

Our drone racing stack's TOPP-RA-style retimer (`planning/trajectory_optimizer.py`, `_topp_retime()`) uses a forward-backward propagation that enforces velocity and centripetal acceleration limits but does **not** enforce jerk continuity. The result is that the speed profile can have abrupt acceleration transitions at segment boundaries — the exact class of discontinuities this paper targets.

**Direct applicability to gate-3 S-turn (0.247m tracking error):**

The S-turn section is a chicane where the drone must rapidly reverse lateral acceleration. Without jerk constraints, the TOPP retimer can assign a fast speed through the S-turn inflection point that is nominally feasible (centripetal acceleration check passes at each individual waypoint) but requires an instantaneous jerk spike to achieve the required acceleration reversal. Our `max_jerk = 50.0 m/s³` in `DroneConstraints` is a stored parameter but our `_topp_retime()` implementation does NOT enforce this limit during the forward-backward sweep — it is only used in the polynomial segment generation phase. This is a gap.

**The 0.09s race-time recovery target:**

The paper shows jerk constraints add only 3–5% to motion time. For our ~12s estimated race time, that would be 0.36–0.60s added. However, the tracking error improvement could recover time by enabling faster segment execution without overshoot. If gate-3 tracking error drops from 0.247m to ~0.15m (matching well-tracked gates), the ILC Q-filter currently compensating for it could be relaxed, and the segment time floor (`max_compression_protected = 0.65`) could be lowered, potentially recovering the 0.09s target.

**Our current jerk parameter is unused in the retimer:**

```python
# DroneConstraints
max_jerk: float = 50.0  # m/s^3 — declared but not enforced in _topp_retime()
```

The `_topp_retime()` function computes `v_max_accel` from centripetal acceleration but has no jerk-based velocity limit. Adding a third constraint type analogous to the paper's A³ matrix would directly connect `max_jerk` to the forward-backward sweep.

**Speed profile smoothness vs. ILC interaction:**

Our current architecture uses ILC offsets to compensate for the tracking error that arises from un-jerk-constrained trajectories. If the trajectory itself were jerk-constrained, the ILC would converge faster (smoother error signal) and the Q-filter cutoff could be set higher (less lag), improving tracking responsiveness.

---

## Actionable Takeaways

1. **Implement jerk-constrained velocity limits in `_topp_retime()`**: At each waypoint k, after computing the curvature-based v_max, add a jerk-based constraint: v_max_jerk = (max_jerk × Δs)^(1/3) or derived from the paper's Eq. 10. Use the minimum of centripetal and jerk-limited velocities at each step of the forward-backward sweep.

2. **Enforce the existing `max_jerk = 50.0 m/s³` parameter**: Connect `DroneConstraints.max_jerk` to the TOPP retimer. This is a one-line addition at the curvature-speed conversion step in `_topp_retime()`. Since 50 m/s³ is already set in our constraints, no parameter tuning is needed — just enforcement.

3. **Target S-turn inflection points specifically**: The paper's A³ matrix is tridiagonal — it links three adjacent path samples. In our waypoint-based retimer, the S-turn inflection (between gate-3 and gate-4) is the region where acceleration must reverse. Apply a tighter jerk ceiling at that segment boundary rather than globally.

4. **Use warm-starting from current TOPP output**: The paper shows warm-starting TOTP3 from the TOPP-RA solution enables convergence in ~7.5ms. Our retimer already produces a segment-time vector; if we add an SLP layer, initialize it from `_topp_retime()` output.

5. **Validate with the 3–5% time overhead rule**: Before deploying, benchmark the jerk-constrained retimer. If the race time increases by more than 5% (>0.7s), the jerk limits are too tight. Start at `max_jerk = 50 m/s³` (current parameter) and tighten only if tracking error warrants it.

6. **Consider jerk continuity at segment boundaries in the polynomial generator**: Our `_generate_trajectory()` uses 7th-order minimum-snap polynomials which enforce C3 continuity (continuous jerk). However, the segment *timing* set by `_topp_retime()` determines whether the jerk values at boundaries are actually within physical limits. Jerk-constrained timing + minimum-snap polynomials is the correct combination.

7. **Lower ILC Q-filter cutoff conservatism post-implementation**: If jerk constraints reduce gate-3 error, the per-section `filter_cutoff_hz` in `section_boundaries` for the S-turn section can be increased (currently conservative to handle overshoot), reducing ILC phase lag.

---

## Limitations & Caveats

**Manipulator vs. drone dynamics**: The paper targets a 7-DOF industrial arm with joint torque/velocity limits. Our system has translational constraints (centripetal acceleration, thrust, drag) and no explicit joint-space representation. The jerk constraint formulation translates directly to Cartesian jerk (which our `max_jerk` parameter already represents), but the paper's dynamic feasibility certificate (torque-space constraint satisfaction) does not apply.

**Jerk at spline knots only**: The authors acknowledge a significant limitation — their jerk constraint is enforced at path discretization points, but between knot points the jerk can be unconstrained if the polynomial basis allows discontinuities. Our minimum-snap polynomials enforce continuous jerk within each segment, but the *magnitude* at segment boundaries is determined by timing, not shape. This partially mitigates the limitation but does not fully resolve it.

**SLP convergence assumption**: The conservative linearization guarantees feasibility at each SLP iterate, but convergence to the globally optimal jerk-constrained trajectory is not proved. The paper shows empirical convergence but notes sensitivity to path discretization (too coarse → poor jerk approximation; too fine → LP becomes large). For our ~20 waypoints per race (from `dt_sample=0.01s`), N is on the order of 1200 points, which is much larger than their test cases. LP solve time may exceed their 7.5ms benchmark.

**No air resistance model**: All path-parameterization approaches assume the robot follows the geometric path exactly. Our drone experiences aerodynamic drag that is velocity-dependent, meaning the actual jerk experienced during flight differs from the planned jerk. The paper's drag-free assumption is standard in TOPP literature but is a limitation for aggressive drone racing.

**Computation budget for offline planning**: Their 7.5ms SLP solve time is for online replanning in an industrial arm. Our trajectories are computed offline (at startup, before the race). We can afford longer planning times (100–500ms) since the trajectory is pre-computed and cached, making the real-time constraint irrelevant.

**Conservative linearization causes suboptimality**: The lower bound approximation for hₖ(x) means the LP sees a stricter jerk constraint than the true one. The final solution is feasible but not necessarily time-optimal. For safety-critical systems this is desirable; for drone racing, we want the fastest feasible trajectory, so some exploration of tighter jerk limits is warranted.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Jerk limit tested | 100–1000 rad/s³ | Industrial arm joints; translates to ~50–500 m/s³ for Cartesian motion |
| Motion time overhead (jerk=1000 rad/s³) | 3–5% | Representative case; tighter limits → larger overhead |
| SLP solve time (warm-started) | 7.533 ± 5.848 ms | 7-DOF arm, N≈100–200 points; our N≈1200 will be slower |
| TOPP-RA baseline solve time | 0.273 ± 0.089 ms | For comparison; confirms TOTP3 is ~28× slower |
| RMS torque reduction (primary axis) | 59% | Axis 0, front-place motion |
| RMS torque reduction (secondary axis) | 66% | Axis 4, front-place motion |
| Peak power reduction | ~25% | Front-place motion representative |
| SLP convergence tolerance (ε) | Not explicitly stated | "until convergence" — typical practice is ‖Δx‖₂ < 10⁻⁴ |
| Path discretization | N points over s ∈ [0,1] | Paper tests N≈50–200; sensitivity acknowledged |
| Our current max_jerk | 50.0 m/s³ | Set in `DroneConstraints` but not enforced in `_topp_retime()` |
| Our current S-turn compression floor | 0.65 | `max_compression_protected` — the binding constraint that jerk enforcement could relax |
| Gate-3 tracking error (current) | 0.247m | Primary target for jerk-constraint improvement |
| Race time recovery target | 0.09s | From iteration notes; jerk enforcement + ILC relaxation path to achieve this |
