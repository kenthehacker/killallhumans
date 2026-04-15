# Almost Global Trajectory Tracking for Quadrotors Using Thrust Direction Control on S²
- **URL**: https://arxiv.org/abs/2409.05702
- **Year**: 2024
- **Venue**: CDC 2024 (IEEE 63rd Conference on Decision and Control)
- **Authors**: Mirko Leomanni, Alberto Dionigi, Francesco Ferrante, Paolo Valigi, Gabriele Costante

---

## Key Contribution

This paper presents a geometric trajectory tracking controller for quadrotors that achieves **almost global asymptotic stability** — meaning convergence from all initial conditions except the set-measure-zero antipodal configuration (inverted flight). The primary novelty is how rotational degrees of freedom are handled: rather than controlling full attitude on SO(3), the paper controls only the **thrust direction** on the unit sphere S², directly coupling translational and rotational dynamics through a composite Lyapunov function.

The critical claim over prior work (e.g., Lee et al. 2010, Refs labeled [25] in paper) is:
1. **No ISS (Input-to-State Stability) cascade argument needed.** Standard geometric controllers (including the Lee controller used in our `mpc_tracker.py`) analyze the translational and rotational subsystems separately via cascades. This paper constructs a single composite Lyapunov function V = V_ξ(position/velocity errors) + rotational error term, proving joint stability without invoking ISS.
2. **Simpler tuning**: Fewer independent parameters than hierarchical cascade designs.
3. **Better transient behavior**: The composite Lyapunov function is monotonically decreasing during convergence, whereas the ISS cascade approach can exhibit non-monotonic transients (tracking error temporarily worsens before improving).

This matters because non-monotonic transients in cascade controllers directly translate to gate misses or overshoot in racing. The real-world experiment showed **2x improvement in steady-state hover error** (0.15 m vs 0.30 m) compared to a prior global-stabilization method.

---

## Technical Approach

### System Model

The quadrotor is modeled with full 6-DOF rigid body dynamics in a body-fixed frame:
- Translational: p̈ = R^T ζ f − ζg, where p ∈ ℝ³ is position, R ∈ SO(3) is attitude, f is thrust magnitude (scalar), ζ = [0,0,1]^T is the body z-axis.
- Rotational: Ṙ = −[ω]× R, where ω is angular velocity.

The **thrust direction vector** is defined as x₃ = R·ζ/‖ζ‖ ∈ S², i.e., the third column of R projected onto the unit sphere. This captures the 2 DOF of attitude relevant for position control (pitch and roll), ignoring yaw.

### Error State Transformation

Position and velocity errors:
- x₁ = p − p_r (position error)
- x₂ = ṗ − ṗ_r (velocity error)
- x₃ = Rv/‖v‖ (thrust direction, where v is a virtual control input in ℝ³\{0})

The transformed dynamics are:
```
ẋ₁ = x₂
ẋ₂ = (v/‖v‖)·f − d + R^T(ζ − x₃)·f
ẋ₃ = [x₃]× (ω − R·ω_v)
```
where d = p̈_r + ζg is the reference-plus-gravity term, and ω_v incorporates reference trajectory derivatives.

The key insight: x₃ appears as a **perturbation term** in the translational dynamics. When x₃ → ζ (i.e., thrust aligned with desired direction), the perturbation vanishes and the translational subsystem reduces to a standard double integrator with u_ξ as control input.

### Backstepping-Based Control Law

**Thrust magnitude**:
```
f = ‖u_ξ(x₁, x₂, d)‖
```
where u_ξ is any stabilizing controller for the double integrator (position + velocity errors). The paper uses a linear PD controller: u_ξ = −K[x₁; x₂] + d.

**Tilt angular velocity** (controls thrust direction on S²):
```
(I − ζζ^T)ω = (I − ζζ^T)(R·ω_v + [ζ]×(κ₁x₃ + β))
```
where:
- κ₁ ≥ k₁ is a smooth tuning function (scalar)
- β = (λ/k₂) · [ζ]×x₃ is an adaptation term that **couples position error feedback into the attitude controller**
- λ = ‖u_ξ‖ · (∂V_ξ/∂x₂)^T couples the translational Lyapunov function gradient to the rotational channel
- k₂ > 0 is the attitude tuning gain

This coupling is the mechanism that allows the composite Lyapunov proof: attitude control "knows about" translational error through λ, so both errors decrease together.

### Composite Lyapunov Function

```
V = V_ξ(x₁, x₂) + (1 − ζ^T x₃) / (2k₂(1 + ζ^T x₃))
```

The second term is a stereographic-projection-inspired measure of attitude error on S². It equals 0 when x₃ = ζ (thrust aligned), and diverges to +∞ as x₃ → −ζ (inverted). This is why the inverted configuration is excluded from the domain of attraction: the Lyapunov function is undefined there.

V̇ satisfies:
```
V̇ = V̇*_ξ(x₁, x₂) − κ₁(1 − ζ^T x₃) / (k₂(1 + ζ^T x₃)) ≤ 0
```

Both terms are non-positive, confirming joint decrease of position and attitude errors.

### Adaptive Gain Function

```
κ₁ = k₁                              if ζ^T x₃ ≥ 0 (tilt < 90°)
κ₁ = k₁ / √(1 − (ζ^T x₃)²)         otherwise (large tilt)
```

This adaptive schedule increases attitude gain when thrust direction error is large, accelerating convergence from poor initial conditions without destabilizing the system near equilibrium.

### Heading Control

Yaw is controlled separately and orthogonally: ζ^T ω = 0 enforces that yaw rate has no component along the body z-axis. This decouples heading from the thrust-direction control, simplifying the analysis.

---

## Results

### Numerical Simulations (100 Randomized Trials)

- Reference: p_r(t) = [0.38t, 0.6sin(2πt/10), 1]^T m — a helical sinusoid with forward drift
- Initial conditions: x₀ ∈ [−5,0]m, y₀ ∈ [−2.5,2.5]m, z₀ ∈ [1,6]m, initial attitudes θ₀, φ₀ ∈ (−π,π) rad
- **100/100 trials converged**, including near-inverted initial conditions
- Compared to prior method (Ref. 25): proposed controller had faster convergence in position error and monotonically decreasing Lyapunov function (prior had non-monotonic transients)
- Peak angular velocity higher initially but total energy expenditure lower

### Real-World Experiments

**Platform**: 1.0 kg quadrotor, 0.2 m diagonal, PixRacer Pro + nVidia Jetson Xavier NX, OptiTrack motion capture at 100 Hz.

**Hovering test** (target [0, 0, 1]^T m):
- Proposed method steady-state error: **0.15 m** (X-axis)
- Prior method (Ref. 25) steady-state error: **0.30 m**
- 2x improvement despite no integral action and platform imbalance

**Figure-8 Trajectory Tracking**:
- Successfully tracked reference trajectory with compensation for aerodynamic disturbances and sensor delays

### Control Parameters Used in Experiments

Linear PD gains matrix K (used for u_ξ):
```
K = [4   0   0   2   0   0]
    [0   4   0   0   2   0]
    [0   0   4.5 0   0   3]
```
This corresponds to: kp_xy=4, kd_xy=2, kp_z=4.5, kd_z=3.

Attitude tuning:
- k₁ = 1.5 (base attitude gain)
- k₂ = 0.05 (scales attitude error in Lyapunov function)
- c = 0.1 (continuity/smoothing parameter)

---

## Relevance to Our System

Our system (`control/mpc_tracker.py`) implements a Lee et al. SE(3) geometric controller with:
- kp_xy=6, kd_xy=4, kp_z=8, kd_z=5
- feedforward_accel=0.4 (partial feedforward)
- The kinematic sim uses raw accel_des directly (bypasses attitude dynamics)

**Key relevance points:**

### 1. Feedforward Weight: Should We Use 1.0?

The paper's controller is equivalent to **full feedforward** (weight=1.0 in our terminology). The term `d = p̈_r + ζg` in their control law is exactly the reference acceleration plus gravity — it is included in full in u_ξ. There is no scaling factor applied to the feedforward term.

The paper provides theoretical backing: without full feedforward, the equilibrium of the translational subsystem shifts away from x₁=0, meaning the controller tracks a perturbed reference. The Lyapunov proof only works when d is included in full.

**Conclusion**: Full feedforward (weight=1.0) is theoretically correct. Our current 0.4 weighting is a practical compromise driven by sim-specific drag behavior (noted in TrackerConfig comments). In a real quadrotor or a sim with accurate thrust modeling, weight=1.0 is the correct choice. For the kinematic sim where accel_des feeds directly to the integrator without attitude dynamics lag, reducing feedforward can actually hurt if the trajectory already accounts for gravity — worth testing weight=0.7-1.0.

### 2. Gain Values: Are kp=6, kd=4 Appropriate?

The paper uses kp_xy=4, kd_xy=2, kp_z=4.5, kd_z=3 on a real platform. Our gains (kp=6, kd=4) are **50% higher** in the proportional channel and **100% higher** in derivative. This is not necessarily wrong — our kinematic sim has no attitude dynamics lag or motor model, so the effective bandwidth is higher and higher gains are stable. However, the ratio kp/kd ≈ 1.5 is consistent with the paper's ratio (2.0), suggesting our gains are in the right ballpark structurally.

The paper's Lyapunov matrix P was computed from `(A − BK)^T P + P(A − BK) + I = 0` — this provides a principled way to set gains rather than manual tuning. For our system, if we were to apply this methodology: with kp=6, kd=4, the characteristic polynomial of the closed-loop double integrator is s²+4s+6, giving damping ratio ζ_d = 4/(2√6) ≈ 0.82 (well-damped). With the paper's gains (kp=4, kd=2): s²+2s+4, ζ_d = 2/(2√4) = 0.5 (underdamped). Our gains are actually **better damped** for the translational channel.

### 3. The β Coupling Term and Feedforward

The paper's β term = (λ/k₂) · [ζ]×x₃ represents attitude correction that "knows" about translational error through λ = ‖u_ξ‖ · ∂V_ξ/∂x₂. In our kinematic sim, this coupling is moot because the sim directly applies accel_des without going through attitude control. However, this explains why even small tracking errors in our system cause oscillations: **without this coupling, the attitude controller doesn't "know" how hard the translational controller is working**, leading to overshoot or oscillatory behavior when gains are mismatched.

### 4. Almost Global vs. Local Stability

Our current Lee controller (via the ISS cascade argument used in Lee 2010) technically only guarantees stability for initial attitude errors less than π/2. The paper's approach extends this to all initial conditions except inverted. For racing, this matters during aggressive maneuvers where tilt angles approach or exceed 90° (our `max_tilt_rad=0.85` ≈ 49°, so we are within the Lee stability region, but just barely for extreme maneuvers).

---

## Actionable Takeaways

1. **Test feedforward_accel = 1.0 in kinematic sim**: The theoretical justification is solid. Our 0.4 value was calibrated to compensate for sim drag interaction, but this paper provides strong evidence that full feedforward is correct. If the kinematic sim's gravity handling is correct (accel_des already includes gravity compensation), weight=1.0 should be tested in a controlled benchmark run.

2. **The k₁=1.5 attitude gain is very low**: Their attitude gain k₁ drives the thrust-direction correction. If we were implementing this S² controller instead of Lee's, we'd start with k₁=1.5 and use the adaptive schedule. Our current kr=8.0 corresponds to a much stiffer attitude loop, which can cause oscillations if the translational gains don't match.

3. **Consider the β coupling for real hardware**: When deploying on real hardware via MAVLink, implementing the β coupling term (attitude correction proportional to translational error gradient) would improve tracking significantly. This is the main difference between ISS-cascade (Lee) and this approach.

4. **Gain ratio kp/kd ≈ 1.5–2.0 is structurally validated**: Our kp_xy/kd_xy = 6/4 = 1.5 matches the paper's ratio closely. This ratio determines the damping of the error dynamics and is not arbitrary — it comes from the Lyapunov matrix computation.

5. **Adaptive κ₁ for large-tilt maneuvers**: Implementing the adaptive gain schedule `κ₁ = k₁/√(1−cos²θ_tilt)` for tilt angles > 45° could improve recovery from aggressive corner maneuvers where tilt exceeds 45°. This would not change normal-flight behavior but would aggressively correct large attitude excursions.

6. **No integral action = persistent offset**: The paper explicitly notes this limitation. Our system similarly lacks integral action. For real hardware this causes steady-state offset (as seen in their 0.15 m hover error). For the kinematic sim this matters less since there's no persistent disturbance, but for any real deployment an integral term (or disturbance observer) would be needed.

---

## Limitations & Caveats

1. **No integral action**: The controller has persistent steady-state error under constant disturbances (wind, unmodeled drag, sensor bias). Hover error of 0.15 m was acceptable for their test but would be problematic for gate precision in racing (gates are typically ~0.5–1.0 m in diameter).

2. **Minimum acceleration constraint** (‖d‖ ≥ d_m): The controller requires the desired thrust magnitude to be bounded away from zero. Near-hover with near-zero reference acceleration, the thrust magnitude approaches 1·g, so this is rarely binding. However, during certain maneuvers (e.g., near-ballistic trajectories where the quadrotor deliberately reduces thrust), this constraint can be violated.

3. **Angular velocity dynamics neglected**: The control law directly commands angular velocity ω, treating it as a perfect actuator. In reality, motor dynamics and inertia mean the commanded ω is not instantaneously achieved. The paper notes this as future work (another backstepping step could be added). For our kinematic sim, angular velocity is indeed a perfect actuator, so this limitation doesn't apply — but it means the controller cannot be directly compared to real hardware performance.

4. **Yaw control decoupled but not analyzed**: The yaw channel (ζ^T ω = 0) is controlled separately and its stability proof is not part of the main theorem. In practice, yaw coupling through gyroscopic effects is ignored.

5. **Single vehicle, no formation or multi-agent analysis**: Results apply only to a single quadrotor tracking a pre-computed reference. The paper does not consider obstacle avoidance, inter-vehicle constraints, or online replanning.

6. **Comparison baseline (Ref. 25) may be outdated**: The paper compares against a single prior global method. The Lee 2010 controller (which we use) is not included in the comparison, making it harder to directly assess the gain over our current implementation.

7. **Real-world experiment was hover and figure-8 only**: These are low-speed, low-aggressiveness maneuvers. No aggressive racing trajectories (high-speed gates, split-S maneuvers) were tested. Whether the "almost global" property provides practical benefit in racing scenarios (where tilt rarely exceeds 60°) remains unvalidated.

---

## Key Parameters / Constants

From the paper's experimental implementation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| kp_xy | 4.0 | Translational proportional gain (x, y axes) |
| kd_xy | 2.0 | Translational derivative gain (x, y axes) |
| kp_z | 4.5 | Translational proportional gain (z axis) |
| kd_z | 3.0 | Translational derivative gain (z axis) |
| k₁ | 1.5 | Base attitude (thrust-direction) proportional gain |
| k₂ | 0.05 | Attitude error weighting in Lyapunov function |
| c | 0.1 | Continuity/smoothing parameter |
| Control rate | 100 Hz | OptiTrack-based state feedback loop rate |
| Platform mass | 1.0 kg | Test vehicle mass |
| Platform diagonal | 0.2 m | Rotor-to-rotor diagonal |
| Hover error (proposed) | 0.15 m | X-axis steady-state hover error |
| Hover error (prior) | 0.30 m | X-axis steady-state hover error, prior method |
| Feedforward weight | 1.0 | d = p̈_r + ζg included fully in control law |
| Adaptive κ₁ threshold | ζ^T x₃ = 0 | Below this, adaptive gain kicks in |

**Closed-loop pole analysis** (translational, using paper's gains):
- Characteristic polynomial: s² + kd·s + kp = s² + 2s + 4
- Damping ratio: ζ_d = kd/(2√kp) = 2/(2·2) = 0.5 (underdamped)
- Natural frequency: ωn = √kp = 2 rad/s

**Our system comparison** (kp=6, kd=4):
- Characteristic polynomial: s² + 4s + 6
- Damping ratio: ζ_d = 4/(2√6) ≈ 0.816 (well-damped, near-critically)
- Natural frequency: ωn = √6 ≈ 2.45 rad/s
- Conclusion: Our translational gains provide better damping at slightly higher bandwidth — reasonable for our kinematic sim context.
