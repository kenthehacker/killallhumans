# Quadrotor Trajectory Tracking Using Linear and Nonlinear MPC
- **URL**: https://arxiv.org/abs/2411.06707
- **Year**: 2024
- **Venue**: 25th National Conference on Electronics, Communications and Information Technology (REV-ECIT 2024), arXiv preprint

---

## Key Contribution

This paper presents a direct empirical comparison between two classical model-based control architectures for quadrotor trajectory tracking: a Linear Model Predictive Controller (LMPC) and a Nonlinear Model Predictive Controller (NMPC). The contribution is comparative rather than novel — the work consolidates the two dominant MPC paradigms for quadrotors into a unified framework and evaluates them against identical test conditions (same reference trajectory, same simulation environment, same metrics).

The paper does not introduce a new controller design. Instead it provides a structured pedagogical comparison useful for practitioners selecting between LMPC and NMPC for quadrotor applications. This makes it relevant as a reference baseline, not as a source of algorithmic innovation.

---

## Technical Approach

### Quadrotor Model

The system uses a 12-state representation:

    x = { ξ^T, η, ξ_dot^T, η_dot^T }

where ξ = (x, y, z) is position in the inertial frame, η = (φ, θ, ψ) is the Euler-angle orientation vector (roll, pitch, yaw). The control input is:

    u = { ω₁², ω₂², ω₃², ω₄² }

i.e., squared angular velocities of the four rotors — a direct rotor-speed parameterization rather than the collective-thrust + body-rate parameterization used in, e.g., Romero 2025.

Translational dynamics are derived via Lagrangian mechanics, yielding the standard quadrotor form:

    x_ddot = -g [0,0,1]^T + (T/m) R e₃

where T is total thrust, R is the body-to-inertial rotation matrix, and e₃ is the body z-axis. Euler-angle constraints: φ, θ ∈ (-π/2, π/2), ψ ∈ (-π, π) — gimbal-lock-free regime.

### Linear MPC (LMPC)

LMPC linearizes the full nonlinear model about an operating point (or trajectory) and then applies discrete-time linear MPC. The discretized state-space model:

    x_{k+1} = A x_k + B u_k + V_d F_{e,k}

where V_d F_{e,k} is an external disturbance term. The optimization problem is a standard quadratic program (QP) solved online at each timestep. This is computationally cheap but incurs linearization error that grows with deviation from the linearization point — a direct problem for aggressive maneuvers or high-speed flight.

### Nonlinear MPC (NMPC)

NMPC solves the optimal control problem directly on the continuous-time nonlinear dynamics:

    x_dot = f(x, u)

using a multiple-shooting discretization scheme. This eliminates linearization error at the cost of solving a nonconvex nonlinear program (NLP) at each timestep. The paper uses a standard NLP solver (likely IPOPT or similar) but does not detail the implementation.

### Shared Cost Function

Both controllers minimize the same objective:

    J = ∫₀ᵀ ( ‖e_x(t)‖²_Q + ‖u(t)‖²_R ) dt + ‖e_x(T)‖²_P

where e_x(t) = x(t) - x_ref(t) is the tracking error. Terminal cost uses P = Q.

### Weighting Matrices

    P = Q = diag([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

This penalizes only position and orientation errors, not their rates. Velocity tracking error is not directly penalized, which is a notable design choice — it can reduce damping of velocity errors and make transient response more oscillatory.

    R = diag([0.1, 0.1, 0.1, 0.1])

Low input penalty (0.1) relative to state penalty (1.0) — prioritizes tracking accuracy over control effort, appropriate for racing but potentially aggressive for hardware.

### Constraints

- Input bounds: 0 ≤ u ≤ 10 rad/s (rotor speed, per rotor)
- Input rate limits: |δu| ≤ 2 rad/s (actuator bandwidth constraint)
- Prediction horizon: N = 18 steps
- Control horizon: N_u = 2 steps (much shorter than prediction horizon — common trick to reduce NLP dimension)

A control horizon of 2 steps with an 18-step prediction horizon is a significant approximation. It reduces the optimization from a 4×18 = 72-variable problem to a 4×2 = 8-variable problem, but at the cost of reduced flexibility in the control sequence.

### Reference Trajectory

A helical path used for both controllers:

    x(t) = 2 cos(2t/5)
    y(t) = 2 sin(2t/5)
    z(t) = 0.2t

Radius 2 m, vertical climb rate 0.2 m/s. This is a smooth, continuous, low-curvature trajectory — not representative of aggressive gate traversal.

---

## Results

### Position RMSE

| Controller | Position RMSE (m) | Angle RMSE (rad) |
|------------|-------------------|------------------|
| LMPC       | 0.2395            | 0.0023           |
| NMPC       | 0.2394            | 0.0025           |

The steady-state tracking accuracy is essentially identical between LMPC and NMPC. The difference (0.0001 m in position, 0.0002 rad in angle) is within simulation noise.

### Transient Response

NMPC demonstrates faster convergence from initial error to steady-state tracking: ~5 seconds for NMPC vs ~7-10 seconds for LMPC. NMPC also shows no overshoot in convergence, while LMPC exhibits mild oscillatory transients.

This transient advantage of NMPC is the primary differentiator in this paper. For drone racing, where the drone is never in steady-state and is perpetually in transient response, this distinction matters significantly.

### Overall Assessment from Paper

The paper concludes that NMPC is preferable due to faster convergence and no overshoot. However, the paper does not report computational latency, real-time feasibility, or control loop frequency — critical omissions for any racing application.

---

## Relevance to Our System

Our system uses a **GeometricTracker** (Lee et al. SE(3) controller) — not MPC. The geometric controller is a pure feedback law with PD gains plus feedforward acceleration, running at 50-120 Hz in the kinematic PyBullet simulation. Current tracking error is ~0.174 m average (as of iteration 37).

### Is MPC Likely to Help in a Kinematic Sim?

The answer is nuanced and context-dependent:

**Arguments against switching to MPC:**

1. **Kinematic sim removes the key MPC advantage.** The geometric controller performs near-optimally when the dynamics model is trivial. In a kinematic sim, the "plant" is essentially integrator chains, and a PD controller with feedforward is already near-optimal for a linear plant. NMPC's benefit comes from handling nonlinear dynamics — which are absent in a kinematic sim.

2. **No model mismatch to exploit.** MPC's advantage over PD is handling nonlinear coupling between axes and actuator saturation constraints. In the kinematic PyBullet sim, none of these couplings are present.

3. **Computational cost.** NMPC requires solving a NLP at each control step. Even with N_u=2 and N=18 (as in this paper), this is orders of magnitude slower than the GeometricTracker's closed-form computation. Our current loop runs at >100 Hz; NMPC would likely drop this below 50 Hz unless we use a fast solver (acados, OSQP).

4. **The paper's error (~0.24 m RMSE on a slow helix) is already worse than our current ~0.174 m avg error.** This suggests the LMPC/NMPC implementations in this paper are not tuned for aggressive racing.

**Arguments for MPC in our setting:**

1. **Constraint handling.** Our GeometricTracker clips thrust and tilt with hard limits, but this is not constraint-aware in the sense of MPC — it cannot pre-emptively slow down to respect upcoming saturation. An MPC that sees N=18 steps ahead would anticipate saturation and adjust the trajectory to stay feasible.

2. **Predictive feedforward.** Our feedforward_accel=0.4 provides partial feedforward, but a full predictive horizon could compensate for trajectory curvature better, especially in tight S-turns and helices.

3. **Better velocity tracking.** The Q matrix in this paper does not penalize velocity error (zeros on velocity states), but a well-tuned LMPC that includes velocity weights could reduce phase lag more systematically than our PD tuning.

**Practical verdict:** For a kinematic sim at our current performance level (~0.174 m avg error), the effort-to-payoff ratio of implementing MPC is low. The geometric tracker is near the theoretical optimum for a kinematic sim. Gains from MPC would be marginal and offset by the complexity and loop-rate cost.

If we transition to a **dynamic sim** (full rotor dynamics, motor inertia, drag), MPC — especially NMPC — would become much more valuable, as nonlinear coupling and aerodynamic terms become the primary error sources.

---

## Actionable Takeaways

1. **Do not replace GeometricTracker with LMPC/NMPC for kinematic sim.** The kinematic sim removes the key benefit of MPC. Our current geometric controller is already achieving lower RMSE (0.174 m) than the NMPC in this paper (0.24 m RMSE on a slower, simpler trajectory).

2. **The N_u << N trick is worth noting.** Using a control horizon (N_u=2) much shorter than the prediction horizon (N=18) is a practical way to reduce MPC cost while retaining prediction benefits. If we ever implement LMPC for a dynamic sim, start with N=20, N_u=2-4.

3. **Velocity weighting in the cost function matters.** The paper's Q matrix zeros out velocity error. For trajectory tracking (not just position regulation), include velocity error weights: e.g., Q = diag([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]). Velocity weights reduce phase lag.

4. **NMPC convergence advantage is real but irrelevant at steady-state.** NMPC's 5s vs LMPC's 7-10s convergence benefit matters for the initial transient. In racing, the "initial transient" is the entire race — the drone never reaches steady-state. This slightly tilts toward NMPC if we ever add MPC, but the paper doesn't report computational cost, which is the actual constraint.

5. **The paper's rotor-speed parameterization (u = ω²) is not directly applicable.** Our system uses collective thrust + attitude commands (MAVLink/PX4 interface), matching the "On Your Own" (Romero 2025) architecture. Any MPC implementation should be formulated in thrust/body-rate space, not rotor-speed space, for compatibility with our competition interface.

6. **External disturbance term in LMPC (V_d F_e)** is a useful modeling choice for handling wind or unmodeled aerodynamics. If we add aerodynamic drag to our model, incorporating a disturbance feedforward term in LMPC would help.

---

## Limitations & Caveats

1. **No real-hardware validation.** The entire paper is simulation-only. Real-world deployment would face motor inertia, sensor noise, communication latency, and aerodynamic disturbances that are absent here.

2. **No computational timing reported.** The paper does not report solve times, control loop frequencies, or NMPC solver characteristics. This is a critical omission for any real-time application. Without this, we cannot assess whether the NMPC is tractable at racing speeds.

3. **Slow, smooth reference trajectory.** The helical test trajectory (v ≈ 1-2 m/s, radius 2 m) is far from competitive drone racing conditions (10-20 m/s, tight gate traversals, 2+ g accelerations). Results on this trajectory may not generalize.

4. **No constraint violations reported.** The paper does not analyze how often the optimizer hits input bounds or rate limits. Frequent constraint activity would indicate that the constraints are actually doing work and the MPC is actively solving a non-trivial feasibility problem — this would be the interesting case.

5. **Control horizon N_u=2 is very aggressive.** Using only 2 control degrees of freedom with an 18-step prediction horizon means the controller can only "steer" for 2 steps before holding controls constant. For fast maneuvers, this may be insufficient — the controller sees the obstacle but cannot take sufficient action.

6. **Linearization point not specified for LMPC.** The paper does not specify whether LMPC uses a single hover linearization or a time-varying linearization along the reference trajectory. The latter (trajectory linearization) is substantially more accurate for tracking and is the standard for competitive implementations, but it is not mentioned.

7. **Paper is written in Vietnamese** (conference REV-ECIT 2024 is a Vietnamese national conference). English abstract is available but full technical details may contain nuances lost in extraction.

8. **RMSE is near-identical at steady state.** The paper's headline result (NMPC converges faster) is valid, but the primary practical metric — steady-state RMSE — shows LMPC and NMPC are effectively equal. This weakens the argument for NMPC's added complexity.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| State dimension | 12 | position (3) + angles (3) + velocities (6) |
| Control dimension | 4 | squared rotor angular velocities |
| Prediction horizon N | 18 steps | |
| Control horizon N_u | 2 steps | much shorter than N |
| Input bounds | [0, 10] rad/s | rotor angular velocity per rotor |
| Input rate limit | ±2 rad/s | per rotor, per step |
| Q weight (position+angle) | 1.0 (diagonal) | |
| Q weight (velocities) | 0.0 | velocities not penalized |
| R weight (inputs) | 0.1 (diagonal) | per rotor |
| LMPC position RMSE | 0.2395 m | helix trajectory |
| NMPC position RMSE | 0.2394 m | helix trajectory |
| LMPC angle RMSE | 0.0023 rad | |
| NMPC angle RMSE | 0.0025 rad | |
| NMPC convergence time | ~5 s | from initial offset to tracking |
| LMPC convergence time | ~7-10 s | from initial offset to tracking |
| Reference trajectory radius | 2 m | helix |
| Reference climb rate | 0.2 m/s | helix z-direction |
| Reference angular rate | 2/5 rad/s | helix x-y |

**Our system comparison (iteration 37):**
- Controller: GeometricTracker (Lee et al. SE(3) PD + feedforward)
- Avg tracking error: ~0.174 m (better than this paper's 0.239 m)
- Loop frequency: >100 Hz
- Control parameterization: collective thrust + attitude (not rotor speeds)
