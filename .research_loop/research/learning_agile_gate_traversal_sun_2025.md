# Learning Agile Gate Traversal via Analytical Optimal Policy Gradient

- **URL**: https://arxiv.org/abs/2508.21592
- **Authors**: Tianchen Sun, Bingheng Wang, Nuthasith Gerdpratoom, Longbin Tang, Yichao Gao, Lin Zhao (National University of Singapore)
- **Year**: 2025
- **Venue**: arXiv cs.RO

---

## Key Contribution

This paper presents a hybrid neural network + model predictive control (NN-MPC) framework for agile quadrotor gate traversal through arbitrarily oriented, confined gates. The key problem is that classical MPC requires hand-tuned cost weights and fixed reference trajectories that fail for gates tilted far off-axis, while end-to-end RL provides poor interpretability, sample inefficiency, and weak disturbance rejection. The authors occupy the middle ground: a neural network predicts adaptive reference poses and time-varying MPC cost weights online (a 20-dimensional output vector), while an MPC layer retains full physical optimization and online disturbance rejection.

The core technical novelty is the derivation of **analytical policy gradients** flowing through two difficult layers: (1) a differentiable conic optimization for collision detection, and (2) the MPC's optimality conditions via the discrete-time Pontryagin Minimum Principle (PMP) and backward Riccati recursion. This avoids finite-difference or sampling-based gradient estimation, which are noisy and expensive. The result is a system that trains in 736k steps (~93 minutes on 32 CPU cores) versus PPO's 200M steps on GPU — a 270x sample reduction — while achieving zero-shot sim-to-real transfer with 30 m/s² peak accelerations and recovery from 1146 deg/s disturbances within 0.85 seconds.

---

## Technical Approach

### Quadrotor Dynamics Model

The state vector is `x = [p, v, q, ω]` — position (3), velocity (3), quaternion (4), body rates (3). The control input is `u = [T, ω_cmd]` — collective thrust plus body-rate command. Continuous dynamics are integrated using fourth-order Runge-Kutta with a dt = 0.1 s prediction step and a horizon of N = 20 steps (so the MPC looks 2 seconds ahead).

### MPC Stage Cost

The MPC minimizes a cumulative stage cost with four terms:

```
c(x_k, u_k) = w_p_ref(t_k) ||p_k - p_ref||²
            + w_R_ref(t_k) ||R_k - R_ref||_F²
            + w_p_goal(t_k) ||p_k - p_goal||²
            + w_u ||u_k||²
```

The subscript `_F` denotes Frobenius norm. The attitude reference `R_ref` and the three weight schedules `[w_p_ref, w_R_ref, w_p_goal]` are all predicted by the neural network (they are functions of time step k within the horizon, not scalars).

### Time-Varying Cost Weights

The three cost schedules are parameterized analytically using smooth basis functions, with the neural network predicting their amplitudes `[Q_p_ref, Q_R_ref, Q_p_goal]` and the reference traversal time `t_ref`:

```
w_p_ref(k)  = (Q_p_ref / 2) * (1 + tanh(1000 * (t_ref - k*dt)))
w_p_goal(k) = (Q_p_goal / 2) * (1 + tanh(1000 * (k*dt - t_ref)))
w_R_ref(k)  = Q_R_ref * exp(−γ * (k*dt − t_ref)²)
```

The effect is a smooth handoff: before the gate, position-tracking dominates; after the gate, goal-reaching dominates; around `t_ref`, attitude tracking peaks (the drone must be correctly oriented at the gate). The sharpness parameter γ is learnable or tunable.

### Attitude Representation via SVD Projection

To avoid gradient discontinuities that arise from Rodrigues parameters and gimbal lock in Euler angles, the neural network outputs an unconstrained 3×3 matrix `M_ref`. This is projected to SO(3) via signed SVD:

```
R_ref = SVD⁺(M_ref) := U * diag(1, 1, det(UV^T)) * V^T
```

The `det(UV^T)` correction ensures the result is a proper rotation (det = +1). This makes the mapping from network output to rotation matrix smooth and fully differentiable.

### Neural Network Architecture

- Two hidden layers, 256 neurons each
- SiLU (Swish) activation
- Spectral normalization on weights (for training stability)
- Inputs: quadrotor state `[p, v, q, ω]`, goal position `p_goal`, four gate corner positions in body frame (12 values) — total ~25 inputs
- Outputs: 20-dimensional vector = `[M_ref (9), t_ref (1), Q_p_ref (1), Q_R_ref (1), Q_p_goal (1), ...]` plus control smoothness weights
- Runs at 100 Hz onboard on Radxa ZERO 2 Pro (ARM-class SBC)

### Collision Detection as Conic Optimization

Rather than using a hard indicator function (non-differentiable), the paper reformulates collision as a continuous scaling problem. The quadrotor body is an ellipsoid `C_quad(T_quad, α)` and the gate clearance region is a polytope `C_gate(T_gate, α)`. Both sets are scaled by α, and the optimization finds:

```
α*_n = argmin α_n
s.t.  x ∈ C_quad(T_quad, α_n)
      x ∈ C_gate_n(T_gate_n, α_n)
      α_n ≥ 0
```

When `α*_n ≤ 1`, the drone-body ellipsoid overlaps with the gate boundary polytope (collision). When `α*_n > 1`, there is clearance. The gate traversal loss penalizes near-collision:

```
L_gate = β_gate * Σ_n max(0, 1 - α*_n)²     [β_gate = 100]
```

Crucially, the Envelope Theorem converts the implicit derivative `dα*/dθ` into an explicit form that only requires the Lagrange multipliers at the solution, enabling efficient backpropagation.

### MPC Differentiation via Safe-PDP

Differentiating through the MPC solution `ξ*(z)` (where z is the neural network output) uses the Safe-PDP framework. The key idea is to approximate the inequality constraints of the MPC using a logarithmic barrier in the forward pass, then derive necessary optimality conditions from the resulting unconstrained problem. The gradient of the MPC outputs with respect to inputs reduces to solving a **finite-time linear quadratic regulator (LQR)** backward pass — specifically, a backward Riccati recursion with sensitivity matrices as the quantities being accumulated. This is efficient (O(N) in horizon length) and avoids any numerical Jacobian approximation.

### Total Training Loss

```
L = L_gate + L_goal + L_control
  = β_gate * Σ max(0, 1 - α*_n)²
  + β_goal * Σ_{j=N-h+1}^{N} ||p_j - p_goal||²
  + β_control * Σ ||u_i - u_{i-1}||²
```

Loss weights: β_gate = 100, β_goal = 2, β_control = 0.001. The goal loss averages over the last h = 5 horizon steps to avoid terminal-only sensitivity.

### Training Protocol

- 32 parallel CPU environments (no GPU required)
- Adam optimizer, lr = 0.0002, decay 0.99 per 50 episodes
- Total: 736k simulation steps, ~93 minutes wall-clock time
- Gate randomization: position, orientation ±72° around y-axis, initial drone state varied
- Zero-shot sim-to-real: no domain randomization, no fine-tuning

---

## Results

### Simulation

- Gate dimensions: 0.6 m wide × 0.25 m tall (narrow — only 2.5× the drone diameter)
- Quadrotor ellipsoid: 0.3 m × 0.3 m × 0.1 m
- Gate orientations tested: ±72° tilt around y-axis
- Untrained MPC baseline success rate: **9.38%** (128 trials)
- Trained NN-MPC success rate: **80.46%** (128 trials)
- This is an 8.6× improvement in gate passage reliability

### Real-World Hardware

Hardware: custom quadrotor, 25 cm tip-to-tip, 260 g mass, Radxa ZERO 2 Pro onboard computer, 100 Hz control loop.

| Metric | Value |
|--------|-------|
| Peak acceleration | 30 m/s² |
| Gate orientations covered | 30° to 70° |
| Minimum gate clearance demonstrated | 7.5 cm |
| Recovery time from 1146 deg/s disturbance | 0.85 s |
| Policy gradient compute time | 0.16 s |
| Training samples | 736k steps |
| Training wall-clock time | 93 min (32 CPU cores) |

### Disturbance Rejection (Simulation, Unseen Disturbances)

Applied: 15.5 m/s² linear + 480 rad/s² angular acceleration for 0.1 s.

| Method | Settling Time |
|--------|--------------|
| Proposed NN-MPC (trained) | 0.89 s |
| PPO (RL) | 1.30 s |
| PX4 cascaded PID | 2.18 s |

The proposed method shows 31% faster recovery than PPO and 59% faster than PX4 PID, attributed to the MPC's online replanning capability.

### Training Efficiency

| Method | Gradient Compute | Training Samples |
|--------|-----------------|-----------------|
| Proposed (analytical gradients) | 0.16 s | 736k |
| Wang et al. (finite difference) | 0.29 s | — |
| AC-MPC (sampling-based) | 0.22–0.58 s | — |
| PPO (RL baseline) | — | 200M |

The proposed method requires 270× fewer samples than PPO while achieving comparable or better task success rates.

### Ablation Study (Implicit from Paper)

The tanh-based time-varying cost schedule is critical. Without it (fixed weights), the MPC cannot handle the dynamic priority shift from gate alignment to goal reaching, leading to the low 9.38% success rate of the untrained baseline.

---

## Relevance to Our System

Our system uses min-snap polynomial trajectory planning with L-BFGS-B racing line optimization and a kinematic sim with a PD controller. The current bottleneck is the S-turn at gate-3 (error 0.463 m) caused by approach-side proximity inflation in our racing line optimizer, which pushes the trajectory wide before the gate rather than aiming directly at it.

This paper is directly relevant in the following ways:

**1. Adaptive Reference Pose for Gate Traversal.** Our racing line currently computes a fixed waypoint through each gate center. The NN-MPC approach shows that dynamically adjusting the reference pose and approach angle as a function of the current drone state and gate orientation can dramatically improve gate passage reliability (9% → 80%). For our S-turn, this translates to: instead of always approaching gate-3 from a fixed offset, learn (or compute analytically) the optimal approach pose given the current trajectory state.

**2. Time-Varying Cost Weights as a Design Pattern.** The tanh-sigmoid handoff between gate-alignment cost and goal-reaching cost is a clean engineering pattern we could adopt in our MPC or PD tracker. Before a gate, weight position tracking heavily; at the gate, weight attitude; after the gate, switch to next-waypoint tracking. This is related to our current issue where the racing line optimizer doesn't differentiate between the "approach" and "through-gate" phases.

**3. Differentiable Collision Formulation.** Our L-BFGS-B optimizer currently uses heuristic proximity inflation. Reformulating gate passage as a conic scaling optimization (α* > 1 = safe) would give us analytical gradients through the gate constraint, eliminating the need for conservative empirical inflation margins that cause our gate-3 deviation.

**4. The Gap Between Our System and This Paper.** Our PD controller lacks MPC's online replanning capability. The paper's disturbance rejection advantage (0.89 s vs. 2.18 s for PID) reflects a fundamental architecture difference. However, many of the gate-traversal insights — especially the reference trajectory design and cost weight scheduling — are transferable to our trajectory optimizer without needing a full MPC stack.

---

## Actionable Takeaways

1. **Replace proximity inflation with conic clearance margin.** In `planning/racing_line.py`, replace the current Euclidean distance-to-gate-edge penalty with the α-scaling formulation: represent the drone as a 2D ellipse and the gate as a 1D line segment (in the gate plane), and penalize when α* < 1.2 (20% clearance margin). This gives a differentiable constraint that L-BFGS-B can handle cleanly.

2. **Add gate-approach phase distinction to racing line optimizer.** The tanh cost-switching pattern is directly applicable in our trajectory optimizer. Add a "pre-gate" waypoint 0.5–1.0 m before each gate center, weighted for approach angle alignment, and a "through-gate" waypoint at the center. This mirrors the paper's `w_p_ref` → `w_p_goal` transition and would fix the approach-side inflation problem at gate-3.

3. **Implement SVD-projected attitude targets for non-axis-aligned gates.** If we encounter tilted gates, adopt the SVD⁺ projection for smooth rotation matrix parameterization in our trajectory optimizer, avoiding quaternion interpolation artifacts near ±90° tilts.

4. **Time-varying weights in `mpc_tracker.py`.** In `control/mpc_tracker.py`, implement tanh-based weight scheduling: set `w_position` high before a gate waypoint, spike `w_attitude` within 0.1 s of the gate crossing time, then transition to `w_goal` for the next segment. This is a low-effort, high-leverage change.

5. **Sample efficiency as validation.** When evaluating changes to our L-BFGS-B optimizer, track improvement in simulation success rate across multiple random starts (we already do this with multi-start L-BFGS). The paper's 9% → 80% improvement benchmark gives us a sense of how large the headroom is when gate-approach logic is improved.

6. **Consider NN pre-training for racing line warm-start.** A small MLP trained to predict near-optimal gate approach offsets (conditioned on the previous gate position and next gate position) could warm-start our L-BFGS-B optimizer, avoiding the local minima that cause the gate-3 problem. This is a lower-cost alternative to full NN-MPC: train offline, use as initialization, then refine with gradient descent.

7. **Disturbance rejection benchmark.** Add a disturbance rejection test to our benchmark suite (inject a 5 m/s² linear perturbation during gate-3 traversal) to measure recovery time and motivate future MPC adoption.

---

## Limitations & Caveats

**Not end-to-end racing.** The paper addresses a single gate traversal task, not a multi-gate race sequence. There is no timing optimization (lap time minimization), no multi-gate trajectory planning, and no speed profiling. Our system's racing line optimizer handles the global trajectory; this paper's method addresses local gate-traversal execution.

**Single gate orientation (tilt only).** Gates are randomized in orientation around the y-axis only (roll-tilt), covering ±72°. Yaw-rotated gates (common in DCL racing) are not demonstrated. Our DCL-style courses use predominantly yaw-rotated gates, so the attitude tracking component is less relevant than the approach-trajectory insights.

**Small quadrotor, slow flight.** The hardware is a 260 g, 25 cm drone. Peak acceleration of 30 m/s² is impressive for the platform but corresponds to roughly 3g. Full-size racing drones operate at 10–15g. The MPC horizon of 2 seconds at 100 Hz is appropriate for this scale but would need adjustment for higher-speed operation.

**Gate dimensions are challenging but not race-speed.** The 0.6 m × 0.25 m gate with 7.5 cm minimum clearance is a difficult precision task, but it is not demonstrated at high forward speeds (no lap time metrics provided). The paper focuses on success rate and disturbance rejection, not on minimizing traversal time.

**Sim-to-real transfer without aerodynamics.** Zero-shot transfer succeeds partly because the quadrotor and gates are small, aerodynamic effects (drag, ground effect, propwash) are limited, and the MPC provides online correction. At racing speeds with larger drones, drag and propwash near gates become significant and would require explicit aerodynamic modeling.

**Training requires known gate pose.** The neural network is conditioned on the four gate corner positions in the body frame — it assumes accurate gate pose estimation. Our system uses PnP pose estimation with a Kalman filter, which introduces noise. The paper does not analyze sensitivity to gate pose estimation errors.

**Safe-PDP implementation complexity.** The analytical gradient derivation through the MPC is non-trivial to implement and requires a specific MPC solver (they use a custom implementation supporting backward Riccati passes). Adopting this in our codebase would require significant refactoring of `mpc_tracker.py`.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| MPC horizon N | 20 steps | |
| MPC discretization dt | 0.1 s | → 2s lookahead |
| tanh sharpness | 1000 | Sigmoid steepness in weight schedule |
| Attitude weight γ | Learnable | Temporal sharpness of attitude bell curve |
| β_gate | 100 | Gate collision loss weight |
| β_goal | 2 | Goal-reaching loss weight |
| β_control | 0.001 | Control smoothness loss weight |
| Goal loss window h | 5 steps | Last 5 horizon steps averaged |
| NN hidden size | 256 neurons | |
| NN output dimension | 20 | |
| Adam lr | 0.0002 | |
| Adam lr decay | 0.99 per 50 ep | |
| Training environments | 32 (CPU) | |
| Training steps | 736k | |
| Training time | 93 min | On 32 CPU cores |
| Drone mass | 260 g | |
| Drone tip-to-tip | 25 cm | |
| Drone ellipsoid | 0.3 × 0.3 × 0.1 m | For collision detection |
| Gate size | 0.6 m × 0.25 m | |
| Gate tilt range | ±72° (y-axis) | |
| Minimum clearance | 7.5 cm | |
| Peak acceleration | 30 m/s² (~3g) | |
| Disturbance recovery | 0.85 s | From 1146 deg/s body-rate impulse |
| Control loop rate | 100 Hz | NN + MPC onboard |
| Reference traversal time t_ref | 1.0 s | Preset (not learned) |
