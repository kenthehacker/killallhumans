# Learning Robust Agile Flight Control with Stability Guarantees

- **URL**: https://arxiv.org/abs/2510.12611
- **Authors**: Lukas Pries, Markus Ryll (Autonomous Aerial Systems Lab, Department of Aerospace and Geodesy, TU Munich, Germany)
- **Year**: 2025 (submitted October 14, 2025)

## Key Contribution

This paper introduces the Neural Geometric Tracking Controller (NGTC), a hybrid control architecture that fuses a classical differential-flatness-based geometric feedback controller with a learned neural augmentation while providing formal, universal stability guarantees. The core problem it addresses is the long-standing tension in agile UAV control: pure learning-based controllers (RL, imitation) can match or exceed expert performance but lack stability certificates, whereas classical geometric controllers (DFBC, SE(3) Lee et al.) are provably stable but degrade significantly on aggressive or actuator-infeasible trajectories where their linear error correction becomes over-saturated.

NGTC resolves this tension using two key theoretical tools. First, the Youla Parameterization guarantees that any stabilizing augmentation of the form `u = k(x) + Q(x̃)` inherits the base controller's stability, provided Q is itself a stable (contracting) operator. Second, Recurrent Equilibrium Networks (RENs) provide a neural architecture that is contracting by construction via a structured weight parameterization. Together, these enable unconstrained gradient-based training of Q while structural stability is guaranteed regardless of the learned weights. The practical payoff is substantial: NGTC achieves 40–78% lower tracking error than DFBC on infeasible trajectories, runs at 0.576 ms per inference (5× faster than NMPC), achieves 0% crash rate under 50% drag perturbation where DFBC crashes 7% of the time, and transfers directly from simulation to hardware without domain randomization or fine-tuning.

## Technical Approach

### Control Law Structure

The NGTC control law takes the form:

```
u_t = k(x_t) + Q(x̃_t) + r_t
```

where `k(x_t)` is the base DFBC geometric feedback controller, `Q(x̃_t)` is the REN neural augmentation, `r_t` is the reference feedforward term, and `x̃_t = x_t - x̂_t` is the residual state — the difference between the observed state and the nominal predicted state under zero disturbance. This residual encodes all model mismatch and disturbance effects; Q learns to correct exactly this discrepancy without affecting the nominal closed-loop stability structure.

### Youla Parameterization and Stability

The Youla Parameterization theorem provides that, given a nominal plant P̂ and a base controller k(·) that stabilizes P̂, the augmented controller `k(·) + Q(·)` stabilizes the closed-loop system for all stable (contracting) operators Q. This converts the hard problem of "learn a stable control policy" into an unconstrained problem of "learn any contracting mapping." The stability is structural, not emergent from training.

Contractivity is defined in discrete time as: a system is contracting if any two trajectories from different initial conditions satisfy `|x_1t - x_2t| ≤ β·α^t·|x_10 - x_20|` for β ∈ ℝ⁺ and α ∈ [0,1). This is the discrete-time analog of Lyapunov contraction.

### Recurrent Equilibrium Network (REN) Architecture

The REN is defined by the implicit system:

```
[x_{t+1}]   [A    B_1  B_2 ] [x_t]   [b_x]
[v_t    ] = [C_1  D_11 D_12] [ω_t] + [b_v]
[y_t    ]   [C_2  D_21 D_22] [u_t]   [b_y]

ω_t = σ(v_t)   (element-wise nonlinearity)
```

Contractivity is enforced by requiring D_11 to be strictly lower triangular (acyclic structure), which ensures the implicit equilibrium equation has a closed-form solution and that the network is contracting by construction. The weight matrices are parameterized via Cayley transforms and LMI-based reparameterization so that unconstrained gradient descent preserves contractivity.

The architecture used in the paper:

| Parameter | Value |
|-----------|-------|
| Internal states n | 32 |
| Input dimension m | 96 |
| Equilibrium dimension q | 256 |
| Output dimension p | 3 (force correction) |

### Base Controller: Differential Flatness-Based Control (DFBC)

The DFBC position control law is:

```
F_des = m·(ẍ_ref - K_v·(ẋ - ẋ_ref) - K_x·(x - x_ref) + g·e_3)
```

with attitude control on SO(3) using geometric error. The reference feedforward term `r_t` in the full control law derives from trajectory derivatives. Specifically, the jerk feedforward term for angular velocity is computed via differential flatness:

```
m·R^T·j = [T·Ω_y^B, -T·Ω_x^B, Ṫ]^T
```

where j is position jerk (third derivative of position), T is total thrust, R is rotation matrix, and Ω^B is body angular velocity. This projects the jerk onto the body frame to generate angular velocity references, enabling the controller to anticipate attitude changes needed for aggressive maneuvers.

### Jerk Feedforward: Role and Derivation

The jerk feedforward term is the highest-order derivative used in the controller. It appears in the computation of Ω_x^B and Ω_y^B (body angular velocity references), which are derived by differentiating the thrust direction vector. The derivation follows from the kinematic differential flatness property: thrust direction and its time derivatives are fully determined by the trajectory and its derivatives up to snap (fourth order) in flat-output space, but only jerk is needed to generate Ω^B references.

Jerk feedforward is particularly important for aggressive trajectories where the desired attitude changes rapidly between waypoints. Without it, the attitude controller always lags the required orientation, producing tracking error even when position tracking is tight. Including jerk allows proactive attitude corrections before the positional error has time to accumulate.

### Why Snap Feedforward Was Rejected

The paper explicitly evaluated higher-order derivative feedforward terms and concluded: "Any higher-order feedforward terms did not improve the tracking performance in our real-world experiments." This means snap (fourth derivative) and higher orders were tested and found to provide no benefit.

The explanation is implied by the noise considerations: each additional derivative amplifies measurement noise by a factor proportional to the signal bandwidth. Snap is the fourth derivative of position; if position is measured or estimated at 100 Hz with typical IMU noise, snap estimates will be extremely noisy and the signal-to-noise ratio becomes unfavorable. The nominal improvement from snap's anticipation of attitude rate changes is overwhelmed by the noise-induced control chattering. Jerk represents the practical limit where the feedforward benefit still exceeds the noise cost.

### Noise Handling in Derivative Signals

The paper does **not** explicitly specify filtering parameters (cutoff frequencies, filter orders) for derivative signal conditioning. This is a notable gap. However, several implicit choices reveal the practical approach:

1. **Motor model as de-facto low-pass filter**: The motor dynamics are modeled as a first-order system with time constant τ_mot = 30 ms. This effectively limits the bandwidth of any actuator command to roughly 1/(2π·0.030) ≈ 5.3 Hz. Since the controller commands thrust (not directly rotor speed), the motor model acts as an inherent low-pass filter on control inputs, attenuating high-frequency noise that would otherwise excite structural modes.

2. **Control rate choice (20 Hz)**: The outer position loop runs at 20 Hz (50 ms update period). At 20 Hz, the Nyquist frequency is 10 Hz, which is already below the motor bandwidth. This conservative choice suppresses aliasing of high-frequency derivative noise before it reaches the controller.

3. **Simulation-based training avoids real-sensor noise**: The REN Q is trained in simulation (RK4 at 100 Hz) where state derivatives can be computed analytically with zero measurement noise. The stability guarantee then ensures that bounded real-world noise enters as a bounded disturbance, which Q was trained to reject. This sidesteps explicit derivative filtering by absorbing noise rejection into Q's learned behavior.

4. **Residual state x̃_t as implicit filter**: Rather than computing velocity or acceleration derivatives directly, the controller uses x̃_t = x_t - x̂_t. The predicted state x̂_t comes from integrating the nominal dynamics forward, which is a smooth, noise-free prediction. The residual is thus the difference between a noisy measurement and a smooth prediction, which tends to be smoother than raw derivatives.

The implicit lesson: at 20 Hz, the derivative chain from position to jerk involves only one differentiation step (position → velocity → acceleration → jerk), and the low update rate provides natural anti-aliasing. Snap would require two more differentiations of already-noisy jerk estimates, explaining why it was unhelpful.

### Training

**Loss**: Quadratic tracking cost analogous to NMPC:
```
L = Σ_t [ Q_p‖p_t - p_ref,t‖² + Q_v‖v_t - v_ref,t‖² + R_u‖u_t‖² ]
```

**Optimization**: Analytic Policy Gradient (APG) — direct backpropagation through the RK4 simulation dynamics. No RL reward shaping or sampling variance.

**Dataset**: 100,000 Lissajous trajectories; amplitudes A_x, A_y ∈ [0, 20] m, A_z ∈ [0, 3] m; frequencies ω ∈ [0, 5] rad/s; trajectories exceeding motor limits by >10% filtered out.

**Disturbances during training**: Random constant forces < 20 N, random torques < 0.1 Nm, with 20% Gaussian noise on each.

**No domain randomization**: Stability guarantee absorbs sim-to-real gap structurally.

## Results

### Tracking Accuracy (Position RMSE in meters)

**Feasible trajectories** (within actuator limits):

| Trajectory | NGTC | NMPC | DFBC |
|------------|------|------|------|
| Horizontal Loop | **0.20** | 0.23 | 0.25 |
| Vertical Loop | **0.17** | 0.18 | 0.20 |
| Lemniscate | 0.15 | 0.22 | **0.14** |

NGTC matches or beats NMPC on feasible trajectories. DFBC edges ahead on the lemniscate (purely linear trajectory, no actuator saturation).

**Infeasible trajectories** (exceeding actuator limits — the critical regime):

| Trajectory | NGTC | NMPC | DFBC | NGTC improvement over DFBC |
|------------|------|------|------|----------------------------|
| Horizontal Loop* | **1.42** | 1.77 | 2.39 | -40.6% |
| Vertical Loop* | 1.19 | **1.06** | 5.47 | -78.2% |
| Lemniscate* | 1.13 | **0.84** | 2.04 | -44.6% |

NGTC beats DFBC by 40–78% on infeasible trajectories. NMPC wins on some infeasible cases by leveraging its full optimization horizon, but NGTC is 5× faster.

### Robustness to Perturbations (Table III — 30 runs each)

| Perturbation | NGTC RMSE | DFBC RMSE | NMPC RMSE | NGTC crash | DFBC crash | NMPC crash |
|---|---|---|---|---|---|---|
| +50% drag | **0.67 m** | 0.74 m | 0.73 m | **0%** | 7% | 0% |
| +30% motor τ | 0.44 m | 0.49 m | **0.42 m** | 0% | 0% | 0% |
| -30% mass | **0.59 m** | 0.71 m | 0.73 m | **0%** | 3% | 3% |
| +30% mass | **0.62 m** | 0.89 m | 0.78 m | **0%** | 7% | 10% |
| 10 N external force | **0.22 m** | 0.33 m | 0.29 m | 0% | 0% | 0% |
| 15 N external force | **0.34 m** | 0.65 m | 0.71 m | 3% | — | 27% |

Under 15 N lateral persistent force (strong wind), NMPC crashes 27% of runs and NGTC crashes only 3% with 0.34 m RMSE — a 48% reduction. Under +50% drag, NGTC eliminates the 7% DFBC crash rate entirely.

### Computational Efficiency

| Controller | Inference time | Platform |
|------------|---------------|---------|
| DFBC | < 0.025 ms | i7 single core |
| NGTC (DFBC + REN) | **0.576 ms** | i7 single core |
| NMPC | ~3.0 ms | i7 single core |

NGTC adds only 0.55 ms overhead over pure DFBC and runs 5.2× faster than NMPC. At 0.576 ms, the REN forward pass consumes < 6% of a 10 ms budget at 100 Hz.

### Real-World Deployment

Deployed on Jetson Orin Nano without domain randomization or fine-tuning. The paper demonstrates sim-to-real transfer on aggressive race-style trajectories with mid-flight wind disturbance. NGTC maintained trajectory while NMPC became unstable and DFBC exhibited significant error recovery lag.

## Relevance to Our System

Our stack uses a geometric PD controller with feedforward acceleration in `control/mpc_tracker.py`. This is architecturally identical to the DFBC base controller in this paper (PD position gains + acceleration feedforward from differential flatness). NGTC is therefore a direct neural augmentation on top of what we already have.

**ILC offset computation and second derivatives**: Our ILC pipeline computes position correction offsets that need to be differentiated to compute acceleration corrections for feedforward. This paper's treatment of derivative noise is directly relevant. The key lesson: differentiating once (offset → velocity correction) is feasible; differentiating twice (offset → acceleration correction) will amplify noise significantly at 100 Hz operation. The paper's implicit solution — using a 20 Hz outer loop and a smooth nominal predictor x̂_t to compute residuals rather than raw derivatives — translates directly: we should compute ILC acceleration corrections by integrating the ILC position offsets backward through the nominal dynamics (computing what feedforward acceleration change would produce the desired position offset) rather than double-differentiating the offsets in the time domain.

**Filtering for second derivatives**: Although the paper does not give explicit filter parameters, the combination of (a) first-order motor model at τ=30 ms, (b) 20 Hz outer loop rate, and (c) residual-based neural correction effectively acts as a cascade of low-pass filters. For our 100 Hz loop, if we compute ILC acceleration offsets from position offset data, we should apply at least a second-order low-pass filter (Butterworth or Savitzky-Golay) with cutoff around 5–10 Hz before using any second-derivative quantity in the controller. The motor time constant of 30 ms corresponds to a 5.3 Hz bandwidth — this is the physical limit that defines how fast we can actually act on acceleration estimates.

**Gate-7 helix and infeasible trajectory regime**: The 40–78% RMSE improvement on infeasible trajectories maps directly to our helix section performance. If gate-7's demanded lateral acceleration momentarily exceeds our thrust envelope, DFBC saturates and lags, while an NGTC-style augmentation would compensate. This is likely the dominant source of our current 0.284 m gate-7 tracking error.

**No domain randomization, stable training**: Because the closed-loop is provably contracting for any Q, we can train the REN augmentation aggressively in our PyBullet simulation without careful reward shaping or stability monitoring. Failed training attempts cannot destabilize the vehicle, enabling faster iteration.

**REN timing at 100 Hz**: At 0.576 ms on an i7, the REN forward pass costs approximately 2–4× more on Jetson-class hardware (~1.2–2.3 ms). At 100 Hz, that is 12–23% of the 10 ms budget. With a smaller REN (n=16 internal states vs. n=32), the overhead would be approximately halved.

## Actionable Takeaways

1. **Use jerk feedforward, do not add snap feedforward.** The paper's explicit finding that snap and higher-order terms did not help in real-world experiments is a direct recommendation. Our current feedforward chain should stop at acceleration (second derivative). Adding snap would require computing the fourth derivative of position, amplifying noise by a factor of roughly (2πf)⁴ and providing no trajectory tracking benefit.

2. **For ILC acceleration corrections, compute them from trajectory-space inverse dynamics, not from time-domain differentiation.** Instead of differentiating ILC position offsets twice to get acceleration corrections, express the desired position offset as a modified trajectory and compute the differential-flatness feedforward acceleration analytically. This avoids double differentiation of noisy offset estimates.

3. **Apply a low-pass filter with cutoff ≤ 5–10 Hz before using any second derivative quantity.** The 30 ms motor time constant (5.3 Hz physical bandwidth) defines the practical upper limit. A Butterworth or Savitzky-Golay filter at 5 Hz on ILC position offsets before differentiation will prevent noise amplification from degrading the acceleration feedforward.

4. **Augment `control/mpc_tracker.py` with a contracting REN.** Keep existing PD gains. Add a small REN (n=16 states, p=3 outputs) operating on x̃_t = x_t - x̂_t and outputting a 3D force correction added before attitude extraction. Use the Youla parameterization structure for guaranteed stability.

5. **Use analytic policy gradient (APG) training via differentiable PyBullet.** Backpropagate through the RK4 simulator with a quadratic tracking cost. Avoid RL — APG gives lower-variance gradients and requires no reward shaping. Train on our actual gate trajectories (helix, S-turn), not just Lissajous curves.

6. **Train the REN with persistent disturbance forces representative of gate wash.** Scale the paper's 20 N training forces to our drone mass: for a 0.72 kg drone, 20 N = 2.8 g lateral acceleration; for our drone, scale proportionally. Target 3–5 N constant lateral force during training, which represents realistic gate-wash effects.

7. **Validate specifically on the infeasible regime.** After training, test on the gate-7 helix section as an "infeasible" trajectory — measure position RMSE with and without the REN augmentation. This should show the 40–78% improvement range observed in the paper.

8. **Consider residual-based state representation over raw error.** Our current controller uses direct position error (x - x_ref). The NGTC residual x̃_t = x_t - x̂_t is a richer signal that captures not just current error but the deviation from expected closed-loop behavior. This makes Q more informative and training more sample-efficient.

## Limitations & Caveats

**20 Hz outer loop mismatch**: The paper's controller runs at 20 Hz (50 ms), which is 5× slower than our 100 Hz target. The conservative rate choice suppresses derivative noise by providing natural anti-aliasing — the jerk computation is well-conditioned at 20 Hz but noisier at 100 Hz. At 100 Hz, the same jerk feedforward may require explicit low-pass filtering that the paper does not specify.

**No explicit filter parameters**: The paper does not give filter cutoff frequencies, filter orders, or any discussion of signal conditioning for derivative signals. All implicit filtering comes from the 20 Hz loop rate and motor model. For our 100 Hz system, this gap must be filled with explicit design choices.

**Snap rejected empirically, not theoretically**: The conclusion that snap feedforward is unhelpful is empirical, specific to their hardware and 20 Hz rate. At higher control rates with lower-noise state estimation, snap might be beneficial. We should not treat this as a general rule.

**No gate-passing evaluation**: The paper benchmarks on loops and lemniscates, not on gate-traversal tasks with pass/fail criteria. RMSE improvements may not translate directly to gate pass rate improvements if the residual errors are spatially concentrated at the gate plane.

**REN outputs force only, not torque**: The 3-output REN corrects linear force but not torques. For our helix with significant yaw demands, a 4D or 6D output REN (or a separate yaw-axis correction) would be needed.

**Training distribution**: Lissajous trajectories with |A_x|, |A_y| ≤ 20 m, |A_z| ≤ 3 m, ω ≤ 5 rad/s. Our helix may exceed the vertical amplitude or frequency range. Explicitly including helix-shaped training trajectories is necessary to cover this regime.

**Sim-to-real gap assumed bounded**: The stability guarantee requires that the sim-to-real model mismatch enters as a bounded disturbance. If our PyBullet model has systematic errors (e.g., wrong drag model, incorrect motor response), the disturbance may not be bounded in the sense required. The paper's validation on a single hardware platform may not generalize to all sim-to-real gaps.

**Jetson timing not reported**: The 0.576 ms figure is measured on an i7. For a Jetson Orin Nano or Nano Super (competition-class hardware), expect 2–4× slowdown. Budget planning should assume ~1.5–2.3 ms for the REN forward pass.

## Key Parameters / Constants

| Parameter | Value | Usage |
|-----------|-------|-------|
| K_x (position proportional gain) | (18, 18, 18) N/m | DFBC base controller |
| K_v (velocity derivative gain) | (8, 8, 8) N·s/m | DFBC base controller |
| k_q,xy (attitude proportional) | 150 rad/s² per rad | Attitude control |
| k_q,z (yaw proportional) | 3 rad/s² per rad | Yaw control |
| Angular velocity damping gains | (20, 20, 8) | Rate damping |
| REN internal states n | 32 | Network architecture |
| REN input dimension m | 96 | Residual state features |
| REN equilibrium dimension q | 256 | Implicit layer width |
| REN output dimension p | 3 | 3D force correction |
| Motor time constant τ_mot | 30 ms | First-order motor model (~5.3 Hz bandwidth) |
| Outer control loop rate | 20 Hz (50 ms) | Outer loop rate |
| Simulation integration rate | 100 Hz | RK4 step |
| Training dataset size | 100,000 trajectories | Lissajous corpus |
| Training disturbance forces | < 20 N | Persistent lateral force |
| Training disturbance torques | < 0.1 Nm | Persistent torque |
| Training trajectory amplitude A_x, A_y | [0, 20] m | Lissajous range |
| Training trajectory amplitude A_z | [0, 3] m | Vertical Lissajous range |
| Training trajectory frequency ω | [0, 5] rad/s | Frequency range |
| Infeasibility threshold | > 10% over motor limit | Trajectory filter cutoff |
| NGTC inference time (i7) | 0.576 ms | Computational budget |
| NMPC inference time (i7) | ~3.0 ms | Baseline computation |
| Drone mass (paper) | 0.72 kg | Physical parameter |
| Thrust limits (paper) | 0–8.5 N per rotor | Actuator constraint |
| Moment of inertia (paper) | diag(2.5, 2.1, 4.3) g·m² | Physical parameter |
| Horizontal Loop* RMSE — NGTC | 1.42 m | Infeasible tracking result |
| Horizontal Loop* RMSE — DFBC | 2.39 m | Infeasible tracking baseline |
| Vertical Loop* RMSE — DFBC | 5.47 m | Worst-case DFBC failure |
| 15 N disturbance RMSE — NGTC | 0.34 m | Wind disturbance rejection |
| 15 N crash rate — NMPC | 27% | NMPC fragility under disturbance |
| +50% drag crash rate — NGTC | 0% | Robustness result |
| +50% drag crash rate — DFBC | 7% | Robustness baseline |
| Tilt angle constraint β | 56° | Maximum tilt from vertical |
