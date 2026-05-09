# Learning Robust Agile Flight Control with Stability Guarantees (NGTC)

- **URL**: https://arxiv.org/abs/2510.12611
- **Authors**: Lukas Pries, Markus Ryll
- **Year**: 2025 (submitted October 14, 2025)

## Key Contribution

This paper introduces the Neural Geometric Tracking Controller (NGTC), a control architecture that fuses a classical geometric feedback controller with a learned neural augmentation while providing formal stability guarantees. The fundamental challenge addressed is that pure learning-based controllers (RL, imitation) achieve high performance but lack stability certificates, while classical geometric controllers (DFBC, SE(3)) are stable but underperform on aggressive or infeasible trajectories. NGTC resolves this tension by applying the Youla Parameterization: any stabilizing controller can be written as a fixed base controller plus a stable operator Q on the residual. By constraining Q to be a Recurrent Equilibrium Network (REN) — a neural architecture that is contracting by construction — the closed-loop system inherits global stability guarantees regardless of what Q learns.

Concretely, NGTC achieves 40% lower tracking error than Differential Flatness-Based Control (DFBC) on infeasible trajectories (those exceeding actuator limits), matches NMPC accuracy on feasible trajectories, runs at 0.576 ms per inference step (5× faster than NMPC), and maintains 0% crash rate under 50% drag perturbation where DFBC crashes 7% of the time. The controller transfers directly from simulation to hardware on a Jetson Orin Nano without domain randomization or fine-tuning.

## Technical Approach

### Control Law Structure

The NGTC control law is:

```
u_t = k(x_t) + Q(x̃_t) + r_t
```

where:
- `k(x_t)` is the base geometric (DFBC) controller — a PD controller using differential flatness
- `Q(x̃_t)` is the REN neural augmentation operating on the residual state `x̃_t = x_t - x̂_t`
- `r_t` is the reference feedforward term
- `x̂_t` is the nominal state trajectory predicted by the base controller in the absence of disturbances

The key insight is that x̃_t represents model error and disturbance: the difference between where the drone *is* and where the unaugmented controller *predicted* it would be. Q learns to correct exactly this discrepancy without touching the base controller's stability structure.

### Youla Parameterization for Stability

The Youla Parameterization theorem guarantees that if the base controller k(·) stabilizes the nominal system, then the closed-loop system with augmented controller k(·) + Q(·) remains stable for all stable (contracting) operators Q. This means:
- Q can be any contracting neural network
- No stability-constrained optimization is needed during training
- Stability is a structural property of Q, not an emergent property of training

This converts the hard constrained optimization problem ("learn a control policy that is stable") into an unconstrained optimization problem ("learn any contracting mapping").

### Recurrent Equilibrium Network (REN) Architecture

The REN is a recurrent neural network whose implicit equilibrium equations are structured to guarantee contractivity (and hence stability). The architecture:

| Parameter | Value |
|-----------|-------|
| Internal states n | 32 |
| Input dimension m | 96 |
| Equilibrium dimension q | 256 |
| Output dimension p | 3 |
| Structure | Acyclic (strictly lower-triangular D₁₁ matrix) |

The acyclic structure means the equilibrium equations have a closed-form solution (no fixed-point iteration needed), enabling constant execution time. The REN outputs 3D force corrections that are added to the base DFBC thrust vector.

Contractivity is enforced by a parameterization of the weight matrices (using Cayley transforms and LMI-based reparametrization) such that *any* unconstrained gradient descent step on the weights produces a contracting network. This makes training with standard Adam/SGD straightforward.

### Base Controller: Differential Flatness-Based Control (DFBC)

The base controller uses differential flatness of the quadrotor system:

- **State**: [position p ∈ ℝ³, velocity v ∈ ℝ³, orientation R ∈ SO(3), angular velocity ω ∈ ℝ³]
- **Position control**: PD law

```
F_des = m·(ẍ_ref - K_v·(ẋ - ẋ_ref) - K_x·(x - x_ref) + g·e₃)
```

- **Attitude control**: geometric error on SO(3)

DFBC gains (Table I from paper):

| Gain | Value | Units |
|------|-------|-------|
| K_x (proportional) | (18, 18, 18) | N/m |
| K_v (derivative) | (8, 8, 8) | N·s/m |
| k_q,xy (attitude P) | 150 | rad/s² per rad |
| k_q,z (yaw P) | 3 | rad/s² per rad |
| ω gains (angular velocity D) | (20, 20, 8) | — |

### Training Procedure

**Loss function**: The NMPC optimal control cost (weighted quadratic tracking + control effort):

```
L = Σₜ [ Q_p‖p_t - p_ref,t‖² + Q_v‖v_t - v_ref,t‖² + R_u‖u_t‖² ]
```

**Optimization method**: Analytic Policy Gradient (APG) — direct backpropagation through the differentiable simulation dynamics (RK4 integrator at 100 Hz). No RL reward shaping or trajectory sampling variance.

**Training dataset**: 100,000 Lissajous trajectories with randomized parameters:
- Amplitudes: A_x, A_y ∈ [0, 20] m; A_z ∈ [0, 3] m
- Frequencies: ω_x, ω_y, ω_z ∈ [0, 5] rad/s
- Filtered to exclude trajectories exceeding motor thrust limits by >10%

**Training disturbances**: Random constant forces < 20 N and torques < 0.1 Nm with 20% Gaussian noise applied during each rollout.

**No domain randomization**: The direct sim-to-real transfer is enabled by the stability guarantee — the closed-loop system is contracting under any bounded disturbance, so sim-to-real gap is handled structurally rather than by training diversity.

### Physical Parameters

| Parameter | Value |
|-----------|-------|
| Quadrotor mass | 0.72 kg |
| Moment of inertia | diag(2.5, 2.1, 4.3) g·m² |
| Thrust limits | 0–8.5 N per rotor |
| Motor time constant | 30 ms (low-pass filter model) |
| Control update rate | 50 ms (20 Hz outer loop) |
| Simulation integrator | RK4 at 100 Hz |

## Results

### Tracking Accuracy (Position RMSE, meters)

**Feasible trajectories** (within actuator limits):

| Trajectory | NGTC | NMPC | DFBC |
|------------|------|------|------|
| Horizontal Loop | **0.20** | 0.23 | 0.25 |
| Vertical Loop | **0.17** | 0.18 | 0.20 |
| Lemniscate | 0.15 | 0.22 | **0.14** |

On feasible trajectories, NGTC matches or beats NMPC and DFBC.

**Infeasible trajectories** (exceeding actuator limits — the hard case):

| Trajectory | NGTC | NMPC | DFBC | NGTC vs DFBC |
|------------|------|------|------|--------------|
| Horizontal Loop* | **1.42** | 1.77 | 2.39 | -40.6% |
| Vertical Loop* | 1.19 | **1.06** | 5.47 | -78.2% |
| Lemniscate* | 1.13 | **0.84** | 2.04 | -44.6% |

On infeasible trajectories, NGTC beats DFBC by 40–78% RMSE. NMPC is slightly better on some infeasible cases (by accessing the full optimization horizon), but NGTC runs 5× faster.

### Robustness Testing (Table III — crash rate and RMSE under perturbations)

| Condition | NGTC RMSE | NGTC crashes | DFBC crashes | NMPC crashes |
|-----------|-----------|--------------|--------------|--------------|
| Nominal | — | 0% | 0% | 0% |
| +50% drag | 0.67 m | **0%** | 7% | — |
| ±30% mass variation | — | **0%** | 3–10% | 3–10% |
| 15 N external force | 0.34 m | 3% | — | 27% |

Under 15 N lateral force (equivalent to strong wind), NMPC crashes 27% of runs; NGTC crashes 3% and maintains 0.34 m RMSE.

### Computational Efficiency

| Controller | Inference time | Platform |
|------------|---------------|---------|
| DFBC | < 0.025 ms | i7 single core |
| NGTC (DFBC + REN) | **0.576 ms** | i7 single core |
| NMPC | ~3.0 ms | i7 single core |

NGTC is 5.2× faster than NMPC and adds only 0.55 ms overhead over DFBC. At 0.576 ms per call, it fits comfortably in our 10 ms budget for a 100 Hz control loop.

### Real-World Validation

Deployed on Jetson Orin Nano (onboard). Racing trajectory tracking with simulated wind disturbance applied mid-flight. NGTC maintained trajectory while NMPC became unstable and DFBC showed significant error recovery lag. No additional training or fine-tuning needed for real-world deployment.

## Relevance to Our System

Our system uses a PD controller with feedforward acceleration in `control/mpc_tracker.py`. The DFBC in this paper is architecturally identical to our current controller (PD position gains + feedforward). NGTC is therefore a direct drop-in neural augmentation on top of what we already have.

**Direct applicability**: We could train a REN Q-network to correct the residual state x̃_t = x_t - x̂_t at each timestep, using our existing DFBC/PD as the base controller k(x_t). The training would use our PyBullet simulation (already available) with our current benchmark trajectories as the dataset. No changes to the base controller gains are needed.

**Gate-7 helix connection**: Infeasible trajectories in this paper correspond to situations where the reference exceeds actuator capacity — exactly what happens at gate-7's helix if the trajectory demands high lateral acceleration. The 40–78% RMSE reduction on infeasible trajectories is directly relevant. Our 0.284 m gate-7 error may partly stem from the trajectory being locally infeasible given our drone's thrust limits, and NGTC's learned augmentation would compensate.

**Disturbance rejection**: The 15 N force rejection (0.34 m RMSE, 3% crash) is relevant for competition environments where gate wash and draft effects create persistent lateral forces. Our current DFBC has no learned disturbance model.

**Stability guarantee = safe experimentation**: Because the closed-loop is provably stable for any contracting Q, we can train NGTC aggressively without fear of destabilizing the vehicle. This is a major practical benefit for rapid iteration.

**REN fits our loop budget**: At 0.576 ms, NGTC fits well within our 10 ms control loop budget (100 Hz). Our current PD + feedforward runs at < 0.1 ms; adding 0.576 ms overhead is fully acceptable.

**No domain randomization needed**: The paper transfers directly from sim to real without domain randomization. Since we train and evaluate primarily in PyBullet sim, the stability guarantee means we can expect real-world generalization without elaborate transfer learning.

## Actionable Takeaways

1. **Augment `control/mpc_tracker.py` with a REN**: Keep the existing PD gains exactly as-is (K_x, K_v). Add a REN forward pass that takes the residual state x̃_t as input and outputs a 3D force correction δF. Apply δF additively to the DFBC desired force before attitude extraction. Start with n=16 internal states (half of paper's 32) to minimize overhead.

2. **Implement analytic policy gradient training**: Use PyBullet + our benchmark trajectory set as the simulation environment. Backpropagate through the RK4 integrator with a quadratic tracking + control cost. Use the 100,000-trajectory Lissajous dataset approach, but generate from our actual gate trajectory shapes (helix, S-turn) rather than generic Lissajous curves.

3. **Train with persistent disturbance forces**: During REN training, inject random constant forces < 5 N (scaled from 20 N to our drone mass) and torques. This teaches Q to compensate for gate-wash effects and wind.

4. **Use the paper's DFBC gains as a starting point**: K_x = (18, 18, 18) N/m, K_v = (8, 8, 8) N·s/m, re-scaled to our drone's mass. Our current gains may be sub-optimal relative to these literature values.

5. **Add infeasibility detection**: Before the control step, check if the reference acceleration exceeds our thrust envelope. When infeasible, allow the REN a larger authority (increase the gain on δF). When feasible, reduce authority to avoid unnecessary neural intervention.

6. **Evaluate on infeasible trajectory subset**: After training, evaluate NGTC specifically on our helix section (gate-7 approach) as an "infeasible" case — this is where the most improvement is expected. Measure RMSE before and after adding the REN augmentation.

7. **Implement the REN weight parameterization**: Use the Cayley transform / lower-triangular D₁₁ structure to ensure the REN is contracting by construction. This is available in open-source REN implementations (e.g., the `contracting_rnn` package).

## Limitations & Caveats

- **20 Hz outer loop**: The paper runs the full NGTC at 20 Hz (50 ms update period). Our system targets 100 Hz. The REN may need to be smaller (n=8–16 states) to maintain sub-10 ms inference at full 100 Hz. At 0.576 ms on a fast i7, on a Jetson Orin Nano the time may be 2–4× longer (~1–2 ms), still feasible.

- **3D force output only (p=3)**: The REN outputs only a 3D force correction, not torque correction. Yaw errors are not directly compensated. For our helix which requires significant yaw rotation, a torque output dimension would be needed (p=4 or p=6).

- **Quadrotor mass 0.72 kg**: The paper's drone is lighter than typical competition drones. Gains K_x = (18, 18, 18) N/m and K_v = (8, 8, 8) N·s/m are tuned for 0.72 kg — they need rescaling proportional to our drone's mass.

- **No gate-passing constraints**: NGTC optimizes tracking accuracy but has no gate clearance or collision avoidance structure. The neural augmentation could in principle learn to avoid gates but this isn't guaranteed. Gate pass rate is not reported.

- **Training data coverage**: The Lissajous training set covers amplitudes up to 20 m and frequencies up to 5 rad/s. Our helix gate involves a helical 3D trajectory that may be outside this distribution. Including helix-shaped trajectories in the training set is recommended.

- **Jetson Orin Nano timing not reported**: The paper uses an i7 for timing (0.576 ms). The actual onboard Jetson Orin Nano inference time is not given. Budget 2–4× overhead for the competition compute platform.

## Key Parameters / Constants

| Parameter | Value | Usage |
|-----------|-------|-------|
| K_x (position P gain) | (18, 18, 18) N/m | DFBC base controller |
| K_v (velocity D gain) | (8, 8, 8) N·s/m | DFBC base controller |
| k_q,xy (attitude P) | 150 | Attitude control |
| k_q,z (yaw P) | 3 | Yaw control |
| Angular velocity gains | (20, 20, 8) | Rate damping |
| REN internal states n | 32 | Network size |
| REN input dimension m | 96 | Residual state features |
| REN equilibrium dim q | 256 | Implicit layer width |
| REN output p | 3 | Force correction axes |
| Training trajectories | 100,000 | Lissajous dataset size |
| Training disturbance forces | < 20 N | Random persistent force |
| Training disturbance torques | < 0.1 Nm | Random persistent torque |
| A_x, A_y amplitude range | [0, 20] m | Training trajectory |
| A_z amplitude range | [0, 3] m | Training trajectory |
| Frequency range | [0, 5] rad/s | Training trajectory |
| Motor time constant | 30 ms | Low-pass filter model |
| Control rate | 20 Hz (50 ms) | Outer loop (they use) |
| RK4 integration rate | 100 Hz | Sim integrator |
| NGTC inference time | 0.576 ms | i7 single core |
| NMPC inference time | ~3.0 ms | i7 single core |
| Quadrotor mass | 0.72 kg | Paper's drone |
| Thrust limits | 0–8.5 N per rotor | Paper's drone |
| Moment of inertia | diag(2.5, 2.1, 4.3) g·m² | Paper's drone |
| Horizontal Loop* RMSE (NGTC) | 1.42 m | Infeasible tracking |
| Horizontal Loop* RMSE (DFBC) | 2.39 m | Infeasible tracking baseline |
| 15 N disturbance RMSE (NGTC) | 0.34 m | Disturbance rejection |
| 15 N crash rate (NMPC) | 27% | Disturbance fragility |
| +50% drag crash rate (NGTC) | 0% | Robustness |
| +50% drag crash rate (DFBC) | 7% | Robustness baseline |
