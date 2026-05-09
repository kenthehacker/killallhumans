# Dynamics-Invariant Quadrotor Control using Scale-Aware Deep Reinforcement Learning

**Authors:** Varad Vaidya, Jishnu Keshavan
**Published:** March 2025, arXiv:2503.09622
**Venue:** IROS 2025

## Key Contribution

This paper presents a deep reinforcement learning framework that achieves dynamics-invariant quadrotor control by directly outputting force/torque commands, bypassing traditional intermediate control layers (PID cascades, angular rate controllers). The central novelty is scale-aware domain randomization: rather than randomizing physical parameters independently, they parameterize randomization by arm length L, leveraging known scaling laws (mass ~ L^3, inertia ~ L^5, torque-thrust ratio ~ L) to maintain physically consistent parameter combinations. This enables a single policy to control quadrotors spanning 30g to 2.1kg — a 70x mass range — with consistent sub-5cm tracking RMSE. The system achieves 85% improvement over DRL baselines in tracking accuracy, validated across 200+ real-world flights on a Crazyflie 2.1 with wind, payloads, and ground effect disturbances.

## Technical Approach

The architecture has four components trained end-to-end with PPO:

**1. Trajectory Encoder.** Processes a finite horizon of H=100 future reference position/velocity deviations through 3 convolutional layers (32 filters each, kernel size 3, stride 1, ReLU) followed by a fully connected layer, producing a 32-dimensional embedding. This captures the "shape" of upcoming trajectory demands — whether the drone needs to prepare for a sharp turn, deceleration, or straight-line sprint. The convolution over the time axis is a clever choice: it extracts temporal patterns (frequency content, acceleration profiles) more efficiently than an MLP would.

**2. Dynamics Encoder (privileged, teacher).** Takes a 10-dimensional physical parameter vector [mass m, inertia J (3D), arm length L, thrust-to-weight ratio TWR, Km/Kx ratio, wind velocity vw (3D)] and maps it through 2 hidden layers (64 neurons, tanh) to an 8-dimensional dynamics embedding. This is only available during teacher training — at deployment, it's replaced by the adaptation module.

**3. Adaptation Module (student).** Processes the last 50 timesteps of state-action history through a 3-layer MLP + 3-layer 1D CNN (64 filters/layer) to predict the dynamics embedding z_hat_t. This is the sim-to-real bridge: the module implicitly infers mass, inertia, and aerodynamic properties from how the drone responds to commands, without requiring system identification.

**4. Base Policy (Actor).** A 3-layer MLP (64 neurons, tanh) that takes pose error, trajectory embedding, and dynamics embedding to output 4-dimensional normalized thrust and torque commands (CTBT — Combined Thrust and Body Torques). Crucially, this outputs in the body frame, avoiding the singularities and computational overhead of attitude parameterization.

**Scale-Aware Domain Randomization.** The key insight: physical parameters of quadrotors are not independent — they are governed by scaling laws. Mass scales as L^3 (volume), inertia as L^5 (mass * length^2), and torque-to-thrust coefficient ratio as L. By sampling arm length L and deriving other parameters from these relationships (with some noise), the randomization covers physically plausible drones rather than wasting training budget on impossible parameter combinations. TWR is randomized between 2.0-2.75. Wind disturbances sampled from U(0, 1.5) + N(0, 0.01), capped at 2.0 m/s during training.

**State Noise Injection:** Position N(0, 0.01), velocity N(0, 0.01), quaternion N(0, 0.005), angular velocity N(0, 0.001).

**Reward Function:**

| Term | Margin | Weight |
|------|--------|--------|
| Position error ||p_d - p|| | 0.75 | 0.50 |
| Proximity reward ||p_d - p|| | 0.05 | 0.1625 |
| Velocity error ||v_d - v|| | 0.50 | 0.20 |
| Angular velocity error | 0.50 | 0.0625 |
| Action smoothness ||a_t - a_{t-1}|| | 0.50 | 0.0375 |
| Yaw error | 0.50 | 0.0375 |
| Crash penalty | -100 | — |

Position error dominates at 50% weight. The dual position terms (error + proximity) create a coarse-to-fine reward landscape: the position error term guides from far away, while the proximity reward (margin 0.05) provides a sharp bonus for sub-5cm accuracy.

## Results

**Simulation (MuJoCo, 100 Hz, 100M training steps, 64 parallel envs):**

Tracking RMSE across arm lengths (cm):

| Trajectory | L=0.05m | L=0.11m | L=0.16m | L=0.21m |
|-----------|---------|---------|---------|---------|
| Butterfly | 2.0±0.1 | 2.0±0.1 | 2.0±0.1 | 2.0±0.1 |
| Satellite | 2.0±0.1 | 1.9±0.1 | 1.9±0.1 | 2.0±0.1 |
| Random Spline | 1.2±0.1 | 1.2±0.1 | 1.2±0.1 | 1.3±0.1 |
| Octahedron | 1.6±0.1 | 1.6±0.1 | 1.7±0.1 | 1.8±0.1 |

Remarkably consistent across a 4x arm length range — the dynamics invariance claim is well-supported.

**vs. Baselines:** 85% improvement over DRL baselines, up to 95% RMSE reduction vs. "Extreme Adaptation" baseline.

**Robustness:** Handles 10 m/s wind (5x training maximum) and 4.5 m/s speed (3x training speeds) — significant out-of-distribution generalization.

**Real-World (Crazyflie 2.1, 39g, >200 flights):**

| Trajectory | Baseline RMSE | Wind RMSE | Payload RMSE |
|-----------|--------------|-----------|-------------|
| Butterfly | 0.044±0.003m | 0.062±0.002m | 0.048±0.004m |
| Satellite | 0.046±0.002m | 0.048±0.004m | 0.049±0.003m |
| Random Spline | 0.051±0.002m | 0.076±0.003m | 0.084±0.003m |
| Octahedron | 0.039±0.002m | 0.043±0.002m | 0.047±0.004m |

Sub-5cm RMSE across all conditions. Wind (2.5 m/s gusts) increases error by ~40% on worst-case trajectories. 5g swinging payload (10% mass) increases error by ~65% on random splines.

**Ground Effect (Table IV):** At 0.08-0.20m altitude, RMSE ranges 0.019-0.054m — the adaptation module successfully compensates for ground-effect disturbances without explicit modeling.

## Relevance to Our System

This paper is relevant primarily through its adaptation architecture and robustness techniques, not as a direct replacement for our control pipeline:

1. **Adaptation module for our EKF.** Their 50-timestep history-based dynamics encoder implicitly performs system identification. We could apply the same concept to our EKF: use a small CNN over recent state-action history to predict a residual correction to our process model. This would capture aerodynamic effects (prop wash near gates, ground effect on low passes) that our current process noise model cannot represent.

2. **Trajectory encoder for gain scheduling.** Their H=100 lookahead trajectory encoder produces a 32-dimensional embedding that conditions the controller. This is a learned version of gain scheduling — the controller automatically adjusts its behavior based on upcoming trajectory demands. For our gate-3 problem, a trajectory-aware controller could preemptively adjust gains before entering the difficult section rather than reacting after error accumulates.

3. **Scale-aware randomization as robustness framework.** Even though we don't need to generalize across drone sizes, the principle of physically-consistent parameter randomization applies to our sim-to-real gap. Rather than independently randomizing mass, inertia, and drag, we should use scaling laws to ensure our domain randomization covers plausible drones.

4. **Direct force/torque output.** Their CTBT output bypasses the PID cascade. While our geometric tracker (SE(3) Lee controller) is already relatively direct, their approach suggests that even the attitude-rate inner loop may be limiting. For future iterations, a learned low-level controller outputting motor commands directly could eliminate cascaded controller delay.

5. **Action smoothness reward.** Weight 0.0375 for ||a_t - a_{t-1}|| — relatively small but present. This confirms that even aggressive RL controllers benefit from explicit smoothness regularization, supporting the idea of adding smoothness terms to our trajectory cost.

## Actionable Takeaways

- **History-based adaptation for EKF:** Implement a lightweight CNN (3-layer, 64 filters) over the last 50 control timesteps to predict a process model correction. This could reduce gate-3 error if prop wash or unmodeled aerodynamics near the gate are a factor.
- **Trajectory lookahead for gain scheduling:** Feed the next 100 reference points through a small encoder to condition our tracker gains. The convolution-over-time architecture (3 conv layers, 32 filters, kernel 3) is lightweight enough for real-time use at 100 Hz.
- **Dual position reward structure:** Their coarse (margin 0.75) + fine (margin 0.05) position terms could inform our ILC target design — large corrections when far from the gate, micro-corrections when within 5cm.
- **Training regime:** 100M steps, 64 parallel envs, 6s episodes at 100 Hz. If we move to learned components, this sets expectations for training scale.
- **Robustness validation protocol:** Test at 5x training wind and 3x training speed to verify out-of-distribution robustness — a useful benchmark methodology for our system.

## Limitations & Caveats

- Maximum tested speed is 2.1 m/s on a 39g Crazyflie — far below racing speeds. The approach is validated for slow, precise tracking rather than aggressive racing. At 15+ m/s, the dynamics become significantly more nonlinear and the 100 Hz control rate may be insufficient.
- The 100M training step requirement (64 parallel MuJoCo envs) is computationally significant. Adapting this to our PyBullet setup would require substantial infrastructure work.
- The adaptation module needs ~50 timesteps (0.5s at 100 Hz) to converge on dynamics estimates. During this period, tracking accuracy may degrade — potentially problematic at racing speeds where 0.5s covers 7-10 meters.
- Real-world validation uses motion capture for state estimation, not onboard sensing. The sub-5cm RMSE numbers would likely degrade with VIO or EKF-based estimation.
- The policy outputs body-frame forces/torques, requiring an accurate mass estimate for the thrust normalization. Errors in mass estimation would directly scale the position tracking error.
- Ground effect results at <0.20m altitude are impressive but not relevant to racing (we fly well above ground effect altitudes near gates).

## Key Parameters/Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Trajectory horizon H | 100 timesteps | Lookahead for encoder |
| Trajectory encoder | 3 conv layers, 32 filters, kernel 3 | 32-dim output |
| Dynamics encoder | 2 hidden layers, 64 neurons, tanh | 8-dim output |
| Adaptation history | 50 timesteps | State-action history |
| Adaptation module | 3-layer MLP + 3-layer 1D CNN, 64 filters | Predicts dynamics embedding |
| Actor network | 3 hidden layers, 64 neurons, tanh | 4-dim CTBT output |
| TWR randomization | 2.0 - 2.75 | Thrust-to-weight ratio |
| Wind randomization | U(0, 1.5) + N(0, 0.01), cap 2.0 m/s | Training disturbances |
| Position noise | N(0, 0.01) | State injection |
| Velocity noise | N(0, 0.01) | State injection |
| Training steps | 100M | PPO, 64 parallel envs |
| Episode duration | 6 seconds at 100 Hz | 600 steps/episode |
| Control frequency | 100 Hz | MuJoCo physics |
| Position error weight | 0.50 | Dominant reward term |
| Proximity margin | 0.05 m | Fine-grained accuracy bonus |
| Action smoothness weight | 0.0375 | Jerk penalty |
| Crash penalty | -100 | Episode termination |
| Real-world RMSE | 0.039 - 0.084 m | Across conditions |
| Sim RMSE | 0.012 - 0.020 m | Across scales |
| Arm length range | 0.05 - 0.21 m | 30g to 2.1kg drones |
