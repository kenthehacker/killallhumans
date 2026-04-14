# DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control
- **URL**: https://arxiv.org/abs/2310.09053
- **Authors**: Kevin Huang, Rwik Rana, Alexander Spitzer, Guanya Shi, Byron Boots
- **Year**: 2023
- **Venue**: CoRL 2023

---

## Key Contribution

DATT introduces a unified feedforward-feedback-adaptive control architecture for quadrotor trajectory tracking that combines a learned feedforward policy (trained via PPO reinforcement learning) with an L1 adaptive control disturbance estimator. The central claim is that precise trajectory tracking — including on infeasible, non-smooth reference paths — can be achieved without requiring trajectory derivatives (velocity, acceleration, jerk, snap) by instead encoding a short horizon of future reference positions directly into the neural network policy. This sidesteps the differential flatness requirement common to classical MPC and geometric controllers, while simultaneously handling unknown dynamics through adaptive compensation. The system achieves sub-10 cm tracking error on hardware in real wind disturbances, with inference at 3.17 ms — roughly 4x faster than competing MPC baselines.

---

## Technical Approach

### Feedforward-Feedback-Adaptive Structure

DATT decomposes control into three additive components:

**Feedforward (learned)**: A convolutional encoder phi processes the next H=0.6 seconds of reference positions (10 future waypoints, sampled at 60 ms intervals) transformed into the drone's current body frame. Three Conv1D layers (16 filters, kernel size 3) compress this into a 32-dimensional embedding capturing trajectory shape, curvature, and future direction. This embedding is concatenated with current state and fed into a 3-layer MLP (64 neurons/layer, ReLU) that outputs a collective thrust + 3-axis angular rate command.

The key insight for curvature handling: by expressing future waypoints in the body frame rather than world frame, the network implicitly sees local curvature geometry. A straight segment looks different from a sharp turn in body coordinates, even if world-frame coordinates differ only in absolute position. The paper found that body-frame representation is mandatory — world-frame encoding caused training to fail entirely. The encoder thus learns a trajectory shape embedding without ever being given explicit curvature, snap, or jerk signals.

**Feedback (explicit)**: The position error `R_t^T (p_t - p_t^d)` (tracking error rotated into body frame) is concatenated directly as a separate input to the MLP alongside the feedforward embedding. This provides reactive correction on top of the anticipatory feedforward. Notably the paper finds this explicit error input slightly improves accuracy even though the error could theoretically be inferred from the feedforward context — suggesting the two channels encode different timescales of information (anticipatory vs reactive).

**Adaptive (L1)**: An L1 adaptive controller estimates translational disturbances (wind, drag, payload) in real-time using closed-loop velocity prediction. The estimated disturbance is low-pass filtered and added as a feedforward compensation term. This module requires no fine-tuning and updates online at control frequency. Importantly, the paper compared L1 adaptation against RMA (Rapid Motor Adaptation — a learned adaptation module), and L1 won on hardware despite RMA winning in simulation. The explanation is state-action distribution shift: RMA's learned adaptation network overfits the simulation distribution and degrades under the domain gap, while L1's model-based estimator is distribution-agnostic.

### Policy Training

The policy is trained with PPO (20M steps, ~3 hours on a single RTX 3080). A critical curriculum trick: the first 2.5M steps use a single fixed reference trajectory before randomizing. Randomizing trajectories from the start produced 8.5x worse final performance, indicating the policy needs an initial stable target to bootstrap its value function estimate. Training mixes degree-5 polynomial trajectories (smooth, feasible) and zigzag waypoint sequences (infeasible), ensuring the policy generalizes across the smooth-to-infeasible spectrum.

The reward is: `r = -||p - p^d|| - 0.5|yaw_error| - 0.1||v||` plus a small bonus for surviving the episode. No snap or jerk penalties. Disturbances during training are Brownian-motion force perturbations with covariance Σ=0.01I, bounded to [-3.5, 3.5] m/s².

---

## Results

**Without disturbances (Table 1, mean ± std tracking error in meters):**
- DATT on smooth trajectories: **0.049 ± 0.017 m** vs. MPC 0.088 m, geometric controller 0.104 m
- DATT on infeasible trajectories: **0.083 ± 0.023 m** vs. MPC 0.181 m; nonlinear controller crashed
- Inference time: **3.17 ms** vs. 12.62–13.10 ms for MPC variants

**With wind + drag plate (Table 2):**
- DATT smooth+wind: **0.095 ± 0.053 m** — 34–48% lower error than adaptive baselines
- DATT infeasible+wind: **0.161 ± 0.056 m** — 34% better than L1-MPC at 1/4 the compute cost

**Ablation (feedforward contribution):**
- Removing the feedforward encoder and using only feedback+adaptive degrades smooth-trajectory error from 0.049 to ~0.071 m (+45%). On infeasible trajectories the degradation is larger, confirming the encoder is doing meaningful anticipatory work beyond what reactive feedback can provide.

---

## Relevance to Our System

Our current system uses a geometric SE(3) tracker (`control/mpc_tracker.py`) with pre-computed min-snap trajectories (`planning/trajectory_optimizer.py`). The DATT architecture is directly relevant in two ways:

1. **Feedforward trajectory context**: Our geometric controller currently receives only the instantaneous setpoint (position, velocity, acceleration from the polynomial). DATT shows that providing a short horizon of future positions as a learned feedforward embedding — even without explicit derivatives — substantially reduces tracking error (45% on smooth trajectories). For our gate-to-gate segments, we already have the full trajectory available; adding a lookahead window to the controller is straightforward.

2. **Adaptive disturbance rejection**: Our EKF handles state estimation but our controller has no explicit disturbance compensation loop. The L1 adaptive module in DATT is model-based and requires no training — it could be bolted onto our existing geometric controller as a translational force compensator, directly addressing wind or unmodeled aerodynamic effects in the PyBullet sim.

The architecture is also relevant to our gate tracking error problem. Per-gate errors at sharp turns (gate 3, gate 7) are highest in our benchmarks. DATT's body-frame horizon encoder naturally handles high-curvature segments by seeing the upcoming turn in local coordinates — exactly the information our current controller lacks.

---

## Actionable Takeaways

1. **Add a trajectory lookahead window to the geometric controller.** Express the next N=10 future reference positions in the current drone body frame and pass them as additional inputs. Even a simple MLP mapping this context to a thrust correction offset (no RL required) could reduce tracking error at high-curvature gates. Start with a hand-tuned centripetal feedforward: `F_ff = m * v^2 / R * n_hat` where R is the radius of curvature estimated from the lookahead positions.

2. **Implement L1 adaptive compensation.** The L1 estimator is a simple first-order velocity predictor + low-pass filter with no training required. It can be added to `control/mpc_tracker.py` as a translational force correction. Expected improvement: 34–48% error reduction in disturbance conditions per DATT's results.

3. **Body-frame lookahead is mandatory.** If implementing a feedforward encoder, transform future waypoints to body frame before encoding. World-frame encoding failed in DATT's ablations. This is consistent with SE(3) equivariance — the control signal should be frame-independent.

4. **Curriculum for any learned components.** If we add RL to tune controller gains or learn a feedforward correction, start training on a single fixed trajectory for the first ~10% of steps before randomizing. This dramatically stabilizes early learning (8.5x performance difference in DATT).

5. **Do not replace L1 with a learned adapter for hardware/sim transfer.** DATT's RMA vs. L1 comparison is a cautionary tale: learned adaptation performs better in simulation but worse on hardware due to distribution shift. For our PyBullet-to-competition pipeline, a model-based L1 compensator is more robust than a learned disturbance encoder.

---

## Limitations & Caveats

- **Planar-only trajectories**: DATT's hardware experiments are confined to xy-plane trajectories due to the Crazyflie's low thrust-to-weight ratio (<2). Our racing drone has a higher TWR, but the paper doesn't validate 3D aerobatic maneuvers.
- **Simplified simulator**: The sim uses a first-order thrust/rate delay model. Real propeller dynamics (blade flapping, rotor inflow) are unmodeled. This limits the agility of learned behaviors that transfer zero-shot.
- **Brownian disturbance model**: Training disturbances are temporally correlated Brownian noise, but real wind has structured spatial coherence. The L1 estimator handles real wind better than the learned policy component, suggesting the policy's robustness to structured disturbances may be limited.
- **Training sensitivity**: The paper reports high variance across PPO seeds and requires careful hyperparameter tuning. Reproducing results requires the exact curriculum schedule and body-frame normalization. This is a practical barrier to integrating RL components into our pipeline.
- **3.17 ms inference on RTX 3080**: Inference time will be higher on embedded hardware (Jetson, Pixhawk). At 50 Hz control, the budget per step is 20 ms, so the network fits, but tight.
- **No gate-passing task**: DATT evaluates on trajectory tracking error, not gate passage. It does not address the discrete gate detection and sequencing problem central to our competition setup.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Feedforward horizon H | 0.6 s | Future positions window |
| Reference waypoints per step | 10 | Spaced at 60 ms intervals |
| Conv1D layers in encoder | 3 | 16 filters, kernel size 3 |
| Encoder output dimension | 32 | Trajectory embedding size |
| MLP hidden layers | 3 x 64 neurons | ReLU activations |
| Control rate | 50 Hz | dt = 0.02 s |
| PPO training steps | 20M | ~3 hours on RTX 3080 |
| Curriculum warmup steps | 2.5M | Fixed trajectory before randomizing |
| Disturbance covariance | Σ = 0.01I | Brownian motion magnitude |
| Disturbance bound | ±3.5 m/s² | Translational force disturbance |
| Reward: position weight | 1.0 | Tracking error penalty |
| Reward: yaw weight | 0.5 | Yaw error penalty |
| Reward: velocity weight | 0.1 | Velocity magnitude penalty |
| L1 time delay constant k | 0.4 | Adaptation filter parameter |
| Tracking error (smooth, no wind) | 0.049 ± 0.017 m | Best hardware result |
| Tracking error (infeasible, wind) | 0.161 ± 0.056 m | Hardest condition |
| Inference time | 3.17 ms | vs. 12.62 ms for MPC |
