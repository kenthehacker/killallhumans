# Reinforcement Learning Based Prediction of PID Controller Gains for Quadrotor UAVs
- **URL**: https://arxiv.org/abs/2502.04552
- **Year**: 2025
- **Venue**: arXiv preprint

---

## Key Contribution

This paper presents a DDPG-based (Deep Deterministic Policy Gradient) online gain-tuning framework that fine-tunes the PID gains of a cascaded quadrotor attitude controller during flight. The central claim is that RL can close the loop on gain adaptation in real time, dynamically adjusting inner-loop proportional and derivative gains in response to observed attitude error, outperforming a carefully hand-tuned baseline by 12–34% in RMSE across simulation and real outdoor flight.

The contribution is framed as a practical alternative to lengthy manual tuning: rather than searching for static optimal gains offline, an actor network learns a continuous policy that maps the current 12-dimensional state vector to five normalized gain-adjustment outputs, which are then blended into the running PID computation at each 0.05 s control step.

A secondary contribution is the deployment methodology — the network is code-generated to C++ and flashed onto a Pixhawk 2.1 Cube Black running PX4, demonstrating that a 128×128 two-hidden-layer actor network is compact enough to run on embedded flight hardware with tight memory budgets.

---

## Technical Approach

### Controller Architecture

The paper uses a standard cascaded position–attitude controller common in PX4-based systems:

- **Outer loop (position)**: proportional-derivative control on position errors, producing desired attitude references.
- **Inner loop (attitude)**: dual-proportional + derivative (P–P–D) structure separately for roll/pitch, yaw, and altitude channels.

The RL agent only adjusts inner-loop attitude gains (not position gains), which is a deliberate design choice — attitude dynamics are faster and less stable, so online adaptation is highest-value there.

### State Space (S ∈ ℝ¹²)

The agent observes:
- World-frame position (x, y, z)
- Euler angles (roll φ, pitch θ, yaw ψ)
- Position errors (ex, ey, ez) relative to the reference trajectory
- Attitude errors (eφ, eθ, eψ)

### Action Space (A ∈ ℝ⁵, range [-1, 1])

Five normalized weight adjustments for inner-loop gains:
- nP₁,φθ — roll/pitch outer proportional scaling
- nP₁,ψ — yaw outer proportional scaling
- nP₂,φθ — roll/pitch inner proportional scaling
- nP₂,ψ — yaw inner proportional scaling
- nD,φθ — roll/pitch derivative scaling

Gain update equation: **k_new = k_original × (1 + 0.4 × n)**

The parameter `a = 0.4` (the search rate) bounds the RL output to ±40% of the hand-tuned baseline. This is a key hyperparameter: too large allows instability, too small limits adaptation benefit.

### Reward Function

Piecewise step function on attitude error norm ‖eη‖ (radians):

| Condition | Reward |
|-----------|--------|
| ‖eη‖ ≥ 0.04 | −25 |
| 0.01 ≤ ‖eη‖ < 0.04 | −15 |
| 0.001 ≤ ‖eη‖ < 0.01 | −10 |
| 0.0005 ≤ ‖eη‖ < 0.001 | −5 |
| 0.0001 ≤ ‖eη‖ < 0.0005 | −1 |
| ‖eη‖ < 0.0001 | +10 |

This is a purely attitude-focused reward — position tracking is not directly rewarded, only attitude accuracy. This is appropriate for the inner-loop framing but means position tracking improvement is only indirect.

### Network Architecture

**Actor:**
- Input: 12 neurons
- Hidden 1: 128 neurons, tanh activation
- Hidden 2: 128 neurons, tanh activation
- Output: 5 neurons, tanh activation (bounded [-1, 1])

**Critic:**
- Input: 12 neurons (state only — note: does not receive action as input, which deviates from standard DDPG where critic takes both state and action)
- Hidden 1: 128 neurons, ReLU
- Hidden 2: 128 neurons, ReLU
- Output: 1 neuron (Q-value estimate)

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sampling time | 0.05 s (20 Hz control loop) |
| Discount factor γ | 0.99 |
| Actor learning rate | 1 × 10⁻³ |
| Critic learning rate | 1 × 10⁻³ |
| L₂ regularization | 1 × 10⁻⁵ |
| Optimizer ε | 1 × 10⁻⁸ |
| Minimum batch size | 1024 |
| Experience replay buffer | 10⁶ transitions |

Training is conducted entirely in MATLAB/Simulink using the UAV Toolbox with a Newton-Euler dynamics model. The trained actor is then exported to C++ for hardware deployment.

### Test Trajectory

A 45-second scenario: 10 s takeoff → 2.5 s hover → 20 s circular lap → 2.5 s hover → 10 s landing. The circular trajectory exercises sustained attitude error in a repeatable, measurable way.

---

## Results

### Simulation (MATLAB/Simulink)

| Method | RMSE of ‖eη‖ |
|--------|-------------|
| Manual tuning | 12.75 × 10⁻³ rad |
| RL fine-tuning | 11.17 × 10⁻³ rad |
| Improvement | ~12.4% |

### Outdoor Real-Hardware Flight (Pixhawk 2.1, PX4)

| Method | RMSE of ‖eη‖ |
|--------|-------------|
| Manual tuning | 33.93 × 10⁻² rad |
| RL fine-tuning | 22.55 × 10⁻² rad |
| Improvement | ~33.5% |

The much larger improvement on real hardware (33% vs. 12%) is noteworthy: it suggests that manual gains are particularly suboptimal in the presence of real-world disturbances (wind, vibration, motor variability), and RL adaptation compensates for these nonlinear effects that the Simulink model does not capture.

### Baseline Gains (hand-tuned, for reference)

| Gain | Value |
|------|-------|
| kP₁,z (altitude outer P) | 8.9 |
| kP₂,z (altitude inner P) | 19.8 |
| kP₁,xy (horizontal outer P) | 0.6 |
| kP₂,xy (horizontal inner P) | 3.9 |
| kD,xy (horizontal D) | 0.29 |
| kP₁,ψ (yaw outer P) | 22.0 |
| kP₂,ψ (yaw inner P) | 5.4801 |
| kP₁,φθ (roll/pitch outer P) | 44.0 |
| kP₂,φθ (roll/pitch inner P) | 11.467 |
| kD,φθ (roll/pitch D) | 0.81905 |

### Vehicle Parameters

| Parameter | Value |
|-----------|-------|
| Mass | 1.2 kg |
| Arm length | 0.225 m |
| Ixx = Iyy | 0.0131 kg·m² |
| Izz | 0.0234 kg·m² |
| Max thrust per rotor | 8.43 N |

---

## Relevance to Our System

Our current setup uses a geometric (SE(3)) tracker with a PD position controller at the outer loop: kp_xy=6, kd_xy=4, kp_z=8, kd_z=5, plus a feedforward acceleration weight of 0.4. These are structurally analogous to the paper's cascaded outer-loop position gains (kP₁,xy=0.6, kP₂,xy=3.9, kD,xy=0.29 in their normalized convention).

Several insights are directly applicable:

1. **The ±40% adaptation band is a reasonable starting bound.** Their search rate a=0.4 (applied multiplicatively as k_new = k_original × (1 + 0.4 × n)) is an empirically validated range. For our kp_xy=6, that implies a plausible search range of [3.6, 8.4]. For kd_xy=4, the range would be [2.4, 5.6]. These bounds confirm our hand-tuned values are in a sane region but suggest headroom remains.

2. **Attitude gains dominate position tracking.** The paper isolates attitude inner-loop gains as the highest-leverage tuning target. In our geometric tracker, the analogous gains are kr=8.0 (attitude proportional) and kw=2.5 (angular velocity damping). These are currently hand-tuned and not subject to any adaptive mechanism. The paper implies these are the highest-value targets for data-driven optimization.

3. **RL-discovered gains outperform hand-tuning in the presence of real disturbances.** In our kinematic sim, drag is a fixed 0.5 coefficient providing "free" velocity damping. If we ever move to a more realistic sim or hardware, the gain improvement from RL is likely to be substantially larger than our current 10–15% iteration headroom.

4. **Feedforward weight 0.4 appears in both systems.** The paper's search rate a=0.4 and our feedforward_accel=0.4 represent the same design philosophy: a modest fraction of the nominal value as a blending coefficient. This likely converges empirically for quadrotors near this mass/inertia range.

5. **20 Hz control loop with 1024-sample batch is tractable.** Our benchmark runs the full pipeline at 50–120 Hz. An RL-tuned adaptation layer running at 20 Hz (one gain update per 5 control steps) would add negligible overhead.

6. **The piecewise step reward is simple to replicate.** If we wanted to train a DDPG agent for our gain adaptation, the reward is straightforward: thresholds on cross-track error (our `avg_tracking_error_m`) rather than attitude error norm. The 12-dimensional state space is similar in structure to what our EKF outputs.

The primary limitation for direct adoption is that this paper trains and deploys on PX4 hardware, and their gain structure (nested P–P–D for attitude) differs from our geometric SE(3) tracker. Direct transfer of their trained policy is not possible. However, the methodology — offline DDPG training in simulation, then transfer to deployment — is directly applicable.

---

## Actionable Takeaways

1. **Run a gain sweep over [kp_xy, kd_xy, kp_z, kd_z] within ±40% of current values.** The paper empirically validates this band as sufficient to capture RL-optimal gains. A 5×5×5×5 grid search (3,125 configurations) at our benchmark's 20-second sim mode would systematically cover this space. Estimated time: ~2 hours.

2. **Tune kr and kw (attitude gains) alongside position gains.** These are currently fixed at kr=8.0, kw=2.5 and are the most likely bottleneck per the paper's analysis. The paper's inner-loop gain ranges suggest quadrotor attitude P gains are typically 10–50× position gains, which is consistent with our kr=8 vs. kp_xy=6 ratio.

3. **Implement adaptive gain scheduling keyed on tracking error.** The simplest form: if avg_tracking_error > threshold, scale kp_xy and kp_z up by a small delta. This is the zero-RL baseline that the paper's method improves upon, and it may already close some of our current error gap.

4. **If implementing RL gain tuning, use DDPG with the paper's exact architecture.** The 128×128 actor with tanh outputs, critic with ReLU, batch size 1024, discount 0.99 are solid defaults validated on hardware. The only change needed is replacing their attitude-error reward with our cross-track-error reward.

5. **The gain update equation k_new = k_original × (1 + 0.4 × n) is a clean implementation pattern.** Rather than outputting absolute gain values (which require careful normalization), output a multiplicative delta bounded to ±40%. This keeps the RL output dimensionless and independent of the absolute gain scale.

6. **Do not expect large simulation gains.** The paper shows only 12% improvement in simulation vs. 33% on hardware. Our kinematic sim is well-modeled and hand-tuning has already iterated 37 times; the marginal improvement from RL in the current sim is likely in the 5–15% range. The real value of RL gain tuning emerges on hardware.

---

## Limitations & Caveats

1. **Attitude-only adaptation.** The RL agent in this paper only adjusts inner-loop attitude gains (5 parameters). It does not adapt position loop gains, feedforward weights, or trajectory parameters. For our system where the trajectory and feedforward are the primary performance drivers, this is a partial solution.

2. **Reward is attitude-only, not trajectory-following.** The reward punishes attitude error, not position tracking error or gate-crossing. For racing, minimizing attitude error is necessary but not sufficient — the drone needs to be at the right *position* with correct *velocity*, and a pure attitude-error reward may not optimize for that.

3. **20 Hz adaptation loop.** The paper's 0.05 s sampling period means gains update at 20 Hz. Our control loop runs at 50–120 Hz. If disturbances occur on the intra-update timescale (e.g., gate proximity effects), the adaptation loop cannot respond in time.

4. **Small absolute improvement in simulation.** The 12.4% RMSE reduction in simulation (12.75e-3 → 11.17e-3 rad) is modest and may not translate meaningfully to our tracking error metric. The hardware improvement (33%) is more compelling but cannot be reproduced without physical hardware.

5. **Critic architecture deviation.** The paper's critic takes only state as input (not state-action pair), which is non-standard for DDPG. This may reduce Q-function quality and make training less stable. The standard DDPG critic architecture (Lillicrap et al., 2015) concatenates state and action; the paper's deviation is not discussed or justified.

6. **No position gain adaptation.** For our system, kp_xy and kd_xy are the highest-level tunable parameters (they directly determine cross-track error, which is our primary benchmark metric). The paper does not address adaptation of these outer-loop gains.

7. **Circular trajectory only.** Results are reported for a single circular test trajectory. Racing involves high-curvature gates, sudden direction changes, and variable speed profiles that are qualitatively different. Generalization of the trained policy to novel trajectories is not evaluated.

8. **MATLAB/Simulink training environment.** The sim-to-real gap between a MATLAB Newton-Euler model and PyBullet dynamics is non-trivial. Policy trained in our PyBullet sim would need re-validation if deployed to the MATLAB/Simulink pipeline.

---

## Key Parameters / Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| a | 0.4 | Search rate (gain adaptation bound: ±40% of baseline) |
| S | ℝ¹² | State space dimension |
| A | ℝ⁵, [-1,1] | Action space (5 normalized gain deltas) |
| γ | 0.99 | RL discount factor |
| αᵃ | 1×10⁻³ | Actor learning rate |
| αᶜ | 1×10⁻³ | Critic learning rate |
| Batch | 1024 | Minimum training batch size |
| Buffer | 10⁶ | Experience replay capacity |
| Ts | 0.05 s | Control/adaptation sampling period (20 Hz) |
| Actor hidden | 128×128, tanh | Actor network layers |
| Critic hidden | 128×128, ReLU | Critic network layers |
| kP₁,φθ | 44.0 | Baseline roll/pitch outer P gain |
| kP₂,φθ | 11.467 | Baseline roll/pitch inner P gain |
| kD,φθ | 0.81905 | Baseline roll/pitch D gain |
| kP₁,xy | 0.6 | Baseline horizontal outer P gain |
| kP₂,xy | 3.9 | Baseline horizontal inner P gain |
| kD,xy | 0.29 | Baseline horizontal D gain |
| kP₁,z | 8.9 | Baseline altitude outer P gain |
| kP₂,z | 19.8 | Baseline altitude inner P gain |
| RMSE improvement (sim) | ~12.4% | Attitude error reduction in simulation |
| RMSE improvement (real) | ~33.5% | Attitude error reduction on hardware |
| k_new equation | k_orig × (1 + a × n) | Gain update formula |
