# Deep Q-Learning-Based Gain Scheduling for Nonlinear Quadcopter Dynamics
- **URL**: https://arxiv.org/abs/2603.03127
- **Year**: 2026

---

## Key Contribution

This paper proposes a deep Q-network (DQN)-based gain-scheduling framework for quadcopter trajectory tracking that bridges model-free reinforcement learning with classical control theory. The central insight is that instead of learning control inputs directly (which risks instability and requires massive safe-exploration overhead), the RL agent learns only to *select gain vectors* from a pre-certified, finite set of stabilizing configurations. The controller architecture remains a flatness-based feedback law; the DQN acts as a supervisory policy that picks which row from a pre-computed gain table to activate at each timestep.

This framing is significant because it decouples the learning problem from the stability problem. All actions in the DQN's action space are provably stabilizing by construction, so the agent cannot learn a policy that destabilizes the vehicle. The learning objective becomes purely performance optimization — minimize tracking error, attitude excursions, and control effort — rather than simultaneously discovering stability and performance. The result is a system that inherits both the adaptivity of RL and the safety guarantees of classical gain scheduling, without requiring trust-region methods, barrier certificates, or other post-hoc safety filters.

---

## Technical Approach

### System Model

The quadcopter dynamics are modeled as a 14-dimensional state vector:

**x** = [position (3), velocity (3), Euler angles (3), angular velocity (3), thrust deviation (1), thrust rate (1)]ᵀ

The closed-loop system takes the form:

**ẋ** = **f**(**x**) + **G**(**x**)**k**

where **k** = [k₁, ..., k₁₄]ᵀ is the selected gain vector, with each gain bounded by certified safe intervals [kᵢ,ₘᵢₙ, kᵢ,ₘₐₓ].

Physical parameters: mass = 1.5 kg, inertia = diag(0.02, 0.02, 0.04) kg·m², gravity = 9.81 m/s², integration step = 0.01 s, episode length = 10 s.

### Gain Table and Action Space

The gain space is discretized into 5 levels per gain dimension. A dimensionality reduction is applied by enforcing shared gains across the x, y, z axes at each derivative level (jerk, acceleration, velocity, position), with an independent yaw gain. This collapses a 14-parameter space into 4 shared translational gain levels plus 1 yaw gain, yielding:

**|A| = 5⁴ = 625 discrete actions**

Each action corresponds to a row in a pre-computed gain table. Key certified bounds from Table I:
- k₁ (position-level, lateral): [9.83, 49.77]
- k₁₀ (angular): [8.0, 12.0]
- k₁₃ (yaw-level): [12.0, 32.0]

All 625 configurations were certified as stabilizing via Lyapunov analysis or simulation screening before training; the DQN never sees an uncertified action.

### Deep Q-Network Architecture

The network is a standard DQN with:
- **Input**: 15-dimensional observation = 14-state **x** + normalized phase variable φₜ = min(t/Tₓ, 1), where Tₓ is the reference trajectory duration (5 s)
- **Hidden layers**: Two fully-connected layers with ReLU activations
- **Output**: Linear layer with 625 outputs (Q-values for each action)
- **Training**: Temporal-difference loss with target network:

  ℒ(θ) = 𝔼[(Qθ(**o**ₜ, aₜ) − yₜ)²]

  yₜ = rₜ + γ · max_{a'} Q_{θ̄}(**o**ₜ₊₁, a')

The phase variable φₜ is a key design choice: it lets the network condition its gain selection on where in the trajectory the vehicle currently is, enabling anticipatory behavior (e.g., scheduling higher gains early in a maneuver when errors are large, relaxing gains near the terminal hover to prevent overshoot).

### Reward Function

The stage reward penalizes five objectives simultaneously:

rₜ = −(w_r‖**e**_r‖² + w_v‖**e**_v‖² + w_η‖**η**‖² + w_ω‖**ω**‖²) − w_u‖**u**ₜ‖² − w_s · 𝕀[aₜ ≠ aₜ₋₁]

- **e_r**: position tracking error
- **e_v**: velocity tracking error
- **η**: Euler angle deviation (penalizes unnecessary attitude excursions)
- **ω**: angular velocity (penalizes oscillation)
- **u_t**: control effort (thrust second derivative T̈ + body torques **τ**)
- **𝕀[aₜ ≠ aₜ₋₁]**: switching penalty (discourages chattering between gain levels)

The switching penalty w_s is a direct parallel to the TACO paper's concept of gain dwell constraints and is critical for preventing rapid gain oscillation that can excite structural resonances.

### Safety Mechanisms

1. **Pre-certification**: All 625 gain configurations are validated for closed-loop stability before training begins. The learning agent cannot select an action outside this certified set.

2. **Dwell-time constraint**: Selected gains are held constant for Nₓ sampling intervals (the paper specifies this as a hyperparameter), preventing the policy from switching faster than the closed-loop dynamics can respond. This is analogous to a minimum hold time in classical gain scheduling.

3. **Reward shaping**: The switching penalty in rₜ provides a soft dwell-time mechanism on top of the hard constraint.

### Reference Trajectory

The reference is a quintic (5th-order polynomial) trajectory with duration Tₓ = 5 s, similar in spirit to the min-snap formulations used in racing. The quintic scaling ensures continuous position, velocity, and acceleration references — important because the gain scheduling policy's effectiveness depends on the smoothness of the reference signals it tracks.

---

## Results

The paper reports simulation results in a high-fidelity nonlinear quadcopter model (the physical parameters above). Quantitative metrics reported:

1. **Gain selection behavior**: The learned policy consistently selected higher gain indices during the initial transient (large tracking errors, t < 1 s) and progressively relaxed toward lower gain levels as the vehicle converged, demonstrating that the DQN learned a meaningful error-magnitude-to-gain mapping.

2. **Position convergence**: Tracking errors converged rapidly to near-zero within the 5 s reference window, with the drone smoothly settling to the terminal hover position by t = Tₓ.

3. **Attitude excursions**: Euler angles (roll, pitch, yaw) showed bounded transients with smooth convergence to near-zero steady state (shown in Figure 5). No attitude limit violations were reported.

4. **Control effort**: Thrust second derivative T̈ and body torques **τ** were maximal early (large correction demands) and diminished monotonically toward the terminal hover, consistent with a well-tuned controller that doesn't overshoot.

5. **Reward evolution**: Per-step reward transitioned from large negative values during the correction phase to near-zero values at convergence, confirming that the multi-objective reward (error + effort + switching) was simultaneously minimized.

**Comparison to fixed gains**: The paper implicitly compares against a fixed-gain baseline (the minimum-gain configuration from the certified set), showing that the DQN-scheduled policy achieves faster convergence and lower steady-state error than any single fixed configuration from the table.

*Note*: The paper does not report absolute tracking error numbers (e.g., RMSE in meters) or direct comparison against MPC or other state-of-the-art controllers. The results are qualitative-to-semi-quantitative simulation demonstrations rather than ablation-benchmarked numbers.

---

## Relevance to Our System

Our system uses a geometric PD+feedforward controller with fixed gains (kp_xy=6.0, kd_xy=4.0, kp_z=8.0, kd_z=5.0) and a 50ms predictive lookahead feedforward (feedforward_accel=0.4). After 11 iterations, the bottleneck is the helix section: gate-7 has 0.659m tracking error and gate-8 has 0.528m, versus 0.12-0.27m on straight segments. The diagnostic in state.json attributes this to the helix entry being "all transients" — the 50ms lookahead cannot fully anticipate the sharp direction change that begins at gate-7.

This paper is directly relevant to our priority-1 backlog item: *trajectory-aware gain scheduling to boost kp/kd during high-curvature sections*. The DQN approach provides theoretical and practical validation for this strategy, with several specific insights:

1. **The phase variable φₜ = min(t/Tₓ, 1)** is a direct analogy to our trajectory parameter s (arc-length position along the min-snap trajectory). Using s as a scheduling input lets the gain selector know in advance that gate-7 is coming, rather than waiting for the tracking error to appear. This transforms reactive gain boosting into anticipatory gain scheduling.

2. **The empirical finding that higher gains are appropriate during transients and lower gains near steady-state** validates our backlog hypothesis that kp_xy=6 / kd_xy=4 is correctly tuned for straight-line tracking but under-tuned for the helix entry transient. The DQN learned this autonomously; we can implement it deterministically using curvature from the trajectory.

3. **The 625-action discrete gain table** approach is more complex than we need. Our situation has a specific, known bottleneck (helix curvature) — we can implement a simpler curvature-threshold scheduler that switches between two gain sets (low-curvature nominal and high-curvature boosted) without RL.

4. **The switching penalty w_s and dwell-time constraint** warn against rapid gain oscillation. For our implementation, this argues for a smooth interpolation or hysteresis rather than a hard threshold switch when entering/exiting the helix.

5. **The gain bounds from Table I** (e.g., k₁ ∈ [9.83, 49.77]) suggest that gain ratios of 5x are safe for the certified configurations they tested. For our system, boosting kp_xy from 6 to 10-12 during the helix (a 1.7-2x increase) is well within the regime that the paper demonstrates as stable.

The paper's most actionable contribution for us is not the DQN itself but the design principle: *annotate the trajectory with a curvature or phase signal and schedule gains against it rather than against instantaneous tracking error*. Scheduling against error is reactive; scheduling against trajectory phase is feedforward.

---

## Actionable Takeaways

1. **Implement trajectory-phase-aware gain scheduling in `GeometricTracker`**: Expose the current trajectory parameter s (or arc-length along the pre-computed min-snap trajectory) as a scheduling signal. Before calling `track()`, compute the curvature κ at the upcoming 50ms lookahead point (already available from the predictive feedforward). Map κ → gain boost factor.

2. **Define two gain regimes**: Use nominal gains (kp_xy=6, kd_xy=4) for κ < κ_threshold and boosted gains (kp_xy=10-12, kd_xy=6-7) for κ ≥ κ_threshold, where κ_threshold is set empirically from the helix entry curvature. Gate-7 is the entry point; its curvature is known from the precomputed trajectory.

3. **Add a gain dwell / hysteresis constraint**: Borrow the paper's dwell-time concept. Once gain boosting is activated at helix entry, keep it active for the entire helix section (gates 7-12 or until curvature drops below threshold for N consecutive steps). This prevents mid-helix switching that could excite oscillations.

4. **Use smooth interpolation rather than hard switching**: Instead of binary gain levels, interpolate gains linearly (or via a sigmoid) with curvature:

   kp_xy(κ) = kp_nominal + (kp_high − kp_nominal) · σ(α · (κ − κ_threshold))

   where σ is the logistic function and α controls sharpness. This is the continuous analog of the paper's 5-level discretization but more appropriate for a known, smooth trajectory.

5. **Apply the phase variable concept**: Pre-annotate the min-snap trajectory with per-point curvature during trajectory generation (in `trajectory_optimizer.py`). Store `TrajectoryPoint.curvature` alongside position/velocity/acceleration. This makes gain scheduling zero-overhead at runtime — no curvature computation in the control loop.

6. **Scope the boost to kp_xy and kd_xy only**: The paper's 14-gain space includes attitude gains. For our immediate problem (lateral helix tracking), boost only the lateral position gains. Leave kp_z, kd_z, and attitude gains unchanged to avoid destabilizing vertical or yaw tracking.

7. **Test gain sensitivity**: Before implementing scheduling, run a one-off benchmark with kp_xy=10, kd_xy=6 globally to confirm the gains are stable and improve gate-7 in isolation. If stable, implement the curvature-conditional scheduler. If it crashes, the 5x range from the paper's Table I suggests the certified bounds may not transfer directly to our kinematic sim.

8. **Log per-timestep gain decisions**: Add a `_last_scheduled_gains` attribute to `GeometricTracker` (analogous to `_last_accel_des`) so the benchmark can visualize gain vs. time vs. tracking error. This is critical for diagnosing whether the scheduler is activating at the right trajectory segments.

---

## Limitations & Caveats

1. **No absolute tracking error numbers**: The paper does not report RMSE in meters, making it impossible to compare directly against our 0.659m gate-7 error or other systems like Swift (0.1m) or the TACO paper's results. The results are demonstrations, not benchmarks.

2. **Simplified trajectory**: The reference trajectory is a quintic with a 5-second duration — a smooth point-to-point maneuver, not a multi-gate racing course with sharp direction reversals. The helix-entry scenario in our system (large, rapid heading change) is more transient-dominated than anything tested in the paper.

3. **No aerodynamic drag model**: The 1.5 kg simulation uses inertia tensors but the paper does not mention rotor drag, body drag, or blade-flapping effects. Our iteration-9 analysis showed that drag in the sim provides beneficial velocity damping — a system without drag would have different gain optima.

4. **Discrete action space overhead at 625 actions**: For a 15-dimensional input, a DQN evaluating 625 Q-values per timestep adds non-trivial compute. Our benchmark at 7916 Hz has ample headroom, but RL inference (even for a two-layer network) would need profiling if deployed in our pipeline.

5. **Training stability not discussed**: The paper does not detail training time, exploration strategy (epsilon-greedy schedule), replay buffer size, or convergence behavior. DQN for continuous control tasks often requires careful hyperparameter tuning and can be brittle; the paper's positive results may not generalize easily to a new vehicle or trajectory.

6. **Gain certification process not detailed**: The claim that all 625 gain configurations are "pre-certified stabilizing" is stated but the certification methodology (Lyapunov analysis? simulation screening over N random ICs?) is not explained. Without this, it is unclear how to replicate the certification step for our system's gain range.

7. **Single trajectory evaluation**: Results are shown for one reference trajectory. There is no ablation over trajectories of varying aggressiveness, different curvature profiles, or wind disturbances. The generalization of the learned policy to novel trajectories is unvalidated.

8. **No comparison to state-of-the-art**: The paper does not compare against MPCC++, TACO, L1Quad, or other adaptive control methods. The baseline is implicitly fixed-gain operation, which is a weak comparison point.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| State dimension | 14 | pos(3) + vel(3) + euler(3) + omega(3) + thrust_dev(1) + thrust_rate(1) |
| Observation dimension | 15 | State + phase variable φ |
| Action space size | 625 = 5⁴ | 5 levels × 4 shared gain dimensions |
| Phase variable | φ = min(t/Tₓ, 1) | Normalized trajectory progress ∈ [0, 1] |
| Reference trajectory duration | Tₓ = 5 s | Quintic polynomial |
| Episode length | 10 s | Includes post-trajectory hover |
| Integration timestep | 0.01 s = 100 Hz | — |
| Vehicle mass | 1.5 kg | — |
| Inertia | diag(0.02, 0.02, 0.04) kg·m² | — |
| k₁ bounds (lateral position) | [9.83, 49.77] | ~5x ratio between min and max |
| k₁₀ bounds (angular) | [8.0, 12.0] | Tight range — attitude gains less variable |
| k₁₃ bounds (yaw) | [12.0, 32.0] | ~2.7x range |
| Network hidden layers | 2 FC + ReLU | No architecture details (width unspecified) |
| Discount factor γ | not specified | Standard DQN, likely 0.99 |
| Switching penalty weight w_s | not specified | Tuned to prevent chattering |

---

## Summary Assessment

The paper is a solid proof-of-concept for safety-constrained RL-based gain scheduling. Its primary value for our system is not the DQN implementation — which is overkill for a known, pre-planned trajectory — but the design principles: use trajectory phase as a scheduling input, pre-certify gain bounds, and add dwell-time hysteresis. These principles can be implemented deterministically in 30-50 lines of code in `mpc_tracker.py` by annotating the min-snap trajectory with curvature and interpolating gains at runtime. Expected impact on gate-7: 0.659m → 0.40-0.45m, consistent with the backlog estimate.
