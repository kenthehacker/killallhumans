# What Matters in Learning Zero-Shot Sim-to-Real RL Policy (SimpleFlight)

- **URL**: https://arxiv.org/html/2412.11764
- **Year**: 2024

---

## Key Contribution

SimpleFlight identifies and rigorously ablates five factors critical to achieving zero-shot sim-to-real transfer for quadrotor trajectory tracking using reinforcement learning. The central claim is that architectural and training choices — rather than algorithmic novelty — explain the gap between simulation performance and real-world deployment. The paper demonstrates a greater than 50% reduction in trajectory tracking error (mean Euclidean distance) compared to state-of-the-art RL baselines (DATT, Fly) on a physical Crazyflie 2.1 platform, and achieves performance comparable to a finely-tuned nonlinear MPC baseline (PAMPC) without any online adaptation or model-based components.

The five critical factors, in order of presentation:
1. Using rotation matrices (not Euler angles or quaternions) plus linear velocity in actor inputs.
2. Adding a time vector to critic inputs as privileged information (asymmetric actor-critic).
3. Action difference regularization to penalize large command changes between timesteps.
4. Selective system identification with targeted domain randomization only on sensitive parameters.
5. Large batch sizes during training to improve generalization even when simulation metrics are insensitive.

---

## Technical Approach

### Policy Architecture

SimpleFlight uses an **asymmetric actor-critic** design. The actor and critic receive different inputs, allowing the critic access to privileged training-time information without contaminating the deployed actor.

**Actor inputs:**
- Relative positions from the drone to the next N=10 reference trajectory points (world frame), providing a 0.5s look-ahead horizon at 100 Hz.
- Linear velocity (body or world frame).
- Rotation matrix R ∈ R^{3x3} representing full 3D orientation.

**Critic inputs:** All actor inputs plus a time vector f_t = [t, ..., t]^T ∈ R^k (k=1 for trajectory tracking), which prevents out-of-distribution behavior when trajectories exceed training duration.

**Network structure:** Three-layer MLPs with ELU activation and LayerNorm between layers; hidden dimension of 256 neurons throughout.

**Action space:** CTBR (Collective Thrust and Body Rate commands), a common mid-level interface that delegates attitude rate tracking to the onboard flight controller firmware.

### State Representation

The paper provides a key insight: rotation representations in four or fewer dimensions are topologically discontinuous, making them harder for neural networks to learn. Replacing the rotation matrix with a quaternion causes approximately 63.6% degradation in training performance. Euler angles fare even worse. The 9-element rotation matrix, despite its redundancy, provides a continuous and smooth manifold for the network to learn from.

### Training Setup

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Simulator:** GPU-accelerated OmniDrones at 100 Hz, 0.01s timesteps
- **Training duration:** 15,000 epochs
- **Parallel environments:** 1024 to 16,384 (batch size ablation)
- **Trajectory mix:** Balanced combination of smooth polynomial trajectories and infeasible zigzag paths with sharp directional changes

### Reward Function

The composite reward is:

```
r = r_task + λ * r_smooth
```

Both terms are normalized to [0, 1]. The smoothness term is defined as the L2 norm of the action difference:

```
r_smooth = ||u_t - u_{t-1}||_2
```

The optimal smoothness coefficient is λ = 0.4. The paper explicitly tested and rejected alternative smoothness formulations:
- Acceleration penalty (||acc_t||_2): poor performance on fast trajectories
- Jerk penalty (||jerk_t||_2): fails on fast trajectories
- Snap penalty (||snap_t||_2): fails on normal and fast trajectories
- Action clipping: limits agility (0.310m MED on fast trajectories)
- Low-pass filtering: complete failure (∞ MED — untrackable)

The conclusion is that penalizing the first difference of the control output is uniquely effective at bridging the sim-to-real gap without sacrificing agility.

### Real-World Deployment

- State feedback from OptiTrack motion capture system at 100 Hz
- CTBR commands sent via 2.4 GHz radio to Crazyflie 2.1
- Zero-shot deployment: no fine-tuning or domain adaptation after training
- At 50 Hz deployment frequency, "minor performance drop" is observed but performance still exceeds DATT baseline

---

## Domain Randomization and System Identification

This is the most directly applicable section. The paper evaluates selective DR on four dynamical parameters, explicitly comparing calibrated (SysID) values versus randomized ranges.

### Parameters Studied

| Parameter | SysID Tool | DR Range Tested | Conclusion |
|---|---|---|---|
| Mass (m) | Direct weighing | ±10%, ±30% | DR is **counterproductive** — calibrated value is sufficient and DR degrades performance |
| Inertia matrix I = diag(Ix, Iy, Iz) | Calibration rig | ±10%, ±30% | Low sensitivity; DR neither helps nor hurts significantly |
| Motor time constant Tm | Static propeller test stand | ±10%, ±30% | Minimal benefit from DR; calibration is preferred |
| Thrust coefficient kf | Static propeller test stand | ±10%, ±30% | DR **improves robustness** when accurate calibration is not feasible |

### System Identification Methodology

Physical parameter identification uses a static propeller test stand to directly measure thrust/drag coefficients and motor time constants. These are then used as ground-truth values in the simulator. The key insight is that calibrating these parameters accurately makes broad domain randomization unnecessary and potentially harmful for non-sensitive parameters, while targeted DR on parameters that are genuinely difficult to calibrate (like thrust coefficients that may shift with battery level or wear) provides meaningful robustness.

The recommendation is: "For sensitive parameters like thrust coefficients, when accurate calibration is not feasible, DR can effectively enhance robustness. For other parameters, accurately calibrated simulation values are sufficient."

### What Is NOT Randomized

- Observation noise / sensor noise (not addressed in this work)
- Wind disturbances or aerodynamic perturbations (not addressed)
- Visual/perceptual domain (explicitly noted as future work)
- Mass and inertia (after accurate SysID, randomization hurts)

---

## Results

All experiments use Mean Euclidean Distance (MED) in meters between actual and target positions, averaged across trajectory instances.

### Crazyflie 2.1 Results

| Trajectory | SimpleFlight (m) | DATT (m) | Fly (m) |
|---|---|---|---|
| Figure-eight, slow (0.6 m/s) | 0.016 ± 0.002 | 0.050 | 0.093 |
| Figure-eight, normal (1.6 m/s) | 0.028 ± 0.000 | 0.050 | 0.181 |
| Figure-eight, fast (2.5 m/s) | 0.051 ± 0.002 | 0.113 | 0.281 |
| Pentagram, fast (infeasible) | 0.045 ± 0.002 | ∞ (failed) | ∞ (failed) |
| Zigzag (infeasible) | 0.052 ± 0.003 | 0.114 (60% SR) | ∞ (failed) |

SimpleFlight is 1.8x to 6.5x more accurate than DATT on feasible trajectories and is the only method to reliably complete infeasible trajectories. Against PAMPC (nonlinear MPC with careful tuning), SimpleFlight achieves comparable or slightly better performance.

### Large Batch Size Effect

Increasing parallel environment count from 1024 to 16,384 improves real-world performance substantially despite negligible change in simulated tracking error. This highlights the importance of using simulation metrics cautiously — simulation overfitting to small batches does not predict real-world robustness.

---

## Relevance to Our System

Our system currently uses a geometric SE(3) tracker (Lee et al.) with manually tuned gains rather than a learned policy. SimpleFlight is directly relevant in several ways:

1. **Rotation representation**: If we add any learned components (e.g., learned residuals, gain prediction), we must use rotation matrices rather than Euler angles or quaternions. Our EKF currently outputs Euler angles for some interfaces — this should be checked.

2. **Action smoothness**: Our MPC tracker outputs attitude rate commands at 100 Hz. If we observe high-frequency chatter, adding an explicit action-difference penalty (analogous to λ=0.4 on the action delta) or a smoothing layer could help. The paper's finding that low-pass filtering fails completely suggests we should smooth the *command policy* rather than the *output signal*.

3. **SysID before DR**: For any future RL or learned controller experiments, the paper recommends: first characterize mass and inertia accurately (weigh the drone, measure inertia), then use a static thrust stand to characterize kf and Tm. Apply DR only to kf (±10–30%) and not to mass.

4. **Trajectory lookahead**: The N=10 point, 0.5s horizon lookahead is directly analogous to our MPC prediction horizon. This validates our current architectural choice of providing future reference points to the controller.

5. **CTBR action space**: We currently target body rates + collective thrust via the SE(3) controller. This matches SimpleFlight's action space, meaning their findings on command smoothness regularization transfer directly.

6. **Zero-shot deployment frequency**: At 50 Hz (half of our target 100 Hz), SimpleFlight still outperforms baselines. Our 100 Hz control loop is adequate.

---

## Actionable Takeaways

1. **State representation audit**: Verify that any learned or semi-learned component (future RL experiments, neural augmentation) uses 9-element rotation matrices, not quaternions or Euler angles. The 63.6% performance degradation from switching to quaternions is too large to accept.

2. **SysID protocol**: Before sim-to-real transfer of any trained policy, conduct: (a) direct mass measurement, (b) inertia estimation from CAD or swing tests, (c) static propeller stand measurements for kf and Tm. Apply ±10–30% DR only to kf.

3. **Action smoothness loss**: Add a smoothness penalty r_smooth = ||u_t - u_{t-1}||_2 with coefficient λ ≈ 0.4 to any RL training reward. This alone substantially closes the sim-to-real gap. Do not use low-pass filtering or jerk penalties as alternatives.

4. **Asymmetric actor-critic**: If training an RL policy, give the critic privileged time-index information to prevent out-of-distribution value estimates at long horizons. The actor should not see this.

5. **Batch size**: Use at least 4,096–16,384 parallel environments during RL training. Simulation convergence metrics do not predict real-world performance when batch sizes are too small.

6. **Infeasible trajectory handling**: Train on a mix of smooth and infeasible (zigzag, sharp-turn) trajectories. A policy trained only on smooth paths fails on infeasible gate sequences. This is directly relevant to our racing context where consecutive gates may require non-smooth transitions.

7. **Do not use DR for mass/inertia if SysID is available**: Broad randomization on accurately measured parameters actively hurts performance.

---

## Limitations & Caveats

1. **Single platform tested**: Primary real-world experiments use Crazyflie 2.1 (small, lightweight, low thrust-to-weight). Results on larger racing drones (250mm 5" racing quads) are not demonstrated. Our platform differs significantly — mass, inertia, and motor dynamics will require re-characterization.

2. **No wind or disturbance testing**: The paper does not study robustness to external forces, wind gusts, or aerodynamic effects. In outdoor or high-speed racing conditions, this omission is significant. DR over aerodynamic drag parameters is not addressed.

3. **No sensor noise analysis**: OptiTrack ground truth is used. Robustness to measurement noise from VIO, GPS, or onboard IMU is unstudied. Visual domain adaptation is explicitly deferred to future work.

4. **Fair comparison caveat**: The authors acknowledge that comparisons with DATT and Fly are confounded by different simulators, modeling approaches, and input/output spaces. The 50% improvement claim should be interpreted cautiously.

5. **Curriculum-free training**: No progressive difficulty curriculum is used. For very agile maneuvers (>5 m/s), training from scratch on mixed trajectories may not suffice — curriculum learning may still be needed.

6. **No gate detection or localization**: The paper assumes perfect state feedback (OptiTrack). Integration with a real perception pipeline (camera + EKF + gate detector) is not addressed.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|---|---|---|
| Trajectory lookahead points | N = 10 | Spaced 0.05s apart → 0.5s horizon |
| Control frequency | 100 Hz | Timestep = 0.01s |
| Deployment frequency | 100 Hz (50 Hz viable) | Via 2.4 GHz radio |
| Network hidden dim | 256 neurons | Three MLP layers |
| Activation function | ELU | With LayerNorm between layers |
| Time vector dimension | k = 1 | Scalable for complex tasks |
| Smoothness coefficient | λ = 0.4 | Action difference reward weight |
| Training epochs | 15,000 | PPO |
| Parallel environments | 1,024–16,384 | More = better real-world transfer |
| DR range for kf | ±10% to ±30% | Thrust coefficient only |
| DR for mass | Not recommended | Use calibrated value |
| DR for inertia | Not recommended | Use calibrated value |
| Body rate limits (Crazyflie) | [-π, π] rad/s | |
| Acceleration limits (Crazyflie) | [0, 1.6g] | |
| Best MED (figure-eight slow) | 0.016 ± 0.002 m | At 0.6 m/s |
| Best MED (infeasible pentagram) | 0.045 ± 0.002 m | At 1.0 m/s |
