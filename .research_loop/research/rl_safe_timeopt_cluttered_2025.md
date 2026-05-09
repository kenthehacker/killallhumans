# Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing
- **URL**: https://arxiv.org/abs/2512.09571
- **Authors**: Feng Yu, Yu Hu, Yang Su, Yang Deng, Linzuo Zhang, Danping Zou (Shanghai Jiao Tong University)
- **Year**: 2025
- **Venue**: arXiv / IEEE RA-L (IEEE Robotics and Automation Letters)

---

## Key Contribution

This paper presents what the authors claim is the first vision-based autonomous drone racing system that operates robustly in diverse, previously unseen, and physically cluttered environments — without GPS, prior maps, or course-specific tuning. The central contribution is a two-phase reinforcement learning (RL) training pipeline that separates exploration-friendly soft-collision training from safety-enforcing hard-collision refinement, combined with an adaptive noise-augmented curriculum that progressively eliminates policy reliance on privileged gate-position commands and replaces it with depth-image-based perception.

Unlike prior racing systems (Swift, MPCC, min-snap + MPC) that assume obstacle-free corridors or pre-surveyed tracks, this work targets realistic courses where the drone must navigate cluttered, unknown layouts using only onboard depth imagery at up to 5 m/s. The two key enabling insights are: (1) staged RL training separates "learning to go fast" from "learning to go safely," and (2) zigzag track primitives are explicitly included in training to ensure the policy acquires the rapid left-right direction reversal capability needed for sequential gates — the most demanding geometry in any racing course.

---

## Technical Approach

### Two-Phase Reinforcement Learning

**Phase 1 — Soft-Collision Training**: The drone trains without rigid-body collision termination. When the drone intersects an obstacle, a mild penalty proportional to the number of mesh contact points is applied. This penalty is small enough that the policy can still discover high-speed trajectories without early termination. The key effect is that the policy explores a broad region of the state-action space, including aggressive maneuvers through tight gate sequences. Forcing hard-collision termination from the start collapses exploration and converges to slow, overly conservative behavior.

**Phase 2 — Hard-Collision Fine-Tuning**: The soft-phase policy is loaded as a checkpoint and fine-tuned with realistic collision physics: any collision is terminal with a penalty of -1 (scaled to 100 in the reward table). Smoothness and safety penalty weights are increased. The policy preserves the fast trajectory shapes discovered in Phase 1 while learning to avoid obstacles strictly. This staged approach is directly analogous to curriculum learning in contact-rich manipulation: first learn the task, then learn to do it safely. Critically, without the Phase 1 warm-start, Phase 2 training alone fails to find fast solutions.

### Zigzag Track Primitives and Sequential Turn Handling

The paper explicitly identifies three archetypal track families used for training, and the zigzag family is given particular emphasis because it tests the fundamental capability of rapid direction reversal:

1. **Circular tracks** (clockwise and counterclockwise variants) — develops turning skill in a single direction, including centripetal attitude shaping.
2. **Zigzag tracks** — incorporates straight-line flight segments between gates placed alternately to the left and right. This requires the drone to exit one gate moving in one lateral direction, decelerate transversally, and enter the next gate moving in the opposite lateral direction. This is the structural analog of an S-turn.
3. **Elliptical tracks** — combined layout requiring behavior switching between straight-line and curved flight.

The paper's framing is that these three primitives cover all "fundamental motion modes" needed to generalize to arbitrary real-world course layouts. In practice, the zigzag primitive is the key one for handling S-turn geometry, because it forces the policy to learn how to set up its approach angle and speed at gate N to correctly position itself for gate N+1 — a multi-step planning behavior that single-gate training cannot develop.

The paper does not give explicit gate-to-gate distances or angles for the zigzag primitive, but the real-world validation shows successful navigation through a physical zigzag course with gates offset laterally and achieved speeds exceeding 5 m/s through the sequence.

### Adaptive Noise-Augmented Curriculum

A critical challenge in vision-based racing is the training-to-deployment gap: during training, gate positions can be provided exactly from simulation ground truth, but at deployment the drone must localize gates from noisy depth perception. The paper resolves this through a dynamically adjusted noise schedule injected into the gate-position command fed to the actor network:

- If the policy passes more than 3 gates per episode, noise amplitude increases by growth rate α₁.
- If fewer than 3 gates are passed, noise amplitude decreases by decay rate α₂.
- The asymmetry condition α₁ < α₂ is enforced so that noise decreases faster on failure than it grows on success. This prevents catastrophic forgetting: when the policy fails, it quickly returns to a noise level it can handle.

As noise grows large (up to 1.2 m per axis in ablations), the actor cannot rely on commanded gate positions and must extract gate features directly from the 96×72 depth image. The critic retains ground-truth commands throughout training, forming an asymmetric actor-critic (the actor trains from noisy commands, the critic evaluates from clean ground truth). This structure stabilizes policy gradient updates while the actor learns perceptual robustness.

For sequential gates (zigzag/S-turn geometry), the noise curriculum has a particularly important effect: large noise in the first gate position forces the policy to fly an approach that still works when the gate could be 1 m in any direction from where it expected. This is equivalent to planning with a constraint that the approach trajectory must be robust to gate localization uncertainty — which is exactly the problem that causes high tracking error at the first gate of an S-turn when the system must commit to an approach angle before it has a confident gate-2 fix.

### Observation Space and Gate Representation

The actor's input vector includes:
- Body-frame linear velocity (3D)
- Rotation matrix row 3 (encodes roll and pitch)
- Relative translation from the drone to gate N (3D, expressed in body frame)
- Relative translation from gate N to gate N+1 (3D, expressed in body frame)
- Previous action (4D: mass-normalized collective thrust + 3-axis body rates, CTBR format)
- Downsampled depth image: 96×72, processed by a CNN

The inclusion of the N→N+1 relative gate translation is directly relevant to S-turn handling. The policy knows not just where it is going but where it needs to be after the current gate. This is a lookahead of one gate that allows the policy to shape its trajectory through gate N to be well-positioned for gate N+1. A policy without this lookahead would optimize only for gate N, potentially arriving at a heading and speed that is poorly suited for gate N+1 — exactly the scenario causing high error at gate 3 when the gate-4 offset is not accounted for in the approach.

### Perception Reward for FOV Alignment (Gate Facing)

The reward function includes a perception component with weight 0.1:

```
r_perception = Normalize(R_{t-1} · (p_gate^w - p_drone^w)) · [1, 0, 0]^T
```

This term rewards the drone for aligning its forward body axis toward the next gate. At high speeds through turns, aggressive maneuvers can push the gate outside the camera's field of view before the drone has crossed it, causing a loss of gate-pose feedback. The perception reward counteracts this by penalizing large deviations between the drone's heading and the gate-facing direction. For an S-turn, this means the policy is incentivized to not over-rotate during the first turn, keeping the second gate in view during the transition.

This is an integrated planning-perception incentive baked into the RL reward, not a post-hoc constraint. It implicitly shapes the approach angle through the first gate of an S-turn so that the drone exits with a heading that gives visual coverage of the second gate.

### Lipschitz Continuity Constraint (L2C2)

Action oscillation is a persistent problem in RL-based controllers: high-frequency noise in thrust and body rate commands that physical actuators cannot follow. The paper applies a local Lipschitz continuity regularizer that penalizes large output differences between consecutive observation states:

```
λ₁ · D(π(o_t), π(ô_t)) + λ₂ · ||V(o_t) - V(ô_t)||²
```

where `ô_t` is a linearly interpolated observation between consecutive timesteps, and D is the squared Hellinger distance between action distributions. This suppresses action rate oscillation by approximately 25% without sacrificing peak speed, and directly improves sim-to-real transfer by reducing high-frequency command noise that physical actuators cannot track.

### Simulation Infrastructure: DiffLab

The authors developed "DiffLab," a parallelized simulator on NVIDIA Isaac Lab featuring:
- First and second-order air drag modeling (aerodynamic fidelity)
- Control delay modeling (important for angular rate tracking)
- Parameter randomization across mass, inertia, and drag coefficients
- Real flight data calibration for drag coefficient identification

The real-world platform is a 0.46 kg quadrotor with Intel RealSense D435i depth camera and Radxa Zero3W onboard computer (1.6 GHz quad-core A55, 1 TOPS NPU), running the full pipeline onboard at real-time inference rates.

---

## Results

**Simulation robustness (gate-position noise tolerance):**
- 100% success rate with up to ±0.6 m gate-position noise injected per axis.
- Over 50% success rate at ±1.2 m noise per axis.
- Adaptive curriculum achieves ~80% success rate at 2.1 m total noise vs. 54% for fixed-noise baselines — a substantial gap validating the asymmetric curriculum design.

**Real-world deployment:**
- Factory environment: 10/10 success at ±0.3 m noise; 7/10 at ±0.9 m noise.
- Forest environment: 10/10 success at ±0.6 m noise; 8/10 at ±1.2 m noise.
- Peak speeds exceed 5 m/s in both simulation and real-world tests.

**Ablation results:**
- Two-phase (soft + hard) training achieves higher average speed than one-phase (hard only) training with comparable success rates. Hard-only training converges to slow, cautious behavior.
- L2C2 regularization reduces action rate oscillation by ~25% in controlled ablation.
- Adaptive noise curriculum outperforms both noise-free and fixed-noise-level baselines significantly.

**Training cost:** approximately 4 hours on an Intel i7 + RTX 4090 using PPO.

The paper does not provide direct quantitative comparison against Swift, TOGT, or other time-optimal systems, which limits head-to-head benchmarking. The 5 m/s peak speed is substantially below what time-optimal racing systems achieve (10–25+ m/s), reflecting the different objectives (cluttered-environment robustness vs. raw time minimization).

---

## Relevance to Our System

Our system is a drone racing autonomy stack with min-snap polynomial trajectories optimized by L-BFGS, using a geometric SE(3) tracker. The current bottleneck is the S-turn pair at gates 3–4, where gate-3 has a 0.463 m tracking error. The first turn of the S-turn has high error because the approach trajectory commits to an angle that works well for gate 3 in isolation but is poorly set up for gate 4.

### Zigzag/S-Turn Approach Management

The paper's most directly applicable insight is that **the approach to gate N must be shaped by the requirement for gate N+1**, not just by gate N geometry. The RL policy accomplishes this implicitly through the one-gate lookahead in the observation (the N→N+1 gate-relative translation). For our optimization-based pipeline, the equivalent is to ensure our L-BFGS racing line optimizer considers not just the exit velocity and angle at gate 3, but the full path from gate 3 to gate 4.

Our current min-snap optimizer already does this globally, but the L-BFGS initial conditions and gate-passage waypoints must place gate-3 waypoints such that the drone exits gate 3 with a heading that minimizes the curvature required to reach gate 4. If gate-3 waypoints are specified at the gate plane only (no exit vector constraint), the optimizer is free to exit at any angle, and it may choose one that minimizes snap for the gate-2-to-gate-3 segment but creates high curvature (and therefore high tracking error) in the gate-3-to-gate-4 segment.

**Mitigation:** Add exit-vector constraints at gate 3 that point toward gate 4. This is a direct translation of the paper's one-gate lookahead observation into the trajectory optimization formulation.

### Perception Reward as Gate-Facing Yaw Constraint

The perception reward (weight 0.1) keeps the camera pointing at the gate during approach. For our EKF + PnP pipeline, gate-facing yaw has a direct functional equivalent: if the drone's attitude during the approach to gate 3 is well-aligned to gate 3's face normal, PnP pose estimation will be more accurate and the EKF correction will have lower uncertainty. The perception reward insight translates into a yaw regularization term in our trajectory optimizer that biases the drone to face the approaching gate.

### One-Gate Lookahead in Trajectory Optimization

The observation vector's N→N+1 relative translation is the mechanistic basis for S-turn handling in this system. In our trajectory optimizer, this corresponds to ensuring that waypoints at gate N are accompanied by derivative constraints (velocity vector direction) that point toward gate N+1. Without this, the L-BFGS optimizer may find a local minimum where gate N is traversed correctly but the exit trajectory is misaligned with gate N+1 — which is the structural cause of the gate-3 tracking error spike.

### Soft-to-Hard Constraint Analogy for Trajectory Refinement

The two-phase training (soft collision → hard collision) has a structural analog in our iterative benchmark loop. If we first optimize for pure minimum-snap (no tracking error penalty, just geometric path quality), then refine with an explicit tracking error penalty added to the L-BFGS objective, we may find better local optima than optimizing both simultaneously. The paper's insight is that global exploration must precede local refinement for difficult trajectory shapes.

---

## Actionable Takeaways

1. **Add exit-vector constraints at gate 3 pointing toward gate 4.** In `planning/trajectory_optimizer.py` and `planning/racing_line.py`, enforce a velocity-direction constraint at the gate-3 waypoint that biases the exit vector toward the gate-4 center. This is the most direct translation of the paper's one-gate lookahead and should reduce gate-3 tracking error by narrowing the curvature required in the gate-3-to-gate-4 segment.

2. **Include the N→N+1 gate offset in the L-BFGS racing line cost function.** The racing line optimizer currently minimizes curvature and tracking error gate-by-gate. Extend the cost to penalize heading deviation at gate N relative to the N→N+1 direction. This mirrors the paper's observation-space lookahead.

3. **Add yaw-toward-gate regularization during approach segments.** In `planning/trajectory_optimizer.py`, add a soft yaw penalty term that incentivizes the drone to face the approaching gate during the final 30–50% of each inter-gate segment. This directly improves PnP quality and EKF correction accuracy at the moment of gate crossing, reducing the state estimation error that propagates through the S-turn.

4. **Two-phase L-BFGS optimization for S-turn segments.** First optimize the gate-3-to-gate-4 segment in isolation (no snap penalty, just endpoint constraints) to find a geometrically feasible path, then refine with full snap and tracking error penalties. This avoids the local minima that arise when aggressive curvature penalties dominate at initialization.

5. **Test trajectory optimizer with zigzag-class track primitives.** Generate synthetic test cases with lateral gate offsets matching the gate-3/gate-4 geometry (alternating left-right placement) and verify that the optimizer consistently finds low-curvature solutions. If it fails on these test primitives, the gate-3 error will recur on any similar track layout.

6. **Apply Lipschitz-style control rate smoothing.** Add a penalty on the derivative of thrust and body rate commands in `control/mpc_tracker.py` beyond what min-snap provides. The 25% oscillation reduction from L2C2 is significant; in our classical pipeline, this maps to a jerk penalty in the attitude command generation.

7. **Scale EKF gate-correction noise covariance with approach speed.** The paper's noise curriculum implicitly shows that perception quality degrades at higher speeds. In `estimation/gate_pnp.py`, scale the measurement noise covariance R upward as a function of body speed, especially during the approach to gate 3 where speed is likely near peak. This avoids overweighting a noisy high-speed PnP measurement that could corrupt the EKF state going into the S-turn.

---

## Limitations & Caveats

1. **No direct speed comparison with state-of-the-art.** The 5 m/s peak speed is substantially below Swift (18+ m/s) or time-optimal systems like TOGT. The paper addresses robustness and obstacle avoidance, not raw time minimization. Insights from this paper are about trajectory shaping and approach management, not about speed profiles.

2. **External gate-position commands still required.** Despite "vision-based" branding, the system injects (noisy) gate position commands from an external localization source (motion capture in real-world tests). Full autonomy from pure onboard depth imagery without any gate-position prior is not demonstrated. In our competition setting, we have GPS-referenced gate positions from course maps, so this limitation is less relevant.

3. **RL policy — not directly extractable.** The system is end-to-end RL; the specific behaviors (exit-vector shaping, FOV maintenance) are implicit in learned weights, not in explicit planning rules. The insights must be translated into geometric principles for use in our optimization-based pipeline.

4. **Clutter handling untested at competition speeds.** The 5 m/s performance in cluttered environments does not validate whether the approach generalizes to 15–25 m/s racing speeds. At those speeds, the aerodynamic regime changes significantly.

5. **Zigzag primitive angle/distance unspecified.** The paper does not give exact geometric parameters for the zigzag track primitive (gate spacing, lateral offset, approach angle). This limits direct reuse of the training distribution as a benchmark for our optimizer.

6. **Sim-to-real gap remains for speed.** The authors note that "distribution of scenes will greatly affect the speed of drones." Their fastest real-world speed is not reported numerically. Domain shift is an acknowledged open problem.

7. **Hardware inference latency not reported.** The Radxa Zero3W with 1 TOPS NPU runs the full depth-CNN pipeline onboard, but the end-to-end latency from depth image capture to control command is not quantified. For our system's compute budget analysis, this is a missing data point.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Depth image resolution | 96 × 72 pixels | Downsampled CNN input |
| Gate-crossing threshold | 0.35 m | Gate pass-through detection |
| Noise growth rate α₁ | < α₂ (exact unreleased) | Curriculum increase on success |
| Noise decay rate α₂ | > α₁ | Curriculum decrease on failure |
| Curriculum switch criterion | 3 gates per episode | Noise adjustment trigger |
| Progress reward weight | 1.0 | Both phases |
| Perception reward weight | 0.1 | FOV alignment, both phases |
| Body rate penalty (Phase 1) | -0.02 | Smoothness, soft phase |
| Body rate penalty (Phase 2) | -0.1 | Smoothness, hard phase |
| Action rate penalty (Phase 1) | -0.01 | Actuator smoothness, soft |
| Action rate penalty (Phase 2) | -0.05 | Actuator smoothness, hard |
| Collision penalty (Phase 1, soft) | 50 | Penetration-based penalty |
| Collision penalty (Phase 2, hard) | 100 | Terminal collision penalty |
| Success reward (Phase 1) | 10 | Gate-crossing bonus |
| Success reward (Phase 2) | 20 | Gate-crossing bonus, higher stakes |
| Bad pose penalty (Phase 2 only) | -30 | Anti-tumbling |
| L2C2 action oscillation reduction | ~25% | Measured in ablation |
| Peak flight speed (sim and real) | >5 m/s | Performance benchmark |
| Success rate at ±0.6 m noise | 100% (sim) | Robustness benchmark |
| Success rate at ±1.2 m noise | >50% (sim) | Robustness benchmark |
| Real-world success (factory, ±0.3 m) | 10/10 | Real-world validation |
| Real-world success (factory, ±0.9 m) | 7/10 | Real-world validation |
| Real-world success (forest, ±0.6 m) | 10/10 | Real-world validation |
| Real-world success (forest, ±1.2 m) | 8/10 | Real-world validation |
| Quadrotor mass | 0.46 kg | Real-world platform |
| Depth camera | Intel RealSense D435i | Real-world hardware |
| Onboard compute | Radxa Zero3W, 1.6 GHz A55, 1 TOPS NPU | Real-world hardware |
| Training time | ~4 hours | RTX 4090 + Intel i7, PPO |
| Adaptive curriculum advantage | ~80% vs. 54% at 2.1 m noise | vs. fixed-noise baseline |
