# Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing
- **URL**: https://arxiv.org/abs/2512.09571
- **Authors**: Feng Yu, Yu Hu, Yang Su, Yang Deng, Linzuo Zhang, Danping Zou
- **Year**: 2025
- **Venue**: IEEE Robotics and Automation Letters

---

## Key Contribution

This paper presents the first vision-based autonomous drone racing system that operates robustly in diverse, previously unseen, and physically cluttered real-world environments — without requiring GPS, prior maps, or course-specific tuning. The central technical insight is a two-phase reinforcement learning (RL) training pipeline that separates exploration-friendly soft-collision training from safety-enforcing hard-collision refinement, combined with an adaptive noise-augmented curriculum that progressively weeds out policy reliance on privileged gate-position commands and replaces it with depth-image-based perception.

Unlike prior racing systems (including Swift and most MPCC/min-snap approaches) that assume obstacle-free corridors or pre-mapped tracks, this work targets the more realistic setting where the drone must navigate cluttered, unknown courses using only onboard depth imagery. The authors claim this is the first system to demonstrate vision-based racing in such conditions with quantified robustness at real-world speeds.

---

## Technical Approach

### Two-Phase Reinforcement Learning

**Phase 1 — Soft-Collision Training**: The drone is trained in simulation without rigid-body collision termination. When the drone intersects obstacles, a mild penetration-point penalty is applied (proportional to the number of mesh contact points). This preserves broad exploration of the state-action space and allows the policy to discover high-speed trajectories without premature termination. Without this phase, hard-collision training collapses exploration and the policy converges to slow, over-cautious behavior.

**Phase 2 — Hard-Collision Fine-Tuning**: After the soft-phase policy is learned, the simulation switches to realistic collision dynamics: a terminal penalty of -1 on any collision. Smoothness and collision penalty weights are also increased during this phase. The policy is fine-tuned from the Phase 1 checkpoint, learning to preserve the fast trajectories discovered earlier while strictly avoiding obstacles.

This staged approach is a direct parallel to curriculum learning in contact-rich manipulation: first learn to accomplish the task, then learn to do it safely.

### Adaptive Noise-Augmented Curriculum

A core challenge in vision-based racing is the gap between training with privileged gate-position commands (easy, fast convergence) and inference from noisy depth perception alone. The paper introduces a dynamically adjusted noise schedule applied to the gate-position commands fed to the actor:

- If the policy passes more than 3 gates per episode, noise amplitude increases (growth rate α₁).
- If fewer than 3 gates are passed, noise decreases (decay rate α₂).
- α₁ < α₂, so noise decreases faster on failure than it grows on success — this asymmetry prevents catastrophic forgetting.

As noise grows large, the actor cannot rely on the commanded gate positions and must extract gate features directly from the depth image. The critic, however, retains clean ground-truth commands throughout, forming an asymmetric actor-critic that stabilizes training while driving perception-based policy emergence.

### Depth-Based Perception Architecture

The perception input is a downsampled depth image at 96×72 resolution, processed by a CNN. The CNN output is concatenated with MLP-processed state features (body-frame linear velocity, rotation matrix row 3 for roll/pitch, previous action) and the noisy gate position commands. The combined representation drives the control policy. A separate, offline-trained gate-crossing detection head uses network embeddings but is excluded from policy gradient updates to avoid destabilizing the main policy.

The depth image provides implicit obstacle awareness and gate localization without requiring explicit object detection or bounding-box pipelines — important for real-time operation on constrained hardware.

### Perception Reward for FOV Alignment

The reward includes a perception component (weight 0.1) defined as:

```
r_perception = Normalize(q_{t-1} ⊙ (p^w_gate - p^w_b)) · [1, 0, 0]^T
```

This term rewards the drone for orienting its forward axis toward the next gate, effectively incentivizing the drone to keep the gate within the camera's field of view. At high speeds, aggressive turns can push the gate outside the depth camera's FOV before the drone crosses it; this reward counteracts that tendency by shaping approach angles. Critically, this is an integrated planning-perception incentive baked directly into the RL reward, not a post-hoc constraint or replanning trigger.

### Lipschitz Continuity Constraint (L2C2)

To reduce action oscillation — a persistent problem in RL-based controllers — the authors apply a local Lipschitz continuity regularization. The constraint penalizes large differences in the policy's output distribution between linearly interpolated observations. Using squared Hellinger distance D between distributions at interpolated states, the regularizer suppresses action rate by approximately 25% without sacrificing speed. This directly improves sim-to-real transfer by reducing high-frequency command noise that physical actuators cannot track.

### Track-Primitive Generator

Training is performed on three synthetic track families:
1. **Circular tracks** (clockwise and counterclockwise) — develops turning primitives.
2. **Zigzag tracks** — develops straight-line flight and rapid direction reversals.
3. **Elliptical tracks** — combined patterns requiring behavior switching.

These primitives are designed to cover the fundamental motion modes required for arbitrary course layouts, enabling zero-shot generalization to novel real-world tracks.

### Sim-to-Real Transfer

Domain randomization covers: first and second-order air drag coefficients, mass, inertia, and PID gain perturbations. Control delay is explicitly modeled to improve angular velocity tracking fidelity. The real hardware uses an Intel RealSense D435i depth camera and a Radxa Zero3W computer (1.6 GHz quad-core A55, 1 TOPS NPU) on a 0.46 kg quadrotor with Betaflight low-level control. The full pipeline runs onboard at real-time speeds on this constrained hardware.

---

## Results

**Simulation**:
- Policy reaches over 5 m/s flight speeds.
- 100% success rate with up to 0.6 m gate-position noise injected.
- Over 50% success rate at 1.2 m gate-position noise.
- L2C2 reduces action rate oscillation by 25%.

**Real-World (two environments)**:
- Factory environment: 10/10 success at ±0.3 m noise; 7/10 at ±0.9 m noise.
- Forest environment: 10/10 success at ±0.6 m noise; 8/10 at ±1.2 m noise.

**Training cost**: ~4 hours on Intel i7 + RTX 4090 using PPO.

The system is compared qualitatively to prior work. The authors claim this is the first to demonstrate agile, vision-based, obstacle-aware racing in cluttered real-world environments. Direct quantitative benchmarks against Swift or other state-of-the-art systems are not provided, which is a notable gap.

---

## Relevance to Our System

### Field of View Handling

This is where the paper is most directly relevant to our competition system. Our current pipeline (SE(3) geometric tracker + min-snap trajectories) makes no explicit effort to keep gates within the camera's FOV during aggressive maneuvers. The paper's perception reward approach is a clean, low-overhead integration: a single reward term shaped to align the drone's forward axis with the next gate. For our system, an analogous constraint could be embedded in the trajectory optimizer as a soft penalty on yaw deviation from gate-facing, or as a CBF-style constraint in the MPC tracker.

The paper uses **integrated perception-planning incentives** (baked into RL reward) rather than staged or post-processing perception constraints. For our non-RL, optimization-based pipeline, the equivalent is to add a yaw-regularization term in the min-snap trajectory optimizer that biases the drone to face the next gate during the approach segment — not just minimize snap.

### Perception Robustness During Aggressive Maneuvers

The depth-based perception system handles aggressive maneuvers through two mechanisms:
1. The perception reward keeps the gate in FOV during high-speed approaches.
2. The two-phase training ensures the policy was exposed to high-speed trajectories near obstacles before the safety constraint was tightened.

For our pipeline, the lesson is that our EKF + PnP gate-pose estimator will degrade during extreme attitude excursions (gate leaves the camera frame). The mitigation is to pre-plan yaw profiles that ensure continuous gate visibility, similar to perception-aware planning approaches (see `perception_aware_planning_eth_2026.md`).

### Speed vs. Perception Tradeoff

The paper addresses this tradeoff through the adaptive noise curriculum: faster flight is rewarded, but gate-position noise (representing perception uncertainty at speed) is increased as performance improves, forcing the policy to develop robust perception rather than relying on accurate localization. The key insight is that perception quality degrades at higher speeds (motion blur, limited FOV during turns), so the training distribution must reflect that coupling.

For our system: at higher speeds, our EKF gate-correction updates will be less frequent (gates pass through FOV faster), increasing drift. One mitigation is to prioritize EKF correction opportunities by slightly reducing speed at gate approach segments — a direct tradeoff that this paper quantifies implicitly through its noise-to-success-rate curves.

### Cluttered Environment Handling

Our current competition track (VQ1) likely does not have physical obstacles beyond the gates themselves, so the cluttered-environment capability is not directly applicable. However, the soft/hard collision training philosophy is relevant as a general principle: exploration under soft constraints before enforcement of hard limits. This maps well to our iterative benchmark refinement loop.

### RL vs. Classical Pipeline Applicability

This paper uses end-to-end RL, which is architecturally different from our min-snap + SE(3) tracker approach. The core insights (perception reward, curriculum, two-phase training) are not directly transplantable. However, the **perception reward formulation** and **gate-facing yaw incentive** are extractable as geometric design principles that apply to any trajectory optimizer.

---

## Actionable Takeaways

1. **Add yaw-toward-gate regularization in the trajectory optimizer.** In `planning/trajectory_optimizer.py`, add a soft penalty on yaw angle deviation from the vector pointing toward the next gate during approach segments. This directly mirrors the paper's perception reward and keeps gates in the depth camera FOV without requiring a separate perception module.

2. **Increase EKF gate-correction frequency during high-speed segments.** High-speed flight means gates transit the FOV quickly. Consider triggering EKF updates more aggressively (lower confidence threshold) during fast segments to compensate for reduced observation time, accepting slightly noisier individual measurements.

3. **Model gate-position uncertainty as a function of approach speed.** The paper's noise injection curriculum implicitly encodes this: faster flight → larger effective gate-position uncertainty. In our `gate_pnp.py`, scale the measurement noise covariance R as a function of estimated body speed to reflect reduced perception quality at high speeds.

4. **Apply Lipschitz-style action smoothing.** The L2C2 constraint's 25% reduction in action rate oscillation is significant for sim-to-real transfer. In our MPC tracker (`control/mpc_tracker.py`), add an explicit penalty on control derivative (jerk/snap in attitude commands) beyond what min-snap already provides. This is the classical analog to L2C2.

5. **Track-primitive-style trajectory diversity during testing.** During benchmark development, test with circular, zigzag, and elliptical sub-track segments to verify our trajectory optimizer and tracker generalize, not just on the specific VQ1 layout.

---

## Limitations & Caveats

1. **No direct comparison with state-of-the-art speeds.** The 5 m/s peak speed is substantially below the 70+ m/s seen in elite human drone racing and below the 18+ m/s targets in time-optimal systems like Swift. The paper does not compare against Swift, TOGT, or similar. This limits direct performance benchmarking.

2. **Relies on external localization for gate position commands.** Despite claiming vision-based operation, the system still injects (noisy) gate position commands from an external source (motion capture in real-world tests). Full autonomy from pure vision is not demonstrated. This is a meaningful gap for competition settings without infrastructure.

3. **RL policy — not interpretable or analytically tunable.** Unlike our min-snap + MPC pipeline, RL policies offer limited levers for targeted improvement. The paper's approach cannot be directly fine-tuned via gain schedules or waypoint modifications.

4. **Clutter handling untested at competition speeds.** Real drone racing involves speeds well above 5 m/s. Whether the cluttered-environment performance holds at 10–20+ m/s is unvalidated.

5. **Sim-to-real gap acknowledged but not fully closed.** The authors note that "distribution of scenes will greatly affect the speed of drones" — real-world domain shift limits achievable speed across diverse environments. Their fastest real-world run is not reported numerically.

6. **Hardware constraints.** The 1.6 GHz A55 CPU with 1 TOPS NPU is representative of competition-class constrained hardware, but the depth CNN inference latency is not reported. For our system with potentially tighter compute budgets, the feasibility of a similar depth-CNN approach must be verified.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Depth image resolution | 96 × 72 pixels | Downsampled for CNN input |
| Noise growth rate α₁ | < α₂ (exact values not released) | Curriculum noise increase on success |
| Noise decay rate α₂ | > α₁ | Curriculum noise decrease on failure |
| Gate-passing threshold | 3 gates per episode | Curriculum switching criterion |
| Towards reward weight | 1.0 | Phase 1 and 2 |
| Body rate penalty weight | -0.02 (Phase 1), -0.1 (Phase 2) | Smoothness term |
| Action rate penalty weight | -0.01 (Phase 1), -0.05 (Phase 2) | Actuator smoothness |
| Collision penalty weight | 50 (Phase 1, soft), 100 (Phase 2, hard) | Safety enforcement |
| Perception reward weight | 0.1 | FOV alignment incentive |
| Success reward | 10 (Phase 1), 20 (Phase 2) | Gate-crossing bonus |
| Bad pose penalty | -30 (Phase 2 only) | Anti-tumbling |
| L2C2 action oscillation reduction | ~25% | Measured in ablation |
| Quadrotor mass | 0.46 kg | Real-world platform |
| Peak flight speed | >5 m/s | Sim and real |
| Success rate at 0.6 m noise | 100% | Robustness benchmark |
| Success rate at 1.2 m noise | >50% | Robustness benchmark |
| Training time | ~4 hours | RTX 4090 + Intel i7, PPO |
| Depth camera | Intel RealSense D435i | Real-world hardware |
| Onboard compute | Radxa Zero3W, 1.6 GHz A55, 1 TOPS NPU | Real-world hardware |
