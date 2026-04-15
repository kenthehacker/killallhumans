# DiffLab: Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing

**Authors:** Feng Yu, Yu Hu, Yang Su, Yang Deng, Linzuo Zhang, Danping Zou
**Published:** December 2025, arXiv:2512.09571
**Venue:** IEEE Robotics and Automation Letters (RA-L, accepted)

## Key Contribution

This paper presents a vision-based autonomous drone racing framework that handles diverse, partially unknown, and obstacle-cluttered environments — conditions where prior RL-based methods fail due to limited exploration and inability to distinguish gates from obstacles. The core contributions are: (1) a two-phase training strategy (soft-collision then hard-collision) that decouples velocity optimization from collision avoidance, (2) an adaptive noise-augmented curriculum that progressively degrades gate position information to build robustness, (3) Local Lipschitz Continuous Constraints (L2C2) that reduce action oscillation by ~25%, and (4) a track-primitive generator that trains on canonical track shapes (circular, zigzag, elliptical) for generalization. The system achieves >5 m/s flight speeds through cluttered environments with real-world deployment on computationally constrained hardware.

## Technical Approach

**Two-Phase Training:** The key insight is that training with realistic collision termination from the start severely limits exploration — the drone never learns to fly fast because it crashes before discovering high-speed behaviors. Phase 1 (soft-collision) allows the drone to pass through obstacles with a mild penalty `r_soft = -sum_p I[p]` where `I[p] in [0,1]` indicates mesh penetration depth. Phase 2 (hard-collision) fine-tunes with termination on collision (`r_hard = {-1 if collided, 0 else}`) and increases smoothness/collision penalty weights by 2-5x. This is reminiscent of curriculum learning but applied to the physics engine rather than the task difficulty.

**Reward Function (Phase 1 / Phase 2 weights):**
- Towards reward (velocity toward target): 1 / 1
- Body rate penalty: -0.02 / -0.1
- Action rate penalty: -0.01 / -0.05
- Collision penalty: 50 / 100
- Perception reward (facing target): 0.1 / 0.1
- Success reward (distance < 0.35m): 10 / 20
- Bad pose penalty: 0 / -30

**Observation Space:** `o_t = [^b v_t, r_3, ^b delta_p_t^1, ^b delta_p_t^2, a_{t-1}, I_t]` — body-frame velocity, third rotation matrix row (compact attitude), relative positions to next two gates (in body frame, with curriculum noise), previous action, and depth image (96x72).

**Action Space:** CTBR (Collective Thrust + Body Rates) — mass-normalized thrust and body angular rates.

**Asymmetric Actor-Critic:** The actor receives noisy gate commands (simulating perception uncertainty), while the critic receives ground-truth gate positions. This forces the policy to be robust to gate localization errors while still having a well-informed value function for stable training.

**Adaptive Noise Curriculum:** Gate position noise follows `U(-mu, mu)` with adaptive update: `mu_{t+1}^i = mu_t^i * (1+alpha_1)^I[n>3] * (1-alpha_2)^I[n<3]`, where n is gates passed, alpha_1 is growth rate, alpha_2 is decay rate (alpha_2 > alpha_1). This automatically finds the noise level that challenges but doesn't overwhelm the policy.

**L2C2 Regularization:** `L_L2C2 = lambda_1 D(pi_theta(o_t), pi_theta(o_hat_t)) + lambda_2 ||V_phi(o_t) - V_phi(o_hat_t)||^2` where `o_hat_t = o_t + (o_{t+1} - o_t) * u, u ~ U(-1,1)`. Uses squared Hellinger distance D. This penalizes policy sensitivity to small observation changes, reducing the jerky control outputs that plague RL policies.

**Track Primitive Generator:** Trains on three canonical shapes — circular (clockwise + counterclockwise), zigzag, and elliptical — capturing left turns, right turns, and straights. This avoids overfitting to specific track layouts.

**Domain Randomization:** Air drag coefficients (first and second-order), mass, inertia, PID gains for angular velocity controller, observation noise, and control delay.

## Results

**Simulation (DiffLab on NVIDIA Isaac Lab, PPO, ~4 hours on RTX 4090):**

| Track | Noise Level | Success Rate | Max Velocity |
|-------|-------------|-------------|-------------|
| Factory | [0.0, 0.0]m | 10/10 | 5.1 m/s |
| Forest | [0.0, 0.0]m | 10/10 | 5.3 m/s |
| Factory | [-0.6, 0.6]m | 10/10 | 4.4 m/s |
| Forest | [-0.6, 0.6]m | 10/10 | 4.6 m/s |
| Factory | [-1.2, 1.2]m | 5/10 | 4.3 m/s |
| Forest | [-1.2, 1.2]m | 8/10 | 4.6 m/s |

**Ablation — Noise Curriculum:** At 2.1m noise, adaptive curriculum achieves 80% success vs 54% for fixed noise and ~0% for noise-free training.

**Ablation — L2C2:** Reduces action rate (oscillation) by ~25% with negligible impact on speed.

**Ablation — Two-Phase:** Higher average speed than single-phase training due to better velocity exploration in Phase 1.

**Real-World:** >5 m/s through zigzag, circular, and U-shaped tracks, with successful navigation despite >1m gate position noise.

## Relevance to Our System

While our system uses a pre-computed min-snap trajectory rather than end-to-end RL, several ideas from this paper are transferable:

1. **Two-phase optimization for TOPP.** Our TOPP compression floors exist because the optimizer must simultaneously satisfy speed and tracking constraints. A two-phase approach — first optimize for speed ignoring tight error bounds, then tighten — could find faster solutions that the single-pass optimizer misses. This directly parallels their soft-collision/hard-collision decomposition.

2. **Noise-augmented ILC.** Their adaptive noise curriculum is conceptually similar to our ILC corrections but applied to gate positions rather than trajectory offsets. We could inject controlled noise into our ILC target during optimization to find more robust corrections that don't overfit to a single trajectory execution.

3. **L2C2 for controller smoothness.** Our gate-3 binding constraint might be partially caused by controller oscillation near the gate approach. The L2C2 concept — penalizing sensitivity to small state changes — could be applied to our geometric tracker's gain scheduling to reduce jitter without a full RL rewrite.

4. **CTBR action space.** They use mass-normalized collective thrust + body rates, which is the same abstraction level as our tracker output. This confirms our architecture choice is compatible with state-of-the-art RL approaches if we later move to learned controllers.

5. **Asymmetric information for robustness.** Training with degraded gate estimates while evaluating against ground truth is a powerful idea. For our EKF, we could validate performance by artificially degrading PnP gate detections during testing.

## Actionable Takeaways

- **Two-phase TOPP optimization:** First pass: optimize time with relaxed error bounds (2x current thresholds). Second pass: tighten bounds while warm-starting from the fast solution. This could break through compression floors.
- **Adaptive noise injection in ILC:** When computing ILC corrections, add noise to the reference trajectory proportional to current gate headroom. Gates with large headroom get more exploration; gate-3 with 0.024m headroom gets minimal noise.
- **Action smoothness regularization:** Add an action-rate penalty term to our trajectory cost function: `lambda * ||u_k - u_{k-1}||^2`. Their weights suggest the penalty should be 5-10x smaller than the tracking penalty in normal flight, increasing near gates.
- **Track-primitive decomposition:** Instead of optimizing the full racing line at once, decompose into primitives (turns, straights, gate approaches) and optimize each separately. This could reveal that gate-3's issue is a specific primitive type.
- **Success radius of 0.35m:** They define gate success at 0.35m distance — close to our 0.25m threshold. This suggests 0.25m is genuinely tight for high-speed racing.

## Limitations & Caveats

- Maximum speeds of 5 m/s are significantly slower than competitive drone racing (15-25 m/s). At our racing speeds, the dynamics are much more nonlinear and the control margins thinner.
- The RL policy is end-to-end from depth images — fundamentally different from our modular pipeline. Direct technique transfer requires adaptation to our trajectory-tracking architecture.
- 4 hours of training on an RTX 4090 sounds fast, but their DiffLab simulator is differentiable (built on Isaac Lab), which is not available in our PyBullet setup.
- Real-world results are qualitative (success/failure) rather than quantitative tracking error measurements, making direct comparison difficult.
- The two-phase approach may not work as cleanly for trajectory optimization as it does for RL — our optimizer is deterministic and may not benefit from the "exploration" that soft-collision enables in RL.

## Key Parameters/Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Depth image resolution | 96 x 72 | Downsampled input |
| Success distance | 0.35 m | Gate pass threshold |
| Body rate penalty (Phase 1/2) | -0.02 / -0.1 | 5x increase in Phase 2 |
| Action rate penalty (Phase 1/2) | -0.01 / -0.05 | 5x increase in Phase 2 |
| Collision penalty (Phase 1/2) | 50 / 100 | 2x increase in Phase 2 |
| Noise curriculum alpha_1 | growth rate | Slower than alpha_2 |
| Noise curriculum alpha_2 | decay rate | Faster than alpha_1 |
| L2C2 action oscillation reduction | ~25% | From ablation study |
| Training time | ~4 hours | RTX 4090, PPO |
| Max velocity achieved | 5.3 m/s | Forest track, no noise |
| Noise tolerance | up to 1.2m | With 50-80% success |
| Algorithm | PPO | Proximal Policy Optimization |
