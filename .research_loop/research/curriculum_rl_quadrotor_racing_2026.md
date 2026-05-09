# Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles (2026)

**Paper:** arXiv:2602.24030
**Authors:** SJTU-ViSYS team (Shanghai Jiao Tong University)
**Venue:** arXiv preprint, February 2026
**Code:** https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing

## Key Contribution

This paper presents a complete end-to-end RL framework for autonomous quadrotor racing that achieves 100% gate pass rates across multiple track configurations while handling randomly placed obstacles. The key innovation is a three-stage curriculum learning strategy combined with a multi-scene updating approach that prevents catastrophic forgetting across diverse obstacle configurations. Unlike prior RL racing work that operates in obstacle-free environments, this system demonstrates robust real-world deployment at 8 m/s with a depth camera and onboard Raspberry Pi, making it one of the most practically validated RL racing systems to date.

## Technical Approach

**Three-stage curriculum:**

- **Level 1 (Foundation):** Obstacle-free racing at lower desired speeds. The drone learns basic gate traversal and trajectory following. Training continues until the drone can complete full tracks consistently. All reward components are active.

- **Level 2 (Obstacle introduction):** Random obstacles are placed in cuboid regions between consecutive gates. Speed remains low. The drone learns obstacle avoidance while maintaining gate traversal. Domain randomization on obstacle placement begins here.

- **Level 3 (Speed push):** Desired speed is increased to maximum. The velocity penalty term is removed from the reward, encouraging the policy to find its own speed-accuracy tradeoff. Obstacle density is increased. This stage produces the final racing policy.

**RL algorithm:** PPO (Proximal Policy Optimization) with:
- Learning rate: 1e-4 decaying to 1e-5
- Discount factor gamma = 0.99
- Clip range: 0.2
- GAE-lambda: 0.95
- Batch size: 51,200
- 100 parallel environments
- 100M timesteps per track

**Network architecture:**
- **Observation:** 17-dim state vector (relative positions to next 2 gates [6D], body-frame linear velocity [3D], desired speed [1D], quaternion orientation [4D], body-frame angular velocity [3D]) + 64x64 inverse depth map with Gaussian noise.
- **State branch:** 2-layer MLP [192, 96]
- **Depth branch:** 3-layer CNN processing the 64x64 depth image
- **Temporal integration:** GRU with 256-dim hidden state
- **Actor/Critic:** Separate MLP heads [192, 96] each

**Action space:** 4D collective-thrust + body rates (T, omega_x, omega_y, omega_z) — the CTBR interface, which matches our MAVLink command interface.

**Reward function (7 components):**
```
R_t = r_prog + r_theta + r_cmd + r_vd + r_avoid + r_pass + r_crash
```

| Component | Purpose | Weight |
|-----------|---------|--------|
| Progress (lambda_1 = 0.9) | Distance reduction to next gate | Dominant signal |
| Orientation (lambda_2 = 0.05) | Alignment with gate normal | Mild shaping |
| Action penalty (lambda_3 = -0.005) | Command magnitude regularization | Smoothness |
| Action smoothness (lambda_4 = -0.0025) | Command change penalty | Jerk reduction |
| Speed penalty (lambda_5 = -0.05) | Deviation from desired speed | Removed in L3 |
| Obstacle avoidance (lambda_6 = -0.01) | Inverse distance to obstacles | Safety |
| Gate pass bonus (lambda_7 = 5) | Binary gate traversal reward | Sparse but large |
| Crash penalty (lambda_8 = -4) | Binary collision penalty | Episode termination |

**Multi-scene updating strategy:** Instead of training on a single fixed obstacle configuration per rollout, the method maintains multiple scene variants and updates the policy across all of them. This prevents overfitting to specific obstacle arrangements and maintains generalization. The number of scenes is adjusted dynamically per curriculum stage.

**Domain randomization:**
- Drone position: +/-0.5m on all axes
- Gate locations: +/-1.0m (x-y), +/-0.3m (z)
- Obstacle placement: random within cuboid regions, 0.5m safety margin

## Results

**Simulation benchmarks (3 track configurations):**

| Method | S-track | 3D-Circle | J-track | Success Rate |
|--------|---------|-----------|---------|-------------|
| Vision baseline | 5.4s | 3.8s | 3.9s | 30--40% |
| State-based (no curriculum) | 3.5s | 3.8s | 3.1s | 20--30% |
| **Full CRL method** | **3.4s** | **3.6s** | **2.9s** | **100%** |

The full method achieves both the fastest times AND 100% success rates, while baselines show a stark speed-reliability tradeoff.

**Hardware-in-the-loop (10 m/s desired):** 100% success, lap times 3.2--4.1s across tracks.

**Real-world (8 m/s desired, Vicon + depth camera):** 100% success across 3 trials per track. Lap times: S-track 4.3s, 3D-Circle 5.0s, J-track 3.7s. Platform: 0.58 kg quadrotor, 14N max thrust, Raspberry Pi, Intel D435i at 30 Hz inference.

**Robustness to obstacle density:** 100% at 2 obstacles per 2-gate segment, degrades to 70--80% at 5 obstacles per segment.

**Robustness to gate position variance:** 100% at +/-0.3m, degrades to 60--80% at +/-1.0m.

## Relevance to Our System

**Current challenge:** Our system uses a classical pipeline (EKF -> trajectory optimizer -> geometric tracker). At 14.08s with 0.138m avg error, we want to push to 12s while staying under 0.25m error. The question is whether an RL approach could achieve this more effectively than continued classical tuning.

This paper is relevant in several specific ways:

1. **The CTBR action space matches our interface:** The policy outputs collective thrust + body rates, which is exactly what our MAVLink bridge sends to the flight controller. This means an RL policy from this framework could be a drop-in replacement for our geometric tracker, without changing the competition adapter layer.

2. **The curriculum design addresses our exact problem:** We want to trade accuracy for speed. Their Level 3 curriculum explicitly removes the speed penalty and lets the policy discover the speed-accuracy Pareto frontier. We could adapt this: train Level 1 at our current 14s pace, Level 2 at 13s, Level 3 at 12s with relaxed tracking constraints.

3. **The 100% success rate at competitive speeds is compelling:** Our classical pipeline achieves 100% gate pass rate but at a conservative speed. Their RL pipeline achieves 100% at 10 m/s (sim) and 8 m/s (real). For our track geometry, these speeds would correspond to roughly 11--13s race times.

4. **The GRU-based architecture handles temporal dynamics:** Our EKF + geometric controller pipeline has no memory of past performance. Their GRU hidden state (256-dim) implicitly tracks dynamics, disturbances, and trajectory progress. This could be particularly valuable for our ILC-style iterative improvements — the policy might learn lap-to-lap corrections naturally.

5. **30 Hz inference on Raspberry Pi is relevant:** Our control loop runs at >100 Hz, but the RL policy only needs to run at 30 Hz with the inner-loop attitude controller handling high-frequency stabilization. This is compatible with our architecture where the tracker outputs attitude commands at a slower rate than the inner loop.

## Actionable Takeaways

1. **Hybrid approach (medium risk, high reward):** Keep our classical pipeline for state estimation and trajectory planning, but replace the geometric tracker with a trained RL policy that takes (state, next 2 gates) as input and outputs CTBR commands. The reward function from this paper provides a well-validated starting point. Train in our PyBullet sim with domain randomization on gate positions and drone dynamics.

2. **Curriculum for speed progression:** Implement a 3-stage curriculum in our PyBullet environment:
   - Stage 1: Current trajectory (14s), reward gate passage + low tracking error
   - Stage 2: Faster trajectory (13s), same rewards
   - Stage 3: Aggressive trajectory (12s), remove speed penalty, rely on gate pass reward

3. **Borrow the reward function structure:** Even without full RL, the reward weights reveal useful insights for our classical system:
   - Progress reward (0.9 weight) should dominate — we should prioritize forward progress over tracking precision
   - Action smoothness penalty (0.0025) is small but important — our controller should not oscillate
   - Gate pass bonus (5.0) is 10x the progress reward — gate traversal is paramount
   - The speed penalty removal in Level 3 is the key insight: stop penalizing the controller for going fast, only penalize for missing gates and crashing

4. **Use their obstacle avoidance as gate-margin shaping:** Their 1/(d_col + b_omega) obstacle reward could be adapted as a gate-centering reward: penalize distance from gate center, creating a natural accuracy incentive without explicit tracking error penalties.

5. **Training budget estimate:** 100M timesteps with 100 parallel envs at ~6000 FPS = ~4.6 hours of training per track. With our PyBullet sim running at maybe 1000 FPS across parallels, training would take ~28 hours per track configuration. Feasible for a single iteration.

## Limitations

- **Sim-to-real gap at high speeds:** Real-world tests are at 8 m/s, while sim results use 10+ m/s. The 20% speed reduction for real deployment suggests non-trivial sim-to-real transfer losses. For our competition at >15 m/s equivalent speeds, this gap could be larger.
- **Vicon dependency:** Real-world experiments use motion capture for state estimation. Our competition uses onboard vision (gate PnP + EKF). The policy's robustness to noisy state estimates is unvalidated.
- **Fixed track geometry:** Training and evaluation use fixed track configurations (S, J, Circle). Generalization to unseen track layouts is not demonstrated. Our competition track changes between races.
- **No aerodynamic modeling:** The simulation (VisFly/Habitat-Sim) likely uses simplified dynamics without rotor drag or ground effect. At high speeds these effects become significant.
- **Depth camera requirements:** The 64x64 depth map input requires a depth sensor we may or may not have in competition. However, the state-based variant (without depth) also achieves competitive times, just with lower obstacle success rates.
- **100M steps per track:** If the competition track changes, retraining takes hours. No online adaptation mechanism exists.
- **Degradation at high gate variance:** Performance drops to 60--80% at +/-1.0m gate position uncertainty. Competition gates may have similar or larger uncertainty from our PnP estimation.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| RL algorithm | PPO |
| Learning rate | 1e-4 -> 1e-5 |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Batch size | 51,200 |
| Parallel envs | 100 |
| Training steps | 100M per track |
| GRU hidden dim | 256 |
| State MLP layers | [192, 96] |
| Depth CNN layers | 3 |
| Action space | 4D (T, wx, wy, wz) |
| Observation dim | 17 + 64x64 depth |
| Gate pass reward | +5.0 |
| Crash penalty | -4.0 |
| Progress weight | 0.9 |
| Real-world speed | 8 m/s |
| Sim speed | 10+ m/s |
| Inference rate | 30 Hz |
| Platform mass | 0.58 kg |
| Max thrust | 14 N |
| Sim FPS | ~6000 |
| Gate position randomization | +/-1.0m (xy), +/-0.3m (z) |
| Drone position randomization | +/-0.5m (xyz) |
