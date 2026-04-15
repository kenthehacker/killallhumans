# RAPTOR: A Foundation Policy for Quadrotor Control (2025)

**Paper:** arXiv:2509.11481
**Authors:** Geles et al. (RPG, University of Zurich)
**Venue:** arXiv preprint, 2025

## Key Contribution

RAPTOR introduces a single, ultra-compact neural network policy (2,084 parameters) that can control diverse quadrotors zero-shot — from 32g micro-drones to 2.4kg platforms — without retraining or system identification. The core insight is that a tiny GRU-based network trained via meta-imitation learning across 1,000 randomized quadrotor models develops emergent in-context adaptation: within milliseconds of deployment, it infers the platform's dynamics from recent observation-action history and adjusts its control strategy accordingly. This is the first demonstration of a "foundation model" paradigm applied to low-level motor control, where a single policy replaces per-platform tuning entirely.

## Technical Approach

**Two-stage training pipeline:**

1. **Teacher generation:** 1,000 quadrotors are sampled from a broad distribution (mass 0.02--5 kg, thrust-to-weight 1.5--5.0, randomized inertias, thrust curves, and motor time constants). For each, a dedicated 3-layer MLP teacher (hidden dim 64, ~55x larger than the student) is trained with SAC (Soft Actor-Critic) for 1M steps (~31 min per teacher on a single CPU core, ~34 hours total on a consumer laptop).

2. **Meta-imitation learning:** The 1,000 teachers are distilled into a single GRU-based student policy (hidden dim 16, 3 layers, 2,084 params). Training runs for 1,000 epochs with a 10-epoch warm-up phase using teacher rollouts. Sequence length during training is 500 timesteps (5 seconds at control frequency). Total distillation takes 1.9 hours.

**Observation space:** Position error, rotation matrix, linear velocity, angular velocity, previous action.

**Action space:** Four normalized motor commands in [0, 1].

**Reward function:** r = -||p||_2 - 0.2*arccos(1-|q_z|) - ||a_t - a_{t-1}||_2 + 1.5 - 100*1[terminal]. This penalizes position error, orientation error, action jerk, and terminal crashes while providing a survival bonus.

**Domain randomization ranges:** Mass via cubic scaling (0.02--5 kg), thrust-to-weight ratio (1.5--5.0 uniform), inertia perturbation (+/-10%), quadratic thrust curve coefficients scaled by TWR*mass, variable motor time constants for rising and falling edges.

**Key architectural insight:** The GRU hidden state acts as a compressed system identification module. A linear probe on the hidden state achieves R^2 = 0.949 for predicting thrust-to-weight ratio, confirming the network learns to identify platform dynamics online. The context window extrapolates 10x beyond training (50s vs. 5s training sequences) without degradation.

## Results

- **Trajectory tracking (10s figure-eight):** Mean RMSE 0.11m (std 0.04m) across the fleet. Dedicated per-platform policy achieves 0.17m on Crazyflie vs. foundation's 0.19m — the foundation policy is competitive despite being 55x smaller.
- **Aggressive tracking (5.5s figure-eight):** Mean RMSE 0.20m (std 0.04m). Dedicated policy: 0.15m vs. foundation: 0.19m.
- **Real-world fleet:** 10 platforms from 31.9g to 2.4kg, zero-shot deployment on all.
- **Wind robustness:** 7 m/s sustained, gusts to 10 m/s, max ground speed 10 m/s, relative wind >15 m/s.
- **Recovery:** Mid-air activation from 4.5 m/s with arbitrary orientation stabilizes within milliseconds.
- **Compute:** Uses <10% of available compute on smallest microcontrollers.

## Relevance to Our System

**Current challenge:** We run a geometric SE(3) tracker (Lee et al.) with hand-tuned gains in `control/mpc_tracker.py`. Our avg tracking error is 0.138m at 14.08s race time. To push toward 12s, we need the controller to handle more aggressive trajectories (higher speeds, tighter turns) without tracking error blowing past 0.25m.

RAPTOR's approach is relevant in two ways:

1. **Adaptive control without gain scheduling:** Our current tracker requires manual gain tuning, and gains that work at moderate speed may be too conservative or unstable at higher speeds. RAPTOR's in-context adaptation could eliminate this — the network automatically adjusts to the current flight regime. However, replacing our entire control stack with an RL policy is a major architectural change.

2. **Disturbance rejection at high speed:** At 12s race time, aerodynamic effects (rotor drag, ground effect near gates) become significant. RAPTOR shows robust performance under wind and payload perturbations, suggesting its learned dynamics model compensates for unmodeled effects better than our model-based controller.

3. **The meta-imitation framework as a training recipe:** Even if we don't use RAPTOR directly, the two-stage distillation approach (many specialized teachers -> one generalist student) could be applied to train a racing-specific controller that generalizes across different speed profiles and trajectory aggressiveness levels.

## Actionable Takeaways

1. **Near-term (low risk):** Use RAPTOR's reward function structure as inspiration for tuning our tracker — specifically the action-smoothness penalty (||a_t - a_{t-1}||) which prevents oscillatory commands. We could add jerk penalization to our trajectory optimizer.

2. **Medium-term:** Implement an adaptive gain schedule in our geometric tracker inspired by RAPTOR's implicit system ID. The key idea: use recent tracking error history (a sliding window) to modulate controller gains, mimicking what the GRU hidden state does.

3. **Long-term:** If we hit a wall with model-based control at high speeds, RAPTOR's meta-imitation pipeline is a viable path to a learned controller. Training 1,000 teachers takes ~34 hours on a laptop, which is feasible. The 2,084-parameter student runs on microcontrollers, so latency is not a concern.

4. **Specific parameter insight:** RAPTOR achieves 0.20m RMSE on aggressive (5.5s) figure-eights. Our current 0.138m on a 14.08s race is comparable in difficulty. Pushing to 12s will likely increase our error to the 0.18--0.22m range based on RAPTOR's scaling, which still fits within our 0.25m budget.

## Limitations

- **No racing-specific evaluation:** RAPTOR is tested on figure-eight tracking, not gate racing. Racing involves discrete waypoints, high-curvature segments, and proximity to obstacles — regimes not directly validated.
- **Motor-level control assumption:** RAPTOR outputs direct motor commands. Our competition interface uses attitude commands (collective thrust + body rates via MAVLink). Adapting RAPTOR would require either changing our interface or adding a motor-mixing layer.
- **Training data requirements:** 1,000 teachers x 1M steps each = 1 billion total training steps. While feasible, this is a significant compute investment for uncertain racing performance gains.
- **No explicit constraint handling:** RAPTOR has no mechanism for actuator saturation constraints, speed limits, or collision avoidance — all critical for racing.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Student hidden dim | 16 (GRU) |
| Student params | 2,084 |
| Teacher hidden dim | 64 (MLP) |
| Num teachers | 1,000 |
| RL algorithm (teachers) | SAC |
| Training steps/teacher | 1M |
| Distillation epochs | 1,000 |
| Sequence length | 500 steps (5s) |
| Mass range | 0.02--5 kg |
| TWR range | 1.5--5.0 |
| Discount factor | 0.99 |
| Tracking RMSE (aggressive) | 0.20m mean |
| Tracking RMSE (moderate) | 0.11m mean |
| Total training time | ~36 hours (laptop) |
