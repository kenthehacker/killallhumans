# SPIRAL: Self-Play Incremental Racing Algorithm for Learning
- **URL**: https://arxiv.org/abs/2510.22568
- **Authors**: Onur Akgün
- **Year**: 2025
- **Venue**: IEEE ASYU 2025

---

## Key Contribution

SPIRAL presents a self-play training framework for multi-drone racing that autonomously generates escalating competitive difficulty. Rather than relying on hand-crafted curricula or fixed opponent policies, the approach bootstraps from a single-drone baseline and iteratively competes agents against the best previously saved versions of themselves. The core claim is that this produces faster lap times with lower variance than both naive PPO and game-theoretic (SE-IBR) baselines, specifically because competitive pressure from near-equal opponents drives speed maximization in a way that solo reward shaping cannot.

The three-stage curriculum — solo flight, 1v1 racing, 2v2 racing — is the structural contribution. Each stage inherits the policy trained in the previous stage, so the skill progression is compositional rather than from-scratch.

---

## Technical Approach

### Problem Formulation

The racing task is cast as a decentralized multi-agent MDP. Each agent i has:
- State sᵢ: 12D ego-state (position, linear velocity, Euler angles, angular velocity) plus a K=50-step contextual buffer holding relative positions of the Mₒ=2 closest opponents and Mₘ=2 upcoming gates, plus recent actions.
- Action aᵢ: 4D continuous vector [pˣ, pʸ, pᶻ, ψ]ᵀ — a desired position waypoint and yaw heading.
- Objective: maximize discounted cumulative reward Jᵢ(πᵢ) = E[Σ γᵗ Rᵢ,ₜ].

The decentralized structure means there is no shared reward signal; any apparent multi-agent coordination is emergent from individual optimization.

### Physics and Control Stack

Simulation runs in PyBullet with standard quadrotor rigid-body dynamics:

- Translational: m·p̈ = [0, 0, -mg]ᵀ + R·Tᵦ − Kₐ·ṗ (aerodynamic drag term Kₐ·ṗ included)
- Rotational: I·ω̇ = τ − ω × (I·ω)
- Motor allocation maps {ω₁², ω₂², ω₃², ω₄²} to thrust T and torques τ via thrust coefficient kf and drag torque coefficient km.

The control hierarchy is two-level: a 50 Hz high-level RL policy outputs desired position and yaw; a 240 Hz PID inner loop handles attitude stabilization. This separation is significant — the RL policy does not command motor RPMs directly but instead reasons in a slower, position-waypoint space.

### Reward Function

Total reward Rᵢ,ₜ = wₚ·Rₚ,ₜ + wᶜ·Rᶜ,ₜ + wₐ·Rₐ,ₜ + Rₜ,episode_end, where:

- **Progress reward** Rₚ,ₜ = α·Δdₜ + β·Gₜ: rewards forward centerline progress plus a gate-passage bonus.
- **Collision penalty** Rᶜ,ₜ = −C on any collision event.
- **Angular alignment** Rₐ,ₜ = ζ·max(0, cos(θₜ) − cos(θ_threshold)): rewards approaching gates with correct heading.
- **Lap-completion bonus** Rₜ,episode_end = 100/Tₗₐₚ: inversely proportional to lap time, directly incentivizing speed.

All scalar weights (wₚ, wᶜ, wₐ, α, β, ζ, θ_threshold) are tuned empirically rather than derived from theory.

### Self-Play Training Loop

At each stage, the loop is:
1. Roll out racing experience against opponents drawn from the self-play pool (past saved policies).
2. Update the current policy with PPO.
3. Periodically evaluate win rate and lap time.
4. Save improved snapshots as new opponent candidates.
5. Import the best saved model as the next opponent.

This is standard fictitious self-play, conceptually descended from AlphaGo/AlphaStar but applied to continuous-control racing rather than discrete strategy games.

### Staged Curriculum

- **Stage 1**: Single-drone solo flight — learns basic flight stability, gate navigation, lap completion.
- **Stage 2**: 1v1 racing — self-play against best prior policy; learns overtaking, dynamic obstacle avoidance.
- **Stage 3**: 2v2 racing — competes against two opponents simultaneously; learns crowded-environment adaptation.

The track is a PyBullet circuit with 6 gates. Evaluation uses 50 independent runs per method with randomized initial placements.

---

## Results

### 1v1 (Two-Drone) Racing

| Method | Lap Time (s) | Success Ratio |
|---|---|---|
| SPIRAL (PPO + self-play) | **13.71 ± 0.0005** | 0.69 ± 0.05 |
| SE-IBR (game-theoretic baseline) | 14.25 ± 0.30 | **0.81 ± 0.11** |
| PPO (no self-play) | 16.31 ± 0.07 | 0.80 ± 0.02 |

### 2v2 (Four-Drone) Racing

| Method | Lap Time (s) | Success Ratio |
|---|---|---|
| SPIRAL (PPO + self-play) | **13.71 ± 0.01** | 0.56 ± 0.01 |
| SE-IBR (game-theoretic baseline) | 14.39 ± 0.33 | **0.62 ± 0.22** |
| PPO (no self-play) | 17.82 ± 0.27 | 0.31 ± 0.12 |

Key observations:
- SPIRAL achieves the fastest lap times in both scenarios, with dramatically lower variance than SE-IBR.
- SE-IBR has higher success ratios, indicating more conservative/reliable behavior at the cost of ~0.5–0.7 s per lap.
- PPO without self-play degrades sharply from 1v1 to 2v2 (success 0.80 → 0.31), confirming that competitive self-play is load-bearing for multi-agent generalization.
- SPIRAL's 2v2 success ratio of 0.56 (vs 0.69 in 1v1) reveals that its aggressive policy incurs more collisions in crowded environments — a concrete speed-reliability trade-off.

---

## Relevance to Our System

### Progressive Speed Paradigm — Direct Analogy

SPIRAL's core insight maps directly onto a recurring design question in our ILC + trajectory planning pipeline: when is it safe to push speed?

Our current system starts with conservatively inflated segment times in the min-snap trajectory optimizer (`planning/trajectory_optimizer.py`) and applies ILC to improve tracking accuracy over laps. Once tracking error has been reduced by ILC, there is an opportunity to compress segment times toward the physical limits of the drone — but doing this prematurely (before ILC has converged) causes gate misses or crashes.

SPIRAL's staged curriculum is the multi-agent analogue of this progression: it does not attempt 2v2 racing until the drone has demonstrated competent solo flying and 1v1 racing. The underlying principle is identical — **capability must be established before challenge level is raised**. For our system, this translates to a concrete decision rule: reduce trajectory inflation only when ILC has converged (e.g., iteration-over-iteration improvement in avg_tracking_error_m drops below some threshold like 1%).

### Speed-Reliability Trade-off Quantification

The SPIRAL results provide a useful empirical data point: the fastest policy (13.71 s) accepts a 31–44% collision rate in multi-drone scenarios, while a more conservative policy (SE-IBR at 14.25–14.39 s) achieves 62–81% success. This ~0.7 s lap time difference (~5%) comes at a significant reliability cost. For our competition context (VQ1, single-drone, timed runs), crashes are disqualifying, so we sit firmly in the "success-first" regime — but it confirms that the last ~5% of speed comes with disproportionate risk, which should temper how aggressively we compress segment times in later ILC iterations.

### Self-Play vs. ILC as Improvement Mechanisms

Both SPIRAL and our ILC loop are mechanisms for incremental performance improvement through iterative feedback, but they operate differently:

- SPIRAL uses a pool of past policies as a curriculum signal; the "difficulty" increases as the self-play pool improves.
- Our ILC uses cross-track error from prior laps as a feedforward correction signal; the "difficulty" is fixed (the same track) and only the correction improves.

A potential synthesis: once ILC converges on the current trajectory, treating the converged trajectory as a "Stage 1 baseline" and then attempting a speed-up (reduced segment times) is structurally equivalent to SPIRAL's stage transitions. The stage transition criterion in SPIRAL is implicit (policy improvement plateaus); we could make it explicit by tracking ILC correction magnitude.

### State Representation and Predictive Horizon

SPIRAL's K=50-step contextual buffer (encoding the last 50 time steps of opponent and gate positions) is analogous to our `state_predictor.py` latency compensation — both are mechanisms to give the policy more temporal context than a single current observation. The 50-step buffer at 50 Hz represents a 1-second look-back window. Our EKF + latency predictor provides forward-looking state prediction, which is complementary. A combined approach (backward context buffer + forward prediction) could improve gate-approach planning in our MPC tracker.

---

## Actionable Takeaways

1. **Explicit stage transition criterion for trajectory speed-up**: Define a convergence threshold for ILC (e.g., Δavg_error < 1% iteration-over-iteration for 3 consecutive iterations) as the gate to reduce min-snap segment times. This operationalizes SPIRAL's implicit stage-transition logic.

2. **Accept the speed-reliability trade-off explicitly**: SPIRAL quantifies that the fastest 5% of speed comes with a ~30% collision rate penalty. For our single-drone competition use case, targeting 95% of physically optimal speed with <5% crash probability is a more rational operating point than maximum speed.

3. **Lap-time reward shaping**: SPIRAL's Rₜ,episode_end = 100/Tₗₐₚ directly incentivizes speed at episode end without penalizing intermediate conservatism. Our trajectory optimizer could adopt an analogous structure: minimize total segment time subject to tracking error constraints, rather than treating segment time as a fixed parameter.

4. **Multi-step contextual buffer for gate approach**: Implement a short (10–20 step) history buffer in the MPC tracker's state input, particularly encoding gate-relative position history. This could smooth the gate-approach behavior that currently relies solely on instantaneous EKF state.

5. **Opponent awareness as a future extension**: If the VQ1 competition involves simultaneous multi-drone runs on the same course (unclear from current spec), SPIRAL's architecture of incorporating relative opponent positions into the observation is the right approach. Worth confirming the competition format.

---

## Limitations & Caveats

1. **Low success ratios in multi-drone settings**: SPIRAL's best result is 56–69% success, meaning 31–44% of runs end in collision or course failure. For a competition with a single scored attempt, this is unacceptable. The framework optimizes for average speed across many runs, not reliability in a single run — a fundamentally different objective than ours.

2. **Decentralized reward inhibits coordination**: The paper explicitly acknowledges that "any emergent teamwork is a byproduct of individual competitive goals rather than true cooperative strategies." In 2v2, agents cannot be trained to cooperate, only to compete. This limits the framework's value for any team-racing scenario.

3. **No sim-to-real transfer discussion**: The paper operates entirely in PyBullet simulation and does not address sim-to-real transfer. Our system uses PyBullet as a test environment but targets real hardware (MAVLink/MAVSDK interface). SPIRAL's RL policy may not transfer without domain randomization or adaptation, which the paper does not evaluate.

4. **6-gate circuit only**: All experiments run on a single fixed track with 6 gates. Generalization to different track layouts, gate spacings, or gate orientations is not demonstrated. This limits confidence that the trained policies are robust rather than track-specific.

5. **Empirically tuned reward weights**: All scalar weights (wₚ, wᶜ, wₐ, α, β, ζ) are described as "tuned empirically" with no sensitivity analysis or ablation. The policy's behavior may be brittle to reward weight changes, and the paper provides no guidance on how to set these for a new track or drone platform.

6. **SE-IBR comparison is incomplete**: SPIRAL beats SE-IBR on lap time but loses on success ratio. The paper frames this as SPIRAL being "better" without acknowledging that for single-run competition use cases, success ratio is the primary metric. The comparison is somewhat cherry-picked.

7. **Venue and scope**: Published at IEEE ASYU 2025, a mid-tier regional conference. The work is a proof-of-concept rather than a production-grade system. Results should be treated as directionally interesting rather than definitive.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|---|---|---|
| State dimension (ego) | 12 | [pos, vel, euler, ang_vel] |
| Context buffer length K | 50 steps | History window (1 s at 50 Hz) |
| Opponents in observation Mₒ | 2 | Closest N opponents tracked |
| Gates in observation Mₘ | 2 | Upcoming N gates tracked |
| High-level policy frequency | 50 Hz | RL policy step rate |
| Low-level PID frequency | 240 Hz | Inner attitude control loop |
| Action space dimension | 4 | [pˣ, pʸ, pᶻ, ψ] — position + yaw |
| Lap-completion bonus | 100 / Tₗₐₚ | Speed-incentivizing episode reward |
| Training algorithm | PPO | Proximal Policy Optimization |
| Evaluation runs | 50 | Independent runs per method |
| Track gates | 6 | PyBullet circuit track |
| SPIRAL 1v1 lap time | 13.71 ± 0.0005 s | Best achieved |
| SPIRAL 2v2 lap time | 13.71 ± 0.01 s | Best achieved (4-drone) |
| SPIRAL 1v1 success ratio | 0.69 ± 0.05 | Collision-free gate-correct completion |
| SPIRAL 2v2 success ratio | 0.56 ± 0.01 | Collision-free gate-correct completion |
| SE-IBR 1v1 success ratio | 0.81 ± 0.11 | Highest reliability baseline |
