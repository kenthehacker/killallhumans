# Adaptive Gain Scheduling using Reinforcement Learning for Quadcopter Control

- **URL**: https://arxiv.org/abs/2403.07216
- **Year**: 2024
- **Authors**: Not explicitly listed in HTML version (March 2024 preprint)

---

## Key Contribution

This paper proposes using Proximal Policy Optimization (PPO) to learn a policy that dynamically adjusts the proportional gains of a cascaded feedback controller during flight, rather than relying on a single set of manually tuned static gains. The core insight is that optimal gains vary across flight regimes — what works for straight-line cruise is suboptimal in sharp turns or during aggressive maneuvers — so an adaptive policy that continuously modulates gains based on current tracking error state can outperform any fixed configuration.

The main technical contribution is framing gain scheduling as a reinforcement learning problem: the RL agent observes the current tracking error vector and outputs continuous gain adjustments for six control channels simultaneously. This is meaningful because it sidesteps the need for explicit gain-scheduling lookup tables (which require human insight into operating regimes) and instead learns the mapping from error state to optimal gains directly from reward signal. The reported result — roughly 43-49% reduction in Integral Squared Error (ISE) and Integral Time Squared Error (ITSE) across all tested trajectories — is a strong quantitative claim that motivates the approach.

---

## Technical Approach

### Cascaded Controller Structure

The underlying controller is a cascaded proportional feedback architecture operating in 2D (3-DOF: x-translation, y-translation, planar rotation). The cascade has three stages:

1. **Position loop**: Position error → desired velocity command via `Kpx`, `Kpy`
2. **Velocity loop**: Velocity error → desired angle command via `KpVx`, `KpVy`
3. **Attitude loop**: Angle error → thrust differential via `Kpθ`, `Kpω`

This is architecturally similar to a PD-cascade, where each gain controls one error channel at one cascade level. The RL policy's job is to set all six gains at each timestep.

### Quadcopter Dynamics Model (2D)

The planar quadcopter model has six continuous states:

```
x = [px, py, θ, Vx, Vy, ω]
```

where `px`, `py` are positions, `θ` is heading angle, `Vx`, `Vy` are linear velocities, and `ω` is angular velocity. Forces are two scalar thrusts `T₁`, `T₂` from left/right rotors, with aerodynamic drag. Physical parameters: mass = 2.5 kg, propeller span = 1.0 m.

The equations of motion follow standard rigid-body dynamics with proportional drag in both translational and rotational axes. This 2D model was intentionally simplified to allow fast simulation during training without hardware risk.

### RL Problem Formulation (MDP)

The environment is implemented via the Gymnasium API as a discrete-time MDP:

**State space** (ℝ⁶): Current tracking error vector
```
s = [ex, eVx, eθ, eω, ey, eVy]
```
where `ex = px_ref - px`, `eVx = Vx_ref - Vx`, etc. Crucially, the policy observes only the *current* error — it does not see upcoming waypoints or trajectory curvature. This means the policy is reactive, not predictive.

**Action space** (ℝ⁶, bounded in [-1, 1], then linearly rescaled):

| Gain   | Min   | Max   |
|--------|-------|-------|
| Kpx    | 0.5   | 2.0   |
| KpVx   | -0.5  | -0.1  |
| Kpθ    | 5.0   | 10.0  |
| Kpω    | 10.0  | 16.0  |
| Kpy    | 0.5   | 3.0   |
| KpVy   | 5.0   | 15.0  |

Note `KpVx` is strictly negative — this encodes the directional convention of the velocity-to-angle mapping in that cascade layer.

**Reward function**: A multi-term shaped reward:
- **Timeout** (episode > 1.2× expected duration): -1
- **Out-of-bounds** (deviation > 10 m): -5
- **Success** (reached target): `+10 / ISE` — rewards precision, not just arrival
- **Intermediate progress**: `+0.05 × min(prev_deviation / curr_deviation - 1, 2)` — shaped reward for getting closer, capped at 2 to prevent exploitation

The success reward `10/ISE` is particularly well-designed: it incentivizes the agent to not just reach the goal but to track the path precisely, which is exactly what distinguishes good trajectory following from crude waypoint chasing.

### PPO Algorithm

PPO was selected for its favorable sample efficiency through clipped surrogate objective, which allows multiple gradient steps per batch of experience without catastrophic policy collapse. The policy gradient update is:

```
∇θ J(θ) = E_π [ Q(s, a) ∇θ log π(a|s) ]
```

with the clipped surrogate to bound the policy ratio update. The actor-critic architecture uses a shared value function `V(s)` for advantage estimation.

**Training hyperparameters:**

| Parameter               | Value         |
|-------------------------|---------------|
| Total timesteps         | 2.4 × 10⁵     |
| Parallel environments   | 3             |
| Steps per env per update| 2,048         |
| Effective batch per update | 6,144      |
| Mini-batch size         | 64            |
| Discount factor γ       | 0.99          |
| Learning rate η         | 3 × 10⁻⁴     |

Training is relatively lightweight (2.4×10⁵ steps) — convergence was observed around 16,000 steps, suggesting the problem is relatively low-dimensional and the shaped reward accelerates learning.

### Policy Network

The paper does not specify the exact MLP architecture (layer count, widths, activation functions), which is a gap in reproducibility. Given the 6-input, 6-output structure and the small action space, a shallow MLP (2-3 layers, 64-128 units) is likely sufficient.

---

## Results

### Convergence During Training

- Mean episodic reward converged to ~25 by ~16,000 steps
- Entropy loss decreased monotonically (policy becomes more deterministic as it specializes)
- Explained variance reached 0.94 (value function fits well; stable advantage estimation)
- Value loss stabilized at 1.15

### Quantitative Tracking Performance

Three test trajectories were evaluated, comparing static (manually tuned) gains vs. RL-adaptive gains:

| Trajectory | ISE (static) | ISE (RL) | ISE Improvement | ITSE (static) | ITSE (RL) | ITSE Improvement |
|------------|-------------|----------|-----------------|---------------|-----------|------------------|
| 1          | 0.582       | 0.330    | **43.3%**       | 71.651        | 37.551    | **47.6%**        |
| 2          | 0.526       | 0.290    | **44.9%**       | 86.409        | 44.137    | **48.9%**        |
| 3          | 0.357       | 0.205    | **42.6%**       | 66.681        | 36.120    | **45.8%**        |
| **Average**|             |          | **43.6%**       |               |           | **47.4%**        |

ITSE improvements are consistently larger than ISE improvements (47.4% vs. 43.6%), which indicates the RL controller not only has lower peak errors but also converges faster — the time-weighting penalizes sustained errors more, so ITSE improvement implies faster error decay dynamics.

The baseline is static gains that were presumably hand-tuned, not PID-optimized by Bayesian optimization or grid search, so the improvement margin may be more modest against a properly optimized static controller. However, the consistency across all three trajectories (42-49% range) suggests this is robust, not cherry-picked.

---

## Relevance to Our System

Our system uses a PD+feedforward controller with fixed gains (kp=6, kd=4, feedforward_accel=0.4) with 50ms predictive lookahead. The current performance bottleneck is helix turn tracking: gate-7 has 0.659m error compared to ~0.12-0.27m on straight segments. This 2.4-5.5× error ratio between turns and straights is exactly the regime-varying problem this paper addresses.

The fundamental insight transfers directly: our fixed kp=6 and kd=4 are likely overfit to one operating regime (either turns or straights, probably straights given that's where most gates lie). During high-curvature segments, the drone needs higher derivative gain (more damping relative to proportional) to resist centripetal deviation; on straights, higher proportional gain can be tolerated without oscillation. A static gain compromise between these two regimes is inherently suboptimal.

Our architecture already has the right structure for this adaptation. The 50ms lookahead in `state_predictor.py` provides trajectory-aware context — we could include upcoming curvature or speed profile as additional state inputs to the gain scheduler, making it predictive rather than purely reactive (which is the main limitation of this paper's approach). This would be a direct architectural improvement over the paper.

The ITSE improvement (47.4% average) is more relevant to us than ISE, since gate-7's 0.659m error is sustained through the helix, not just a transient spike. If we could apply this and achieve even half the improvement, we would bring gate-7 from 0.659m to ~0.33m, which would beat our 0.5m threshold.

---

## Actionable Takeaways

1. **Implement trajectory-aware gain scheduling in `control/mpc_tracker.py`**: Instead of fixed `kp=6, kd=4`, parameterize gains as functions of a context vector. Start simple: compute curvature of upcoming trajectory (from `planning/trajectory_optimizer.py`) and scale `kd` up during high-curvature segments. Rule of thumb from this paper: gains should vary ~2-4× across their feasible range.

2. **Define gain search bounds by analogy to paper's structure**: The paper's action space bounds (e.g., Kpx in [0.5, 2.0] — a 4× range, Kpω in [10, 16] — a 1.6× range) were tuned to avoid instability. Our equivalent would be kp in [3, 9] and kd in [2, 8], with tighter limits on whichever axis caused oscillation in prior experiments.

3. **Use curvature and speed as scheduling inputs, not just current error**: The paper's policy is purely reactive (observes only current errors). Our system knows the reference trajectory ahead — use upcoming curvature radius, reference speed, and segment type (helix vs. straight) as inputs to the gain scheduler. This gives predictive gain adjustment, correcting the main limitation of this work.

4. **Tune for ITSE, not ISE**: The paper shows ITSE improvements exceed ISE improvements (47.4% vs. 43.6%), confirming that the RL policy learns to reduce *sustained* errors more than peak errors. When evaluating our adaptive gains, use a time-weighted metric to measure whether gains recover faster after disturbances — this is the relevant quantity for race time.

5. **Warm-start with PPO if implementing RL**: If taking the full RL approach, replicate their lightweight setup: 3 parallel envs, 2.4×10⁵ steps, reward = 10/ISE for success + shaped progress reward. Use our existing PyBullet adapter for the sim environment. Expected convergence in ~16k steps implies this can be prototyped in under an hour of training.

6. **Design a rule-based scheduler as a prior**: Before RL, implement a hand-crafted two-regime scheduler (helix: kp=5, kd=6; straight: kp=7, kd=3) to test the hypothesis that variable gains help. Benchmark this first — if it improves gate-7 error, then RL can refine it further.

7. **Add gain state to benchmark logging**: Extend `simulation.per_gate_avg_error` output to also log which gains were active at each gate. This will help identify whether gain schedule transitions are correctly timed relative to gate entry/exit.

8. **Consider gain interpolation, not hard switching**: Abrupt gain switches cause transient control spikes. Smooth the gain schedule by interpolating over a 0.1-0.2s window centered on trajectory curvature inflection points. This avoids the oscillation that rigid regime boundaries can introduce.

---

## Limitations & Caveats

1. **2D only (3-DOF)**: The entire paper is in planar simulation. The cascade structure for a full 6-DOF quadcopter is substantially more complex — attitude control (roll/pitch/yaw) introduces coupling that the 2D model ignores. The 43% improvement figure may not transfer directly to 3D racing.

2. **Purely reactive policy (no trajectory preview)**: The observation space is only current error, not upcoming trajectory shape. This means the policy cannot pre-emptively adjust gains before entering a curve — it must react after the error grows. Our system can do better by feeding upcoming curvature into the scheduler.

3. **No stability guarantees**: The authors explicitly note this gap. A gain scheduler that occasionally drives the system unstable (selecting very high gains in a poorly understood regime) would be catastrophic in a race. Any implementation needs hard gain bounds and emergency fallback to known-stable static gains.

4. **Baseline quality unclear**: The static baseline is "manually tuned" but details are sparse. If the static baseline was not itself optimized (e.g., via Bayesian optimization), the 43% improvement is an upper bound on what RL adds over a good static controller.

5. **Single step reference used for training**: Training used a 1m step reference (point-to-point), which is simpler than continuous trajectory tracking. Performance on complex helical paths (our bottleneck) may differ from the reported test trajectories.

6. **No noise model during training**: The training environment appears noise-free. Real-world application (and our PyBullet simulation) has motor noise, sensor delays, and aerodynamic effects. Domain gap may reduce the 43% improvement when applied to simulation with realistic disturbances.

7. **Hardware not tested**: No physical drone experiments. The paper is purely simulation-validated.

---

## Key Parameters / Constants

| Parameter                     | Value           | Source         |
|-------------------------------|-----------------|----------------|
| Drone mass                    | 2.5 kg          | Dynamics model |
| Propeller span                | 1.0 m           | Dynamics model |
| Kpx range                     | [0.5, 2.0]      | Action space   |
| KpVx range                    | [-0.5, -0.1]    | Action space   |
| Kpθ range                     | [5.0, 10.0]     | Action space   |
| Kpω range                     | [10.0, 16.0]    | Action space   |
| Kpy range                     | [0.5, 3.0]      | Action space   |
| KpVy range                    | [5.0, 15.0]     | Action space   |
| PPO learning rate             | 3 × 10⁻⁴       | Training config|
| PPO discount γ                | 0.99            | Training config|
| PPO mini-batch size           | 64              | Training config|
| PPO total timesteps           | 2.4 × 10⁵      | Training config|
| Parallel envs                 | 3               | Training config|
| Steps per env per update      | 2,048           | Training config|
| Convergence step (approx.)    | ~16,000         | Training curves|
| Timeout penalty               | -1              | Reward function|
| Out-of-bounds penalty         | -5              | Reward function|
| Success reward                | 10 / ISE        | Reward function|
| Progress reward               | 0.05 × min(prev/curr - 1, 2) | Reward function |
| Avg ISE improvement           | ~43.6%          | Results        |
| Avg ITSE improvement          | ~47.4%          | Results        |
| Explained variance (converged)| 0.94            | Training curves|
| Value loss (converged)        | 1.15            | Training curves|
