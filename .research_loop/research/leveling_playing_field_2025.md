# Leveling the Playing Field (2025)

- **URL**: https://arxiv.org/abs/2506.17832
- **Authors**: Pratik Kunapuli, Jake Welde, Dinesh Jayaraman, Vijay Kumar (University of Pennsylvania)
- **Year**: 2025
- **Venue**: Not yet published in venue (preprint as of mid-2025)
- **Code**: https://pratikkunapuli.github.io/rl-vs-gc/

---

## Key Contribution

The paper's central claim is that prior RL-vs-GC comparisons suffered from three systematic, asymmetric advantages given to RL over geometric control (GC):

1. **Objective function access** — RL is trained with an explicit reward function, while GC was hand-tuned without access to the same formal optimization target.
2. **Task-aligned data access** — RL is trained on the exact task distribution at test time; GC was often tuned on simpler hovering tasks and then transferred.
3. **Feedforward information** — GC's control laws depend on up to 4th-order derivatives of the reference trajectory, but many published comparisons omitted these terms.

When these asymmetries are eliminated — GC is tuned via Bayesian optimization against the same reward, on the same task, with full feedforward — the performance gap shrinks dramatically. The authors conclude that RL and well-tuned GC are much closer competitors than the literature suggested, and that the superiority of each depends heavily on the agility regime.

---

## Technical Approach

### Geometric Controller (GC)

The GC follows the Lee et al. SE(3) geometric control structure, organized as a hierarchical two-loop controller:

- **Outer loop (position)**: Computes desired acceleration from position and velocity errors plus feedforward reference acceleration:

  ```
  p_ddot_des = -Kp*(p - p_d) - Kv*(v - v_d) - m*g*z_W + p_ddot_d
  ```

  where `Kp` and `Kd` (velocity gain `Kv`) are PD gains, and `p_ddot_d` is the feedforward term from the trajectory.

- **Inner loop (attitude)**: Computes torques from orientation and angular rate errors, incorporating feedforward angular velocity and angular acceleration derived from differentiating the desired orientation.

The feedforward terms require up to **4th-order position derivatives** and **2nd-order yaw derivatives** from the reference trajectory. In practice, these are approximated by appending a sequence of future reference positions and yaws to the observation, then finite-differencing over the horizon. This is computationally cheap and does not require an analytic trajectory representation — any trajectory that can supply a look-ahead window is sufficient.

The controller has **8 tunable parameters** (PD gains for x/y treated symmetrically, z treated separately, plus attitude gains). These are optimized using **Optuna (Bayesian optimization)** against the same reward function used to train the RL policy.

### Reinforcement Learning (RL) Controller

- Architecture: 3-layer MLP, 256 units per layer
- Algorithm: Proximal Policy Optimization (PPO)
- Training: 200 million timesteps, ~30 minutes, 4096 parallel environments in IsaacLab
- Control frequency: 50 Hz; simulation: 100 Hz
- Domain randomization applied to physical parameters

The RL policy receives current state and a look-ahead window of future reference positions/yaws as input — the same information given to the GC for finite-difference feedforward computation.

### Reward Function (Table III)

The shared reward function used for RL training and GC optimization:

| Term | Weight | Notes |
|------|--------|-------|
| Position | λp = 15.0·dt | Tolerance δp annealed 0.8→0.1 |
| Orientation | λR = −4.0·dt | |
| Velocity | λv = −0.05·dt | |
| Angular velocity | λω = −0.01·dt | |

The position term uses a smooth tolerance function (δp) rather than a hard penalty, and δp is annealed from 0.8 m down to 0.1 m during training to focus the policy progressively on tighter tracking.

---

## Results

### Quadrotor Lissajous Trajectory Tracking (Table IV, primary benchmark)

| Controller | Avg Reward | Position RMSE (m) | Yaw RMSE (rad) |
|------------|-----------|-------------------|----------------|
| RL | 14.196 ± 0.48 | **0.119 ± 0.05** | **0.274 ± 0.15** |
| GC | 13.447 ± 1.61 | 0.158 ± 0.20 | 0.483 ± 0.29 |

RL wins on average RMSE, but GC's standard deviation is much larger — GC achieves near-zero steady-state error on portions of the trajectory but occasionally exhibits larger transient spikes, pulling up the mean.

### Aerial Manipulator (heavier, more complex system)

| Controller | Position RMSE (m) | Yaw RMSE (rad) |
|------------|-------------------|----------------|
| RL | 0.118 ± 0.05 | 0.487 ± 0.26 |
| GC | **0.136 ± 0.10** | **0.405 ± 0.29** |

GC achieves better yaw tracking; RL better position. The gap is narrow.

### Ball-Catching (agility task, Table V)

The ball-catching task requires the drone to intercept a ballistic projectile — a highly agile, time-critical task. At the shortest catch-time window (0.79 s):

- RL catch success: **65%**
- GC catch success: **30%**

As the catch time window increases (more time available), the gap narrows. This confirms RL's advantage specifically in the high-agility transient regime.

### Domain Randomization (Table VI)

Under parameter uncertainty (0–40% randomization of mass, inertia, etc.):

- RL maintains consistent performance across randomization levels
- GC degrades noticeably — its gains are tuned for nominal parameters

RL's implicit robustness from training-time domain randomization is a genuine structural advantage.

### Realistic vs. Simple Motor Dynamics (Table VII)

Adding first-order motor dynamics (thrust lag) and control delays:

- RL trained on realistic dynamics: reward **13.053 ± 1.17** (best overall)
- GC: minimal sensitivity to modeling complexity

GC's robustness here comes from its model-free feedback structure; RL can match or exceed it only when trained on a sufficiently realistic model.

---

## Relevance to Our System

Our system uses the SE(3) geometric tracker in `/control/mpc_tracker.py` with a `TrackerConfig` gains structure. The paper's findings are directly actionable:

1. **Feedforward terms are the most important single fix.** If our geometric tracker is not using reference acceleration, reference angular velocity, and reference angular acceleration from the trajectory, we are leaving significant tracking error on the table. The paper quantifies this as the largest single contributor to GC's poor performance in prior comparisons.

2. **Bayesian gain optimization matters.** Our gains are likely hand-tuned. The paper shows that even without RL, optimizing 8 gain parameters with Optuna against the actual task reward substantially closes the gap with RL.

3. **GC achieves zero steady-state error; RL does not.** For our application — following a pre-planned racing line through gates — the trajectory is not adversarial and the drone has time to settle. GC's structural property of integral-free zero steady-state error (via accurate feedforward) is a direct advantage for gate-passing accuracy.

4. **RL's advantage is in highly agile transient responses.** Our race scenario involves sustained high-speed flight through a fixed gate sequence. Unless we are doing aggressive snap maneuvers between gates with very short segments, GC with full feedforward should be competitive with RL.

5. **The finite-difference feedforward trick is cheap.** Appending a look-ahead window of future reference points and finite-differencing to get derivatives avoids requiring analytic polynomial derivatives. Our `trajectory_optimizer.py` already produces min-snap polynomials, which means we can compute exact derivatives — this gives us even better feedforward than the paper's finite-difference approximation.

---

## Actionable Takeaways

### Priority 1: Verify and enable full feedforward in `mpc_tracker.py`

The geometric controller must include:
- Reference acceleration `p_ddot_d` in the position loop
- Reference angular velocity `omega_d` in the attitude loop
- Reference angular acceleration `alpha_d` in the attitude loop

Our min-snap trajectories provide analytic access to all derivatives up to snap (4th order). Use them. The feedforward thrust term should be:

```
F_ff = m * (p_ddot_d + g*z_W)
```

The feedforward attitude terms require computing the desired orientation from `p_ddot_d` and then differentiating that orientation in time to get `omega_d` and `alpha_d`. Lee et al. (2010) provides the exact formulas.

### Priority 2: Tune gains with Bayesian optimization

Replace manual gain tuning with Optuna against a simulation reward matching the benchmark thresholds. The paper used 8 parameters (Kp_xy, Kv_xy, Kp_z, Kv_z, KR, Komega, plus possibly feedforward scaling factors). Even a grid search over these 8 parameters in simulation would substantially improve over hand-tuning.

### Priority 3: Accept the steady-state/transient tradeoff

If we observe persistent offset errors in steady-state cruise segments, that points to missing feedforward. If we observe large overshoot at gate entries/exits (transient), that is acceptable GC behavior that RL would partially suppress — but for gate passing, the steady-state accuracy is what determines success.

### Priority 4: Do not implement RL unless feedforward GC fails

The paper's conclusion is that a properly implemented GC is competitive with RL for trajectory tracking tasks. Given our timeline (VQ1 May 2026), implementing and training RL correctly (200M steps, domain randomization, realistic motor model) is a high-risk, high-cost path. Fix the GC feedforward first.

---

## Limitations & Caveats

- **Simulation only.** All experiments are in IsaacLab. Sim-to-real transfer issues (rotor wash, vibration, camera latency) are not evaluated. GC may be more robust to sim-to-real gap due to its model-free feedback structure.

- **Exact gain values not published.** The paper does not release the Bayesian-optimized gain values from their experiments. The 8-parameter search space is described but the specific solution is not tabulated.

- **Lissajous trajectory only (primary benchmark).** The Lissajous figure is a smooth, moderate-agility trajectory. Racing lines through gate sequences may have sharper curvature changes, favoring RL more.

- **No gate detection / perception loop.** The paper assumes perfect state knowledge. Our system's EKF uncertainty and gate detection errors will add tracking error beyond what the controller comparison captures.

- **Motor model matters for RL.** RL trained on simple dynamics underperforms GC in some conditions (Table VII). The quality of the motor model used for RL training critically affects the result.

---

## Key Parameters / Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| Position reward weight λp | 15.0·dt | Table III |
| Orientation reward weight λR | −4.0·dt | Table III |
| Velocity reward weight λv | −0.05·dt | Table III |
| Angular velocity reward weight λω | −0.01·dt | Table III |
| Position tolerance δp (initial) | 0.8 m | Table III / text |
| Position tolerance δp (final) | 0.1 m | Table III / text |
| RL network size | 3-layer MLP, 256 units | Text |
| RL training timesteps | 200M | Text |
| Control frequency | 50 Hz | Text |
| Simulation frequency | 100 Hz | Text |
| Number of GC tunable parameters | 8 | Text |
| Feedforward derivative order (position) | Up to 4th (snap) | Text |
| Feedforward derivative order (yaw) | Up to 2nd (angular accel) | Text |
| RL position RMSE (quadrotor, Lissajous) | 0.119 ± 0.05 m | Table IV |
| GC position RMSE (quadrotor, Lissajous) | 0.158 ± 0.20 m | Table IV |
| RL ball-catch rate (0.79 s window) | 65% | Table V |
| GC ball-catch rate (0.79 s window) | 30% | Table V |
| RL domain randomization robustness | Consistent 0–40% | Table VI |
| GC domain randomization robustness | Degrades under uncertainty | Table VI |
