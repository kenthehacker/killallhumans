# TACO: Trajectory-Aware Controller Optimization for Quadrotors
- **URL**: https://arxiv.org/abs/2511.02060
- **Authors**: Hersh Sanghvi, Spencer Folk, Vijay Kumar, Camillo Jose Taylor
- **Year**: 2025
- **Venue**: arXiv preprint (submitted to ICRA 2026), November 2025

---

## Key Contribution

TACO introduces a framework for adapting quadrotor controller gains in real time based on the upcoming reference trajectory and the current vehicle state. The core insight is that static, hand-tuned gains are a compromise: gains that work well on gentle curves are too weak for aggressive turns, while gains that handle tight corners cause oscillation on smooth straight segments. TACO replaces this compromise with a learned predictive model that, given the current 8-dimensional gain vector plus a 1-second lookahead of the reference trajectory, predicts a multi-dimensional performance vector (tracking errors, control effort). A lightweight sampling-based optimizer then searches for the gain vector that minimizes the predicted cost, re-running every 0.5 seconds with only ~20 ms of compute per iteration.

A secondary contribution is a trajectory adaptation mechanism that modifies the reference path itself to be more dynamically feasible, operating through null-space parameterization of the polynomial spline constraints so waypoint boundary conditions are never violated. The two mechanisms (gain adaptation + trajectory adaptation) are complementary and combine for the best tracking performance. The work also delivers a highly parallelized quadrotor simulator built in PyTorch (refactoring RotorPy), achieving ~17,000 frames per second for batches of 10,000 simulated vehicles, enabling the 8 million datapoints needed for training.

---

## Technical Approach

### Controller

TACO wraps a standard **geometric tracking controller** (Lee et al. SE(3) formulation), identical in structure to the one already in our system. The controller computes:

```
Thrust:  f  = (-kp·ep - kv·ev - mg·e3 + m·p̈_d) · R·e3
Moment:  M  = -kR·eR - kΩ·eΩ + (Ω×JΩ) - J(Ω̂R^T Rd Ωd - R^T Rd Ω̇d)
```

where ep, ev, eR, eΩ are position, velocity, attitude, and angular velocity errors. The gain vector optimized by TACO is:

```
g = [kp^x, kp^y, kp^z, kv^x, kv^y, kv^z, kR, kΩ] ∈ ℝ^8
```

This is an 8-dimensional per-axis gain set, which is richer than the 4-scalar setup in our current `TrackerConfig` (kp_xy, kd_xy, kp_z, kd_z). The attitude and angular rate gains (kR, kΩ) are also included in the optimization.

### Learned Predictive Model

**Architecture:** MLP with hidden layers [512, 512, 256, 256] and ReLU activations.

**Inputs (per optimization query):**
1. Gain vector g ∈ ℝ^8
2. Current robot state o_n = [v_n, q_n, ω_n] ∈ ℝ^10 (velocity, quaternion, angular velocity)
3. Trajectory lookahead τ̄_{n:n+H}: H=100 steps at Δt=0.05 s intervals (= 1 second horizon), expressed as position-relative offsets from the drone's current pose

**Outputs:** 8-dimensional cost vector c containing:
- Average absolute tracking error per axis (x, y, z)
- Average velocity tracking error
- Average absolute pitch/roll rate (proxy for oscillation)
- Average control effort (thrust + moment magnitude)
- Terminal position error at end of horizon

The model is trained with random gains drawn uniformly from [g_min, g_max], random initial state offsets, and trajectories from four families: MinSnap (constant velocity), MinSnap (varying velocity), random polynomial (via null-space sampling), and ZigZag straight-line segments.

### Optimization Procedure (TACO Algorithm)

Executed every T = 0.5 s (2 Hz replanning):

1. Generate N random gain samples G_r = U(g_min, g_max)
2. Concatenate each with current τ̄ and o_n
3. Forward-pass through MLP → predicted cost vectors
4. Select lowest-cost sample g*
5. In subsequent iterations: apply N_r perturbations around previous g*
6. Evaluate perturbed samples through MLP
7. Deploy g* to controller

This is essentially random-restart hill climbing (CEM-lite) with the MLP as a cheap surrogate. The MLP forward pass is far cheaper than rolling out a simulation, enabling ~20 ms per full TACO iteration on an Apple M3. The approach is "orders of magnitude faster than Bayesian Optimization baselines" (which took ~45 minutes for 50 trajectories).

### Trajectory Adaptation

Given a polynomial spline with constraint matrix A (encoding waypoint passage, continuity of velocity/acceleration/jerk), the constraint Aσ = b is preserved by any modification ϕ in Null(A). TACO parameterizes modifications in the null-space basis V:

```
τ̄(t) = α · S · [V·ϕ_v + σ]
```

where α is a speed-scaling factor and S is a smoothness matrix. Gradient updates flow through the MLP via autodifferentiation:

```
ϕ_v ← ϕ_v - η · (∂ĉ/∂ϕ_v)
```

This deforms the reference path to reduce predicted tracking error without violating waypoint constraints or destroying smoothness — a clean mechanism to slightly relax overly aggressive curvature or speed profile while keeping gates on the planned path.

---

## Results

### Simulation Benchmarks (Table II — Average Tracking Error in meters)

| Method | MinSnap | MinSnap-Hard | MinSnap-Varying | ZigZag | Lissajous (OOD) |
|---|---|---|---|---|---|
| Nominal (hand-tuned) | 0.44 (5 crashes) | 0.80 (11 crashes) | 0.62 (6 crashes) | 0.47 (1 crash) | 0.17 |
| Oracle Static (BO, 45 min) | 0.20 | 0.40 (1 crash) | 0.25 | 0.33 | 0.12 |
| **TACO Full** | **0.18** | **0.30 (1 crash)** | **0.20** | **0.28** | **0.11** |
| Oracle Adaptive (BO, online) | 0.10 | 0.24 | 0.13 | 0.26 | 0.08 |
| MPC (data-driven, 1s horizon) | 0.19 | 0.21 (3 crashes) | 0.20 | 0.13 | 0.14 |

TACO Full matches or beats the static oracle (which had 45 minutes of Bayesian Optimization per trajectory). Notably, TACO generalizes zero-shot to Lissajous curves (not seen in training) at 0.11 m, close to Oracle Adaptive at 0.08 m.

### Trajectory Adaptation Ablation (MinSnap-Hard)
- Static trajectory + fixed gains: 0.31 m average keypoint error
- Adapted trajectory + fixed gains: 0.21 m (−32%)
- Adapted trajectory + TACO gain optimization: **0.15 m** (−52%)

### Real-Robot Transfer (CrazyFlie 2.0)
Average tracking error reduced by ~5 cm vs. hand-tuned parameters on physical hardware. Sim-to-real transfer was achieved without domain randomization. Z-axis tracking improved most (vertical acceleration is the hardest axis without trajectory-aware gain boosting).

### Computational Cost
- TACO iteration: ~20 ms on Apple M3 (suitable for 2 Hz replanning in a 50–120 Hz control loop)
- Parallelized simulator: ~17,000 FPS for batch of 10,000 vehicles on CPU (25× speedup vs. serial)
- Dataset generation: 8 million datapoints in ~7 hours

---

## Relevance to Our System

Our system is a near-direct match for TACO's target architecture: a geometric SE(3) controller tracking min-snap polynomial trajectories, with fixed gains kp=6 (xy), kd=4 (xy), kp_z=8, kd_z=5, plus feedforward_accel=0.4. The bottleneck identified in iteration 10 is gates 7–12 (helix turns), where avg error is 0.64 m vs. 0.31 m on straight gates 1–6.

**Why TACO is directly relevant:**

1. **Trajectory-dependent gain mismatch is exactly our problem.** The 0.64 m error on helix gates vs. 0.31 m on straight gates is the hallmark of static gains that are tuned for one regime and fail in another. TACO was designed to solve this precise failure mode.

2. **Our controller structure matches TACO exactly.** The thrust/moment equations in TACO (equations above) are identical to our `GeometricTracker`. The gain vector [kp^x, kp^y, kp^z, kv^x, kv^y, kv^z, kR, kΩ] maps directly to our `TrackerConfig` fields.

3. **The trajectory adaptation idea is applicable to our min-snap optimizer.** Our `planning/trajectory_optimizer.py` generates polynomial splines. The null-space trajectory deformation could slightly relax curvature on the helix segment to reduce kinematic infeasibility without moving the gate waypoints.

4. **Our kinematic simulator is similar to RotorPy.** The sim applies `accel = accel_des - drag*vel`, clamped to max_accel=15. This is a close analog to the RotorPy dynamics TACO uses, so a simplified version of TACO's training procedure would be tractable here.

5. **2 Hz adaptation rate is comfortable.** Our control loop runs at 50–120 Hz. Running TACO every 0.5 s (20 ms compute) is well within budget.

**Specific module impact:**
- `control/mpc_tracker.py` — primary target: add per-axis gains and a gain scheduler
- `planning/trajectory_optimizer.py` — secondary target: trajectory adaptation via null-space perturbation on the helix segments
- `race_pipeline.py` — needs a hook to pass trajectory lookahead to the gain scheduler

---

## Actionable Takeaways

1. **Implement per-axis position gains.** Split kp_xy → (kp_x, kp_y) and kd_xy → (kd_x, kd_y). On helix turns, the x/y plane is the limiting direction; boosting kp on the curved axis while keeping kd moderate may reduce turn tracking error without causing oscillation on straights.

2. **Build a simple lookup-table gain scheduler.** A full learned model is complex to train, but a simpler version — a dictionary mapping trajectory "phase" (curvature, current speed, lookahead curvature) to a set of pre-optimized gains — captures the main benefit. Pre-compute optimal gains for "helix" and "straight" phases offline using grid search over the sim.

3. **Boost kp_z and kd_z during climbing/descending helix segments.** The Z-axis is the hardest axis in turns that combine lateral and vertical motion. TACO's results show Z-tracking improves most from trajectory-aware gain tuning. Specifically: at the helix entry (gate 7), increase kp_z from 8 to ~12 and kd_z from 5 to ~7.

4. **Try null-space trajectory deformation on the helix segment.** Before the race starts, apply a small null-space perturbation to the polynomial spline over gates 7–12 to reduce peak curvature. Constraint: gate waypoint positions are fixed; only the inter-waypoint shape is modified. This is a one-time offline optimization step requiring no learned model.

5. **Increase lookahead for feedforward on turns.** Currently feedforward_accel=0.4 uses the instantaneous desired acceleration. TACO's trajectory-lookahead input (100-step, 1-second window) suggests that using a slightly earlier point on the trajectory for feedforward (e.g., 0.1–0.2 s ahead) could phase-lead the feedforward and help turn anticipation.

6. **Separate kR and kΩ from kp/kd and tune them per phase.** TACO optimizes attitude gains jointly with position gains. On tight turns, higher kR (attitude proportional) is needed to bank quickly; on straights, high kR causes roll/pitch oscillation. Current kR=8.0 is a compromise. Reducing kR to ~5 on straight segments and boosting to ~12 on helix turns would match TACO's adaptive behavior.

7. **If training data is available: train a 2-layer MLP gain predictor.** Given the parallelized sim (which we can implement via vectorized numpy), 8 million datapoints at 17k FPS would take under an hour. The MLP (hidden [512, 512, 256, 256]) predicts tracking cost for (gains, state, trajectory_lookahead) and can be used with TACO's sampling optimizer at 2 Hz.

---

## Limitations & Caveats

1. **Max speed is ~3.5 m/s — far below racing speeds.** TACO's training distribution samples velocities only up to 3.0 m/s, and experiments reach ~3.5 m/s maximum. Competitive drone racing is 10–25 m/s. It is unclear whether the learned model generalizes to these regimes. The gain optimization landscape may be qualitatively different at racing speeds (aerodynamic effects, propeller wash, structural flex).

2. **CrazyFlie dynamics, not racing drone dynamics.** All physical experiments use a CrazyFlie 2.0, a slow, light platform (mass ~27g, propeller diameter ~65mm). A racing quad (e.g., 5-inch freestyle, ~250g, max_accel ~15g) has very different inertia tensor J, motor time constants, and drag coefficients. The trained MLP would need retraining for racing-scale dynamics.

3. **2 Hz adaptation is slow for rapid gate sequences.** Our race has gates that may be only 0.5–1.5 seconds apart. A 0.5 s replanning cycle means gains are adapted at most once or twice per gate passage. This is adequate if transitions are detected early but insufficient if the controller needs to change behavior mid-passage.

4. **Trajectory adaptation preserves waypoint timing, not just position.** The null-space formulation fixes both the waypoint positions and the time at which they are reached. For racing, some degree of time-flexibility (arriving at a gate slightly early or late) might be more beneficial than shape-only deformation.

5. **No handling of state estimation noise.** TACO conditions on ground-truth velocity and quaternion (from Vicon/motion capture in real experiments). Our system uses an EKF with non-trivial uncertainty (~0.1 m). Noisy state input to the MLP could degrade gain predictions; the paper does not study this.

6. **Training requires a dynamics model that matches the test environment.** TACO achieves sim-to-real transfer on CrazyFlie by using RotorPy (a high-fidelity CrazyFlie sim). Our kinematic sim (`accel = accel_des - drag*vel`, clamped) is a simplified model. Gains trained in this sim may not transfer to a higher-fidelity sim or real drone, but they are likely internally consistent — so the approach is still useful for self-improvement within our benchmark loop.

7. **Sampling-based optimizer can miss sharp optima.** The TACO sampling procedure (uniform random + perturbations) is not gradient-based in the gain space. For our 8-dimensional gain space, a CEM or gradient-through-model approach could find tighter optima. The paper notes Oracle Adaptive (BO) still outperforms TACO, indicating headroom.

---

## Key Parameters / Constants

These are directly usable values from the paper:

| Parameter | Value | Notes |
|---|---|---|
| Optimization horizon H | 100 steps | 1 second lookahead |
| Trajectory discretization Δt | 0.05 s | 20 Hz lookahead sampling |
| Replanning period T | 0.5 s (2 Hz) | TACO gain update rate |
| TACO compute time | ~20 ms | Apple M3; trivially fits in 0.5s budget |
| MLP hidden layers | [512, 512, 256, 256] | ReLU activations |
| Gain vector dimension | 8 | [kp^x, kp^y, kp^z, kv^x, kv^y, kv^z, kR, kΩ] |
| Velocity training range | [0.5, 3.0] m/s | Note: below racing speeds |
| Simulation timestep δt | 0.01 s (100 Hz) | For training rollouts |
| Dataset size | 8,000,000 samples | ~7 hours to generate |
| Parallelized sim throughput | ~17,000 FPS | Batch of 10,000 on CPU |
| Tracking error improvement (hard MinSnap) | 0.80 → 0.30 m | TACO vs. nominal |
| Trajectory adaptation improvement | 32% | Fixed gains, adapted path |
| Combined improvement | 52% | TACO gains + adapted path |
| Real-robot improvement | ~5 cm reduction | CrazyFlie 2.0 |

**Suggested gain ranges for our system (extrapolated from paper, to be verified):**
- kp: [2, 15] per axis
- kd (velocity): [1, 10] per axis
- kR (attitude): [3, 20]
- kΩ (angular velocity): [0.5, 8]

The paper's CrazyFlie gains will be scaled differently than our sim's gains (different mass/inertia), but these ranges give reasonable bounds for a grid-search or sampling-based optimizer over our kinematic sim.
