# AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization

**URL:** https://arxiv.org/abs/2509.17340
**Authors:** Xin Chen, Rui Huang, Longbin Tang, Lin Zhao
**Year:** 2025 (submitted September 22, 2025; revised March 21, 2026)
**Venue:** ICRA 2026

---

## 1. Key Contribution

AERO-MPPI presents a GPU-accelerated trajectory optimization framework for agile, mapless drone navigation in cluttered 3D environments. The central insight is that a single MPPI optimizer is highly susceptible to local minima in obstacle-dense settings — the sampled perturbations tend to cluster around the current nominal trajectory, so if that trajectory is locally trapped, the optimizer cannot escape. AERO-MPPI overcomes this by running **M = 15 parallel MPPI instances**, each initialized ("anchored") to a structurally different trajectory class derived from LiDAR point cloud geometry. Each optimizer thus explores a different basin of attraction, and the best solution across the ensemble is selected for execution. This single idea — diversify initialization rather than inflate sample count — is the paper's core novelty. Secondary contributions include a two-resolution spherical partition for anchor extraction and a two-stage cost evaluation scheme that separates trajectory shaping (within each optimizer) from final trajectory selection (across optimizers).

---

## 2. Technical Approach

### 2.1 Anchor Extraction from LiDAR

The system processes accumulated LiDAR point clouds (10 consecutive frames, ~200,000 points) within a look-ahead sphere of radius r_max = 10 m. The sphere around the drone is partitioned at two resolutions:

- **High-resolution:** 3° angular resolution → 120×60 spherical cone cells (I,J)
- **Coarse:** 18° resolution → 20×10 cells, formed by pooling 6×6 high-res blocks

For each coarse cell, the system identifies the direction with the maximum safe clearance and places a 3D anchor point at `p_ref = p_0 + ℓ · d_IJ` where ℓ = 5.0 m is a fixed look-ahead distance. Each anchor becomes the waypoint for one MPPI optimizer's guiding trajectory.

### 2.2 Polynomial Guiding Trajectories

Each anchor is connected to the drone's current state and a goal direction via **three fifth-order polynomials** (one per spatial axis): `f_μ(t) = Σ(i=0..5) a_i,μ t^i`. Boundary conditions (current position, velocity, acceleration; anchor position; goal direction) uniquely determine the six coefficients per axis. These guiding polynomials define the tracking reference fed to each independent MPPI optimizer.

### 2.3 Ensemble MPPI

The framework runs M = 15 parallel MPPI instances simultaneously (M_h = 5 horizontal × M_v = 3 vertical orientations, plus additional). Each optimizer runs the standard MPPI update:

```
u_nom ← u_nom + Σ_k ε_k δu_k
```

with K = 128 trajectory samples, N = 25 prediction horizon steps, Δt = 50 ms, and temperature λ = 0.1.

The cost function used **within** each optimizer is a composite stage cost:

```
S^(1) = J_track + J_vnorm + J_ctrl + S^(2)
S^(2) = J_goal + J_col
```

where:
- `J_track = Σ Q_track ||p_t - p_traj(t)||` — tracks the anchor's guiding polynomial
- `J_vnorm = Σ Q_vnorm ||v_t||^2` — penalizes speed buildup in constrained spaces
- `J_ctrl = Σ Q_c ||u_t||^2 + Q_cΔ ||Δu_t||^2` — control smoothness
- `J_goal` — terminal position/velocity/attitude penalty
- `J_col` — exponential collision penalty with hard threshold at d_obs_min = 0.4 m

### 2.4 Final Trajectory Selection

After all M optimizers converge, each produces an optimized control sequence. These sequences are **re-rolled out via RK4 integration** of the drone dynamics, then re-evaluated under a simplified cost S^(2) (goal + collision only, no tracking term). The control sequence achieving the lowest S^(2) cost is selected and executed. This two-stage evaluation is important: it decouples diversity (each optimizer optimizes a different guiding polynomial) from selection (all candidates judged by a common, unbiased objective).

### 2.5 GPU Implementation

The entire framework is implemented using NVIDIA Warp GPU kernels. On a GeForce RTX 4080 SUPER, it runs at 500 Hz with less than 600 MB GPU memory (N=25, K=256). On an NVIDIA Jetson Orin NX 16GB (embedded hardware), it achieves 10 Hz planning, sufficient for real-world deployment. Onboard simulation runs at 50 Hz.

---

## 3. Results (Quantitative)

Evaluated in three simulated obstacle environments: Forest (random cylinders), Verticals (vertical pole arrays), and Inclines (inclined barriers). All baselines fail above ~4 m/s; AERO-MPPI sustains flight above 7 m/s.

**Average Velocity (m/s):**

| Scenario   | AERO-MPPI       | Best Baseline (method)         |
|------------|-----------------|-------------------------------|
| Forest     | 5.60 ± 0.11     | 3.27 ± 0.24 (Topo-MPPI)      |
| Verticals  | 3.52 ± 0.21     | 1.42 ± 0.08 (Topo-MPPI)      |
| Inclines   | 5.08 ± 0.19     | 1.16 ± 0.10 (Fast-Planner)   |

**Trajectory Smoothness (higher = smoother):**

| Scenario   | AERO-MPPI       | Best Baseline                 |
|------------|-----------------|-------------------------------|
| Forest     | 7.74 ± 0.22     | 4.52 (Topo-MPPI)             |
| Verticals  | 6.33 ± 0.57     | 4.89 (Topo-MPPI)             |
| Inclines   | 8.09 ± 0.66     | 2.57 (Fast-Planner)          |

**Success Rates:** AERO-MPPI achieves >80% success across all scenarios at velocities up to 7 m/s. Baselines drop below 50% at comparable speeds.

**Real-world flight** confirmed safe, agile navigation in complex indoor environments, executing at 10 Hz on Jetson Orin NX 16GB without prebuilt maps.

---

## 4. Relevance to Our System

Our system uses L-BFGS to optimize a min-snap polynomial trajectory (via `planning/trajectory_optimizer.py` and `planning/racing_line.py`). The optimizer takes gate waypoints as hard constraints and solves for polynomial coefficients minimizing snap, with a racing line pass that does lateral offset adjustments and curvature-aware speed profiling.

The local minima problem AERO-MPPI addresses is directly analogous to what we face: L-BFGS is a gradient-based method that can only descend from its initialization. If the initial trajectory guess is in a poor basin (e.g., a trajectory that cuts too sharply before a gate cluster, or that enters a tight gate at a suboptimal angle), L-BFGS will find the local minimum in that basin and stop. There is no mechanism in our current system to explore fundamentally different trajectory topologies.

AERO-MPPI's key insight — **run multiple optimizers initialized to structurally different trajectories, then select the winner** — is directly transferable. In our context, this means:

- Generate multiple candidate racing lines with different lateral offset profiles or speed scaling
- Optimize each independently with L-BFGS
- Evaluate all optimized trajectories under a common cost (total time or integrated tracking error)
- Select and execute the best

Unlike AERO-MPPI (which extracts anchors from LiDAR), our anchors can be derived analytically: different polynomial splines connecting the same gate sequence with different intermediate waypoints, different speed profiles (aggressive vs. conservative through tight turns), or randomized initial coefficient perturbations. Our problem is simpler (known gate positions, no obstacle avoidance) but the optimization landscape may still be multi-modal due to the nonlinear coupling between trajectory timing, curvature, and speed feasibility.

---

## 5. Actionable Takeaways

1. **Implement ensemble trajectory initialization in `trajectory_optimizer.py`.** Before L-BFGS, generate M = 5–15 candidate initial trajectories with different lateral offset profiles (e.g., sample `lateral_offset` in `racing_line.py` at several values: −0.3m, −0.1m, 0, +0.1m, +0.3m per gate). Run L-BFGS from each initialization independently.

2. **Add a common selection cost.** After all M optimizations complete, re-evaluate each trajectory under total estimated race time (or integrated squared tracking error from simulation) and select the minimum. Do not use the L-BFGS loss as the selection criterion — the paper's two-stage evaluation showed this matters.

3. **Perturb segment time allocation, not just spatial waypoints.** Our trajectory's local minima may stem from poor time allocation between gates rather than spatial shape. Add initializations with different inter-gate time budgets (e.g., ±20% on per-segment durations).

4. **Use polynomial guides as warm starts.** AERO-MPPI initializes each MPPI optimizer with a fifth-order polynomial guide. Similarly, pre-fit a smooth polynomial spline through gate centers (with different intermediate waypoints per ensemble member) and use its coefficients as the starting point for L-BFGS, rather than starting from the previous iteration's solution.

5. **Exploit GPU parallelism if available.** The 15-way parallelism in AERO-MPPI costs almost nothing on GPU. Our L-BFGS runs on CPU and is likely cheap enough that 5–10 instances fit within the benchmark's timing budget without significant overhead — profile before assuming it's too slow.

6. **Apply ensemble diversity to speed profiles in `racing_line.py`.** The `curvature_aware_speed_profiling` function could produce multiple speed profiles (e.g., different aggressiveness multipliers: 0.8×, 1.0×, 1.2×) and feed each as a separate trajectory candidate into the ensemble.

7. **Use RK4 re-evaluation for final selection.** AERO-MPPI's winning move is re-rolling out all M candidates under simplified cost. In our system, this maps to running each candidate trajectory through a lightweight forward simulation (or the existing `state_predictor.py`) to estimate actual tracking error, then selecting the candidate with minimum simulated error.

---

## 6. Limitations & Caveats

- **Obstacle-free racing context:** AERO-MPPI is designed for obstacle avoidance. Its anchor generation is driven by LiDAR obstacle geometry, which has no direct analog in our gate-racing setting. The ensemble principle transfers cleanly, but the anchor source must be redesigned (derived from gate geometry and racing line variants rather than free-space directions).

- **No time-optimality guarantee.** AERO-MPPI optimizes for collision-free agility, not minimum lap time. Our problem has a fundamentally different cost structure (minimize time, not maximize clearance). The ensemble principle works regardless, but our cost function must stay consistent with lap time.

- **Success rate not 100%.** At 7 m/s, AERO-MPPI achieves ~80% success — meaning ~20% of runs crash or fail. For racing (where one crash = DNF), this success rate may be insufficient if adopted directly. Our problem's deterministic gate positions make it easier to achieve near-100% reliability.

- **10 Hz on embedded hardware.** Real-world AERO-MPPI runs at only 10 Hz on Jetson Orin NX. Our benchmark requires >100 Hz control loop. If we adopt ensemble MPPI as a trajectory replanner (not a low-level controller), we must ensure it runs as an offline or infrequent replanning layer, not in the main control loop.

- **Parameter sensitivity.** The paper uses a fixed look-ahead ℓ = 5.0 m and fixed anchor count M = 15. Performance sensitivity to these values is not fully ablated. In our system, the optimal ensemble size and diversity strategy for the racing-line optimizer would need empirical tuning.

- **Implicit limitations unaddressed.** The paper does not discuss dynamic gates, wind disturbance, or perception noise effects on trajectory quality. These matter for our VQ1 competition setting.

---

## 7. Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 15 (M_h=5, M_v=3) | Number of parallel MPPI instances |
| K | 128 | MPPI trajectory samples per instance |
| N | 25 | Prediction horizon steps |
| Δt | 50 ms | Time step per horizon step |
| λ | 0.1 | MPPI temperature (exploration-exploitation) |
| ℓ | 5.0 m | Look-ahead distance for anchor placement |
| r_max | 10.0 m | LiDAR perception range |
| High-res partition | 3° (120×60 cells) | Fine angular resolution for obstacle finding |
| Coarse partition | 18° (20×10 cells) | Pooled resolution for anchor selection |
| d_obs_min | 0.4 m | Hard collision threshold distance |
| d_obs_max | 1.0 m | Soft collision onset distance |
| Q_track | 15.0 | Tracking cost weight |
| Q_p | 3.0 | Goal position cost weight |
| Q_v | 0.25 | Goal velocity cost weight |
| Q_q | 1.0 | Goal attitude cost weight |
| Q_vnorm | 0.15 | Velocity norm cost weight |
| Q_c | 0.5 | Control cost weight |
| Q_cΔ | 0.5 | Control smoothness cost weight |
| C_col | 10^6 | Collision penalty constant |
| a_col | 5.0 | Collision penalty slope |
| Drone mass | 1.0 kg | Simulated quadrotor mass |
| Thrust range | 0.3–16.35 N | Per-rotor thrust limits |
| Planning freq (sim) | 50 Hz | Optimization update rate in simulation |
| Planning freq (real) | 10 Hz | Optimization update rate on Jetson Orin NX |
| GPU freq | 500 Hz | Rate achievable on RTX 4080 SUPER |
| GPU memory | <600 MB | At N=25, K=256 on RTX 4080 SUPER |
| LiDAR frames | 10 | Frames accumulated per point cloud |
| Point cloud size | ~200,000 pts | Points per accumulated scan |
| Polynomial degree | 5th order | Per-axis guiding trajectory polynomials |
| RK4 re-evaluation | Applied to all M | Final selection uses simplified cost S^(2) |
