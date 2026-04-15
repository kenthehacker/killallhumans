# A Data-Driven Aggressive Autonomous Racing Framework with Velocity Prediction
- **URL**: https://arxiv.org/abs/2410.11570
- **Authors**: Zhouheng Li, Bei Zhou, Cheng Hu, Lei Xie, Hongye Su (Zhejiang University)
- **Year**: 2024 (submitted October 2024, revised March 2025)
- **Venue**: arXiv preprint

---

## Key Contribution

This paper introduces VPMPCC (Velocity Prediction based Model Predictive Contouring Control), a data-driven framework for aggressive autonomous racing on tracks with sharp corners. The central problem it solves is that standard MPCC underperforms on high-curvature sections because it cannot look ahead to anticipate the need to decelerate — it optimizes velocity greedily within a short prediction horizon, causing it to enter corners too fast and lose time recovering. VPMPCC addresses this by encoding a Reference Velocity Profile (RVP) derived from the minimum-curvature racing line into the MPCC cost function, giving the planner a velocity target that reflects global track geometry, not just local curvature.

The second major contribution is a Bayesian Optimization (BO) framework that automatically learns the nine VPMPCC parameters using a novel Objective Function adapted to Racing (OFR). The OFR is designed to avoid two failure modes that plague naive BO for racing: a pure lap-time objective causes BO to discover aggressive, crashing trajectories; a pure safety objective converges to slow, conservative behavior. OFR combines both concerns via a multiplicative penalty that activates only when safety thresholds are violated. This design enables convergence in 42.86% fewer iterations than alternative BO formulations.

---

## Technical Approach

### MPCC Background

Model Predictive Contouring Control frames the racing problem as minimizing contouring error (lateral deviation from the racing line) and lag error (progress slowdown) while maximizing projected velocity along the arc-length parameter `s`. The standard MPCC cost is:

```
J = sum over horizon [ q_c * e_c^2 + q_l * e_l^2 - gamma * v_p ]
```

where `e_c` is the contouring error, `e_l` is the lag error, `v_p` is the projected velocity (arc-length progress rate), and `gamma` is a weighting term. This works well on smooth tracks but fails on sharp corners because the optimizer has no mechanism to pre-slow for an upcoming turn.

### VPMPCC: Velocity Prediction Extension

VPMPCC adds a velocity prediction cost term to the MPCC formulation:

```
f_VP = (q_v / v_delta_max) * sum_{k=0}^{N_p} (v_k - v_RVP(s_k))^2
```

This term penalizes deviation of the planned velocity `v_k` at each prediction step from the Reference Velocity Profile value `v_RVP(s_k)` at the corresponding arc-length position. The key insight is that `v_RVP(s)` is computed from global track geometry — it encodes where the vehicle *needs* to be slow — while the MPCC horizon operates locally. By embedding the global velocity target into the local cost, the planner anticipates corners and begins braking early.

The full state space is `[x, y, phi, s]^T` (position, heading, arc-length) and the control space is `[v, delta, v_p]^T` (longitudinal velocity, steering angle, projected velocity). The decoupling of `v` (actual longitudinal velocity) from `v_p` (arc-length rate) is what allows the velocity prediction term to be formulated independently.

### Reference Velocity Profile (RVP) Construction

The RVP is derived from the minimum-curvature path through the track. The velocity at each arc-length position is set proportional to the local turning radius: `v_RVP(s) ~ sqrt(a_lat_max * R(s))` where `R(s)` is the local radius of curvature and `a_lat_max` is the lateral acceleration limit. In practice, the exact shape of this mapping is a learnable function that BO optimizes — the paper parameterizes it and searches over the parameter space.

### Bayesian Optimization of Nine Parameters

The nine parameters optimized by BO are:

```
theta = [N_p, q_v, gamma, e_con, e_lag, q_delta_v, q_delta_delta, q_delta_vp, xi]
```

Where:
- `N_p` — prediction horizon length
- `q_v` — velocity prediction cost weight
- `gamma` — progress maximization weight
- `e_con`, `e_lag` — contouring and lag error weights
- `q_delta_v`, `q_delta_delta`, `q_delta_vp` — control smoothness weights
- `xi` — track boundary margin scaling (range 0.01–0.4)

The Gaussian Process surrogate model uses a Matern kernel with `nu = 5/2` and Expected Improvement (EI) as the acquisition function — a standard but well-validated choice for continuous hyperparameter optimization.

### Objective Function Adapted to Racing (OFR)

The OFR has three components:

1. **Lap Time Term**: `L(z) = T_lap + lambda_1 * [t - t_lb]^-`
   Minimizes lap time, with `t_lb = 17.6s` as a lower bound penalty to prevent physically impossible configurations.

2. **Trajectory Length Penalty**: `I(z) = lambda_2 * tanh(lambda_3 * (trajectory_length - D_ref))`
   Penalizes trajectories much shorter than the reference length `D_ref = 62.8m`, catching cases where the vehicle cuts corners excessively.

3. **Distance Barrier**: `B(z) = lambda_4 * log([max(|d|) / d_tol, 1]^+)^{-1}`
   Penalizes lateral deviations beyond tolerance `d_tol = 0.5m`, acting as a safety barrier function.

4. **Failure Penalty**: `J_fail` applied when the high-fidelity dynamics model predicts the planned trajectory is physically unrealizable on real hardware.

The multiplicative structure (lap time × safety penalty) means the safety terms only dominate when violated, allowing BO to pursue fast but realizable trajectories otherwise.

### Corner-Specific Behavior (Empirical Findings)

The learned velocity profiles show three consistent patterns on sharp corners:
- **Early deceleration**: braking begins 1–2 vehicle lengths before the apex, not at it
- **Apex minimum**: the velocity minimum is achieved at or slightly before the geometric apex
- **Early acceleration**: power is applied at or just past the apex, not at corner exit
- **Compound turn behavior**: for chicanes (two consecutive opposite-direction turns), velocity remains suppressed across the entire compound maneuver, not just at each individual apex

These patterns emerge from the optimization without being explicitly programmed — they are the controller's learned response to the combination of velocity prediction cost and lap-time objective.

### Sim-to-Real Transfer

A high-fidelity vehicle dynamics model, identified offline from real hardware data, is used during BO training to filter trajectories. Any planned trajectory that the dynamics model predicts would fail on hardware receives penalty `J_fail`. This closed-loop filtering during training enables direct sim-to-real transfer without retraining, a significant practical advantage.

---

## Results

### Simulation Results

| Method | Lap Time | Training Iterations |
|--------|----------|---------------------|
| Standard MPCC | 30.44s (baseline) | N/A |
| VPMPCC with OFR (this paper) | 16.12s | 76 |
| VPMPCC with alternative BO objective | 17.73s | 133 |

Relative improvements:
- Lap time reduced by **47.1%** vs standard MPCC
- Training iterations reduced by **42.86%** vs alternative BO formulation
- Optimal configuration found 9.07% faster in lap time than alternative BO

### Real Vehicle Results (F1TENTH, 25 laps, 1:10 scale)

| Metric | Value |
|--------|-------|
| Mean projected velocity | 3.68 m/s |
| Velocity utilization | 93.18% of vehicle limits |
| Lap time | 17.05s |
| Max velocity | 6.39 m/s |
| Mean computation time | 7.04 ms |
| Computation time std | 4.64 ms |
| Trajectory length | ~62.8m (D_ref) |

The 93.18% velocity utilization (vs 70.4% for standard MPCC — a 22.78 percentage point improvement) demonstrates that VPMPCC operates very close to the physical limit while maintaining safe track boundaries across 25 laps of prolonged racing on a sharp-cornered track.

### Comparison: VPMPCC vs Standard MPCC

| Metric | Standard MPCC | VPMPCC | Improvement |
|--------|--------------|--------|-------------|
| Mean projected velocity | 2.16 m/s | 3.68 m/s | +70.4% |
| Lap time (sim) | 30.44s | 16.12s | -47.1% |
| Velocity utilization | ~70% | 93.18% | +23pp |

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories with TOPP (Time-Optimal Path Parametrization) retiming, followed by a geometric SE(3) tracker (MPC). The worst-performing gates are gate-7 (0.284m avg error) and gate-8 (0.235m avg error), which lie in the helical section where the trajectory curves continuously in 3D — a structural analog to the paper's sharp-corner problem.

### Connection: Velocity Pre-Planning at High-Curvature Sections

The fundamental insight of VPMPCC directly applies: our TOPP retimer assigns time to trajectory segments locally, based on curvature at each point. It does not look ahead to see that a sequence of curved segments (the helix) requires sustained slow speed through the entire section, not just at individual curvature peaks. The result is that our tracker enters the helical section at a velocity it cannot sustain through consecutive gates, producing the tracking errors we observe at gate-7 and gate-8.

The VPMPCC paper's finding that "for chicanes (consecutive turns), velocity stays low between turns" is directly analogous. For our helix, the retimer should treat the entire helical section as a compound maneuver and assign a sustained time inflation across gates 5–8, not just at the sharpest individual curvature points.

### Connection: Early Deceleration

Our trajectory approach begins slowing only at the gate entry point. VPMPCC's learned profiles begin decelerating 1–2 vehicle lengths earlier. For our system, this means the segment approaching gate-7 (the gate-6 exit to gate-7 approach) should be inflated, not just gate-7's through segment. This gives the tracker time to shed lateral velocity from the previous gate before entering the next helix turn.

### Connection: Compound Maneuver Detection

VPMPCC treats chicanes as single compound maneuvers. In our `racing_line.py`, we could detect consecutive gates where the lateral offset reverses direction (same as opposite-sign turns in ground racing) within a distance threshold, and apply a compound inflation factor. For the helix, all consecutive gates qualify since the racing line spirals continuously.

### Connection: BO for Parameter Tuning

VPMPCC uses BO over 9 parameters with ~76 iterations. Our inflation parameters for the helical section (which gates to inflate, by how much) are a similar parameter space. Our `scripts/benchmark.py --mode sim --duration 20` completes in a few seconds, making 100+ BO iterations feasible in under 10 minutes. This is actionable: run BO over `helix_inflation_start_gate`, `helix_inflation_end_gate`, and `helix_inflation_factor` with our benchmark as the black-box objective.

---

## Actionable Takeaways

1. **Inflate the approach segment, not just the gate segment**: For gate-7 and gate-8, add time inflation to the segments immediately preceding these gates (gate-6 exit → gate-7 approach, gate-7 exit → gate-8 approach). The VPMPCC evidence suggests early deceleration is more effective than deceleration at the apex.

2. **Treat the entire helix as a compound maneuver**: Rather than inflating individual gate-through segments, apply a sustained inflation factor across the entire helical section (gates 5–8 or wherever the helix begins). Sustained low velocity across compound curves is the empirically validated pattern from VPMPCC.

3. **Use Bayesian Optimization for inflation parameter search**: Define a 3–5 parameter search space (helix start gate, end gate, inflation factor, possibly separate entry/through/exit inflation) and use BO with `scripts/benchmark.py` as the objective. With ~7.04ms compute times in VPMPCC and our sim similarly fast, 100 iterations is tractable.

4. **Compute a Reference Velocity Profile from trajectory curvature**: Build a 1D curvature profile of our 3D trajectory (curvature at each waypoint) and use `v_ref(s) = sqrt(a_max * R(s))` to define a target velocity profile. Add a soft cost in the retimer that penalizes deviation from this global profile, analogous to the VPMPCC velocity prediction term `f_VP`.

5. **Gate-specific compound detection heuristic**: In `racing_line.py`, detect gates where the racing line offset reverses direction within 15m (the analog of chicane detection). For such gate pairs, increase the effective curvature used for time allocation by 1.3–1.5x when computing the time inflation.

6. **OFR-style objective for our BO**: Do not optimize pure race time — BO will find configurations that crash. Combine race time with a tracking error penalty that activates when per-gate error exceeds 0.3m, analogous to OFR's distance barrier. This prevents BO from discovering aggressive but unstable configurations.

---

## Limitations & Caveats

1. **Ground vehicle, not quadrotor**: VPMPCC uses a kinematic bicycle model on a 2D surface. Our system operates in 3D with different dynamics (gravity, thrust vectoring, no lateral friction constraint). The velocity profiling principle transfers but specific parameter values (like the 1–2 vehicle length deceleration anticipation distance) do not.

2. **Continuous track vs. discrete gates**: VPMPCC optimizes along a continuous arc-length parameter. Our trajectory has discrete gate constraints that must be exactly satisfied. The velocity profile must be consistent with gate-pass timing requirements, adding a constraint absent in VPMPCC.

3. **Online vs. offline planning**: VPMPCC does local trajectory planning online at ~7ms per cycle. Our system uses pre-computed min-snap trajectories. The BO insight (data-driven parameter tuning) applies to our offline planner, but the online replanning capability of VPMPCC does not directly transfer.

4. **Track-specific learned profiles**: The 93.18% velocity utilization is for a specific track and vehicle. Our helix geometry is quite different. The qualitative principles (early braking, compound turn suppression) transfer; the specific numeric ratios do not.

5. **F1TENTH scale effects**: At 1:10 scale with `v_max = 15 m/s` (scaling to 150 m/s full scale), the aerodynamic and inertial effects differ substantially from our drone. The kinematic assumption underlying VPMPCC's planning model is a stronger approximation at drone velocities.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Prediction horizon (`N_p`) | Optimized by BO | One of 9 BO parameters |
| Prediction interval (`T_s`) | 0.1 s | Fixed, not optimized |
| Max velocity (`v_max`) | 15 m/s | Vehicle limit |
| Distance tolerance (`d_tol`) | 0.5 m | Safety barrier threshold in OFR |
| Reference track length (`D_ref`) | 62.8 m | Custom track perimeter |
| Lap time lower bound (`t_lb`) | 17.6 s | OFR penalty activation threshold |
| Wheelbase (`L`) | 0.32 m | F1TENTH 1:10 scale |
| BO iterations to converge (OFR) | 76 | 42.86% fewer than alternative |
| BO iterations (alternative) | 133 | Comparison baseline |
| Mean computation time | 7.04 ms | Real-time capable |
| Velocity utilization achieved | 93.18% | Of physical hardware limits |
| MPCC velocity utilization | ~70.4% | Baseline comparison |
| Matern kernel smoothness | nu = 5/2 | GP surrogate model |
| BO acquisition function | Expected Improvement (EI) | Standard choice |
| xi (track boundary margin) | 0.01–0.4 | Searched range |
| Deceleration anticipation | 1–2 vehicle lengths | Before corner apex |
| Lap time improvement vs MPCC | 47.1% | Simulation results |
| Lap time improvement vs alt BO | 9.07% | Simulation results |
