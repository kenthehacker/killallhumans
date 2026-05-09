# Optimizing Control-Friendly Trajectories with Self-Supervised Residual Learning

- **URL**: https://arxiv.org/abs/2601.02738
- **Authors**: Kexin Guo, Zihan Yang, Yuhang Liu, Jindou Jia, Xiang Yu
- **Year**: 2026

---

## Key Contribution

This paper addresses a fundamental mismatch in trajectory-following robotics: nominal trajectory planners assume a perfect model of system dynamics, but real hardware always exhibits residual dynamics (aerodynamic drag, motor nonlinearities, sensor latency, flexible structure) that analytical models cannot capture. When a controller tries to track a trajectory planned under the nominal model, these residuals cause persistent tracking error that neither the planner nor the controller fully accounts for.

The central contribution is a **self-supervised minimum-residual trajectory optimization** framework. Rather than treating the residual as pure disturbance to be rejected by the controller, the planner learns it (offline, from flight data alone, without labeled ground-truth residuals) and then actively optimizes trajectories that minimize exposure to those residuals. The result is a "control-friendly" trajectory that is physically easier for any controller to follow, without requiring that controller to change. In quadrotor agile flight experiments this reduced mean position tracking error from 0.113 m to 0.098 m under a baseline DFBC controller, and from 0.104 m to 0.084 m under a neural disturbance-observer controller — roughly 13–19% improvement — on the same hardware and controller, simply by changing the planned trajectory.

---

## Technical Approach

### Hybrid Dynamics Model

The full closed-loop dynamics are written as:

```
x_dot = phi(x, x_ref) + d(x, xi)
```

where `phi` is the nominal closed-loop dynamics (known analytically) and `d(x, xi)` is the residual term parameterized by a neural network with weights `xi`. The state vector is `x = [p, v, Theta, omega]^T` in R^12 (position, velocity, orientation, angular rate) and the control input is the four motor thrusts `u = [T1, T2, T3, T4]^T`.

### Self-Supervised Residual Learning

The network is trained purely from closed-loop flight trajectories — no labeled residual forces are needed. The learning objective minimizes prediction error over multi-step rollouts:

```
min_xi  sum_{k=1}^{N-1} l_k(x_k, x_{k,r}, xi) + l_N(x_N, x_{N,r})
s.t.    x_{k+1} = Phi_RK4(x_k, x_ref_k, xi)
```

where `x_{k,r}` is the real measured state at step k and `Phi_RK4` is a 4th-order Runge-Kutta rollout of the hybrid model. The loss at each step is:

```
l_k = (x_{k,r} - x_k(xi))^T L_k (x_{k,r} - x_k(xi))
```

with `L_k` a diagonal weighting matrix. Using multi-step RK4 rollouts (rather than single-step prediction) is crucial: it avoids the noise amplification problem inherent in finite-difference derivative estimation, and it maintains accuracy over longer horizons needed for trajectory optimization.

### Analytic Gradients via Adjoint / Hamiltonian Method

To train the network from multi-step rollouts without differentiating through noisy measured derivatives, the paper applies Hamiltonian-based optimal control theory (Pontryagin adjoint method). The gradient `d(loss)/d(xi)` is computed by integrating adjoint states backward in time (Eqs. 6–9 in the paper), yielding exact analytic gradients through the RK4 integration at negligible cost. This is the key technique that makes the self-supervised approach numerically stable — derivative labels are never needed, so label noise cannot contaminate the gradient.

This is directly analogous to how neural ODEs and differentiable physics engines compute gradients, applied here to a hybrid nominal + residual model.

### Minimum-Residual Trajectory Optimization

Once the residual model `d(x, xi)` is learned, trajectory optimization minimizes the residual squared along the trajectory:

```
min_{u, t_f}  integral_0^{t_f} [ d_xi(x)^T d_xi(x) + lambda_r * u^T u ] dt
s.t.          x_dot = f(x, u) + d_xi(x),   x in X,  u in U
```

The first term drives the trajectory toward regions of state space where residual forces are small. The second term (`lambda_r = 0.1`) is a standard control regularizer. The constraint enforces the full hybrid dynamics — so the optimizer implicitly accounts for residual effects on the trajectory rollout.

Discretization uses **direct multiple shooting** with variable step sizes `h_k` (bounded between `h_lb` and `h_ub`), giving `N*(n+p+1)` decision variables where `n=12` (state dimension) and `p=4` (control dimension). This allows the optimizer to place time nodes more densely in high-curvature or high-residual regions. Ipopt solves the resulting NLP.

### Network Architecture

The residual MLP is kept deliberately small to (a) be differentiable analytically, (b) generalize to unseen trajectories, (c) be fast enough for trajectory optimization:

- **Simulation**: 2 hidden layers × 32 units, input = `[vx, vy, vz, Theta_x, Theta_y, Theta_z]^T` (6 states), output = 3D aerodynamic acceleration residual
- **Real-world**: 3 hidden layers × 32 units, same input format, output = 3D velocity residuals

The input being only 6 dimensional (velocity + attitude) is a deliberate modeling choice: aerodynamic drag and attitude-dependent effects are the dominant residuals for agile quadrotor flight, and they depend primarily on velocity magnitude and orientation, not on absolute position.

### Smoothness After Correction

The minimum-residual objective does not explicitly constrain jerk, but two mechanisms maintain trajectory quality:

1. The control regularizer `lambda_r * u^T u` penalizes aggressive inputs that would increase aerodynamic loading, which indirectly penalizes jerk.
2. The direct multiple shooting formulation with variable step sizes allows the optimizer to choose time discretization that avoids under-resolved segments, preventing oscillatory solutions.

A notable finding (Table III in the paper): min-residual trajectories have *higher* maximum jerk (46.7 vs 22.7 m/s³ for min-snap) despite lower tracking error. This means the optimizer is trading jerk for reduced residual exposure — the trajectory asks more of the actuators in terms of sharp inputs but exposes the drone to less drag at those moments. This is important: **trajectory friendliness to the controller is not the same as smoothness**. A trajectory with higher jerk but lower drag force can be easier to track.

---

## Results

### Residual Learning

| Metric | Nominal model (simulation) | Hybrid model (simulation) |
|--------|---------------------------|--------------------------|
| Acceleration RMSE (train) | 0.684 m/s² | 0.260 m/s² |
| Acceleration RMSE (test) | 0.603 m/s² | 0.173 m/s² |

The hybrid model reduces test-set acceleration prediction error by 71% over the nominal model. For real-world flight data (16 training trajectories, ~15s each):

| Dataset | Position RMSE | Velocity RMSE |
|---------|--------------|--------------|
| Training | 0.064 m | 0.137 m/s |
| Test (same conditions) | 0.067 m | 0.129 m/s |
| Out-of-distribution | 0.093 m | 0.199 m/s |

Generalization to out-of-distribution maneuvers holds reasonably well (0.067→0.093 m, +39%), suggesting the MLP has learned a physically meaningful residual rather than overfitting to specific trajectories.

### Trajectory Tracking

Real-world 7-waypoint pretzel flight:

| Controller | Min-Snap mean pos error | Min-Residual mean pos error | Improvement |
|------------|------------------------|----------------------------|-------------|
| Baseline DFBC | 0.113 m | 0.098 m | -13.3% |
| Neural disturbance observer | 0.104 m | 0.084 m | -19.2% |

Notably, the improvement is larger when a stronger controller (neural DO) is used — suggesting the min-residual trajectory is revealing its benefit more fully when the controller can actually execute it precisely. The aerodynamic drag along the trajectory is also reduced from 1.845 N (min-snap) to 0.817 N (min-residual) in simulation — a 55.7% reduction in drag loading, which directly translates to less unmodeled force that the controller must reject.

### Computational Cost

Minimum-residual optimization is reported as having comparable wall-clock time to minimum-snap (Fig. 7), running on an Intel i7-10750H with Ipopt. Training converges in ~100 epochs for simulation, ~40 epochs for real-world data with ADAM.

---

## Relevance to Our System

Our current ILC implementation (`planning/trajectory_optimizer.py`, `compute_ilc_offset_table`) computes **position offsets** from repeated sim trials: at each time step it measures the cross-track error, filters it, and adds a correction to the reference position. We also compute velocity offsets as smooth derivatives of the position offsets (iteration 41). This is correct ILC but is fundamentally reactive — it reduces error by shifting the reference after the fact.

This paper offers a complementary and more proactive angle: instead of correcting for tracking error after seeing it, optimize the trajectory from the start to minimize the physics that *cause* tracking error. In our context, the dominant residuals are likely:

1. **Aerodynamic drag** at high speed (the paper addresses this directly)
2. **Motor response lag** on aggressive pitch/roll commands
3. **Frame flexibility** (minor for small frames)

The min-residual approach applies most directly to the `trajectory_optimizer.py` planning phase. If we learn a small MLP on PyBullet sim rollout data (position, velocity, orientation vs. commanded trajectory), we can characterize the sim's implicit residual physics (since the sim is not purely analytical — it has drag, rotor inertia, etc.) and then optimize trajectories that minimize exposure to those effects.

Regarding the ILC → acceleration correction extension: this paper provides the key theoretical insight. Our ILC currently corrects position (and velocity). The paper shows that for agile flight, **the dominant residual is in acceleration** (from aerodynamic drag entering the force balance). If we extended ILC to also correct acceleration feed-forward inputs, those corrections would look exactly like the `d(x, xi)` term in this paper — a learned function of velocity and attitude. The paper's result that a 3-hidden-layer × 32-unit MLP on 6 inputs can predict these residuals accurately (0.067 m position RMSE in test) is direct evidence that such a correction function is learnable at low complexity.

The self-supervised training protocol (use flight trajectory data without labeled residuals, multi-step RK4 rollout loss, Hamiltonian adjoint gradients) is the correct way to fit this model without introducing label noise from numerical differentiation.

---

## Actionable Takeaways

1. **Learn a residual dynamics MLP on PyBullet rollout data.** Collect ~40 varied trajectories through the sim, fit a 2-layer × 32-unit MLP with inputs `[vx, vy, vz, phi, theta, psi]` and outputs 3D acceleration residuals. Use multi-step RK4 rollout loss (not single-step) to avoid noise amplification. The analytic Hamiltonian gradient approach from the paper can be adopted, or standard autodiff (PyTorch/JAX) through the RK4 rollout achieves the same thing.

2. **Add a minimum-residual term to the trajectory optimizer's cost.** In `trajectory_optimizer.py`, after fitting the residual MLP, augment the min-snap cost with `lambda_res * sum_k ||d_xi(x_k)||^2`. This encourages the planner to route through low-drag regions. Use `lambda_res` around 0.1 as a starting point (matching the paper's `lambda_r`).

3. **Extend ILC from position corrections to acceleration feed-forward corrections.** The MLP residual model provides a direct prediction of what acceleration offset to inject at each state. This is equivalent to the paper's `d(x, xi)` feed-forward term and can replace or augment the current cross-track position offsets. The benefit is that acceleration corrections act before the error accumulates, while position offsets act after.

4. **Use the 6-dimensional input representation for the residual model.** The paper's empirical result confirms that `[vx, vy, vz, roll, pitch, yaw]` is sufficient to predict aerodynamic and attitude-dependent residuals. Do not include absolute position in the residual model input — position is not a physical cause of drag.

5. **Apply per-section scaling based on local velocity.** The paper's minimum-residual cost naturally weights corrections by speed (higher speed → more drag → larger `||d_xi||`). Our current per-section ILC already applies section-specific alpha scaling (iteration 42 velocity correction). The min-residual approach would make this principled: sections with higher speed automatically receive more correction because the learned residual is larger there.

6. **Check trajectory jerk vs. tracking error tradeoff.** The paper shows min-residual trajectories can have higher jerk (46.7 vs 22.7 m/s³) while achieving lower tracking error. Do not penalize jerk too heavily in our cost — jerk constraints may be inadvertently forcing the trajectory into high-drag regions.

7. **Use out-of-distribution generalization as a validation criterion.** The paper tests on 3 trajectories outside the training distribution and shows only modest degradation (0.067→0.093 m). We should similarly validate any learned residual model on held-out maneuver types before relying on it in the trajectory optimizer.

8. **Prefer the ADAM learning rate of 1e-3 for real-world fitting** (the paper used 1e-2 for simulation, 1e-3 for real hardware, likely because hardware data is noisier). In our PyBullet sim context, 1e-2 with segment size N=50 and batch size 10 is appropriate.

---

## Limitations & Caveats

**Not time-optimal.** The minimum-residual objective minimizes drag/residual exposure, not race time. It is orthogonal to time-optimal planning. In a racing context, one must balance residual minimization against lap time. The paper uses a fixed-time pretzel trajectory; for us, the trajectory duration is also a degree of freedom. A combined cost (time + residual) would be needed, and the tradeoff weight requires tuning.

**MLP trained offline on historical data.** The residual model reflects the dynamics of the specific vehicle and environment from which training data was collected. In competition, wind, altitude, or payload changes could shift the residual distribution. The out-of-distribution test shows degradation but not failure; this is acceptable for a fixed-course race but should be noted.

**No guarantees on correction magnitude.** Unlike our current ILC with explicit `max_correction_m` clamping, the minimum-residual approach does not bound how far the trajectory can deviate from the nominal path to avoid drag. If the optimizer finds a drastically different route that minimizes drag but violates gate constraints, it would break the gate-passing requirement. Gate constraints must be added explicitly to the optimization (as waypoint constraints or corridor constraints).

**Variable jerk can stress actuators.** As noted, min-residual trajectories can exhibit higher jerk than min-snap. This is fine if the actuators can handle it, but the paper's experiments are on a specific hardware platform. Our thrust-to-weight ratio and motor bandwidth determine whether higher-jerk trajectories are feasible.

**Assumes closed-loop dynamics are stationary.** The residual MLP is trained once and not updated online. If battery voltage droops mid-race or the drone accumulates rotor imbalance, the learned model becomes inaccurate. For the ~14s race duration this is likely acceptable, but extended operation would need online adaptation.

**Simulation-to-real gap applies in reverse.** The paper's sim results show 55.7% drag reduction; real-world results show 13–19% tracking error reduction. The gap between "reduced residual force" and "reduced tracking error" is large, suggesting the residual is not the only source of tracking error. Our ILC error signal in the sim directly measures tracking error, which integrates all error sources. The residual MLP only captures a portion.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Control regularizer `lambda_r` | 0.1 | Trajectory optimization cost |
| Simulation MLP: hidden layers | 2 × 32 units | Aerodynamic residual learning |
| Real-world MLP: hidden layers | 3 × 32 units | Velocity residual learning |
| MLP input dimension | 6 | `[vx, vy, vz, phi, theta, psi]` |
| Training dataset size (sim) | 40 trajectories, 8,000 states | 8:2 train-test split |
| Training dataset size (real) | 16 trajectories × ~15s each, >10,000 states | Plus 3 OOD test trajectories |
| Mini-batch segment length N (sim) | 50 | Multi-step RK4 rollout |
| Mini-batch segment length N (real) | 25 | Multi-step RK4 rollout |
| Batch size (sim) | 10 | ADAM mini-batch |
| Batch size (real) | 16 | ADAM mini-batch |
| Learning rate (sim) | 1e-2 | ADAM |
| Learning rate (real) | 1e-3 | ADAM |
| Training epochs to convergence (sim) | ~100 | |
| Training epochs to convergence (real) | ~40 | |
| Trajectory horizon (sim) | 10 s, 400 nodes, 0.025 s step | Direct multiple shooting |
| Trajectory horizon (real) | N=280 nodes, 0.03 s step | 7-waypoint pretzel |
| Loss weighting L (real) | diag{1, 1, 0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0, zeros} | Emphasizes position/velocity, not angles |
| Min-residual tracking error (DFBC) | 0.098 m | vs. 0.113 m min-snap, −13.3% |
| Min-residual tracking error (neural DO) | 0.084 m | vs. 0.104 m min-snap, −19.2% |
| Aerodynamic drag (min-residual, sim) | 0.817 N | vs. 1.845 N min-snap (−55.7%) |
| Max acceleration, min-snap | 9.392 m/s² | Real flight |
| Max acceleration, min-residual | 9.091 m/s² | Real flight (−3.2%) |
| Max jerk, min-snap | 22.684 m/s³ | Real flight |
| Max jerk, min-residual | 46.676 m/s³ | Real flight (+105.8%) |
| Acceleration RMSE improvement (sim) | 0.603→0.173 m/s² | Nominal→hybrid model, test set |
| Position RMSE, OOD test | 0.093 m | 16-trajectory real-world model |
