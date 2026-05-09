# Why Change Your Controller When You Can Change Your Planner: Drag-Aware Trajectory Generation
- **URL**: https://arxiv.org/abs/2401.04960
- **Authors**: Hanli Zhang, Anusha Srikanthan, Spencer Folk, Vijay Kumar, Nikolai Matni
- **Year**: 2024
- **Venue**: L4DC 2024 (Learning for Dynamics and Control Conference), submitted January 10, 2024

---

## Key Contribution

This paper proposes a data-driven trajectory planning framework that accounts for aerodynamic drag effects on quadrotors by modifying the planner rather than the controller. The central insight is that when a quadrotor experiences aerodynamic drag (from velocity-dependent forces on the airframe, payload, or rotors), the conventional approach is to redesign the feedback controller to reject this disturbance or compensate it. The authors argue that a better approach is to plan reference trajectories that are *intrinsically easier for the existing controller to track* — trajectories that lie within the controller's effective tracking capability envelope given the drag dynamics.

The methodological contribution is a layered decomposition of the optimal control problem. By introducing a redundant reference variable and relaxing the equality constraint between planned and executed trajectories into a soft penalty, the authors derive a two-layer formulation: an outer planning layer that generates reference trajectories minimizing path cost plus a "tracking penalty," and an inner tracking layer where the fixed SE(3) feedback controller executes the reference. The tracking penalty is learned from simulation data (200,000 rollouts) using a shallow neural network, enabling fast gradient-based optimization at test time via projected gradient descent. This results in up to 83% reduction in position tracking error in simulation and eliminates controller saturation on hardware (Crazyflie 2.0), all without touching the controller.

---

## Technical Approach

### Problem Formulation

The paper starts from a standard finite-horizon optimal control problem over discrete time N:

```
min_{x,u}  C(x_{0:N}) + sum_t ||D_t u_t||^2
subject to: x_{t+1} = f(x_t, u_t),  x_0 = z_0
```

where `f` is the quadrotor dynamics including drag, `C` is a path cost, and `||D_t u_t||^2` penalizes control effort.

To decouple planning from execution, the authors introduce a separate reference trajectory variable `r_{0:N}` and relax the strict constraint `r_t = x_t` into a soft penalty via parameter `ρ`:

```
min_{r_{0:N}} [ C(r_{0:N}) + min_{x,u} sum_t ( ||D_t u_t||^2 + ρ||r_t - x_t||^2 ) ]
subject to: x_{t+1} = f(x_t, u_t),  x_0 = z_0
```

The inner minimization over `(x, u)` for a fixed reference `r_{0:N}` defines the **tracking penalty**:

```
g^track_{ρ̄, π_q}(z_0, r_{0:N}) := min_{x,u} sum_t ( ρ̄||u_t||^2 + ||r_t - z_t||^2 )
  subject to: z_{t+1} = f_q(z_t, u_t),  z_0 given
```

where `ρ̄ = 1/ρ` is a reparameterization that improves numerical stability (ρ̄ ≈ 0 emphasizes tracking over control effort, giving stable training). The key insight is that this penalty is a function of *which controller is used* (`π_q`). When the inner minimization is evaluated under the actual SE(3) geometric feedback controller (not an optimal controller), it captures how well *that specific controller* can follow any given reference, including all its limitations: bandwidth limits, gain tuning, saturation.

### Drag Model

The aerodynamic forces modeled are:

```
f_a = -C||v||_2 R^T v - K η_s R^T v
```

where:
- `C` is a diagonal matrix of parasitic (form) drag coefficients
- `K` is a diagonal matrix of rotor drag coefficients
- `η_s` is the collective thrust
- `R` is the rotation matrix (body-to-world)

This captures two distinct drag mechanisms: velocity-dependent form drag (dominant at high speeds) and thrust-dependent rotor drag (dominant during aggressive thrust changes). Both are expressed in the body frame.

### Trajectory Parameterization

Reference trajectories are parameterized as **polynomial splines of order 7 with 3 segments** (total 96 coefficient vector `c` ∈ ℝ⁹⁶). This is a minimum-snap-style parameterization. The planning problem becomes:

```
min_c  [ c^T H c + g^track_{ρ̄, π_q}(z_0, c) ]
subject to:  A c = b
```

where `c^T H c` is the standard minimum-snap cost and `A c = b` encodes waypoint constraints. The constraint `A c = b` is handled via projection during gradient descent, i.e., gradient steps on the unconstrained coefficients `c` are projected back to the feasible set after each step.

### Learning the Tracking Penalty

Rather than computing `g^track` analytically (intractable for a nonlinear controller), the authors use supervised learning:

1. **Data collection**: Generate 200,000 random polynomial trajectories from the 96-dimensional coefficient space. For hardware experiments, 709,382 trajectories were used.
2. **Simulation rollout**: Each trajectory is executed in RotorPy simulation under the SE(3) geometric feedback controller. The resulting tracking cost (sum of squared tracking errors) is recorded.
3. **Network training**: A 3-layer MLP with hidden layers `{100, 100, 20}` neurons and ReLU activations maps trajectory coefficients `c` to tracking cost `g^track`. Four separate networks are trained for `ρ̄ ∈ {0, 0.1, 0.5, 1}`. Training uses SGD with momentum 0.9, 80/20 train/validation split.

### Test-Time Optimization

At deployment time, the drag-aware planning problem is solved using **projected gradient descent** with a maximum of 30 iterations. The gradient of `g^track` with respect to `c` is provided by backpropagation through the learned network. The projection step enforces waypoint constraints `A c = b` after each gradient update.

This is fast (30 iterations of backprop through a small MLP) and runs offline before the race, consistent with a pre-computed trajectory approach.

---

## Results

**Simulation (RotorPy):**
- Compared against: (1) standard minimum-snap (Mellinger & Kumar, 2011) and (2) minimum-snap with drag-compensated controller (Svacha et al., 2017).
- **Position tracking error reduced by up to 83%** relative to baseline minimum-snap, across tested waypoint sequences.
- The learned tracking penalty varied noticeably across `ρ̄` values, suggesting the hyperparameter has meaningful effect. Best results were generally at `ρ̄` close to 0 (pure tracking emphasis).
- The method outperformed the drag-compensated controller baseline, demonstrating that modifying the planner provides additional benefit beyond the controller-side approach.

**Hardware (Crazyflie 2.0):**
- With standard minimum-snap trajectories, the Crazyflie's controller **saturated and the vehicle crashed** when tracking aggressive waypoint sequences in the presence of drag.
- With drag-aware planned trajectories, the same waypoint set was **successfully tracked** without saturation.
- State feedback provided by motion capture at 100 Hz.
- The hardware network required more training data (709K trajectories) due to the sim-to-real gap in drag modeling.

**Baseline comparison summary:**

| Method | Tracking Error (sim) | Hardware feasibility |
|---|---|---|
| Min-snap (Mellinger 2011) | Baseline | Crash/saturation |
| Min-snap + drag controller (Svacha 2017) | Reduced | Not tested |
| Drag-aware planner (this paper) | -83% vs baseline | Successful |

---

## Relevance to Our System

Our system is directly relevant to this paper in several ways, though the direction of applicability is somewhat inverted from what might first appear.

**Our current situation**: We use a min-snap polynomial trajectory optimizer with L-BFGS time allocation. Our geometric (SE(3)) controller has feedforward acceleration weight of 0.4 (deliberately partial). The reason feedforward is at 0.4 and not 1.0 is explicitly documented in `mpc_tracker.py`: the kinematic sim has linear drag (`0.5 * vel`) that the controller does not model, so full feedforward causes overshoot at high-speed segments. This is exactly the drag mismatch problem the Zhang et al. paper addresses.

**Direct module connection**: The paper targets the same planning-controller architecture we have. Their "planning layer" corresponds to `trajectory_optimizer.py` and their "tracking layer" corresponds to `mpc_tracker.py`.

**Why the approach matters for us**: Our average tracking error is 0.227m (22.7 cm) and our race time is 16.69s, with a target of <14s. To close the race time gap, we need to push velocities higher. But higher velocity means more drag force (`f_a ∝ ||v||`), which means our partial feedforward becomes increasingly inaccurate. The Zhang et al. paper shows that drag-aware planning can enable faster trajectories that stay within the controller's tracking envelope — exactly what we need.

**Where our situation diverges**: The paper learns the tracking penalty from simulation rollouts. In our case, the sim uses linear drag (coefficient 0.5), which is a known, parametric model. This means we do not need to learn the tracking penalty from data — we can compute it analytically or use the drag coefficient identification approach (Faessler 2018) directly. The Zhang et al. contribution is most valuable when the drag model is unknown or complex (e.g., payload delivery with unknown aerodynamics). For our sim, the drag model is fully known.

**Planning-side drag compensation**: The core insight — generate trajectories that are conservative in drag-heavy regimes and aggressive in drag-light regimes — can be implemented in our `trajectory_optimizer.py` without any ML. By incorporating the sim's linear drag model (`f_drag = -0.5 * v`) into the trajectory feasibility constraints, we can push velocities higher on straight segments (where drag is predictable) and slow down through tight turns (where the drag-induced cross-track force is largest). This is a planning-side analogue of what the paper does via the learned tracking penalty.

**Feedforward completeness vs. planning conservatism**: The Faessler (2018) paper in our research archive provides the complementary controller-side fix (feedforward drag compensation). Zhang et al. (2024) provides the planning-side fix. Ideally, both are applied: fix the controller's feedforward first (to reduce model mismatch), then adjust the planner to be drag-aware (to push speeds higher within the corrected model).

---

## Actionable Takeaways

1. **Incorporate linear drag into trajectory feasibility constraints (planning side)**: In `trajectory_optimizer.py`, the constraint on maximum acceleration should account for drag. At velocity `v`, the net thrust available for acceleration is `thrust - drag = thrust - 0.5*v`. A drone traveling at 10 m/s with 0.5 drag coefficient has 5 N/kg of thrust budget consumed by drag. This means the effective acceleration budget is `max_accel - 0.5 * v`, not the constant `max_accel = 20.0`. Adding this speed-dependent derating to the constraint checker would prevent the optimizer from generating trajectories that are nominally feasible but practically infeasible under drag.

2. **Reduce planned velocity at tight turns to match tracking envelope**: The Zhang et al. tracking penalty is highest when the reference trajectory requires rapid direction changes at high speed (the controller cannot generate the needed lateral force fast enough). In our planner, this corresponds to segments with high curvature at high speed. Adding a curvature-weighted speed limit (e.g., `v_max_segment = v_limit / (1 + k_drag * curvature * speed)`) to the L-BFGS time allocation would act as a proxy for the learned tracking penalty without any neural network.

3. **Validate the approach by sweeping feedforward weight**: The paper's result (planning modification beats controller modification) assumes the controller is already reasonably well-tuned. Our feedforward at 0.4 is suboptimal due to the drag mismatch. Before implementing full drag-aware planning, try implementing the Faessler 2018 drag feedforward in the controller (set dx = dy = 0.5 s⁻¹ matching the sim's 0.5 linear drag coefficient), then re-tune feedforward_accel upward from 0.4 toward 1.0. If full feedforward becomes stable after drag compensation, we confirm the model mismatch is the root cause.

4. **Use the paper's ρ̄ parameterization insight**: The tracking penalty weight `ρ̄ ≈ 0` consistently performed best in the paper, meaning tracking accuracy matters more than control effort minimization. In our context, this suggests the objective function during time allocation should weight tracking error more heavily than control energy — consistent with the current min-snap formulation which minimizes snap (proxy for control effort) without an explicit tracking error term.

5. **If racing-line speed profile is added**: The `racing_line.py` module could benefit from a drag-aware speed profile. The minimum-curvature racing line already reduces lateral acceleration demand; adding a drag-derating factor `v_max(curvature) = v_rated / sqrt(1 + (0.5 * v / g)^2)` would naturally slow the drone in regions where drag-induced lateral deviation is highest (high speed, high curvature), matching the spirit of the Zhang et al. tracking penalty without any data-driven learning.

6. **For a more complete implementation (medium effort)**: Collect trajectory rollout data in our PyBullet sim (with the known 0.5 linear drag) and train a shallow MLP on the polynomial coefficients → tracking cost mapping, exactly as the paper does. This requires 50,000–200,000 rollouts (each is a short sim run), a 3-layer MLP, and projected gradient descent at planning time. The benefit is automatic adaptation to complex drag effects (including any nonlinearity in our sim), at the cost of an offline data collection step.

---

## Limitations & Caveats

1. **Drag model for our sim is already known and simple**: Our sim uses linear drag `f = -0.5 * v`, which is a parametric model with a single known coefficient. The paper's method is designed for *unknown* or complex drag (payload delivery scenarios). For our case, the Faessler 2018 approach (analytical feedforward using known drag) is likely more effective and requires zero training data.

2. **Polynomial order and segment count are fixed**: The paper uses 7th-order polynomials with 3 segments (96 coefficients). Our optimizer is also polynomial-based but may use different parameterization. The learned tracking penalty network is specific to the coefficient space dimensionality it was trained on, so we cannot directly reuse their trained network even if we had it.

3. **Projected gradient descent may not converge**: The paper explicitly notes that the gradient-based solver does not always converge to a feasible solution. They suggest future work on convex approximations. For racing, where reliability is critical, a non-convergent planner could produce invalid trajectories in edge cases.

4. **30-iteration limit may not be sufficient for complex courses**: The paper's waypoint sequences appear relatively simple (a few waypoints with smooth geometry). Our race track has 8 gates with varying orientations. More complex geometry may require more iterations or a larger coefficient vector (more segments), which increases network input dimensionality.

5. **ρ̄ hyperparameter requires tuning**: Four separate networks are trained for `ρ̄ ∈ {0, 0.1, 0.5, 1}`. The best value is selected based on validation performance. This hyperparameter effectively controls how conservative the drag-aware planner is — too small and the planner ignores the tracking penalty, too large and it over-constrains the trajectory (slower race). Racing requires tuning this on the actual track.

6. **Sim-to-real gap in tracking penalty**: The tracking penalty is learned in simulation. If the real drone's drag differs from the sim model (which it does for our hardware target, MAVLink competition drones), the learned penalty network is miscalibrated. The paper addresses this by using more data for hardware (709K vs. 200K trajectories), which reduces variance but does not fix systematic model mismatch. For our PyBullet-only benchmark loop, this is not an issue; for competition hardware, it would require re-training.

7. **Assumes convex feasible set for projection**: The waypoint constraints `A c = b` form a linear (convex) constraint set, enabling efficient Euclidean projection. If we add nonlinear gate constraints (orientation, FOV), the projection step becomes expensive or infeasible.

8. **No real-time re-planning**: The method generates a fixed trajectory offline. If a gate is detected in a different position than expected, the drag-aware trajectory cannot be updated in real time. Our system already has this limitation (pre-computed trajectories), so this does not add a new constraint.

---

## Key Parameters / Constants

| Parameter | Symbol | Value (paper) | Notes |
|---|---|---|---|
| Polynomial order | — | 7 | Min-snap compatible |
| Number of segments | — | 3 | For their tested courses |
| Coefficient vector dim | — | 96 | 3 segments × 7th order × 4 flat outputs |
| Network hidden layers | — | {100, 100, 20} neurons | 3-layer MLP, ReLU activations |
| Training trajectories (sim) | — | 200,000 | Random sampling from coefficient space |
| Training trajectories (hardware) | — | 709,382 | Larger for sim-to-real robustness |
| Train/val split | — | 80 / 20 | Standard split |
| SGD momentum | — | 0.9 | Standard SGD |
| Projected GD iterations | — | 30 (max) | Test-time optimization |
| Tracking penalty weight range | ρ̄ | {0, 0.1, 0.5, 1} | ρ̄ = 0 often best |
| State feedback rate (hardware) | — | 100 Hz | Motion capture |
| Platform mass | — | ~30 g | Crazyflie 2.0 |
| Drag reduction (sim) | — | up to 83% | vs. min-snap baseline |
| Drag model term 1 | — | -C||v||_2 R^T v | Parasitic form drag (velocity-squared) |
| Drag model term 2 | — | -K η_s R^T v | Rotor drag (thrust × velocity) |

**For our system specifically**: The sim's linear drag coefficient of 0.5 N/(m/s)/kg translates to a body-frame drag force of `0.5 * |v|` m/s² of deceleration at every velocity. At 10 m/s, this is 5 m/s² of drag deceleration — 25% of our max acceleration budget (20 m/s²). At 15 m/s (our `max_velocity` constraint), drag alone consumes 7.5 m/s² = 37.5% of the budget. This is significant enough that drag-aware planning (whether via the Zhang et al. method or simpler curvature-speed constraints) should yield measurable gains at our target race times.
