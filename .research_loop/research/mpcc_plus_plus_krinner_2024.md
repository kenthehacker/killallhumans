# MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints
- **URL**: https://arxiv.org/abs/2403.17551
- **Authors**: Maria Krinner, Angel Romero, Leonard Bauersfeld, Melanie Zeilinger, Andrea Carron, Davide Scaramuzza
- **Year**: 2024
- **Venue**: Robotics: Science and Systems (RSS) 2024

## Key Contribution

MPCC++ extends the classic Model Predictive Contouring Control (MPCC) framework for drone racing with three synergistic additions that together achieve 100% real-world gate-passage success while matching RL-level lap times. The original MPCC formulates racing as a single optimization that simultaneously maximizes path progress and minimizes contouring error — but in practice it collides with gates because the soft cost alone cannot guarantee spatial feasibility. MPCC++ fixes this by (1) introducing hard prismatic tunnel safety constraints in a Frenet-Serret frame, separating safety enforcement from the performance objective entirely; (2) augmenting the nominal rigid-body dynamics with a data-driven polynomial residual model that captures aerodynamic rotor effects; and (3) using Trust-Region Bayesian Optimization (TuRBO) rather than weighted maximum-likelihood to tune the eight MPC hyperparameters directly on lap time.

The combination is the first model-based controller to achieve 100% success in real-world high-speed drone racing (up to 80+ km/h) while reducing lap time variance to ±0.14 s, compared to 59.3% success for baseline MPCC and 85% for RL. This is a significant practical result: prior competitive controllers either required millions of simulation episodes (RL) or accepted gate-collision risk (classical MPCC).

## Technical Approach

**Contouring Control Foundation.** The MPCC cost function minimizes a combination of lag error `e^l(θ)` (deviation along the path tangent) and contour error `e^c(θ)` (deviation perpendicular to the path), while penalizing angular rate, thrust-rate changes, and progress rate, minus a progress reward term:

```
J = Σ [ ||e^l(θ_k)||²_Ql + ||e^c(θ_k)||²_Qc + ||ω_k||²_Qω
        + ||v_θk||²_Rvθ + ||Δf_k||²_RΔf - μ·v_θk ]
```

The key insight is that the path parameter θ is treated as an additional optimization variable updated online, so the MPC simultaneously selects how fast to progress along the reference path and how closely to track it. This naturally encodes the time-optimality objective through progress maximization.

**Prismatic Tunnel Safety Constraints.** The core safety innovation is reformulating the gate-passage requirement as hard convex constraints defined in the Frenet-Serret frame of the reference path. Four halfspace inequalities bound the drone position within a rectangular cross-section tunnel of width W(θ) and height H(θ):

```
(p_k - p_0(θ_k)) · n(θ_k) ≥ 0
2H(θ_k) - (p_k - p_0(θ_k)) · n(θ_k) ≥ 0
(p_k - p_0(θ_k)) · b(θ_k) ≥ 0
2W(θ_k) - (p_k - p_0(θ_k)) · b(θ_k) ≥ 0
```

where `n(θ)` and `b(θ)` are the normal and binormal vectors of the Frenet frame. The corridor width transitions smoothly through a sigmoid from a nominal wide value `W_n` (allowing flexible maneuvering between gates) to the tight gate opening `W_gate` (enforcing passage through the gate), parametrized over path length. This spatial separation means that between gates the optimizer has full freedom to select the racing line, while at gates safety is hard-constrained.

**Terminal Set for Recursive Feasibility.** A periodic reference trajectory is computed offline that passes through the center of each gate; the MPC terminal state is constrained to lie on this trajectory. This guarantees that the online optimization always has a feasible fall-back (the center-line) and prevents the control problem from becoming infeasible during aggressive flight. The offline reference solves:

```
min Σ||p^d(θ_k) - p_k||²  subject to x_0 = x_M  (periodicity constraint)
```

**Learned Aerodynamic Residual.** The nominal point-mass quadrotor model cannot represent high-speed aerodynamic effects (rotor drag, induced velocity, airframe drag). MPCC++ augments the nominal dynamics with polynomial residual terms fit to collected real-world data (IMU + VICON motion-capture). Each residual force and torque component is expressed as a polynomial in velocities and rotor angular speed Ω:

```
f_x = C_fx · [v_x, v_x³, Ω², v_x·Ω²]ᵀ
f_y = C_fy · [v_y, v_y³, Ω², v_y·Ω²]ᵀ
f_z = C_fz · [v_z, v_z³, v_xy, v_xy², v_xy·Ω², v_z·Ω², v_xy·v_z·Ω²]ᵀ
τ_x = C_τx · [v_y, Ω², v_y·Ω²]ᵀ
τ_y = C_τy · [v_x, Ω², v_x·Ω²]ᵀ
τ_z = C_τz · [v_x, v_y]ᵀ
```

Coefficients are identified via ordinary least-squares, making the augmentation lightweight (no online learning) while adding meaningful fidelity to the prediction model. In simulation this reduces lap time from 5.30→5.15 s in the high-fidelity BEM (Blade Element Momentum) environment.

**TuRBO Hyperparameter Optimization.** Eight cost-function weights (Q_l, Q_c, Q_ω, R_vθ, R_Δf, μ, and two others) are optimized using Trust-Region Bayesian Optimization with 8 parallel instances. The reward signal is negative mean lap time over 3-lap rollouts plus a failure penalty of γ=100 per collision. This takes 600 episodes total. The key result is that TuRBO-tuned MPCC++ achieves lower variance (±0.02 s in sim) than WML-tuned MPCC (±0.1 s), suggesting the optimization better finds the basin of a well-conditioned solution.

**Solver Implementation.** The MPC runs at 100 Hz using ACADOS with the SQP_RTI (Real-Time Iteration) solver, which performs a single SQP step per control cycle. Prediction horizon N=20 at 25 Hz (0.8 s lookahead). The soft-constraint log-sum-exp barrier uses α=100 to approximate hard constraints while maintaining smoothness for gradient-based solvers.

## Results

All experiments use a 7-gate Split-S aerobatic track.

| Environment | Method | Lap Time (s) | Success Rate (%) |
|---|---|---|---|
| Simple Sim | MPCC (WML) | 5.38 ± 0.10 | 100 |
| Simple Sim | MPCC++ (TuRBO) | 5.16 ± 0.02 | 100 |
| Simple Sim | MPCC++ w/ augmented model | 5.09 ± 0.10 | 100 |
| BEM Sim | MPCC (WML) | 5.51 ± 0.06 | 100 |
| BEM Sim | MPCC++ (TuRBO) | 5.30 ± 0.02 | 100 |
| BEM Sim | MPCC++ w/ augmented model | 5.15 ± 0.03 | 100 |
| Real World | MPCC (TuRBO) | 5.67 ± 1.06 | 59.3 |
| Real World | MPCC++ (TuRBO) | 5.41 ± 0.14 | **100** |
| Real World | MPCC++ w/ augmented model | 5.38 ± 0.26 | **100** |
| Real World | RL (reference) | 5.35 ± 0.15 | 85.0 |

Key observations:
- MPCC++ achieves first-ever 100% real-world success for model-based racing
- Lap time variance collapses from ±1.06 s (baseline MPCC) to ±0.14 s
- MPCC++ matches RL performance (5.41 vs 5.35 s) while providing interpretable guarantees
- The learned aerodynamic augmentation adds ~0.03–0.15 s further improvement
- At 80+ km/h, the aerodynamic terms become significant enough to measurably affect trajectory

## Relevance to Our System

Our system uses a geometric SE(3) controller (GeometricTracker in `control/mpc_tracker.py`) with `kp=6, kd=4, feedforward_accel=0.4` tracking min-snap polynomial trajectories at 100 Hz. The current bottleneck is the helix section (gates 7–12), averaging 0.64 m error vs 0.31 m for straight gates 1–6, with overall avg tracking error 0.481 m and race time 13.34 s.

**Directly applicable insights:**

1. **Contouring control vs. trajectory tracking.** Our GeometricTracker is a pure trajectory tracker: it receives a time-parametrized reference `p(t), v(t), a(t)` and minimizes error to the scheduled point. MPCC-style contouring control decouples progress rate from path geometry — the controller chooses how fast to advance along the path online. In tight turns, our controller over-shoots the scheduled time because the drone physically cannot execute the pre-planned timing. MPCC-style re-parametrization would let the optimizer slow down (reduce θ velocity) through the helix rather than accumulate large position errors.

2. **Safety constraints for gate passage.** The prismatic tunnel approach is directly applicable to our gate-sequencing logic. Currently `gate_sequencing/sequencer.py` detects gates passively; the planning has no hard constraint enforcing passage through gate openings. Encoding gate openings as hard corridor constraints in a replanning or MPC layer would eliminate the class of helix errors where the drone flies outside the gate opening.

3. **Learned residual dynamics.** Our kinematic sim uses `accel = accel_des - drag*vel` with `drag=0.5`, which is the same polynomial structure MPCC++ uses (linear-in-velocity drag term). The paper's full polynomial basis `[v, v³, Ω², v·Ω²]` is more expressive. If we transition to a full dynamics sim or real hardware, this residual structure is the right way to capture rotor drag without re-deriving from first principles.

4. **TuRBO for gain tuning.** Our TrackerConfig has 7+ tunable gains (kp_xy, kd_xy, kp_z, kd_z, kr, kw, feedforward_accel). Bayesian optimization of these jointly on lap time is exactly what TuRBO does. This is more principled than our current manual iteration and could resolve the helix vs. straight trade-off automatically.

5. **100 Hz ACADOS SQP_RTI.** The paper achieves real-time MPC at 100 Hz — the same as our target loop rate — using a single SQP step. This validates that real-time nonlinear MPC is feasible at our rate, countering the concern that MPC would be too slow for our 100 Hz budget.

The most immediately applicable module is `race_pipeline.py` (trajectory dispatch logic) and `planning/trajectory_optimizer.py` (reference path). Replacing the time-parametrized dispatch with a contouring-style path parameter `θ` would let the controller adapt progress speed online, which directly addresses the helix tracking bottleneck.

## Actionable Takeaways

1. **Implement path-parameter re-dispatch in GeometricTracker.** Instead of dispatching reference points by elapsed time `t`, maintain a path parameter `θ` and use velocity-tracking error to advance or retard θ online. At minimum: if position error exceeds a threshold (e.g., 0.3 m), slow the reference advancement. This is a low-cost approximation of the full MPCC formulation and can be implemented entirely within `control/mpc_tracker.py` and `race_pipeline.py`.

2. **Add sigmoid gate-corridor narrowing to trajectory optimizer.** In `planning/trajectory_optimizer.py`, encode each gate as a corridor-width constraint that narrows from `W_n` (wide, say 2.0 m) to `W_gate` (actual gate width, ~0.8 m) over a 0.5 m approach distance. Use the sigmoid parametrization `W(θ) = W_gate + (W_n - W_gate) * sigmoid(...)`. This directly prevents the helix drift from missing gate openings.

3. **Use TuRBO (or standard BO) to tune TrackerConfig gains.** Define a 7-dimensional search space over `[kp_xy, kd_xy, kp_z, kd_z, kr, kw, feedforward_accel]`. Reward = negative race time, failure penalty for crash. Run 50–100 episodes with a Gaussian Process surrogate (scipy or botorch). The paper's 600-episode budget is for hardware; in sim this should converge in 50–100 episodes.

4. **Adopt polynomial aerodynamic basis for drag compensation.** Replace the current scalar drag coefficient (currently 0.0) with the MPCC++ polynomial basis: at minimum add a `v³` term and an `Ω²` cross-term. Even without full rotor speed data, fitting `[v, v³]` per-axis using our sim data would improve prediction in the helix where `v_xy` is highest.

5. **Replace WML gain tuning with lap-time-based reward.** Currently gains are tuned manually by inspecting per-gate error. MPCC++ shows that optimizing directly on lap time with a collision penalty (γ=100) is more effective. Even without BO, changing our tuning criterion from "minimize avg tracking error" to "minimize race time subject to no crash" would align our iteration loop with the competition metric.

6. **Set MPC horizon to N=20 at 25 Hz lookahead (0.8 s).** If we implement even a simple linear MPC or predictive feedforward, use a 0.8 s horizon — this spans approximately one helix turn (at 3 m radius, 13 m/s) and allows the controller to pre-bank for curvature.

7. **Enforce recursive feasibility via offline center-line.** Pre-compute a center-line trajectory through all gate centers (`planning/racing_line.py` already does something close). Use this as the terminal cost/constraint in any MPC formulation to guarantee feasibility when the main cost function drives aggressive deviations.

## Limitations & Caveats

1. **ACADOS dependency.** The real-time MPC performance relies on ACADOS with SQP_RTI, which is a specialized C solver not in our current dependencies. Our `control/mpc_tracker.py` uses scipy, which is 10–100× slower for comparable horizon lengths. Implementing the full MPCC++ would require integrating ACADOS (or CasADi + HPIPM), a non-trivial engineering task.

2. **Frenet-Serret frame computation at high curvature.** The prismatic tunnel uses Frenet-Serret normal/binormal vectors, which are undefined or numerically ill-conditioned at high curvature (the torsion is singular). The helix section of our track has high curvature, so the tunnel parametrization would need special handling near zero-curvature inflection points.

3. **Real-world vs. kinematic sim.** MPCC++ is validated on a physical quadrotor with accurate thrust models and VICON ground truth. Our sim is a kinematic model with scalar drag — the aerodynamic augmentation terms (rotor Ω coupling) are not representable in our sim. The learned residual only helps when transitioning to hardware.

4. **7-gate Split-S vs. our track.** MPCC++'s 5.4 s lap covers a 7-gate aerobatic course. Our 13.34 s race over presumably more gates means the absolute lap times are not comparable, but the per-gate error improvement should scale.

5. **No perception.** MPCC++ assumes perfect state estimation (VICON in real-world experiments). In competition conditions without VICON, EKF drift would degrade the tunnel-constraint feasibility. Combining MPCC++ with gate-based drift correction (as in `estimation/gate_pnp.py`) would be necessary for deployment.

6. **Terminal set periodicity.** The recursive feasibility guarantee requires a periodic track. Our track may not be exactly periodic; the terminal set would need to be re-defined as a fixed terminal trajectory segment rather than a full lap orbit.

## Key Parameters / Constants

- **MPC horizon**: N=20 steps at 25 Hz → 0.8 s lookahead
- **Control rate**: 100 Hz (SQP_RTI, single step per cycle)
- **Soft-constraint barrier coefficient**: α=100 (log-sum-exp formulation)
- **Failure penalty in TuRBO reward**: γ=100 (lap time units equivalent)
- **TuRBO budget**: 600 episodes, 8 parallel instances, 3 laps per episode
- **Sigmoid corridor transition**: over ~0.5–1.0 m approach to gate (exact value not stated but implied by gate width transition)
- **Nominal corridor width** W_n: wide (implied ~1.5–2.0 m from track geometry)
- **Gate corridor width** W_gate: tight (actual gate opening, implied ~0.6–0.8 m for competition gates)
- **Learned drag basis**: [v, v³, Ω², v·Ω²] per axis for forces; [v, Ω², v·Ω²] for torques
- **Hyperparameters tuned**: 8 parameters (Q_l, Q_c, Q_ω, R_vθ, R_Δf, μ, and 2 others)
- **Lap time improvement (sim, BEM)**: 5.51 → 5.15 s (6.5% over baseline MPCC, 100% success)
- **Lap time improvement (real world)**: 5.67 → 5.38 s (5.1% over baseline MPCC, from 59% → 100% success)
- **Max flight speed**: >80 km/h (~22 m/s) in real-world experiments
