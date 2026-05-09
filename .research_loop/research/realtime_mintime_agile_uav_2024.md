# Real-time Planning of Minimum-time Trajectories for Agile UAV Flight

- **URL**: https://arxiv.org/abs/2409.16074
- **Authors**: Krystof Teissing, Matej Novosad, Robert Penicka, Martin Saska — Multi-robot Systems Group, Faculty of Electrical Engineering, Czech Technical University in Prague
- **Year**: 2024
- **Venue**: IEEE Robotics and Automation Letters (vol. 9, no. 11, pp. 10351–10358, November 2024)
- **DOI**: 10.1109/LRA.2024.3471388
- **Code**: https://github.com/ctu-mrs/pmm_uav_planner (GPL-3.0, C++)

---

## Key Contribution

Teissing et al. introduce a real-time minimum-time multi-waypoint trajectory planner for multirotors that runs in milliseconds on onboard hardware. The paper makes three interlocking contributions that, together, close most of the gap between computationally cheap point-mass model (PMM) planning and expensive full-model optimization:

1. **Limited Thrust Decomposition (LTD)**: An iterative algorithm that constrains the *norm* of the total thrust acceleration vector (`‖a_T‖ ≤ a_Tmax`) rather than bounding each axis independently. Per-axis constraints, used by all prior PMM-based methods, are overly conservative because they do not allow one axis to use the slack of another. LTD iteratively finds consistent per-axis acceleration bounds that together satisfy the thrust-norm constraint, enabling the drone to exploit its full motor capacity. Empirically, this alone yields trajectories more than 20% faster than equivalent per-axis-constrained plans.

2. **Gradient-based multi-waypoint velocity optimization**: At intermediate via-waypoints, the drone's 3D velocity is unknown. Prior methods sampled or searched over these velocities. Teissing derives the closed-form gradient of total trajectory time with respect to each via-waypoint velocity and applies gradient descent. Convergence is approximately 100× faster than the sampling-based state-of-the-art (cone-refocusing).

3. **Drag modeling inside PMM**: The first PMM trajectory planner to incorporate a linear aerodynamic drag model. Drag is evaluated analytically at the critical switch-time instants (velocity extrema) and fed into the LTD computation. This substantially reduces tracking error at speeds above ~60 km/h, where drag forces are no longer negligible relative to thrust.

The combined result is a planner capable of generating aggressively time-optimal trajectories — validated up to 3.5g and over 100 km/h in real-world outdoor tests — with planning latency suitable for onboard real-time replanning (10–100 Hz rate).

---

## Technical Approach

### Problem Formulation

Given a sequence of n+1 waypoints with fixed start and end states (position and velocity), and unknown velocities at the n-1 intermediate via-waypoints, the problem is:

```
{v*_1, ..., v*_{n-1}} = argmin  T_Π(v_1, ..., v_{n-1})
```

where `T_Π` is total trajectory duration. Each segment is planned independently with boundary conditions set by its two endpoint velocities, so the optimization variable space is (n-1) × 3 dimensional. Treating waypoint velocities as the primary decision variable (rather than polynomial coefficients or segment times directly) is the key structural insight — it decouples the problem into independent segments solvable analytically.

### Segment Structure: Bang-Bang and Bang-Singular-Bang

For each axis independently, the time-optimal single-axis problem is solved analytically using Pontryagin's Maximum Principle. There are two canonical cases:

- **Bang-bang**: Two acceleration phases (full +a_max, then full -a_max or vice versa). Duration: `T_ax = t₁ + t₂`, computed from the boundary conditions via four closed-form cases covering all sign combinations.
- **Bang-singular-bang**: Three phases, where a constant-velocity coast phase (at the velocity bound) is activated when the unconstrained bang-bang would exceed `v_max`. Duration: `T_ax = t₁ + t_σ + t₂`.

The segment time is the maximum over all three axes (axis synchronization). Per-axis accelerations are then scaled down by a factor `γ ∈ (0, 1]` so that all axes complete in the same duration, enabling synchronized multi-axis motion.

### Limited Thrust Decomposition (LTD)

The thrust constraint `‖a_T‖ ≤ a_Tmax`, where `a_T = a - d - g` (acceleration minus drag minus gravity), couples all three axes. LTD resolves this coupling iteratively without a nonlinear solver:

1. **Initialize**: Assume equal thrust allocation: `a_max_i = a_Tmax / sqrt(3)` for each axis.
2. **Critical instants**: Identify the four time instants within a segment where any axis switches acceleration sign. These are the points where the thrust vector direction changes most rapidly.
3. **Scaling per instant**: At each critical time l, compute `β_l` solving:
   ```
   a_Tmax = ‖β_l · a_l - d_l - g‖
   ```
   where `a_l` is the current per-axis acceleration vector and `d_l` is drag at that instant (computed from the linear drag model using current velocity and attitude estimate).
4. **Update per-axis bounds**: `a_max_i = min_{l : a_l_i > 0} { β_l · a_l_i }` — take the tightest constraint across all critical instants and positive-acceleration axes.
5. **Recompute and iterate**: Recompute segment durations with updated bounds. Repeat until `|max_l ‖a_l_T‖ - a_Tmax| < ε_a`.

The algorithm has no convergence proof but empirically converges in a small constant number of iterations. The drag model is:
```
d_l = R(t_l) · D · R^T(t_l) · v(t_l)
```
where D is a diagonal matrix of per-axis drag coefficients, and the attitude R is estimated from the velocity direction (small-angle approximation for PMM).

### Gradient-Based Multi-Waypoint Velocity Optimization

For a trajectory with via-waypoint velocities `v_k`, the gradient of total time with respect to `v_k` is:

```
∇T_Π(v_k) = ∂T_k / ∂v_k  +  ∂T_{k+1} / ∂v_k
```

where T_k and T_{k+1} are the durations of the segments before and after waypoint k. Because each segment duration is an analytic function of its endpoint velocities (via the bang-bang formula), these partial derivatives have closed-form expressions. The update rule:

```
v_k ← v_k - α · ∇T_Π(v_k)
```

A subtlety arises from axis synchronization: the segment duration is the maximum-axis duration. The gradient contribution from a "slave" axis (not the bottleneck) is zero, since its per-axis time is scaled to match the master (and thus does not appear in the total time). Only the master (M) axis contributes gradient for that segment. When the master axis identity changes during optimization, a step-size reduction by factor η = 0.5 is applied to prevent oscillation.

**Initialization**: Via-waypoint velocities are initialized along the chord direction to the next waypoint, with magnitude limited by a rotation-angle parameter `r = 0.6` to avoid initializing with sharp heading reversals.

### Constraint Handling

- **Thrust norm**: Enforced by LTD iteration (exact, up to convergence threshold ε_a).
- **Per-axis velocity limits**: Enforced exactly by activating the bang-singular-bang coast phase when needed.
- **Feasibility (non-negative times)**: Handled by detecting non-differentiable points of `|t₁| + |t₂|` and reflecting infeasible cases.
- **Gate passage**: Waypoints are placed at gate centers; the method does not enforce gate aperture constraints (only position passage, not orientation or clearance margin).

---

## Results

| Metric | Value |
|--------|-------|
| Planning time | Milliseconds per trajectory (on Khadas VIM3 onboard computer) |
| Convergence speedup vs sampling baseline | ~100× |
| Trajectory time reduction vs per-axis constraint | >20% |
| Max tested acceleration | 3.5g (~34 m/s²) |
| Max tested speed | >100 km/h (~28 m/s) |
| Tracking error vs full-model optimizer | Similar or smaller |

**Ablation findings**: Each of the three main contributions (thrust norm constraint, gravity modeling, drag modeling) independently reduces tracking error. Drag modeling is especially impactful above 60–70 km/h. Removing any one of the three measurably degrades performance.

**Validation**: Simulation benchmarks and real-world outdoor flights with a 1.21 kg quadrotor (68 N collective thrust capacity, ~5.7g). The NMPC tracker used for tracking handles the rotational dynamics that the PMM plan ignores, which is why PMM-based trajectories are trackable in practice despite being formally dynamically infeasible for the rotational DOF.

---

## Relevance to Our System

Our system uses minimum-snap polynomial trajectories with L-BFGS time allocation and TOPP-style retiming. Current race time is ~13.68 s against a target of <14 s, with Gate 7 (helix section) as the persistent worst gate at 0.284 m tracking error. The Teissing approach is relevant on several dimensions:

**1. Thrust norm constraint as a drop-in improvement for time allocation**

Our `DroneConstraints` enforces a per-axis `max_acceleration = 12.0 m/s²`. The Teissing result shows that constraining the *norm* instead (same numerical value) would allow the drone to apply 12 m/s² to the dominant axis during a turn rather than 12/sqrt(3) ≈ 6.9 m/s² per axis. For Gate 7's helix segment, where the horizontal centripetal acceleration dominates, this is directly exploitable: the vertical axis has surplus capacity during the horizontal turn and could provide more lift, or more horizontal thrust could be allocated. Expected gain: 5–10% reduction in segment time for high-curvature segments.

**2. Waypoint velocity as the optimization variable**

Our L-BFGS optimizes `log(T_i)` (segment times) using finite-difference gradients. Reparameterizing to optimize via-waypoint velocities `v_k` instead gives access to analytic gradients (Teissing's derivation applies directly to our PMM approximation). This is cleaner than our current approach and would allow more reliable convergence. The closed-form gradient enables hundreds of iterations in the same budget that L-BFGS currently uses for ~200 iterations with numerical gradients.

**3. Drag correction for Gate 7 helix**

Gate 7's helix requires sustained lateral acceleration over a curved path. At our max_velocity of 15 m/s (54 km/h), drag forces are in the 2–3 m/s² range (using Teissing's δ_x=0.28, δ_y=0.35 coefficients). Our current planner does not account for drag — meaning the polynomial assumes the drone can sustain 15 m/s around the curve, but in practice it slows to ~12–13 m/s due to drag-induced deceleration. Adding a linear drag correction to the segment time estimate for helix segments would reduce the tracking error at Gate 7 specifically, since the reference trajectory would be consistent with what the drone can physically do.

**4. Bang-bang as warm-start for our polynomial optimizer**

The PMM-based bang-bang solution gives a lower-bound trajectory time and an associated set of via-waypoint velocities. Using this as the initial guess for our L-BFGS (rather than our heuristic speed-factor initialization) would give the optimizer a much better starting point, likely reducing the number of L-BFGS iterations to convergence and potentially finding a better local minimum.

**5. Computational efficiency for replanning**

Our current planner runs once per race setup (offline). The Teissing approach enables *online* replanning at 10–100 Hz, which would allow us to react to gate position updates during flight — relevant if our EKF gate estimates drift. This is a longer-term capability but the architecture supports it if we adopt the PMM formulation.

---

## Actionable Takeaways

Ordered by expected impact on our current bottleneck (Gate 7 helix, 0.284 m error; overall race time ~13.68 s):

**1. Implement thrust norm constraint in `_optimize_time_allocation()` (HIGH IMPACT)**
Replace the per-axis acceleration penalty with a norm check: `‖[Δv_x/T, Δv_y/T, Δv_z/T]‖ ≤ max_accel_norm`. For our helix segments, this will allow more horizontal acceleration without being penalized by the vertical axis constraint, directly reducing the time allocated to Gate 7's segment. Estimated impact: 3–7% reduction in helix segment time.

**2. Add linear drag correction to segment time estimates (MEDIUM IMPACT)**
For segments with average speed above 10 m/s, add a drag correction factor:
```
T_corrected = T_nominal * (1 + δ_eff * v_avg / a_max)
```
where `δ_eff ≈ 0.3` (average of Teissing's horizontal drag coefficients). This makes the segment time estimate physically consistent with drag-limited speed, reducing the gap between planned and achieved trajectory. Particularly relevant for Gate 7's helix where sustained high speed is required.

**3. Reparameterize velocity optimization to use analytic gradients (MEDIUM-HIGH IMPACT)**
Implement Teissing's gradient `∂T_k/∂v_k` for our PMM-approximated segments. Use this in place of L-BFGS finite-difference optimization for the per-gate velocity magnitude at entry/exit waypoints. This is an additive improvement — keep the polynomial fitting but optimize entry/exit velocities with analytic gradients first, then fit polynomials to the resulting boundary conditions.

**4. Increase optimizer iteration budget and improve initialization (MEDIUM IMPACT)**
Our L-BFGS uses maxiter=200. Increase to 500 for race-critical segments (helix, tight turn). Initialize straight-section velocities at 0.8× max_velocity (vs current 0.45–0.65×) — Teissing shows gradient descent quickly finds the right speed even from aggressive initialization, and we may be leaving time on straights due to conservative initialization.

**5. Use PMM bang-bang solution as warm-start (LOW-MEDIUM IMPACT)**
Call the open-source `pmm_uav_planner` binary for our 12-gate race course (via subprocess or pybind11 wrapper) and extract its segment time allocation. Feed this into our L-BFGS as the starting point. The PMM plan is the theoretical lower bound; our polynomial optimizer then adds smoothness while staying near-optimal.

**6. Evaluate drag model parameters for our hardware (LOW IMPACT, REQUIRED FOR ABOVE)**
Teissing uses δ_x=0.28, δ_y=0.35, δ_z=0.70 for a 1.21 kg drone. Our drone characteristics differ. Extract drag coefficients from our simulation or use system identification if real hardware is available. For PyBullet testing, these can be estimated from the deceleration profile when thrust cuts to gravity-only.

---

## Limitations & Caveats

**Point-mass model vs. full rotational dynamics**: PMM ignores attitude settling time. Switching from +a_max to -a_max requires the drone to slew its attitude, which takes 50–200 ms at high aggressiveness. The actual trajectory deviates from the PMM plan during direction reversals. Teissing compensates with an NMPC tracker, but the plan itself is not attitude-feasible. Our min-snap polynomials implicitly enforce smooth attitude changes via the max_tilt_angle parameter, which is an advantage for tracking fidelity at the cost of trajectory time.

**Gate aperture constraints not enforced**: Placing waypoints at gate centers does not guarantee the trajectory stays within the gate opening. For narrow gates (our aperture is ±0.6 m), the PMM trajectory may clip the frame at high speeds. Our entry/exit waypoint approach (explicit approach vectors 0.4 m before/after gate center) is more conservative but safer for gate clearance.

**Local vs. global optimality**: Gradient descent on via-waypoint velocities is provably local. The initialization with `r=0.6` mitigates but does not eliminate the risk of local minima. Multiple restarts from different initializations may be needed for global-quality solutions, reducing the computational advantage.

**Limited cross-segment coupling**: The gradient `∂T_k/∂v_k` only involves segments k and k+1. Long-range trade-offs (e.g., slowing before Gate 6 to enable a faster Gate 7 entry) are captured only implicitly through sequential gradient updates. Our L-BFGS, despite using approximate gradients, optimizes all segment times jointly and can in principle capture such interactions directly.

**Drag model linearity**: `F_drag = δ·v` is accurate only at moderate speeds. Above ~80 km/h, drag grows quadratically. At our target speeds (15 m/s = 54 km/h), linear drag is a reasonable approximation but will underestimate drag at the highest velocity peaks.

**C++ implementation**: The open-source planner is compiled C++. Direct use from Python requires ctypes or pybind11 wrapping, or subprocess invocation with JSON I/O. This is feasible but non-trivial to integrate cleanly.

**No formal convergence guarantee for LTD**: The iterative thrust decomposition has no convergence proof. Empirically it converges in few iterations, but pathological cases (near-singular thrust configurations, high drag asymmetry) may oscillate or diverge.

---

## Key Parameters / Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `η` | 0.5 | Step size reduction factor on master-axis role change in gradient descent |
| `r` | 0.6 | Via-waypoint velocity initialization: limits heading rotation angle to prevent bad starts |
| `ε_a` | 0.01 m/s² | LTD convergence: stop iterating when thrust norm error is below this |
| `ε_t` | 1×10⁻⁴ s | Velocity optimization convergence: stop when time improvement drops below this |
| `a_Tmax` | ~5.7g (68 N / 1.21 kg) | Maximum collective thrust acceleration of their test platform |
| `δ_x` | 0.28 | Linear drag coefficient, x-axis (body frame) |
| `δ_y` | 0.35 | Linear drag coefficient, y-axis (body frame) |
| `δ_z` | 0.70 | Linear drag coefficient, z-axis (vertical/downwash dominated) |
| LTD critical times | 4 per segment | Number of critical acceleration-switch instants checked per segment |
| Speedup vs sampling | ~100× | Gradient descent vs cone-refocusing sampling baseline |
| Time speedup vs per-axis | >20% | Norm constraint vs per-axis constraint in trajectory time |
| Max validated accel | 3.5g (~34 m/s²) | Real-world outdoor flight validation |
| Max validated speed | ~28 m/s (>100 km/h) | Real-world outdoor flight validation |
| Platform mass | 1.21 kg | Test hardware (for drag coefficient context) |
