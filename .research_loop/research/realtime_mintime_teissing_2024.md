# Real-time Planning of Minimum-time Trajectories for Agile UAV Flight

- **URL**: https://arxiv.org/abs/2409.16074
- **Year**: 2024
- **Venue**: IEEE Robotics and Automation Letters (vol. 9, no. 11, pp. 10351–10358, November 2024)
- **Authors**: Krystof Teissing, Matej Novosad, Robert Penicka, Martin Saska — Multi-robot Systems Group, Czech Technical University in Prague
- **DOI**: 10.1109/LRA.2024.3471388
- **Code**: https://github.com/ctu-mrs/pmm_uav_planner (GPL-3.0)

---

## Key Contribution

Teissing et al. present a real-time method for planning minimum-time multi-waypoint trajectories that runs onboard in milliseconds, even on modest flight computers. The three interlocking innovations that make this possible are:

1. **Limited Thrust Decomposition (LTD)**: a novel iterative algorithm that constrains the *norm* of the collective thrust acceleration (rather than bounding each axis independently), allowing the drone to use its full motor capacity. Per-axis constraints, used in prior work, are overly conservative; norm-based constraints produce trajectories that are over 20% faster.

2. **Gradient-based multi-waypoint velocity optimization**: rather than sampling or searching over intermediate waypoint velocities, the paper derives the gradient of total trajectory time in closed form and applies gradient descent. This converges to optimal solutions approximately 100× faster than the prior state-of-the-art sampling-based approach.

3. **Drag modeling inside the Point-Mass Model (PMM)**: the first paper to incorporate a linear drag model into PMM-based trajectory planning, which substantially reduces tracking error at high speeds.

The combination produces minimum-time trajectories reaching 3.5g and speeds exceeding 100 km/h, with tracking errors comparable to (or smaller than) methods that optimize over the full multirotor model at a cost of hours.

---

## Technical Approach

### Problem Formulation

The core optimization problem is:

```
{v*_1, ..., v*_n} = argmin T_Π(v_1, ..., v_n)
```

where `T_Π` is total trajectory duration and `v_i` are the unknown velocities at intermediate via-waypoints. Start and end states are fully specified; only waypoint positions are given as input. The key insight is that **treating waypoint velocities as the primary decision variables** dramatically simplifies the problem structure compared to optimizing segment times or polynomial coefficients directly.

### Segment Structure: Bang-Bang and Bang-Singular-Bang

Each segment uses Pontryagin's Maximum Principle to derive time-optimal single-axis trajectories analytically. There are two canonical structures:

- **Bang-bang** (no velocity constraint active): two acceleration phases with durations t₁, t₂. Total: T_ax = t₁ + t₂.
- **Bang-singular-bang** (velocity bound hits): three phases t₁, tσ, t₂, where tσ is a constant-velocity coast at the velocity limit.

Given start/end states and per-axis acceleration bounds, the segment duration T_ax is computed analytically (four explicit closed-form cases). No polynomial fitting; no iterative solver per segment.

### Limited Thrust Decomposition (LTD) Algorithm

The multirotor constraint is that total thrust acceleration magnitude is bounded: `‖a_T(t)‖ ≤ a_Tmax`, where `a_T = a - d - g` (acceleration minus drag minus gravity). This couples the three axes, so per-axis limits must be derived iteratively.

The LTD algorithm proceeds:

1. Initialize per-axis limits assuming equal allocation: `a_max_i = a_Tmax / sqrt(3)`.
2. Identify the four critical time instants when any axis switches acceleration sign.
3. At each critical time l, compute the scaling factor `β_l` solving `a_Tmax = ‖β_l * a_l - d_l - g‖`, where `a_l` is the current acceleration vector and `d_l` is drag at that instant.
4. Update per-axis limit: `a_max_i = min{β_l * a_l_i | a_l_i > 0}` (take the tightest binding constraint across all critical instants).
5. Recompute segment durations with updated per-axis limits. Repeat until `|max‖a_l_T‖ - a_Tmax| < ε_a`.

Drag at each critical time uses a linear model: `d_l = R(t_l) D R^T(t_l) · v(t_l)`, with D a diagonal drag coefficient matrix. Estimating attitude from velocity direction closes the loop. The algorithm has no formal convergence proof but empirically converges in a small number of iterations.

### Gradient-Based Multi-Waypoint Velocity Optimization

For a trajectory of n segments with via-waypoint velocities `v_1, ..., v_{n-1}`, the gradient of total time is:

```
∇T_Π(v_{k}) = ∂T_{k}/∂v_{k} + ∂T_{k+1}/∂v_{k}
```

where T_k is the duration of the k-th segment (which depends on v_k and v_{k-1}). Because each segment duration is an analytic function of its endpoint velocities (via the bang-bang formula), this gradient is computed in closed form. The update rule is standard gradient descent:

```
v_{k+1} = v_k - α ∇T_Π(v_k)
```

A key subtlety for synchronization: when axes are "synchronized" (i.e., the slowest axis determines segment time and the others are scaled to match), only the master axis M contributes to the gradient. Slave axes have zero gradient contribution for that segment. The step size α is reduced by factor η = 0.5 at each "role change" (when the master axis switches). This prevents oscillation around the optimum when the bottleneck axis changes.

### Constraint Handling

- **Velocity bounds**: optional per-axis velocity limits are enforced via the bang-singular-bang structure. If a velocity limit would be violated, the coast phase (tσ) is activated automatically.
- **Thrust norm**: enforced by LTD iteration. The coupling between axes is resolved without a nonlinear solver.
- **Feasibility**: non-negative segment durations are enforced by detecting non-differentiable points of `T_Π_abs = |t₁| + |t₂|` and reflecting infeasible cases.
- **Gate passage**: waypoints are placed at gate centers; the paper does not explicitly use entry/exit approach waypoints (this is a difference from our system).

### Key Numerical Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| LTD convergence threshold `ε_a` | 0.01 m/s² | Stops LTD iteration when thrust error is small |
| Time convergence threshold `ε_t` | 1×10⁻⁴ s | Stops velocity optimization when improvement is tiny |
| Step size reduction factor `η` | 0.5 | Halves gradient step when master axis changes |
| Velocity heading init factor `r` | 0.6 | Limits initial heading rotation to prevent bad initialization |
| Drone mass (test platform) | 1.21 kg | Hardware context |
| Maximum collective thrust | 68 N (~5.7g) | Motor capacity |
| Drag coefficients | δ_x=0.28, δ_y=0.35, δ_z=0.7 | Used in drag model |
| Max tested acceleration | 3.5g | Real-world flight |
| Max tested speed | >100 km/h | Real-world flight |

---

## Results

- **Computation time**: milliseconds per multi-waypoint trajectory, on a Khadas VIM3 onboard computer — suitable for real-time replanning at 10–100 Hz.
- **Speed vs prior work**: 100× faster convergence than the cone-refocusing sampling baseline.
- **Speed vs axis-constraint methods**: >20% reduction in trajectory time by using norm constraints instead of per-axis limits.
- **Tracking error**: similar to or smaller than full multirotor model optimizers (which require hours of computation), validated by ablation study.
- **Ablation findings**: gravity modeling, drag modeling, and norm-constrained thrust all individually contribute to error reduction. Removing any one increases tracking error at high speeds. Drag modeling is especially important at velocities above 60–70 km/h.
- **Validation**: both simulation and real-world outdoor flights.

---

## Relevance to Our System

Our current system uses minimum-snap polynomial trajectories with L-BFGS optimization over log-segment-times. We have 25 segments (entry/exit waypoints per gate × 12 gates + start + virtual finish) and a race time of ~23 s against a target of 14 s. The gap is ~65% — a fundamental time allocation problem, not a tracking problem.

The Teissing approach is highly relevant in three ways:

**1. Time allocation philosophy**: Our L-BFGS objective minimizes `sum(times) + penalty`, but the penalty terms for velocity and acceleration violations are quadratic soft constraints calibrated by hand. The Teissing approach enforces constraints *exactly* (via analytic bang-bang solutions) and optimizes time as the *only* objective with no competing penalty terms. This is architecturally cleaner and avoids the problem where high penalty weights slow down the optimizer convergence but low weights lead to infeasible trajectories.

**2. Gradient information**: Our L-BFGS uses numerical finite-difference gradients (implicit in scipy's L-BFGS-B). Teissing derives closed-form gradients of total time with respect to waypoint velocities. If we parameterize our problem in terms of waypoint velocities rather than segment times (or in addition to segment times), we gain access to analytic gradients that are orders of magnitude cheaper to compute and more numerically stable. This could allow far more optimizer iterations in the same wall-clock budget.

**3. Norm constraint on thrust**: Our `DroneConstraints` uses a per-axis `max_acceleration = 12.0 m/s²` (roughly 1.2g). If instead we constrained the thrust *norm* to `max_acceleration = 12.0 m/s²` total, the drone could allocate more thrust to the dominant axis (e.g., horizontal during a turn), potentially completing segments faster. This is Teissing's main speedup — a direct analogue for our system.

**4. Drag modeling**: At speeds above 10–12 m/s (our max_velocity is 15 m/s), aerodynamic drag causes the actual velocity at segment endpoints to be lower than the polynomial predicts. This means our trajectory over-estimates speed, leading to tracking error. Adding a simple linear drag correction to segment time estimates (as Teissing does) would reduce this error.

---

## Actionable Takeaways

These are concrete changes to make to our system, ordered by expected impact:

**1. Loosen per-axis constraints, implement norm constraint (HIGH IMPACT)**
Replace the per-axis max_acceleration check in `_optimize_time_allocation()` with a norm check: `‖[a_x, a_y, a_z]‖ ≤ max_acceleration`. Currently, the quadratic penalty fires when any single axis's average speed change exceeds 12 m/s². With a norm constraint, sharp turns that demand mostly lateral acceleration can trade off against reduced vertical acceleration, achieving the same turn faster. This alone could recover 5–10% of lap time.

**2. Reparameterize from segment times to waypoint velocities (HIGH IMPACT)**
Instead of optimizing `log(T_i)` for each of the 25 segments, optimize the 3D velocity vectors at the 23 intermediate waypoints (or just their magnitudes and directions). For each candidate waypoint velocity, segment duration is computable in closed form via the bang-bang formula (no inner optimization needed). The gradient of total time with respect to each waypoint velocity is analytic (Teissing eq. derivation). This replaces L-BFGS with finite-difference gradients with gradient descent using exact gradients — likely 10–50× faster per iteration and more reliable convergence.

**3. Increase initial speed estimates (MEDIUM IMPACT)**
Our `_initial_time_allocation()` uses speed factors of 0.45–0.65 of max_velocity. Teissing's method initializes with the velocity pointing along the chord to the next waypoint, scaled to `r=0.6` of maximum — comparable, but Teissing then immediately runs gradient descent, which quickly finds faster solutions. We should increase `maxiter` in L-BFGS from 200 to 500–1000, and try initializing closer to `0.8 * max_velocity` for straight segments (turn_angle < 0.3 rad), since the optimizer will pull back if needed.

**4. Add linear drag correction to segment time estimates (MEDIUM IMPACT)**
Add a drag-aware correction to `_initial_time_allocation()`. For a segment of length d at average speed v, the drag force decelerates the drone by `F_drag = δ * v`, reducing achievable speed. Add a factor `1 / (1 - δ_eff * v / a_max)` to time estimates for long, fast segments. This avoids the case where the polynomial assumes the drone can sustain speed that drag makes physically impossible.

**5. Remove the FOV penalty from the time-allocation optimizer (LOW-MEDIUM IMPACT)**
The FOV penalty in `_optimize_time_allocation()` competes with the primary time-minimization objective. It currently forces the optimizer to slow down segments to keep the gate in view — even when the tracking controller can handle gate acquisition without a perfectly centered view. Move FOV enforcement to the post-optimization `_relax_for_fov()` pass only. This lets the primary optimizer find a faster trajectory, then makes the minimum necessary slowdowns for perception.

**6. Consider warm-starting with Teissing's bang-bang solution as initial guess (LOW IMPACT)**
Before running our polynomial optimizer, compute a PMM trajectory using Teissing's approach (available as open-source C++ code). Use its segment time allocation as the initial guess for L-BFGS rather than our heuristic. The PMM solution is a lower bound on achievable time and may already be close to the polynomial optimum, giving L-BFGS a much better starting point.

---

## Limitations & Caveats

**Point-mass model vs. full model**: PMM ignores rotational dynamics and attitude settling time. A drone cannot instantaneously switch from +a_max to -a_max; the attitude must slew first. For very agile maneuvers (>2g), the attitude transient can take 50–200 ms, meaning the true trajectory deviates from the PMM plan during direction reversals. Teissing mitigates this by using an NMPC tracker that handles the attitude dynamics online, but the reference trajectory itself is not dynamically feasible for the rotational DOF. Our min-snap polynomials do implicitly constrain attitude via the max_tilt_angle parameter, which is an advantage for trajectory smoothness.

**Gate passage constraints not addressed**: Teissing's method places waypoints at gate centers but does not enforce that the trajectory stays within the gate aperture (±0.6m in our case). For gates with narrow clearance, the PMM trajectory may clip the frame. Our entry/exit waypoint approach (0.4m before/after gate center along the gate normal) is a more conservative but safer choice for gate-constrained racing.

**Gradient descent convergence**: While 100× faster than the cone-refocusing baseline, the paper does not prove global optimality — only local. The gradient can get stuck in local minima, particularly when the via-waypoint velocity initialization is poor. The `r=0.6` initialization trick mitigates this but does not guarantee the global minimum.

**No segment time coupling across the full trajectory**: The gradient `∂T_{k}/∂v_k` only involves segments k and k+1. Long-range interactions (e.g., that slowing down before a hard turn may allow a faster exit that compounds over several gates) are only captured implicitly through the gradient propagation. Our L-BFGS, despite using a rough gradient, optimizes *all* segment times jointly and can in principle find cross-segment trade-offs.

**Drag model accuracy**: Linear drag (`F = δ·v`) is only accurate at low Reynolds numbers and moderate speeds. Above ~80 km/h, the drag is better modeled quadratically. At the speeds we target (15 m/s = 54 km/h), linear drag is reasonable but may underestimate at peak velocities.

**C++ implementation**: The open-source code (`ctu-mrs/pmm_uav_planner`) is compiled C++, not Python. Direct integration into our Python codebase would require a Python wrapper (ctypes or pybind11) or calling it as a subprocess. This is non-trivial but feasible as a warm-start generator.

---

## Key Parameters / Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `η` | 0.5 | Step size reduction factor on master-axis role change |
| `r` | 0.6 | Initial velocity heading rotation limit |
| `ε_a` | 0.01 m/s² | LTD convergence: thrust norm error threshold |
| `ε_t` | 1×10⁻⁴ s | Velocity optimization convergence: time change threshold |
| `a_Tmax` | 5.7g (68 N / 1.21 kg) | Maximum collective thrust acceleration (test platform) |
| `δ_x, δ_y` | 0.28, 0.35 | Linear drag coefficients, horizontal axes |
| `δ_z` | 0.70 | Linear drag coefficient, vertical axis |
| max tested speed | ~28 m/s (>100 km/h) | Real-world validated speed |
| max tested accel | 3.5g (~34 m/s²) | Real-world validated acceleration |
| LTD critical times | 4 | Number of critical instants checked per segment |
| Speedup vs sampling | 100× | Gradient descent vs cone-refocusing baseline |
| Time speedup vs per-axis | >20% | Norm constraint vs per-axis constraint |
