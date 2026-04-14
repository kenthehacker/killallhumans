# Real-time Planning of Minimum-time Trajectories for Agile UAV Flight

- **URL**: https://arxiv.org/abs/2409.16074
- **Authors**: Krystof Teissing, Matej Novosad, Robert Penicka, Martin Saska (Multi-robot Systems Group, Faculty of Electrical Engineering, Czech Technical University in Prague)
- **Year**: 2024
- **Venue**: IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 10351–10358, November 2024 (DOI: 10.1109/LRA.2024.3471388)
- **Code**: https://github.com/ctu-mrs/pmm_uav_planner (GPL-3.0)

---

## Key Contribution

Teissing et al. present a real-time Point-Mass Model (PMM) trajectory planner that generates minimum-time multi-waypoint trajectories in milliseconds, making it feasible to replan at 10–100 Hz onboard modest flight computers. The primary innovation over prior PMM planners is the **Limited Thrust Decomposition (LTD)** algorithm: an iterative method that enforces the physically correct constraint — that the *norm* of collective thrust acceleration is bounded — rather than bounding each axis independently. Prior PMM methods imposed per-axis acceleration limits, which are overly conservative because they prevent the drone from directing all available thrust along the dominant axis of motion. By replacing per-axis constraints with a norm constraint and iteratively solving for per-axis limits consistent with that norm, LTD produces trajectories that are **over 20% faster** than per-axis-constrained methods.

The second major contribution is a **gradient-based multi-waypoint velocity optimizer**. For a trajectory of n segments, the unknowns are the velocities at intermediate via-waypoints. Because each segment duration is an analytic function of its endpoint velocities (via bang-bang theory), the gradient of total trajectory time with respect to each via-waypoint velocity is computable in closed form. Gradient descent on this analytic gradient converges approximately **100 times faster** than the prior state-of-the-art sampling-based approach. A third contribution is the incorporation of a **linear aerodynamic drag model** directly inside the PMM planner — the first PMM paper to do so — which significantly reduces tracking error at high speeds by making the reference trajectory physically realistic.

---

## Technical Approach

### Point-Mass Model Foundation

The PMM models the drone as a point mass subject to gravity, linear drag, and a bounded collective thrust acceleration. The three translational axes (x, y, z) are treated independently for trajectory computation, with coupling introduced only through the shared thrust magnitude. The key decision variables are the velocities `v_1, ..., v_{n-1}` at the n-1 intermediate via-waypoints (start and end states are fully specified). The optimization problem is:

```
{v*_1, ..., v*_{n-1}} = argmin  T_Π(v_1, ..., v_{n-1})
```

where `T_Π = Σ T_k(v_{k-1}, v_k)` is the sum of per-segment durations and each `T_k` is computed analytically.

### Single-Segment Time-Optimal Solution

For a single axis with bounded acceleration `[a_min, a_max]` and optional velocity bound `[v_min, v_max]`, Pontryagin's Maximum Principle gives two canonical control structures:

**Bang-bang** (no velocity limit active): two phases with accelerations `a_1, a_2 ∈ {a_min, a_max}`.

```
p_1 = p_0 + v_0*t_1 + 0.5*a_1*t_1^2
v_1 = v_0 + a_1*t_1
p_2 = p_1 + v_1*t_2 + 0.5*a_2*t_2^2
v_2 = v_1 + a_2*t_2
T_ax = t_1 + t_2
```

There are four sign combinations for (a_1, a_2), and for each the equations reduce to a quadratic in t_1 (or t_2), yielding a closed-form solution. The valid case (positive phase durations) is selected analytically.

**Bang-singular-bang** (velocity limit active): three phases — accelerate to velocity limit, coast at limit, then decelerate. Adds a "singular" coast phase with `a = 0` and duration `t_sigma`. The coast duration is determined by the distance budget remaining after the acceleration and deceleration phases.

Crucially, no iterative solver is needed per segment. Given per-axis `(a_min, a_max, v_min, v_max)`, the segment duration `T_ax` is evaluated in O(1) time.

### Axis Synchronization

When the three axes have different unconstrained durations `(T_x, T_y, T_z)`, they are synchronized to a common duration `T_s = max(T_x, T_y, T_z)`. The slower axes become the "master" (M), the faster axes are "slaves" (S) — their accelerations are scaled by `γ ∈ (0, 1]` to stretch their durations to match `T_s`. Only one master axis exists per segment; all others are slaves. This synchronization is what makes the gradient computation tractable: the gradient of `T_s` with respect to a via-waypoint velocity only involves the master axis. Slave axes contribute zero gradient (their durations are determined by the master, not by their own dynamics constraints).

### Limited Thrust Decomposition (LTD) Algorithm

A multirotor's total thrust acceleration magnitude is bounded: `‖a_T(t)‖ ≤ a_Tmax`, where the thrust acceleration is:

```
a_T(t) = a(t) - d(t) - g
```

Here `a(t)` is the 3D trajectory acceleration, `d(t)` is the aerodynamic drag acceleration (computed from a linear drag model), and `g` is the gravity vector. This constraint couples all three axes — you cannot simply set `a_max_x = a_max_y = a_max_z = a_Tmax/sqrt(3)` without being overly conservative.

The LTD algorithm resolves this coupling iteratively:

1. **Initialize**: set per-axis limits as `a_max_i = a_Tmax / sqrt(3)` for each axis.
2. **Compute trajectory**: given current per-axis limits, compute bang-bang/bang-singular-bang solutions for each axis and synchronize.
3. **Find critical times**: identify the at most 4 critical time instants `t_l` when any axis switches acceleration sign. These are the only times when `‖a_T(t)‖` can be at a local maximum.
4. **Compute drag at critical times**: using the linear drag model `d_l = R(t_l) D R^T(t_l) · v(t_l)`, where `D = diag(δ_x, δ_y, δ_z)` is the drag coefficient matrix. Attitude `R(t_l)` is estimated from the velocity direction (assuming the drone points into the flow).
5. **Compute scaling factors**: for each critical time `l`, find `β_l` such that `a_Tmax = ‖β_l * a_l - d_l - g‖`. This is a scalar equation solved analytically.
6. **Update per-axis limits**: `a_max_i = min{β_l * a_l_i | a_l_i > 0, all l}`. Take the most restrictive binding constraint across all critical instants.
7. **Repeat** from step 2 until `|max{‖a_l_T‖} - a_Tmax| < ε_a`.

The key insight is that at the critical times, the trajectory is at its acceleration extremes. Scaling these extremes to exactly saturate the thrust norm constraint (instead of the per-axis limits) allows the drone to use full thrust. Empirically, 3–5 iterations suffice for convergence (no formal proof of convergence is given, but the iteration is observed to be monotonically contracting in practice).

### Gradient-Based Multi-Waypoint Velocity Optimization

For a multi-segment trajectory, the gradient of total time with respect to via-waypoint velocity `v_k` is:

```
∇_{v_k} T_Π = ∂T_k/∂v_k + ∂T_{k+1}/∂v_k
```

Only the two adjacent segments `k` and `k+1` share the via-waypoint velocity `v_k`, so the gradient is local. Each partial derivative `∂T_k/∂v_k` is computed from the closed-form bang-bang expression for `T_k`. For the master axis M:

```
∂T_k/∂v_k^i  = ∂T_M/∂v_k^i    (if axis i is master in segment k)
∂T_k/∂v_k^i  = 0                (if axis i is slave in segment k)
```

The update rule is gradient descent:
```
v_k^{n+1} = v_k^n - α * ∇_{v_k} T_Π(v_k^n)
```

The step size `α` is initialized to `α_init` and reduced by factor `η = 0.5` whenever the master axis role changes between iterations (a sign that the step overshot). Convergence is declared when `|T_Π^{n+1} - T_Π^n| < ε_t`. Velocity bounds on via-waypoints are enforced by clamping.

**Initial velocity estimation**: for a waypoint between two segments, the initial velocity is set along the bisector of the incoming and outgoing directions, scaled by a factor that depends on turn angle. Specifically, for a sharp turn (angle `θ` between segment directions), the velocity magnitude is limited to prevent initialization too close to the feasibility boundary. The formula uses a blend factor `r = 0.6`:

```
θ_n = ((1 - r)/2) * θ + r * (l_prev / (l_prev + l_next)) * θ
```

where `l_prev, l_next` are the lengths of adjacent segments.

### Velocity Bounds and Feasibility

Per-axis velocity bounds are enforced through the bang-singular-bang structure automatically: if the planned velocity would exceed `v_max`, the coast phase activates. Infeasibility of a particular via-velocity (where one phase duration goes negative) is detected by checking the non-differentiable points of `T_abs = |t_1| + |t_2|`, which correspond to degenerate cases. These points define the feasibility boundary for the gradient search.

---

## Results

### Computational Performance

- Multi-waypoint trajectory generation: **milliseconds** on a Khadas VIM3 SBC (ARM Cortex-A73, 4 cores), enabling real-time replanning at 10–100 Hz.
- Velocity optimization convergence: approximately **100× faster** than the prior sampling-based (cone-refocusing) approach.
- LTD iteration: typically **3–5 iterations** to converge (ε_a = 0.01 m/s²).

### Trajectory Time Improvement

- Norm-constrained trajectories are **>20% faster** than per-axis-constrained trajectories on identical courses. This is a direct consequence of allowing full thrust utilization along the dominant axis during sharp turns and aggressive climb/dive segments.

### Real-World Flight Validation

- **Maximum sustained acceleration**: 3.5g (~34 m/s²).
- **Maximum speed**: >100 km/h (~28 m/s).
- **Tracking error**: comparable to or smaller than full multirotor model optimizers that require several hours of offline computation. The NMPC tracker used in experiments closed the gap between the PMM reference and the full-model reality.

### Ablation Study

The paper performs an ablation with four configurations:
1. Per-axis only (baseline).
2. Per-axis + drag.
3. Norm constraint, no drag.
4. Norm constraint + drag (full method).

Results show that each component independently reduces tracking error. Drag modeling is especially critical at high speeds (above ~17 m/s / 60 km/h), where the linear drag force represents a significant fraction of the achievable thrust. Removing drag from the reference model causes the tracker to accumulate speed errors that never fully recover on straight sections.

---

## Relevance to Our System

Our system is an autonomous drone racing stack with:
- A min-snap polynomial trajectory optimizer using L-BFGS over log-segment-times.
- Per-axis acceleration constraints in the optimizer (`max_acceleration = 20 m/s²` as a soft penalty bound).
- Current race time of ~16.69s through 12 gates (target: <14s — a ~16% reduction needed).
- 25 waypoints (start + entry/exit per gate + virtual finish), min-snap polynomials across each segment.

### How the per-axis vs norm constraint applies directly

Our `DroneConstraints.max_acceleration = 20.0` is applied in the time-allocation optimizer as a **per-axis** soft penalty: if the estimated acceleration along any single axis exceeds 20 m/s², a penalty is added to the L-BFGS objective. This matches exactly the conservative formulation that Teissing et al. replace with the norm constraint.

The physical reality: our drone's collective thrust is bounded in magnitude, not per-axis. When the drone is in a banked turn, most thrust goes horizontal. The per-axis formulation incorrectly penalizes this and forces the optimizer to allocate more time (slower the turn). With a norm constraint of 20 m/s² total, the optimizer can allocate 20 m/s² horizontally during a level turn (instead of being constrained to 20/sqrt(3) ≈ 11.5 m/s² per axis), potentially completing that segment significantly faster.

The Teissing paper quantifies this at >20% improvement in trajectory time. Applied to our 16.69s trajectory, this could yield a reduction of ~3.3s — from 16.69s to ~13.4s — which is below the 14s target. This is the single highest-leverage change identified in this research cycle.

### Architecture fit

Our L-BFGS optimizer operates on segment times `T_i` directly (or `log(T_i)` for positivity). We could either:

(a) **Replace the per-axis penalty** with a norm-based penalty in the existing L-BFGS objective (low risk, easy to implement).

(b) **Adopt the PMM bang-bang time allocation** as the primary optimizer and use min-snap only for smooth trajectory generation (higher risk, higher reward — would require replacing `_optimize_time_allocation`).

Option (a) is the most actionable near-term change. Option (b) would require significant refactoring but could close the full gap.

### Speed considerations

At our target 15 m/s, aerodynamic drag is non-negligible. A linear drag model with `δ ≈ 0.3` (reasonable for a 1 kg racing quad) gives a drag acceleration of `δ * v = 0.3 * 15 = 4.5 m/s²` — roughly 20% of our `max_acceleration = 20 m/s²`. This means our polynomial assumes the drone can maintain 15 m/s on long segments, but drag continuously decelerates it. The effect is that our trajectory under-predicts time on fast straight segments, causing the controller to be perpetually behind the reference — a contributor to tracking error that grows with lap speed.

---

## Actionable Takeaways

1. **Replace per-axis acceleration penalty with norm penalty in `_optimize_time_allocation`** (HIGH IMPACT). Change the soft constraint from `penalty if a_x > a_max OR a_y > a_max OR a_z > a_max` to `penalty if sqrt(a_x^2 + a_y^2 + a_z^2) > a_max`. This is a one-line change in the penalty function and directly mirrors the LTD insight. Expected benefit: 5–20% reduction in optimized trajectory time based on Teissing's >20% result.

2. **Increase `max_acceleration` to reflect actual thrust capability** (HIGH IMPACT). If our drone can sustain 3g collective thrust (as test platforms in Teissing et al. do), our `max_acceleration = 20 m/s²` (~2g) is conservative. Raising it to 25–30 m/s² combined with the norm constraint would allow the optimizer to find faster trajectories, while the actual constraint remains the norm. Benchmark carefully after each increase to detect crashes.

3. **Implement the LTD iterative algorithm for accurate per-segment per-axis limits** (MEDIUM IMPACT). Rather than a single fixed `max_acceleration`, compute per-axis limits dynamically based on which axes are dominant in each segment. For a predominantly horizontal segment, raise `a_max_horizontal` and lower `a_max_vertical` proportionally. This is the core of LTD and can be implemented in pure Python in ~50 lines.

4. **Add linear drag correction to segment time estimates** (MEDIUM IMPACT). In `_initial_time_allocation`, add a multiplicative factor `1 / (1 - δ_eff * v_avg / a_max)` to segment times for long straight segments. This prevents the optimizer from targeting a trajectory that drag makes physically unreachable, reducing chronic tracking lag on fast segments. Use `δ_eff ≈ 0.3` (horizontal) and `δ_eff ≈ 0.7` (vertical), matching Teissing's measured drag coefficients for racing-class quads.

5. **Warm-start L-BFGS with PMM bang-bang time allocation** (MEDIUM IMPACT). Compute a quick PMM trajectory (analytically, without LTD iterations) for each segment as the initial time guess instead of the current heuristic. PMM times are lower bounds on achievable time; using them as initialization puts L-BFGS closer to the true optimum and reduces the number of iterations needed for convergence. This could reduce `maxiter` requirements or improve solution quality within the same iteration budget.

6. **Consider adopting the gradient-based via-velocity optimizer as an outer loop** (HIGH IMPACT, HIGH EFFORT). Restructure the optimizer to treat via-waypoint 3D velocities as the primary decision variables (not segment times). For each candidate via-velocity set, compute segment times via PMM bang-bang (closed-form, O(1) per segment), compute total time and its closed-form gradient, and run gradient descent. This replaces finite-difference L-BFGS gradients with analytic exact gradients — O(n) instead of O(n^2) per iteration — and would dramatically accelerate convergence. Min-snap polynomials can then be constructed from the resulting per-segment state boundary conditions for smooth tracking.

7. **Test with `max_acceleration = 25.0` immediately** (LOW EFFORT, POTENTIALLY HIGH IMPACT). As a quick experiment, raise the limit and run the benchmark. The norm-based penalty change (item 1) should prevent crashes by ensuring the constraint is physically correct, not just per-axis conservative. If the benchmark shows faster trajectory without crash, keep the increase.

---

## Limitations & Caveats

**Point-mass model ignores rotational dynamics**: PMM assumes the drone can instantaneously apply any translational acceleration up to the thrust bound. In reality, changing thrust direction requires slewing the attitude (pitching/rolling), which takes 50–200 ms for aggressive maneuvers. This means PMM-generated acceleration profiles that switch sign rapidly (bang-bang reversals) are not instantaneously achievable. Our min-snap polynomials implicitly limit this via the `max_tilt_angle` and `max_jerk` parameters, which is an advantage over raw PMM. The NMPC tracker in Teissing's experiments compensates for this model-reality gap, but introduces some trajectory deviation near sharp reversal points.

**Gate aperture constraints not handled**: Teissing places waypoints at gate centers but does not constrain the trajectory to stay within the gate opening. For our 1.2m × 1.2m gates with 0.4m entry/exit offsets, this is a real concern: a PMM trajectory computed only to the gate center may pass through the frame rather than the opening. Our entry/exit waypoint approach (0.4m before and after the gate center along the gate normal) addresses this but may need to be preserved even if we adopt PMM time allocation.

**No global optimality guarantee**: gradient descent finds a local minimum. Different initializations may yield different trajectories. Teissing's r=0.6 initialization heuristic is good but not guaranteed to land in the global basin. For a 12-gate course, there is one dominant basin (fly fast through all gates), but local minima can exist near transitions between bang-bang cases.

**Gradient structure assumes independent segments**: the gradient `∂T_k/∂v_k` only couples two adjacent segments. Long-range effects (e.g., entering a straight fast by braking hard from the previous turn) are only captured through gradient accumulation over many iterations, not through any look-ahead. Our current L-BFGS jointly optimizes all segment times and can in principle find multi-gate trade-offs in a single step, though in practice its finite-difference gradients are noisier.

**Drag model accuracy**: Linear drag is a good approximation at moderate speeds. At 28 m/s (our max target), the true drag is likely between linear and quadratic. Teissing's drag coefficients (δ_x=0.28, δ_y=0.35, δ_z=0.70) are measured on a specific 1.21 kg platform; our platform may differ. A drag coefficient calibration experiment (coast test on a real or simulated drone) would improve accuracy.

**C++ implementation**: the open-source `ctu-mrs/pmm_uav_planner` is compiled C++. Using it as a Python library requires ctypes or pybind11 bindings. However, the core algorithm (bang-bang time computation + LTD + gradient descent) is simple enough to re-implement from scratch in Python in ~200 lines, which is preferable for tight integration with our existing optimizer.

**Convergence not proven**: LTD has no formal convergence proof. The step-size reduction on master-axis role changes is a heuristic that prevents oscillation but is not guaranteed to converge in all configurations. For pathological waypoint layouts (e.g., U-turns with very short segments), the algorithm may cycle. A maximum iteration cap with a fallback to the initial per-axis estimate is needed for robustness.

---

## Key Parameters / Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `ε_a` | 0.01 m/s² | LTD convergence threshold: max thrust norm error |
| `ε_t` | 1×10⁻⁴ s | Velocity optimization convergence: time change per step |
| `η` | 0.5 | Step size halving factor when master axis changes |
| `r` | 0.6 | Via-velocity initialization: heading blend factor |
| `a_Tmax` | ~5.7g = 55.9 m/s² | Max collective thrust accel on test platform (68 N / 1.21 kg) |
| `δ_x` | 0.28 | Linear drag coefficient, x-axis (m/s per m/s²) |
| `δ_y` | 0.35 | Linear drag coefficient, y-axis |
| `δ_z` | 0.70 | Linear drag coefficient, z-axis (higher: rotor downwash) |
| LTD critical instants | 4 per segment | Times when any axis switches acceleration sign |
| LTD iterations (typical) | 3–5 | Sufficient for convergence in practice |
| Max real-world accel | 3.5g (~34 m/s²) | Validated in outdoor flight |
| Max real-world speed | >28 m/s (100 km/h) | Validated in outdoor flight |
| Speedup vs sampling | ~100× | Gradient descent vs prior cone-refocusing baseline |
| Time improvement vs per-axis | >20% | Norm constraint vs per-axis constraint (same platform) |
| Drone mass (test) | 1.21 kg | Hardware context for drag/thrust parameters |
| Test platform collective thrust | 68 N | 4-motor racing quad |
