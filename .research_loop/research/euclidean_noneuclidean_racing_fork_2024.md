# Euclidean and Non-Euclidean Trajectory Optimization for Quadrotor Racing

- **URL**: https://arxiv.org/abs/2309.07262
- **Authors**: Thomas Fork, Francesco Borrelli
- **Year**: 2024 (submitted September 2023, revised July 2024)
- **Venue**: arXiv preprint (cs.RO), UC Berkeley

---

## Key Contribution

Fork and Borrelli present two distinct trajectory optimization formulations for computing minimum-time racing trajectories through gates, differing fundamentally in how vehicle position is described. The central result is a **100x reduction in compute time** versus comparable published methods (specifically Scaramuzza et al.'s CPC) while maintaining high-fidelity quadrotor dynamics.

The most important departure from prior art is that **neither approach approximates gates as waypoints**. Instead, gates are treated as spatial regions (feasible passage sets) through which the trajectory must pass, with the optimizer free to choose the precise crossing location within the opening. This is the same insight as TOGT (Qin 2024) but arrived at through a fundamentally different mathematical framework — continuous-time optimal control with finite element discretization rather than MINCO polynomials + L-BFGS.

Two contributions of comparable significance:
1. The **Euclidean approach**: a multiphase optimal control problem in global inertial coordinates, with variable-time finite elements allocated to gate pairs and gate passage enforced as spatial feasibility constraints.
2. The **non-Euclidean approach**: a curvilinear coordinate system aligned with a racetrack centerline, in which gate passage becomes simple lateral bound constraints on the (y, n) coordinates, and obstacle avoidance reduces to adding convex inequalities without any mixed-integer formulation.

---

## Technical Approach

### Problem Statement

Both methods solve a **minimum-time periodic trajectory** problem:

```
minimize   t_f  (total lap time)
subject to:
  - quadrotor rigid-body dynamics (full 6-DOF, not point mass)
  - gate traversal: trajectory passes through gate region at some time t_i
  - actuator constraints: individual rotor thrust 0 <= T_i <= T_max
  - tilt angle constraints: maximum attitude deviation
  - periodicity: start state = end state (for closed-loop racing)
```

The trajectory is parameterized using a **direct finite element (collocation) method** rather than differential flatness polynomials. Each finite element spans a gate-to-gate segment, and each element is assigned a **variable time duration** — the optimizer adjusts both the trajectory shape and the time budget per segment simultaneously.

### Gate Representation as Regions

This is the central innovation distinguishing both Fork/Borrelli methods from earlier waypoint-based planners.

**Prior (waypoint) approach**: Gate traversal is modeled as an equality constraint:
```
p(t_i) = p_gate  (position equals gate center at some crossing time)
```
This forces the drone through a single fixed point, discarding the fact that physical gates have finite opening dimensions and the drone has freedom of where within that opening to cross.

**Fork/Borrelli approach**: Gate traversal is modeled as a feasibility constraint:
```
p(t_i) in Gate_i
```
where `Gate_i` is a convex set (polygon, rectangle, or polytope) describing the gate opening in 3D space.

**In the Euclidean formulation**, the gate region is expressed as a polytope in global coordinates. At the finite element corresponding to crossing time t_i, the position state is bounded within the gate's linear inequality set `A*p <= b`, where `A` and `b` encode the gate's edges, orientation, and world-frame position. The optimizer is free to choose any crossing point satisfying these bounds.

**In the non-Euclidean formulation**, gate passage is even more naturally represented. Because the s-coordinate parameterizes progress along the centerline and (y, n) are lateral offsets, a gate becomes simply a **tightening of the lateral bounds** at the corresponding s-value. Gate opening width maps directly to the allowed range of y, and gate opening height maps to the allowed range of n. No coordinate transformation or extra constraint structure is needed — the gate geometry is baked into the coordinate system bounds at the gate s-location.

This is visible in Figure 1 of the paper: lateral coordinate limits "shrink to enforce gate passage" at each gate location, widening back out in the free-flight segments between gates.

### Euclidean Formulation

Position kinematics in the global (inertial) frame:

```
∂_t x^g = R^g_b * v^b
∂_t R^g_b = R^g_b * ω̂^b
```

where `x^g` is position in world frame, `R^g_b` is the body-to-world rotation matrix, `v^b` is body-frame velocity, and `ω̂^b` is the angular velocity skew-symmetric matrix.

Full rigid-body dynamics (Euler equations in body frame):

```
m(v̇^b + ω̂^b v^b) = F^b    (translational Newton-Euler)
I^b ω̇^b + ω̂^b I^b ω^b = K^b  (rotational Newton-Euler)
```

The four-propeller thrust model combines individual rotor thrusts T_1 through T_4 into net body-frame force F^b and torque K^b via a fixed mixer matrix that depends on rotor positions and arm lengths.

The track is divided into **L+1 finite elements** (one per inter-gate segment). Each element has a fixed number of collocation nodes but a **variable time duration** Δt_i, which is part of the decision variable set. The optimizer jointly determines trajectory states at each node and the duration of each element.

Gate constraints appear as polytope bounds at the finite element boundary corresponding to each gate crossing:

```
A_i * p(s_i) <= b_i  for each gate i
```

where `s_i` is the s-parameter (element boundary) at gate i, and (A_i, b_i) encode the gate opening geometry in world coordinates.

**Advantage of the Euclidean approach**: Straightforward extension to obstacle avoidance — static obstacles can be added as additional polytope exclusion constraints on the same finite elements. No coordinate transformation complexity.

**Disadvantage**: Initialization is harder because there is no geometric prior embedded in the coordinate system. The solver must discover the gate-traversing structure from scratch.

### Non-Euclidean (Curvilinear) Formulation

The non-Euclidean approach introduces a **Darboux frame** moving along a user-defined centerline curve x^c(s):

```
x = x^c(s) + y * e^c_y(s) + n * e^c_n(s)
```

where `s` is the along-track parameter (not required to be arc-length), `y` and `n` are lateral deviations, and `e^c_y`, `e^c_n` are the frame's lateral unit vectors.

The three curvature quantities characterizing the frame's rotation along the centerline are:

- κ^c_n: geodesic curvature  `= -(x^c_ss × x^c_s) · e^c_n / ||x^c_s||_2^3`
- κ^c_y: normal curvature    `(derived from orthogonality conditions via matrix inversion)`
- κ^c_s: torsion             `(twist of the frame about the centerline tangent)`

These curvature terms appear in the kinematic equations relating the time derivatives of s, y, n to the vehicle velocity:

```
ṡ = v · e^c_s / [(1 - κ^c_y * n - κ^c_n * y) * ||x^c_s||_2]   (12a)
ẏ = v · e^c_y + n * κ^c_s * ||x^c_s||_2 * ṡ                   (12b)
ṅ = v · e^c_n - y * κ^c_s * ||x^c_s||_2 * ṡ                   (12c)
```

**Critical constraint**: Equation (12a) has a singularity when `κ^c_y * n + κ^c_n * y = 1`. This must be avoided by the explicit inequality:

```
κ^c_y * n + κ^c_n * y <= λ < 1    (singularity avoidance, Eq. 15)
```

where λ is a tunable parameter controlling how close to the singularity the trajectory is allowed to approach. Higher λ (closer to 1) permits trajectories far from the centerline (useful for wide lateral excursions at low-curvature track sections) but risks numerical instability near high-curvature sections.

**Gate passage in curvilinear coordinates**: A gate at centerline position s = s_i has lateral bounds:

```
y_min(s_i) <= y <= y_max(s_i)
n_min(s_i) <= n <= n_max(s_i)
```

These bounds are tight at gate locations (set by the physical gate opening) and loose in between (no constraint, or wide safety margins). The optimizer naturally finds the least-cost crossing point within the gate opening, without any waypoint approximation.

**Centerline choice**: The centerline x^c(s) is not prescribed by the method — it is a design parameter that encodes a trajectory prior. The authors fit it as a spline to approximate nominal gate centers. An important result: **the centerline does not need to be arc-length parameterized**. This generalizes prior curvilinear methods (Arrizabalaga 2022, Giuseppe drone racing) which required arc-length parameterization and, as the authors show, contained geometric modeling errors in the curvature formulas.

**Advantage over prior curvilinear work**: The integral connection between Euclidean and curvilinear coordinates prevents integration error accumulation across long tracks. Previous formulations accumulated numerical drift because they used approximate local updates rather than exact global integrals.

### Actuator and Tilt Constraints

Both approaches use the same physical constraint set on the quadrotor control inputs:

**Rotor thrust bounds**:
```
0 <= T_i <= T_max   for i = 1, 2, 3, 4  (individual motor bounds)
```

**Tilt angle limit**: A constraint on maximum attitude deviation from vertical, typically expressed as a bound on the angle between the body z-axis and the gravity vector:
```
cos(θ_tilt) >= cos(θ_max)
```
or equivalently `R_body[2,2] >= cos(θ_max)`, where θ_max is the maximum allowed roll/pitch angle. This prevents excessive bank angles that would cause the drone to lose altitude uncontrollably and limits the instantaneous horizontal acceleration.

The full rigid-body model (not a point-mass or differential flatness model) means these constraints are imposed directly on the actual control inputs throughout the finite element discretization, not via a post-hoc feasibility check on polynomial derivatives. This is more conservative than the TOGT approach (which samples feasibility at discrete points) but more rigorous.

**Specific vehicle parameters** are deferred to Appendix A of the paper ("Vehicle Model Parameters"), which the HTML renders do not expose fully. Parameters include mass m, inertia matrix I^b, maximum rotor thrust T_max, arm length, and rotor geometry. The paper uses "realistic vehicle parameters" consistent with competition-class racing quadrotors.

### Comparison Between the Two Approaches

| Aspect | Euclidean | Non-Euclidean |
|--------|-----------|---------------|
| State space | Global (x, y, z, R) | Curvilinear (s, y, n, R) |
| Gate constraints | Polytope `Ap <= b` in world frame | Lateral bound tightening at s_i |
| Initialization | No geometric prior | Centerline encodes prior |
| Obstacle avoidance | Direct polytope exclusion | Convex bounds on (y, n) |
| Singularity risk | None | Must enforce `κy·n + κn·y < λ` |
| Applicable to | Any environment | Environments with good centerline |
| Tracks corrected errors in | Prior fixed-wing methods | Arrizabalaga 2022, Giuseppe drone racing |
| Convergence | Lower — less prior | Higher — centerline warm-starts |

### Comparison to the Dual-Waypoint Approach

The "dual-waypoint" approach (used in some prior planners) approximates each gate as two waypoints — one on the approach side and one on the exit side of the gate plane. This is better than a single-waypoint approximation because it implicitly enforces gate-normal-aligned crossing (the direction from waypoint 1 to waypoint 2 is approximately the gate normal), but still has several deficiencies:

1. **No position freedom within gate**: Both waypoints are fixed at specific positions (e.g., 0.5 m before and after the gate center). The drone must pass through these specific points, discarding the available lateral freedom within the gate opening.

2. **Velocity direction is over-constrained**: The trajectory must be heading from waypoint 1 to waypoint 2 at the crossing, which pins the exit velocity direction. For gates at oblique angles to the racing line, this forces unnecessary turns.

3. **Increased problem size**: Two waypoints per gate doubles the number of intermediate constraints, making the NLP larger.

Fork and Borrelli's region approach eliminates all three issues. The gate region formulation provides freedom within the gate opening, does not constrain the exit velocity direction (the optimizer chooses), and represents each gate as a single constraint set rather than two points.

### Computational Method

Both approaches use a **direct multiple shooting** or **direct collocation** discretization:
- The trajectory over each inter-gate segment is discretized into multiple collocation nodes.
- Each element has variable time duration Δt_i.
- The full NLP (nonlinear program) is solved by a standard NLP solver (likely IPOPT or similar).

The **100x speedup** over CPC is attributed to:
1. The finite element allocation strategy — variable element sizes adapted to the complexity of each segment.
2. The centerline encoding (non-Euclidean) that provides a high-quality initial guess, reducing NLP iterations.
3. Avoidance of the mixed-integer structure that some prior methods needed for obstacle avoidance.

Approximate compute times based on the "100x faster" claim and the paper's reference baseline (~30 minutes for CPC on a racetrack scenario):
- Comparable CPC baseline: ~30 minutes (1800 s)
- Fork/Borrelli approach: ~18 seconds

---

## Results

### Performance on Three Test Scenarios

**Racetrack (oval-shaped, multiple gates)**:
- Achieves comparable lap times to CPC at ~100x lower computation cost.
- Both Euclidean and non-Euclidean approaches converge; non-Euclidean has better initial convergence due to centerline prior.

**Figure-eight trajectory**:
- Periodic trajectory through a figure-8 gate pattern.
- Validates the periodicity constraint formulation.
- Non-Euclidean approach handles the topology (two loops sharing a common crossing point) cleanly.

**Obstacle avoidance (static barriers)**:
- Unique capability not demonstrated by comparable methods.
- Static obstacles are modeled as convex polytope exclusion zones.
- In the non-Euclidean formulation, obstacles simply add lateral bound tightening at corresponding s-values — no MILP decomposition needed.
- Multiple obstacles can be added with linear constraint overhead.

### Key Quantitative Claims

- **100x faster** computation than CPC (Scaramuzza et al.'s prior method)
- **Improved solver convergence** — lower NLP failure rate vs CPC
- **High-fidelity dynamics** — full rigid-body model, not point mass
- **Gate region freedom** — optimizer exploits lateral gate width, routing through gate corners when shorter than gate center

Specific absolute lap times and computation times are not extracted from the HTML renderings, but the relative performance is well-documented.

---

## Relevance to Our System

Our system (`planning/trajectory_optimizer.py`) uses minimum-snap polynomial segments with gate centers as hard waypoints. The Fork/Borrelli paper is directly relevant to three aspects of our planning stack:

### 1. Gate Parameterization

Our `TrajectoryOptimizer._generate_trajectory()` targets gate centers as fixed waypoints. This means we always pass through the center of each gate, never cutting corners even when the optimal racing line would pass through a corner of a large gate opening. On a competition track with 0.5–1.0 m gate openings and a drone with ~0.35 m collision radius, the effective corridor through each gate is only ±0.15–0.33 m from center. The Fork/Borrelli region approach would let the optimizer find crossing points that minimize path curvature — which can meaningfully reduce the trajectory length and tracking error on tight tracks.

**Concrete benefit**: On a gate where the optimal racing line passes 0.2 m off-center (within the safe opening), forcing center-passage adds unnecessary curvature into the adjacent segments. For 12 gates, these small deviations compound into measurable lap time differences.

### 2. Dynamic Feasibility via Direct Collocation vs. Polynomial Sampling

Our min-snap framework enforces dynamics constraints indirectly — the polynomial is fit to minimize snap (4th derivative), and then we check actuator limits post-hoc via `_check_feasibility()`. This is the same gap the TOGT paper addresses via cubic penalty functions. Fork/Borrelli use direct collocation, which enforces dynamics at every collocation node throughout the solve. The result is a trajectory that is guaranteed feasible at all collocation points, not just at polynomial evaluation samples.

For our use case, the practical implication is that our polynomial trajectories may occasionally violate thrust limits at intermediate points between segment boundaries, even if the endpoints are feasible. This can cause controller saturation and tracking error spikes, especially near high-curvature gates.

### 3. Curvilinear Coordinate Frame for Gate-Relative Control

The non-Euclidean approach's (s, y, n) coordinate system is directly applicable to our gate-relative control problem. Currently, `race_pipeline.py` tracks the trajectory in global ENU/NED coordinates. A gate-relative frame (with s along the trajectory, y/n as lateral deviations) would make it easier to:
- Define gate passage as a lateral constraint on (y, n) rather than a 3D position proximity check
- Switch control modes near gates (tighten lateral gain) vs. between gates (tighten longitudinal gain)
- Naturally detect gate passage as a sign change in s (passing through the gate s-location)

This is architecturally more sophisticated than our current `_check_pass_through()` plane-intersection check, but would improve robustness for non-axis-aligned gates.

### 4. Periodic Trajectory for Loop Racing

The periodicity constraint in Fork/Borrelli (start state = end state) is relevant if the competition includes multiple laps. Our current system generates a one-shot trajectory from start to last gate. If we need to fly multiple laps, the trajectory would need to be extended or wrapped — the Fork/Borrelli formulation handles this naturally.

---

## Actionable Takeaways

### High Priority

**1. Model gates as lateral bounds, not point waypoints**

In `trajectory_optimizer.py`, replace the fixed gate-center waypoints with gate-region constraints. For rectangular gates of width w and height h, allow crossing offsets (dy, dn) subject to `|dy| < w/2 * safety_margin` and `|dn| < h/2 * safety_margin`. Even a simple post-processing step that shifts the waypoint off-center to minimize segment curvature would capture some of this benefit:

```python
# For each gate, compute the optimal lateral offset within the gate opening
# that minimizes the curvature of the incoming + outgoing segments
gate_center = gate.position
approach_dir = normalize(gate_center - prev_waypoint)
exit_dir = normalize(next_waypoint - gate_center)
# Optimal offset minimizes |approach_angle - exit_angle| (straightens the path)
offset = compute_racing_line_offset(approach_dir, exit_dir, gate.width, gate.height)
waypoints[i] = gate_center + offset
```

**2. Enforce singularity avoidance in curvilinear coordinates**

If we adopt a curvilinear frame for gate proximity detection, we must ensure `κ_y * n + κ_n * y < λ < 1`. For our track geometries, this means keeping the drone within `λ / max(|κ|)` meters of the centerline. For typical racing tracks with curvature radii > 2 m and lateral deviations < 0.5 m, λ = 0.9 provides comfortable margin.

**3. Use variable-time finite elements for segment time allocation**

Our `_optimize_time_allocation()` computes times with a fixed heuristic (distance / nominal_speed + curvature_bonus). The Fork/Borrelli approach makes times free variables in the same solve, letting the optimizer discover the ideal time distribution. This is the most impactful single change for reducing lap time. Implementing it requires moving from separate time-then-trajectory optimization to a joint NLP — a significant refactor but with measurable gains.

### Medium Priority

**4. Add post-gate extension waypoints to avoid trajectory endpoint stall**

Consistent with the gate-aware online planning paper's finding: never end a trajectory at a gate. Add a waypoint 2–3 m past gate-12 along its exit normal to ensure non-zero velocity at crossing. Fork/Borrelli's formulation never has this issue because gate crossing is a pass-through constraint, not a segment endpoint.

**5. Consider the non-Euclidean frame for gate proximity switching**

Near each gate (within 2 m in s), tighten lateral tracking gain and reduce longitudinal gain. Between gates, invert this weighting. The (s, y, n) frame makes this switching natural and avoids the awkward 3D proximity checks in our current `_update_gate_proximity()`.

### Lower Priority

**6. Obstacle avoidance via curvilinear lateral bounds**

If the competition course has static obstacles or safety corridors, the non-Euclidean formulation handles these as simple tightening of y and n bounds — no convex decomposition or safe flight corridors needed. This would be the cleanest way to add obstacle constraints if course information is provided pre-race.

**7. Periodicity for multi-lap races**

If the competition requires N laps, constrain the trajectory endpoint state to equal the start state. Fork/Borrelli's formulation supports this directly.

---

## Limitations & Caveats

1. **No implementation code provided**: The paper is theoretical; no open-source implementation is available. Reproducing the non-Euclidean formulation requires implementing the Darboux frame kinematics, singularity constraints, and full NLP from scratch.

2. **Centerline dependence**: The non-Euclidean approach's quality depends on centerline choice. A poor centerline (high curvature, misaligned with the actual optimal path) forces the optimizer to operate near the singularity `κy·n + κn·y → 1` and degrades convergence. For our 12-gate track, the gate centers themselves make a reasonable centerline, but this requires pre-specifying the track layout.

3. **High-fidelity dynamics increase NLP size**: The full rigid-body model has more states (6-DOF vs. 3-DOF point mass) and more constraints (per-motor thrust bounds vs. collective thrust bound). This makes each NLP evaluation more expensive. The 100x speedup relative to CPC is impressive, but the absolute computation time is still measured in tens of seconds — not real-time.

4. **Not real-time**: The method is an offline trajectory planner. It does not handle runtime disturbances, state estimation errors, or gate position corrections (e.g., from EKF updates). A separate online controller (e.g., geometric MPC) must track the precomputed trajectory. This is compatible with our architecture.

5. **No perception modeling**: Gate visibility and EKF observability are not considered. The planner can generate trajectories that pass through gates at angles that make PnP estimation degenerate (e.g., nearly edge-on approach). A perception-aware extension would be needed.

6. **Tilt angle limits not quantified**: The paper mentions tilt angle constraints but does not specify the numerical limit used in experiments. For our drone, typical competition-class limits are 45–70° from vertical. Using a too-conservative limit (e.g., 30°) would reduce achievable lateral acceleration and increase lap times unnecessarily.

7. **Singularity avoidance parameter λ**: The parameter λ in the singularity constraint `κy·n + κn·y <= λ` is not quantified in the paper. Its optimal value depends on track curvature and maximum allowable lateral deviation. Too-small λ excessively constrains the lateral deviations; too-large λ risks numerical issues. Tuning this parameter for a specific track requires experimentation.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Formulation type (E) | Direct collocation, variable-time elements | Euclidean method discretization |
| Formulation type (NE) | Curvilinear (s, y, n) + direct collocation | Non-Euclidean method discretization |
| Objective | Minimize total lap time t_f | Both approaches |
| Periodicity | p(0) = p(t_f), v(0) = v(t_f) | Closed-loop racing constraint |
| Rotor thrust | 0 <= T_i <= T_max (per motor) | From 4-propeller model |
| Tilt angle | θ <= θ_max (not numerically specified) | Attitude constraint |
| Curvature terms | κ^c_n, κ^c_y, κ^c_s | Darboux frame curvatures |
| Singularity margin | κ^c_y * n + κ^c_n * y <= λ < 1 | Non-Euclidean constraint (Eq. 15) |
| Singularity param λ | Not specified (tunable per track) | Proximity to coordinate singularity |
| Centerline param | Arbitrary (not arc-length required) | Non-Euclidean approach |
| Compute speedup | ~100x vs CPC baseline | Key performance claim |
| Baseline CPC compute | ~30 min (estimated from "100x" claim) | Reference method |
| Estimated approach compute | ~18 s (estimated from "100x" claim) | Fork/Borrelli compute |
| Gate constraint type (E) | Polytope: A*p <= b at element boundary | Euclidean gate passage |
| Gate constraint type (NE) | Lateral bound tightening: y_min(s_i) <= y <= y_max(s_i) | Non-Euclidean gate passage |
| Finite element allocation | Variable element size per segment | Adaptive to segment complexity |
| Time per element | Free variable Δt_i | Jointly optimized with trajectory |
| Test scenarios | Racetrack, figure-eight, obstacle field | Three validation cases |
| Obstacle modeling (NE) | Convex bounds on (y, n) | No MILP required |
| Dynamics model | Full 6-DOF rigid body | Not point mass or flat output |
| Vehicle params | Appendix A (specific values not exposed in HTML) | Realistic competition quadrotor |
| Corrects errors in | Arrizabalaga 2022, Giuseppe drone racing | Prior curvilinear methods |
