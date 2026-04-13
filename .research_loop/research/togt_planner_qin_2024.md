# Time-Optimal Gate-Traversing Planner (Qin 2024)

- **URL**: https://arxiv.org/abs/2309.06837
- **Authors**: Chao Qin, Maxime S.J. Michet, Jingxiang Chen, Hugh H.-T. Liu
- **Year**: 2024
- **Venue**: ICRA 2024

---

## Key Contribution

The central contribution is a trajectory optimization framework that plans **time-optimal trajectories through race gates by treating gate openings as extended feasible regions** rather than single waypoint targets. Prior planners (e.g., CPC) modeled gates as points or very small waypoints, forcing the drone through a single pass-through pose. TOGT instead specifies a gate as a set of valid traversal positions (a ball or convex polygon in 3D space), then jointly optimizes *where* the drone crosses the gate and *how long* each segment takes. This unlocks more aggressive cornering and shorter lap times, especially on complex or multi-constraint gates.

A secondary contribution is extreme scalability: TOGT solves 61-gate tracks in ~5 seconds, while the prior CPC baseline exceeds 8 hours and fails to converge.

---

## Technical Approach

### Problem Formulation

The task is formulated as a minimum-time optimal control problem:

```
minimize  t_f
subject to:
  - initial and terminal state constraints
  - quadrotor rigid-body dynamics (13-state: position, quaternion, velocity, body rates)
  - single-rotor thrust bounds: f_min <= f_i <= f_max (per motor)
  - angular rate limits: |omega| < omega_max
  - gate traversal constraints: p(t_i) in Gate_i  for each gate i
```

The key difficulty is that gate traversal constraints are time-triggered (existence of crossing time t_i) and space-constrained simultaneously.

### Trajectory Parameterization: MINCO Polynomials

Rather than discretizing the control problem, the authors use **MINCO (Minimum Control) trajectory functionals**: degree-5 piecewise polynomials in flat output space `y(t) = [p^T, psi]^T` (position + yaw). Differential flatness of the quadrotor means that all states and control inputs can be recovered analytically from the flat output and its derivatives up to order `s`:

```
x = Psi_x(y^[s-1])
u = Psi_u(y^[s])
```

This eliminates the system dynamics and boundary condition constraints from the optimization variables entirely — the polynomial coefficients encode the full trajectory, and feasibility checks are purely algebraic evaluations of polynomial derivatives.

**Why degree 5?** Degree 5 (quintic) polynomials have 6 coefficients per dimension per segment, which is the minimum degree to impose position, velocity, and acceleration boundary conditions at both endpoints of each segment — a natural match for differentially flat quadrotor planning.

### Gate Parameterization: Two Primitive Types

This is the core innovation. Every gate is modeled as a **spatial region** rather than a point.

**1. Ball Gate**
A spherical feasible region:
```
G_ball = { p in R^3 | ||p - p_w||_2 <= delta }
```
where `p_w` is the gate center and `delta` is the radius. These represent circular or near-circular openings, and also approximate spherical margins around waypoints.

**2. Convex Polygon / Polyhedron Gate**
A linear inequality set:
```
G_poly = { p in R^3 | A*p <= b }
```
This directly models square gates, rectangular gates, pentagonal gates, tunnel cross-sections, and any other convex opening. The matrix `A` and vector `b` are derived from the gate's geometry and orientation in world frame.

**Complex Gates by Composition**
Dive gates, tilted gates, or directional gates are modeled as ordered sequences of primitives — e.g., a dive gate requires passing through one polyhedron (entrance face) then another (exit face) in order. This subsumes arbitrary gate geometry.

### Decision Variables After Decomposition

The trajectory is segmented into `L+1` pieces for `L` gates. The optimization variables are:

- `P in R^(3xL)`: one traversal waypoint per gate (the actual crossing location within the gate opening)
- `T in R^+(L+1)`: time allocation for each segment between waypoints

Given `P` and `T`, the MINCO polynomial is uniquely determined (up to boundary derivatives). The original constrained problem becomes:

```
minimize  sum(T) + I_{T_hat(P)}(T)
```

where `I` is an indicator/penalty function that equals zero when dynamic feasibility holds for the given `P, T` assignment, and is large otherwise.

### Constraint Elimination: Smooth Surjections

A key numerical trick: rather than handling inequality constraints `p_i in Gate_i` and `T_j > 0` explicitly (which requires constrained optimization or projection), the authors **reparameterize with smooth surjections** — differentiable mappings from unconstrained variables to the feasible set.

For ball gates:
```
g_B(d) = p_w + [ 2*delta*d / (d^T*d + 1) ]_3
```
This stereographic-projection-like map sends any `d in R^3` to a point strictly inside the ball. As `||d|| -> inf`, it approaches the boundary.

For polygon gates:
```
g_P(d) = o + V * [ [d]^2 / (d^T*d)^2 ]_v
```
where `o` is a reference point inside the polygon, `V` encodes vertex structure, and the map guarantees the output lies within the polygon for all inputs `d`.

These surjections convert the **constrained** problem over `(P, T)` into an **unconstrained** problem over `(D, K)` — making L-BFGS directly applicable without any projected gradient or augmented Lagrangian outer loop.

### Dynamic Feasibility: Penalty via Sampling

Dynamic feasibility (thrust bounds + rate limits) is not analytically eliminated. Instead, violations are penalized via a cubic integral over sampled trajectory points:

```
I_{T_hat(P)}(T) ≈ sum_{i=1}^{L+1} sum_{j=0}^{kappa_i} max[ h_Psi(y^[s](t_{i-1} + j*Delta_t_i)), 0 ]^3 * Delta_t_i
```

where `kappa_i` controls sampling density per segment and `h_Psi` encodes the motor thrust and rate constraint residuals. The cubic exponent `3` ensures: (1) the gradient is zero at constraint boundary (smooth), (2) the function is zero when constraints are satisfied, and (3) large violations are heavily penalized.

### Time Allocation Strategy

Segment times `T_j` are jointly optimized with waypoint positions. There is no fixed heuristic for how long each segment should be — the L-BFGS solver freely adjusts segment durations to minimize total lap time while keeping feasibility penalties low.

**Curvature-implicit allocation:** Short, high-curvature segments naturally require more time to stay within thrust limits; the penalty function captures this because high-curvature polynomials require large angular accelerations that push actuator constraints. The optimizer therefore allocates more time to sharp turns implicitly, without a separate curvature heuristic.

**Initial guess sensitivity:** The paper acknowledges that L-BFGS convergence quality depends on initialization. They initialize segment times proportionally to Euclidean distance between gates — a simple but effective warm start. Gate traversal points `D` are initialized to zero (mapping to gate centers via the surjection).

### Solver

L-BFGS (limited-memory BFGS quasi-Newton) is used for the final unconstrained optimization. Gradients of the objective with respect to `D` and `K` are computed analytically via chain rule through the polynomial evaluation and surjection maps. No finite-difference gradients are used.

---

## Results

### Simulation (7-gate square track, QuadA: 0.85 kg)

| Method    | Compute Time | Lap Time |
|-----------|-------------|----------|
| CPC       | 466 s       | 6.85 s   |
| TOGT      | 0.36 s      | 7.53 s   |
| TOGT-WP   | 0.14 s      | 8.45 s   |

On small tracks, CPC achieves slightly shorter lap times because it uses bang-bang thrust profiles (true time-optimal control) while TOGT uses smooth polynomials. TOGT sacrifices ~10% lap time for a 1300x speedup in planning.

### Scalability (increasing gate count)

| Gates | CPC compute | TOGT compute |
|-------|-------------|--------------|
| 7     | 466 s       | 0.36 s       |
| 31    | >8 hours    | 1.87 s       |
| 61    | DNF         | 5.18 s       |

At 31+ gates, TOGT actually achieves **shorter lap times than CPC** (by ~1%) because it can optimally exploit gate geometry (crossing through corners of the gate to save distance) while CPC is forced to pass through gate centers.

### Real-World Experiments (QuadB: 1.05 kg, motion capture lab, 4x4x2 m)

**4-gate rectangular track (10 gates total with ball regions):**
- TOGT lap time: 5.96 s
- TOGT-WP lap time: 6.14 s
- Tracking RMSE: 0.3 m (MPC at 100 Hz)

**Tunnel course (4 square tunnels, 2 m depth each):**
- Completion time: 5.56 s
- Peak speed: 6.83 m/s
- Average tracking error: 0.15 m
- Demonstrated optimal cornering through confined passages

**Dive gate challenge:**
- Peak diving speed: 2.90 m/s
- Validated multi-polyhedron ordered gate constraints

---

## Relevance to Our System

Our system uses `planning/trajectory_optimizer.py` (min-snap polynomials) and `planning/racing_line.py` (lateral offset + curvature-aware speed profiling). TOGT directly addresses two weaknesses:

1. **Gate parameterization as waypoints**: Our current planner targets gate centers. TOGT shows that optimizing the traversal *point within* the gate opening can reduce lap times by several percent and enable more aggressive cornering lines.

2. **Segment time allocation**: Our `racing_line.py` uses curvature-aware speed profiling as a heuristic, while TOGT jointly optimizes segment times as free variables in the same L-BFGS solve. This is more principled and converges to better solutions on tracks with complex geometry.

3. **Tunnel/complex gate support**: Our `gate_sequencing/sequencer.py` detects gates by center position. TOGT's polyhedron model would handle gates with directional constraints or non-circular openings more faithfully.

The MINCO polynomial basis is directly compatible with our existing min-snap framework — degree-5 polynomials with matched boundary conditions are equivalent to min-snap piecewise polynomials.

---

## Actionable Takeaways

### High Priority

**1. Curvature-Aware Segment Time Initialization**
Replace the current distance-proportional time allocation in `trajectory_optimizer.py` with a curvature-weighted scheme:

```python
# Estimate curvature at each gate using the angle between consecutive gate-to-gate vectors
# Allocate more time to high-curvature segments
segment_angle = angle_between(v_prev, v_next)  # turning angle at gate i
base_time = distance / nominal_speed
curvature_bonus = k_curv * segment_angle  # k_curv ~ 0.1-0.3 s/rad
T_i = base_time + curvature_bonus
```
This directly mimics what TOGT's optimizer discovers implicitly.

**2. Gate Traversal Point Optimization**
Instead of always targeting gate centers, add a small 2D offset within the gate plane as a free variable. For a rectangular gate of width `w` and height `h`, allow offsets `(dx, dz)` with `|dx| < w/2 * margin` and `|dz| < h/2 * margin`. Optimize these offsets to minimize total trajectory length or snap cost.

**3. Ball-Gate Safety Margins**
When modeling gates as ball regions with radius `delta` (e.g., `delta = 0.3 m` for safety on a 1.0 m gate), the optimizer can cut corners more aggressively. Our current system likely uses a single target point with no margin.

### Medium Priority

**4. Smooth Surjection for Gate Constraints**
If we add gate traversal point optimization, use the ball-gate surjection `g_B(d) = p_w + [2*delta*d / (d^T*d + 1)]_3` to keep the problem unconstrained. This avoids projection steps and enables direct gradient descent.

**5. Joint Time + Position Optimization**
Bundle segment times into the same scipy optimizer call as trajectory coefficients. Currently, time allocation is a pre-processing step separate from polynomial fitting. Making them jointly optimizable (even with a simple gradient-free search over `T`) would likely improve lap times.

### Lower Priority

**6. Polyhedron Gate Modeling**
For square gates in the competition, model the gate as `Ap <= b` with `A, b` derived from the gate's 4 edges projected into world frame. This ensures the trajectory doesn't clip corners.

---

## Limitations & Caveats

1. **Smooth polynomials are not bang-bang optimal**: On short tracks with few gates, bang-bang thrust profiles (true time-optimal) can be ~10% faster than smooth polynomial trajectories. TOGT trades this for tractability.

2. **Local optima sensitivity**: L-BFGS finds local minima. The quality of the initialization (distance-proportional `T`, zero-offset `D`) significantly affects the solution. Complex tracks with many local minima may need multi-start or annealing.

3. **Sampling-based feasibility**: Dynamic feasibility is checked at discrete time samples. A finely-tuned trajectory could violate constraints between samples. The paper uses `kappa_i` (sampling density) as a hyperparameter — denser sampling is safer but slower.

4. **No perception constraints**: The planner does not model camera field-of-view or gate visibility. Our system needs gate visibility for EKF updates; a perception-aware extension (e.g., ETH 2025 paper) would be needed to enforce that.

5. **Yaw optimization is limited**: The paper optimizes yaw (`psi`) as part of the flat output, but in practice yaw is often fixed to face the next gate. A more principled yaw planning strategy could improve gate detection.

6. **Motion capture ground truth**: Real-world experiments use a 100 Hz motion capture system for state feedback. Translating to on-board VIO adds noise and latency that can degrade tracking error significantly beyond the reported 0.15–0.3 m.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Polynomial degree | 5 | MINCO flat-output polynomial order |
| QuadA mass | 0.85 kg | Simulation quadrotor |
| QuadB mass | 1.05 kg | Real-world quadrotor |
| Arm length | 0.125–0.15 m | Motor-to-center distance |
| Max motor thrust | 6.375–6.88 N | Per-motor limit |
| Max roll/pitch rate | 8–15 rad/s | Body rate limit |
| Max yaw rate | 3 rad/s | Body rate limit (yaw) |
| Ball gate radius delta | 0.3 m | Safety margin on square gates |
| Drone frame radius | ~0.34 m | Collision footprint |
| MPC control rate | 100 Hz | Trajectory tracking frequency |
| State estimation rate | 100–200 Hz | MoCap + VIO fusion |
| TOGT compute (7 gates) | 0.36 s | L-BFGS wall time |
| TOGT compute (61 gates) | 5.18 s | L-BFGS wall time |
| CPC compute (7 gates) | 466 s | Comparison baseline |
| TOGT lap time vs CPC delta | +10% on small tracks, -1% on 31+ gates | Lap time tradeoff |
| Penalty exponent | 3 (cubic) | Constraint violation penalty order |
| Time init strategy | distance / nominal_speed | Segment time warm start |
| Gate offset init | D = 0 (gate centers) | Traversal point warm start |
