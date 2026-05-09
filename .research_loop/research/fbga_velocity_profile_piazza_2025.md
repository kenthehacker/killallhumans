# FBGA: Real-time Velocity Profile Optimization for Time-Optimal Maneuvering with Generic Acceleration Constraints
- **URL**: https://arxiv.org/abs/2509.26428
- **Authors**: Mattia Piazza, Mattia Piccinini, Sebastiano Taddei, Francesco Biral, Enrico Bertolazzi
- **Year**: 2025
- **Venue**: IEEE RA-L 2026 (vol. 11, no. 2, pp. 1674-1681)
- **Code**: https://github.com/DRIVEWISE/FBGA (C++, CC BY-NC-ND 4.0)

## Key Contribution

FBGA is a forward-backward algorithm that computes time-optimal velocity (speed) profiles along a prescribed spatial path, subject to **arbitrary, non-convex acceleration constraints** (the "g-g-v diagram"). It closes a critical gap in the literature: prior methods either (a) used optimal control / QSS solvers that are too slow for online use (seconds to minutes), or (b) used fast forward-backward schemes limited to conservative box-shaped acceleration bounds. FBGA achieves both: it handles diamond-shaped, speed-dependent, non-convex acceleration envelopes while running 2-3 orders of magnitude faster than optimal control baselines. On five racetracks with two vehicle classes (car and motorcycle), lap times match OCP solutions within 0.11%-0.36%.

The key insight is that the classic forward-backward integration scheme can be extended to generic acceleration constraints by introducing a **signed distance function** from the g-g-v envelope boundary, and using root-finding (rather than closed-form expressions) to find the maximum feasible acceleration at each discretized path segment. This makes the algorithm agnostic to the constraint shape — it only needs to evaluate the constraint functions, not assume convexity or differentiability.

## Technical Approach

### Problem Formulation

The problem is: given a fixed spatial path (curvilinear abscissa `s` with curvature `kappa(s)`), find the longitudinal acceleration `a_x(s)` that minimizes total traversal time `T = integral(ds / v_x(s))`, subject to:

- **Double-integrator dynamics**: `ds/dt = v_x`, `dv_x/dt = a_x`
- **Lateral acceleration bounds**: `Gamma_y^-(v_x) <= a_y = kappa * v_x^2 <= Gamma_y^+(v_x)`
- **Longitudinal acceleration bounds**: `Gamma_x^-(a_y, v_x) <= a_x <= Gamma_x^+(a_y, v_x)`
- **Boundary conditions**: `v_x(0) = v_ini`, `s(T) = L`

The coupling between lateral and longitudinal bounds (via the g-g-v diagram) is the key challenge. For a drone, the g-g-v captures thrust limits, drag, and the coupling between lateral and longitudinal acceleration through tilt angle constraints.

### Change of Variable: Space-Domain

The time-optimal problem is reformulated in the spatial domain: minimize `integral_0^L ds/v_x(s)` subject to `v_x'(s) * v_x(s) = a_x(s)`. Under constant `a_x` within a segment, the analytical solution is:

```
v_x(s) = sqrt(2 * s * a_x + v_ini^2)
T_segment = (-v_0 + sqrt(2 * a_x * L + v_0^2)) / a_x
```

### Signed Distance Function D±

A crucial building block is Algorithm 1: a signed distance function `D±(a_x, a_y, v_x)` that returns:
- Negative values when the point is inside the g-g-v envelope (feasible)
- Zero on the boundary
- Positive values outside (infeasible)

It works by: (1) clipping `a_y` to the lateral bounds `Gamma_y±(v_x)`, (2) computing longitudinal bounds `[a_x_min, a_x_max]` at the clipped `a_y` and `v_x`, (3) applying a pyramid-shaped function `Lambda(x, y) = max(x-1, -1-x, y-1, -1-y)` composed with a normalization `Phi(x, x_min, x_max) = 2*(x - x_min)/(x_max - x_min) - 1` that maps the constraint range to [-1, +inf). This is a general-purpose feasibility test that works for any shape of g-g-v envelope.

### Three-Phase Algorithm

**Phase 1 — VELSAT (Maximum Speed from Lateral Bounds):**
For each path point, compute the maximum speed `v_sat` such that lateral acceleration `kappa * v_sat^2` stays within `Gamma_y±(v_x)`. This is a root-finding problem: solve `H(V) = kappa * V^2 - Gamma_y±(V) = 0` on `[0, v_max]`. This gives an upper bound on speed at each point.

**Phase 2 — Forward Pass:**
Iterate forward through segments. At each segment, given initial speed `v_0`:
1. Compute the clipped lateral acceleration and corresponding longitudinal bounds `[a_x_min, a_x_max]`.
2. Check if maximum acceleration `a_x_max` keeps the end-state inside the g-g-v envelope using `D±`.
3. If yes, use `a_x_max` (bang-bang: accelerate as hard as possible).
4. If not, use root-finding to find the largest `a_x` that places the segment endpoint exactly on the g-g-v boundary.
5. Clamp final speed to `min(v_sat, v_from_integration)`.
6. If no feasible `a_x` exists (speed must drop below what forward integration can achieve), mark segment as invalid — the backward pass will fix it.

**Phase 3 — Backward Pass:**
Iterate backward through segments. For each segment:
1. If the forward solution is still valid (end speed reachable from start), keep it.
2. If the average acceleration `a_x_avg = (v_1^2 - v_0^2) / (2L)` is feasible inside the g-g-v envelope at both endpoints, accept it.
3. Otherwise, apply maximum deceleration `a_x_min` or find the deceleration that places the start state on the g-g-v boundary via root-finding.

The backward pass always yields a feasible solution and overrides the initial velocity if infeasible.

### Differences from Classic TOPP-RA

| Aspect | TOPP-RA | FBGA |
|--------|---------|------|
| Constraints | Second-order cone (convex) | Arbitrary g-g-v (non-convex OK) |
| Solver | LP/SOCP per grid point | Root-finding per segment |
| Phase space | (s, s_dot) | (s, v_x) with explicit a_x |
| Formulation | Path parameterization s(t) | Direct velocity planning |
| Jerk limits | Some extensions support it | Not yet (future work) |
| Optimality proof | Yes (for convex) | Open question for non-convex |

FBGA's key advantage over TOPP-RA is that it does not require convexifying the acceleration constraints. For a drone, the thrust envelope coupling lateral and longitudinal acceleration is naturally non-convex (especially at high speeds where drag is significant), and FBGA handles this natively.

### Discretization

The path is uniformly sampled into N points (1m spacing typical for cars). Within each segment, `a_x` is assumed piecewise constant — identical to how direct optimal control methods discretize. The algorithm is robust to coarse discretization: reducing from 300 to 100 mesh points on a 300m segment changes maneuver time by only 0.3%, while reducing CPU time from 0.177ms to 0.062ms.

## Results

### Lap Time Accuracy (Table I)
Five circuits, two vehicle classes (car, motorcycle), comparing FBGA vs. OCP_bench (indirect optimal control via Pins solver):

| Circuit | Vehicle | OCP_bench [s] | FBGA [s] | Delta | Delta % |
|---------|---------|--------------|----------|-------|---------|
| Catalunya (4.66km) | Car | 112.461 | 112.204 | -0.257 | 0.23% |
| Catalunya | Motorcycle | 105.381 | 104.999 | -0.382 | 0.36% |
| Sepang (5.52km) | Car | 135.480 | 135.086 | -0.394 | 0.29% |
| Palm Beach (3.17km) | Car | 79.963 | 79.869 | -0.094 | 0.12% |

FBGA slightly *underestimates* lap times (by 0.094-0.449s) because it allows instantaneous traction-braking transitions (no jerk limit).

### Computational Performance (Table I, M2 Max)
| Circuit | Vehicle | OCP_bench [ms] | FBGA [ms] | Speedup |
|---------|---------|---------------|-----------|---------|
| Catalunya | Car | 8017 | 9.86 | 813x |
| Catalunya | Motorcycle | 694 | 3.31 | 210x |
| Palm Beach | Motorcycle | 706 | 2.39 | 295x |

### Short-Horizon (300m, Table II)
| Method | Mesh points | Time [s] | CPU [ms] |
|--------|------------|----------|----------|
| OCP_bench | 300 | 8.7936 | 97.533 |
| FBGA | 300 | 8.7765 | 0.177 |
| FBGA | 100 | 8.8029 | 0.062 |

Speedup: 550x at same mesh, 1573x at coarse mesh, with 0.19% time difference.

### Sensitivity
CPU time scales linearly with N. Lap time changes by only 0.06% when increasing mesh from ~4000 to ~40000 segments. Time split: ~20% VELSAT, ~60% Forward, ~20% Backward.

### Comparison with Double-Track Model
On Catalunya with a high-fidelity double-track car model (Pacejka tires, real engine curves, LSD): FBGA lap time 113.96s vs. MLT-DT 114.05s, a difference of only 73ms.

## Relevance to Our System

Our drone racing pipeline currently uses:
1. **L-BFGS trajectory optimization** (min-snap polynomials) with heuristic time allocation
2. **Post-optimization selective compression** (`_compress_times`) that heuristically shrinks "easy" segments
3. **Curvature-aware speed profiling** in `racing_line.py` using approximate speed limits at high-curvature points

This is exactly the problem FBGA solves, but properly. Our current approach has several weaknesses that FBGA would address:

1. **Heuristic segment timing**: We use `_initial_time_allocation` (distance-based) followed by L-BFGS optimization with a global objective. FBGA would give us the *provably near-optimal* time allocation for each segment, given the drone's actual acceleration envelope.

2. **Box constraints are conservative**: Our `DroneConstraints` uses independent `max_velocity=15`, `max_acceleration=20` limits. In reality, the drone's acceleration capability depends on speed (drag) and the coupling between lateral/longitudinal acceleration (tilt angle). A g-g-v diagram captures this coupling. At high speed, available lateral acceleration decreases due to drag consuming thrust budget.

3. **The compression heuristic is fragile**: `_compress_times` selectively compresses "easy" segments by checking if speed is below 75% of max velocity. This is a proxy for what FBGA does optimally — it finds the true maximum speed at each path point.

4. **Integration path**: We have min-snap polynomial trajectories that output position as a function of arc length. FBGA takes exactly this as input: a spatial path with curvature values. We would:
   - Generate the min-snap spatial path (positions through gates) as we do now
   - Compute curvature along the path
   - Run FBGA to get the optimal velocity profile
   - Re-parameterize the trajectory in time using the FBGA velocity profile

For a drone, the g-g-v diagram would encode: `a_max_lateral(v) = sqrt((T_max/m)^2 - (g + drag(v)/m)^2)` where lateral acceleration capability decreases with speed due to drag. The longitudinal bounds couple to lateral through the thrust vector allocation.

## Actionable Takeaways

1. **Implement a Python FBGA**: Port the core algorithm (Algorithms 2-5) to Python/NumPy. The algorithm is simple — three passes over N segments with root-finding at each. With N=100 segments for our ~50m race course, this would run in <1ms even in Python.

2. **Define a drone g-g-v diagram**: Create `Gamma_y±(v_x)` and `Gamma_x±(a_y, v_x)` functions for our drone. The key physics:
   - Total thrust magnitude: `T_max = max_thrust` (from DroneConstraints)
   - At speed v with drag D(v): vertical thrust needed = `m*g + D_vertical`
   - Available horizontal thrust: `T_horiz = sqrt(T_max^2 - (m*g + D(v))^2)`
   - Lateral + longitudinal budget: `a_x^2 + a_y^2 <= (T_horiz/m)^2`
   - This is an elliptical g-g diagram that shrinks with speed — exactly what FBGA handles.

3. **Replace `_compress_times` and `_optimize_time_allocation`**: After generating the spatial path through gates, run FBGA instead of heuristic time allocation. This replaces three functions: `_initial_time_allocation`, `_optimize_time_allocation`, and `_compress_times`.

4. **Use FBGA for racing line evaluation**: In `racing_line.py`, the objective function estimates time via path length and curvature. Replace this with FBGA calls — evaluate each candidate racing line by running FBGA to get the true time-optimal lap time. With 0.06ms per FBGA call, we can evaluate hundreds of candidates in the L-BFGS loop.

5. **Implement the D± signed distance function**: This is a clean, reusable primitive for checking whether a drone state is within its performance envelope. Useful beyond just FBGA — could be used in the MPC tracker to check constraint feasibility.

6. **Consider wrapping the C++ implementation**: The open-source C++ code at github.com/DRIVEWISE/FBGA could be wrapped via pybind11 for maximum performance, though a pure Python implementation is likely fast enough for our course lengths (~100 segments).

## Limitations & Caveats

- **No jerk constraints**: FBGA allows instantaneous acceleration transitions. For a drone, motor dynamics limit how fast thrust can change. The authors acknowledge this and plan jerk-limited extensions. For our system, this means FBGA may produce velocity profiles that are slightly too aggressive at accel/decel transitions — our downstream geometric tracker will smooth these out, but there may be tracking error at transitions.

- **No formal optimality proof for non-convex constraints**: Optimality is proven only for convex/rectangular g-g shapes. For non-convex envelopes (which a drone at high speed with drag has), the solution is empirically near-optimal but not guaranteed. The 0.11-0.36% gap vs. OCP suggests this is not a practical concern.

- **Piecewise-constant acceleration**: Within each segment, `a_x` is constant. This is standard in direct optimal control but means the solution quality depends on discretization. With 1m segments on our ~50m course (N=50), this should be more than adequate.

- **Fixed path assumption**: FBGA optimizes speed along a given path, not the path itself. We still need our racing line optimizer for path planning. The two are complementary.

- **Assumes quasi-steady-state**: Lateral acceleration is `a_y = kappa * v^2`. This is exact for a point mass on a curved path but ignores transient dynamics (e.g., yaw rate buildup). For a drone, this is a good approximation since yaw dynamics are fast relative to translation.

- **CC BY-NC-ND license**: The C++ code cannot be modified or used commercially under this license. We would need to implement our own version, which is straightforward given the algorithm's simplicity.

## Key Parameters / Constants

- **N (mesh points)**: 100-300 for short horizon (300m), 3000-5000 for full lap. For our ~50m course, N=50-100 is sufficient.
- **v_max (top speed)**: Our `DroneConstraints.max_velocity = 15 m/s`.
- **v_ini (initial speed)**: Speed at trajectory start, typically 0 or current speed.
- **Root-finding tolerance**: The SOLVE routine precision controls accuracy vs. speed tradeoff. Default in the C++ code uses a custom non-derivative method.
- **g-g-v functions**: Need to define `Gamma_y±(v_x)` and `Gamma_x±(a_y, v_x)` specific to our drone. For a symmetric quadrotor: `Gamma_y+ = -Gamma_y- = sqrt((T_max/m)^2 - g^2 - drag_accel(v)^2)` and `Gamma_x+(a_y, v) = sqrt(a_horiz_max(v)^2 - a_y^2) - drag_accel(v)`, `Gamma_x-(a_y, v) = -sqrt(a_horiz_max(v)^2 - a_y^2) - drag_accel(v)`.
- **CPU time budget**: FBGA runs in 0.06-0.18ms for 100-300 points (C++, M2 Max). Python implementation ~10-50x slower, still well under 1ms for N=100.
- **Time breakdown**: ~20% VELSAT, ~60% Forward, ~20% Backward. Forward pass dominates due to more root-finding iterations.
