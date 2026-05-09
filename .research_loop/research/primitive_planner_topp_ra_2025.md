# Primitive-Planner: An Ultra Lightweight Quadrotor Planner with Time-optimal Primitives

- **URL**: https://arxiv.org/abs/2502.16882
- **Authors**: Jialiang Hou, Neng Pan, Zhepei Wang, Jialin Ji, Yuxiang Guan, Zhongxue Gan, Fei Gao
- **Year**: 2025
- **Submitted**: February 24, 2025
- **Category**: Robotics (cs.RO)

---

## Key Contribution

Primitive-Planner proposes a computationally minimal quadrotor trajectory planner built around a pre-computed library of time-optimal motion primitives. Unlike real-time optimization-based planners (e.g., EGO-Planner-v2, MINCO), which solve expensive trajectory optimization problems online, Primitive-Planner performs all heavy computation offline — generating a library of 73 geometric primitives (circular arc paths of varying curvature and heading deviation) and applying TOPP-RA to each primitive to produce time-optimal, dynamically-feasible velocity profiles. At runtime, the planner only needs to: (a) check which primitives are collision-free in the current local environment and (b) select the minimum-cost safe primitive. Both steps complete in under 4 ms on embedded hardware.

The secondary contribution is a novel fast collision-checking algorithm whose time complexity is independent of the resolution of obstacle sampling. By pre-computing which voxels each primitive occupies, online checking reduces to a fixed-cost batch lookup over a 2000-point point cloud sample of the environment. This deterministic, O(1)-in-resolution complexity is a significant departure from prior art, which scales either with the number of obstacle samples or the resolution of the occupancy map.

---

## Technical Approach

### Motion Primitive Library — Offline Generation

The path library consists of 73 geometric paths, each with a fixed arc length of 5 meters:

- **Radii**: r ∈ {6, 8, 12, 20, 36, 78, ∞} meters (7 values; ∞ corresponds to a straight line)
- **Heading deviations**: θ ∈ {0°, −10°, −20°} as a discrete set, with 30° interpolation intervals filling in intermediate headings
- This gives 7 × (number of heading samples) = 73 total primitive paths

Each path is stored as a sequence of 1000 discrete points sampled at uniform arc-length intervals, achieving O(Δ²) interpolation error where Δ is the arc-length step.

### TOPP-RA Velocity Profile Generation

For each geometric path, an offline TOPP-RA (Time-Optimal Path Parameterization via Reachability Analysis) run produces the time-optimal velocity profile subject to velocity and acceleration bounds.

The core transformation is the standard path-parameter substitution. Let s be arc-length and define the square speed variable:

```
ṡ² = (ds/dt)²
```

Second derivatives transform via:

```
s̈ = (ṡ²ᵢ₊₁ - ṡ²ᵢ) / (2Δs)
```

The dynamic constraints take the second-order form:

```
a(s)s̈ + b(s)ṡ² + c(s) ∈ C(s)
```

where a(s), b(s), c(s) are functions derived by substituting path derivatives into the drone's equations of motion. The constraint set C(s) encodes:

- **Velocity bounds**: v_min ≤ ‖q̇‖ ≤ v_max
- **Acceleration bounds**: a_min ≤ ‖q̈‖ ≤ a_max
- **Velocity norm constraint**: ‖q̇‖² ≤ v_norm²

Velocity bounds in the path-parameter domain become:

```
ṡ²_min = 0
ṡ²_max = v²_max / (q'ᵀ q')
```

where q' = dq/ds is the unit tangent (since the path is arc-length parameterized, ‖q'‖ = 1, so ṡ²_max = v²_max on straight segments).

**TOPP-RA Two-Pass Algorithm:**

1. **Backward pass**: Starting from the end point with a known terminal velocity (ṡ²_N = 0 for offline library generation), compute the feasible speed-square interval [I_min(s), I_max(s)] at each discretized path point by solving linear programs backward.

2. **Forward pass**: Starting from the known start velocity, propagate forward by selecting the maximum feasible speed-square at each point (greedy speed maximization), constrained to remain within the backward-computed reachability intervals.

The LP at each point uses Seidel's algorithm (O(d) expected time for d constraints), keeping per-point cost very low.

**Variable start velocities**: The library is generated for a grid of start velocities q̇₀ ∈ {0, 0.1, ..., v_max} to enable seamless primitive chaining during receding horizon planning. End velocities are fixed at 0 for offline library generation but matched to continuity requirements online.

### Fast Deterministic Collision Checking

**Offline pre-computation:**
- The space swept by the primitive library is covered by a virtual voxel grid
- For each voxel, pre-compute which primitives pass through it (via kd-tree queries at distance d)
- Store a lookup table: voxel_id → set of unsafe primitive indices

**Online checking (per planning step):**
1. Sample N_pc = 2000 points from the local point cloud (random sampling)
2. For each sampled point, compute its voxel index
3. Batch-mark all primitives that intersect any occupied voxel as unsafe
4. The safe primitive set = all 73 primitives minus the union of unsafe sets

Time complexity: O(N_pc) = O(2000) regardless of obstacle density or primitive sampling resolution. Measured runtime: 3.2–3.6 ms.

Safety margin: the query distance is inflated to d + r_inflated to provide clearance.

### Trajectory Selection

Given the set of safe primitives, the planner selects the minimum-cost one via:

```
c = λ_g · c_goal + λ_b · c_bound
```

where:
- c_goal = ‖p_end − p_goal‖ − ‖p_start − p_goal‖  (progress toward global goal)
- c_bound penalizes deviation from the planning boundary

The selected primitive must also match the drone's current speed (continuity matching via the variable start-velocity library).

### Receding Horizon Planning with Coordinate Transformation

Re-planning occurs at **10 Hz**. To guarantee C¹ continuity (position + velocity) between concatenated primitives across re-planning steps, a velocity-frame coordinate transformation is applied:

- x-axis of frame {V}: current velocity direction (normalized)
- y-axis: cross(x, [0,0,-1])
- z-axis: cross(x, y)

All 73 primitives are stored tangent to the x-axis of {V}. When the frame is aligned to the current velocity, every primitive is automatically tangent to the current flight direction, guaranteeing smooth concatenation without explicit boundary conditions.

### Smoothness Properties

The approach guarantees C⁰ (position matching) and C¹ (velocity direction matching) continuity between primitives by construction (the coordinate transform ensures tangency). Higher-order continuity (C² in acceleration, C³ in jerk) is **not guaranteed** — this is a deliberate trade-off for computational speed. The TOPP-RA velocity profiles are piecewise-linear in the ṡ² domain, so the resulting trajectory acceleration is piecewise-constant (zero higher derivatives between knot points), which introduces finite-magnitude acceleration steps at segment boundaries.

---

## Results

### Simulation Benchmark (Dense Environment, 200 obstacles)

| Method | Flight Time (s) | Path Length (m) | Computation (ms) |
|--------|-----------------|-----------------|-----------------|
| Mapless (primitive-based baseline) | 19.747 | 48.541 | 10.794 |
| EGO-Planner-v2 (optimization-based) | 21.707 | 41.934 | 9.974 |
| **Primitive-Planner (proposed)** | **13.479** | **41.336** | **3.595** |

Key observations:
- **32% shorter flight time** vs Mapless, **38% shorter** vs EGO-Planner-v2
- **Path length** is competitive with EGO-Planner-v2 despite being primitive-based (no continuous optimization)
- **Computation** is ~3× faster than either baseline, concentrated almost entirely in collision checking (3.54 ms); trajectory selection costs only 0.055 ms

### Collision Checking Timing (across environment densities)

Collision checking time: 3.2–3.6 ms regardless of sparse/medium/dense obstacle density. This confirms the O(1)-in-density property of the algorithm.

### Success Rates by Library Size

| Library Size | Sparse | Medium | Dense |
|---|---|---|---|
| Low (25 primitives) | 100% | 80% | 65% |
| Medium (37 primitives) | 100% | 90% | 75% |
| High (73 primitives) | **100%** | **100%** | **100%** |

The full 73-primitive library achieves 100% success across all environment densities, validating the coverage design.

### Real-World Validation

- 109 time-optimal trajectories computed for physical quadrotor implementation
- v_max = 2 m/s, a_max = 6 m/s² (conservative real-world limits)
- Sustained ~2 m/s in dense obstacle environments
- Robust navigation confirmed

---

## Relevance to Our System

Our system (`planning/trajectory_optimizer.py`) currently uses minimum-snap polynomial trajectories with a custom `_topp_retime()` pass that does curvature-based speed profiling. Our `_inflate_sharp_turns()` heuristic compounds centripetal acceleration estimates and helix/S-turn detection to slow down tight sections. We also maintain a `planning/racing_line.py` optimizer that offsets pass-through points within gate openings to cut corners.

**Where Primitive-Planner's TOPP-RA formulation is directly relevant:**

1. **TOPP-RA two-pass algorithm for our speed profiling**: Our `_topp_retime()` function is described as "TOPP-RA-style" but the paper provides the full algorithmic specification (backward reachability via LP, forward maximization) that could replace the heuristic centripetal thresholds in `_inflate_sharp_turns()`. The `a(s)s̈ + b(s)ṡ² + c(s) ∈ C(s)` form with LP solvers (Seidel's algorithm) is the key detail. Our current implementation uses ad-hoc curvature estimates; replacing with proper TOPP-RA would give dynamically-correct speed limits.

2. **Variable start-end velocity support**: The paper explicitly handles non-zero start and end velocities (q̇₀ grid from 0 to v_max). Our `_topp_retime()` must handle gate-to-gate transitions where entry speed is non-zero. The TOPP-RA backward pass from non-zero end velocity is directly applicable.

3. **The helix and S-turn sections** (gates 7–8 helix, gates 3–4 S-turn) are precisely the sections where our heuristics break down (per-gate errors of 0.171m avg in iteration 35). A proper TOPP-RA implementation enforcing centripetal acceleration constraints via the path-parameter LP would produce tighter, correct speed limits for these sections without the empirical threshold tuning.

4. **Primitive-Planner's architecture is conceptually different from ours**: We plan gate-constrained polynomial trajectories offline, not receding-horizon primitives with obstacle avoidance. Their 10 Hz replanning loop is irrelevant to our competition setting where the course is known in advance. The TOPP-RA speed profiling component is the transferable piece, not the broader architecture.

5. **The 73-primitive library approach** is not applicable to gate racing (our waypoints are fixed and ordered by the competition rules). However, the offline computation philosophy (pre-compute TOPP-RA speed profiles for fixed paths) aligns with how we use `trajectory_optimizer.py` — compute once before the race, execute from cache.

---

## Actionable Takeaways

1. **Implement proper TOPP-RA backward-forward pass** in `_topp_retime()`. Replace the heuristic centripetal inflation in `_inflate_sharp_turns()` with LP-based reachability analysis. At each of the 1000 sampled path points, solve for the feasible ṡ² interval via Seidel's LP, then forward-propagate to find the time-optimal speed profile. This should eliminate the threshold magic (`a_centripetal_threshold = 4.5`) and helix/S-turn special-casing.

2. **Enforce velocity norm constraints alongside centripetal acceleration**. The paper includes `‖q̇‖² ≤ v_norm²` as a third constraint type alongside velocity and acceleration. In our path-parameterized form, the constraint ṡ²_max = v²_max / ‖q'‖² is trivially satisfied on arc-length-parameterized paths (‖q'‖ = 1), but our polynomial paths are NOT arc-length-parameterized. Pre-reparameterize by arc-length before applying TOPP-RA.

3. **Use Seidel's LP (O(d) expected)** rather than scipy's LP for the per-point feasibility problems. At 1000 path points with ~4 constraints each, Seidel's algorithm runs in microseconds per point. Total per-trajectory TOPP-RA cost < 1 ms. This replaces our current O(N) heuristic inflation which misses intra-segment violations.

4. **Store variable start-velocity profiles** for the gate-to-gate segments. Currently our `segment_times` represent a scalar time budget per segment; there is no mechanism to ensure the exit velocity from one segment matches the entry condition for the next TOPP-RA pass. Build a velocity-continuous TOPP-RA stitching procedure: after initial time allocation, run TOPP-RA on the full path (not per-segment) with boundary velocities propagated end-to-end.

5. **Apply TOPP-RA on the full arc-length-reparameterized path** rather than segment-by-segment. Our current approach inflates segments individually, missing inter-segment coupling (the approach speed to a tight gate depends on how the previous straight was executed). A single full-path TOPP-RA pass automatically handles this coupling.

6. **Replace per-gate TOPP "floor"** (iteration 35's `helix TOPP floor` commit) with a proper reachability-bounded floor. The current floor in `_topp_retime()` prevents the helix from being over-slowed; a proper backward pass naturally gives a non-zero lower bound on ṡ² from the acceleration reachability constraint (cannot decelerate faster than a_max), eliminating the need for a manually set floor.

7. **Pre-compute the TOPP-RA result and cache it** alongside the racing line cache (already done in `racing_line_cache.json`). Since the path is fixed before the race, there is no reason to recompute TOPP-RA at planning time. Cache the ṡ²(s) profile indexed by the same hash as the racing line offsets.

---

## Limitations & Caveats

1. **Low-speed evaluation only**: The paper tests at v_max = 2–3 m/s and a_max = 6 m/s². Our competition requires up to 15 m/s with 20 m/s² acceleration. At these speeds, drag forces (neglected in their formulation), rotor saturation asymmetries, and gyroscopic effects become significant. The simple velocity/acceleration constraint model may underestimate feasible speed in some segments and overestimate it in others.

2. **No jerk or snap constraints**: The formulation only enforces velocity and acceleration bounds — no jerk (third derivative) or snap (fourth derivative) limits. Minimum-snap polynomial trajectories exist specifically because bang-bang acceleration (piecewise constant) produces infinite jerk at knot points, which excites structural resonances and saturates attitude controllers. For our system, snap constraints matter because the SE(3) geometric controller uses feedforward jerk; sudden jerk changes degrade tracking. A TOPP-RA variant with snap constraints (e.g., via convex constraints on the third derivative of q) would be needed for seamless integration with our min-snap backbone.

3. **Arc paths only**: The primitive library uses circular arcs and straight lines. This is sufficient for obstacle avoidance in cluttered environments but inadequate for gate-constrained racing trajectories, which require polynomial curves (min-snap) to satisfy pass-through position/velocity constraints at gate boundaries.

4. **No gate constraints**: The planner has no concept of passing through a specific window at a specific orientation. It is designed for collision-free navigation in unstructured environments, not for precision gate traversal. The racing line optimization problem (finding the optimal lateral offset within each gate opening) is orthogonal to TOPP-RA and not addressed.

5. **Receding horizon replanning (10 Hz) is irrelevant to our setting**: We plan offline with full course knowledge. The 10 Hz loop, coordinate frame transformation, and primitive chaining mechanisms are inapplicable.

6. **C⁰ and C¹ continuity only**: As noted above, higher-order continuity is not guaranteed. Concatenating primitives at 10 Hz produces piecewise trajectories with acceleration discontinuities, which would cause unacceptable tracking errors for our SE(3) controller that relies on smooth reference acceleration.

7. **No comparison to polynomial or minimum-snap methods**: The paper compares against Mapless (another primitive planner) and EGO-Planner-v2 (gradient-based ESDF). Min-snap, MINCO, or TOGT-style polynomial methods are not evaluated. It is unclear whether primitive planning outperforms polynomial methods on gate-structured courses.

8. **Real-world speed limited to 2 m/s**: The physical experiments operate at conservative limits. Whether TOPP-RA-derived profiles remain accurate at 10+ m/s (where aerodynamic drag, battery voltage sag, and rotor dynamics become nonlinear) is not evaluated.

---

## Key Parameters / Constants

The following numerical values from the paper are directly usable or serve as reference baselines:

| Parameter | Value | Context |
|---|---|---|
| Path arc length per primitive | 5 meters | Fixed primitive length for stable coverage |
| Arc radii | {6, 8, 12, 20, 36, 78, ∞} m | 7 discrete curvature levels |
| Heading deviation angles | {0°, ±10°, ±20°} + 30° intervals | Heading coverage |
| Total primitives | 73 | Full library achieving 100% success |
| Path discretization points | 1000 | Satisfies O(Δ²) interpolation error |
| Online point cloud samples | N_pc = 2000 | Fixed-cost collision checking |
| Replanning frequency | 10 Hz | Receding horizon loop rate |
| Start velocity grid | {0, 0.1, ..., v_max} m/s | Variable entry velocity support |
| Terminal velocity (offline) | 0 m/s | Library generation boundary condition |
| Collision checking time | 3.2–3.6 ms | Independent of obstacle density |
| Trajectory selection time | 0.055 ms | Negligible compared to collision check |
| Test v_max | 3 m/s (sim), 2 m/s (real) | Evaluation speed regime |
| Test a_max | 6 m/s² | Evaluation acceleration bound |
| LP solver | Seidel's algorithm | O(d) expected per point |
| Interpolation error bound | O(Δ²) where Δ = arc-length step | Accuracy guarantee |
| Flight time improvement | 13.479s vs 19.747s (Mapless) | 31.8% reduction |
| Computation improvement | 3.595ms vs 10.794ms (Mapless) | 66.7% reduction |

For direct application to our TOPP-RA implementation: the 1000-point discretization with Seidel's LP per point is the key design decision. At our path lengths (typically 3–8m per inter-gate segment), 1000 points gives 3–8 mm spacing, well below the 0.1m accuracy requirement.
