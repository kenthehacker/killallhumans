# Time-Optimal Planning for Long-Range Quadrotor Flights

- **URL**: https://arxiv.org/abs/2407.17944
- **Authors**: Wenliang Shao, Yunlong Song, Davide Scaramuzza (ETH Zurich Robotics and Perception Group)
- **Year**: 2024
- **Venue**: arXiv preprint (2024), building on prior ICRA/RAL work

---

## Key Contribution

The paper introduces **Automatic Optimal Synthesis (AOS)**, a polynomial-based trajectory planning method for minimum-time quadrotor flight over long distances with many waypoints. The central claim is that prior methods suffer from a scaling pathology: as flight range or waypoint count grows, discretization-based approaches require exponentially more decision variables (knot points), making them computationally intractable. AOS resolves this by leveraging the **analytical structure of time-optimal control** — specifically, the bang-bang and bang-singular structure derivable from Pontryagin's Maximum Principle — to automatically determine the minimum number of polynomial pieces needed to represent the optimal trajectory.

The practical outcome is that AOS produces trajectories that are only ~1.7% slower than the best known reference solutions, while running **orders of magnitude faster** computationally. In real-world flight, the method demonstrated peak velocities of 8.86 m/s over multi-waypoint courses in a motion capture lab.

This is distinct from TOGT (Qin 2024, arXiv:2309.06837), which focuses on racing gate traversal with spatial gate regions. AOS focuses on the general minimum-time fixed-endpoint and multi-waypoint problem, with a rigorous theoretical analysis of why a specific number of polynomial pieces is sufficient — something TOGT takes as a heuristic.

---

## Technical Approach

### Quadrotor Dynamics Model

The paper analyzes a **planar (2D) quadrotor model** to make the theoretical derivations tractable:
- States: horizontal position `x`, vertical position `z`, pitch angle `θ`, and their time derivatives
- Controls: collective thrust `F_T` and pitch rate `ω`
- Constraints: bounded thrust `F_T ∈ [F_min, F_max]` and bounded rotational rate `|ω| ≤ ω_max`

A non-dimensional formulation is used to remove mass and gravity parameters, scaling time by `ω_max` and position by `ω_max²/g`. This makes the theoretical results system-independent: the structural conclusions about the number of polynomial pieces needed hold for any quadrotor given its normalized constraints.

### Bang-Bang and Bang-Singular Control Structure

The core theoretical result, derived via PMP, is:

1. **Collective thrust is bang-bang with at most 5 switches.** This means the optimal thrust profile alternates between full thrust and minimum thrust at most 5 times over the entire trajectory.

2. **Rotational rate is bang-singular with at most 2 singular arcs and 3 isolated bang arcs.** On singular arcs, the control is neither at its maximum nor minimum — it takes an intermediate value determined by the state trajectory.

These results bound the complexity of the optimal trajectory. Knowing the maximum number of control switches tells us directly how many polynomial segments are needed: each switch introduces a new polynomial piece. This is the mechanism that makes "automatic optimal synthesis" possible — you do not need to guess how many knot points the optimizer should have; you can derive it.

### Piecewise Polynomial Representation

Rather than discretizing the trajectory into many short time steps (as in shooting methods or direct collocation), AOS represents the trajectory as **N piecewise polynomials**, where N is determined by the control structure analysis above. Each piece is a polynomial in time over a segment `[t_{k-1}, t_k]`, expressed in the flat output space `y(t) = [p^T, ψ]^T` (position + yaw).

The differential flatness property of quadrotors means that given `y(t)` and its derivatives up to order 4, the full state (position, velocity, acceleration) and control inputs (thrust, body rates) can be recovered analytically without integrating dynamics. This removes the system dynamics as explicit constraints in the optimization — they are satisfied by construction once the flat output polynomial is determined.

**Continuity conditions** between pieces enforce matching of position, velocity, and acceleration (and higher derivatives as needed) at each knot point. These are linear equality constraints on the polynomial coefficients, drastically reducing the effective dimension of the problem.

### Endpoint Constraint Handling

This is directly relevant to our system's issue of trajectory degradation near the last gate.

AOS formulates the planning problem as a **boundary value problem** with fixed initial and terminal states. The boundary conditions specify:
- Start state: position, velocity, acceleration (and higher derivatives for jerk continuity)
- End state: position, velocity, acceleration — **all three can be specified independently**

Critically, the method handles **arbitrary endpoint constraints** — not just hover-to-hover (zero velocity at both ends) but also cases where the drone arrives at the final point with nonzero velocity, which is the racing scenario. The NLP explicitly includes the terminal state as a constraint rather than assuming it.

For intermediate waypoints (gates), the method supports specifying position equality constraints at crossing times, allowing full multi-waypoint trajectories.

The AOS approach solves endpoint boundary conditions via numerical NLP rather than a shooting method. Shooting methods are notoriously sensitive to initial guesses for the co-state variables and can fail to converge for distant endpoints. AOS avoids this by parameterizing the trajectory directly in coefficient space and solving a well-conditioned finite-dimensional optimization.

### Time Allocation Strategy

The segment durations `T_k` (time for each polynomial piece) are **free variables in the optimization**, not pre-specified. The optimizer jointly minimizes total time `sum(T_k)` subject to:
1. Continuity constraints at knot points (linear in polynomial coefficients)
2. Endpoint and waypoint constraints (linear in polynomial coefficients)
3. Actuator constraint satisfaction (checked by sampling at dense time points)

The non-dimensional scaling (`t̂ = ω_max * t`, `x̂ = ω_max² * x / g`) makes the optimization landscape more uniform and reduces the conditioning sensitivity that typically arises when segment durations span different scales.

**Key insight for our system**: AOS does not use a distance-proportional or curvature-weighted heuristic to initialize segment times. Instead, it uses the analytical structure (bang-bang switch count) to determine N, and then solves for both coefficients and durations simultaneously. This is more principled than our current approach of allocating time based on arc length in `racing_line.py`.

### Minimal Polynomial Piece Inference

The algorithm has two modes:

1. **Known switch count**: If the number of thrust switches and singular arcs is known from analysis, N is directly computed and the NLP is solved once.

2. **Unknown switch count**: N is iteratively increased starting from a small value. If the optimizer converges to a feasible trajectory with a given N, it is sufficient. Otherwise N is incremented. This iterative procedure is guaranteed to terminate because the bound from PMP guarantees an N large enough always exists.

This is the "automatic" in "automatic optimal synthesis" — no manual tuning of polynomial piece count is required.

---

## Results

### Hover-to-Hover Flight Quality

Across varied start/end configurations (horizontal, diagonal, vertical, and random orientations):
- AOS trajectories are only **1.7% slower** on average than the best reference solutions computed by computationally intensive methods
- This near-optimality gap is consistent across all tested configurations

### Computational Speed

The benchmark comparison shows AOS is **orders of magnitude faster** than discretization-based state-of-the-art methods. While exact wall-clock numbers vary by problem complexity, the speedup is reported as multiple orders of magnitude (10x–1000x depending on the scenario).

This is especially pronounced for long-range flight: discretization methods scale quadratically or worse with flight duration (more knot points needed), while AOS's polynomial piece count stays bounded by the PMP-derived maximum.

### Long-Range Multi-Waypoint Flights

- Successfully plans 19-waypoint trajectories in motion capture environments
- Peak measured velocity in hardware flight: **8.86 m/s**
- Demonstrates that the method remains efficient even as waypoint count grows — the key innovation over prior approaches that struggle beyond a handful of waypoints

### Real-World Validation

Hardware experiments were conducted in a motion capture lab. Aggressive maneuvers with bang-bang thrust profiles were successfully executed, confirming that the theoretical bang-singular structure derived for the 2D model translates to real 3D flight behavior.

---

## Relevance to Our System

Our system (`planning/trajectory_optimizer.py`) uses min-snap polynomial trajectories with a two-phase approach: (1) curvature-aware speed profiling in `racing_line.py` to allocate segment times, then (2) polynomial fitting given those fixed times. This sequential approach is suboptimal because the time allocation does not account for the polynomial feasibility cost.

**The specific issue: trajectory ends before last gate, causing tracking degradation.** This is a direct manifestation of poor endpoint constraint handling. When the trajectory optimizer reaches the last gate, it needs to satisfy the terminal boundary conditions (position, velocity, acceleration at the final point). If the terminal constraints are inconsistent with the available segment time or if the optimizer implicitly assumes a hover-to-hover endpoint (zero terminal velocity), the trajectory will decelerate prematurely — the polynomial must "slow down" to meet the zero-velocity endpoint before the gate, making the last gate an endpoint rather than an intermediate pass-through.

AOS directly addresses this through two mechanisms:

1. **Explicit non-zero terminal velocity constraints**: The method can specify any terminal velocity, not just hover. For racing, the drone should arrive at the last gate with near-peak velocity and exit with momentum toward the finish. Specifying this terminal state explicitly prevents the premature deceleration artifact.

2. **Joint time-coefficient optimization**: Because segment times and polynomial coefficients are optimized together, the planner does not "run out of trajectory" near the endpoint — it naturally extends the trajectory as needed to satisfy both the gate constraint and the dynamic feasibility requirements.

3. **Minimal-segment parameterization**: By using the minimum N polynomial pieces for the given endpoint problem, AOS avoids over-constraining the trajectory near endpoints. Our current optimizer may be using a fixed segment structure that implicitly forces the trajectory to end at a hover point.

The bang-bang and bang-singular structure analysis also suggests that our `mpc_tracker.py` geometric tracker should expect thrust bang-bang inputs near time-optimal trajectories — this could explain why the tracker's gain tuning is sensitive near the last gate.

---

## Actionable Takeaways

### 1. Fix Terminal Velocity Constraint (Highest Priority — Directly Addresses Current Bug)

In `planning/trajectory_optimizer.py`, the endpoint constraints for the last segment should specify a **nonzero terminal velocity** (ideally, the approach velocity toward the last gate from the second-to-last gate direction). Currently the min-snap optimizer likely defaults to zero velocity at the final waypoint.

```python
# Instead of: terminal_velocity = [0, 0, 0]
# Use: terminal_velocity = direction_to_last_gate * nominal_speed * 0.5
# Or for pass-through: terminal_velocity = entry_velocity_at_last_gate
```

This single change would likely eliminate the trajectory degradation near the last gate.

### 2. Distinguish Endpoint Type from Waypoint Type

Treat intermediate gates as **pass-through waypoints** (nonzero velocity, position-only constraint) and the trajectory endpoint as a **full-state boundary condition**. AOS does this explicitly. Our current optimizer may be applying the same endpoint treatment to all gates.

In the polynomial fitting, this means:
- For gates 1 through N-1: constrain only position at the gate crossing time; leave velocity free
- For the final endpoint (if there is one after the last gate): constrain full state (position + velocity + acceleration)

### 3. Adopt Joint Time-Coefficient Optimization

Replace the sequential (time allocation → polynomial fit) pipeline with a joint optimization where segment times `T_k` are decision variables alongside polynomial coefficients. Even a simple 1D line search over a global time-scale factor, applied jointly to all segments, would improve on the current decoupled approach.

### 4. Use PMP Structure to Bound Polynomial Segment Count

For a course with K gates, the minimum number of polynomial pieces N needed is bounded by the bang-bang switch analysis (at most 5 thrust switches per segment + singular arc count). This suggests N = O(K) suffices, rather than the denser discretization we may currently use. Reducing N reduces computation and improves optimizer convergence.

### 5. Non-Dimensional Time Scaling

Normalize segment times by `ω_max` and positions by `ω_max²/g` before passing to the optimizer. This improves conditioning, especially for aggressive trajectories where segment times span a wide range.

### 6. Multi-Waypoint Long-Range Capability

If the competition track has more than ~10 gates, verify that our planner does not degrade due to knot-point explosion. AOS's iterative N inference would guarantee tractability; our current fixed-structure polynomial may not.

---

## Limitations & Caveats

1. **2D model analysis, 3D application**: The bang-bang switch count bounds are derived for a planar quadrotor. The 3D case has more complex control structure (three-axis rotations vs. one). The authors apply results to 3D flight, but the theoretical guarantees are strictly only proven for 2D. In practice this appears to work, but the minimal N for 3D may exceed the 2D prediction.

2. **No gate spatial constraints**: Unlike TOGT, AOS does not model gates as spatial regions (balls or polyhedra). Waypoints are point constraints. For a race with gates of finite aperture, the gate margin must be handled separately (either through safety margins or by combining with TOGT's gate parameterization).

3. **Near-optimal, not globally optimal**: The iterative N inference guarantees sufficiency but not that the local minimum found by the NLP solver is the global optimum. Multiple starts or annealing may be needed for complex tracks.

4. **Motion capture state feedback**: Hardware experiments use high-precision motion capture at 100+ Hz. With on-board VIO, state noise is higher and tracking performance will degrade relative to reported figures.

5. **Bang-singular control near hardware limits**: Singular arcs (intermediate thrust values) may be sensitive to actuator nonlinearities (motor deadband, ESC response) that smooth polynomial models do not capture. Bang-bang portions are more robust to these effects.

6. **No perception awareness**: The planner does not enforce gate visibility for the on-board camera. Our EKF requires the next gate to be in the camera's FOV for drift correction. A perception-constrained extension (ETH 2025 paper) would be needed to maintain gate visibility during aggressive maneuvers.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Time normalization | `t̂ = ω_max * t` | Non-dimensional time scaling |
| Position normalization | `x̂ = ω_max² * x / g` | Non-dimensional position scaling |
| Max thrust switches | 5 | Bang-bang thrust switch count upper bound (2D, PMP) |
| Singular arc count | ≤ 2 | Maximum singular arcs in rotational rate (2D) |
| Isolated bang arcs | ≤ 3 | Maximum isolated bang arcs in rotational rate (2D) |
| Optimality gap vs reference | ~1.7% | AOS solution slower than best known reference |
| Compute speedup vs state-of-art | orders of magnitude | Wall-clock time comparison |
| Peak flight velocity (hardware) | 8.86 m/s | Demonstrated in motion capture lab |
| Waypoints in long-range test | 19 | Multi-waypoint trajectory complexity |
| Polynomial representation | piecewise, degree TBD | Flat output space polynomials |
| Control rate in hardware | 100 Hz | Trajectory tracking loop (typical) |
| N inference strategy | iterative (start small, increase) | Minimum polynomial piece count discovery |
| Singular flow condition | `-p₂cos(Θ) + p₄sin(Θ) = 0` | Adjoint state condition for singular arc |
| Flat singular condition | `c₂c₃ = c₁c₄` | Condition for constant-pitch singular flow |

---

## Relationship to Other Papers in This Research Loop

- **TOGT (Qin 2024)**: Complementary. TOGT handles gate spatial regions and joint waypoint-time optimization via L-BFGS; AOS provides rigorous justification for polynomial piece count via PMP analysis. Combining both (AOS for N selection + TOGT for gate region constraints) would be stronger than either alone.

- **Fast Min-Snap (Burke 2020)**: Burke's fast min-snap uses QP with fixed time allocation; AOS explicitly optimizes time as a free variable. AOS is more principled for time-optimal (vs. snap-optimal) problems.

- **ETH Perception-Aware (2026)**: Orthogonal concern. AOS provides the trajectory backbone; the ETH paper adds FOV constraints on top. Both are needed for a complete racing pipeline with on-board gate detection.

- **Our system**: The most directly applicable insight is fixing terminal velocity constraints in `trajectory_optimizer.py` to use nonzero endpoint velocity, which should resolve the observed tracking degradation near the last gate.
