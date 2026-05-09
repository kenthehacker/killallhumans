# Sampling-Based Motion Planning with Online Racing Line Generation for Autonomous Driving on Three-Dimensional Race Tracks

- **URL**: https://arxiv.org/abs/2403.18643
- **Authors**: Levent Ögretmen, Matthias Rowold, Alexander Langmann, Boris Lohmann
- **Year**: 2024 (submitted March 27, 2024; revised May 13, 2024)
- **Venue**: 2024 IEEE Intelligent Vehicles Symposium (IV), Jeju Island, Republic of Korea, pp. 811–818

---

## Key Contribution

This paper addresses two distinct but related failures of existing sampling-based trajectory planners for autonomous racing. The first failure is that jerk-optimal trajectory primitives — the standard edge type used in sampling-based planners — cannot adequately represent the geometry of an optimal racing line on complex circuits. On a simple oval, a jerk-optimal arc can match the racing line well; on a complex multi-apex circuit with varying corner radii and elevation changes, the mismatch between jerk-optimal primitives and the true racing line forces the planner into suboptimal solutions. The second failure is that prior work only considers two-dimensional track geometry, ignoring the effects of elevation changes and banking on tire grip limits and vehicle dynamics.

The paper's answer to both problems is the same: generate the racing line online, directly from the current vehicle state and local track geometry, rather than using a pre-computed global offline racing line as a fixed reference. The online racing line is generated over a limited spatial horizon using optimal control, respecting real-time velocity constraints derived from velocity/acceleration (gg) diagrams. The sampling-based planner then generates trajectory candidates that are evaluated against this locally-optimal reference, which is continuously updated as the vehicle progresses.

A secondary contribution is the explicit 3D extension: the planning framework uses Euler angles derived from track geometry to correctly account for slope-induced changes in available lateral and longitudinal grip, captured through velocity-dependent gg-diagrams that vary with terrain slope.

---

## Technical Approach

### Overall Architecture

The system comprises six integrated components, as described in the associated GitHub repository (TUMRT/sampling_based_3D_local_planning):

1. **gg-Diagram Generation (offline)**: For each velocity and slope combination, computes the envelope of achievable accelerations (the "gg-diagram"). These are approximated as diamond shapes for computational tractability. The velocity-dependent gg-diagram encodes that available grip changes with speed and with track inclination.

2. **Track Processing (offline)**: Smooths raw 3D track boundary data (x, y, z coordinates for left and right boundaries) to remove measurement noise while preserving topological structure. Computes Euler angles (roll, pitch, yaw) at each track point from boundary geometry.

3. **Global Racing Line (offline)**: Computes a traditional minimum-lap-time racing line around the full circuit. This serves as a fallback reference and for lap-time comparison, but is not the primary reference for the online planner.

4. **Online Racing Line Generation**: The key innovation. Given the current vehicle state and local track geometry over a finite spatial horizon, solves a minimum-time optimal control problem to generate a locally optimal racing line from that starting point. This uses the Acados optimal control framework (open-source, based on HPIPM and BLASFEO for high-performance QP/NLP solving).

5. **Sampling-Based Trajectory Generation**: Generates multiple candidate trajectories (samples) from the current state. Candidates are jerk-optimal in the spatial domain — an extension from the temporal domain used in prior work, which better captures brake points and apexes.

6. **Trajectory Selection**: Scores each candidate trajectory against a cost function that penalizes deviation from the online racing line (cross-track and along-track error) and rewards progress. The lowest-cost feasible trajectory is executed.

### Online Racing Line Algorithm

The online racing line builds on Rowold et al. (2023) "Online Time-Optimal Trajectory Planning on Three-Dimensional Race Tracks" (IEEE IV 2023). The key properties are:

- **Limited spatial horizon**: Rather than solving for the full lap, the algorithm plans over a fixed spatial horizon ahead of the vehicle. This makes the problem tractable for real-time execution.
- **Time-optimal formulation**: Minimizes traversal time over the horizon subject to track boundary constraints and gg-diagram feasibility constraints at each arc-length position.
- **Frenet frame representation**: The problem is formulated in track-aligned Frenet coordinates (arc-length s, lateral offset d, heading deviation Δψ), enabling efficient boundary constraint encoding.
- **Acados NLP solver**: The resulting problem is a nonlinear program solved with a real-time iteration scheme (RTI) for online execution.
- **3D coupling**: The slope-induced changes to available acceleration are incorporated via the velocity-dependent gg-diagrams, so the optimizer knows that a downhill section allows higher speeds and an uphill section constrains them.

### Spatial vs. Temporal Jerk-Optimality

A key technical contribution, elaborated in a follow-up paper by the same group (Ögretmen et al. 2025, arXiv:2505.05157), is the transition from temporal-domain to spatial-domain sampling. Traditional jerk-optimal trajectory edges minimize ∫ j(t)² dt subject to boundary conditions in time. This works well when following a velocity profile that matches the racing line.

However, when the sampled velocity deviates significantly from the racing line (e.g., braking harder or later into a corner), the temporal jerk-optimal arc places the apex at a different longitudinal position than intended, creating a structural mismatch. Spatial-domain jerk-optimal edges minimize ∫ j(s)² ds with arc-length as the independent variable. This produces trajectories where apex placement (the longitudinal coordinate of minimum lateral velocity) is correctly associated with corner geometry regardless of the velocity profile, better representing the space of feasible racelines.

### 3D Effects

The track is parameterized as a 3D surface described by boundary coordinates (x(s), y(s), z(s)) plus Euler angles (φ, θ, ψ) computed from the tangent and normal vectors at each arc-length position. The gg-diagram is a function of both speed v and slope angle θ, capturing the physical reality that:
- On an uphill section, gravitational force reduces available traction for acceleration
- On a downhill section, available braking force is supplemented by gravity
- Banked corners increase the effective normal force (and thus available lateral grip)

This 3D coupling was absent in prior 2D-only planners.

### Multi-Vehicle Scenario

The paper specifically addresses multi-vehicle racing, where a fixed offline racing line becomes suboptimal because it ignores opponents. The online racing line naturally adapts: if the optimal path is blocked by a competitor, the optimizer solves for the best reachable line given the current dynamic boundary constraints (which can include exclusion zones around opponents). The authors demonstrate that this online adaptation provides significant lap time gains compared to rigidly following an offline racing line when opponents are present.

---

## Results

Simulation experiments on complex multi-corner circuits demonstrate:

- **Lower lap times** than prior sampling-based planners that use an offline racing line reference
- **Improved utilization of dynamic limits** (the vehicle operates closer to the gg-diagram boundary throughout the lap)
- **Multi-vehicle gains**: Particularly significant lap time improvements when the online racing line generation is active in multi-vehicle scenarios, because the planner can dynamically reroute around blocked paths

The follow-up work (arXiv:2505.05157) provides a specific data point: their extended spatial-domain planner achieves **1.42 seconds faster** lap times in test sections compared to a temporal-domain baseline when online velocity profiles are used instead of offline references.

From the companion repository and papers, key performance characteristics are:

- Real-time capable on standard racing computing hardware (the Acados-based online racing line solver runs within planning cycle budgets of ~50–100 ms)
- Applicable to circuits as complex as full F1-style tracks with multiple corner types and elevation changes
- The system is implemented in Python 3.10.12 with Acados as the core solver

---

## Relevance to Our System

This paper is relevant to our racing line determinism problem from two angles.

**First, the online racing line as a replacement for our offline `racing_line.py`**: Our current pipeline pre-computes a static racing line using `racing_line.py`'s lateral offset optimization, and `_select_by_sim()` tries to choose among multiple candidate racing lines by simulating each one. This paper shows that generating the racing line online (per planning cycle) from the current vehicle state is feasible and outperforms offline racing lines, particularly in dynamic scenarios. The key advantage is that the online racing line is generated fresh each cycle — there is no pre-computed reference to have basin-selection non-determinism over.

**Second, the spatial-domain sampling architecture for our trajectory optimizer**: Our `trajectory_optimizer.py` uses min-snap polynomial segments. The paper's insight that spatial-domain primitives better capture apex placement and racing line geometry applies directly: if we parameterize our min-snap segments by arc-length rather than time, the resulting trajectories will better approximate the racing line regardless of speed profile variations.

**For the specific `_select_by_sim()` problem**: The paper's approach suggests an alternative selection criterion. Rather than running full trajectory optimization + kinematic simulation for each candidate, score candidates by:
1. Deviation from an online racing line (computed once per planning cycle)
2. Dynamic limit utilization (how close the trajectory brings the drone to thrust/velocity limits)

This decouples candidate selection from trajectory optimization: select the racing line using the online optimizer (or a geometric criterion), then run trajectory optimization only once on the selected line.

The TUM Racing Team's GitHub repository (TUMRT/sampling_based_3D_local_planning) provides open-source Python code for the gg-diagram generation, track processing, global and online racing line computation, and sampling-based planner. This is directly usable as a reference implementation.

---

## Actionable Takeaways

1. **Generate the racing line deterministically per-run using a lightweight online optimizer**: Replace `_select_by_sim()` with a single call to a spatial-horizon time-optimal optimizer (using Acados or a simple convex relaxation). The optimizer takes gate positions as boundary constraints and produces a unique, deterministic racing line without basin-selection ambiguity.

2. **Switch trajectory primitives to spatial domain**: Modify `trajectory_optimizer.py` to generate min-snap polynomials in arc-length s rather than time t. This fixes apex placement alignment with gate geometry and produces trajectories that better follow the racing line when velocity profiles change.

3. **Build a gg-diagram for our drone**: Characterize the drone's achievable acceleration envelope as a function of speed (and potentially attitude) using data from our PyBullet simulation. Store this as a lookup table and use it to constrain the online racing line optimizer. For a drone, the gg-diagram is roughly an ellipse in the horizontal plane bounded by maximum collective thrust minus gravity.

4. **Use the online racing line as the scoring function for candidate trajectories**: Instead of scoring by simulated lap time (which requires full optimization), score each candidate trajectory by how closely it tracks the online racing line. The online racing line is computed once per planning cycle and provides a stable, deterministic reference.

5. **Integrate 3D effects via elevation-aware gate sequencing**: If competition tracks have significant elevation changes, incorporate the z-coordinate of gate centers into our `sequencer.py` gate pass detection and into the racing line's speed profile. This is the drone-racing analog of the paper's slope-corrected gg-diagrams.

6. **Adopt the Acados solver for online racing line generation**: The existing Python bindings for Acados make it straightforward to set up a receding-horizon trajectory optimization problem. This would replace the current scipy/custom solver in `trajectory_optimizer.py` with a purpose-built RTI (real-time iteration) scheme.

7. **Reference the TUMRT repository for implementation**: `TUMRT/sampling_based_3D_local_planning` provides working Python code for all six system components. Start from their `local_sampling_based/sim_sampling_based_planner.py` entry point and adapt the vehicle parameters to quadrotor dynamics.

---

## Limitations & Caveats

**Ground vehicle dynamics, not UAV**: The entire system is built around a 4-wheel vehicle with tire friction constraints and a point-mass dynamics model with μ-slip curves. The gg-diagram structure, slip angle constraints, and longitudinal/lateral decoupling assumptions do not apply directly to a quadrotor. However, the gg-diagram concept generalizes: a drone's achievable acceleration set at a given speed can be precomputed and stored in an analogous table.

**Real-time compute requirements**: The online racing line generation solves a receding-horizon NLP at each planning step. The paper demonstrates real-time feasibility for ground vehicles (~10 ms planning cycles), but the complexity depends on horizon length and track discretization. For drone racing at 100+ Hz control rates with short planning horizons, this may require careful problem sizing.

**No gate-passing constraints**: The paper's racing line formulation is for continuous tracks with boundaries on both sides. Our drone racing problem has gate constraints — the drone must pass within a specific rectangular aperture. These are point constraints at discrete locations, not continuous boundary constraints. The Frenet-frame optimization would need to be extended to handle gate aperture constraints.

**Spatial-domain primitives require invertible parameterization**: Transitioning to arc-length parameterized trajectory segments requires that the trajectory is monotonically increasing in arc-length (no reversals). For aggressive drone maneuvers that might involve near-zero or backward velocity (e.g., split-S turns), this assumption could break down.

**Multi-vehicle focus**: Much of the paper's motivation and gains come from multi-vehicle scenarios. In a time-trial (solo race) setting, the online racing line is still beneficial but the gains are smaller — the offline racing line is already near-optimal when there are no opponents to avoid.

**Acados dependency**: Adopting the Acados framework introduces a significant new dependency (C library with Python bindings) that conflicts with our "no unnecessary dependencies" policy. A simpler convex approximation might be more appropriate for our use case.

---

## Key Parameters / Constants

- Acados optimal control framework (open-source, C + Python bindings): primary solver for online racing line
- Spatial resolution: consistent with Δs = 2.0 m used in related TUM work
- Python version: 3.10.12 (per repository)
- Test environment: Ubuntu 22.04.3 LTS
- Lap time improvement (spatial vs temporal domain): 1.42 s per test section (from follow-up paper arXiv:2505.05157)
- gg-diagram approximation: diamond underapproximation of achievable acceleration ellipse
- Planning horizon: limited spatial horizon (exact value not publicly specified; typically 50–200 m for ground vehicle racing)
- Online racing line solver: RTI (real-time iteration) scheme within Acados NLPF framework
- Track boundary representation: (x, y, z) coordinates for left/right boundaries + Euler angles (φ, θ, ψ) from tangent/normal geometry
- Gate constraint representation (our adaptation): rectangular aperture as hard constraint at discrete arc-length positions
