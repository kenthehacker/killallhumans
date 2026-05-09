# Online Velocity Profile Generation and Tracking for Sampling-Based Local Planning in Autonomous Racing

- **URL**: https://arxiv.org/abs/2505.05157
- **Authors**: Alexander Langmann, Levent Ögretmen, Frederik Werner, Johannes Betz (Technical University of Munich — TUM School of Engineering and Design, Chair of Automatic Control, Institute of Automotive Technology, Professorship of Autonomous Vehicle Systems)
- **Year**: 2025

---

## Key Contribution

This paper addresses a core gap in sampling-based autonomous racing planners: the velocity profile used for trajectory evaluation is typically precomputed offline and becomes invalid whenever dynamic constraints change (grip loss, tire wear, rubber accumulation) or when the vehicle deviates laterally from the nominal racing line. The authors contribute:

1. **An online forward-backward velocity profile solver** that runs in real time (43 ms on a single Python/CPU core), continuously adapting to a grip-scaling parameter α that linearly modulates the maximum feasible lateral and longitudinal accelerations.

2. **An apex-aligned spatial sampling strategy** for local trajectory generation that places trajectory samples at track features (brake points, apexes) rather than at fixed temporal intervals, ensuring that critical speed-limiting constraints are not missed when the planner's horizon spans a braking zone.

3. **A 3D track representation** using parametrized Frenet-Serret frames, which correctly handles banked corners and elevation changes that cause apparent gravitational components to appear in the vehicle's lateral and longitudinal axes.

The combination allows the planner to legally and safely respond to reduced-grip events (demonstrated with α = 0.7, simulating a 30% reduction in peak acceleration) and to overtake maneuvers that take the vehicle significantly off the nominal racing line, with documented lap-sector time advantages of 1.18–1.42 seconds over approaches that use a frozen offline velocity profile.

---

## Technical Approach

### 3D Track Representation

The track spine is a parametrized curve C = {c(s) ∈ ℝ³ | s ∈ [0, s_lap]}, with Frenet-Serret road frames constructed at each arc-length station s. Euler angles (φ, μ, θ) in zyx convention describe the local road-plane orientation. This is important because on banked corners the gravitational acceleration vector projects onto the vehicle's lateral and longitudinal axes, modifying the effective acceleration limits.

Apparent accelerations are expressed as:

    ã_x = a_x + g·sin(μ)
    ã_y = a_y - g·cos(μ)·sin(φ)
    ã_z = a_z - g·cos(μ)·cos(φ)

These transformed accelerations are what appear in the feasibility constraint, not the raw body-frame values.

### Vehicle Dynamics Model (Point-Mass Quasi-Steady-State)

The paper uses a point-mass model with g-diagram acceleration limits. A combined acceleration feasibility constraint is expressed as:

    1 ≤ (ã_x / (α · ã_{x,lim}))^ρ + (ã_y / (α · ã_{y,max}))^ρ

where ρ controls the shape of the g-diagram (circular at ρ = 2, diamond at ρ = 1, rounded rectangle at ρ > 2), and α ∈ (0,1] is a real-time grip-scaling factor. Setting α < 1 shrinks the entire g-diagram, immediately reflecting reduced traction.

### Apex Detection

The algorithm scans the precomputed race line for local curvature maxima Ω_{z,max} to identify apex candidates. An initial velocity guess at each apex candidate is:

    V_{guess,cand} = √α · V_off

where V_off is the offline profile speed at that location. The apex is then refined iteratively:

    V_new = √(â_y / Ω_z)

until |V_new - V_old| < ε. This iterative refinement is cheap because curvature and the current α are already known; convergence typically requires only a few iterations.

### Forward-Backward Solver

Starting from apex speeds as boundary constraints, the solver sweeps forward (acceleration phase) and backward (braking phase):

    ã_{x,fw} = min{ ã_{x,eng}, ã_{x,max} · (1 - (ã_y/(α·ã_{y,max}))^ρ)^(1/ρ) }
    ã_{x,bw} = ã_{x,min} · (1 - (ã_y/(α·ã_{y,max}))^ρ)^(1/ρ)
    V_feas   = min{ V_fw, V_bw }

Engine torque limits bound the forward sweep (ã_{x,eng}); braking limits bound the backward sweep. The final feasible velocity at each discretization node is the minimum of the two sweeps. This is the classical "bang-bang" velocity profile algorithm extended for 3D tracks and real-time α updates.

### Spatial-Domain Sampling

Conventional sampling-based planners use a fixed temporal horizon T. During a braking zone, a large fraction of the horizon is consumed by a relatively short spatial distance, meaning the planner can miss the apex entirely. The paper proposes converting all temporal trajectory attributes to the spatial domain:

    s̈(s) = s̈(t) / ṡ(t)
    n'(s) = ṅ(t) / ṡ(t)
    n''(s) = (n̈(t) - n'·s̈(t)) / ṡ(t)²

Samples are then placed at fixed spatial intervals plus forced insertions at each detected apex and brake point. This guarantees that the velocity profile's critical features are always captured regardless of current vehicle speed.

### Cost Functional for Trajectory Evaluation

Candidate trajectories are scored against six weighted terms:

    C = Σ_{i=0}^{N} ∫_0^T w_i · c_i(t) dt

Terms include: lateral deviation from race line, curvature deviation, velocity deviation from the online profile, collision risk, collision severity (relative speed), and acceleration limit violations. The online velocity profile contributes directly to two of these (velocity deviation and acceleration feasibility), so its accuracy is load-bearing for trajectory ranking.

---

## Results

All experiments use the Yas Marina Circuit (Abu Dhabi) in simulation. The vehicle is a Super Formula car with A2RL modifications, running on an Intel Core i7-1270P.

### Scenario 1 — Longitudinal Recovery
The vehicle starts ~20 m/s below the offline velocity profile. The spatial sampling strategy correctly places samples at the upcoming brake point, allowing the trajectory cost to accurately penalize infeasible velocities. Temporal sampling missed the brake point in at least one condition, producing a plan that violated the acceleration constraint.

### Scenario 2 — Grip Reduction (α = 0.7), No Obstacle
A 600 m track sector has grip reduced to 70% of nominal. The online profile lowers the apex speed at s = 2520 m. Using the online profile, the planner tracks a feasible trajectory through the sector. Compared to using the frozen offline profile (which over-estimates available grip), the online approach gains **1.42 seconds** in sector time.

### Scenario 3 — Lateral Deviation + Grip Reduction (Multi-Vehicle)
A static obstacle forces the vehicle 6 m left of the racing line while α = 0.7 applies across the same sector and a 60 m/s speed cap is in effect. Online velocity profile integration yields a **1.18-second sector advantage**. The planner selects a trajectory that remains feasible under the reduced grip while navigating around the obstacle.

### Runtime
- Total planning step: **114 ms** average (10 Hz planning loop)
- Online velocity profile generation alone: **43 ms** average (Python, single core)
- Runtime scales linearly with the number of detected apexes and the discretization resolution Δs

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories with TOPP (Time-Optimal Path Parameterization) retiming and curvature-aware speed profiling for drone gate racing. Gate-7 in our helix section is persistently the worst gate at 0.284 m tracking error. Several aspects of this paper connect directly:

**Forward-backward solver pattern.** Our curvature-based speed profiling in `planning/racing_line.py` already applies a heuristic speed cap at high-curvature waypoints, but it does not implement a proper forward-backward sweep with coupled longitudinal-lateral acceleration constraints. Gate-7 sits at a helix curve with simultaneously high curvature and a 3D elevation change; a correct 3D forward-backward solver would respect the gravitational projection onto the drone's thrust axis, potentially producing a tighter but physically legal speed assignment that reduces the control tracking burden through that segment.

**Spatial-domain trajectory sampling.** Our trajectory is fully parametrized by arc length after min-snap, so the infrastructure for spatial-domain evaluation already exists. The paper's insight that a fixed temporal horizon can miss apex/brake features maps onto our setting: a fixed-time MPC horizon may not reach the Gate-7 entry point when the drone is decelerating hard approaching the helix. Switching from a temporal to a spatial replanning horizon for local trajectory refinement is directly applicable.

**Per-segment α scaling.** While our drone does not experience tire grip variation, our equivalent of α is effective thrust headroom, which varies with battery state and motor temperature. A segment-level constraint scaling factor that feeds the velocity optimizer could allow us to tighten the speed profile through Gate-7 without over-conservatively slowing the entire helix.

**Apex detection for gate alignment.** The paper's apex detection (curvature maxima → velocity boundary condition) maps directly to our gate-passing constraints. Each gate is a hard lateral constraint that also implies a speed boundary: too fast and cross-track error spikes; too slow and we lose lap time. The paper's iterative apex refinement (V_new = √(â_y / Ω_z)) could be adapted for drones using the thrust-to-weight ratio and gate approach curvature in place of the lateral g-limit.

---

## Actionable Takeaways

1. **Implement a proper forward-backward sweep in `planning/racing_line.py`.** Replace the current heuristic curvature speed cap with a sweep that propagates velocity constraints forward (thrust-limited acceleration) and backward (drag + gravity-limited deceleration) from gate waypoints. Gate-7's high curvature will naturally receive a tighter apex velocity, reducing the overshoot that drives its 0.284 m error.

2. **Force spatial sample placement at gate centers.** In the local trajectory replanning (if any exists in the pipeline), ensure that gate positions are always included as explicit sample nodes rather than relying on temporal discretization to land near them. Even in a pre-planned min-snap trajectory, re-parameterizing the time allocation to guarantee minimum segment times around Gate-7 could help.

3. **Add gravitational projection to 3D speed profiling.** Our helix involves significant elevation change. The drone's effective horizontal thrust is reduced by the cosine of the bank angle. Adding the gravitational component (a_z component of the flight path) into the speed profile optimizer — as this paper does for banked corners — would produce tighter, more physically grounded velocity assignments through the helix.

4. **Consider asynchronous velocity profile updates.** The paper runs its velocity optimizer at 43 ms on a single Python core; our pipeline runs at >100 Hz. If we pre-plan the velocity profile offline and cache a per-gate speed table, we get the main benefit (gate-aligned apex speeds) with zero runtime cost. Only dynamic replanning (e.g., battery sag) would require online recalculation.

5. **Adapt the g-diagram shape parameter ρ for drone dynamics.** For a quadrotor, the combined thrust-torque constraint approximates ρ ≈ 2 (circular) in roll-pitch coupling. Calibrating ρ from our observed tracking error distribution could improve how tightly the speed profile packs the gate passages without causing crashes.

---

## Limitations & Caveats

**Point-mass model.** The paper explicitly acknowledges that its quasi-steady-state point-mass assumption ignores transient dynamics (pitch transients, rotor inflow lag for drones). For a ground vehicle at 60 m/s on a 1 m-resolution track, this is a reasonable approximation; for a drone executing a 0.5-second gate passage at 15 m/s, transient aerodynamic effects may be non-negligible, meaning the forward-backward speed profile may still be slightly over-optimistic.

**Race-line dependency.** The feasibility guarantee holds only on the nominal race line. For lateral deviations (e.g., our drone drifting 0.28 m off the gate center at Gate-7), the velocity profile may no longer be strictly feasible. The paper acknowledges this and lists it as a future work item.

**Jerk discontinuities.** The forward-backward solver produces bang-bang velocity profiles with instantaneous acceleration switches. For a drone with attitude-command actuation, large jerk inputs excite flexible body modes and IMU aliasing. Smoothing the velocity profile (e.g., via jerk-limited rounding of the velocity profile at switch points) would be necessary before using it directly in the attitude command chain. Our min-snap formulation already minimizes snap (4th derivative), so the velocity profile from this approach would need to be re-fit through a min-snap layer.

**Python runtime.** 43 ms in Python for the velocity optimizer is too slow for our 100 Hz loop. A C++ or NumPy-vectorized implementation would be needed for online use. For offline pre-computation this is not a concern.

**No aerodynamic drag in velocity model.** The paper targets ground vehicles where drag is significant but modeled separately from the g-diagram. For a drone, aerodynamic drag is velocity-squared dependent and materially affects deceleration limits. Ignoring drag would make the backward sweep too pessimistic (over-estimate braking distance) or too optimistic depending on which direction the approximation rounds.

---

## Key Parameters / Constants

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Δs | 1.0 m | Race line discretization resolution |
| h_opt | 600 m | Velocity optimization lookahead horizon |
| T | 4 s | Temporal planning horizon (baseline, replaced by spatial) |
| Planning step | 100 ms | Sampling-based planner update rate |
| Online profiler runtime | 43 ms | Per-step wall time (Python, single core) |
| Total planning step | 114 ms | Full planning cycle including trajectory ranking |
| α | 0.7 (test) | Grip scaling factor (1.0 = nominal grip) |
| ρ | vehicle-specific | G-diagram shape exponent (ρ = 2 → circular) |
| ε | small (unspecified) | Apex velocity convergence threshold |
| Speed cap (scenario 3) | 60 m/s | Regulatory speed limit in multi-vehicle test |
| Sector time gain (scenario 2) | 1.42 s | Online vs. offline profile advantage |
| Sector time gain (scenario 3) | 1.18 s | Online vs. offline profile in multi-vehicle case |
| Track | Yas Marina (Abu Dhabi) | Simulation environment |
| Hardware | Intel Core i7-1270P | Test processor |
