# Online Velocity Profile Generation for Autonomous Racing (2025)

**Paper:** arXiv:2505.05157
**Authors:** Ogretmen et al.
**Venue:** IEEE Intelligent Vehicles Symposium (IV 2025), Cluj, Romania

## Key Contribution

This paper presents an online forward-backward velocity profile optimizer that adapts in real time to changing dynamic constraints (e.g., grip variations from tire temperature, surface conditions) during autonomous racing. The key advance over offline velocity profiles is the ability to recompute the feasible speed envelope at each planning step, enabling the vehicle to exploit higher speeds when grip is available and slow down appropriately when it degrades. Combined with a novel spatial (rather than temporal) trajectory sampling strategy, the planner achieves 1.42s faster times over a 600m section compared to a fixed offline profile under reduced-grip conditions.

## Technical Approach

**Forward-backward velocity optimization:**

The algorithm divides the upcoming track (horizon h_opt = 600m, discretized at ds = 1.0m) into segments at detected apex locations (local curvature maxima). It then runs two passes:

1. **Forward pass** — integrates acceleration potential:
   `a_x,fw = min{a_x,eng, a_x,max * (1 - (a_y / (alpha * a_y,max))^rho)^(1/rho)}`
   where alpha is a grip scaling factor in (0,1], rho = 1.3 is a shape morphing exponent for the g-diagram, and a_y is the lateral acceleration demand from curvature. The velocity update is: `V_{i+1} = min(sqrt(max{V_i^2 + 2*a_x,fw*ds, 0}), V_max)`.

2. **Backward pass** — integrates deceleration limits analogously, producing V_bw.

3. **Feasible profile:** V_feas = min{V_fw, V_bw} at each discretization point.

Apex locations are refined iteratively using a fixed-point convergence algorithm, with initial velocity guesses at apexes computed as `V_guess = sqrt(alpha) * V_offline`.

**G-diagram constraint model:**
The combined longitudinal-lateral acceleration envelope is modeled as: `(a_x / (alpha * a_x,lim))^rho + (a_y / (alpha * a_y,max))^rho <= 1`, where rho = 1.3 produces a diamond-like shape more realistic than a simple friction circle. The grip scaling factor alpha linearly scales all vertices of the diamond, representing global grip changes.

**3D track representation:**
The track is parameterized as C = {c(s) in R^3 | s in [0, s_lap]} with Euler angle orientation (phi, mu, theta) in zyx convention. Apparent accelerations incorporate banking angles and gravity: `a_z = w_y*v + g*(cos_mu * cos_phi)`. This enables proper velocity planning on banked turns and elevation changes.

**Spatial sampling for local trajectory planning:**
Instead of temporal sampling (fixed time horizon), the planner samples trajectories in the spatial domain (fixed arc-length horizon S). The key transformation converts temporal derivatives to spatial: `n'(s) = n_dot(t) / s_dot(t)`, `n''(s) = (n_ddot(t) - n'*s_ddot(t)) / s_dot(t)^2`. This produces third-order polynomial longitudinal profiles (vs. fourth-order temporal), and the paper demonstrates that spatial sampling tracks the velocity profile and brake/apex points with significantly higher geometric fidelity.

## Results

- **Computational cost:** 43ms for velocity profile generation alone, 114ms total trajectory planning cycle (Python on Intel i7-1270P). Fits within 100ms planning step.
- **Reduced grip scenario (alpha = 0.7, 600m section):** Online planner 13.55s vs. offline 14.97s — 1.42s faster (9.5% improvement).
- **Lateral deviation + reduced grip scenario:** 1.18s time advantage for online over offline.
- **Spatial vs. temporal ablation:** Temporal sampling decelerates too early and accelerates before the apex; spatial sampling accurately captures brake points and apex locations.

**Test track:** Yas Marina Circuit (Abu Dhabi), turns 6-7 featuring mixed high/low-speed sections with high curvatures.

## Relevance to Our System

**Current challenge:** We pre-compute a min-snap polynomial trajectory with curvature-aware speed profiling in `planning/racing_line.py`. The speed profile is computed offline and remains fixed during the race. At 14.08s race time, our trajectory is moderately aggressive. To push toward 12s, we need to increase speeds on straights and through gates, but our fixed profile cannot adapt to actual tracking performance or aerodynamic conditions during flight.

This paper is directly applicable in several ways:

1. **Adaptive speed profiling:** Our `racing_line.py` computes a fixed velocity profile based on curvature and drone constraints. The forward-backward solver from this paper could replace our static profile with one that adapts online. When tracking error is low (drone is performing well), alpha can be set higher to allow faster speeds. When error is high (approaching a difficult gate), alpha drops to provide more margin.

2. **The spatial sampling insight is critical for drones:** Our min-snap trajectory is parameterized in time, but the paper shows spatial parameterization produces better geometric fidelity. For gate racing, hitting the right spatial point (gate center) matters more than hitting it at exactly the right time. Re-parameterizing our trajectory segments by arc length could improve gate tracking.

3. **The g-diagram concept translates to drone dynamics:** For drones, the combined acceleration constraint is the thrust envelope: total thrust must provide both lateral acceleration (for turns) and vertical acceleration (for altitude). The diamond-diagram constraint with rho = 1.3 is a good approximation for drone thrust allocation between horizontal and vertical channels.

4. **Online re-optimization at 43ms is fast enough:** Our control loop runs at >100 Hz (10ms), and trajectory re-optimization at ~50ms intervals is feasible as a slower outer loop. This matches the hierarchical architecture we already use (trajectory planner -> controller).

## Actionable Takeaways

1. **Implement a forward-backward velocity optimizer in `planning/racing_line.py`:** Replace the current static curvature-based speed profile with the FW-BW solver. Key parameters: ds = 0.5m (finer than the paper's 1.0m for our smaller track), rho = 1.3, h_opt = full track length (our tracks are <200m total).

2. **Add an adaptive grip factor alpha:** Initialize alpha = 1.0, then modulate based on recent tracking error. If avg error over last 2 gates is below 0.15m, increase alpha toward 1.2 (push harder). If error exceeds 0.20m, reduce alpha to 0.8 (back off). This creates a natural speed-accuracy tradeoff mechanism.

3. **Consider spatial re-parameterization of trajectory segments:** Convert our time-parameterized min-snap polynomials to arc-length parameterization for the tracking controller. This decouples "where to be" from "when to be there" and lets the velocity profile handle timing independently.

4. **Use apex detection for segment time allocation:** Our current trajectory optimizer allocates time per segment heuristically. The paper's apex detection (curvature maxima) could inform better time allocation — spend less time on straights, more time on tight turns.

5. **Estimated speed gain:** The paper shows ~9.5% time improvement from online vs. offline velocity profiles under degraded conditions. For our system, going from static to adaptive profiling could yield 1--1.5s improvement (14.08s -> 12.6--13.0s), getting us close to the 12s target.

## Limitations

- **Designed for ground vehicles:** The g-diagram, tire grip model, and track representation assume a car. Direct application to drones requires replacing the friction model with a thrust envelope model and adapting the 3D track to a gate sequence.
- **Point-mass dynamics:** The paper uses a point-mass model that neglects transient effects (yaw dynamics, suspension, etc.). For drones, this means ignoring attitude dynamics, which matters at high angular rates during aggressive gate transitions.
- **No jerk regularization:** The velocity profile produces "instantaneous changes in acceleration" — discontinuous acceleration profiles that would cause aggressive snap in our min-snap trajectory. We would need to add smoothing or jerk constraints.
- **Feasibility guarantee only on racing line:** The profile is guaranteed feasible only when the drone is exactly on the reference path. Off-path deviations (common when tracking error is nonzero) may violate the assumed curvature and hence the velocity limits.
- **Python implementation at 43ms:** This is borderline for real-time. A C++/Cython implementation would be needed for reliable real-time performance.

## Key Parameters

| Parameter | Paper Value | Suggested Drone Value |
|-----------|-------------|----------------------|
| Discretization ds | 1.0 m | 0.5 m |
| Optimization horizon h_opt | 600 m | Full track (~150m) |
| Planning step | 100 ms | 50--100 ms |
| Shape exponent rho | 1.3 | 1.3 (start here) |
| Grip factor alpha | 0.7--1.0 | 0.8--1.2 (adaptive) |
| V_max | 60 m/s | 15--20 m/s (drone limit) |
| Temporal planning horizon T | 4 s | 2--3 s |
| Compute time (velocity opt) | 43 ms (Python) | Target <20 ms |
| Compute time (full planner) | 114 ms | Target <50 ms |
