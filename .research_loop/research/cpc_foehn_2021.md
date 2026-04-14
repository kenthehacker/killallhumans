# CPC: Complementary Progress Constraints for Time-Optimal Quadrotor Trajectories
- **URL**: https://arxiv.org/abs/2007.06255
- **Authors**: Philipp Foehn and Davide Scaramuzza (Science Robotics version adds Angel Romero)
- **Year**: 2021
- **Venue**: Science Robotics (extended version of arXiv 2020 preprint); DOI: 10.1126/scirobotics.abh1221

## Key Contribution

CPC introduces a fundamentally new way to solve the time-optimal trajectory planning problem for quadrotors passing through multiple waypoints. The central insight is that the **time allocation to waypoints** (i.e., when the drone should reach each gate) should not be fixed a priori but optimized jointly with the trajectory itself. Prior polynomial-based methods (min-snap) fix the time allocation then optimize the trajectory, or iterate between them. Prior numerical optimization methods assign waypoint constraints to specific discrete time nodes, which presupposes the time allocation. CPC sidesteps both limitations by introducing **complementary progress variables** that let the optimizer freely decide when each waypoint is visited, enabling truly simultaneous optimization of the spatial trajectory and temporal allocation.

To the authors' knowledge, this is the first approach that produces truly time-optimal trajectories for quadrotors through multiple waypoints while respecting full rotational dynamics and individual rotor thrust limits.

## Technical Approach

### The Complementarity Constraint Formulation

The core idea borrows from optimization under contacts (complementarity problems in robotics). For each waypoint j, a **progress variable** lambda_j tracks whether the waypoint has been completed:
- lambda_j = 1 means waypoint j is incomplete
- lambda_j = 0 means waypoint j has been completed

A **progress change variable** mu_k^j represents progress at time node k for waypoint j. The evolution is:

    lambda_{k+1} = lambda_k - mu_k

The key constraint is the **complementarity condition**:

    mu_k^j * (||p_k - p_wj||^2 - nu_k^j) = 0,   0 <= nu_k^j <= d_tol^2

This enforces that progress change (mu) can only be nonzero when the drone position p_k is within distance d_tol of waypoint p_wj. In other words, a waypoint can only be marked complete when the drone is actually near it. When the drone is far from a waypoint, mu must be zero and the progress variable cannot change. This elegantly encodes "pass through each waypoint" without specifying when.

Additional constraints enforce ordering: lambda_k^j - lambda_k^{j+1} <= 0, ensuring waypoints are visited in sequence. Boundary conditions require lambda_0 = 1 (all incomplete) and lambda_N = 0 (all complete).

### NLP Structure

The optimization problem is a direct multiple-shooting NLP:

**Objective:** minimize t_N (total trajectory time)

**Decision variables per node k:** dynamic state x_dyn,k, control input u_k, progress lambda_k, progress change mu_k, slack nu_k

**Equality constraints:**
- RK4 dynamics integration: x_{k+1} - x_k - dt * f_RK4(x_k, u_k) = 0
- Progress evolution: lambda_{k+1} - lambda_k + mu_k = 0
- Boundary conditions on lambda
- Complementarity constraints (above)

**Inequality constraints:**
- Individual rotor thrust bounds: T_min <= T_i <= T_max
- Progress ordering constraints
- mu_k^j >= 0

The dynamics model includes full 3D rotational dynamics with quaternion attitude, individual rotor thrusts (not just collective thrust), and a linear aerodynamic drag model:

    v_dot = g + (1/m) R(q) T - c_D * v

where the drag coefficient c_D is chosen so that at maximum horizontal thrust, steady-state velocity equals v_max.

### Why This Outperforms Min-Snap

Polynomial trajectories are inherently smooth. A minimum-snap polynomial can only touch input limits at infinitesimally brief instants -- it cannot sustain bang-bang-like actuation profiles that exploit the full actuator potential. CPC trajectories exhibit extended periods of actuator saturation (full thrust on some rotors, zero on others), extracting maximum performance from the hardware. The paper demonstrates this visually: CPC trajectories show characteristic thrust profiles with sustained saturation, while min-snap trajectories have sinusoidal, smooth thrust profiles that never fully exploit the motors.

## Results

### Trajectory Time Comparisons

**Hover-to-hover (15m):** CPC achieves 1.933s vs 1.895s (2D optimal) and 1.885s (2D optimal variant). CPC is 2-2.7% slower in this simple case because it models full 3D rotational dynamics (the 2D methods ignore rotation time).

**Multi-waypoint (5 waypoints, 50m):** 2.430s

**Slalom track (10 waypoints):** 8.644s

**NeurIPS AirSim Qualification (21 waypoints):** 24.11s vs competition best 30.11s (20% theoretical improvement)

**vs Human Pilots (simulation, figure-8):** CPC 9.621s vs human 10.338s (7.1% faster)

**vs Human Pilots (real-world, hairpin):** CPC 0.874s vs human 0.984s (11.2% faster)

### Solve Times

This is the critical weakness: solve times range from **1-40 minutes** for simple scenarios to **hours** for longer tracks (e.g., 21 waypoints). The method uses IPOPT or similar interior-point NLP solvers. Iteration counts range from 216 (good initialization) to 303 (poor initialization) for a hairpin turn.

## Relevance to Our System

Our system uses a two-phase approach:
1. **Racing line optimization** (L-BFGS on lateral offsets within gate openings)
2. **Min-snap polynomial trajectory** with L-BFGS time allocation optimization

This is architecturally different from CPC in several important ways:

**What CPC does better:** CPC jointly optimizes the spatial path and time allocation in a single NLP, with full quadrotor dynamics including individual rotor limits. Our approach separates spatial path (racing line) from temporal allocation (segment times), and uses a simplified dynamics model (velocity/acceleration/jerk limits rather than rotor thrust limits). CPC can find trajectories that exploit actuator saturation, while our min-snap polynomials are inherently smooth and suboptimal in this regard.

**What our approach does better:** Our approach runs in seconds (L-BFGS converges fast on smooth polynomial objectives), while CPC takes minutes to hours. For a real-time racing system that needs to replan, CPC is completely impractical. Our two-phase approach is a pragmatic compromise: it may not find the globally time-optimal trajectory, but it finds a good one fast enough to actually use.

**The gap:** CPC represents the theoretical performance ceiling. If CPC finds a 24.11s trajectory for a 21-waypoint track while a similar min-snap approach might find ~30s, the performance gap is roughly 20%. For our system (current race time ~14.6s on our track), adopting CPC-style optimization could theoretically shave 2-3 seconds -- but only offline.

**Practical hybrid:** The most actionable approach is to use CPC-style ideas for **offline precomputation** of reference trajectories, then track them with our existing controller. Specifically, the key insight about sustained actuator saturation could inform how we set our segment time bounds -- our current L-BFGS time optimizer may be too conservative because the underlying min-snap polynomials cannot represent the aggressive maneuvers that a truly time-optimal trajectory requires.

## Actionable Takeaways

1. **Offline trajectory precomputation:** CPC could generate a truly time-optimal reference trajectory offline (accepting multi-minute solve times). This trajectory would then be tracked by our existing geometric controller. This is exactly the approach the authors later used in their real-world experiments.

2. **Better time allocation initialization:** Our L-BFGS time optimizer could be warm-started with CPC's insight that time allocation should allow sustained actuator saturation between waypoints. Currently, our segment time optimization may converge to smooth, conservative solutions.

3. **Actuator-aware segment time bounds:** The drag model and thrust saturation analysis from CPC can inform tighter bounds on our segment times. The formula c_D = sqrt((4*T_max/m)^2 - g^2) / v_max provides a principled way to set maximum velocity given thrust limits.

4. **Progress variable concept for gate sequencing:** The complementarity constraint idea could improve our gate sequencing -- rather than hard-coding gate passage detection, a progress-variable approach could provide smoother transitions.

5. **Don't over-invest in polynomial smoothness:** CPC demonstrates that the smoothness of min-snap polynomials is fundamentally limiting. If we hit a performance ceiling with our current approach, the next step is moving to direct collocation with full dynamics, not further tuning polynomial parameters.

## Limitations & Caveats

1. **Computational cost is prohibitive for online use:** Minutes to hours for a single trajectory. This is strictly an offline planning tool. Any replanning during a race must use faster methods.

2. **Non-convex problem with local optima:** The acceleration space is non-convex when T_min > 0. Different initializations yield different solutions (1.3% variation observed). The authors mitigate this with a point-mass convex pre-solve for initialization, but global optimality is not guaranteed.

3. **No control authority margin:** Time-optimal trajectories saturate actuators by definition, leaving zero margin for disturbance rejection. In practice, the trajectory must be slightly suboptimal to allow the controller to compensate for model mismatch, wind, etc. The authors acknowledge this but do not address it.

4. **Simplified drag model:** The linear drag approximation c_D*v is a rough model. Real aerodynamic effects (blade flapping, ground effect, prop wash interactions) are not captured. Model mismatch grows at high speeds where these effects matter most.

5. **No perception constraints:** CPC optimizes purely for speed with no consideration of whether the drone can see upcoming gates. For our vision-based system, a time-optimal trajectory that points the camera away from the next gate is useless. Perception-aware planning (ETH 2025/2026 work) addresses this gap.

6. **Scaling:** The NLP grows linearly with the number of collocation nodes and waypoints. For long tracks, the solve time becomes impractical even for offline use without warm-starting or decomposition.

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_tol | 0.1-0.4m | Waypoint proximity tolerance for progress change |
| N (nodes) | 50-3360 | Total collocation nodes (50-100 per waypoint typical) |
| N_w | 50-100 | Nodes allocated per waypoint |
| dt | t_N/N | Variable timestep (optimized as part of t_N) |
| T_min | 0 or >0 | Minimum individual rotor thrust (T_min>0 makes problem non-convex) |
| T_max | platform-specific | Maximum individual rotor thrust |
| c_D | derived | Linear drag coefficient: sqrt((4*T_max/m)^2 - g^2) / v_max |
| Integration | RK4 | 4th-order Runge-Kutta for dynamics propagation |
| Solver | IPOPT | Interior-point NLP solver |
| Initialization | Convex pre-solve | Point-mass model solved first for warm start |
