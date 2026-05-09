# Gate-Aware Online Planning for Two-Player Autonomous Drone Racing

- **URL**: https://arxiv.org/abs/2402.18021
- **Authors**: Fangguo Zhao, Jiahao Mei, Jin Zhou, Yuanyi Chen, Jiming Chen, Shuo Li (Zhejiang University; Zhejiang University of Technology)
- **Year**: 2024 (submitted February 28, 2024; revised September 23, 2024)
- **Venue**: arXiv preprint, cs.RO

---

## Key Contribution

The paper introduces **Pairwise Model Predictive Control (PMPC)**, an online real-time planning framework for two-drone competitive racing. Its central novelty is the **Magnetic Induction Line (MIL)** — a spatial reference curve that connects consecutive gates while being, by construction, **perpendicular to each gate plane at entry and exit**. This eliminates the need to manually place entry/exit waypoints along gate normals: the MIL naturally encodes the gate traversal geometry from the physics of magnetic dipole fields.

The second contribution is joint two-agent optimization: rather than treating the opponent as a static obstacle, PMPC co-optimizes both drones' trajectories with a collision-avoidance cost that is only active until the first predicted collision time, avoiding over-conservatism when paths diverge post-crossing.

The key departure from prior work (TOGT, MINCO, min-snap) is that those planners reduce gate traversal to waypoint-passing and rely on trajectory smoothness to accidentally satisfy gate orientation constraints. PMPC enforces perpendicular gate crossing as a hard geometric constraint embedded in the reference curve.

---

## Technical Approach

### 1. Point-Mass Reference Trajectory

Before computing the MIL, the system generates a time-optimal point-mass trajectory via a two-step Dijkstra-based velocity search:

**Step 1 (coarse direction-fixed search):** For each waypoint `i`, the nominal velocity direction is set as the unit vector from waypoint `i-1` to waypoint `i`. Twenty velocity magnitudes `{1*v_i, 2*v_i, ..., 20*v_i}` are sampled. A search graph is built where edge cost encodes kinematic feasibility under point-mass dynamics. Dijkstra's algorithm finds the globally optimal speed sequence.

**Step 2 (refined directional search):** At each waypoint, a cone of 20 velocity directions around the optimal magnitude from Step 1 is sampled (different directions, same optimal speed). Dijkstra runs again on this refined graph to find the truly optimal velocity vector at each waypoint — allowing the trajectory to deviate from the straight inter-gate direction when beneficial.

The polynomial connecting adjacent waypoints uses the **minimum angular acceleration yaw polynomial** with coefficients computed via qpOASES. Segment time equals the point-mass travel time `T_i*`.

This two-stage sampling achieves near-time-optimal speed profiles without a full nonlinear optimization. The point-mass trajectory is used as a "backbone" initialization that the MIL and MPC refine near gates.

### 2. Magnetic Induction Line (MIL) — the Core Gate Traversal Method

**Physical analogy:** Each gate is modeled as a 3D magnetic dipole. The dipole moment vector `m_i` is set to be **aligned with the gate normal** (perpendicular to the gate plane). A magnetic dipole field has the property that field lines connect the north and south poles perpendicularly — by setting the dipole along the gate normal, field lines automatically arrive at and depart from the gate plane at right angles.

**Field computation:** The magnetic field at an arbitrary point in space due to two adjacent dipole gates is:

```
B_k = sum_{i=0,1}  (mu_0 / 4*pi) * [ 3*(m_i . r_i)*r_i / ||r_i||_2^5  -  m_i / ||r_i||_2^3 ]
```

where `r_i = xi_k^m - xi_i^g` is the displacement from gate `i` to the query point `xi_k^m`, and `mu_0` is the vacuum permeability constant.

**MIL optimization problem:** The MIL is not computed by analytically integrating field lines; instead, it is found by solving a discrete optimization over `N_m` sample points:

```
minimize  sum_{k=1..N_m}  [ ||xi_k^m - xi_k^pm||_2^2  +  ||xi_k^m - xi_{k-1}^m||_2^2 ]

subject to:
  (xi_k^m - xi_{k-1}^m) / ||xi_k^m - xi_{k-1}^m||_2  =  B_k     (tangency: each step aligns with local B field)
  B_k . B_{k-1}  >  0.99                                           (smoothness: no field reversal = no curling)
```

The two objective terms serve complementary roles:
- **Term 1** (`||xi_k^m - xi_k^pm||_2^2`): Minimizes deviation from the point-mass reference trajectory. Between gates where field lines are nearly straight (weak dipole interaction), the MIL closely follows the point-mass path, preserving the time-optimal character.
- **Term 2** (`||xi_k^m - xi_{k-1}^m||_2^2`): Minimizes the distance between adjacent MIL points, preventing curling or folding near the strong-field regions close to each gate. This is the key curvature-control term: it penalizes sharp bends in the MIL.

The smoothness constraint (`B_k . B_{k-1} > 0.99`) additionally ensures the field direction does not reverse between adjacent points, which would indicate the MIL is wrapping around a pole — a sign of excessive local curvature. This dot-product constraint is a sufficient condition for bounded curvature along the MIL.

**Perpendicularity at gate entry/exit:** This is the critical geometric property. Because each dipole moment `m_i` is set parallel to the gate normal, the magnetic field exactly at the gate plane is parallel to `m_i` (i.e., parallel to the gate normal). Since the MIL is tangent to `B` everywhere (by the tangency constraint), the MIL direction at the gate plane crossing is automatically perpendicular to the gate plane. No special parameterization of entry/exit offsets or velocities is needed — the perpendicularity condition emerges from the physics of the dipole field.

**Only two dipoles per segment:** "MILs are assumed to be only affected by the two adjacent magnets/gates." Each inter-gate MIL segment is computed using only the fields from gates `k` and `k+1`. This is a simplification that keeps the computation tractable and maintains local spatial coherence.

### 3. Gate Heading Angle Optimization

Optimal yaw angles at each waypoint are found by minimizing a discounted future-gate visibility cost:

```
minimize_{psi_i}  sum_{k=i..N_wp-1}  gamma^(k-i) * || R(psi_i) * (xi_{k+1}^g - xi_i^g) ||_2^2
```

where `gamma in (0,1)` is a discount factor that down-weights distant future gates, and `R(psi_i)` is the 2D rotation matrix for yaw `psi_i`. The effect: at each gate, the drone yaws toward the next gate (high weight), with some awareness of gates further ahead (decaying weight). This mimics how expert human pilots steer — already looking at the next gate while exiting the current one.

Consecutive yaw angles are connected via **minimum angular acceleration polynomials**, computed with the same qpOASES solver as the positional trajectory.

### 4. PMPC Cost Function and MIL Activation

The PMPC operates at 200 Hz over a 20-step prediction horizon with **adaptive time step** (1–40 ms per step; finer near gates, coarser between them):

**State:** `x_j = [xi_j, v_j, q_j]` (position, velocity, quaternion) for drone `j`
**Control:** `u_j = [T_s^j, omega_j]` (thrust + angular velocity)
**Dynamics:** Full nonlinear quadrotor model

**Total cost:**
```
J_PMPC = J_track^1 + J_racing^1 + J_mil^1
       + J_track^2 + J_racing^2 + J_mil^2
       - J_col
```

**Tracking cost** (`J_track`): Quadratic deviation from the point-mass reference trajectory. Quaternion error is split: yaw weighted higher (`Q_yaw > Q_tilt`) since heading accuracy matters for gate visibility, while roll/pitch are left to the dynamics optimizer.

**Racing cost** (`J_racing`): Minimizes remaining distance to the next gate center `xi^g`:
```
J_racing^j = sum_{k=0..N}  ||xi_k^j - xi^g||_{Q_r}^2
```
Dynamic sigmoid weighting: large `Q_r` when far from gate (encourages aggressive approach), smaller `Q_r` near gate (shifts priority to MIL accuracy).

**MIL alignment cost** (`J_mil`): Only activated when the drone is within a proximity threshold of the gate. Penalizes perpendicular distance from the MIL reference curve:
```
J_mil^j = sum_{k=0..N}  ||d_tilde_k^j||_{Q_m}^2
```
where `d_tilde_k = d_k - (d_k . m_i) * m_i` is the component of the position error perpendicular to the gate normal (i.e., the off-axis deviation from the MIL). Motion along the gate normal is unconstrained — this term does not impede forward progress through the gate, only corrects lateral/off-axis error.

**Collision avoidance cost** (`J_col`): Maximizes separation between the two drones, but only from `t=0` to `t_c` (the first collision time predicted from the point-mass trajectories). After `t_c`, no collision cost applies, exploiting the high replanning frequency to handle post-collision-time divergence reactively.

### 5. Solver

- Backend: **acados** with **hpipm** embedded QP solver
- Solve time: ~1 ms (laptop CPU), ~4 ms (Coolpi 4B ARM embedded computer)
- Control frequency: 200 Hz

---

## Results

### Simulation (7-gate figure-8 track)

| Method | Top Speed | Lap Time | Compute Time |
|--------|-----------|----------|--------------|
| TOGT (single-drone, MINCO, offline) | 17.5 m/s | 12.56 s | 0.018 s offline |
| Swarm-CPC (multi-drone, offline) | 22 m/s | 10.6 s | 23,250 s offline |
| **PMPC (proposed, online)** | **18.7 m/s** | **12.51 s** | **~4 ms online** |

PMPC matches TOGT lap time to within 0.05 s while running fully online. Swarm-CPC is 2 seconds faster but requires 6+ hours of offline computation. Min inter-drone distance: 0.68 m (above 0.5 m safety threshold).

Gate traversal test at varying gate orientations (0°, 30°, 45°, 60° yaw): PMPC achieved correct perpendicular crossing at all orientations. Naive waypoint-based planners failed at 45° and 60°.

### Real-World Experiments

- Arena: 5 m × 4 m × 2 m (compact — arena-limited, not dynamics-limited)
- Track: figure-8 with 7 waypoints, mixed gate orientations
- Top speed: **6.1 m/s**
- Collision-free across all trials
- Correct heading toward subsequent gates at all times

---

## Relevance to Our System

Our system generates min-snap polynomial trajectories with entry/exit waypoints placed at fixed ±0.4 m offsets along gate normals. The problem: at the helix entry (gate-7, 68.5° turn), the 0.4 m pre- and post-gate segments are very short relative to the curvature demanded by the turn. The min-snap optimizer, in order to satisfy endpoint positions and velocities within these short segments, produces extremely high polynomial coefficients — effectively infinite curvature in the limit. The drone physically cannot follow this; it either crashes or deviates massively, causing high tracking error.

### How the MIL paper directly addresses this

**The MIL approach fundamentally reframes the entry/exit problem.** Instead of placing explicit entry/exit waypoints at fixed offsets from the gate, the MIL lets the dipole physics determine where the perpendicular approach begins. Near each gate, the field lines curve from the inter-gate "straight" region into the gate-perpendicular direction. The **transition from straight to perpendicular happens gradually over a physically determined length** — not abruptly at a fixed 0.4 m offset.

For a sharp turn (e.g., 68.5°), the MIL field line from the previous gate would curve smoothly over a longer spatial extent, distributing the direction change across a larger arc length. The second objective term in the MIL optimization (`||xi_k^m - xi_{k-1}^m||_2^2`) explicitly penalizes sharp local bends — this is curvature regularization. The smoothness constraint (`B_k . B_{k-1} > 0.99`) enforces a bound on the local field direction change per step, equivalent to bounding curvature.

**The key insight for our 0.4 m offset problem:** the 0.4 m fixed offset creates a curvature problem because we are specifying both position and velocity at two points only 0.4 m apart in a high-angle turn. The MIL avoids this by not specifying an explicit offset distance. Instead, the approach direction constraint (perpendicular to gate normal) is satisfied asymptotically, with the field line becoming perpendicular only exactly at the gate plane. The transition region is allowed to be as long as needed by the physics of the dipole field — for a 68.5° turn between adjacent gates, the turn radius is effectively spread over the entire inter-gate segment, not crammed into 0.4 m.

### Practical implication for our 0.4 m offset parameterization

The paper suggests two actionable changes to our waypoint placement:

1. **Variable offset distance based on turn angle:** For shallow turns, 0.4 m may be fine. For sharp turns, the offset should be larger to give the polynomial enough distance to curve. The MIL field line length (from the gate plane to where it joins the inter-gate straight) grows with turn angle — at 68.5°, a 2–3 m offset would be more appropriate than 0.4 m.

2. **Decoupling position and velocity constraints at entry/exit:** Our current approach sets both position (offset point) and velocity (along next gate direction) as hard constraints. The MIL approach effectively only constrains the *direction* at the gate plane (perpendicular) while allowing position to float along the gate plane and velocity magnitude to be determined by the optimizer. Loosening the position constraint (or using a longer offset) while keeping the velocity direction constraint would reduce the curvature demanded by the polynomial.

### Adaptive time step near gates

PMPC uses dt = 1–40 ms, with fine steps near gates. Our system uses fixed dt = 0.01 s. For the helix entry where tracking error spikes, reducing dt in the 1–2 m approach window would give the tracker finer reference points and reduce the jump between consecutive waypoints.

### The heading lookahead mechanism

PMPC's heading optimizer is particularly relevant to our helix entry. At gate-7 (68.5° turn), the drone must yaw sharply. The gamma-discounted heading cost causes the drone to begin yawing toward gate-8 while still crossing gate-7, distributing the yaw change over a longer time window. Our current yaw generation from instantaneous velocity direction means yaw lags the trajectory, creating a moment of high angular rate exactly at the gate crossing — the worst time for a precision maneuver.

---

## Actionable Takeaways

1. **Increase entry/exit offset distance for high-curvature gates proportional to turn angle.** For gate-7 (68.5° turn), replace the fixed 0.4 m offset with a turn-angle-dependent offset: `d_offset = max(0.4, 0.4 * tan(turn_angle / 2))`. At 68.5°, this gives ~0.7 m. Or more aggressively, use `d_offset = 0.4 + k * sin(turn_angle)` tuned empirically. The MIL analogy suggests the offset should scale with the spatial extent of the turning arc.

2. **Relax the position constraint at gate entry/exit waypoints; keep only the velocity direction constraint.** Currently `TrajectoryOptimizer` enforces both position (at the offset point) and velocity (toward the next gate). For sharp turns, the combination over a short segment is the curvature culprit. Consider removing the entry/exit position waypoints entirely and instead using **velocity boundary conditions at the gate center itself** — perpendicular to the gate normal — which is exactly what the MIL enforces.

3. **Implement MIL-inspired curvature regularization in the segment time allocation.** The MIL's second term penalizes adjacent-point distance (equivalently, constrains curvature). Our `_optimize_time_allocation()` does not account for curvature. Add a curvature penalty: when the turn angle between consecutive gate segments exceeds a threshold (e.g., 45°), increase the segment time for both the pre-gate and post-gate segments, reducing curvature by stretching the polynomial over a longer time interval.

4. **Two-stage velocity search for time-optimal speed profiling.** Replace the single L-BFGS pass in `_optimize_time_allocation()` with the PMPC paper's two-step Dijkstra approach: first sample 20 speeds per segment along the nominal direction, then resample 20 directions around the optimal speed. This is more systematic and less prone to local optima, especially at the helix entry where the nominal direction changes sharply.

5. **Heading lookahead toward next gate.** Replace instantaneous velocity-derived yaw with a lookahead yaw that blends `atan2` to the *next* gate rather than the current velocity direction. For gate-7→gate-8, begin yawing toward gate-8's center 1–2 m before reaching gate-7. This is the direct implementation of the PMPC heading objective (Eq. 3), simplified by dropping the discount factor and using only the next gate.

6. **Adaptive sampling density near high-curvature gates.** Near gates where `|turn_angle| > 45°`, halve `dt_sample` in the trajectory sampler for the 1–2 m approach and departure windows. This provides the MPC tracker with finer reference points and reduces tracking error at sharp transitions.

7. **Do not end the trajectory at a gate center; always extend past the last gate along its exit normal.** (This is the MIL's fundamental message: gates are pass-through constraints, not endpoints.) For our last gate (gate-12), add an extension waypoint 2–3 m past the gate along its normal. For gate-7 specifically (helix entry), the exit waypoint should be placed further downstream — enough to ensure the post-gate segment has a smooth, low-curvature connection to the helix arc.

---

## Limitations & Caveats

1. **Small-scale experiments only.** Real-world validation is in a 5×4×2 m arena at 6.1 m/s. Competition-scale courses (20+ m, 15+ m/s) are not demonstrated. The MIL curvature control may need retuning at higher speeds where aerodynamic drag and rotor dynamics matter.

2. **Two-drone-specific.** The PMPC formulation is explicitly for two drones. The collision avoidance term `J_col` is paired between exactly two agents. For single-drone racing (our case), `J_col` disappears and the framework reduces to single-agent gate-aware MPC — still directly applicable but the paper's discussion of `J_col` is irrelevant.

3. **Static gate configuration.** The MIL requires known, fixed gate positions and orientations (dipole moments must be set from gate normals). Dynamic or unknown gates would require MIL recomputation online.

4. **MIL only spans adjacent gates.** The dipole model only considers the two gates bounding each inter-gate segment. For our helix where gates are closely spaced and have large orientation changes, the next-segment gate's field could non-negligibly influence the current MIL. The authors note this as a simplification; for our helix, it means the MIL near gate-7 does not anticipate gate-8's orientation constraint.

5. **200 Hz MPC requirement.** The system uses acados/hpipm at 200 Hz. Our pipeline targets 100 Hz with a simpler geometric tracker. Full PMPC adoption would require solver integration and likely embedded hardware upgrades for competition.

6. **No state estimation treatment.** The paper assumes perfect state knowledge. Position errors from VIO drift or EKF uncertainty are not addressed. At 15+ m/s, a 0.1 m position error at gate approach creates significant crossing error; MIL-based references are sensitive to this.

7. **MIL offset distance not explicitly quantified.** The paper does not state how far from each gate plane the MIL transitions from perpendicular to parallel. For implementation, this spatial extent must be determined by the dipole field geometry (depends on the ratio of inter-gate distance to gate size), adding a tuning burden.

---

## Key Parameters / Constants

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Velocity samples per waypoint, Step 1 | 20 | Coarse Dijkstra graph magnitudes |
| Velocity samples per waypoint, Step 2 | 20 | Refined Dijkstra graph directions |
| MPC prediction horizon | N = 20 steps | Online lookahead |
| Adaptive dt range | 1–40 ms per step | Fine near gates, coarse between |
| Effective MPC lookahead (total) | 0.02–0.8 s | Depending on proximity to gate |
| Rotor thrust bounds | 1–40 N | Actuator limits |
| Angular velocity: pitch/roll | ±5 rad/s | Actuator limits |
| Angular velocity: yaw | ±1 rad/s | Actuator limits |
| Heading discount factor | gamma in (0,1) | Future-gate visibility weight |
| Q_yaw weighting | > Q_tilt | Prioritize heading accuracy |
| MIL smoothness constraint | B_k . B_{k-1} > 0.99 | Bounded curvature along MIL |
| Collision safety margin | delta_col = 0.5 m | Minimum inter-drone separation |
| Downwash relief matrix | E = diag(1, 1, 1/3) | Relax vertical separation req. |
| MIL gate proximity threshold | Not explicitly stated (~1–2 m) | Switches J_track → J_mil |
| Control frequency | 200 Hz | Online MPC replanning rate |
| Solver backend | acados + hpipm | Embedded QP |
| Solve time (laptop) | ~1 ms | |
| Solve time (Coolpi 4B ARM) | ~4 ms | Embedded computation budget |
| Drone mass (real-world) | 300 g | Quadrotor platform |
| Max speed (simulation) | 18.7 m/s | |
| Max speed (real-world) | 6.1 m/s | Arena-limited |
| Lap time (simulation, 7 gates) | 12.51 s | |
| Arena size (real-world) | 5 m × 4 m × 2 m | |
| Number of gates tested | 7 | Simulation and real-world |
| Gate orientations tested (real) | 0°, 30°, 45°, 60° yaw | Position-swap experiments |
