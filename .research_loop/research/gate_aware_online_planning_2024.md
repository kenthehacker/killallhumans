# Gate-Aware Online Planning for Two-Player Autonomous Drone Racing

- **URL**: https://arxiv.org/abs/2402.18021
- **Authors**: Fangguo Zhao, Jiahao Mei, Jin Zhou, Yuanyi Chen, Jiming Chen, Shuo Li
- **Year**: 2024 (submitted February 28, revised September 23, 2024)
- **Venue**: arXiv preprint (cs.RO)

---

## Key Contribution

The paper introduces **Pairwise Model Predictive Control (PMPC)**, a real-time online planning framework for two-drone competitive racing. Its central novelty is twofold:

1. **Gate-awareness via Magnetic Induction Lines (MIL)**: Instead of treating gates as waypoints (simple 3D points to pass through), each gate is modeled as a 3D magnetic dipole. The spatial curve induced by the combined dipole fields of consecutive gates — the MIL — provides a smooth reference that is by construction **perpendicular to each gate plane at the entry and exit crossing points**. This guarantees proper gate traversal geometry regardless of gate orientation.

2. **Two-agent simultaneous optimization**: Rather than treating the opponent as a static obstacle, the PMPC jointly optimizes both drones' trajectories with a shared cost term. Collision avoidance is handled predictively by penalizing states only up to the first predicted collision time, which avoids over-conservative behavior when the drones separate after a near-pass.

The key departure from prior single-drone work (TOGT, MINCO) is explicitly modeling the gate orientation constraint, not just position. Most prior planners reduce gate traversal to a waypoint-passing problem and rely on the trajectory optimizer to accidentally satisfy the orientation requirement through smoothness alone. PMPC enforces perpendicular crossing as a hard geometric constraint embedded in the reference curve.

---

## Technical Approach

### 1. Point-Mass Reference Trajectory

Before constructing the MIL, the system generates a time-optimal point-mass trajectory using a two-step velocity sampling procedure inspired by Dijkstra:

- **Step 1**: For each waypoint, sample 20 velocity vectors along the waypoint-to-waypoint direction with varying magnitudes. Build a graph and find the globally optimal speed sequence using Dijkstra's shortest path on a graph where edge costs encode kinematic feasibility.
- **Step 2**: Around the optimal speed at each waypoint, resample velocities in a cone around the nominal direction to allow for direction deviation. Run Dijkstra again on this refined graph to find globally better velocity vectors. This two-stage sampling achieves near-optimal speed profiles without a continuous nonlinear optimization.

The point-mass trajectory is used as a "backbone" — MIL then wraps around it near each gate.

### 2. Magnetic Induction Line (MIL) for Gate Traversal

Each gate is modeled as a 3D magnetic dipole with its dipole moment vector set perpendicular to the gate plane (aligned with the gate normal). The magnetic field at an arbitrary position is:

```
B_k = sum_i (mu_0 / 4*pi) * [ 3*(m_i . r_i)*r_i / ||r_i||^5  -  m_i / ||r_i||^3 ]
```

where `m_i` is the magnetic moment of gate `i` and `r_i` is the displacement vector from gate `i` to the query point.

The MIL is the field line of this combined dipole field that connects two consecutive gates. By construction:
- The field line terminates **perpendicularly to each gate plane** at entry and exit, since the dipole moment is aligned with the normal.
- The field line deviates minimally from the point-mass trajectory in the inter-gate regions (between gates, the dipole fields are weak, so the line is nearly straight).
- Gate orientation is automatically handled: tilted or yaw-rotated gates produce the correct entry/exit approach angle without any explicit parameterization.

The MIL is computed numerically by integrating the field line from gate `k` to gate `k+1`.

### 3. Heading Trajectory Optimization

Optimal heading angles at each waypoint are found by minimizing a weighted future-gate visibility cost:

```
minimize  sum_i  gamma^(k-i) * || R(psi_i) * (xi_{k+1}^g - xi_i^g) ||^2
```

where:
- `gamma in (0,1)` discounts far-future gates
- `R(psi_i)` is the rotation matrix for yaw angle `psi_i`
- `xi^g` are gate positions in 2D (horizontal plane)

This causes the drone to yaw toward the next gate while approaching, keeping the gate in the camera field of view — directly analogous to the perception-aware planning objective in the ETH 2025 work.

### 4. Pairwise MPC Formulation

The PMPC operates over a prediction horizon of 20 steps with **adaptive time step** (1–40 ms, shorter near gates for precision, longer between gates for computational efficiency).

**Dynamics model**: Full quadrotor model with states (position, velocity, quaternion orientation) and controls (collective thrust and body angular rates). Constraints:
- Rotor thrust: 1–40 N per rotor
- Angular velocity: ±5 rad/s pitch/roll, ±1 rad/s yaw

**Cost function**:
```
J_PMPC = J_track + J_racing + J_mil - J_col
```

- `J_track`: Quadratic tracking error relative to the point-mass reference trajectory (maintains general course)
- `J_racing`: Minimizes remaining distance to the next gate (encourages aggressive progress)
- `J_mil`: Penalizes perpendicular distance to the MIL curve when within a gate proximity threshold (ensures proper gate-crossing geometry). This term is only active within a gate entry/exit region.
- `J_col`: Collision avoidance term — maximizes separation between both drones' predicted positions, but **only from t=0 to t=t_c** (the first predicted collision time). After t_c, no constraint is applied, allowing the solver to find solutions where drones diverge post-crossing.

Dynamic weighting between `J_track` and `J_racing` is used: far from gates, tracking dominates (prevents divergence from the nominal path); near gates, racing term dominates (allows aggressive approach).

### 5. Solver and Timing

- Backend: **acados** with hpipm QP solver
- Solve time: ~1 ms on laptop, ~4 ms on embedded Coolpi 4B (ARM)
- Control frequency: **200 Hz**
- Prediction horizon: 20 steps (adaptive dt = 1–40 ms → effective lookahead 0.02–0.8 seconds)

---

## Results

### Simulation Benchmarks (7-gate track)

| Method | Top Speed | Lap Time | Solve Time |
|--------|-----------|----------|------------|
| TOGT (single-drone MINCO) | 17.5 m/s | 12.56 s | 0.018 s (offline) |
| Swarm-CPC (offline multi-drone) | 22 m/s | 10.6 s | 23,250 s (offline) |
| **PMPC (proposed)** | **18.7 m/s** | **12.51 s** | **~4 ms (online)** |

PMPC nearly matches TOGT lap time (within 0.05 s) while running fully online in real time. Swarm-CPC is faster but requires >6 hours of offline computation, which is incompatible with dynamic racing.

Two simulation experiments:
1. **Position swap**: Two drones exchange positions through a single gate, with the gate at varying orientations (0°, 30°, 45°, 60°). PMPC achieved correct perpendicular crossing at all orientations without any failure. Naive waypoint-based planners failed at 45° and 60° gates (clipped a corner).
2. **Racing track**: 7-gate oval with mixed gate orientations. Both drones completed all gates collision-free at 18.7 m/s top speed.

### Real-World Experiments (5m × 4m × 2m arena)

Two 300g quadrotors flew a figure-8 track with 7 waypoints and varying gate orientations:
- Top speed: **6.1 m/s** (arena-limited, not dynamics-limited)
- Collision-free throughout
- Heading correctly oriented toward subsequent gates at all times
- Successful gate traversal at non-axis-aligned gate orientations

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories with a PD/geometric tracker and a kinematic PyBullet simulation. The identified problem is that **the trajectory ends at the last gate (gate-12) center without any post-gate extension**, causing the drone to reach the trajectory endpoint before the gate sequencer confirms passage, then fall into a gate-seeking fallback mode with ~2x worse tracking error.

### Direct relevance to the gate-12 trajectory endpoint problem

The MIL formulation in PMPC is directly relevant to our last-gate issue. PMPC never terminates a trajectory at a gate — it always extends the reference beyond each gate via the MIL field line, which continues outward from the gate plane on the exit side. The key insight:

**Gates are not trajectory endpoints; they are pass-through constraints in the middle of the trajectory.**

Our `TrajectoryOptimizer._generate_trajectory()` uses gate centers as hard waypoint endpoints (line 537–562 in `trajectory_optimizer.py`). The final segment ends with `end_vel = dir_to_gate / dist * min(dist / T, max_velocity * 0.3)` — a slow approach toward gate-12's center. After the trajectory ends at gate-12, the `RaceTrajectory.sample()` method clamps to `self.points[-1]` (the last point), which is at or near gate-12 but has near-zero velocity. The drone stalls there waiting for the sequencer to confirm gate passage.

The PMPC fix: extend the trajectory **through** and **past** each gate by adding a post-gate waypoint along the gate's exit normal. For gate-12 specifically, a waypoint 2–5 meters beyond the gate center along the gate normal would:
1. Give the trajectory a non-zero velocity at the gate (it's still moving toward the post-gate waypoint)
2. Ensure the sequencer's plane-crossing detection fires (the drone physically crosses the gate plane at speed)
3. Eliminate the stall/fallback condition entirely

### MIL perpendicular entry constraint

Our current planner does not enforce perpendicular gate crossing. In `_generate_trajectory()`, the velocity at each waypoint is computed as `next_dir * next_speed` (pointing toward the next gate), not perpendicular to the current gate's plane. For most gates this works since the next-gate direction roughly coincides with the gate normal. But for gates with significant yaw (non-axis-aligned), the trajectory can approach at an oblique angle, leading to the sequencer's `_check_pass_through()` not detecting the crossing (if the drone skims the gate plane without cleanly crossing).

The MIL approach — enforcing that the reference curve is tangent to the gate normal at the crossing point — would fix this. Even a simplified version (just ensuring exit velocity is along the gate normal at each gate) would improve sequencer reliability.

### Adaptive MPC horizon and gate proximity switching

The PMPC's switching between `J_track` (far from gate) and `J_mil + J_racing` (near gate) is analogous to what we should do near gate-12: increase the trajectory's local reference density (shorter dt_sample) in the 2–3 m approach to ensure the tracker has fine-grained reference points near the gate, rather than coarse interpolation of a near-zero-velocity endpoint.

---

## Actionable Takeaways

### 1. Add post-gate extension waypoints (highest priority — directly fixes gate-12 issue)

In `TrajectoryOptimizer.optimize()`, after building `waypoints = [start_position] + [gate.position for gate in gates]`, append a final waypoint that is `N` meters past the last gate along its exit normal:

```python
# After last gate, extend trajectory along gate exit normal
last_gate = gates[-1]
normal = np.array(last_gate.normal)
extension_dist = 3.0  # meters past the gate center
extension_point = np.array(last_gate.position) + normal * extension_dist
waypoints.append(extension_point)
```

This converts gate-12 from a trajectory endpoint into a mid-trajectory pass-through, ensuring non-zero velocity at the gate crossing and reliable plane-crossing detection by the sequencer.

### 2. Enforce gate-normal-aligned exit velocity at each gate

In `_generate_trajectory()`, replace the `next_dir * next_speed` end velocity with a blend of (a) the gate normal direction and (b) the direction to the next gate:

```python
gate_normal = np.array(gates[i].normal)  # perpendicular to gate plane
blend = 0.5  # 50% gate normal, 50% toward next gate
end_vel_dir = normalize(blend * gate_normal + (1 - blend) * next_dir)
end_vel = end_vel_dir * next_speed
```

This improves gate sequencer reliability for non-axis-aligned gates.

### 3. Use adaptive sampling density near gates

The PMPC uses dt = 1 ms near gates and dt = 40 ms between them. Our system uses a fixed `dt_sample = 0.01` s throughout. Near each gate (within 1–2 m), reduce dt_sample to 0.005 s to give the tracker finer reference points and avoid the trajectory "jumping" through the gate in one large step.

### 4. Two-pass velocity sampling for time allocation

Our `_optimize_time_allocation()` uses a single L-BFGS pass with rough distance-based estimates. The PMPC paper's two-stage Dijkstra approach (coarse directional samples → refined directional cone) is more systematic and achieves near time-optimal speeds without a full nonlinear optimization. This could improve our speed profile for gates where the drone approaches at an angle requiring yaw correction.

### 5. Distance-to-next-gate racing term in controller

In `race_pipeline._control_callback()`, add a supplementary feedforward term when within 3 m of a gate that increases pitch (forward acceleration) toward the gate center, analogous to PMPC's `J_racing` term. This would counteract the trajectory's tendency to slow down at segment endpoints.

### 6. Heading lookahead for gate visibility

PMPC's heading optimizer (the `gamma`-weighted future-gate term) ensures the camera stays pointed toward the upcoming gate. Our system computes yaw from instantaneous velocity (`atan2(dy, dx)` in `_generate_trajectory()`). For tight turns, yaw lags the trajectory direction, potentially losing gate visibility. Pre-computing heading as a lookahead toward the next gate's center (interpolated over the preceding segment) would improve PnP detection reliability.

---

## Limitations & Caveats

1. **Small arena / low speed**: Real-world results are in a 5×4×2 m arena at 6.1 m/s. Competition-scale arenas with 15–20 m/s flight are not demonstrated. The MIL computation and MPC horizon may need retuning for larger-scale courses.

2. **Two-drone focus**: The paper is explicitly designed for two drones. The collision avoidance formulation is paired (joint optimization of both agents). Scaling to N>2 drones is listed as future work and non-trivially harder (combinatorial collision pairings).

3. **Gate count**: The simulation track has 7 gates; real hardware also 7 waypoints. Our system has 12 gates. The MIL field computation cost scales linearly with gate count, but the Dijkstra velocity sampling scales as O(G * V^2) where V is the velocity sample count per gate.

4. **MIL requires analytic gate normals**: The magnetic dipole formulation assumes exact knowledge of gate orientation (the dipole moment vector is the gate normal). Our system already has `gate.normal` in `GateWaypoint`, so this is compatible — but noisy or estimated gate orientations from PnP would degrade MIL quality.

5. **Offline point-mass trajectory**: PMPC precomputes a point-mass reference offline, then refines online via MPC. If gate positions change dynamically, the offline reference becomes invalid. For our competition scenario (static gates, known map), this is not a limitation.

6. **No integration with drift-corrected EKF**: The paper does not address state estimation or VIO drift. In practice, at 15+ m/s the gate approach timing is sensitive to position error. The MIL reference is only as good as the state estimate feeding the MPC.

7. **200 Hz MPC requirement**: The system requires 200 Hz online solve. Our current pipeline targets 100 Hz and uses a simpler geometric tracker. The full PMPC formulation may not fit within our 10 ms budget without the acados solver and embedded hardware profile.

---

## Key Parameters / Constants

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Velocity samples per waypoint (Step 1) | 20 | Coarse Dijkstra graph |
| Velocity cone half-angle (Step 2) | Not specified (approx. 15–30°) | Refined Dijkstra graph |
| MPC prediction horizon | 20 steps | Online lookahead |
| Adaptive dt range | 1–40 ms | Fine near gates, coarse between |
| Rotor thrust bounds | 1–40 N per rotor | Dynamics constraint |
| Angular velocity limits | ±5 rad/s (pitch/roll), ±1 rad/s (yaw) | Dynamics constraint |
| Heading discount factor (gamma) | (0, 1), tuned per track | Future gate visibility weighting |
| Collision safety margin | 0.5 m | Minimum separation |
| Collision E-matrix | diag(1, 1, 1/3) | Downwash deweighting in vertical axis |
| Yaw weight vs tilt weight | Q_yaw > Q_tilt | Prioritizes heading accuracy |
| Gate proximity threshold (MIL activation) | Not explicitly stated (approx. 1–2 m from gate plane) | Switch from J_track to J_mil |
| Control frequency | 200 Hz | Online MPC rate |
| Solver | acados + hpipm | Embedded QP solver |
| Solve time (laptop) | ~1 ms | Computational budget |
| Solve time (Coolpi 4B ARM) | ~4 ms | Embedded budget |
| Arena size (real-world) | 5m × 4m × 2m | Experimental constraint |
| Max speed (real-world) | 6.1 m/s | Arena-limited |
| Max speed (simulation) | 18.7 m/s | Dynamics-limited |
| Lap time (simulation, 7 gates) | 12.51 s | Performance benchmark |
| Drone mass | 300 g | Quadrotor platform |
