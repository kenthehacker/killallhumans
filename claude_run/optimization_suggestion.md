# Flight Path Planning — Current Logic & Optimization Suggestions

## 1. The Full Control Pipeline (Top to Bottom)

```
GateSequencer          → which gate to target next
        ↓
_get_target()          → converts gate position to TargetState
        ↓
MPCPlanner.plan()      → converts TargetState into a desired velocity vector
        ↓
PID (vx, vy, vz, yaw) → converts velocity error into acceleration command
        ↓
TRPYMixer.mix()        → converts acceleration into (throttle, roll, pitch, yaw)
        ↓
QuadrotorDrone         → applies forces/torques in PyBullet
```

---

## 2. Gate Selection: GateSequencer (`sim_pybullet/sequencer.py`)

Gates are pre-sorted at startup by `sequence_index`. The drone always targets the
**next gate in the fixed list** — there is no dynamic nearest-gate lookup during the race.

**Pass-through detection** uses a *signed-distance plane-crossing* test:
1. The gate's forward axis (local +X after rotation) is treated as a plane normal.
2. Each physics tick, the signed distance from the gate plane is computed for the
   previous and current drone positions.
3. If those two signed distances have opposite signs, the drone crossed the plane.
4. The crossing point is found by linear interpolation between the two positions.
5. If that crossing point falls within 1.5× the gate opening dimensions, the gate
   is counted as passed and the sequencer advances to the next gate.

The 1.5× tolerance margin is deliberately generous — it was added to forgive
imprecise flight through the center.

---

## 3. Target Computation (`sim_pybullet/runner.py: _get_target`)

In **sim-metadata mode** (the default, no real detector):

```python
target_pos = (gate.pose.x, gate.pose.y, gate.pose.z)   # gate center
target_yaw = atan2(dy, dx)                              # 2D angle to gate
TargetState(position=target_pos, velocity=(0, 0, 0), yaw=target_yaw)
```

Key observations:
- The target is literally the **gate center** with **zero target velocity**.
  This tells the rest of the stack: "arrive at the gate and stop."
  In a racing context this is the wrong objective — you want to *pass through*
  at speed with the velocity vector aligned with the gate's normal.
- The yaw is computed from the 2D (x, y) projection only. If the gate is at a
  very different altitude, the drone's heading is correct horizontally but the
  vertical approach angle is left entirely to the MPC/PID.
- In **detection mode**, `gate_detection_to_target()` (`flight_control/adapter.py`)
  unprojects the pixel-space bounding box center into a 3D world position using the
  estimated distance from the detector and the camera FOV. The resulting
  `TargetState` also has `velocity=(0,0,0)` and `yaw=drone_state.yaw`
  (it inherits current yaw, not gate-aligned yaw).

---

## 4. The "MPC" Planner (`flight_control/mpc.py`)

### What it is labelled vs. what it actually does

It is called `MPCPlanner` but it is **not a true Model Predictive Controller**.
A real MPC formulates a constrained optimization problem (usually a quadratic
program or nonlinear program) and solves it with a dedicated solver. This code
instead does a **brute-force grid search** over a fixed set of candidate
accelerations.

### How it works step by step

1. **Candidate generation** — builds a 5×5×5 grid of candidate constant
   accelerations from `(-max, -0.5×max, 0, +0.5×max, +max)` on each axis:
   125 candidates total.

2. **Forward simulation** — for each candidate `(ax, ay, az)`, simulates the
   drone forward for `horizon_steps=15` ticks at `dt=0.05s` (0.75 s lookahead)
   using simple Euler integration with velocity clamping.

3. **Cost function** — accumulates over all horizon steps:
   ```
   cost += position_weight    * ||pos - target_pos||²
         + velocity_weight    * ||vel - target_vel||²
         + acceleration_weight * ||accel||²
   ```
   Then adds a heavier **terminal cost** at the final step for position and
   velocity error.

4. **Selection** — the candidate with the lowest total cost wins. Its resulting
   velocity vector (after 15 steps of simulation) is returned as `best_velocity`.

5. **Output** — `plan()` returns `(best_velocity, target.yaw)`. The `best_velocity`
   is fed directly into the PID controllers as the **desired velocity**.

### Problems with this approach

| Problem | Explanation |
|---------|-------------|
| Coarse grid | 5 levels per axis = 125 candidates. Large portions of the feasible acceleration space are never evaluated. A diagonal maneuver between grid points is never considered. |
| Constant acceleration assumption | A real MPC applies a *time-varying* sequence of inputs. This planner commits to the same `(ax, ay, az)` for all 15 steps and returns only the resulting velocity — much less expressive. |
| Wrong output interpretation | The returned `best_velocity` is the velocity *after 15 steps of simulation*, not a "desired velocity for the next tick." These are very different quantities, especially when far from the target. |
| Zero target velocity | `TargetState.velocity` defaults to `(0,0,0)`. The planner is therefore always trying to decelerate the drone to a stop at the gate — wrong for racing. |
| No feedforward | There is no notion of "fly through the gate at speed in a particular direction." The entire pipeline is reactive/feedback only. |
| No heading alignment with gate normal | The planner minimizes Euclidean distance to the gate center. It does not penalize arriving at the gate from the wrong approach angle. |

---

## 5. PID Layer (`flight_control/pid.py`, `flight_control/controller.py`)

Four independent PID controllers run after the planner:

```
pid_vx.update(desired_vx, actual_vx, dt)  → ax (world-frame)
pid_vy.update(desired_vy, actual_vy, dt)  → ay (world-frame)
pid_vz.update(desired_vz, actual_vz, dt)  → az (world-frame)
pid_yaw.update(wrap(desired_yaw - actual_yaw), 0.0, dt)  → yaw_rate
```

All three velocity PIDs share **identical gains**: `kp=1.8, ki=0.1, kd=0.25`.
In reality, a drone's vertical axis (z) has very different dynamics from the
horizontal axes because gravity must be actively overcome, so you'd typically
want different gains there.

The yaw PID tracks the error between `desired_yaw` (just `target.yaw` passed
through unchanged from the planner) and the current drone yaw. This is fine
in principle, but the desired yaw is only a 2D bearing to the gate — it doesn't
account for roll that would be needed to execute a banking turn at speed.

---

## 6. TRPY Mixer (`flight_control/mixer.py`)

Converts world-frame accelerations `(ax, ay, az)` to competition-format:

```
1. Rotate (ax_world, ay_world) into body frame using current yaw
2. pitch = atan2(ax_body, g)        — forward accel → nose-down angle
3. roll  = atan2(ay_body, g)        — lateral accel → bank angle
4. throttle = mass * (g + az_world) / (cos(roll)*cos(pitch)) / max_thrust
5. Normalize all to [-1, 1]
```

This is a standard quadrotor decomposition and is mathematically correct for
small-to-moderate angles. The approximation degrades at extreme attitudes
(past ~30°), which is common in aggressive racing maneuvers.

Current limits: `max_roll = max_pitch = 0.6 rad (~34°)`, `max_yaw_rate = 3.0 rad/s`.
For aggressive racing these are conservative — competition drones often fly at
60°+ bank angles.

---

## 7. Why the Drone Veers in the Wrong Direction

Based on the screenshot issue noted in `sim_flight_control_tuning.md`, here is
the most likely root cause chain:

1. **After passing gate 1**, the drone still carries significant velocity in the
   direction it was flying (toward gate 1).

2. **Gate 2 is to the left** (from the FPV). The target suddenly jumps to gate 2's
   world position.

3. The MPC grid-searches 125 constant-acceleration candidates. The drone's
   existing momentum (rightward/forward from gate 1 approach) means many
   candidates that *point toward gate 2* still produce a forward-simulated
   position that overshoots or arcs away. The cost function sums error over
   all 15 steps — so a candidate that decelerates the existing momentum first
   and then pushes toward gate 2 may actually beat a candidate that immediately
   hard-turns.

4. The PID integrators also hold state from the gate 1 approach and add a
   lag in building up the correct lateral velocity.

5. Combined effect: the drone commits to a deceleration+arc trajectory that
   *looks like veering right* even though gate 2 is on the left.

The deeper structural issue is that there are **no intermediate waypoints** and
**no velocity direction planning**. The controller is purely reactive — it just
minimizes distance to the gate center at each tick without any awareness of
where it came from or what approach angle it needs.

---

## 8. Literature Background — What Real Drone Racing Stacks Do

### 8.1 Differential Flatness & Minimum-Snap Trajectories

Quadrotors are **differentially flat** with flat outputs `(x, y, z, yaw)`.
This means: if you define a smooth trajectory through those four variables
(including their derivatives up to 4th order), you can compute exactly what
thrust and body rates are needed — no iterative solving required.

The standard approach (Mellinger & Kumar 2011, Richter et al. 2016) is:
- Represent the trajectory as a **piecewise polynomial** (typically 7th-order
  per segment, one segment per gate-to-gate leg).
- Minimize **snap** (the 4th derivative of position) along the entire trajectory.
  This keeps actuator demands smooth and small.
- Enforce **gate-passage constraints**: at each gate waypoint, constrain position,
  velocity direction, and sometimes higher derivatives.
- Solve the resulting QP (quadratic program) offline before the race, then
  track the pre-planned trajectory with a tracking controller.

Why minimize snap specifically? Because:
```
position  →  velocity  →  acceleration  →  jerk  →  snap
                                             ↑
                                  Quadrotor thrust/torque
                                  maps to acceleration.
                                  Minimizing snap ≈ minimizing
                                  change in thrust commands.
```

### 8.2 Gate Passage Velocity

In a real racing stack, the **TargetState at a gate is not "position=center, velocity=0"**.
It is:
```
position = gate_center
velocity = v_cruise * gate_normal_vector
```
Where `gate_normal_vector` is the unit vector pointing through the gate's opening
(the gate's local +X axis after rotation), and `v_cruise` is a desired pass-through
speed. This forces the trajectory planner to arrive at the gate moving in the
right direction, not just arriving at the right point from any direction.

### 8.3 True MPC (Model Predictive Control)

A proper MPC loop for drone racing:
- Formulates a **nonlinear or linear-time-varying QP** at each control tick.
- Uses a **linearized model** of the drone dynamics around the current operating point.
- Solves the QP with a fast solver (ACADOS, OSQP) in < 5 ms.
- Produces a **time-varying control sequence** (not a single constant acceleration).
- Applies only the first control input, then re-solves next tick (receding horizon).
- Can enforce hard constraints: `v_max`, `a_max`, attitude limits, obstacle avoidance.

For our current scale this would be a significant engineering step, but the key
conceptual upgrade is: **solve for a sequence, not a single step.**

### 8.4 Geometric Control on SE(3)

Rather than the PID-over-velocity approach, geometric controllers (Lee et al. 2010)
operate directly on the rotation group. They compute thrust and angular velocity
commands from position and attitude errors without linearizing the dynamics.
This is more accurate at large attitude angles (steep banks, aggressive pitch).

For our mixer, the approximation `pitch ≈ atan2(ax, g)` is already a small-angle
geometric relationship — it's valid but degrades past ~30° pitch.

---

## 9. Summary of Key Gaps vs. What's Needed for Racing

| Current | What Racing Needs |
|---------|-------------------|
| Target = gate center, velocity = 0 | Target = gate center, velocity = v × gate_normal |
| Grid-search over 125 constant accels | Minimum-snap polynomial trajectory planned gate-to-gate |
| No intermediate waypoints | Approach waypoints set up correct entry angle per gate |
| Single reactive tick, no lookahead beyond 0.75s | Full race trajectory planned ahead offline or with rolling horizon |
| Same PID gains for all axes | Axis-specific and speed-adaptive gains |
| Yaw computed from 2D bearing only | Yaw aligned to gate normal at each gate, interpolated between |
| No velocity feedforward | Flat-output feedforward from trajectory derivatives |
| 34° max attitude | Competition drones fly 60°+ at speed |

---

## 10. Suggested Incremental Improvements (No Full Rewrite Required)

These are rough directions — not final decisions — for discussion:

1. **Fix gate passage velocity**: In `_get_target`, set `TargetState.velocity` to
   a non-zero vector aligned with the gate's normal direction at some reasonable
   cruise speed (e.g., 3–5 m/s). This alone is the highest-leverage single change.

2. **Add an approach waypoint**: Instead of targeting the gate center directly,
   target a point slightly before the gate (e.g., 1–2 m along the negative gate
   normal). This forces the drone to line up on approach before the gate plane.

3. **Fix MPC output interpretation**: The planner should return a *desired velocity
   for the next tick* (i.e., `current_velocity + best_accel * dt`), not the
   simulated terminal velocity after 15 steps.

4. **Per-axis PID gains**: vz should have higher kp than vx/vy since vertical
   dynamics fight gravity. Tune separately.

5. **Add velocity feedforward**: After planning, pass `desired_velocity` through
   to the mixer as a feedforward term rather than relying entirely on PID feedback.

6. **Log drone + gate positions per frame**: Already flagged in tuning notes.
   Crucial for diagnosing the veering issue quantitatively.
