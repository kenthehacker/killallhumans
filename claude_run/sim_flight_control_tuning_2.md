# Flight Control Tuning — Session 2

> This file is written as an instruction set for an AI coding agent.
> Read `claude_run/optimization_suggestion.md` first for a deep dive on the
> current architecture. This document focuses on decisions, bugs, and concrete tasks.

---

## Decision: RL vs. Deterministic Flight Planning

**Use deterministic trajectory planning for now. RL is a future upgrade.**

### Why deterministic first

The state-of-the-art in autonomous drone racing (AlphaPilot, Swift, TOGT-Planner)
all use trajectory optimization + MPC + a geometric or PID attitude controller —
not end-to-end RL. RL wins on raw aggressiveness at the cost of weeks of training,
careful reward shaping, and sim-to-real transfer risk.

For Phase 1 of the competition:
- The environment is a known, deterministic virtual world.
- Gate positions are available from simulation metadata.
- We can pre-compute a trajectory through all gates before the run begins.
- Deterministic code is debuggable; RL failures are opaque.
- Our deterministic stack currently has multiple concrete bugs that explain the
  bad behavior. Fix those first — they are low-hanging fruit.

### Where RL becomes attractive (future)

- After deterministic works, train an RL speed optimizer in the PyBullet sim
  to learn the optimal velocity profile through each gate (how fast to fly in,
  what banking angle maximizes exit speed).
- RL for the high-level speed/racing-line decision; deterministic for low-level
  attitude control. This hybrid is the best of both worlds.
- The telemetry logging task below (Task 5) is a prerequisite for RL: we need
  standardized frame-by-frame logs of (state, action, reward) before training.

### Target deterministic architecture

```
[Gate positions from sequencer]
          ↓
[Trajectory planner: gate-to-gate Bezier/polynomial]
  - Pre-compute smooth path through all gates
  - Set passage velocity = v_cruise × gate_normal at each gate
          ↓
[Trajectory tracker: sample desired pos+vel at current time]
          ↓
[Existing MPC/PID stack — with bugs fixed below]
          ↓
[TRPY Mixer → QuadrotorDrone]
```

The trajectory planner is the missing layer. Everything below it exists but
is buggy. Fix the bugs first, then add the planner.

---

## Bug Catalog

Fix these in order — earlier bugs cause downstream symptoms that mask later ones.

---

### BUG 1 (Critical): Gate yaw values are wrong in race_01.json

**File:** `sim_pybullet/configs/race_01.json`

**What's wrong:**
The gate `yaw` values define the gate's normal vector used for pass-through detection.
`_gate_normal(pose)` in `sequencer.py` returns `[cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]`.
This is the direction a drone must cross to trigger a pass-through event.

Gate-1 is at `(x=8, y=0)` with `yaw=1.5708` (90°).
Gate normal = `[cos(90°), sin(90°), 0] = [0, 1, 0]` → pointing in +Y.
The gate plane is therefore perpendicular to Y. The drone starts at `(0,0)` and
approaches gate-1 by moving in +X. It never crosses the gate's Y-normal plane
unless it oscillates in Y. Pass-through is detected by luck (y-oscillation), not design.

**For a drone approaching in +X**, the gate normal must be `[1, 0, 0]`, which
requires `yaw = 0.0`. More generally: the gate yaw should be the bearing FROM
the previous gate/start TO the current gate.

**Expected gate yaw values** (computed from approach directions):
```
gate-1 at (8,0):    approach from (0,0)  → atan2(0-0, 8-0)   = 0.0
gate-2 at (18,4):   approach from (8,0)  → atan2(4-0, 18-8)  ≈ 0.38 rad
gate-3 at (28,-2):  approach from (18,4) → atan2(-2-4, 28-18) ≈ -0.54 rad
gate-4 at (38,1):   approach from (28,-2)→ atan2(1-(-2), 38-28) ≈ 0.29 rad
```

**Fix:** Update all gate yaw values in `race_01.json` to the values above.

---

### BUG 2 (Critical): MPC forces zero velocity at every gate

**File:** `sim_pybullet/runner.py`, lines 213–217 and `flight_control/types.py` line 16

**What's wrong:**
`_get_target()` creates:
```python
TargetState(position=target_pos, yaw=target_yaw)
```
`TargetState.velocity` defaults to `(0.0, 0.0, 0.0)`. The MPC cost function
penalizes `||vel - target_vel||²` = `||vel||²` — it's always trying to decelerate
the drone to a full stop at the gate center. This is the worst possible racing behavior.

**Fix:** Compute a non-zero passage velocity in `_get_target`:
```python
gate_normal_x = math.cos(gate.pose.yaw) * math.cos(gate.pose.pitch)
gate_normal_y = math.sin(gate.pose.yaw) * math.cos(gate.pose.pitch)
gate_normal_z = math.sin(gate.pose.pitch)
cruise_speed = 3.5  # m/s — tune this
target_vel = (
    gate_normal_x * cruise_speed,
    gate_normal_y * cruise_speed,
    gate_normal_z * cruise_speed,
)
return TargetState(position=target_pos, velocity=target_vel, yaw=target_yaw)
```
This tells the MPC: arrive at the gate moving through its opening at `cruise_speed`.

---

### BUG 3 (High): MPC returns terminal velocity, not next-step velocity

**File:** `flight_control/mpc.py`, lines 10–27

**What's wrong:**
`plan()` returns `best_velocity` which is the velocity at the END of a 15-step
(0.75 s) constant-acceleration simulation. This gets used by the PID controllers
as the *desired velocity for the very next control tick*.

When the drone is 10m from a gate at 4 m/s, the "desired velocity" returned is
what the velocity would be 0.75s later under the best constant acceleration. That
could be any value, and the PID controllers then try to hit that velocity in one
tick — impossible and destabilizing.

The correct interpretation is: the planner should return the desired velocity for
the *next tick* — i.e., `current_velocity + best_accel * config.dt`.

**Fix:** In `plan()`, after selecting `best_cost` / `best_velocity`, also track
`best_accel`, then return:
```python
next_vel_x = _clamp(state.velocity[0] + best_accel[0] * config.dt,
                    -config.max_velocity[0], config.max_velocity[0])
next_vel_y = _clamp(state.velocity[1] + best_accel[1] * config.dt,
                    -config.max_velocity[1], config.max_velocity[1])
next_vel_z = _clamp(state.velocity[2] + best_accel[2] * config.dt,
                    -config.max_velocity[2], config.max_velocity[2])
return (next_vel_x, next_vel_y, next_vel_z), best_yaw
```
This is the intended velocity for the PID to track over the next `dt` seconds.

---

### BUG 4 (High): Adapter ignores gate direction for yaw in detection mode

**File:** `flight_control/adapter.py`, line 45

**What's wrong:**
```python
return TargetState(position=target_position, yaw=drone_state.yaw)
```
In detection mode, the yaw is set to the drone's *current* yaw. The drone never
gets told to turn toward the gate — it just keeps facing wherever it was already
pointing.

**Fix:** Compute yaw toward the unprojected gate position:
```python
dx = target_position[0] - drone_state.position[0]
dy = target_position[1] - drone_state.position[1]
target_yaw = math.atan2(dy, dx)
return TargetState(position=target_position, yaw=target_yaw)
```

---

### BUG 5 (Medium): Shared velocity PID gains across all three axes

**File:** `flight_control/types.py`, lines 43–45

**What's wrong:**
```python
velocity_pid: PIDConfig = field(
    default_factory=lambda: PIDConfig(1.8, 0.1, 0.25, 2.0, 8.0)
)
```
All three velocity PIDs (vx, vy, vz) share identical gains. The vertical axis
fights gravity — it needs a higher `kp` and possibly `ki` to prevent altitude
sag. Horizontal axes don't fight a constant disturbance.

**Fix:** Split into `horizontal_pid` and `vertical_pid` in `ControllerConfig`,
and wire them separately in `FlightController.__init__`:
```python
@dataclass
class ControllerConfig:
    mpc: MPCConfig = field(default_factory=MPCConfig)
    horizontal_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=1.8, ki=0.1, kd=0.25,
                                          integrator_limit=2.0, output_limit=8.0)
    )
    vertical_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=2.5, ki=0.3, kd=0.2,
                                          integrator_limit=3.0, output_limit=10.0)
    )
    yaw_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=3.0, ki=0.05, kd=0.2,
                                          integrator_limit=1.5, output_limit=4.0)
    )
```
In `FlightController.__init__`, use `horizontal_pid` for `pid_vx` and `pid_vy`,
and `vertical_pid` for `pid_vz`.

---

### BUG 6 (Medium): MPC horizon too short relative to gate spacing

**File:** `flight_control/types.py`, line 22

**What's wrong:**
`horizon_steps=15, dt=0.05` → 0.75 s lookahead.
At `max_velocity=6 m/s`, the planner sees 4.5 m ahead.
Gates in `race_01.json` are ~10 m apart. The drone cannot "see" the next gate
in its planning window when it's far away — the planner is essentially flying
blind beyond 4.5 m.

**Fix:** Increase `horizon_steps` to 25–30, or increase `dt` to 0.1 (1.5–3 s
lookahead). Watch the computational cost: 30 steps × 125 candidates = 3750
simulations per control tick. Profile and tune accordingly.

```python
@dataclass
class MPCConfig:
    dt: float = 0.05
    horizon_steps: int = 25       # was 15 → now ~1.25 s lookahead at dt=0.05
    ...
```

---

### BUG 7 (Low): Camera pitch offset uses already-modified forward vector

**File:** `sim_pybullet/drone.py`, lines 357–362

**What's wrong:**
```python
forward = forward * ca + up * sa        # forward is now modified
up = -forward * sa + up * ca            # BUG: uses the NEW forward, not the original
```
The rotation matrix formula requires both `forward` and `up` to be computed from
the *original* forward. The second line uses the already-overwritten `forward`.

This bug is dormant when `camera_pitch_offset = 0.0` (default).

**Fix:**
```python
new_forward = forward * ca + up * sa
new_up = -forward * sa + up * ca        # original forward used here
forward = new_forward
up = new_up
```

---

### BUG 8 (Low): WASD orbit is intercepted by PyBullet GUI on macOS

**File:** `sim_pybullet/runner.py`, lines 411–424

**What's wrong:**
Two issues:
1. PyBullet's GUI window steals keyboard focus. When the PyBullet window is in
   focus, OpenCV's `waitKey(1)` receives nothing.
2. Arrow key codes 81–84 are Linux-specific. On macOS (Darwin), OpenCV returns
   different codes for arrow keys.

**Fix (arrow keys):** Remove the Linux arrow-key codes and rely only on WASD.
Add a print statement or HUD indicator confirming orbit angle so the user knows
it's working. Ensure the OpenCV window has focus (click on it first).

**Fix (PyBullet stealing focus):** The call to
`p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, ...)` should
prevent PyBullet from consuming WASD, but the window focus is outside our control.
No code change needed — document this as a usage note for macOS.

---

## Task List for the AI Agent

Implement the fixes above in this order. After each task, verify the simulation
runs without crashing before moving to the next.

### Task 1 — Fix gate yaw values (BUG 1)
- Edit `sim_pybullet/configs/race_01.json`.
- Recompute each gate's `yaw` as `atan2(gate.y - prev.y, gate.x - prev.x)`.
- Verify by adding a debug print in `GateSequencer._check_pass_through` that logs
  `d_prev` and `d_curr` each time a gate is checked — confirm they flip sign when
  the drone passes through.

### Task 2 — Fix pass-through velocity target (BUG 2)
- In `sim_pybullet/runner.py: _get_target()`, compute `target_vel` from the gate
  normal and a configurable `cruise_speed` (start at 3.0 m/s).
- The gate normal is `(cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch))`.
- Pass `velocity=target_vel` into the returned `TargetState`.

### Task 3 — Fix MPC output interpretation (BUG 3)
- In `flight_control/mpc.py: MPCPlanner.plan()`, track `best_accel` alongside `best_cost`.
- Return `(next_tick_velocity, best_yaw)` where `next_tick_velocity` is
  `current_vel + best_accel * dt` (clamped to `max_velocity`).
- Update the docstring to reflect the corrected output semantics.

### Task 4 — Fix adapter yaw (BUG 4)
- In `flight_control/adapter.py: gate_detection_to_target()`, compute
  `atan2(target_position[1] - drone_state.position[1], target_position[0] - drone_state.position[0])`
  and use it as `TargetState.yaw`.

### Task 5 — Split velocity PID gains (BUG 5)
- Add `horizontal_pid` and `vertical_pid` to `ControllerConfig` in `flight_control/types.py`.
- Wire them in `FlightController.__init__` in `flight_control/controller.py`.
- Keep the old `velocity_pid` field as a deprecated alias that maps to `horizontal_pid`
  so existing tests don't break.

### Task 6 — Extend MPC horizon (BUG 6)
- Set `horizon_steps = 25` in `MPCConfig`.
- Benchmark: time `FlightController.step_trpy()` once and confirm it runs in < 5 ms
  (add a `time.perf_counter()` check, print a warning if exceeded).

### Task 7 — Add telemetry logger
- Create `sim_pybullet/telemetry.py` with a `TelemetryLogger` class.
- Each frame, log: `sim_time`, drone `position`, `velocity`, `yaw`, `roll`, `pitch`,
  current gate ID, gate position, TRPY command, and distance to current gate.
- Write to a `.jsonl` file (one JSON object per line) in a `runs/` directory.
- Wire the logger into `RaceRunner.run()` after step 5 (compute control command).
- This produces the data format needed for RL training later.

### Task 8 — Visualize planned path in the simulation
- In `RaceRunner`, after computing `target` in `_get_target()`, draw a debug line
  in PyBullet from the drone's current position to the target gate center using
  `pybullet.addUserDebugLine()`.
- Additionally, draw lines connecting all future gate centers in sequence (a
  "racing line preview") at the start of each lap using a different color.
- Clear and redraw the drone→target line each frame using the `replaceItemUniqueId`
  parameter of `addUserDebugLine`.
- This satisfies the "display the planned path like a racing game racing line"
  requirement from `sim_flight_control_tuning_2.md`.

### Task 9 — Fix camera pitch offset bug (BUG 7)
- In `sim_pybullet/drone.py: _compute_fpv_matrices()`, fix the two-line rotation
  to use a temp variable for the original `forward` before computing the new `up`.

### Task 10 — Add gate height variation to race_01.json
- Currently: z values are 1.5, 2.2, 1.3, 2.8 — already varied but small range.
- Add a 5th gate and ensure heights span 1.0 m to 3.5 m with no single step > 1.5 m
  (so the next gate stays in the FPV frame).
- After fixing BUG 1 (gate yaw), recompute all gate yaws for the updated course.

---

## Integration Notes for the AI Agent

- Do not change the competition-facing `TRPYCommand` output format — that is the
  interface to the DCL platform.
- Do not remove `FlightController.step()` (non-TRPY version) — it is used by unit
  tests in `flight_control/tests/`.
- All tunable constants (`cruise_speed`, updated PID gains, `horizon_steps`) should
  be exposed as constructor parameters or dataclass fields, not hardcoded, so they
  can be tuned without touching logic.
- After all fixes, run `python3 -m sim_pybullet.runner --no-gui --max-time 60` and
  confirm all 4 gates are passed.
