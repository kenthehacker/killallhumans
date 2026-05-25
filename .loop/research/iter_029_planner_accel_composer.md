# iter-029 → iter-030: planner acceleration vs PyBullet maneuver budget (Composer)

## Diagnosis (code-grounded)

Two independent gaps explain “reference / tracker asks for ~50–80 m/s² peaks while `QuadrotorDrone` only delivers ~10 m/s² maneuvering” on `scripts/smoke_quadrotor_drone_race.py` + `race_01.json`.

### A) Smoke harness plans much faster than the PyBullet matrix path

`scripts/benchmark.py` (`_run_pybullet_bench`, ~914–920) resolves trajectory speed as:

1. explicit `max_velocity_mps` from track JSON, else  
2. `planner_overrides.plan_max_speed_mps`, else  
3. `derive_safe_max_velocity(gates)`.

`race_01.json` sets `"plan_max_speed_mps": 4.0`, so the **matrix PyBullet** run plans at **4 m/s**.

`scripts/smoke_quadrotor_drone_race.py` (~76) calls **only** `derive_safe_max_velocity(specs)` and ignores the `planner` block entirely. On a 12-gate helix with many shallow triplets, the centripetal walk often **never binds**, so the function returns the **absolute cap** `DEFAULT_DRONE_MAX_SPEED_MPS` = **15 m/s** (`planning/auto_velocity.py`).

Same gates, **3.75× higher** execution speed → proportionally harsher lateral acceleration demand from the polynomial and from the tracker’s PD path.

### B) Tracker tilt envelope ≠ PyBullet plant tilt envelope

`GeometricTracker` clamps attitude to `TrackerConfig.max_tilt_rad`, defaulting from `competition.drone_spec.DEFAULT_MAX_TILT_RAD` ≈ **0.85 rad** (`control/mpc_tracker.py`).

`QuadrotorDrone.step_reference` converts `cmd.roll_rad` / `pitch_rad` to normalized stick deflections by dividing by `DroneConfig.max_roll_angle` / `max_pitch_angle` = **0.35 rad** (`sim_pybullet/drone.py`, lines 301–304, 347–348). Any attitude beyond ±0.35 rad is **saturated away** before `apply_command`.

Rough horizontal authority from gravity tilt scales like **g·tan(θ)**. At θ = 0.35 rad, tan θ ≈ 0.365 → **~3.6 m/s² per horizontal axis** from tilt geometry alone; combined with differential thrust and vertical thrust modulation, **~8–12 m/s²** total maneuver magnitude is plausible — matching the “~10 m/s² delivered” observation.

Meanwhile the planner + tracker still *assume* up to **0.85 rad** bank and **15 m/s²** class accelerations (`DEFAULT_MAX_ACCEL_MPS2` in `competition/drone_spec.py`). The plant cannot reproduce that command set.

### C) “50–80 m/s²” is not necessarily the polynomial alone

`TrajectoryOptimizer._generate_trajectory` post-clamps **velocity** overshoots by scaling velocity/accel/jerk together, but there is **no hard per-sample ‖a‖₂ ≤ max_acceleration** projection on the stored `TrajectoryPoint.acceleration`.

More importantly, `GeometricTracker.track` forms  
`accel_des = kp·e_p + kd·e_v + feedforward_accel·a_ref` with **no acceleration vector saturation** before thrust/tilt extraction (`control/mpc_tracker.py`, ~195–199). When the drone lags an aggressive reference, **PD alone** can demand tens of m/s² even if `a_ref` is modest.

So the logged “peaks” are often **desired** acceleration from the controller, not what the rigid body integrates.

---

## iter-030 recommendations (ordered by ROI / risk)

### 1) Fix smoke trajectory speed selection (low risk, high leverage)

Factor the benchmark’s three-step `max_velocity` resolution into a tiny helper (e.g. `planning/race_config.py` or `scripts/_race_speed.py`) and call it from `smoke_quadrotor_drone_race.py` after parsing `race_01.json` the same way `RaceConfig` does.

**Acceptance:** smoke uses **4 m/s** on stock `race_01.json`, matching `_run_pybullet_bench`. This should collapse early PD blow-ups and ref-curvature stress without touching the kinematic bench.

### 2) Unify tilt limits across tracker output and PyBullet saturation (medium risk)

Pick one policy and implement it cleanly:

- **Preferred:** set `DroneConfig.max_roll_angle` / `max_pitch_angle` to **match** `DEFAULT_MAX_TILT_RAD` (or `min(tracker_max, stability_derived_cap)`), then re-tune `attitude_kp/kd` and `max_differential` in `apply_command` if oscillation appears. This removes the deliberate 0.35 vs 0.85 mismatch called out in `step_reference`’s own docstring.

- **Alternative (safer but duplicative):** keep conservative PyBullet tilts, but instantiate `GeometricTracker(TrackerConfig(max_tilt_rad=0.35))` inside `step_reference` so commanded bank never exceeds what the plant maps 1:1.

Avoid silently dividing by 0.35 while the tracker thinks 0.85 is legal.

### 3) Optional: mirror kinematic bench saturation in the tracker (low–medium risk)

After computing `accel_des`, apply the same vector clamp the kinematic loop uses (`scripts/benchmark.py`, `DEFAULT_MAX_ACCEL_MPS2`). That caps PD wind-up spikes and keeps PyBullet and matrix controllers comparable.

### 4) Optional (later): hard polynomial peak check

If traces still show `‖a_ref‖` ≫ `constraints.max_acceleration` at interior samples, add a post-pass on segment times (sqrt scaling in time for excess peak accel) between time allocation and `_generate_trajectory` export — the iter-016 class gap — **after** 1–3 are proven in smoke.

---

## What not to do first

Lowering `DEFAULT_MAX_ACCEL_MPS2` / `DEFAULT_MAX_VELOCITY_MPS` globally in `competition/drone_spec.py` “to match PyBullet” fixes the symptom by **regressing the entire kinematic matrix** and invalidates iter-009/iter-010 convergence work, without proving the PyBullet attitude stack is the long-term binding constraint. Try harness parity + tilt honesty first; rebaseline drone_spec only if telemetry says the real vehicle matches the stricter envelope.

---

## Verification checklist for iter-030

1. `python3 scripts/smoke_quadrotor_drone_race.py` → non-zero gates passed, no ground strike at ~1.2 s.  
2. `python3 scripts/benchmark.py --mode sim` (or matrix PyBullet slice) unchanged or improved — no accidental velocity regression on other tracks.  
3. If tilt caps widen: add a short headless stability smoke (max roll/pitch rate, no tumble) before widening defaults on `main`.
