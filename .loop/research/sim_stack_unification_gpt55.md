# Recommendation: unify on the AIGP-sized PyBullet quad, keep synthetic as smoke

The cleanest path is **not** to chase the 27 g Crazyflie demo. It is also not to keep treating the synthetic matrix as the race truth. The repository currently has three different "drones": the synthetic 1 kg / 20 N point-mass envelope in `competition/drone_spec.py`, the `QuadrotorDrone` PyBullet rigid body in `sim_pybullet/drone.py`, and the Crazyflie CF2X `GPDDrone`/`DSLPIDControl` stack in `sim_pybullet/gpd_drone.py`. The observed split is exactly what that architecture predicts: the matrix passes in a permissive kinematic model, while the visual demo flies a much smaller airframe with a different controller and crashes.

My recommendation: **promote an AIGP-sized `QuadrotorDrone` PyBullet backend to the canonical local simulator, freeze its physical values in `competition/drone_spec.py`, route benchmark and visual demo through the same `DroneRaceEnv`, and demote the synthetic benchmark to a fast consistency/regression tier.**

## 1. Swap `visual_demo.py` to `QuadrotorDrone`

Yes. This should be the first change because it removes the most misleading discrepancy with the least conceptual risk.

`visual_demo.py` currently assumes `DroneRaceEnv` creates a CF2X/GPD backend, then hardcodes tracker display constants for that platform: `mass=0.027`, `max_thrust_n=0.6`, 4 m/s plan/cmd speeds, and GPD-style `target_pos/target_vel/target_acc` stepping. That makes the demo a Crazyflie demo, not an AIGP racing-class demo.

Implement this by making `DroneRaceEnv` accept a backend selector, defaulting to the AIGP quad once migrated:

- `backend="aigp_quad"`: use `sim_pybullet.drone.QuadrotorDrone`.
- `backend="cf2x"`: keep `GPDDrone` available only for legacy comparison.

Then update `visual_demo.py` to request the default AIGP backend and use `GeometricTracker` commands as the authoritative control path: sample trajectory, call `tracker.track(...)`, then call `env.drone.apply_command(cmd.thrust, normalized_roll, normalized_pitch, normalized_yaw_rate_or_yaw_error)` or add a thin `step_reference(...)` helper on `QuadrotorDrone` that converts target reference to tracker command internally. Avoid carrying over the CF2X-only 4 m/s assumptions unless they remain explicitly tied to `backend="cf2x"`.

## 2. Swap matrix to PyBullet

Partially. The matrix should become **PyBullet-primary**, but the synthetic benchmark should not be deleted.

Current `scripts/benchmark.py --mode full` already runs both synthetic and PyBullet when available, but the synthetic path is far richer: ILC offsets, controller trace, plan validation, and the 0.089 m headline metric are all coming from the kinematic model. That makes the matrix too easy to overfit.

Change the interpretation:

- `--mode sim` / PyBullet is the release gate for race performance.
- `--mode synthetic` is a fast smoke test for planner/sequencer/controller math and should fail only on obvious regressions.
- `overall_passed` in `--mode full --strict` should require PyBullet success when PyBullet is installed.
- Matrix reports should label synthetic results as "kinematic proxy", not "simulation" or final race evidence.

Do not port every synthetic-specific trick into PyBullet immediately. First make PyBullet and visual demo use the same drone, spec, sequencer config, planner config, and controller path. Then re-introduce ILC/feedforward only if it improves PyBullet metrics.

## 3. Freeze AIGP drone-spec values

Yes, but freeze them as **declared local evaluation assumptions**, not as claimed competition facts.

The AIGP spec gives 280 x 280 x 160 mm but not mass or thrust. A sane local proxy should match the chassis dimensions and use conservative racing-class values:

- `mass_kg = 0.8`
- `max_thrust_n = 24.0` total
- `arm_length_m = 0.14`
- `body_size_m = (0.28, 0.28, 0.16)` for collision/visual envelope, with a smaller central body if arms/motors are modeled separately
- `max_velocity_mps = 15.0`
- `max_acceleration_mps2 = 15.0`
- `max_tilt_rad = 0.85`
- `max_body_rate_rad_s = 6.0`
- `max_yaw_rate_rad_s = 4.0`
- `linear_drag_per_mass = 0.5` initially

This yields a thrust-to-weight ratio of about 3.1:1, plausible for a 280 mm racing quad without being as forgiving as 1 kg / 20 N with kinematic acceleration injection or as irrelevant as a 27 g microdrone. The exact numbers can be calibrated later, but they must live in one place and all local demos must identify which spec they are using.

Rename/document `competition/drone_spec.py` away from "synthetic-bench drone" toward "local AIGP proxy spec". If the CF2X backend remains, give it its own explicit spec module or config class so it cannot silently impersonate the AIGP drone.

## 4. Regression test story

Use a two-tier regression gate:

1. **Fast deterministic tier**: unit tests plus synthetic smoke. This catches planner math, sequencer legality, EKF stability, trajectory generation, and obvious controller output regressions. It should stay cheap enough for every iteration.
2. **Truth tier**: headless PyBullet with `aigp_quad` on `race_01`. This is the pass/fail race gate. Track gates passed, crash/DQ, race time, avg/p95/max tracking error, loop Hz, and first failure reason.

Add a parity invariant: `scripts/benchmark.py --mode sim` and `scripts/visual_demo.py --no-render` must instantiate the same backend and produce compatible gate count / termination results. The visual demo can have rendering and CSV logging, but its flight stack should not diverge from the benchmark stack.

Add one regression test that constructs `DroneRaceEnv(..., backend="aigp_quad")`, asserts `env.drone.config.mass_kg == DroneSpec().mass_kg`, `max_thrust_n` matches, and `body_id` participates in `gate_contact()`. Add another smoke test that runs a short no-render PyBullet race for a few seconds and asserts no import/backend mismatch.

## 5. Concrete code-change list

1. Update `competition/drone_spec.py` to define the frozen AIGP proxy values, including arm length/body dimensions, and update comments to stop calling it synthetic-only.
2. Update `sim_pybullet/drone.py` `DroneConfig` defaults to source from `competition.drone_spec`; add a `step_reference(...)` method or keep `apply_command(...)` and expose a small command adapter.
3. Update `sim_pybullet/env.py` to support `backend="aigp_quad" | "cf2x"` and default to `"aigp_quad"`; create gates in the selected backend's PyBullet client; make `get_sim_time()` work for both.
4. Update `scripts/benchmark.py` `run_sim_benchmark()` to instantiate `DroneRaceEnv(..., backend="aigp_quad")`, run the same `GeometricTracker` path used by the real pipeline, and report `drone_backend` plus `drone_spec`.
5. Update `scripts/visual_demo.py` to use the same backend and command path as `run_sim_benchmark()`. Keep `--backend cf2x` only as an explicit legacy flag.
6. Update `race_pipeline.py` defaults only where they conflict with the frozen spec; avoid baking race_01-specific 4 m/s CF2X knobs into the competition path.
7. Add tests for spec propagation, backend construction, and benchmark/visual no-render parity.
8. Update README/CLAUDE guidance so future agents optimize PyBullet AIGP metrics first and treat synthetic numbers as secondary.

## 6. Risks

The biggest risk is that switching to PyBullet truth will temporarily make metrics worse. That is good signal, not regression. The current 0.089 m synthetic result is not evidence that the physical stack is race-ready.

The second risk is controller impedance mismatch. `QuadrotorDrone.apply_command()` accepts normalized attitude/throttle commands, while the GPD path accepts position references. The unification must choose one control boundary. I recommend using `GeometricTracker -> AttitudeCommand -> QuadrotorDrone.apply_command()` because it mirrors the competition-facing abstraction better than a position-PID wrapper.

The third risk is false confidence from guessed mass/thrust. Mitigate by naming the values "AIGP proxy v1", recording them in every benchmark artifact, and making future calibration a spec change with regression diffs.

The fourth risk is losing CF2X learnings. Keep the CF2X backend behind an explicit flag for comparison, but do not let it drive default demos, thresholds, or planner tuning.

Bottom line: unify all default demos and release gates on a single AIGP-sized PyBullet quad now. Keep synthetic for speed, keep CF2X for legacy comparison, but stop optimizing against either as the primary race truth.
