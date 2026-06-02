# Adversarial review: iter-010 `f1505ee`

## Verdict

No BLOCKER found. I do not see an active callsite still overriding `DroneConstraints(max_acceleration=20.0)`, and the default `DroneConstraints().max_acceleration` now resolves to `competition.drone_spec.DEFAULT_MAX_ACCEL_MPS2 == 15.0`.

But the "single source of truth" claim is only partially true. The exact regression class is fixed for `DroneConstraints`, while several other modules still define the same synthetic-drone limits independently and one doc comment still says the optimizer assumes `20.0`.

## BLOCKER

None.

## MAJOR

1. `planning/auto_velocity.py` is now stale and still owns duplicate drone constants.

   It defines `DEFAULT_DRONE_MAX_ACCEL = 15.0` and `DEFAULT_DRONE_MAX_SPEED_MPS = 15.0` independently instead of importing `DEFAULT_MAX_ACCEL_MPS2` / `DEFAULT_MAX_VELOCITY_MPS`. Worse, its header comment still says:

   > `DroneConstraints.max_acceleration` in trajectory_optimizer.py is *20.0*

   That is false after iter-010. The behavior is currently numerically aligned because both are `15.0`, but future drift is still possible and the doc now points maintainers toward the pre-fix mental model.

2. Several synthetic-bench physics paths bypass `drone_spec`, so `drone_spec` is not actually the stack-wide source of truth.

   Examples:

   - `planning/trajectory_optimizer.py::compute_ilc_offset_table`: local `max_accel = 15.0`, `max_speed = 15.0`, `drag = 0.5`.
   - `scripts/benchmark.py::run_synthetic_benchmark`: local `max_accel = 15.0`, `max_speed = 15.0`, `drag = 0.5`, `yaw_rate_max = 4.0`.
   - `planning/racing_line.py::_kinematic_eval`: default `max_speed_mps=15.0`, `max_accel_mps2=15.0`, local `drag = 0.5`.
   - `scripts/helix_offset_search.py`: `DroneConstraints(max_velocity=15.0)`.
   - `control/mpc_tracker.py::TrackerConfig`: `mass=1.0`, `gravity=9.81`, `max_thrust_n=20.0`, `max_tilt_rad=0.85`, `max_body_rate=6.0` remain inline.

   Some of these may be intentionally deferred, but `competition/drone_spec.py` currently says "bench / planner / tracker all see the same numbers." The tracker does not read the module, and neither do the benchmark clamps that are cited as provenance.

3. Provenance comments are over-specific and partially inaccurate.

   `competition/drone_spec.py` cites current line numbers such as `scripts/benchmark.py:286 max_speed = 15.0`; that line is not the speed clamp in the inspected file. More importantly, `DEFAULT_MASS_KG` and `DEFAULT_MAX_THRUST_N` cite `TrackerConfig`, but `TrackerConfig` still owns its own literals, so the relationship is descriptive, not enforced. The file reads like it centralizes these values already; in practice it centralizes only the `DroneConstraints` defaults.

## MINOR

1. The `competition -> planning` layering is not a Python cycle, but it is conceptually muddy.

   `planning.trajectory_optimizer` importing `competition.drone_spec` works today because `competition/__init__.py` is inert and `drone_spec.py` has no planning imports. The concern is architectural: a generic planning module now depends on a package named for competition adapters/interfaces. If this module is really "synthetic bench vehicle profile," a neutral `config`, `vehicle`, or `dynamics` layer would make the dependency direction clearer. Not a blocker for iter-010.

2. Not every `15.0`, `1.0`, or `20.0` should be moved.

   `1.0` sequencer proximity/pass margins and `20.0` ceiling checks are gate/airspace semantics, not drone dynamics. `RacingLineConfig.select_velocity_mps = 15.0` is documented as a legacy basin-selection knob rather than the execution max speed, so sourcing it from `drone_spec` may re-couple the path-selection/execution split iter-009 fixed. The drone-spec candidates are the physical envelope literals: acceleration, velocity, drag, yaw-rate, mass, gravity, thrust, tilt, and body-rate.

## Direct answers

- Still depends on `max_acceleration=20.0`? No active callsite found. Only stale comments and unrelated numeric uses remain.
- Is `competition <- planning` sound? Runtime-safe, but the package boundary is questionable.
- Are docstrings/provenance accurate? Partially. The key stale `20.0` comment in `auto_velocity` is inaccurate, and `drone_spec` overstates how widely it is consumed.
- Should other literals be sourced from `drone_spec`? Yes for duplicate synthetic drone/bench envelope values listed above; no for gate margins, airspace ceilings, and intentional racing-line selection knobs.
