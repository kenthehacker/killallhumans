# iter-029 follow-up: planner acceleration recommendation for iter-030

## Recommendation

Choose option 1 for iter-030: cap planner aggression, but do it as an explicit
QuadrotorDrone/PyBullet-proxy profile rather than by changing global drone
constants. Do not increase `DEFAULT_MAX_THRUST_N` yet, and do not abandon the
existing CF2X/GPD path. The smoke failure is a planner/plant-envelope mismatch:
the new `QuadrotorDrone` path is being asked to follow polynomial acceleration
spikes that neither its thrust nor its 0.35 rad attitude cap can deliver.

I measured race_01 trajectory variants from current HEAD:

- `smoke_current`: derived max_v = 5.96 m/s, total_time = 18.35 s, max accel =
  91.83 m/s^2, p95 accel = 46.8 m/s^2.
- `race_json_4mps`: max_v = 4.0 m/s, total_time = 23.81 s, max accel =
  62.86 m/s^2, p95 accel = 29.4 m/s^2.
- `v=3.0` with max_acceleration=4.5: total_time = 30.89 s, max accel =
  53.51 m/s^2, p95 accel = 19.17 m/s^2.

So lowering velocity alone is not enough. The min-snap boundary conditions and
post-TOPP compression still create short-segment acceleration spikes. Iter-030
needs a sampled-trajectory feasibility pass.

## Why not options 2 or 3 first

Option 2, increasing thrust-to-weight, is not a one-line fix. With
`DroneConfig.max_roll_angle = max_pitch_angle = 0.35`, lateral acceleration is
tilt-limited before it is thrust-limited. A 40 N drone still cannot follow
50-90 m/s^2 lateral spikes at 20 degrees of tilt. Raising thrust also rebaselines
`competition/drone_spec.py`, `TrackerConfig`, synthetic clamps, trajectory
velocity derivation, ILC schedules, and existing benchmark expectations.

Option 3, staying CF2X, is the right path for the existing `DroneRaceEnv`
benchmark, but it does not answer the iter-029 blocker. `scripts/benchmark.py`
already uses GPD/CF2X through `DroneRaceEnv`; the failing artifact is the new
`QuadrotorDrone + step_reference` smoke harness. Keep CF2X working, but make the
Quad smoke honest instead of hiding the mismatch.

## Concrete iter-030 implementation

1. Add opt-in sampled acceleration retiming to
   `planning/trajectory_optimizer.py`.

   Add `PlannerConfig` fields, default disabled:
   - `sampled_accel_cap_mps2: float = 0.0`
   - `sampled_accel_retime_quantile: float = 1.0`
   - `sampled_accel_retime_margin: float = 1.10`
   - `sampled_accel_retime_max_iters: int = 5`

   After `_topp_retime()` and `_inflate_vertical_climbs()`, but before the final
   `_generate_trajectory()`, call a new `_enforce_sampled_accel_cap(...)` when
   the cap is positive. The helper should:
   - Generate a temporary trajectory with current `segment_times`.
   - Compute sampled acceleration norms.
   - Attribute samples to segment time ranges via cumulative segment times.
   - For each segment whose peak or configured quantile exceeds the cap, stretch
     that segment by `sqrt(observed / cap) * margin`.
   - Iterate until all segments are within cap or the max iteration count is hit.

   Use sampled acceleration, not the existing rough average-velocity penalty:
   the failure is from polynomial peaks, not just segment-average speed.

2. Add a Quad-specific planner profile in
   `scripts/smoke_quadrotor_drone_race.py`.

   Stop using only `derive_safe_max_velocity(specs)` for this smoke. Load
   `PlannerConfig(**cfg["planner"])`, then override for the Quad proxy:
   - `plan_max_speed_mps = 3.0`
   - `cmd_max_speed_mps = 3.0`
   - `sampled_accel_cap_mps2 = 4.0`
   - `sampled_accel_retime_margin = 1.10`

   Construct `TrajectoryOptimizer` with both:
   - `DroneConstraints(max_velocity=3.0, max_acceleration=4.0)`
   - the Quad planner config above

   Also print trajectory diagnostics in the JSON: `trajectory_time_s`,
   `planned_max_accel_mps2`, `planned_p95_accel_mps2`, `quad_hover_lateral_cap_mps2`
   where hover lateral cap is `gravity * tan(max_roll_angle)`.

3. Adjust only the smoke duration, not benchmark thresholds.

   Set smoke `duration = max(30.0, trajectory.total_time + 5.0)`. A physically
   honest Quad proxy may need 60-90 s on race_01 after acceleration retiming.
   That is acceptable for this diagnostic. Do not relax `scripts/benchmark.py`
   race-time thresholds for the main synthetic/GPD benchmarks.

4. Add a unit-level guard.

   In `scripts/benchmark.py` unit tests, add a small trajectory test that enables
   `sampled_accel_cap_mps2=4.0` on a tight three-gate course and asserts the
   generated sampled max acceleration is within `4.0 * 1.15`. This catches future
   regressions without requiring PyBullet.

## Iter-030 success criteria

- `python3 scripts/benchmark.py --mode unit` passes.
- `python3 scripts/smoke_quadrotor_drone_race.py` no longer falls at t=1.16 s.
- Smoke JSON reports planned acceleration capped near 4-5 m/s^2.
- Gate completion may still be partial after one iteration; the immediate blocker
  is physical infeasibility and ground crash, not sub-30 s race performance.

This keeps the CF2X benchmark path stable, avoids guessing a fake AIGP thrust
number, and creates the missing planner invariant: a trajectory handed to a
thrust/tilt-limited rigid-body drone must have sampled accelerations inside that
drone's envelope.
