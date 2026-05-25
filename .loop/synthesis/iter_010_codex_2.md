# iter-010 adversarial review #2 (`f1505ee`)

Scope: `competition/drone_spec.py` plus the `DroneConstraints` default rewiring in `planning/trajectory_optimizer.py`. This pass intentionally looked for import-time side effects, circular imports, and edge cases not covered by review #1's test-coverage / duplicate-literal critique.

## Finding

### M1. `DroneSpec` is a public dataclass, but it is not shape-compatible with the planner object it is supposed to centralize

`competition/drone_spec.py` introduces:

- `DroneSpec.max_velocity_mps`
- `DroneSpec.max_acceleration_mps2`
- `DroneSpec.max_jerk_mps3`
- `DroneSpec.max_tilt_rad`
- `DroneSpec.max_body_rate_rad_s`
- `DroneSpec.max_thrust_n`
- `DroneSpec.mass_kg`
- `DroneSpec.gravity_mps2`

`TrajectoryOptimizer`, however, still consumes a `DroneConstraints`-shaped object with names like:

- `max_velocity`
- `max_acceleration`
- `max_jerk`
- `max_tilt_angle`
- `max_body_rate`
- `max_thrust`
- `mass`
- `gravity`

The commit's current implementation is safe because `DroneConstraints` imports scalar constants from `drone_spec`, not a `DroneSpec` instance. The edge case is API drift: the new module docstring says `DroneSpec` should be instantiated "at module boundaries where downstream code expects a dataclass" and that `DroneConstraints` "becomes a thin wrapper that reads its defaults from here." A future caller can reasonably try:

```python
TrajectoryOptimizer(constraints=DroneSpec())
```

That will fail later with `AttributeError` when the optimizer reads `self.constraints.max_velocity` or `self.constraints.max_acceleration`. This is not an import-time failure, so smoke imports and unit tests will miss it unless a caller actually passes the new dataclass across the boundary.

Risk: medium-low today, because no current call site passes `DroneSpec`. It is still a real edge-case footgun created by the new public type. The fix direction would be one of: add `DroneSpec.to_constraints()`, make `DroneConstraints.from_spec(spec)`, align field names, or explicitly document `DroneSpec` as constants/provenance only and not a planner config object.

## Non-findings

No circular import found in the touched path. The relevant graph is one-way:

`planning.trajectory_optimizer -> competition.drone_spec -> dataclasses`

`competition/__init__.py` is only a docstring, so importing `competition.drone_spec` does not pull in MAVSDK adapters, PyBullet, calibration code, or `planning`.

No meaningful import-time side effects found. `competition.drone_spec` only defines constants and a frozen dataclass. `planning.trajectory_optimizer` already had heavier import-time dependencies (`numpy`, `scipy.optimize`) before this change; the new import adds only the lightweight `competition.drone_spec` module.

One minor hygiene note: `competition.drone_spec` imports `field` but never uses it. There is no repo lint config found, and `py_compile` does not care, so I would not block on it. If a future lint gate uses pyflakes/ruff F401, this becomes a trivial failure.

I am not re-raising review #1's broader "single source of truth is not test-enforced" issue. It remains true that `scripts/benchmark.py`, `planning/auto_velocity.py`, `planning/racing_line.py`, and the ILC helper still have independent 15.0 literals or stale comments, but that was already covered in `.loop/synthesis/iter_010_composer_2.md`.

## Verification run

- `python3 -m py_compile competition/drone_spec.py planning/trajectory_optimizer.py` passed.
- Import smoke passed for `competition.drone_spec`, `planning.trajectory_optimizer`, `planning.auto_velocity`, `control.mpc_tracker`, `race_pipeline`, and `scripts.benchmark`.
- `python3 scripts/benchmark.py --mode unit 2>/dev/null` passed: 9/9 unit benchmark checks, `overall_passed: true`.

Bottom line: no blocker in the scoped import/circular-import surface. The main missed edge case is the newly exposed `DroneSpec` type looking like a config object while not satisfying the existing `DroneConstraints` protocol.
