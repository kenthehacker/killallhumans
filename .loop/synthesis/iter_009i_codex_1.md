# iter-009i adversarial review (`b926734`)

## BLOCKER

None found. The F9-specific change appears to remove the direct `max_velocity_mps` -> `_select_by_sim()` coupling in `planning/racing_line.py`: the scorer now constructs `DroneConstraints(max_velocity=self.config.select_velocity_mps)`, and the synthetic benchmark explicitly holds that at `15.0` while the final trajectory optimizer still uses the derived execution velocity.

## MAJOR

1. **The patch decouples waypoint selection, not the final path geometry.**

   The comments/tests repeatedly describe "path-velocity decoupling" and "racing-line geometry" as velocity-independent, but the final `TrajectoryOptimizer` still uses execution `max_velocity` in `scripts/benchmark.py` after the fixed-offset selection. That optimizer is not a pure retimer: its initial segment times scale with `constraints.max_velocity`, endpoint velocities are derived from neighboring segment times and the same velocity cap, and the polynomial samples are regenerated from those time/velocity boundary conditions. So the optimized gate offsets can be invariant while the continuous min-snap curve between those gates still changes with execution velocity.

   This matters because the original diagnosis explicitly said "same offsets can produce a different space-time curve" when `max_velocity` changes. iter-009i prevents the BO basin from flipping, but it does not prove the executed curve is geometrically invariant or safe at all execution velocities. A regression test that compares only returned gate offsets would miss a velocity-induced bulge, early gate-plane crossing, or strut clearance loss in the final trajectory.

2. **`RacingLineConfig.max_velocity_mps` is now a write-only API trap.**

   `RacingLineConfig` says `max_velocity_mps` is "the velocity the trajectory will be EXECUTED at downstream", but `RacingLineOptimizer` never returns or propagates that value. In the synthetic benchmark, execution velocity comes from the local `max_velocity` variable; in the PyBullet benchmark, it comes from top-level `race_config.max_velocity_mps`, `planner_overrides.plan_max_speed_mps`, or auto-derive; in `race_pipeline.py`, it comes from `PipelineConfig.max_speed`. A user can set `racing_line.max_velocity_mps` in overrides, have it accepted by the dataclass filter, and get no behavior change in either selection or execution.

   That ambiguity is likely to recreate F9-style confusion. If the value is intentionally informational, it should not live as a normal config knob on the optimizer, or it needs a much more explicit name such as `execution_velocity_hint_mps` plus logging/metadata showing it is ignored by selection and not used for trajectory generation.

3. **The new test suite is too tautological for the claimed behavior.**

   `test_max_velocity_does_not_affect_racing_line_geometry()` is useful as a guard against accidentally wiring `max_velocity_mps` back into `_select_by_sim()`, but it only uses `use_cache=False`, a toy four-gate layout, fixed yaw/width defaults, and compares optimized waypoint offsets. It does not cover the real `aigp_default` failure shape, the final trajectory generated at different execution velocities, `validate_trajectory()`, or cache reuse.

   The second test is more misleading: `test_select_velocity_DOES_change_geometry()` does not assert that `select_velocity_mps` changes geometry despite its name and docstring. It only checks shape/finite values, so it would pass if `select_velocity_mps` were accidentally ignored everywhere. Corner cases that would expose regressions: run the real F9/aigp_default gate layout at `{5, 8, 12, 15}` and validate the final sampled trajectory; run with `use_cache=True` across `select_velocity_mps=15 -> 6 -> 15`; assert the cache key changes with `select_velocity_mps` but not `max_velocity_mps`; and include rotated/non-default-width gates so offset comparison exercises actual gate-local geometry.

## MINOR

1. **Cache semantics are directionally correct but under-tested and still single-entry.**

   Including `select_velocity_mps` and excluding `max_velocity_mps` matches the new intended semantics. Not bumping `version` is not fatal because the key material changed, so old caches should miss. The remaining weakness is operational: `racing_line_cache.json` still stores only one cache entry, so switching tracks or selection velocities overwrites the previous result instead of maintaining separate entries. That is not a correctness blocker, but it makes "cache key split" less useful than it sounds and should be tested directly.

2. **`select_velocity_mps` is not a fully coherent scorer speed.**

   `_select_by_sim()` uses it for the candidate trajectory optimizer, but `_kinematic_eval()` still hardcodes `max_speed = 15.0` and `max_accel = 15.0`. With the default `select_velocity_mps=15.0`, this is benign. If a future caller changes `select_velocity_mps`, the trajectory timing and the lightweight tracker physics are no longer consistently parameterized by that selection speed.

3. **The comments overstate the theory.**

   The code invokes Heilmeier/Kapania and "path-velocity decoupling", but the implementation is a fixed-reference-velocity selector plus normal velocity-coupled min-snap generation. That is a pragmatic F9 fix, not the minimum-curvature/path-then-time decomposition described in the comments. Future readers may trust the literature framing and miss the remaining coupling above.

4. **The test naming/commentary claims bit-identical geometry but uses tolerance.**

   The test says "bit-identical offsets" but uses `assert_array_almost_equal(..., decimal=6)`. Tolerance is fine, but the wording should match the assertion.

## Bottom line

No blocker for the narrow F9 fix: execution `max_velocity_mps` no longer controls BO basin selection, and the cache key now follows the actual selector velocity. The main adversarial concern is scope creep in the claim: this is not full path-velocity decoupling. It freezes waypoint-offset selection at a legacy reference velocity, then still generates a velocity-coupled final trajectory. The tests should lock that distinction down so a future "cleanup" does not either reintroduce the basin switch or assume the continuous path is velocity-invariant when it is not.
