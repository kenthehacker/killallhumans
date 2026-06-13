# Iter 005 Adversarial Review — GPT-5.5 Extra High

## Summary
The synthetic matrix improvement is real: forcing `max_velocity_mps=8.0` makes 5 previously failing tracks complete, and `slalom` needs the new `6.0` override to pass. The critical gap is that this velocity "global default" only exists in `run_synthetic_benchmark`; the PyBullet benchmark, JSON loader, and competition-facing `RacePipeline` ignore `max_velocity_mps`, so the shipped fix does not reach the surfaces that matter most. I also found that `8.0`/`6.0` are sweep-derived magic numbers, and the plan validator does not replay the same sequencer config as runtime.

## Findings (ordered by severity)

### F1. Velocity unlock is synthetic-only, not global — BLOCKER
- **File(s)**: `scripts/benchmark.py:344`, `scripts/benchmark_matrix.py:78`, `scripts/benchmark.py:799`, `sim_pybullet/env.py:213`, `race_pipeline.py:90`
- **Issue**: Iter-005 claims the trajectory max velocity was lowered "globally" with per-track JSON override, but the new key is only read by `run_synthetic_benchmark`. The matrix that flipped to 6/7 passing calls only `run_synthetic_benchmark`, while `run_sim_benchmark` builds `DroneConstraints(max_velocity=planner_cfg.plan_max_speed_mps)` and `DroneRaceEnv.load_config()` only carries `planner`, `racing_line`, and `sequencer` sections. The competition-facing `RacePipeline` still defaults to `PipelineConfig.max_speed = 12.0` and passes that directly into `DroneConstraints`, so the `8.0` default and `slalom`'s top-level `max_velocity_mps: 6.0` do not affect PyBullet or the actual adapter path.
- **Repro**: Code inspection is enough: `slalom.json:3` is a top-level key, but `sim_pybullet/env.py:213-228` drops it. `scripts/benchmark_matrix.py:78-80` uses the synthetic path that reads it, while `scripts/benchmark.py:799-803` and `race_pipeline.py:249-253` do not. My in-memory sweep confirmed the synthetic effect: `slalom` at `8.0` crashes at gate-3, `slalom` at `6.0` passes 8/8, but that evidence is only for the synthetic kinematic harness.
- **Fix sketch**: Make one canonical speed source. Prefer moving this under the existing `planner.plan_max_speed_mps` config, or teach `RaceConfig`/`PlannerConfig` to ingest `max_velocity_mps` and pass it through both synthetic and PyBullet paths. Add a regression test that loads `slalom.json` through `DroneRaceEnv.load_config()` and proves the planner max velocity is `6.0`, plus a `RacePipeline` test/config path so competition execution cannot silently run at `12.0`.
- **Confidence**: high — the line-level plumbing is unambiguous.

### F2. `8.0` and `slalom: 6.0` are new course-specific magic numbers — MAJOR
- **File(s)**: `.loop/specs/0_charter.md:12`, `scripts/benchmark.py:337`, `sim_pybullet/configs/slalom.json:2`, `planning/trajectory_optimizer.py:1273`, `planning/trajectory_optimizer.py:1430`
- **Issue**: The charter forbids course-specific magic numbers, but iter-005 replaces one swept constant (`15.0`) with another swept constant (`8.0`) and adds a course-specific `6.0` for `slalom`. The code comment says the values were selected because the current bench passes, not because they are derived from gate spacing, curvature, available lateral acceleration, or controller limits. This is especially awkward because `TrajectoryOptimizer` already has a TOPP-style curvature limit (`a_centripetal = 10.0`, `v_limit = sqrt(a/k)`), but the new outer speed cap bypasses that machinery instead of making it authoritative.
- **Repro**: My in-memory sweep shows the overfit shape: `slalom` at `15.0` crashes at gate-8, `8.0` still crashes at gate-3, and `6.0` passes; `grand_tour`, `straight_hairpin`, `vertical_cliff`, and `aigp_default` pass at `8.0`. That proves lower velocity helps, but not that `8.0` or `6.0` generalize to an unseen course.
- **Fix sketch**: Replace the flat default/override with an auto-derived speed ceiling from track geometry: minimum gate spacing, turn angle/curvature, gate opening margin, and a measured controller lateral acceleration envelope. If manual overrides remain, mark them experimental and fail the matrix unless an auto-derived value is present.
- **Confidence**: high — the behavior is empirically real, but the selected constants are not derived.

### F3. Plan validator does not replay the same sequencer config as runtime — MAJOR
- **File(s)**: `planning/plan_validator.py:53`, `scripts/benchmark.py:325`, `scripts/benchmark.py:356`, `scripts/benchmark.py:779`
- **Issue**: The validator is supposed to distinguish "plan illegal" from "tracker cannot follow legal plan", but it constructs its own `SequencerConfig` with `proximity_pass_distance=0.0` by default. The synthetic runtime sequencer uses `proximity_pass_distance=1.0`, and PyBullet uses `1.2` plus any course sequencer overrides. `run_synthetic_benchmark` calls `validate_trajectory(trajectory, gate_specs, dt=dt)` without forwarding the runtime config, so `plan_validation` can disagree with the exact sequencer that later scores the run.
- **Repro**: Any trajectory that approaches inside the current gate opening and within the runtime proximity radius without a sampled plane crossing can be credited by runtime but reported incomplete by the validator. Conversely, future per-track sequencer overrides in JSON will be ignored by plan validation even though the sim uses them.
- **Fix sketch**: Change `validate_trajectory` to accept either a full `SequencerConfig` or the same config kwargs used by the caller. In `run_synthetic_benchmark`, build one `SequencerConfig`, pass it to both `GateSequencer` and `validate_trajectory`, and add a test where proximity pass is the only reason runtime completes.
- **Confidence**: medium-high — exact false-negative frequency depends on trajectories, but the config mismatch is definite.

### F4. Figure8 is not necessarily "fundamental"; current planner just uses the overlapping part of gate-1 — MAJOR
- **File(s)**: `sim_pybullet/configs/figure8.json:16`, `sim_pybullet/configs/figure8.json:20`, `gate_sequencing/sequencer.py:444`
- **Issue**: The brief's diagnosis says figure8's coplanar gate-1/gate-5 overlap is fundamental. It is fundamental for the current centerline-ish trajectory, but not mathematically impossible. Gate-1 and gate-5 share `x=5`, `y=0`, and `yaw=0`, with `z=2.0` vs `z=2.2`; with `1.2m` interior height, gate-1's opening spans roughly `z=[1.4, 2.6]` while gate-5's spans `z=[1.6, 2.8]`. There is a narrow legal strip through gate-1 below gate-5's opening. The sequencer's future-gate scan correctly DQs any gate-5 opening crossing, but the planner has no constraint telling it to aim for the non-overlap strip.
- **Repro**: Current matrix at `8.0` reports `plan_validation.reason = DQ at t=0.67s: out_of_order:gate-5`; at `6.0`, the sim still DQs `out_of_order:gate-5`. That is consistent with the trajectory crossing the overlapping zone, not proof that no legal gate-1 crossing exists.
- **Fix sketch**: Keep strict sequencing, but make the next planning phase solve gate traversal as a polygon/corridor problem: for each gate, choose a point inside the current gate interior and outside all future-gate interiors on the same plane, with a safety margin. If the non-overlap strip is too narrow after margins, then report the track as geometrically invalid with a proof.
- **Confidence**: medium — the available non-overlap strip is small, so safety margins may eliminate it, but the current "fundamental" claim is stronger than the geometry supports.

### F5. `tracker_config_overrides` is raw, unvalidated config injection — MINOR
- **File(s)**: `scripts/benchmark.py:414`, `scripts/benchmark.py:420`, `control/mpc_tracker.py:33`
- **Issue**: The new `tracker_config_overrides` and course-level `tracker_overrides` are blindly merged into `tracker_kwargs` and passed to `TrackerConfig(**tracker_kwargs)`. `TrackerConfig` is a plain dataclass with no range validation, so typo keys crash the bench and nonsensical values like negative gains, zero thrust, or invalid tilt limits can silently produce meaningless results. Course-level overrides also take precedence over per-call overrides, which is surprising for an experimentation API.
- **Repro**: A config containing `"tracker_overrides": {"kp-x": 1.0}` will raise `TypeError` in `TrackerConfig`; `"max_thrust_n": 0.0` can break normalized thrust math. The matrix catches exceptions as regressions, but the API gives no clear validation error or allowed-key filtering.
- **Fix sketch**: Reuse `_dataclass_from_overrides` for known keys, report unknown keys explicitly, and add `TrackerConfig.validate()` for positive mass/thrust, finite gains, and sane tilt/thrust bounds. Decide and document precedence: call-time overrides should probably beat course config for experiments.
- **Confidence**: high for the validation gap; medium on precedence preference.

## Things iter-005 got right
- The velocity diagnosis is directionally right in the synthetic harness: lowering speed reduces max tracking error and converts several plan-ok/sim-fail tracks into completions.
- Race_01 did not violate the 30s synthetic threshold after the default drop: my run finished in `14.92s` versus `13.69s` at `15.0`.
- The plan validator is useful despite the config mismatch: it correctly separates figure8 (`plan_ok=false`) from the other failing-at-15 tracks (`plan_ok=true`, tracker/speed problem).
- Keeping strict future-gate DQ is correct; figure8 should be fixed in planning geometry, not by weakening gate order.

## What I did NOT review
- I did not run the PyBullet sim or visual demo; this review is based on code inspection plus synthetic matrix/velocity sweeps.
- I did not inspect every historical ILC artifact or retune ILC for the lower-speed trajectories.
- I did not review MAVLink adapter behavior beyond the `RacePipeline` planning speed plumbing.
