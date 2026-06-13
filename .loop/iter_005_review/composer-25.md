# Iter 004/005 Adversarial Review — Composer 2.5

## Summary

The iter-004 plan validator is a high-value diagnostic: on the current matrix, every passing track has `plan_ok=True`, and figure8 fails at all tested speeds (6/8/15 m/s) with the same `out_of_order:gate-5` — velocity cannot fix it. The iter-005 velocity drop (15→8 m/s + slalom 6 m/s) is a real win (6/7 matrix pass, race_01 at 14.9 s under the 30 s cap) but the stated root cause is partly wrong for 5/6 recovered tracks: those plans were already legal at v=15; the synthetic kinematic loop could not track them without overshoot. The fix is closer to “slow the reference so the bench PD model stays inside the gate sequence” than “planner exceeded centripetal envelope.” Charter risk remains: 8.0 and 6.0 are sweep-tuned globals, and the validator still replays with different sequencer settings than the sim.

## Findings (ordered by severity)

### F1. Plan validator replays with stricter sequencer config than the bench — [MAJOR]
- **File(s)**: `scripts/benchmark.py:325-328`, `scripts/benchmark.py:355-356`, `planning/plan_validator.py:58-58`, `planning/plan_validator.py:93-96`
- **Issue**: `run_synthetic_benchmark` builds `GateSequencer` with `proximity_pass_distance=1.0`, but calls `validate_trajectory(trajectory, gate_specs, dt=dt)` without forwarding that flag, so the validator defaults to `proximity_pass_distance=0.0`. Plans can be judged under different pass rules than the sim. Today most tracks agree at both settings; the mismatch is latent for tracks that skim gates within 1 m.
- **Repro**: Call `validate_trajectory(..., proximity_pass_distance=0.0)` vs `1.0` on a trajectory engineered to pass only via proximity credit; bench `plan_validation.ok` may disagree with what the runtime sequencer would do.
- **Fix sketch**: Pass `proximity_pass_distance=seq.config.proximity_pass_distance` (and `pass_through_margin`) into `validate_trajectory`; add a test in `tests/test_plan_validator.py` that asserts validator/bench config parity.
- **Confidence**: high

### F2. Velocity diagnosis oversells “centripetal envelope”; true failure mode was tracker overshoot on legal plans — [MAJOR]
- **File(s)**: `scripts/benchmark.py:337-347`, `scripts/benchmark.py:440-443`, commit `7530f1f` message
- **Issue**: Iter-004 already showed 5/7 failing tracks with `plan_ok=Y` and `sim_ok=N`. Re-running `validate_trajectory` on grand_tour, slalom, and straight_hairpin at **v=15** still yields `plan_ok=True` with full gate credit. Lowering `DroneConstraints.max_velocity` to 8 m/s fixes sim completion by shrinking reference speed/acceleration so the geometric tracker + kinematic integrator (`max_accel=15`, `max_speed=15` at `scripts/benchmark.py:441-443`) cross fewer future-gate planes — not because the optimizer was emitting illegal geometry.
- **Repro**: `python3 -m scripts.benchmark_matrix` pre-7530f1f vs post; or offline validator at v=15 on `grand_tour.json` (plan passes, historical sim DQ’d).
- **Fix sketch**: Document the failure mode as “reference-tracking overshoot in synthetic bench”; tie kinematic `max_speed` to `data.get("max_velocity_mps", 8.0)`; keep SFC/corridor work for true planner failures (figure8).
- **Confidence**: high

### F3. Global 8.0 m/s and slalom 6.0 m/s are charter-style magic numbers, not geometry-derived — [MAJOR]
- **File(s)**: `scripts/benchmark.py:344-344`, `sim_pybullet/configs/slalom.json:2-3`, `.loop/specs/0_charter.md:12-12`
- **Issue**: Default `max_velocity_mps` is 8.0 because a sweep made the matrix pass; slalom adds 6.0 for the same reason. `SpeedProfiler` already implements `v = sqrt(max_accel / κ)` at `planning/racing_line.py:712-714` but the bench path does not wire track `max_velocity` from that derivation. Competing on unknown courses with fixed 8/6 is overfitting to this harness.
- **Repro**: Remove `max_velocity_mps` from `slalom.json` — slalom drops to 2/8 gates at v=8 (`crash_gate:gate-3` in local run); matrix “pass” depends on the hand-picked 6.0.
- **Fix sketch**: Set `DroneConstraints.max_velocity` from min(gate-spacing heuristic, `SpeedProfiler` backward pass, kinematic `max_speed`); allow per-track JSON only as an override with `_derived_from` metadata, not as the primary knob.
- **Confidence**: high

### F4. Kinematic sim still caps speed at 15 m/s while trajectory caps at 8 — [MAJOR]
- **File(s)**: `scripts/benchmark.py:344-347`, `scripts/benchmark.py:441-443`, `scripts/benchmark.py:591-593`
- **Issue**: Trajectory reference is limited to 8 m/s (or 6 on slalom), but the integrator clamps `vel` to `max_speed = 15.0`. Aggressive tracking error can still produce >8 m/s motion and future-gate crossings that a perfect-tracking validator will not see.
- **Repro**: Inspect `controller_trace` / velocity norm on a tight track when `avg_tracking_error` spikes; compare to `trajectory.sample(t).velocity` magnitude.
- **Fix sketch**: `max_speed = float(data.get("max_velocity_mps", 8.0))` (and optionally `max_accel` from `DroneConstraints.max_acceleration`) in `run_synthetic_benchmark`.
- **Confidence**: medium-high

### F5. figure8 failure is planner-topology (coplanar gates), not velocity — diagnosis correct, fix path still open — [MAJOR]
- **File(s)**: `sim_pybullet/configs/figure8.json:16-20`, `sim_pybullet/configs/figure8.json:28-28`, `planning/plan_validator.py:103-114`
- **Issue**: Gates 1 and 5 share `(x=5, y=0)` with different `z` (2.0 vs 2.2). Min-snap from start toward gate-1 crosses gate-5’s plane early. Validator: `DQ at t=0.67s: out_of_order:gate-5` at v=8; same DQ at v=6 and v=15. Sim terminates `crash_gate:gate-5` at ~1.0 s (tracker deviation / strut hit, not the same terminal class as validator). No velocity sweep fixes `plan_ok`.
- **Repro**: `run_synthetic_benchmark(config=figure8.json)` → `plan_validation.ok=False`, `termination_reason=crash_gate:gate-5`.
- **Fix sketch**: Iter-006 SFC / gate-polygon (research synthesis Phase 2); optional explicit start-segment z detour in optimizer for coplanar stacks; **do not** weaken `enforce_in_order` to “credit both gates” — that violates VQ1 rules.
- **Confidence**: high

### F6. `tracker_config_overrides` bypasses safe dataclass filtering — [MINOR]
- **File(s)**: `scripts/benchmark.py:59-65`, `scripts/benchmark.py:414-427`, `control/mpc_tracker.py:33-80`
- **Issue**: `_dataclass_from_overrides` exists for other configs but tracker path does `tracker_kwargs.update(tracker_config_overrides)` then `TrackerConfig(**tracker_kwargs)`. Unknown keys raise `TypeError`; invalid types (e.g. string gain) fail late with opaque errors. `TrackerConfig` has no range validation.
- **Repro**: `run_synthetic_benchmark(..., tracker_config_overrides={"kp_xy": "seven"})`.
- **Fix sketch**: Use `_dataclass_from_overrides(TrackerConfig, overrides)` for both call-site and `tracker_overrides` JSON block.
- **Confidence**: high

### F7. Validator `is_complete=False` reason does not separate trajectory exhaustion vs recovery — [MINOR]
- **File(s)**: `planning/plan_validator.py:133-137`, `gate_sequencing/sequencer.py:490-496`
- **Issue**: Incomplete plans always get `"incomplete: N/M gates passed during full-trajectory replay"`. The sequencer can enter `RaceState.RECOVERY` when far from the target gate, but the validator never surfaces that state — indistinguishable from “trajectory ended too soon.”
- **Repro**: Shorten `trajectory.total_time` on a multi-gate course; same reason string whether the drone was in RECOVERY or simply never reached the next gate plane.
- **Fix sketch**: After replay, if `seq._state == RaceState.RECOVERY`, set `reason` to `"stuck_in_recovery: ..."`; expose `last_event` in `ValidationResult.extras`.
- **Confidence**: medium

### F8. `samples_evaluated` uses a bogus `dir()` guard — [NIT]
- **File(s)**: `planning/plan_validator.py:150-150`
- **Issue**: `step + 1 if "step" in dir() else samples` — `dir()` in function scope always includes `step` after the loop; the else branch is dead. If `samples == 0` and the loop never runs, this would `NameError` (edge case for degenerate trajectories).
- **Repro**: Empty positive `total_time` edge case.
- **Fix sketch**: Initialize `step = -1` before the loop; use `samples_evaluated = step + 1 if step >= 0 else 0`.
- **Confidence**: high

### F9. ILC left on race_01 helix schedule at new reference speed — [MINOR]
- **File(s)**: `sim_pybullet/configs/race_01.json:15-26`, `scripts/benchmark.py:374-380`, `.loop/synthesis/research_synthesis.md:40-43`
- **Issue**: Research synthesis asked to verify per-track ILC reset. Non-race_01 tracks correctly use curvature-derived sections. race_01 still uses hand-tuned `ilc_section_overrides` from the v=15 era; ILC is recomputed on the v=8 trajectory (different segment timing) but section boundaries are still index-based `[0,200,440,740,...]`, which may mis-align with the slower profile. Matrix still passes (14.9 s, max err ~0.38 m) so this is not blocking, but it is untuned debt.
- **Repro**: Compare ILC section boundaries derived from curvature vs fixed overrides on post-7530f1f race_01 trajectory.
- **Fix sketch**: Re-derive `ilc_section_overrides` from curvature on the v=8 trajectory or drop overrides and measure regression.
- **Confidence**: medium

### F10. `benchmark_matrix` ignores `plan_validation.ok` in acceptance — [MINOR]
- **File(s)**: `scripts/benchmark_matrix.py:99-128`
- **Issue**: Matrix propagates `plan_validation` but only fails on crash/DQ/gate rate. A future change could make `sim_passed=True` while `plan_ok=False` (e.g. proximity mismatch or lucky tracker noise) and still show matrix PASS for production tracks.
- **Repro**: Hypothetical track where sim limps through but validator flags illegal plan.
- **Fix sketch**: For non-placeholder tracks, require `plan_validation["ok"]` or add a separate `plan_regressions` list.
- **Confidence**: medium

### F11. Validator false-negative risk at coarse `dt` (sub-sample strut graze) — [MINOR]
- **File(s)**: `planning/plan_validator.py:56-56`, `planning/plan_validator.py:103-114`, `tests/test_plan_validator.py:136-152`
- **Issue**: Sampling at 0.01 s with perfect tracking can miss a brief strut graze between samples if a future change uses faster motion or coarser `dt`. Tests cover DQ and crash on constructed stubs at `dt=0.02`, not aliasing vs bench `dt=0.01` at high speed.
- **Repro**: Construct a trajectory that crosses the strut annulus only between `t` and `t+dt` with `dt=0.05`.
- **Fix sketch**: Sub-sample segments between waypoints for validator only; or require `dt <= 0.01` and document; add adversarial aliasing test.
- **Confidence**: low-medium

## Things iter-004/005 got right

- Plan validator cleanly separates planner failure (figure8 `plan_ok=N`) from tracker failure (pre-005: legal plan + sim DQ), matching the research-swarm intent at `planning/plan_validator.py:1-21`.
- Matrix regression honestly still fails figure8 while lifting five other tracks; no silent weakening of `enforce_in_order`.
- race_01 time regression is benign: 14.9 s simulated vs 13.7 s baseline, well under `THRESHOLDS["max_total_time_s"]=30.0` at `scripts/benchmark.py:54-54`.
- Six new validator tests cover DQ, crash, and degenerate inputs — better than “bench says pass.”
- `tracker_config_overrides` / `tracker_overrides` plumbing is the right seam for per-track gain experiments without editing globals.

## What I did NOT review

- PyBullet `run_sim_benchmark` path and whether sim velocity limits match synthetic changes.
- `RacePipeline` / MAVLink production path — only `run_synthetic_benchmark`.
- Full `git show` line-by-line hunks beyond the two commit stat summaries.
- Unit test suite beyond `tests/test_plan_validator.py`.
- Whether iter-005 changed `planning/racing_line_cache.json` (git status shows modified; not part of the two commits reviewed).
