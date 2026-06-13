# Iter 004/005 Adversarial Review — Opus 4.7 Max-Thinking

## Summary

The 6/7 matrix flip is real and the plan-validator scaffolding is a
genuinely useful diagnostic, but the iter-005 velocity story has two
under-acknowledged failure modes: (a) the new `max_velocity_mps`
plumbing only reaches `run_synthetic_benchmark` (PyBullet and
`RacePipeline` still use 4.0 and 12.0 respectively, so the matrix's
"global default" is not global), and (b) the win was paid for by an
8.9 % race-time regression on race_01 (13.69 s → 14.92 s) — moving
**away** from CLAUDE.md's aspirational `< 14 s` target on the
production-quality track. I also found that the plan validator
validates the bare polynomial trajectory while the runtime applies
ILC offsets and a gate-seeking fallback to it, opening a new
false-negative class. The 8.0 / 6.0 numbers are sweep-derived
overfits and should be track-geometry-derived per charter.
Reproduced via `python3 -m scripts.benchmark_matrix --json-only`
and a per-track validator/sequencer config inspection.

## Findings (ordered by severity)

### F1. Plan validator replays the BARE polynomial — bench reference is `traj + ilc_offsets`, opening a new false-negative class — [MAJOR]
- **File(s)**: `planning/plan_validator.py:107`, `scripts/benchmark.py:355-356`, `scripts/benchmark.py:516-525`, `scripts/benchmark.py:387-399`
- **Issue**: `validate_trajectory` calls `trajectory.sample(t)` and feeds `ref.position` straight into the sequencer. The bench, however, applies a per-step ILC correction to the same reference at line 520 (`target_pos = target_pos + ilc_offsets[step]`) plus a velocity correction at 525. The validator's "trajectory is legal" verdict therefore covers the un-corrected polynomial only. If the ILC table — which can be ≤ `max_correction_m` (race_01 sets it via `ilc_global_overrides` and `ilc_section_overrides`, currently `max_correction_m` defaults are large enough to matter near gates) — shifts the reference into a future-gate plane that the bare trajectory clears, the validator says `ok=True` and the bench DQs. Same hole for the post-`trajectory.total_time` gate-seeking fallback at `benchmark.py:528-538`, which switches the target to `gate.position` directly — a completely synthesised reference the validator never replays.
- **Repro**: Construct a trajectory whose bare polynomial passes gate-N at z=2.0 (inside gate-N opening) but whose ILC offset[step_at_gate_N] pushes z up by, say, 0.4 m; with a future gate-(N+1) at z=2.4 sharing the same plane (figure8-style), the runtime now crosses gate-(N+1) opening too. `validate_trajectory(trajectory, …)` returns `ok=True`; runtime emits `out_of_order:gate-(N+1)`.
- **Fix sketch**: Either (a) accept an optional `reference_corrections: Callable[[int, TrajectoryPoint], TrajectoryPoint]` so the bench can hand the validator the post-ILC reference, or (b) at minimum, document that `plan_validation.ok` covers the polynomial only and add a SECOND validator pass after the bench's reference-shaping pipeline. The current API silently misses the most common iter-001..003 failure category (cross-track ILC pushing reference off-axis at high-curvature gates).
- **Confidence**: high — the code paths are unambiguously different.

### F2. race_01 race-time regressed from 13.69 s → 14.92 s — CLAUDE.md `< 14 s` aspirational target now violated — [MAJOR]
- **File(s)**: `scripts/benchmark.py:344`, `CLAUDE.md:48-57`, `.loop/state/regression_baseline_2026_05_24.json:48-60`
- **Issue**: The iter-005 commit message says "race_01 still completes 12/12 cleanly, just slightly slower" but doesn't quantify it or check it against the aspirational target. Live numbers from `benchmark_matrix --json-only` on HEAD: race_01 finishes in **14.92 s** at v=8, vs **13.69 s** at v=15 (baseline). The CLAUDE.md aspirational target table reads `Race time < 14s` — race_01 was at the boundary at v=15 and is now **0.92 s above the aspirational ceiling**. The hard threshold (`THRESHOLDS["max_total_time_s"] = 30.0`) is still fine, but the loop's stated goal is to TIGHTEN toward aspirational, not retreat from it.
- **Repro**: `python3 -m scripts.benchmark_matrix --configs race_01 --json-only | jq '.tracks.race_01.sim_time_s'` → `14.92`. Compare `.loop/state/regression_baseline_2026_05_24.json` → `13.69`. CLAUDE.md aspirational table explicitly cites `< 14s` race time.
- **Fix sketch**: Treat the race-time as a regression on its own track and make the velocity ceiling track-geometry-derived (see F3) so race_01 can keep its 15 m/s while slalom is capped at its centripetal-feasible v. Add a per-track race-time regression check in `benchmark_matrix.py` (delta vs `regression_baseline_2026_05_24.json` > 5 % is a regression flag, not just a hard-threshold violation). Update CLAUDE.md aspirational target if the team consciously chose to trade race_01 speed for multi-track survival.
- **Confidence**: high — both numbers measured this session.

### F3. `8.0` global default + `slalom: 6.0` override are sweep-fitted magic numbers; charter forbids per-course constants — [MAJOR]
- **File(s)**: `scripts/benchmark.py:344`, `sim_pybullet/configs/slalom.json:3`, `.loop/specs/0_charter.md:12`, `planning/racing_line.py:362`, `planning/racing_line.py:516`, `planning/racing_line.py:675`, `planning/trajectory_optimizer.py:33`, `planning/trajectory_optimizer.py:286`
- **Issue**: The charter explicitly says: "No course-specific magic numbers… must be parameterised, derived from track geometry, or deleted." Iter-005 replaces `15.0` with `8.0` — chosen because it makes the matrix flip — and adds a course-specific `6.0` for slalom — chosen because the matrix-flip didn't take slalom along for the ride. Neither value is derived from track geometry. `SpeedProfiler` already has the right machinery in `planning/racing_line.py:712-714` (`v = sqrt(a_centripetal / κ)`) but the bench wires `DroneConstraints.max_velocity` directly from JSON, bypassing it. The kinematic sim's own `max_accel = 15.0` at `benchmark.py:441` is the upper bound that **actually** sets the centripetal envelope; nobody derived `8.0` from `15.0` and slalom's minimum κ.

  Worse: there are now **at least six** independent 15.0 m/s constants in the planning tree (`trajectory_optimizer.py:33`, `:286`, `racing_line.py:362`, `:516`, `:675`, `benchmark.py:441`). Lowering one to 8.0 doesn't lower the others — `RacingLineOptimizer._evaluate_with_trajectory` still picks the best racing-line candidate **as scored at v=15** (`racing_line.py:362`), then the trajectory optimizer downstream squashes to v=8. The chosen line is optimized for the wrong speed.
- **Repro**:
  - Remove `max_velocity_mps: 6.0` from slalom.json, re-run matrix → slalom regresses (per composer-25's repro and consistent with the iter-005 commit's sweep table).
  - Grep: `rg -n "max_velocity|max_speed = 15" planning/` returns six call-sites — none lowered with the bench's 8.0.
- **Fix sketch**:
  - Make `max_velocity` derived from the SpeedProfiler curvature envelope for each segment, with a single per-bench `a_centripetal` parameter (kinematic equivalent to drone's actual lateral-G limit, ~15 m/s²) and per-segment κ.
  - Single source of truth: hoist the global `max_velocity` into `DroneConstraints` and have **every** caller (bench, racing_line optimizer, trajectory optimizer's internal sim, kinematic integrator's `max_speed`) read from it.
  - Track configs may carry `max_velocity_mps_override` ONLY with a `_derived_from` field explaining why the auto-value is wrong; matrix should reject overrides without that justification.
- **Confidence**: high — geometry-derivable, six-callsite inconsistency directly grep-verifiable.

### F4. The new `max_velocity_mps` reaches `run_synthetic_benchmark` only; PyBullet, `DroneRaceEnv.load_config`, and `RacePipeline` still use the old `4.0` / `12.0` defaults — [BLOCKER for the competition path]
- **File(s)**: `scripts/benchmark.py:344` (new key), `scripts/benchmark.py:799-803` (PyBullet path uses `planner_cfg.plan_max_speed_mps`), `planning/trajectory_optimizer.py:603` (`PlannerConfig.plan_max_speed_mps: float = 4.0`), `race_pipeline.py:90` (`max_speed: float = 12.0`), `race_pipeline.py:240`, `race_pipeline.py:251`
- **Issue**: Three different defaults now coexist:
  - **`run_synthetic_benchmark`** (matrix path): `data.get("max_velocity_mps", 8.0)` → **8.0**
  - **`run_sim_benchmark`** (PyBullet path): `planner_cfg.plan_max_speed_mps` → **4.0** (from `PlannerConfig`)
  - **`RacePipeline`** (MAVLink / competition path): `PipelineConfig.max_speed` → **12.0**
  This is exactly the kind of plumbing drift the synthesis tried to surface as "platform honesty drift." The matrix says 6/7 pass; the PyBullet bench (which would also reveal regressions) runs a completely different velocity. The competition path runs another. Slalom's `max_velocity_mps: 6.0` override is read by **none** of the non-synthetic paths.
- **Repro**: Grep `max_velocity_mps` across the repo:
  - `scripts/benchmark.py:344` (only synthetic)
  - `sim_pybullet/configs/slalom.json:3` (declared but not consumed by `DroneRaceEnv.load_config`)
  - `gpt-55-xhigh.md` F1 already documents the same chain.
- **Fix sketch**: Make `max_velocity_mps` a first-class field on `RaceConfig` so `DroneRaceEnv.load_config` carries it; wire it into `PlannerConfig.plan_max_speed_mps` for the PyBullet path and into `PipelineConfig.max_speed` for the MAVLink path. Add `tests/test_velocity_plumbing.py` that loads slalom.json through every entry point and asserts the chosen v matches.
- **Confidence**: high — exactly the cross-platform drift the iter-001/002 honesty fixes were meant to prevent.

### F5. Validator does NOT model the bench's airspace safety checks (`pos[2] < 0.05`, `pos[2] > 20.0`); silent false-negative class — [MAJOR]
- **File(s)**: `planning/plan_validator.py:103-114`, `scripts/benchmark.py:506-513`
- **Issue**: The bench has two non-sequencer terminal checks: `pos[2] < 0.05` → `crash_ground`, `pos[2] > 20.0` → `crash_ceiling`. The validator only replays the sequencer's gate logic — a trajectory that dips below z=0.05 (e.g. a min-snap with an early downward overshoot) or above z=20 (a high helix entry, plausibly relevant for race_01's gate-12 at z=8.5 + ILC error margin) would validate `ok=True` and crash at runtime as `crash_ground` / `crash_ceiling`. The validator currently makes a stronger guarantee than it actually checks.
- **Repro**: Construct a trajectory through gate-12 (z=8.5) with a planned upper excursion to z=20.5. Validator says `ok=True`; bench terminates `crash_ceiling`.
- **Fix sketch**: Add an airspace bounds check inside the validator's per-sample loop, parameterised by `(z_min, z_max)` (defaults: `(0.05, 20.0)` to match the bench). Or pull the bounds from `data["field"]["bounds_min"][2]` / `bounds_max[2]` so per-track airspace is honoured.
- **Confidence**: medium-high — the bench-vs-validator divergence is real; whether real trajectories hit it depends on track. race_01 with v=8 + helix peaks at z≈8.5 + ILC margin — well under the ceiling. But cliff/heli tracks could trip it.

### F6. Validator's `samples_evaluated` calculation, dead `extras` field, and unused `RaceState` import — [MINOR]
- **File(s)**: `planning/plan_validator.py:31`, `planning/plan_validator.py:50`, `planning/plan_validator.py:150`
- **Issue**:
  - **L31**: `RaceState` is imported from `gate_sequencing.sequencer` but never referenced in the module. Dead import — and composer-25 F7 sketches a fix that would actually USE it (`if seq._state == RaceState.RECOVERY: reason = "stuck_in_recovery"`), so the import was likely speculative scaffolding never wired up. Either wire it (per composer F7) or drop it.
  - **L50**: `extras: dict = field(default_factory=dict)` is declared and never populated anywhere in the module. The doc-string says "per-event metadata" but nothing emits anything to it. Dead field — and the bench at `benchmark.py:643-652` doesn't surface `extras` either, so even if populated it would be lost.
  - **L150**: `samples_evaluated=step + 1 if "step" in dir() else samples` — composer-25 F8 caught this. `"step" in dir()` is always `True` post-loop because `total_time > 0` guarantees `samples ≥ 1`; the else branch is dead. The expression also relies on Python's loop-variable-leak (`step` is defined after the `for step in range(samples)` loop) — a static analyser would flag this as undefined-after-loop, and a future `else: break`-using refactor could silently break it. Replace with `samples_evaluated=step + 1` (initialise `step = -1` before loop for safety).
- **Repro**: Grep `RaceState\|extras\|samples_evaluated` in `planning/plan_validator.py`.
- **Fix sketch**: Drop `RaceState` import + `extras` field; or use them per composer's F7 fix-sketch. Replace the `dir()` guard with explicit init.
- **Confidence**: high — dead code, observable in source.

### F7. Validator/runtime config divergence — `proximity_pass_distance` 0.0 (validator) vs 1.0 (bench) — [MAJOR — co-flagged by both other reviewers, restated with extra angle]
- **File(s)**: `planning/plan_validator.py:58`, `planning/plan_validator.py:93-96`, `scripts/benchmark.py:325-328`, `scripts/benchmark.py:356`, `scripts/benchmark.py:779-782` (PyBullet uses 1.2 + per-track overrides)
- **Issue**: Both gpt-55-xhigh F3 and composer-25 F1 catch the proximity_pass_distance mismatch. I'll add the missing angle: **the validator also doesn't accept `pass_through_margin` or `crash_margin`** — both are hard-coded to `SequencerConfig` defaults (1.0 each). Once anyone tunes either per-track (per the iter-002 M7 work that aligned bench's pass_through_margin to 1.0), the validator silently uses the dataclass default. The fix isn't "pass proximity_pass_distance through" — it's "make `validate_trajectory` accept a full `SequencerConfig`" so all sequencer settings track.
- **Repro**: Both prior reviewers' repros apply. Additionally: a future track-config that sets `sequencer.proximity_pass_distance: 1.5` in JSON would be honoured by bench's `seq` (via `_dataclass_from_overrides`) but ignored by `validate_trajectory`.
- **Fix sketch**: `def validate_trajectory(trajectory, gates, *, dt=0.01, sequencer_config: Optional[SequencerConfig] = None)`. Caller: `validate_trajectory(trajectory, gate_specs, dt=dt, sequencer_config=seq.config)`. Document deprecation of the individual kwargs.
- **Confidence**: high — direct config-source comparison.

### F8. `tracker_config_overrides` precedence is backwards from typical "explicit > implicit" — [MINOR]
- **File(s)**: `scripts/benchmark.py:414-427`
- **Issue**: Per-call `tracker_config_overrides` is `update()`d into `tracker_kwargs` FIRST, then `course_tracker_overrides = data.get("tracker_overrides", {})` is `update()`d AFTER. Result: **course-config overrides win over the explicit per-call argument**. This violates the usual "call-site explicit beats config-file" precedence — a future caller doing `run_synthetic_benchmark(..., tracker_config_overrides={"kp_xy": 14.0})` will be silently overridden by a track config's `tracker_overrides.kp_xy=7.0`. gpt-55-xhigh F5 noted this in passing; restating because the precedence rule is invisible from the API surface and there's no test guarding it.
- **Repro**:
  ```python
  result = run_synthetic_benchmark(
      config={"tracker_overrides": {"kp_xy": 7.0}, "gates": [...]},
      tracker_config_overrides={"kp_xy": 14.0},
  )
  ```
  Tracker actually uses `kp_xy=7.0`, not `14.0`.
- **Fix sketch**: Swap the order — apply `course_tracker_overrides` first, then `tracker_config_overrides`. Add a doc-string note. Add a test asserting per-call > course.
- **Confidence**: high — code inspection sufficient.

### F9. `tracker_config_overrides` accepts arbitrary keys; `TrackerConfig` has no type/range validation — [MINOR — composer F6 / gpt-55-xhigh F5 also flag; restated with safety angle]
- **File(s)**: `scripts/benchmark.py:414-427`, `scripts/benchmark.py:59-65` (existing `_dataclass_from_overrides`), `control/mpc_tracker.py:33-90`
- **Issue**: Both other reviewers noted the missing dataclass-fields filter. I'll add the safety angle: there's already a helper `_dataclass_from_overrides` at `benchmark.py:59-65` that does exactly the field-filter dance, used by `run_sim_benchmark` for `SequencerConfig` / `RacingLineConfig` / `PlannerConfig`. The new `tracker_kwargs.update(...)` path **bypasses** the existing helper, so callers get inconsistent treatment of overrides: course-level `sequencer` block is filtered, course-level `tracker_overrides` block is not. This is a regression of the iter-001 "config plumbing is uniform" pattern.
- **Repro**: `run_synthetic_benchmark(config={"tracker_overrides": {"kp_x": 1.0}})` raises `TypeError: TrackerConfig.__init__() got an unexpected keyword argument 'kp_x'` (note typo `kp_x` instead of `kp_xy`). `run_sim_benchmark(config={"sequencer": {"pass_through_margin_typo": 1.0}})` silently ignores.
- **Fix sketch**: Replace lines 420-426 with `tracker_kwargs.update(_dataclass_from_overrides(TrackerConfig, tracker_config_overrides or {}).__dict__)` style, or just route through the helper.
- **Confidence**: high.

### F10. `benchmark_matrix.py` does NOT propagate `tracker_config_overrides`; new experimentation seam unreachable through the matrix — [MINOR]
- **File(s)**: `scripts/benchmark_matrix.py:78-80`, `scripts/benchmark.py:233-238`
- **Issue**: `run_synthetic_benchmark` now accepts `tracker_config_overrides` as a positional/kwarg. The matrix runner calls it as `run_synthetic_benchmark(duration=duration, dt=dt, config=data)` — no tracker_config_overrides. So the new seam can only be exercised by direct callers of `run_synthetic_benchmark` (no current caller does). Net effect: the seam is added but unused by the matrix, with no CLI flag to inject it. It's currently scaffolding without a customer.
- **Repro**: `rg "tracker_config_overrides" scripts/` returns the function signature and the merge call inside `run_synthetic_benchmark` — no caller.
- **Fix sketch**: Either add a `--tracker-override key=val` CLI flag to `benchmark_matrix.py`, or drop the parameter until there's a caller that needs it. Otherwise it's dead surface area inviting bit-rot.
- **Confidence**: high.

### F11. `benchmark_matrix.py` doesn't fail on `plan_validation.ok=False` for production tracks; matrix can pass while plan is illegal — [MINOR — composer F10 also catches this; I'll add the figure8-specific evidence]
- **File(s)**: `scripts/benchmark_matrix.py:104-128`, `scripts/benchmark.py:643-652`
- **Issue**: Matrix's acceptance criteria (line 113-128) check `crashed`, `disqualified`, `gate_pass_rate < 0.75` — but never check `plan_validation.ok`. If a future change masks figure8's `crash_gate:gate-5` (e.g. with a tracker that happens to miss gate-5's strut by 1cm under noise), `crashed=False` and `gate_pass_rate` could rise, while `plan_validation.ok` stays `False`. Matrix would report PASS for an illegal plan.

  **Live evidence**: figure8 on HEAD has `plan_validation.ok=False` (DQ at t=0.67s) but matrix's regression line is `figure8: crashed (crash_gate:gate-5)` — i.e., the matrix is failing it on `crashed`, not on `plan_ok`. Remove the bench's `crashed` from this match (via tracker tuning that avoids the strut), and the matrix would pass figure8 despite its illegal plan.
- **Repro**: Hypothetical patch to `benchmark.py` raising `crash_margin` so the strut hit becomes a "miss" rather than "crash" — figure8 would still DQ via `out_of_order:gate-5`, but `crashed=False` and `disqualified=True` (still caught), OR if the planner gets clever enough to graze: `crashed=False`, `disqualified=False`, `gate_pass_rate=87%` (passes >= 75% but plan still illegal).
- **Fix sketch**: Add `if not t["plan_validation"]["ok"]: regressions.append(f"{name}: plan_validation failed: {t['plan_validation']['reason']}")` for non-placeholder tracks.
- **Confidence**: high — straightforward acceptance-criteria gap.

### F12. figure8 coplanar diagnosis is *practically* correct; mathematically there's a 20 mm legal strip — [MINOR — diagnosis nit]
- **File(s)**: `sim_pybullet/configs/figure8.json:16`, `sim_pybullet/configs/figure8.json:20`, `gate_sequencing/sequencer.py:444-488`
- **Issue**: Brief says "openings overlap in z — fundamental coplanar-gate issue not solvable by velocity alone." Velocity-independence is correct; "fundamental" is overstated. Concrete geometry (1.2 m opening, 0.18 m border):
  - gate-1 (z=2.0): pass z ∈ (1.4, 2.6); crash annulus z ∈ [1.22, 1.4] ∪ [2.6, 2.78].
  - gate-5 (z=2.2): pass z ∈ (1.6, 2.8); crash annulus z ∈ [1.42, 1.6] ∪ [2.8, 2.98].
  - Legal flythrough z (in gate-1 pass AND outside gate-5 outer frame): z ∈ (1.4, 1.42) — **a 20 mm strip** at the very bottom of gate-1.
- **Repro**: Ran the analytic check this session; trajectory at v=8 actually crosses x=5 at (y=0.239, z=1.634) — well inside both openings → DQ. To thread the legal strip the planner would need a downward dip from start (z=1.5) to z≈1.41 before gate-1, then up. A min-snap that respects 1.5→1.41→2.2 (gate-2) is geometrically possible but requires a sharper "z-aware" entry-offset policy than the current racing-line optimizer emits.
- **Fix sketch**: Both other reviews already point at SFC corridor work / TOGT gate-polygon traversal as the right structural fix. Restate as: figure8 is solvable IFF the planner can pick a 20 mm-wide z-corridor at gate-1 — well below the typical ~0.1 m racing-line lateral margin. The 20 mm width says you have to also tighten ILC max-corr / tracker overshoot for that gate, which iter-006 should plan for. Also worth posting: if the AIGP V Q 1 spec contains any coplanar gates with similar overlap, the entire matrix-passing approach breaks; verify with the (placeholder-only) `aigp_default.json` and call out a real-AIGP-geometry coverage gap.
- **Confidence**: high on the geometry; medium on the practicality of threading a 2 cm window with the current tracker.

### F13. `benchmark_matrix.py` acceptance bar (75 %) is laxer than per-track `THRESHOLDS["min_gate_pass_rate"]` (100 %) — pre-existing but exposed by iter-005 — [MINOR]
- **File(s)**: `scripts/benchmark_matrix.py:124-128`, `scripts/benchmark.py:53`
- **Issue**: The matrix considers a production track passing if `gate_pass_rate ≥ 0.75`; the per-track bench at `benchmark.py:53` requires `min_gate_pass_rate=1.0`. So a track that passes 11/12 gates is "matrix PASS" but "bench `sim_passed=False`." Iter-005's "6/7 tracks passing" claim is matrix-grade; on the stricter per-track bar all 6 are also passing (100 %), but a future minor regression on one gate would silently disagree between the two acceptance gates.
- **Repro**: Hypothetical 11/12 on race_01 — matrix `all_passed=True`, bench `sim_passed=False`.
- **Fix sketch**: Make the matrix acceptance bar reuse the per-track THRESHOLDS dict (`gate_pass_rate >= THRESHOLDS["min_gate_pass_rate"]` and also wire the avg/max tracking-error thresholds in). Single source of pass criteria.
- **Confidence**: high.

### F14. ILC at v=8 reuses race_01's v=15-tuned step-index section boundaries — [MINOR — composer F9 also flags; I'll add the magnitude check]
- **File(s)**: `sim_pybullet/configs/race_01.json:21-26`, `scripts/benchmark.py:373-380`
- **Issue**: race_01's `ilc_section_overrides = [[0,200,…],[200,440,…],[440,740,…],[740,99999,…]]` are **step indices** in 0.01 s. At v=15, race_01 was ≈ 1370 steps total (13.7 s); at v=8 it's ≈ 1492 steps (14.92 s). The 9 % stretch isn't uniform — the helix segments (the higher-step ranges) stretch more in real time than the straight segments, so the `200/440/740` boundaries land on different geometric phases of the helix than they did at v=15.

  **Empirical magnitude check** (this session): race_01 avg-tracking-error went 0.161 m (v=15 baseline) → **0.106 m** (v=8). So at the bench level the ILC tuning is still helping. But max-tracking-error at the gate hand-off transitions could be regressing in spots where the section boundary now falls mid-helix instead of at an inflection — composer's F9 doesn't quantify, mine doesn't either, but a per-gate err comparison vs v=15 baseline would surface it. The risk is "ILC quietly hurts gate-7→gate-8 transition at v=8 even though average looks great."
- **Repro**: Diff `per_gate_avg_error` for race_01 at v=15 vs v=8 (the iter-005 commit message doesn't include this; the baseline JSON in `.loop/state/` only has aggregate err).
- **Fix sketch**: Either (a) re-derive ILC sections from curvature on the v=8 trajectory (drop the override block from race_01.json), or (b) make `ilc_section_overrides` time-based not step-based so the boundaries scale with `total_time`.
- **Confidence**: medium.

### F15. Validator runs every bench call (≈ 60 ms on a 14 s trajectory at dt=0.01); zero memoisation — [NIT]
- **File(s)**: `scripts/benchmark.py:355-356`
- **Issue**: `validate_trajectory` runs on the SAME trajectory + gates every time the bench is called for a given track. Multi-track matrix runs validate each track once (acceptable), but any caller that re-runs `run_synthetic_benchmark` repeatedly (e.g. ILC sweeps, gain tuning) eats a fresh validator pass each time. Cheap individually, but the seam invites repeat-calls. Memoise on `(id(trajectory), id(gate_specs), dt)`.
- **Repro**: Time a 10-call loop of `run_synthetic_benchmark` with identical args — validator runs 10×.
- **Fix sketch**: `@functools.lru_cache` won't work directly (unhashable args) but a manual cache keyed on object identities is trivial. Or just skip the validator when the trajectory hasn't changed (planner output is deterministic given config).
- **Confidence**: medium — optimisation, not correctness.

### F16. `iter-005` ships zero new tests for the velocity-config change — [NIT]
- **File(s)**: `git show 7530f1f --stat` — `scripts/benchmark.py | 27 ++-`, `slalom.json | 2 +`. No `tests/` modifications.
- **Issue**: The change adds two new config-driven paths (`max_velocity_mps`, `tracker_overrides`) and a new function parameter (`tracker_config_overrides`) — none have a unit test. The single regression line is the matrix integration test; if anyone breaks the JSON-key plumbing (e.g. renames the key, mistyped read, or wires it to the wrong call-site), no fast test catches it. Compounds F4 — the un-plumbed PyBullet / RacePipeline paths could have been caught by a `tests/test_velocity_plumbing.py` that imports each entry point and verifies the read.
- **Repro**: `git show 7530f1f --stat`.
- **Fix sketch**: Add a tiny test that loads slalom.json, runs `run_synthetic_benchmark(config=data, duration=0.1)` (just enough to construct the trajectory), and asserts `traj_opt.constraints.max_velocity == 6.0`. Mirror for `tracker_overrides`.
- **Confidence**: high — the empirical test count is in the commit stat.

### F17. Validator's `dq_reason` says "out_of_order:gate-5" but runtime termination is "crash_gate:gate-5" — disagreement in failure-mode label on figure8 — [NIT]
- **File(s)**: `planning/plan_validator.py:126-127`, `scripts/benchmark.py:643-652`
- **Issue**: On figure8 (`python3 -m scripts.benchmark_matrix --configs figure8`), `plan_validation.reason = "DQ at t=0.67s: out_of_order:gate-5"` but `termination_reason = "crash_gate:gate-5"` and `crashed=true, disqualified=false`. The validator says DQ (perfect-tracking crosses gate-5's strict opening), but the bench's drone, due to overshoot, hits gate-5's crash annulus instead. Both "fail" but with different failure-mode labels. A future iteration trying to "fix the figure8 DQ" via planner changes might be confused that the bench's actual error is a CRASH not a DQ. This isn't dangerous (both are terminal), but it muddies the planner-vs-tracker triage that the validator was supposed to clean up.
- **Repro**: Above command's JSON has both fields with mismatched labels.
- **Fix sketch**: Document in the validator's docstring that its label is the failure mode IF the tracker is perfect; runtime label may differ due to overshoot. Or have the validator report BOTH a DQ-zone hit and a crash-zone hit (`would_dq` and `would_crash` flags) so the triage is fully transparent.
- **Confidence**: high.

## Things iter-004/005 got right

- **Plan validator is the right structural addition** — `plan_ok` ≠ `sim_ok` is the single most useful diagnostic field added in 5 iterations, and it correctly fingers figure8 as a planner problem (not a tracker problem) even at v=6.
- **Multi-track matrix is the right honesty surface** — 6/7 is a real win, and the matrix output now lets a reviewer see at a glance which tracks regressed where (composer-25's F2 reaches a slightly different root-cause attribution, but the matrix surface is what made the diagnosis possible).
- **`enforce_in_order=True` was NOT weakened** to make tracks pass — iter-005 strictly improved the planner+tracker pipeline rather than softening the honesty contract. That's the correct prioritisation.
- **Six validator tests + matrix integration** cover the 80 % of bug shapes you'd plausibly hit (DQ, crash, empty, no-gates, result-shape, clean run). Composer F11 / mine F1 expose the long-tail false-negatives but the easy ones are nailed.
- **race_01 still passes** all hard thresholds — even with the aspirational regression I flag in F2, the bench's pass bar is preserved, so no rollback urgency. iter-005 is a directionally correct net positive; my findings are about not declaring victory.

## What I did NOT review

- **`run_sim_benchmark` (PyBullet path) actual execution** — I read the source but didn't run PyBullet headless this session; F4's claim about the PyBullet velocity drift is from code inspection only (consistent with gpt-55-xhigh's F1 finding).
- **`RacePipeline` MAVLink adapter behaviour** — only inspected `PipelineConfig.max_speed` and the trajectory optimizer instantiation; didn't trace the full command path to the MAVSDK bridge.
- **Visual demo (`scripts/visual_demo.py`)** — not in iter-004/005's diff, didn't open.
- **The `planning/racing_line_cache.json` git-status modification** — flagged by git status as M but not in either of the two reviewed commits. Composer-25 also called this out; I didn't open the file.
- **The full ILC test suite** (`tests/test_ilc*.py` if any) — only inspected `tests/test_plan_validator.py`. Composer F9 / mine F14 raise the v=8 ILC alignment concern without running the per-gate diff.
- **`competition/aigp_geometry.py` and SITL calibration stub** — out of iter-004/005 scope but interact with the velocity story (kinematic envelope vs real-drone envelope); did not re-audit.
- **gpt-55-xhigh F1 and composer-25 F1/F2/F3 overlap** — I've explicitly cross-referenced where I'm restating with a new angle vs adding fresh material; the synthesiser should not double-count F7/F8/F9 against the prior reviewers' findings.
