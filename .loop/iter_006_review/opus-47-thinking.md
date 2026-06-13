# Iter 006 Adversarial Review — Opus 4.7 Max-Thinking

## Summary

Iter-006 is directionally correct (the 8.0 / per-track 6.0 globals were
charter violations and they are gone), but four things are off. (1) The
shipping narrative is wrong: race_01 auto-derives to **5.55 m/s**, not
the brief's claimed ~10.8 m/s, because the binding triplet is the inner
hairpin gates (gate-5,6,7) not the gate-1,2,3 entry — and the iter-006
commit message + `auto_velocity.py` module header both repeat the wrong
claim. (2) The `chord = min(|AB|, |BC|)` step in the formula is a real
quantitative error on asymmetric triplets — on race_01's actual binding
triplet it underestimates the circumradius by **1.44×** (3.21 m vs the
through-3-points 4.63 m), which alone explains ~1.2 m/s of the race-time
loss. (3) iter-006 inherits **all** of iter-005's velocity-plumbing
problems untouched: the synthetic bench uses `derive_safe_max_velocity`,
the PyBullet bench falls through to `PlannerConfig.plan_max_speed_mps =
4.0` (because `RaceConfig` has no `max_velocity_mps` field at all, so
the `getattr` is dead code), and `RacePipeline` defaults to 8.0 — three
independent speed sources, the matrix sees only one. (4) iter-005 F6's
dead code (`RaceState` import, `extras` field, `dir()` guard) was *not*
touched by iter-006b even though that PR's whole subject was the
validator. The figure8 1/8 failure is the only thing both commits
correctly punt to a later iter. Live `benchmark_matrix` run reproduces
the brief's 12/12 race_01 PASS at **18.08 s** (vs 13.7 s baseline =
**32 % regression**, well above the CLAUDE.md `< 14 s` aspirational
ceiling that iter-005 already breached).

## Findings (ordered by severity)

### F1. The commit narrative + module docstring claim race_01 derives ~10.8 m/s; actual is **5.55 m/s** — [MAJOR]
- **File(s)**: `planning/auto_velocity.py:14-19`, commit `06ba521` message lines 26-31, `.loop/specs/8_review_brief_iter_006.md:42`
- **Issue**: `planning/auto_velocity.py:16-18` claims "race_01 (most gates β<30°, |AB|≈8-10m → r≈30m → v_max≈15m/s) gets the full default." The iter-006 brief restates this as "~10.8 m/s." Both are **false on the actual `sim_pybullet/configs/race_01.json` geometry**. The binding triplet is gate-5 → gate-6 → gate-7 (β ≈ 93.9°, |AB|=8.28 m, |BC|=4.69 m) with `r_formula = 3.21 m → v = 5.55 m/s`, and the next-tightest triplets (gate-6,7,8 and gate-9,10,11) sit at ~5.6–5.8 m/s. The narrative implies "race_01 sails through at the cap"; the implementation drops it to 37 % of the cap. This isn't just a documentation issue: the entire iter-006 review brief was written under the assumption race_01 was running at ~10.8 m/s and asked "what else is throttling?". The answer is: nothing, the formula itself is throttling and the comment is wrong.
- **Repro**:
  ```python
  import json
  from gate_sequencing.sequencer import GateSpec
  from planning.auto_velocity import derive_safe_max_velocity
  data = json.load(open('sim_pybullet/configs/race_01.json'))
  specs = [GateSpec(gate_id=g['id'],
                    position=(g['pose']['x'], g['pose']['y'], g['pose']['z']),
                    yaw=g['pose'].get('yaw', 0),
                    sequence_index=g.get('sequence_index', 0))
           for g in data['gates']]
  print(derive_safe_max_velocity(specs))  # → 5.5509...
  ```
- **Fix sketch**: (a) replace the misleading `planning/auto_velocity.py:14-19` example with the actual numbers per track (one line per shipping track, computed at test time, not asserted by hand); (b) emit the resolved `max_velocity_mps_used` into the bench result dict so future reviewers don't have to recompute; (c) add `tests/test_auto_velocity.py::test_race_01_derives_expected_velocity` that loads `race_01.json` and asserts the derived value to lock in the actual behaviour.
- **Confidence**: high — fully reproduced; composer-25 F3 reaches the same conclusion independently.

### F2. The `chord = min(|AB|, |BC|)` step in `derive_safe_max_velocity` is **geometrically wrong** for asymmetric triplets — undercosts by up to 3.6× — [MAJOR]
- **File(s)**: `planning/auto_velocity.py:113-114`, `tests/test_auto_velocity.py:42-57`
- **Issue**: The formula `r = chord / (2·sin(β/2))` (line 114) is the *inscribed-arc* radius for an **isoceles** triplet (|AB| = |BC| = chord). For asymmetric triplets the unique circle through A, B, C has radius `r_circ = (|AB|·|BC|·|AC|) / (4·area(△ABC))`, not `min(|AB|, |BC|) / (2·sin(β/2))`. Worked examples:
  - `|AB|=2`, `|BC|=10`, β=90° at B → `r_circ = 5.10 m`, `r_formula = 1.41 m`. Ratio **3.61×**. v_formula = 3.68 m/s, v_actual = 7.00 m/s.
  - **race_01 binding triplet** (gate-5,6,7): `|AB|=8.28`, `|BC|=4.69`, β=93.9° → `r_circ = 4.63 m`, `r_formula = 3.21 m`. Ratio **1.44×**. v_formula = 5.55 m/s, v_actual_through_3_points = 6.67 m/s.
  Using `min` is documented as "conservative" (it never overestimates safe v), but it is **not** the safe-v formula — it's a hand-wave that happens to be safe and happens to bind tighter when the triplet is asymmetric. The 5.55 m/s race_01 cap is ~17 % below the geometric truth for that triplet, which is a measurable fraction of the 13.7 → 18.1 s race-time regression (linearly extrapolating, v=6.67 m/s gives a race time of ~15 s, still over aspirational but not the disaster 18.08 s is).
  The existing tests don't catch this because every test uses a **symmetric** triplet: `test_tight_90_turn_three_gates_at_3m_spacing` uses |AB|=|BC|=3 m (line 47-51), `test_returns_min_over_all_triplets` puts the tight bend at |AB|=|BC|=10/3 m (line 78-82). There is no test with the |AB|=2, |BC|=10 asymmetric case.
- **Repro**: See the `r_circ` computation above. Reproducible to 0.01 m with `(a*b*c)/(4*area)`.
- **Fix sketch**: Replace lines 113-114 with the through-3-points formula:
  ```python
  ac = math.sqrt(sum((a[k]-c[k])**2 for k in range(3)))
  s = 0.5 * (len_ab + len_bc + ac)
  area = math.sqrt(max(s*(s-len_ab)*(s-len_bc)*(s-ac), 0.0))
  if area < 1e-9:
      continue  # collinear — covered by _MIN_BEND_RAD already
  r = (len_ab * len_bc * ac) / (4.0 * area)
  ```
  Add `test_asymmetric_triplet_uses_circumradius_not_min_chord` with `|AB|=2, |BC|=10, β=90°` expecting `v ≈ 7.0 m/s` (not the current 3.68). Predicted matrix outcome: race_01 race-time drops to ~15 s with no other change, slalom/grand_tour unchanged or slightly higher v_max.
- **Confidence**: high — straight geometry. Composer F6 flagged the same site without the quantitative miss; this restates with numbers.

### F3. `RaceConfig` has no `max_velocity_mps` field, so the PyBullet `getattr` is dead code — iter-005b's "Same value applies on both bench paths now" claim is **false** — [BLOCKER for honesty contract]
- **File(s)**: `scripts/benchmark.py:800-810`, `sim_pybullet/env.py:19-48`, `sim_pybullet/env.py:217-228`
- **Issue**: `scripts/benchmark.py:806-808` reads
  ```python
  pybullet_max_v = float(
      getattr(race_config, "max_velocity_mps", None)
      or planner_cfg.plan_max_speed_mps
  )
  ```
  with the comment at lines 800-805 declaring "Same value applies on both bench paths now." But `RaceConfig` (`sim_pybullet/env.py:19-38`) has **no `max_velocity_mps` field**. `load_config` (`sim_pybullet/env.py:217-228`) never assigns one. So the `getattr` is permanently `None` and the PyBullet path **always** falls through to `planner_cfg.plan_max_speed_mps` — for race_01 that is 4.0 (from `race_01.json:56`), for every other track it is the `PlannerConfig` default 4.0 (`trajectory_optimizer.py:603`). The comment is documenting a fix that wasn't actually implemented. iter-005 F4 BLOCKER, restated in iter-005 gpt-55 F1, was supposedly addressed by commit `2121298` (iter-005b); the code added the `getattr` but never the field it was reading. Net effect now in iter-006:
  - Synthetic bench (matrix): race_01 ≈ 5.55 m/s (auto-derived), slalom ≈ 6.06, grand_tour ≈ 5.93
  - PyBullet bench: race_01 = 4.0 (from JSON), slalom = 4.0 (PlannerConfig default), grand_tour = 4.0
  - `RacePipeline` (MAVLink / competition): 8.0 (`race_pipeline.py:95`)
  The matrix's "6/7 PASS" verdict therefore does not predict what the PyBullet bench would say, and the competition path runs an entirely third trajectory. This is exactly the platform-honesty drift the iter-001/002 work was meant to prevent — and it is the same finding `gpt-55-xhigh.md` F1 and `composer-25.md` F1 surfaced in this review round, which means **three independent reviewers across two iterations have flagged the same plumbing gap and it has not been closed**.
- **Repro**:
  ```bash
  rg -n "max_velocity_mps" sim_pybullet/env.py            # zero hits
  rg -n "max_velocity_mps" sim_pybullet/configs/*.json    # only the stale slalom _about comment
  ```
- **Fix sketch**: Composer F1 / GPT-55 F1 each propose the right shape (`select_plan_max_velocity(...)` helper called from synthetic, PyBullet, demo, and RacePipeline). I want to add a **plumbing test** that `tests/test_velocity_plumbing.py` loads each track JSON, instantiates `run_synthetic_benchmark` setup AND `run_sim_benchmark` setup AND `RacePipeline(config_from_json)` setup, and asserts `DroneConstraints.max_velocity` is identical for all three paths. Without that test, iter-007 can repeat the same drift.
- **Confidence**: high — the missing field is grep-verifiable; the consequence is run-verifiable.

### F4. iter-006b's validator commit subject was the validator, but iter-005 F6's dead-code findings (`RaceState`, `extras`, `samples_evaluated` `dir()` guard) are **all still present** — [MINOR — but it means the iter-006 patch was narrower than declared]
- **File(s)**: `planning/plan_validator.py:31`, `planning/plan_validator.py:50`, `planning/plan_validator.py:188`
- **Issue**: iter-005 review F6 (Opus) was a 3-part dead-code finding that landed in the iter-005 synthesis but is unchanged by iter-006b (the validator's own commit). Specifically:
  - **L31**: `from gate_sequencing.sequencer import (GateSequencer, GateSpec, RaceState, SequencerConfig)` — `RaceState` is still never referenced anywhere in the module. Composer F7 in iter-005 proposed wiring it (`reason = "stuck_in_recovery"` if `seq._state == RaceState.RECOVERY`); neither path was taken.
  - **L50**: `extras: dict = field(default_factory=dict)` is declared on `ValidationResult` but never populated by `validate_trajectory` and never surfaced by the bench at `benchmark.py:646-655`. Dead field.
  - **L188**: `samples_evaluated=step + 1 if "step" in dir() else samples` — the `else samples` branch is unreachable (`total_time > 0` guarantees `samples ≥ 1`, so the `for step in range(samples)` loop body runs at least once and `step` is defined). The expression also relies on Python's loop-variable leak, which is exactly the kind of fragile pattern a future `else: break` refactor would silently break.
  The deeper concern: iter-006b's commit message subject is "F5 + F11" — i.e., it claims to be a validator + matrix improvement. The validator file was touched (airspace bounds added). But the simultaneous cleanup of the **same file's** existing review-flagged dead code was skipped. Reviewer-flagged dead code that survives the very commit that touches the file is process drift.
- **Repro**: `grep -n "RaceState\|extras\|\"step\" in dir" planning/plan_validator.py` returns lines 31, 50, 188.
- **Fix sketch**: Drop `RaceState` from the import (or use it for an actual `reason` branch), delete the `extras` field (or document and populate it), and replace `step + 1 if "step" in dir() else samples` with explicit `step` initialisation:
  ```python
  step = -1
  for step in range(samples):
      ...
  return ValidationResult(..., samples_evaluated=step + 1)
  ```
- **Confidence**: high — direct code inspection.

### F5. `tests/test_plan_validator.py::test_trajectory_within_airspace_no_flag` is now **vacuous** — premise broken by the z=2 flip — [MINOR]
- **File(s)**: `tests/test_plan_validator.py:241-256`, `tests/test_plan_validator.py:59-67`
- **Issue**: The test's stated premise is "gates here are at z=-2 from `_make_line_gates` so the path doesn't pass them. Result will be incomplete but NOT airspace" (lines 251-252). But iter-006b flipped `_make_line_gates` to use `z=2.0` (line 63). With `_make_line_gates(2)` returning gates at (5,0,2) and (10,0,2) and the trajectory at z=2.0 from x=0 to x=10, the trajectory now passes both gates cleanly. `result.ok` becomes `True`, so the inner block:
  ```python
  if not result.ok:
      assert "ground" not in result.reason.lower()
      assert "ceiling" not in result.reason.lower()
  ```
  **never executes**. The test passes vacuously and does not exercise the airspace-no-flag branch any more.
- **Repro**: Add a `print(result)` before the `if`: prints `ok=True, reason='trajectory passes all 2 gates cleanly'`. The two `assert` lines are unreachable.
- **Fix sketch**: Either (a) restore the original "gates intentionally don't intersect" by moving the gates back to z=-2 *and* explicitly using the NED-only path (which becomes the validator's z-down test case — see F6 below), or (b) keep z=2 but make the trajectory deliberately miss the gates (e.g., y=10 throughout) so `result.ok=False, reason="incomplete"` and the `not "ground"/not "ceiling"` checks actually fire.
- **Confidence**: high — direct test execution.

### F6. Validator airspace defaults silently assume z-up convention; `gate_sequencing/tests/test_sequencer_adversarial.py` runs on `z=-2.0` (NED) — the codebase invariant per AGENTS.md is "NED internally" — [MAJOR — competition-path risk]
- **File(s)**: `planning/plan_validator.py:59-60`, `planning/plan_validator.py:132-139`, `gate_sequencing/tests/test_sequencer_adversarial.py:35-41`, `AGENTS.md` ("All pipeline modules use NED internally"), `CLAUDE.md:106-108`
- **Issue**: The validator's signature defaults `ground_z_threshold=0.05`, `ceiling_z_threshold=20.0` (lines 59-60) and the synthetic bench calls `validate_trajectory(trajectory, gate_specs, dt=dt)` without overriding either (`benchmark.py:359`). This is internally consistent for the bench (which also hardcodes `pos[2] < 0.05`, `pos[2] > 20.0` at lines 509-515) — both are **z-up** with ground at 0. But:
  - `gate_sequencing/tests/test_sequencer_adversarial.py` has 35 test calls with `z=-2.0` and a module-level comment "Build a straight line of n gates along +X (NED), facing +X" (line 36). The sequencer is correctly *plane-only* so the z value is benign for **its** logic — but if any future caller feeds a NED-faithful plan into `validate_trajectory` (gates and trajectory at z=-2), the validator's default `ground_z_threshold=0.05` fires `crash_ground` on the first sample.
  - `AGENTS.md` declares "Competition/MAVLink: NED" and "All pipeline modules use NED internally. The adapter layer handles conversion." The validator's default is therefore **violating the documented convention** — if iter-007's MAVLink integration runs the validator on the in-pipeline NED state, every trajectory will appear to crash into the ground at t=0.
  - `CLAUDE.md` and the bench treat z=positive as altitude (the `vertical_cliff` track config has `z=10.5` for its top gate; the bench's `crash_ceiling=20.0` makes sense only in z-up). So the bench is z-up; AGENTS.md says the pipeline is NED. **The codebase has two competing conventions and the validator hardcodes one of them in its defaults.**
  iter-005 F5 (Opus) and the iter-006 brief both proposed making the threshold derive from `data["field"]["bounds_min/max"][2]`. iter-006b did not take that proposal — it just baked the bench's z-up numbers into the signature defaults. Composer F8 and GPT-55 F2 in this round also flag the convention drift.
- **Repro**:
  ```bash
  rg -n "z.*=.*-2|position=\(.+,.+,.-?2\.0" gate_sequencing/tests/test_sequencer_adversarial.py | wc -l
  # → 35 hits — all z=-2.0 (NED convention)
  ```
  And manually: `validate_trajectory(traj_with_z=-2, gates_at_z=-2)` returns `ok=False, reason="airspace exit at t=0.00s: crash_ground"`.
- **Fix sketch**: (a) Move `ground_z_threshold`/`ceiling_z_threshold` to **required** kwargs with no defaults — the bench must supply them. (b) Have `validate_trajectory` accept a `field_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]` and derive the z range from that. (c) Either flip `gate_sequencing/tests/test_sequencer_adversarial.py` to z=2.0 (matching the bench convention) OR keep both and add a `_normalize_z` shim with an explicit convention enum in the adapter layer. (d) Add a regression test that feeds NED-style positions through the validator with appropriate overrides and asserts the result.
- **Confidence**: high — the convention divergence is grep-verifiable; the latent failure is one MAVLink integration away.

### F7. `slalom.json` "no longer has a magic number" but the derived value is **6.06 m/s** — essentially the iter-005 6.0 reused via different math; the cleanup is celebrating without measuring — [MINOR]
- **File(s)**: `sim_pybullet/configs/slalom.json:1-2`, commit `06ba521` message lines 33-35
- **Issue**: The slalom JSON comment now reads "iter-006 replaced the explicit `max_velocity_mps=6.0` override with derive_safe_max_velocity, which computes the centripetal-acceleration-limited velocity from gate spacing + bend angle (no per-track magic numbers)." The commit claims this as a charter-compliance win. But `derive_safe_max_velocity` returns **6.06 m/s** for slalom — within 1 % of the value that was previously the magic literal. The cleanup is not "we proved the right value is geometry-derived" but rather "we found a derivation that happens to land on the value we already had hand-picked." That's still net-positive (the formula will generalise to a new course), but the commit narrative implies validation that the formula is the **right** one when it merely matches the prior hand-tune.
  This isn't a bug, it's a calibration concern. Combined with F2 (the formula systematically undercosts asymmetric triplets, of which slalom has many), the matching value is likely a coincidence — the formula could be off in different directions on different tracks and still produce numbers near the old magic literals on a sample of two.
- **Repro**: `derive_safe_max_velocity` on slalom.json gates → 6.06. Pre-iter-006 slalom override → 6.0.
- **Fix sketch**: When proposing a formula as a charter-compliance fix, document the sensitivity sweep (e.g., what does the formula give for a 5-track battery of synthetic stress courses vs. the hand-tuned values?) and add a per-track regression test that asserts the derived value is within ±X% of the hand-tuned baseline so a future formula change doesn't silently move slalom from 6.06 m/s to 4.0 m/s while race_01 stays at 5.55.
- **Confidence**: medium — the slalom match-up is empirical; the broader "formula needs calibration not just adoption" point is a methodology critique.

### F8. `auto_velocity.py` provides no integration test on real track configs; everything is synthetic stub triplets — [MINOR]
- **File(s)**: `tests/test_auto_velocity.py:1-128`, `sim_pybullet/configs/*.json`
- **Issue**: All 9 tests in `tests/test_auto_velocity.py` use the `_StubGate` dataclass with hand-constructed (3,4)-point triplets. None of the tests load a real `sim_pybullet/configs/*.json` track and assert the derived velocity. As a result:
  - F1 (the wrong narrative about race_01 ≈ 10.8 m/s) survived because no test verified the actual value.
  - F2 (the asymmetric-triplet undercost) is invisible because every test uses symmetric triplets.
  - A future change to gate centers (e.g., a JSON edit that adds a 13th gate to race_01) would silently change the binding triplet and the derived velocity, with no test signal.
  The unit-style tests are good for the formula's edge cases; they don't cover its integration against the shipping geometry.
- **Repro**: `wc -l tests/test_auto_velocity.py` = 128; `rg -n "json.load\|sim_pybullet/configs" tests/test_auto_velocity.py` = 0.
- **Fix sketch**: Add `test_each_track_derives_documented_velocity` that loops over `sim_pybullet/configs/*.json`, skips placeholders, computes `derive_safe_max_velocity` on each track's gates, and asserts against a baseline JSON committed alongside (`tests/data/auto_velocity_baselines.json`). The baseline value plus a note on the binding triplet localises the source of any future regression.
- **Confidence**: high — test surface is grep-verifiable.

### F9. `planning/racing_line.py` still has four `15.0` literals; auto-derive only flows into `DroneConstraints.max_velocity`, the racing-line optimizer still scores lines as if v=15 — [MAJOR — iter-005 F3 unfixed]
- **File(s)**: `planning/racing_line.py:362`, `planning/racing_line.py:515`, `planning/racing_line.py:516`, `planning/racing_line.py:675`, `scripts/benchmark.py:334-351`
- **Issue**: The synthetic bench computes `max_velocity = derive_safe_max_velocity(gate_specs)` and passes it to `DroneConstraints(max_velocity=max_velocity)` for the **trajectory optimizer** (line 348-350). But the **racing line optimizer** (`rl_opt = RacingLineOptimizer(); opt_wps = rl_opt.optimize(gate_waypoints, tuple(start_pos))` at line 334-335) runs BEFORE that and uses its own `15.0` constants internally:
  - `planning/racing_line.py:362`: `constraints=DroneConstraints(max_velocity=15.0)` — used inside `_evaluate_with_trajectory` to score candidate racing lines.
  - `planning/racing_line.py:515-516`: `max_accel = 15.0`, `max_speed = 15.0` — used by `SpeedProfiler.profile` for the backward sweep.
  - `planning/racing_line.py:675`: `max_speed: float = 15.0` — default arg on a public method.
  So for race_01 the racing line is picked **as if** the drone will fly at v=15 m/s through the gate-5,6,7 hairpin, then the trajectory optimizer is told to actually cap at v=5.55 m/s. The chosen line is optimized for the wrong physics. iter-005 review F3 (Opus) flagged six 15.0 sites; iter-006 still leaves four of them. The "auto-derive everywhere" charter goal needs a single-source-of-truth, which this isn't.
- **Repro**: `rg -n "15\.0" planning/racing_line.py | grep -v "comment\|#"` — 4 active uses.
- **Fix sketch**: Thread the same `max_velocity` value (resolved by the proposed `select_plan_max_velocity` helper from F3) into `RacingLineOptimizer`'s constructor and `SpeedProfiler.profile`. Single value, single source. Today the racing line and the trajectory optimizer disagree by 2.7× on race_01.
- **Confidence**: high — direct grep confirms; the consequence (racing line vs trajectory disagreement) is geometric.

### F10. ILC step-indexed section boundaries are now **doubly** misaligned for race_01: tuned at v=15 (steps 200/440/740 of 1370), now running with steps 200/440/740 of **1808** — [MAJOR — composer-25 iter-005 F9 / Opus iter-005 F14 are getting worse, not better]
- **File(s)**: `sim_pybullet/configs/race_01.json:21-26` (ilc_section_overrides), `scripts/benchmark.py:377-383`, `planning/ilc_sections.py` (the curvature-derive path that race_01 bypasses)
- **Issue**: race_01.json carries hand-tuned `ilc_section_overrides=[[0, 200], [200, 440], [440, 740], [740, 99999]]` — step indices in 0.01 s steps, originally tuned when race_01 ran in **1370 steps** (13.7 s at v=15 m/s baseline). At iter-005's v=8 m/s the total was 1492 steps (14.92 s); at iter-006's v=5.55 m/s the live `benchmark_matrix` run shows race_01 at **1808 steps** (18.08 s). The hardcoded boundaries 200/440/740 now land at:
  - v=15 (baseline): 200/1370 = 14.6 %, 440/1370 = 32.1 %, 740/1370 = 54.0 %
  - v=8 (iter-005): 200/1492 = 13.4 %, 440/1492 = 29.5 %, 740/1492 = 49.6 %
  - v=5.55 (iter-006): 200/1808 = 11.1 %, 440/1808 = 24.3 %, 740/1808 = 40.9 %
  So the boundaries that used to mark the four hand-tuned phases of the helix (approx entry / inflection / interior / exit) now fall ~25 % earlier in geometric phase than at v=15. The ILC corrections tuned for "interior helix" (section 3) are being applied to "inflection" geometry, etc. The average tracking error in the live run (`avg_tracking_error_m: 0.106` for race_01) is still within the THRESHOLDS bar, so this is a latent drift — but iter-005 F14 already flagged it and iter-006's velocity drop made the misalignment worse without re-tuning. There is no test asserting the section boundaries are in the right geometric phase.
- **Repro**: Live `benchmark_matrix --configs race_01` → `sim_time_s: 18.08, total_steps in synthetic ≈ 1808`. Race_01 JSON sets the 200/440/740 boundaries as step indices.
- **Fix sketch**: (a) Make `ilc_section_overrides` time-based (seconds), not step-based, and scale to `total_time`. (b) Or drop the manual overrides and use `derive_section_boundaries` (curvature-based) on the new v=5.55 trajectory. (c) Add a smoke test that asserts the four section boundaries fall within ±10 % of the ratios they had at v=15 baseline (or, better, that they correspond to curvature inflection points).
- **Confidence**: medium-high — the boundary drift is arithmetic; whether it actually hurts per-gate error needs the per-gate measurements iter-005 F14 also wanted.

### F11. The matrix's `pv.get("ok") is False` check leaks on **missing field** AND on **placeholder track** AND on **non-bool ok values** — three holes, not just the two prior reviewers found — [MAJOR]
- **File(s)**: `scripts/benchmark_matrix.py:99`, `scripts/benchmark_matrix.py:107-128`, `scripts/benchmark_matrix.py:134-139`
- **Issue**: Both composer-25 F10 and gpt-55-xhigh F3 catch the missing-field hole (`pv = ... or {}`; `pv.get("ok") is False` treats `None` / missing as non-regression). I add two more:
  - **Placeholder tracks completely bypass the check** (line 107 vs line 113 branch split). `aigp_default.json` has `placeholder: true`. The matrix's only check for placeholders is `gate_pass_rate < 0.50` — a placeholder with `plan_validation.ok=False` and 60 % gates would silently PASS. Since `aigp_default.json` is **explicitly the placeholder for the real AIGP VQ1 course** the brief is targeting, this is the worst possible track to skip the plan check on. The fact that the live run shows `aigp_default: pv_ok=True` is a coincidence of current geometry, not a guarantee.
  - **`is False` is identity-strict**. Currently `plan_validation.ok` is always a Python `bool`, so `is False` matches. But if a future serialiser stage round-trips through JSON.parse and back, `False` could become `0` or `"false"` or `numpy.bool_(False)`. The matrix would silently start passing. Should be `pv.get("ok") is not True` so anything that isn't truthy `True` fails (including `None`, `0`, `"false"`, missing field).
- **Repro**: Drop a synthetic test case where `track_summary["plan_validation"]` is `{"ok": numpy.bool_(False), "reason": "DQ"}` — `is False` returns `False` and the regression is missed. Or: mark race_01 as `placeholder: true`, re-run matrix; race_01 would PASS at 12/12 even if its plan validator said `ok=False`.
- **Fix sketch**: (a) Apply the plan-validation check **before** the placeholder branching so both branches enforce it. (b) Change the comparison to `pv.get("ok") is not True` (or `bool(pv.get("ok")) == False` with an explicit `if "ok" not in pv: append("missing plan_validation")` first). (c) Add a `test_matrix_acceptance` suite that constructs synthetic `track_summary` dicts with each leak case and asserts the regression list is non-empty.
- **Confidence**: high — control-flow inspection.

### F12. The PyBullet path has a **ground crash check** (`pos[2] < 0.05`) but **no ceiling crash check** (no `pos[2] > 20.0`) — validator + synthetic bench check both, PyBullet checks one — [MINOR — platform drift]
- **File(s)**: `scripts/benchmark.py:507-515` (synthetic, both checks), `scripts/benchmark.py:883-886` (PyBullet, ground only), `planning/plan_validator.py:130-139` (validator, both)
- **Issue**: The iter-006b commit synced the validator's airspace checks with the **synthetic** bench at scripts/benchmark.py:507-515 (both ground and ceiling). But the PyBullet path at scripts/benchmark.py:883-886 only has the ground check; no ceiling check exists in `run_sim_benchmark`. Net effect: a trajectory that the validator flags as `crash_ceiling` would be flagged by the synthetic bench's `pos[2] > 20.0` check **but** silently survive in PyBullet until something else terminates the run (or the timeout). The three honesty surfaces (validator, synthetic, PyBullet) should agree.
- **Repro**: `rg -n "pos\[2\] >|crash_ceiling" scripts/benchmark.py` → synthetic (line 513), no PyBullet hit.
- **Fix sketch**: Add the symmetric `if pos[2] > 20.0: crashed = True; termination_reason = "crash_ceiling"; break` after line 886 in the PyBullet loop. Better, factor the airspace check into a `_check_airspace(pos, ground_z, ceiling_z) -> Optional[str]` helper used by both bench paths and the validator. Tie all three to the same threshold source (proposed `field.bounds_min/max[2]` from track JSON, per F6 and gpt-55-xhigh F2).
- **Confidence**: high — direct grep.

### F13. `figure8` deferral is sound; nothing in iter-006 actually addresses or even moves the needle on it — [NIT — diagnostic clarity]
- **File(s)**: `sim_pybullet/configs/figure8.json:16-23`, `sim_pybullet/configs/figure8.json:60-65`, `.loop/iter_005_review/opus-47-thinking.md:126-134`
- **Issue**: Iter-006 brief documents figure8 as "1/8 FAIL (coplanar-gate edge case, deferred)" and the commit messages confirm no figure8-specific code was touched. The live matrix run reproduces 1/8 with `plan_validation.ok=False — DQ at t=0.75s: out_of_order:gate-5` and runtime `crash_gate:gate-5`. iter-005 F12 already worked the geometry — there's a 20 mm legal z-strip between gate-1's opening top edge (z = 1.4 to 2.6, gate-1 at z=2.0) and gate-5's outer frame bottom (z ≥ 1.42, gate-5 at z=2.2). The iter-006 plan-validator + matrix-strict gating now correctly **reports** the failure even when the sim happens to also crash (the F11 finding is what makes this useful — the matrix catches a plan-broken figure8 even if the sim weren't crashing). So iter-006's contribution to figure8 is purely diagnostic, not corrective. That's fine to defer, but the loop should note that the figure8 coplanar problem is **not** going to be solved by tighter velocity capping — it needs SFC corridor work or TOGT gate-polygon traversal (per the iter-003 research swarm consensus). Velocity capping alone, even at 0 m/s, would still produce a trajectory that crosses gate-5's plane before gate-1 is credited.
- **Repro**: Live `benchmark_matrix --configs figure8 --json-only` → `plan_validation.reason: "DQ at t=0.75s: out_of_order:gate-5"`, `termination_reason: "crash_gate:gate-5"`.
- **Fix sketch**: Add a one-line note in `figure8.json` (alongside the existing color metadata) flagging the known coplanar issue and the planned remediation path (SFC corridor / iter-007+). Otherwise an iter-007 implementer might guess that "make v smaller" is the right move (it isn't — the validator's perfect-tracking replay also DQ's, so the bare polynomial trajectory is wrong, not the tracker).
- **Confidence**: high — geometry from iter-005 F12; matrix output confirms.

### F14. `derive_safe_max_velocity` ignores the racing-line lateral offset (up to ±0.6 m), so the formula is measuring the **gate-center triangle**, not the **flown arc** — [NIT — formula scope]
- **File(s)**: `planning/auto_velocity.py:90-117`, `planning/racing_line.py` (`max_lateral_offset_m`), `scripts/benchmark.py:334-351`
- **Issue**: `derive_safe_max_velocity` is called on `gate_specs` (raw gate centers) **before** the racing line optimizer runs. The racing line moves each waypoint laterally up to ~0.6 m (per the racing-line config) to smooth the bend. After the racing-line pass, the chord and bend at each gate are different from the raw gate triangle. For race_01's gate-5,6,7 binding triplet:
  - Raw gate-center triangle: r_formula = 3.21 m, r_actual_through_centers = 4.63 m.
  - With a 0.6 m outward lateral offset at gate-6 (corner-cutting), the through-3-points r becomes **larger** (the apex moves outward, so the inscribed circle's radius grows).
  Net: the velocity could safely be higher than 5.55 m/s on the **flown** arc, even if the gate-center arc is tighter. Composer F6 noted this in passing; restating with a fix sketch: the velocity derivation should run on the **post-racing-line** waypoints (or, equivalently, an iterative two-pass — derive v from gate centers, run racing line + trajectory at that v, then re-derive v from the actual flown waypoints to see if there's headroom).
- **Repro**: Compute v on `gate_specs` (5.55) vs on `opt_wps` (after RacingLineOptimizer); the latter should be higher for any track where the racing line does meaningful lateral offsetting.
- **Fix sketch**: Two-pass: (1) Initial v from gate centers (current behaviour). (2) Run racing line. (3) Re-derive v from `opt_wps` positions. (4) Run trajectory at the larger value. Add a metric in the bench output for both passes. Or, simpler: derive directly from `opt_wps` after racing line, accept that the first racing-line pass uses the v=15 default (per F9 fix) and a second pass uses the geometric v. The 2× cost is amortised by the better lap time.
- **Confidence**: medium — the direction of the change is clear; the magnitude depends on how aggressively the racing line offsets gate-6.

## Things iter-006 got right

- **Direction is correct**: the iter-005 `8.0` global + `slalom: 6.0` per-track override were charter violations, and iter-006 removes them. Even if F2 / F9 / F11 say the replacement is incomplete or sometimes geometrically miscalibrated, the principle (derive, don't hand-pick) is the right one to land on.
- **`test_auto_velocity.py` covers the formula edge cases well**: `<3 gates`, all-straight, zero-length segment, safety-factor scaling, accel-√-scaling, absolute cap respected, min-over-triplets. The 9 tests are not redundant. The miss is integration (F8), not unit coverage.
- **Validator airspace bounds match the bench**: ground/ceiling thresholds match `benchmark.py:509-515` exactly (and the defaults document the line numbers, which is a nice future-reader signal). The validator now correctly catches plans that would terminate at airspace exit — composer F11 + my iter-005 F5 were both addressed.
- **`benchmark_matrix.py` plan_validation regression check (F11) IS catching figure8** in the live run — three regressions reported: `crashed`, `gate_pass_rate < 75%`, `plan_validation.ok=False`. That's exactly the honesty surface iter-005 F11 was asking for, modulo my F11 hole-finds.
- **`derive_safe_max_velocity` is dependency-free, easy to read, and explicitly documents its assumptions**. The `_MIN_BEND_RAD`, `DEFAULT_SAFETY_FACTOR`, `DEFAULT_ABSOLUTE_CAP_MPS` constants all carry rationale comments. Even when the rationale is "matches legacy default the race_01 ILC sweep targeted" (which composer F5 fairly calls out as charter-leaky), at least the comment is **honest** about why the value is what it is — exactly the documentation hygiene the loop wants.

## What I did NOT review

- **`run_sim_benchmark` actual PyBullet execution** — I inspected the source (and ran the synthetic matrix) but did not boot PyBullet headless this session. F3 / F12 are from code inspection only; the actual PyBullet behaviour at 4.0 m/s is not directly verified by me (composer F4 verifies that the kinematic integrator's `max_speed = 15.0` is also unrelated to the trajectory cap).
- **`race_pipeline.py` MAVLink command path** — I confirmed `PipelineConfig.max_speed = 8.0` at line 95 and the `DroneConstraints(max_velocity=self.config.max_speed)` at line 256, but didn't trace through to the MAVSDK bridge. F3's claim about three independent defaults is based on these two lines.
- **`scripts/visual_demo.py`** — saw it reads `planner_cfg.plan_max_speed_mps` at line 387 and did not open further.
- **`planning/racing_line_cache.json`** — git status shows it as modified, both iter-006 commits don't touch it. Did not open; assume stale cache from a prior bench run.
- **The full `tests/test_auto_velocity.py` parameterised happy-path tests** — read all 9 but didn't re-run them. The bench/matrix integration **was** re-run live (live matrix output documented above and matches the brief's table modulo F1's race-time number).
- **`competition/aigp_geometry.py` and SITL calibration** — out of iter-006 scope; not re-audited.
- **Overlap with composer-25 and gpt-55-xhigh** — I read both reviews before finalising. F1 (race_01 derive ~5.55, not ~10.8) overlaps with composer F3 — kept here because the wider implication (auto_velocity.py module header is also wrong, not just the brief) is novel. F3 (PyBullet plumbing) overlaps with composer F1 and gpt-55 F1 — kept here because the additional detail (the `getattr` is dead code on a non-existent field) sharpens it. F11 (matrix leaks) extends composer F10 / gpt-55 F3 with the placeholder-skipping and identity-strict-`is False` angles. F2, F4, F5, F9, F10, F12 are novel relative to the other two reviewers; F6 / F7 / F8 / F13 / F14 also novel or carry distinct fix sketches. The synthesiser should not double-count F3 with composer F1 / gpt-55 F1, but should treat my F2 (asymmetric-chord quantitative miss) and F4 (dead-code regression in the validator) as fresh material.
