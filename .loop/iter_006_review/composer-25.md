# Iter 006 Adversarial Review — Composer 2.5

## Summary

Iter-006 correctly removes the iter-005 `8.0` / `6.0` globals and adds valuable validator airspace + matrix strictness, but the centripetal helper is **more conservative than the shipping narrative claims** (race_01 derives to **~5.55 m/s**, not ~10.8), the module docstring overstates race_01 headroom, and the synthetic vs PyBullet bench paths still disagree on how max speed is chosen. Race time on race_01 regressed to **18.08 s** (live matrix run), blowing past the CLAUDE.md **< 14 s** aspirational target with no compensating fix. figure8 was not touched and remains plan-broken.

## Findings (ordered by severity)

### F1. PyBullet bench never calls `derive_safe_max_velocity` — synthetic/PyBullet velocity split persists — [BLOCKER]
- **File(s)**: `scripts/benchmark.py:343-349`, `scripts/benchmark.py:806-814`
- **Issue**: `run_synthetic_benchmark` auto-derives `max_velocity` from gate triplets when `max_velocity_mps` is absent. `run_pybullet_benchmark` still uses `getattr(race_config, "max_velocity_mps", None) or planner_cfg.plan_max_speed_mps` — for race_01 that is **`plan_max_speed_mps: 4.0`** in `sim_pybullet/configs/race_01.json:56-56`, not the derived ~5.55 m/s. Any claim that “the stack” uses geometry-derived velocity is false for the PyBullet path and for `visual_demo` / competition adapters that read planner JSON.
- **Repro**: Compare `DroneConstraints(max_velocity=...)` construction in both functions for the same `race_01.json`; synthetic ≈5.55, PyBullet =4.0.
- **Fix sketch**: Factor a shared `resolve_max_velocity_mps(data, gate_specs) -> float` used by synthetic, PyBullet, and demo; log the resolved value in bench JSON (`max_velocity_mps_used`).
- **Confidence**: high

### F2. race_01 sim time 18.08 s — aspirational < 14 s target abandoned without documentation — [MAJOR]
- **File(s)**: `scripts/benchmark.py:699-700`, `CLAUDE.md:64-64`, `.loop/specs/8_review_brief_iter_006.md:18-18`
- **Issue**: Live `python3 -m scripts.benchmark_matrix --configs race_01` → `sim_time_s: 18.08`, `12/12` gates, `plan_validation.ok: true`. Baseline in `.loop/synthesis/regression_matrix_2026_05_24.md` was **13.7 s** at v=15. Iter-005 Opus F2 already flagged crossing 14 s at v=8 (14.92 s); iter-006 drove reference speed lower (~5.55 m/s derived) and **worsened** time further. Charter says tighten toward aspirational when thresholds pass; this is a deliberate retreat on the flagship track with no CLAUDE.md update or matrix race-time regression guard.
- **Repro**: `python3 -m scripts.benchmark_matrix --configs race_01 --json-only | jq '.tracks.race_01.sim_time_s'`.
- **Fix sketch**: Add per-track `sim_time_s` delta vs `regression_baseline_2026_05_24.json` in `benchmark_matrix.py`; allow race_01 explicit `max_velocity_mps` override *with* geometry-derived floor/ceiling; re-tune ILC section boundaries for the slower profile (see F9).
- **Confidence**: high

### F3. Shipping narrative (~10.8 m/s for race_01) and module docstring contradict measured derivation (~5.55 m/s) — [MAJOR]
- **File(s)**: `planning/auto_velocity.py:16-18`, `planning/auto_velocity.py:58-123`, commit `06ba521` message
- **Issue**: On actual `race_01.json` gate centers, `derive_safe_max_velocity` returns **5.551 m/s** (binding triplet **gate-5 → gate-6 → gate-7**, β≈94°, chord≈4.69 m, r≈3.21 m). The review brief’s “~10.8 m/s” and the file header claim (“most gates β<30° … v_max≈15 m/s”) are **not supported** by the implementation on real geometry. The matrix **does** use auto-derive (no top-level `max_velocity_mps` in race_01.json); the 18 s lap is consistent with ~5.5 m/s planning, not a hidden `plan_max_speed_mps` on the synthetic path.
- **Repro**: `derive_safe_max_velocity(gate_specs)` on loaded race_01 gates (see triplet gate-5/6/7 above).
- **Fix sketch**: Fix docstring/comments; emit `derived_max_velocity_mps` in bench output; consider using **racing-line waypoints** (post-offset) for triplets, not raw gate centers, if the intent is path curvature not gate-frame geometry.
- **Confidence**: high

### F4. Kinematic integrator still hard-caps `max_speed = 15.0` while trajectory caps at derived ~5.55 — iter-005 F4 unfixed — [MAJOR]
- **File(s)**: `scripts/benchmark.py:347-349`, `scripts/benchmark.py:444-445`
- **Issue**: `DroneConstraints(max_velocity=max_velocity)` follows derive (~5.55), but the inner loop still sets `max_speed = 15.0` with comment “increased to match trajectory planner ceiling.” Tracker overshoot can still produce motion faster than the reference; validator/plan replay won’t see it. This was iter-005 composer F4; iter-006 did not wire `max_speed` to the resolved planner cap.
- **Repro**: Inspect velocity norm in `controller_trace` when tracking error spikes; compare to `trajectory.sample(t).velocity` magnitude on race_01.
- **Fix sketch**: `max_speed = max_velocity` (and optionally `max_accel` from `DroneConstraints.max_acceleration`) immediately after resolving `max_velocity`.
- **Confidence**: high

### F5. `0.8` and `15.0` defaults are still charter-exposed magic numbers — [MAJOR]
- **File(s)**: `planning/auto_velocity.py:33-44`, `.loop/specs/0_charter.md:12-12`
- **Issue**: Replacing 8.0/6.0 with `DEFAULT_SAFETY_FACTOR=0.8` and `DEFAULT_ABSOLUTE_CAP_MPS=15.0` swaps hand-tuned globals for **physics-motivated but uncalibrated** globals. `0.8` is explicitly “empirically reasonable” (line 36-38) with no test tying it to tracker overshoot. `15.0` exists to preserve “legacy default that the original race_01 ILC sweep targeted” (line 41-43) — i.e. **race_01 overfitting** baked into the cap. Tracks with all bends <5° return the cap unchanged (lines 111-112, 118-119), so “straight” courses still get 15 m/s regardless of spacing.
- **Repro**: `test_three_gates_in_a_straight_line_returns_cap` passes at 15 m/s; remove `absolute_cap_mps` — tight slalom still bounded by geometry, race_01 helix returns 5.55 not 15.
- **Fix sketch**: Derive cap from `SpeedProfiler` backward pass (`planning/racing_line.py:712-714`) or `DroneConstraints`; load safety_factor from track JSON with `_derived_from`; document sensitivity sweep in tests.
- **Confidence**: high

### F6. Centripetal model uses raw gate centers and `min(|AB|,|BC|)` — can mis-estimate radius vs flown path — [MAJOR]
- **File(s)**: `planning/auto_velocity.py:96-116`, `scripts/benchmark.py:334-350`
- **Issue**: Bend radius uses **gate centroid positions** before racing-line lateral offsets. The binding race_01 triplet (gate-5/6/7) uses leg lengths 8.28 m and 4.69 m — the optimizer may fly a tighter arc than the gate triangle suggests, or wider if corner-cutting applies. `chord = min(len_ab, len_bc)` is conservative for speed **only if** the shorter leg is the binding arc chord; for asymmetric legs it can under/over-estimate vs circumradius of the three points. No test uses **3D helix** geometry (all `test_auto_velocity` cases are planar z=2).
- **Repro**: Compare `derive_safe_max_velocity(gate_specs)` vs same triplets on `opt_wps` positions after `RacingLineOptimizer.optimize`.
- **Fix sketch**: Optional second pass on optimized waypoints; add helix integration test from race_01 excerpt.
- **Confidence**: medium

### F7. `_MIN_BEND_RAD = 5°` discards measurable curvature — falls through to 15 m/s cap — [MINOR]
- **File(s)**: `planning/auto_velocity.py:49-49`, `planning/auto_velocity.py:111-112`
- **Issue**: Triplets with bend 1–5° are skipped entirely; `min_radius` stays inf and the function returns `absolute_cap_mps` (15). A long straight with a subtle 3° kink gets the same ceiling as a truly straight runway. Whether that is safe depends on segment length (not considered).
- **Repro**: Construct three gates with β=3°, spacing 20 m — returns 15.0, not √(a·r).
- **Fix sketch**: Lower threshold to 0.5° or use `max(ε, bend)`; incorporate chord length into a combined limit.
- **Confidence**: medium

### F8. Z-frame convention drift: bench/validator z-up vs sequencer tests NED z=-2 — competition risk — [MAJOR]
- **File(s)**: `planning/plan_validator.py:88-91`, `planning/plan_validator.py:132-138`, `scripts/benchmark.py:509-515`, `gate_sequencing/tests/test_sequencer_adversarial.py:35-41`, `tests/test_plan_validator.py:251-252`
- **Issue**: Validator and synthetic bench treat **z-up** with ground `z < 0.05`. Sequencer adversarial stubs still document “NED” and place gates at **`z=-2.0`**. `_make_line_gates` in `tests/test_plan_validator.py:59-67` correctly uses **`z=2.0`**, but line 251-252 still says “gates at z=-2” (stale comment). `CLAUDE.md` states pipeline modules use **NED internally**; competition configs with negative altitude would spuriously trip `crash_ground` if fed to this validator unchanged.
- **Repro**: `validate_trajectory` on a NED-faithful plan with gates at z=-2 and trajectory at z=-1.5 — fails ground check; bench would too.
- **Fix sketch**: Single `AltitudeConvention` enum in adapter layer; convert before validator/bench; align adversarial tests or mark them `z_up_only`.
- **Confidence**: high

### F9. Airspace thresholds are bench-magic, not track-configurable — [MINOR]
- **File(s)**: `planning/plan_validator.py:59-60`, `planning/plan_validator.py:88-91`, `scripts/benchmark.py:359-359`
- **Issue**: `ground_z_threshold=0.05` and `ceiling_z_threshold=20.0` are function defaults matching synthetic bench lines 509-515, but `run_synthetic_benchmark` calls `validate_trajectory(trajectory, gate_specs, dt=dt)` **without** forwarding field bounds from `data["field"]["bounds_min/max"]`. A course with ceiling 12 m would still validate against 20 m.
- **Repro**: Trajectory at z=15 with gates in a 10 m tall field — validator passes airspace, bench may disagree if bounds added later.
- **Fix sketch**: Pass bounds from track JSON; add test that custom thresholds propagate from CallSite.
- **Confidence**: medium

### F10. `benchmark_matrix` treats missing `plan_validation.ok` as pass — only explicit `False` fails — [MINOR]
- **File(s)**: `scripts/benchmark_matrix.py:134-138`, `scripts/benchmark_matrix.py:99-99`
- **Issue**: `pv = track_summary.get("plan_validation") or {}` then `if pv.get("ok") is False`. Missing key, `None`, or `{}` → **not** a regression. Today `run_synthetic_benchmark` always populates `plan_validation` (`scripts/benchmark.py:646-655`), so this is latent — but a partial bench refactor or exception path could mark matrix PASS with no plan check.
- **Repro**: Manually inject `plan_validation: null` into matrix track summary — production track still passes if sim ok.
- **Fix sketch**: For non-placeholder tracks, require `pv.get("ok") is True` explicitly.
- **Confidence**: high

### F11. Placeholder tracks skip plan strictness — by design, but easy to misread — [NIT]
- **File(s)**: `scripts/benchmark_matrix.py:107-139`, `sim_pybullet/configs/aigp_default.json:5-5`
- **Issue**: `is_placeholder` tracks only require `gate_pass_rate >= 50%`; the iter-006 `plan_validation.ok` check runs **only** in the `else` branch for production tracks. aigp_default can sim-pass with an illegal plan. Intentional per comment, but matrix JSON now includes `plan_validation` for placeholders without failing on it.
- **Repro**: Inspect matrix acceptance for `aigp_default` with `plan_validation.ok=false`.
- **Fix sketch**: Document in matrix human output; optional `plan_validation` column in ASCII table.
- **Confidence**: high

### F12. figure8 unchanged — still 1/8, plan DQ; deferral undocumented in code — [MAJOR]
- **File(s)**: `sim_pybullet/configs/figure8.json:16-28`, commit `06ba521` message
- **Issue**: Iter-006 matrix: **1/8 gates**, `plan_validation.ok: false`, `sim_time_s: 1.0`. Config notes coplanar gates 1/5 and 3/7 — planner crosses gate-5 before gate-1. No code changes in iter-006 targeted SFC/corridor/start-segment policy; commit message defers to “iter-007+”. Research synthesis still lists SFC as #1 fix — stall continues.
- **Repro**: `python3 -m scripts.benchmark_matrix --configs figure8 --json-only`.
- **Fix sketch**: Iter-007 Phase 2 corridor (per `.loop/synthesis/research_synthesis.md`); do not weaken `enforce_in_order`.
- **Confidence**: high

### F13. ILC section indices still race_01 v=15-era; misaligned with ~5.55 m/s / 18 s profile — [MINOR]
- **File(s)**: `sim_pybullet/configs/race_01.json:21-26`, `scripts/benchmark.py:361-380`
- **Issue**: Hand-tuned `ilc_section_overrides` at step boundaries `[0,200,440,740,...]` were calibrated for ~13.7–14.9 s runs. At 18.08 s the same indices land on different geometric phases of the helix; ILC corrections may be suboptimal (iter-005 F9 carried forward).
- **Repro**: Compare curvature-derived section boundaries vs fixed overrides on post-iter-006 trajectory point count/timing.
- **Fix sketch**: Re-derive overrides from curvature at new speed or drop overrides and measure matrix delta.
- **Confidence**: medium

### F14. Inscribed-arc formula is correct for the tested 90° case; hairpin behaves — [NIT] (positive sanity)
- **File(s)**: `planning/auto_velocity.py:114-114`, `tests/test_auto_velocity.py:42-57`
- **Issue**: For 90° at 3 m spacing, `r = 3/(2·sin(π/4)) = 3/√2 ≈ 2.121 m`, `v ≈ 4.51 m/s` — matches `test_tight_90_turn_three_gates_at_3m_spacing`. β→π gives `r → chord/2` (finite). Zero-length segments skipped (lines 103-104). **Not a bug** in the core formula for the documented slalom test.
- **Repro**: Run `tests/test_auto_velocity.py::test_tight_90_turn_three_gates_at_3m_spacing`.
- **Fix sketch**: None on formula; extend tests to 3D helix and opposite-leg collinear U-turns (bend=π).
- **Confidence**: high

## Things iter-006 got right

- Removed iter-005’s explicit `8.0` / slalom `6.0` JSON overrides — charter directionally correct (`scripts/benchmark.py:337-347`, `sim_pybullet/configs/slalom.json` override removed).
- Plan validator airspace checks close the iter-005 Opus F5 gap — trajectories clipping z envelope fail before sim (`planning/plan_validator.py:129-139`).
- Matrix strictness on `plan_validation.ok is False` for production tracks is the right honesty bar (`scripts/benchmark_matrix.py:129-138`).
- `test_auto_velocity.py` gives concrete regression anchors for 90° turns, cap behavior, and degenerate segments.
- Multi-track PASS (except figure8) at conservative speeds proves the matrix is a useful gate — even if race_01 got slower.

## What I did NOT review

- Full `git show` line-by-line for both commits (read current file state instead).
- PyBullet sim run for race_01 at iter-006 (only synthetic matrix spot-check).
- `race_pipeline.py`, MAVLink bridge, vision stack, EKF noise tuning.
- Whether `SpeedProfiler` should subsume `auto_velocity.py` entirely (read `planning/racing_line.py:706-715` only).
- Opus/GPT iter-006 peer reviews (this file is composer-25 only).
