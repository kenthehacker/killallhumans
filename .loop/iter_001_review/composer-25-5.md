# Iter 001 Adversarial Review — Composer 2.5

## Summary

Iter-001 materially closes I-1/I-2/I-3/I-5 on the **bench + unit-test** surface: sequencer DQ, synthetic honesty fields, curvature ILC, and AIGP geometry constants are real improvements. The competition path is still half-wired: vision UDP reassembles in isolation but `MAVLinkBridge.get_camera_frame()` returns `None`, and `RacePipeline` never stops on `is_disqualified` or sequencer crash. The largest correctness gap is **A6 camera tilt**: `pitch_offset_rad` exists on `CameraIntrinsics` but is **not applied** in `gate_pose_to_drone_position`, so PnP→EKF drift correction still assumes a forward-aligned camera. Treat green unit tests as necessary, not sufficient.

## Findings (ordered by severity)

### F1. Vision receiver not wired to competition bridge — [BLOCKER]
- **File(s)**: `competition/mavlink_bridge.py:163-167`, `competition/vision_udp.py:144-235`
- **Issue**: A10/A11 shipped `VisionUdpReceiver` + tests, but `MAVLinkBridge.get_camera_frame()` still hard-returns `None` with a "spec not yet released" comment. The DCL sim will publish JPEG chunks on UDP :5600 per VADR-TS-002; without wiring, `RacePipeline._process_detection` never runs (`detection_active=False`), so the stack races blind on the actual competition interface.
- **Repro**: `await MAVLinkBridge().get_camera_frame()` after connect — always `None`. Any integration test that only mocks telemetry will miss this.
- **Fix sketch**: Instantiate `VisionUdpReceiver` on `connect()`, run an `asyncio` datagram task feeding `feed_packet`, expose `pop_latest_frame()` → `decode_jpeg_to_camera_frame()` in `get_camera_frame()`. Add an integration test that feeds crafted chunks and asserts a non-`None` `CameraFrame` at 640×360.
- **Confidence**: high

### F2. RacePipeline ignores sequencer DQ and crash — [BLOCKER]
- **File(s)**: `race_pipeline.py:271-273`, `race_pipeline.py:324-343`, `gate_sequencing/sequencer.py:161-168`
- **Issue**: `session.should_stop` only checks `sequencer.is_complete`. After `is_disqualified` or a frame-strut `last_crash`, `update()` early-returns on subsequent ticks, but `_control_callback` still runs replanner + tracker and `session.run()` keeps commanding the vehicle until completion or 8-minute session timeout. Bench (A7) terminates on DQ/crash; the production pipeline does not mirror that.
- **Repro**: Force DQ in a harness calling `_control_callback` after `sequencer.update` trips `out_of_order:*` — commands still emitted; `should_stop` stays false.
- **Fix sketch**: Extend `should_stop` to `is_complete or is_disqualified or last_crash is not None`; in `_control_callback`, return hover/zero-thrust immediately when terminal. Mirror in `competition/session.py` metrics (`gates_passed` vs DQ reason).
- **Confidence**: high

### F3. Camera +20° tilt not applied in PnP world transform — [BLOCKER]
- **File(s)**: `estimation/gate_pnp.py:214-258`, `estimation/gate_pnp.py:61-66`, `tests/test_camera_geometry.py:75-91`
- **Issue**: A6 added `CameraIntrinsics.pitch_offset_rad` and analytic horizon tests, but `gate_pose_to_drone_position` maps `camera_in_gate` with `R_gate_world @ camera_in_gate` only — no `R_body_camera(pitch_offset_rad)`. The `drone_orientation` argument is unused. Synthesis A6 explicitly promised tilt "threaded through `gate_pose_to_drone_position`"; that step is missing. EKF position updates from PnP will be systematically biased on the real AIGP camera.
- **Repro**: Compare recovered drone position for a gate at known range with `pitch_offset_rad=20°` vs `0°` — positions coincide today. `test_camera_geometry.py` only checks `cy + fy·tan(pitch)` algebra, not `gate_pose_to_drone_position`.
- **Fix sketch**: Apply `R_pitch(self.camera.pitch_offset_rad)` (body Y) between camera and gate frames per A6 plan; add a regression test with a synthetic gate pose where +20° tilt shifts world position by a predictable Δ (>0.1 m at 5 m range).
- **Confidence**: high

### F4. 8-minute max run enforced only in RaceSession, not bench/pipeline — [MAJOR]
- **File(s)**: `competition/aigp_geometry.py:65`, `competition/session.py:36-170`, `scripts/benchmark.py:928-958`, `race_pipeline.py:255-281`
- **Issue**: `AIGP_VQ1_MAX_RUN_DURATION_S = 480` exists but `run_synthetic_benchmark` / `run_sim_benchmark` default `--duration 30` with no reference to the spec cap. `RacePipeline.run()` delegates timeout to `RaceSession` (good for MAVLink path) but nothing sets `GateSequencer` `TIMED_OUT` or marks bench runs invalid after 480 s. Long synthetic sweeps can report PASS on truncated courses without mirroring competition disqualification semantics.
- **Repro**: `python3 scripts/benchmark.py --mode sim --duration 600` — no spec-aligned timeout failure; only user-provided duration ends the loop.
- **Fix sketch**: Import `AIGP_VQ1_MAX_RUN_DURATION_S` in `benchmark.py` and cap `duration`; set `sequencer._state = TIMED_OUT` when exceeded; add adversarial test that a 481 s configured run fails `sim_passed`.
- **Confidence**: high

### F5. race_01 ILC overrides are course magic in JSON disguise — [MAJOR]
- **File(s)**: `sim_pybullet/configs/race_01.json:15-26`, `scripts/benchmark.py:350-356`, `.loop/specs/0_charter.md:12`
- **Issue**: A9 removed `int(2.0/dt)` literals from `benchmark.py`, but `race_01.json` still carries the iter-47-49 helix schedule verbatim (`[0,200]`, `[200,440]`, `[740,99999]` with tuned α/vel_scale). New tracks use curvature partition; **the default development track still bypasses the generic algorithm**, so ILC regressions on unknown VQ1 geometry are invisible when developers only bench `race_01`.
- **Repro**: Run bench on `aigp_default.json` vs `race_01.json` — section boundaries differ entirely; tuning work on race_01 does not transfer.
- **Fix sketch**: iter-002: run `benchmark_matrix` (A17) as CI gate; document race_01 overrides as legacy-only; add test that configs without `ilc_section_overrides` never load the race_01 tuple literals.
- **Confidence**: high

### F6. `enforce_in_order=True` can false-DQ replanner recovery — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:337-355`, `race_pipeline.py:338-343`, `planning/dynamic_replanner.py:12-20`
- **Issue**: DQ scans `self._gates[self._current_idx + 1:]` for opening-inside plane crossings. After a miss/off-track, `DynamicReplanner` rebuilds a line through remaining gates; a legitimate recovery arc may cut through a **future** gate's opening before the current target is credited. That is a terminal DQ, not a recoverable miss — stricter than many competition interpretations and harsher than pre-fix "silent skip" (different failure mode).
- **Repro**: Simulate recovery spline from current pose to `gate-5` that passes through `gate-7` opening at y≈0 while `current_idx` points at gate-5 — expect `out_of_order:gate-7` even if the planner intended a wide rejoin.
- **Fix sketch**: Option A: DQ only for gates with `sequence_index < current.sequence_index + K` (next-N). Option B: on replan, temporarily set `enforce_in_order=False` until back inside corridor. Add adversarial test: wide replan arc vs strict DQ.
- **Confidence**: medium (product call, but code makes it deterministic)

### F7. Vision reassembly does not validate `jpeg_size` vs assembled bytes — [MAJOR]
- **File(s)**: `competition/vision_udp.py:84-97`, `competition/vision_udp.py:220-225`
- **Issue**: `parse_packet` enforces `payload_size == len(payload)` per datagram, but on completion `assemble()` pre-allocates `bytearray(jpeg_size)` and concatenates chunks without checking `sum(len(chunk)) == jpeg_size` or that assembled length matches. Malicious or buggy sender can declare `jpeg_size=1_000_000` (memory pressure) or mismatched total vs chunks; `cv2.imdecode` may still fail silently downstream.
- **Repro**: Feed chunks whose payloads sum to 500 B with header `jpeg_size=10000`, `total_chunks` satisfied — completion returns 10 KB buffer with trailing zeros; no error.
- **Fix sketch**: After assembly, `if len(jpeg_bytes) != buf.jpeg_size: drop + counter`; cap `jpeg_size` (e.g. 2 MB). Add `test_reassembly_jpeg_size_mismatch_raises_or_drops`.
- **Confidence**: high

### F8. ILC step mapping can drop sections on non-uniform timestamps — [MAJOR]
- **File(s)**: `planning/ilc_sections.py:124-136`, `planning/ilc_sections.py:141-147`
- **Issue**: Point runs convert via `int(points[i].time / dt)`. Multiple trajectory points can map to the same step; when `e_step <= s_step`, the run is **skipped** (`continue`), potentially removing a high-curvature class segment. Non-uniform `dt_sample` (optimizer outputs) makes this worse. Tests use uniform `_StubTraj` spacing — gap not covered.
- **Repro**: Build stub with two "high" points at `t=1.00` and `t=1.001` with `dt=0.01` → both map to step 100; interior high run dropped; partition collapses to fewer sections than acceleration semantics imply.
- **Fix sketch**: Map via `np.searchsorted` on times, or enforce `e_step = max(s_step + 1, _pt_to_step(e_pt))`; add test with clustered timestamps asserting high-class segment survives.
- **Confidence**: medium-high

### F9. Deferred iter-001 actions leave competition gaps — [MAJOR]
- **File(s)**: `.loop/synthesis/iter_001.md:86-92`, `.loop/specs/4_review_brief.md:19-22`
- **Issue**: Six commits landed A1-A11, A13-A15, A18; **A12** (`test_mavlink_bridge_commands.py`), **A16** (`run_pipeline_pybullet.py`), **A17** (`benchmark_matrix.py`) are absent. Synthesis marked A16 as "partial I-7" — file does not exist. No multi-track regression gate; MAVLink command-path locking untested; runner/pipeline collapse still split (`sim_pybullet/runner.py` vs `race_pipeline.py`).
- **Repro**: `glob **/run_pipeline_pybullet.py` → empty; `glob **/benchmark_matrix.py` → empty.
- **Fix sketch**: Prioritize A11→bridge wiring + A12 before new features; A17 before tightening `THRESHOLDS`; A16 to de-risk iter-002 collapse.
- **Confidence**: high

### F10. Calibration stub accepts degenerate thrust — [MAJOR]
- **File(s)**: `competition/calibration.py:90-108`, `tests/test_calibration.py:71-77`
- **Issue**: All-zero `thrust_normalized` samples yield `thrust_per_mass=-0.0`, `drag_per_mass=-0.0`, `rmse=0` via `lstsq` without rank check. No NaN, but silently useless — downstream could write `drone_calibration.json` and zero out thrust limits. Constant-thrust hover data also fits with `k_t=0`.
- **Repro**: `identify_thrust_drag_ratios([CalibrationSample(0.0, v, -9.81) for v in ...])` → zeros.
- **Fix sketch**: Require thrust span ≥ ε, condition number bound on `X`, or `ValueError` on near-singular systems; test `test_zero_thrust_span_raises`.
- **Confidence**: high

### F11. `CameraFrame` and bare `GatePnPEstimator` still default to 640×480 — [MINOR]
- **File(s)**: `competition/adapter.py:146`, `estimation/gate_pnp.py:144`, `estimation/gate_pnp.py:106-107`
- **Issue**: `PipelineConfig` defaults to 360 height, but `CameraFrame.height` defaults to 480. `GatePnPEstimator()` without args uses `from_fov(90, 640, 480)` and `GateGeometry(1.2, 1.2)` — legacy sim shape, not AIGP. `RacePipeline` passes explicit camera/geometry; secondary call sites may not.
- **Repro**: `GatePnPEstimator()` → `image_height==480`, `interior_width_m==1.2`.
- **Fix sketch**: Default `CameraFrame` to AIGP dims; `GateGeometry` from `AIGP_GATE_INTERIOR_M`; `GatePnPEstimator()` default `CameraIntrinsics()` (AIGP) not `from_fov(..., 480)`.
- **Confidence**: high

### F12. `PipelineConfig.camera_pitch_offset_rad` not wired to intrinsics — [MINOR]
- **File(s)**: `race_pipeline.py:76-78`, `race_pipeline.py:123-127`
- **Issue**: Config exposes `camera_pitch_offset_rad` but `__init__` builds `CameraIntrinsics.from_fov(...)` without passing pitch — only dataclass default (+20°) applies. Setting `PipelineConfig(camera_pitch_offset_rad=0)` does **not** disable tilt (F3 already broken, but config is misleading).
- **Repro**: `PipelineConfig(camera_pitch_offset_rad=0.0)` + inspect `pipe.camera.pitch_offset_rad` → still +20°.
- **Fix sketch**: After `from_fov`, set `camera.pitch_offset_rad = config.camera_pitch_offset_rad` or construct `CameraIntrinsics(...)` directly from AIGP constants.
- **Confidence**: high

### F13. Tracker residual clamp order is safe; pipeline terminal gate is not — [MINOR]
- **File(s)**: `control/mpc_tracker.py:214-245`
- **Issue**: Residual deltas are clamped to ±0.05, then added, then re-clamped to `max_tilt_rad` (0.85). The brief's 0.85+0.05 leak scenario does **not** occur. However, safety depends on residual running **after** first tilt clamp and only in `GeometricTracker` — `SimplePositionTracker` has no residual path; future refactors could reorder clamps.
- **Repro**: `use_residual=True` with `TrackerResidualMLP` output `[10,10,10]` → rolls/pitches stay ≤ 0.85.
- **Fix sketch**: Add comment/assert in `track()` that residual must remain post-clamp; optional single final `AttitudeCommand` sanitizer used by both trackers.
- **Confidence**: high for clamp; medium for refactor risk

### F14. `_delivered_ids` sliding window can resurrect duplicate frames — [MINOR]
- **File(s)**: `competition/vision_udp.py:172-233`, `competition/vision_udp.py:49`
- **Issue**: Cap `max_buffered_frames * 8` (64 with default 8). Late chunk for an evicted `frame_id` is treated as new work, not dropped — could emit duplicate `ReassembledFrame` with same `frame_id` after long runs.
- **Repro**: Deliver frame 1, advance 65+ newer frames so ID 1 ages out of `_delivered_ids`, resend chunk for frame 1 — `dropped_late_packets` may be 0 and frame re-emitted.
- **Fix sketch**: Use bounded `collections.OrderedDict` or modulo ring keyed by `frame_id`; treat unknown-late as drop if ID < latest_delivered - window.
- **Confidence**: medium

### F15. Sequencer `TIMED_OUT` never set — [NIT]
- **File(s)**: `gate_sequencing/sequencer.py:71`, `gate_sequencing/sequencer.py:222-228`
- **Issue**: `RaceState.TIMED_OUT` exists but no code path assigns it (F4). Dead enum today; confuses readers expecting spec-aligned timeout in sequencer.
- **Fix sketch**: Wire in bench/pipeline when duration exceeded; or remove enum until implemented.
- **Confidence**: high

## Things iter-001 got right

- **Sequencer DQ (I-1)**: Opening-inside crossings of future gates terminate with `out_of_order:<id>`; U-turn false-complete covered in `gate_sequencing/tests/test_sequencer_adversarial.py`.
- **Bench honesty (I-2)**: Synthetic/PyBullet loops break on `last_crash` and `is_disqualified`; result dict exposes `disqualified`, `dq_reason`, `last_crash_gate` (`scripts/benchmark.py:446-453`, `588-590`).
- **ILC default path (I-3)**: Wall-clock `2.0/4.4/7.4` literals gone from `benchmark.py`; `derive_section_boundaries` + `config/ilc_defaults.json` with tests in `tests/test_ilc_sections.py`.
- **Vision unit tests (I-9 partial)**: Chunk OoO, timeout GC, duplicate handling well specified in `tests/test_vision_udp.py` for the synchronous reassembler.
- **Tracker residual safety default**: `use_residual=False` preserves baseline; hard clamps at consumer (`control/mpc_tracker.py:231-245`).

## What I did NOT review

- Full `planning/dynamic_replanner.py` replan waypoint construction and interaction with `trajectory_optimizer`.
- `scripts/train_tracker_residual.py`, `control/residual_weights.npz`, holdout eval gate from synthesis.
- `sim_pybullet/runner.py` end-to-end behaviour vs `race_pipeline.py` (only spot-checked replan/DQ wiring).
- Executing the full 310-test matrix or PyBullet sim runs (read-only static analysis).
- `competition/mavlink_bridge.py` telemetry subscription loops and offboard command encoding beyond `get_camera_frame`.
- `sim_pybullet/configs/aigp_default.json` placeholder gate layout quality.
- Proto/MAVSDK version skew and Windows DCL binary behaviour (not in worktree).
