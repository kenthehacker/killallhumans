# Iter 001 Plan — GPT-5.5 XHigh

## Summary
Make the harness truthful before optimizing flight: add adversarial tests for out-of-order gates, frame strikes, geometry defaults, camera shape, and vision UDP, then make the smallest code changes that clear those bars. Keep race_01 performance by deriving tuning from track geometry and config, not wall-clock race_01 windows; defer full `sim_pybullet/runner.py` collapse because it is currently a large UI/logging/control harness, not a low-cost adapter swap.

## Actions (ordered)
### A1. Add adversarial sequencer tests
- **File(s)**: `gate_sequencing/tests/test_sequencer_adversarial.py:1`, `gate_sequencing/sequencer.py:155`
- **Change**: Create tests that fly through gate 2 while gate 1 is current, then return through gate 1 and try to finish; assert an out-of-order terminal event is recorded and `is_complete` never becomes true. Include a second test where the path crosses an unpassed non-current gate outside its opening and asserts a missed/out-of-order failure is surfaced rather than ignored.
- **Rationale**: Resolves I-1 and I-11 by forcing the sequencer to inspect every still-unpassed gate plane per tick, not only `current_gate`. This follows the competition rule that gate order is strict.
- **Test**: `pytest gate_sequencing/tests/test_sequencer_adversarial.py::test_out_of_order_gate_crossing_is_terminal -q`
- **Risk**: Existing recovery behavior may depend on permissive U-turns after missed gates and will need explicit non-terminal recovery semantics later.
- **Effort**: S

### A2. Add adversarial benchmark honesty tests
- **File(s)**: `tests/test_benchmark_adversarial.py:1`, `scripts/benchmark.py:233`, `scripts/benchmark.py:397`
- **Change**: Create synthetic-track tests that monkeypatch the race config and trajectory path to force a segment through a gate frame, a segment through a later gate before the current gate, and a ground strike; assert `crashed=true`, `sim_passed=false`, and `threshold_failures` names the concrete terminal reason.
- **Rationale**: Resolves I-2 and I-11 by proving the synthetic benchmark cannot report PASS after a frame hit, wrong-order crossing, or crash.
- **Test**: `pytest tests/test_benchmark_adversarial.py -q`
- **Risk**: The current benchmark function is monolithic, so tests may need a small injectable config/path seam to stay deterministic.
- **Effort**: M

### A3. Enforce strict gate order in the sequencer
- **File(s)**: `gate_sequencing/sequencer.py:51`, `gate_sequencing/sequencer.py:97`, `gate_sequencing/sequencer.py:200`
- **Change**: Add `RaceState.FAILED`, add `out_of_order_gate_ids` and `last_out_of_order` properties, and update `GateSequencer.update()` to scan all gates with `sequence_index > current_idx`; if a segment crosses any later gate plane inside its opening, record `out_of_order`, set `FAILED`, and prevent future pass credit.
- **Rationale**: Fixes I-1 without course-specific assumptions; the sequencer becomes the single platform-agnostic enforcement point for strict gate order.
- **Test**: `pytest gate_sequencing/tests/test_sequencer_adversarial.py gate_sequencing/tests/test_sequencer.py -q`
- **Risk**: `RacePipeline.run()` and benchmark loops currently stop on completion/time/crash and must also treat `FAILED` as terminal.
- **Effort**: M

### A4. Make synthetic crashes terminal
- **File(s)**: `scripts/benchmark.py:277`, `scripts/benchmark.py:408`, `scripts/benchmark.py:534`
- **Change**: Reuse the sequencer's geometric crossing helpers or extract public geometry helpers so each synthetic segment from `prev_pos` to `pos` checks current and unpassed gate planes; call `seq.mark_collision()` or record `crash_gate:<id>` when the crossing is inside the outer frame but outside the bare opening, abort immediately, and include crash details in the JSON.
- **Rationale**: Fixes I-2 and makes the synthetic harness obey the same crash semantics already used by PyBullet contact at `scripts/benchmark.py:746`.
- **Test**: `pytest tests/test_benchmark_adversarial.py::test_synthetic_gate_frame_collision_fails_run -q`
- **Risk**: Sharing private geometry helpers directly would couple tests to internals; prefer a small public method on `GateSequencer` if extraction is needed.
- **Effort**: M

### A5. Replace AIGP geometry and camera defaults
- **File(s)**: `gate_sequencing/sequencer.py:26`, `race_pipeline.py:61`, `estimation/gate_pnp.py:33`, `estimation/gate_pnp.py:71`, `competition/adapter.py:140`, `sim_pybullet/env.py:164`, `sim_pybullet/gpd_drone.py:47`, `sim_pybullet/configs/race_01.json:7`
- **Change**: Introduce an `AIGPGeometry` constants module or dataclass with gate inner 1.5m, border 0.6m, depth 0.26m, drone footprint 0.28m x 0.28m x 0.16m, camera 640x360, fx/fy 320, cx/cy 320/180, and +20deg pitch. Wire these defaults into `GateSpec`, `GateGeometry`, `CameraIntrinsics`, `CameraFrame`, `PipelineConfig`, PyBullet camera config, and the race_01 default gate config while preserving per-config overrides.
- **Rationale**: Fixes I-5 and I-8 by making AIGP dimensions ground truth instead of inherited 1.2m gates and 640x480 camera defaults.
- **Test**: `pytest estimation/tests/test_gate_pnp.py competition/tests/test_adapter.py gate_sequencing/tests/test_sequencer.py sim_pybullet/tests/test_gate_contact.py -q`
- **Risk**: race_01 local performance may shift because the simulated opening becomes larger but the frame collision zone also becomes much wider.
- **Effort**: M

### A6. Derive ILC sections from trajectory geometry
- **File(s)**: `scripts/benchmark.py:310`, `planning/trajectory_optimizer.py:184`, `planning/tests/test_trajectory.py:1`
- **Change**: Add a helper such as `derive_ilc_sections_from_curvature(trajectory, dt, gate_waypoints)` that computes section boundaries from curvature/yaw-rate peaks and gate-spacing changes, then replace `inflection_start`, `inflection_end`, and `helix_start` wall-clock constants with those derived boundaries. Keep per-section alpha/cutoff defaults generic and clamp section count to a small fixed budget.
- **Rationale**: Fixes I-3 by removing race_01-specific time windows while retaining the segment-wise ILC pattern from Bristow/Alleyne and the existing `compute_ilc_offset_table()` API.
- **Test**: `pytest planning/tests/test_trajectory.py::test_ilc_sections_follow_curvature_not_wall_clock tests/test_benchmark_adversarial.py::test_synthetic_benchmark_uses_derived_ilc_sections -q`
- **Risk**: Curvature-derived sections can be noisy on nearly straight courses unless the helper falls back to a single global section.
- **Effort**: M

### A7. Add multi-course regression before tuning ILC
- **File(s)**: `tests/test_benchmark_adversarial.py:1`, `sim_pybullet/configs/race_01.json:36`, `scripts/benchmark.py:338`
- **Change**: Add three tiny synthetic course fixtures in test code: straight, S-turn, and vertical helix-like climb. Assert the benchmark runs all three via a new `--synthetic-config` or injectable config seam and that no single-course ILC parameters are accepted unless all courses meet crash/order/finite-error gates.
- **Rationale**: Fixes I-4 by preventing the race_01 sweep values from dominating the loss; convergence threshold and momentum can remain defaults only after they pass varied geometry.
- **Test**: `pytest tests/test_benchmark_adversarial.py::test_ilc_defaults_pass_three_course_regression -q`
- **Risk**: Adding config injection to `run_synthetic_benchmark()` can expose hidden assumptions about race_01 gate IDs and sequence indices.
- **Effort**: M

### A8. Add DCL vision UDP receiver
- **File(s)**: `competition/vision_udp.py:1`, `competition/mavlink_bridge.py:74`, `competition/mavlink_bridge.py:163`, `competition/tests/test_vision_udp.py:1`
- **Change**: Implement `VisionUdpReceiver` for the 24-byte little-endian JPEG chunk header, keyed by `(frame_id, chunk_id, total_chunks)`, with stale-frame cleanup, duplicate chunk handling, JPEG decode to BGR, and `CameraFrame(timestamp_us=sim_time_ns/1000,width=640,height=360)`. Start it from `MAVLinkBridge.__init__`, connect/disconnect lifecycle, and return the latest decoded frame from `get_camera_frame()`.
- **Rationale**: Fixes I-9; the spec's camera stream is separate UDP on port 5600 and `MAVLinkBridge.get_camera_frame()` currently always returns `None`.
- **Test**: `pytest competition/tests/test_vision_udp.py competition/tests/test_adapter.py -q`
- **Risk**: UDP chunk loss can hold incomplete frames in memory unless cleanup is bounded by frame age and total buffered bytes.
- **Effort**: M

### A9. Add MAVLink2 command-path coverage
- **File(s)**: `competition/mavlink_bridge.py:129`, `competition/mavlink_bridge.py:193`, `competition/tests/test_mavlink_bridge_commands.py:1`, `race_pipeline.py:247`
- **Change**: Add tests with a fake MAVSDK system proving `send_position()` maps to `SET_POSITION_TARGET_LOCAL_NED` semantics through `PositionNedYaw`, `send_attitude()` maps to attitude targets, command cadence stays below 100Hz, and heartbeat/offboard lifecycle remains active while vision frames arrive on the separate receiver.
- **Rationale**: Completes I-9 plumbing beyond image decode and protects both supported control paths from the AIGP spec.
- **Test**: `pytest competition/tests/test_mavlink_bridge_commands.py -q`
- **Risk**: MAVSDK wrappers are hard to unit test unless the bridge accepts an injected `System` factory.
- **Effort**: S

### A10. Add drone calibration harness and lightweight ML dynamics regressor
- **File(s)**: `calibration/dynamics_regressor.py:1`, `calibration/collect_mavlink_calibration.py:1`, `control/mpc_tracker.py:1`, `planning/trajectory_optimizer.py:30`, `tests/test_dynamics_regressor.py:1`
- **Change**: Add a small ridge-regression model that fits acceleration residuals from telemetry windows using features `[thrust, roll, pitch, yaw_rate, velocity]`; write/read `drone_calibration.json` with inferred mass/thrust/drag/residual coefficients and apply it to `TrackerConfig`/`DroneConstraints` when present.
- **Rationale**: Resolves I-6 and I-10 with the cheapest ML option that materially helps sim-to-DCL transfer; it uses MAVLink telemetry instead of requiring image labels.
- **Test**: `pytest tests/test_dynamics_regressor.py control/tests/test_tracker.py planning/tests/test_trajectory.py -q`
- **Risk**: Poor excitation data can fit a misleading residual, so the harness must reject calibration logs without hover, step, and ramp segments.
- **Effort**: L

### A11. Add geometry-grounded race config loading checks
- **File(s)**: `sim_pybullet/env.py:153`, `simulation/model_types.py:28`, `simulation/tests/test_scenarios.py:1`, `sim_pybullet/tests/test_runner_replan_integration.py:44`
- **Change**: Update loaders and tests so missing gate geometry uses AIGP defaults, explicit race config overrides still win, sequence indices are normalized to strict order, and drone/camera/gate geometry are exposed as a single source of truth to the pipeline and sim harness.
- **Rationale**: Fixes I-5, I-6, and I-8 by making the AIGP geometry authoritative at load time rather than copied as literals across modules.
- **Test**: `pytest simulation/tests/test_scenarios.py sim_pybullet/tests/test_runner_replan_integration.py -q`
- **Risk**: Existing tests that assumed 1.0m or 1.2m default gates need updates to assert explicit fixture dimensions instead of old defaults.
- **Effort**: S

## ML choice
Pick drone-dynamics regression for sim-to-real calibration. Data path: `MAVLinkBridge` records timestamped attitude/position/velocity/IMU/control commands during hover, thrust step, lateral step, and ramp maneuvers; `calibration/dynamics_regressor.py` fits a small ridge model for acceleration residuals plus scalar mass/drag estimates; output is `drone_calibration.json`; `TrackerConfig` and `DroneConstraints` consume it if present. Train/eval strategy: deterministic train/validation split by maneuver, reject logs with low excitation, assert residual RMSE improves over the uncalibrated physics baseline, and keep the model linear so it is inspectable and cheap at runtime.

## What NOT to do this iter
- Do not fully collapse `sim_pybullet/runner.py` onto `race_pipeline.py`; the file owns UI rendering, CSV logging, PyBullet reset, ad-hoc Catmull-Rom targeting, and keyboard controls, so the swap is not low-cost for Iter 001.
- Do not tune controller gains or ILC values from a single race_01 benchmark; the benchmark is suspect until the adversarial tests pass.
- Do not add end-to-end RL or a heavyweight detector retrain; the fastest useful ML is calibration from MAVLink telemetry, not a new policy or large perception stack.

## Open questions for the synthesiser
- Should Iter 001 allow changing `race_01.json` geometry to AIGP defaults immediately, or keep a second `aigp_vq1_default.json` while race_01 remains a historical local benchmark?
- Should out-of-order gate crossing be represented as `RaceState.FAILED` or as a terminal `RaceState.RECOVERY` subtype with an explicit failure reason?
- Should the DCL UDP vision receiver live in `competition/vision_udp.py` as a reusable adapter component, or directly inside `MAVLinkBridge` for fewer moving parts?
