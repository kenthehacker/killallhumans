# Iter 002 Adversarial Review — GPT-5.5 extra-high 2

## Summary
Iter-002 closes several iter-001 findings, but two of the new "fixed" paths are still fragile enough to mis-score a real run: the RacePipeline timeout check is effectively dead due mixed clock domains, and the sequencer can record a crash and then overwrite the same tick as an out-of-order DQ. I also found an unclosed AIGP camera-default hole in bare `GatePnPEstimator()`, plus live-vision duplicate-frame semantics that can feed repeated PnP measurements into the EKF at control-loop rate.

## Findings (ordered by severity)

### F1. RacePipeline's VQ1 timeout check is dead because it subtracts `time.time()` from `time.monotonic()` — BLOCKER
- **File(s)**: `race_pipeline.py:296`, `race_pipeline.py:380`, `race_pipeline.py:382`, `race_pipeline.py:383`, `competition/session.py:168`
- **Issue**: Iter-002 claims the 8-minute cap is enforced in RacePipeline, but `_race_start_time` is set with epoch wall-clock `time.time()` while the callback computes elapsed with `time.monotonic()`. On a normal process, `time.monotonic() - time.time()` is a huge negative number, so `elapsed > AIGP_VQ1_MAX_RUN_DURATION_S` never becomes true. `RaceSession` has its own wall-clock break, but that path does not call `sequencer.mark_timed_out()`, so the new `RaceState.TIMED_OUT`, `should_stop` timed-out branch, and callback timed-out early return remain unexercised in the live pipeline.
- **Repro**: Start a `RacePipeline.run()` session; `_race_start_time = time.time()` at race start. In `_control_callback`, even after 480 real seconds, `time.monotonic() - self._race_start_time` remains roughly `-1.7e9`, so lines 383-386 never fire. A unit test can reproduce without waiting by setting `_race_start_time = time.time() - 481` and observing that the monotonic subtraction is still negative on macOS.
- **Fix sketch**: Use one clock consistently. Prefer `self._race_start_time = time.monotonic()` if this is intended as wall-clock elapsed, or have `RaceSession` pass elapsed/sim time into the control callback and call `sequencer.mark_timed_out()` before breaking. Add a targeted test that freezes or stubs both clocks and asserts the sequencer enters `TIMED_OUT` and `session.should_stop()` trips.
- **Confidence**: high — the two clock calls are directly visible and the sign of monotonic-vs-epoch elapsed is deterministic.

### F2. Same-segment crash can be overwritten as DQ, losing the physical failure priority — MAJOR
- **File(s)**: `gate_sequencing/sequencer.py:327`, `gate_sequencing/sequencer.py:330`, `gate_sequencing/sequencer.py:335`, `gate_sequencing/sequencer.py:413`, `gate_sequencing/sequencer.py:437`, `gate_sequencing/sequencer.py:438`, `gate_sequencing/sequencer.py:440`
- **Issue**: A current-gate strut hit appends `last_crash` and sets `_last_event = "crash"`, but it does not terminate the update or suppress the future-gate DQ scan. In the same `prev -> pos` segment, if the drone also crosses a future gate opening, the DQ branch sets `RaceState.DISQUALIFIED`, `_dq_reason`, and `_last_event = "dq"`. The review brief explicitly asks which wins when a future-gate event and current-gate crash happen in one segment; the current answer is "both, with DQ overwriting the sequencer state", which makes crash/DQ reporting order-dependent rather than physics-authoritative.
- **Repro**: With gates at `x=5` and `x=10`, call `seq.update((4.0, 1.0, -2.0))` then `seq.update((10.5, 0.0, -2.0))`. The segment hits `g1`'s strut at about `y=0.846` and crosses `g2`'s opening. Current result: `last_crash == ('g1', ...)`, `is_disqualified == True`, `dq_reason == 'out_of_order:g2'`, `last_event == 'dq'`.
- **Fix sketch**: Define terminal-event precedence and enforce it in one place. For competition scoring, a physical crash should stop classification for the tick before DQ scanning, or the sequencer should gain an explicit `CRASHED` terminal state. At minimum, track `crash_classified_this_update` and skip the future-gate DQ loop once true. Add a regression test for "current strut + future opening in one segment".
- **Confidence**: high — I reproduced the mixed crash/DQ state with the current code.

### F3. Bare `GatePnPEstimator()` still constructs a legacy 640x480 camera — MAJOR
- **File(s)**: `estimation/gate_pnp.py:47`, `estimation/gate_pnp.py:63`, `estimation/gate_pnp.py:64`, `estimation/gate_pnp.py:70`, `estimation/gate_pnp.py:148`, `estimation/gate_pnp.py:153`
- **Issue**: Iter-002 updated `CameraIntrinsics` and `GateGeometry` defaults to AIGP, and the pipeline manually threads its camera config. But a standalone `GatePnPEstimator()` still does `CameraIntrinsics.from_fov(90.0, 640, 480)`, yielding `image_height=480` and `cy=240`. That keeps part of the iter-001 "PnPEstimator standalone defaults still 1.2 m / 640x480" finding alive: the geometry half is fixed, the camera half is not.
- **Repro**: In the worktree, `GatePnPEstimator().camera.image_height` returns `480` and `GatePnPEstimator().camera.cy` returns `240.0`, despite `CameraIntrinsics()` itself defaulting to `360`/`180`.
- **Fix sketch**: Change `self.camera = camera or CameraIntrinsics()` in `GatePnPEstimator.__init__`. If legacy 640x480 support is needed, require explicit constructor args. Add `test_gate_pnp_estimator_defaults_to_aigp_camera()` alongside `test_intrinsics_default_to_aigp()`.
- **Confidence**: high — the constructor path directly hard-codes the old dimensions.

### F4. `latest_frame()` repeatedly decodes and reprocesses the same JPEG — MAJOR
- **File(s)**: `competition/vision_udp.py:257`, `competition/vision_udp.py:259`, `competition/vision_udp.py:400`, `competition/vision_udp.py:407`, `competition/vision_udp.py:410`, `race_pipeline.py:341`, `race_pipeline.py:343`, `race_pipeline.py:576`, `race_pipeline.py:577`
- **Issue**: `VisionUdpReceiver.pop_latest_frame()` is named like a consuming pop, but it returns `_latest_frame` without clearing it. `VisionUdpListener.latest_frame()` then decodes that same JPEG on every call until a newer frame completes. In the live bridge, the 100 Hz control loop polls a 30 FPS camera stream, so the same camera frame can be decoded 3-4 times and passed through detector/PnP multiple times. That is not just CPU waste: `_process_detection()` can feed duplicate PnP position updates into the EKF at control-loop cadence from one camera exposure.
- **Repro**: Feed one complete frame into `VisionUdpReceiver`, then call `VisionUdpListener.latest_frame()` repeatedly without sending another packet. Each call reaches `decode_jpeg_to_camera_frame(rf)` on the same `frame_id`; `RacePipeline._control_callback` treats each returned `CameraFrame` as fresh perception and may call `ekf.update_pnp_position()` again.
- **Fix sketch**: Either make `pop_latest_frame()` actually consume the frame, or cache decoded frames by `frame_id` and expose "new frame since last poll" semantics to the bridge. If downstream intentionally wants latest-sample replay, add a timestamp/frame-id guard in `RacePipeline` so PnP updates happen once per camera frame while the controller can still run at 100 Hz.
- **Confidence**: high on repeated decode/replay; medium on EKF impact because it depends on detector/PnP returning a valid match for the repeated image.

### F5. Calibration accepts implausible fits and unvalidated JSON — MAJOR
- **File(s)**: `competition/calibration.py:104`, `competition/calibration.py:107`, `competition/calibration.py:109`, `competition/calibration.py:113`, `competition/calibration.py:121`, `competition/calibration.py:153`, `competition/calibration.py:159`, `competition/calibration.py:160`, `competition/calibration.py:163`
- **Issue**: The physics equation is fixed, but the accept/reject gate is only `k_t > 0`. Bad convention data can fit to a large positive thrust ratio with large residual and still be accepted; JSON reload accepts arbitrary floats without finite/range/schema checks. Since the calibration path is meant to bridge PyBullet-vs-DCL dynamics, silently accepting garbage can poison controller and planner limits.
- **Repro**: Generate old-convention samples with `accel_z = -22*u - 0.4*v - 9.81`; current `identify_thrust_drag_ratios()` returns `thrust_per_mass=50.98`, `drag_per_mass=0.62`, `rmse=5.45` and does not raise. A hand-written calibration JSON with `thrust_per_mass_1_per_s2 = 1e6`, negative drag, or `NaN` values is also accepted by `read_calibration_json()`.
- **Fix sketch**: Reject non-finite values, negative/implausible ratios, low-rank/low-excitation sample sets, and fits whose RMSE exceeds a documented threshold. Store a schema/version in the JSON and validate it on read. Add tests for old-sign samples, all-constant thrust rank deficiency, and JSON bounds.
- **Confidence**: high — the current code computes RMSE but never gates on it, and the JSON path blindly casts required fields.

## Things iter-002 got right
- The core calibration regression sign is corrected: `y = gravity - a` and `X = [u, v]` now match the NED equation.
- `CameraFrame` defaults are now 640x360, and `PipelineConfig.camera_pitch_offset_rad` is threaded into the pipeline-created intrinsics.
- The UDP receiver now validates zero payloads and total payload size against `jpeg_size`.
- The sequencer's strict-opening DQ check no longer uses the lenient pass-through margin.
- Benchmark results now separate physical `crashed` from rule `disqualified`, which makes downstream scoring less misleading.

## What I did NOT review
I did not run the full benchmark or PyBullet simulation. I did not exercise MAVSDK against a live DCL simulator, so bridge lifecycle concerns are based on static code inspection and UDP listener tests only. I read the specified charter, iter-001 synthesis, iter-002 commit diffs, and the main changed files (`sequencer.py`, `vision_udp.py`, `calibration.py`, `race_pipeline.py`, `mavlink_bridge.py`, `benchmark.py`, camera/PnP code, and relevant tests), but I did not exhaustively audit unrelated planner/controller modules beyond the ILC section changes.
