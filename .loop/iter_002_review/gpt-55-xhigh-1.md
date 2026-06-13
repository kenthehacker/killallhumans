# Iter 002 Adversarial Review — GPT-5.5 xhigh-1

## Summary
Iter-002 closed several iter-001 holes, but the shipped timeout path is still broken in the live `RacePipeline`: it mixes epoch and monotonic clocks, so the 8-minute cap never fires. I also found two production-path vision listener issues, a stale standalone PnP default, and a sequencer crash-ordering bug where a later strut hit can overwrite the first terminal impact in the same segment.

## Findings (ordered by severity)

### F1. RacePipeline 8-minute cap uses incompatible clocks, so it never trips — BLOCKER
- **File(s)**: `race_pipeline.py:296`, `race_pipeline.py:382`, `race_pipeline.py:392`
- **Issue**: `_race_start_time` is set with `time.time()`, but timeout and replanner elapsed time are computed with `time.monotonic() - self._race_start_time`. On this machine that yields about `-1779604690s`, so `elapsed > AIGP_VQ1_MAX_RUN_DURATION_S` is never true. This means the iter-002 "8-minute VQ1 timeout" is still dead on the real `RacePipeline` path, and `_maybe_replan()` receives a huge negative `sim_time`.
- **Repro**: In the worktree, `race_start = time.time(); time.monotonic() - race_start` printed `-1779604690.4636452`. A pipeline run would need billions of seconds of monotonic uptime before the timeout condition becomes positive.
- **Fix sketch**: Set `_race_start_time = time.monotonic()` if wall time remains the chosen source. Better: thread sim/telemetry time into the callback/session and use that for both timeout and replanner, then return immediately after `mark_timed_out()` so no extra command is emitted past the cap.
- **Confidence**: high — this is a direct clock-source mismatch in the current code.

### F2. Bare `GatePnPEstimator()` still uses legacy 640x480 intrinsics — MAJOR
- **File(s)**: `estimation/gate_pnp.py:72`, `estimation/gate_pnp.py:153`, `tests/test_camera_geometry.py:45`
- **Issue**: `CameraIntrinsics()` now defaults to AIGP 640x360, but `GatePnPEstimator.__init__` still constructs `CameraIntrinsics.from_fov(90.0, 640, 480)` when no camera is supplied. That leaves a standalone estimator at `cy=240` / height `480`, contradicting the AIGP default and the iter-001 review concern about standalone PnP defaults. The tests cover `CameraIntrinsics()` but not `GatePnPEstimator()`.
- **Repro**: `GatePnPEstimator().camera.image_width, GatePnPEstimator().camera.image_height` prints `640 480`; `GatePnPEstimator().gate` now correctly defaults to `1.5m`, so only the camera half stayed stale.
- **Fix sketch**: Change the default to `self.camera = camera or CameraIntrinsics()` and add an adversarial test asserting `GatePnPEstimator().camera.image_height == AIGP_CAM_HEIGHT_PX` and `cy == AIGP_CAM_CY`.
- **Confidence**: high — the constructor hard-codes the legacy height.

### F3. Multi-strut segment records a later crash as `last_crash` instead of stopping at first impact — MAJOR
- **File(s)**: `gate_sequencing/sequencer.py:327`, `gate_sequencing/sequencer.py:330`, `gate_sequencing/sequencer.py:447`, `gate_sequencing/sequencer.py:452`
- **Issue**: A current-gate strut hit sets `_last_event = "crash"`, but the later future-gate scan still runs and can append another crash for a farther gate in the same prev→pos segment. The run's `last_crash` then points at the later gate, even though the first strut contact should be terminal under the charter.
- **Repro**: With two AIGP gates at x=5 and x=10, a single segment from `(4.0, 1.0, -2.0)` to `(10.5, 1.0, -2.0)` produces `crashed_gate_ids == ['g1', 'g2']` and `last_crash == ('g2', ...)`. The physical first impact was g1.
- **Fix sketch**: Once `crash_classified` fires for the current target, skip future-gate DQ/crash scanning for that tick. If same-segment multi-event ordering matters more generally, compute all plane intersections with their segment parameter `t`, sort them, and consume only the earliest terminal event.
- **Confidence**: high — the current branch appends both crashes by construction.

### F4. `VisionUdpListener.start()` is not idempotent and can leak transports — MAJOR
- **File(s)**: `competition/vision_udp.py:376`, `competition/vision_udp.py:380`, `competition/vision_udp.py:385`, `tests/test_vision_udp_listener.py:152`
- **Issue**: `start()` always creates a new datagram endpoint and overwrites `self._transport` without closing any existing transport. `stop()` is tested as idempotent, but `start()` followed by `start()` is not. With a fixed port, the second call can raise address-in-use; with `port=0` or a socket option allowing rebinding, the first socket is orphaned and `stop()` only closes the second.
- **Repro**: Starting a `VisionUdpListener(port=0)` twice left the first transport open: after `await listener.stop()`, the second transport was closing but the first transport's `is_closing()` remained `False`.
- **Fix sketch**: Make `start()` a no-op when `is_listening`, or explicitly `await stop()` before rebinding. If `port=0` is used in tests, update `self.port` to the actual bound port. Add a lifecycle test for double-start.
- **Confidence**: high — the old transport handle is overwritten.

### F5. `latest_frame()` re-decodes and re-emits the same stale frame every poll — MAJOR
- **File(s)**: `competition/vision_udp.py:257`, `competition/vision_udp.py:400`, `competition/vision_udp.py:407`, `competition/mavlink_bridge.py:189`
- **Issue**: `VisionUdpReceiver.pop_latest_frame()` does not pop; it returns `_latest_frame` without clearing it. `VisionUdpListener.latest_frame()` decodes that same JPEG on every call. At a 100 Hz control loop, one received frame can be decoded 100 times per second until the next frame arrives, and `MAVLinkBridge.get_camera_frame()` keeps returning a stale image instead of surfacing "no new frame".
- **Repro**: Monkeypatching `decode_jpeg_to_camera_frame` and calling `listener.latest_frame()` twice after one reassembled frame increments the fake decode counter twice and returns frame id 1 both times.
- **Fix sketch**: Track the last decoded `frame_id` and return a cached `CameraFrame` for repeat calls, or change `pop_latest_frame()` to clear `_latest_frame` and make `latest_frame()` return `None` until a new frame arrives. Choose semantics explicitly and test them.
- **Confidence**: high — the method name and implementation disagree, and the decode path is per-call.

### F6. Calibration JSON loader accepts physically impossible or non-finite parameters — MINOR
- **File(s)**: `competition/calibration.py:152`, `competition/calibration.py:159`, `competition/calibration.py:164`
- **Issue**: `read_calibration_json()` casts only the required fields and performs no schema or physics validation. A hand-edited or corrupted calibration can load negative `thrust_per_mass`, `NaN`/`inf` RMSE, non-positive `n_samples`, or string-valued optional fields. That bypasses the positivity guard added to `identify_thrust_drag_ratios()` and can poison downstream controller/planner tuning.
- **Repro**: A JSON file with `"thrust_per_mass_1_per_s2": -1` will deserialize into a `CalibrationResult` without error; optional fields such as `"mass_kg": "heavy"` are returned as strings.
- **Fix sketch**: Validate required keys, finite numeric values, `thrust_per_mass > 0`, `n_samples >= 2`, `rmse >= 0`, optional physical values either `None` or finite positive numbers, and consistency like `max_thrust_n ~= thrust_per_mass * mass_kg` when both are present.
- **Confidence**: high — no validation exists in the loader.

### F7. Benchmark paths still do not model the VQ1 8-minute timeout contract — MINOR
- **File(s)**: `scripts/benchmark.py:422`, `scripts/benchmark.py:636`, `scripts/benchmark.py:763`, `scripts/benchmark.py:898`
- **Issue**: The synthetic loop uses `duration` as the only loop cap and never calls `seq.mark_timed_out()`; PyBullet uses `duration` similarly and its threshold checks omit even the synthetic path's `race_time` failure. Default runs are 30s, but the benchmark contract now has a sequencer `TIMED_OUT` state that the bench never exercises, so timeout regressions like F1 are invisible.
- **Repro**: Grepping the benchmark logic by inspection: synthetic computes `sim_time = step * dt` and breaks only through loop bounds / completion / crash / DQ; PyBullet breaks when `sim_time > duration`; neither path imports or checks `AIGP_VQ1_MAX_RUN_DURATION_S`, and PyBullet threshold checks stop at gate pass rate.
- **Fix sketch**: Cap both bench loops at `min(duration, AIGP_VQ1_MAX_RUN_DURATION_S)`, call `seq.mark_timed_out()` when the cap is exceeded, surface `timed_out` / `timeout_reason`, and add a targeted test with a tiny injected cap or long-duration no-completion course.
- **Confidence**: medium — the default benchmark duration is shorter than 8 minutes, but the timeout state was specifically shipped and remains untested.

## Things iter-002 got right
- Calibration physics was corrected for the hover case; the new hover test would have caught the original negative `k_t` regression.
- The UDP reassembler now rejects zero payloads and drops completed frames whose chunk payload sum does not match `jpeg_size`.
- `MAVLinkBridge.get_camera_frame()` is no longer a hard-coded `None`; it is wired to the listener path.
- DQ and crash are separated in the synthetic benchmark result surface, so a rule violation no longer masquerades as a physical impact.
- Gate geometry defaults are centralized around the AIGP 1.5 m opening, and legacy tracks can still override per gate.

## What I did NOT review
I did not run the full 334-test suite or PyBullet benchmark. I reviewed the required brief files, the iter-001 synthesis, the six commit stats, the key changed source/tests called out by the brief, and ran targeted read-only Python repros for the clock, PnP default, listener lifecycle, repeated decode, and multi-strut crash cases. I did not exercise MAVSDK against a live DCL simulator or validate real JPEG throughput under load.
