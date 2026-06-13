# Iter 002 Adversarial Review — Composer 2.5

## Summary

Iter-002 closes most iter-001 BLOCKERs on paper: calibration physics, pipeline DQ/crash/timeout termination, vision UDP → `MAVLinkBridge`, multi-gate-per-tick crediting, future-gate strut crashes, strict-opening DQ, and jpeg_size validation are real fixes with targeted tests. The remaining gaps are **integration completeness** (8-minute cap not on bench/session honesty path, timeout state often never reaches `RaceState.TIMED_OUT`), **perception stack drift** (`GatePnPEstimator` / `Phase1GateDetector` still 640×480 while transport is 360), and **calibration robustness** (positivity-only guard accepts wildly wrong fits). Treat 334 green unit tests as necessary, not sufficient for VQ1.

## Findings (ordered by severity)

### F1. VQ1 8-minute cap not enforced on benchmark paths — [MAJOR] (iter-001 HARD MISS partial)
- **File(s)**: `scripts/benchmark.py:317-320,421-424,766-768`, `competition/aigp_geometry.py:78`
- **Issue**: `AIGP_VQ1_MAX_RUN_DURATION_S` is wired in `race_pipeline.py:380-386` but **never referenced in `benchmark.py`**. Synthetic bench uses `sim_time = step * dt` with CLI default `duration=30`; PyBullet bench breaks on `sim_time > duration` only. A `--duration 600` run completes without `mark_timed_out`, without `is_timed_out`, and without a threshold failure — contradicting iter-002 claim that the 8-minute cap is enforced end-to-end.
- **Repro**: `grep AIGP_VQ1_MAX_RUN_DURATION_S scripts/benchmark.py` → no matches. Run synthetic with `--duration 600` — loop exits at 600s sim time with no timeout terminal semantics.
- **Fix sketch**: `duration = min(requested, AIGP_VQ1_MAX_RUN_DURATION_S)` in both bench entrypoints; call `seq.mark_timed_out(...)` when `sim_time` (or wall time for synthetic) exceeds cap; add `tests/test_benchmark_adversarial.py::test_bench_marks_timed_out_at_480s`.
- **Confidence**: high

### F2. RaceSession timeout exits without setting sequencer `TIMED_OUT` — [MAJOR]
- **File(s)**: `competition/session.py:168-171`, `race_pipeline.py:380-393`, `gate_sequencing/sequencer.py:184-198`
- **Issue**: `RaceSession._race_loop` breaks when `metrics.elapsed_s >= 480` **before** invoking `on_telemetry`. `mark_timed_out()` only runs inside `_control_callback` when `elapsed > 480` on a callback tick. If the last callback fires at t≈479.99s, the session exits on the next loop iteration without ever calling `mark_timed_out` — `is_timed_out` stays false, honesty/metrics consumers see a silent wall-clock stop, not `RaceState.TIMED_OUT`. Comment at `race_pipeline.py:371` ("set by RaceSession or the bench") is inaccurate for both.
- **Repro**: Mock session with `should_stop` including `is_timed_out`; drive loop to 480s with sparse callbacks — session finishes, `seq.is_timed_out` is false.
- **Fix sketch**: Call `sequencer.mark_timed_out` from `RaceSession` on timeout break, or check timeout inside `_control_callback` **before** replan/tracker and `return` hover on the same tick; unify on sim-time if the sim can run slower than wall clock.
- **Confidence**: high

### F3. Timeout tick continues replanning and tracking after `mark_timed_out` — [MAJOR]
- **File(s)**: `race_pipeline.py:373-378,380-457`
- **Issue**: On the first tick where `elapsed > AIGP_VQ1_MAX_RUN_DURATION_S`, the pipeline calls `mark_timed_out` at lines 384-386 but does **not** return hover until the **next** callback (lines 373-378). That tick still runs `_maybe_replan`, state prediction, trajectory tracking, and may send a non-hover attitude via `session.run()`.
- **Repro**: Instrument `_control_callback`; force `elapsed = 481` on one invocation — observe replan/tracker paths execute after `mark_timed_out`.
- **Fix sketch**: After `mark_timed_out`, immediately `return AttitudeCommand(0, 0, yaw, 0.4)` in the same branch (mirror DQ/crash handling at lines 363-369).
- **Confidence**: high

### F4. Calibration accepts sign-flipped data with large positive `k_t` — [MAJOR]
- **File(s)**: `competition/calibration.py:109-119`, `tests/test_calibration.py:22-49`
- **Issue**: Positivity guard only rejects `k_t <= 0`. Feeding samples generated with the **old wrong** physics (`a = -k_t·u - k_d·v - g`) recovers `k_t ≈ 65.2` with `rmse ≈ 1.77` and no exception — a competition calibration run with inverted thrust convention would silently poison `drone_calibration.json`.
- **Repro**:
  ```python
  # three hover-ish samples, wrong-sign synth
  DroneCalibrator().identify_thrust_drag_ratios([...])  # → k_t=65.24, no raise
  ```
- **Fix sketch**: Reject fits with `rmse` above a ceiling, thrust span below ε, or `k_t` outside plausible band (e.g. 5–50 m/s²); add `test_mismatched_physics_raises_or_high_rmse`.
- **Confidence**: high

### F5. `GatePnPEstimator()` camera defaults remain 640×480 — [MAJOR] (iter-001 partial HARD MISS)
- **File(s)**: `estimation/gate_pnp.py:148-154`, `8214b9c` commit message
- **Issue**: Iter-002e fixed `GateGeometry` to 1.5 m AIGP but left `GatePnPEstimator.__init__` default camera as `CameraIntrinsics.from_fov(90.0, 640, 480)` → `image_height=480`, `cy=240`. `RacePipeline` overrides via `from_fov` + pitch (lines 134-139), but `estimation/tests/test_gate_pnp.py` and any tool using bare `GatePnPEstimator()` still scale PnP with legacy geometry. Competition bridge now decodes 360p frames (`vision_udp.py:283`) into a pipeline that builds 360 intrinsics — secondary call sites remain inconsistent.
- **Repro**: `GatePnPEstimator().camera.image_height` → `480`; `gate.interior_width_m` → `1.5` (split defaults).
- **Fix sketch**: Default `camera or CameraIntrinsics()` (AIGP module defaults); update tests that need 480 to pass explicit `from_fov(..., 480)`.
- **Confidence**: high

### F6. Phase1 detector still defaults to 480p — [MAJOR]
- **File(s)**: `gate_detection/src/phase1_detector.py:63-65`, `race_pipeline.py:152-157`
- **Issue**: `RacePipeline` constructs `Phase1GateDetector` with `image_height=self.config.image_height` (360), but the detector class default is still `image_height=480`. Any caller that omits dimensions (tests, scripts, future wiring) runs steering/PnP geometry at the wrong aspect. Detector internals (FOV-based sizing) are tied to `image_width`/`image_height` — mismatch with 360 frames skews gate center estimates.
- **Repro**: `Phase1GateDetector()` → `image_height == 480` while `PipelineConfig.image_height == 360`.
- **Fix sketch**: Default detector dims from `AIGP_CAM_HEIGHT_PX` / `AIGP_CAM_WIDTH_PX`; grep `480` in `gate_detection/` for stragglers.
- **Confidence**: high

### F7. Wall-clock VQ1 timeout vs simulation time — [MAJOR]
- **File(s)**: `race_pipeline.py:296,380-386,392-393`
- **Issue**: Timeout uses `time.monotonic() - self._race_start_time` (wall clock). DCL sim may run slower/faster than real time or pause for debugging; wall-clock timeout can fire early/late relative to **sim** time the competition scores. `_maybe_replan` also uses the same wall-clock as `sim_time` (line 392), coupling replan cooldown to wall clock on the MAVLink path.
- **Repro**: Run sim at 0.5× real-time for 9 wall-clock minutes with 8 sim-minute cap — false timeout. Paused sim: wall clock advances, sim time frozen — false timeout.
- **Fix sketch**: Thread `TelemetryState` / sim timestamp from bridge; compare against sim-time budget; fall back to monotonic only when sim clock unavailable.
- **Confidence**: medium (depends on DCL exposing sim time; wall clock may be intentional for live VQ1)

### F8. `pop_latest_frame` does not pop; redundant JPEG decode every poll — [MINOR]
- **File(s)**: `competition/vision_udp.py:257-259,400-410`, `competition/mavlink_bridge.py:189-198`
- **Issue**: `pop_latest_frame()` returns `self._latest_frame` without clearing it. `VisionUdpListener.latest_frame()` decodes on **every** call via `decode_jpeg_to_camera_frame`. A 100 Hz control loop re-decodes the same JPEG until a new frame completes — wasted CPU on the competition hot path (brief §73-75).
- **Repro**: Call `latest_frame()` twice without new UDP packets — two `cv2.imdecode` passes, same `frame_id`.
- **Fix sketch**: Clear `_latest_frame` after handoff, or cache last decoded `(frame_id, CameraFrame)`; rename to `peek_latest_frame` if non-destructive semantics are intentional.
- **Confidence**: high

### F9. `VisionUdpListener.start()` not idempotent — transport leak — [MINOR]
- **File(s)**: `competition/vision_udp.py:376-383`, `tests/test_vision_udp_listener.py:152-165`
- **Issue**: `test_listener_lifecycle_idempotent_stop` covers double `stop()`, not double `start()`. A second `await start()` without `stop()` allocates a new datagram endpoint via `create_datagram_endpoint` while `_transport` is overwritten — prior socket may leak; port-in-use on second bind.
- **Repro**: `await listener.start(); await listener.start()` on same instance.
- **Fix sketch**: Guard `if self.is_listening: return` at top of `start()`; test double-start is harmless.
- **Confidence**: medium-high

### F10. `read_calibration_json` has no schema validation — [MINOR]
- **File(s)**: `competition/calibration.py:152-167`
- **Issue**: Loads arbitrary JSON keys into `CalibrationResult` with no range checks. Garbage `thrust_per_mass_1_per_s2: -999` or missing keys (KeyError) can crash startup or pass negative thrust into planner config.
- **Repro**: Write JSON with negative thrust; `read_calibration_json` succeeds.
- **Fix sketch**: Validate keys, positivity of `k_t`, finite `rmse`, optional version field; raise `ValueError` with path context.
- **Confidence**: high

### F11. Synthetic vs PyBullet `pass_through_margin` drift — [MINOR]
- **File(s)**: `scripts/benchmark.py:317-319`, `gate_sequencing/sequencer.py:78`, `sim_pybullet/runner.py:183`
- **Issue**: Synthetic bench hard-codes `pass_through_margin=1.5`; PyBullet bench uses `SequencerConfig` default `1.0` unless `race_config.sequencer_overrides` patches it. DQ now uses strict `crash_margin` (F14 fix — good), but **pass crediting** and multi-gate drain still use lenient margin — bench platforms disagree on how wide a crossing counts as a pass.
- **Repro**: Same trajectory on synthetic (1.5) vs sim (1.0) — different `gates_passed` for borderline approaches.
- **Fix sketch**: Document intentional difference or align both to 1.0 for honesty; keep 1.5 only in `sim_pybullet/runner` where documented.
- **Confidence**: high

### F12. `enforce_in_order` still false-DQs legitimate replanner recovery — [MAJOR] (deferred, now production-critical)
- **File(s)**: `gate_sequencing/sequencer.py:413-441`, `race_pipeline.py:459-497`, `planning/dynamic_replanner.py`
- **Issue**: Deferred from iter-001; iter-002 added **stricter** future-gate handling (F3/F14) without replan exemption. After a miss, `_maybe_replan` can rebuild a path whose segment crosses a future gate's strict opening — terminal DQ, not recoverable miss. Dynamic replanner is active on the MAVLink path (lines 388-393).
- **Repro**: Miss gate N, replan arc through gate N+2 opening at y=0.5 m while `current_idx` still N — `out_of_order` on first crossing.
- **Fix sketch**: Temporarily relax `enforce_in_order` during replan cooldown, or DQ only for gates with `sequence_index < current + 2`; adversarial test with `DynamicReplanner` + sequencer integrated.
- **Confidence**: medium-high

### F13. B5 pitch threaded to intrinsics but not consumed in PnP world transform — [MINOR] (known deferral)
- **File(s)**: `race_pipeline.py:134-139`, `estimation/gate_pnp.py:223-267`, `tests/test_camera_geometry.py:129+`
- **Issue**: `camera.pitch_offset_rad` is set on the pipeline intrinsics object; `gate_pose_to_drone_position` still maps `camera_in_gate` with `R_gate_world @ camera_in_gate` only — no `R_pitch(pitch_offset_rad)`, `drone_orientation` unused. Acceptable for position-only PnP with coincident origins (brief §33), but any orientation-from-gate work will be wrong; test only covers intrinsics field, not transform.
- **Repro**: `gate_pose_to_drone_position(...)` invariant to `drone_orientation` and to `pitch_offset_rad` on intrinsics.
- **Fix sketch**: Apply body←camera rotation in transform; regression test with predictable Δ at 5 m range.
- **Confidence**: high (intentional partial fix)

### F14. `_delivered_ids` sliding window still allows duplicate frame re-emit — [MINOR] (deferred)
- **File(s)**: `competition/vision_udp.py:248-253,180-181`
- **Issue**: Cap `max_buffered_frames * 8` (64 with default buffer 8). After 64 newer `frame_id`s, an old ID falls off `_delivered_ids` and a late duplicate chunk can re-complete and update `_latest_frame` — downstream may see a stale image twice.
- **Repro**: Deliver frames 1..70; replay chunk for frame 1 — may not increment `dropped_late_packets`.
- **Fix sketch**: Use bounded `collections.deque` + `set` for O(1) membership; or monotonic epoch in frame_id if encoder guarantees it.
- **Confidence**: medium

### F15. Multi-gate drain does not record second strut crash in same segment — [NIT]
- **File(s)**: `gate_sequencing/sequencer.py:378-385,447-457`
- **Issue**: Crash dedupe uses `if self._last_event != "crash"` (line 379) inside the drain loop, so only one crash entry per tick from drain. Future-gate loop uses a looser condition (lines 448-451). Unlikely in practice but differs from `mark_collision` idempotency semantics.
- **Repro**: Construct segment grazing struts on g2 and g3 in one drain iteration after g1 pass — only one crash recorded.
- **Fix sketch**: Append per-gate if `not any(c[0]==gid for c in self._crashes)` instead of `_last_event` gate.
- **Confidence**: low

## Iter-002 punch-list grading (iter-001 → iter-002)

| Item | Verdict | Notes |
|------|---------|-------|
| B1 calibration physics | **FIXED** | `y = gravity - a`, hover test `test_hover_only_samples_recover_g_over_u` |
| B2 pipeline DQ/crash stop | **FIXED** | `should_stop` + early-return at `race_pipeline.py:287-293,363-369` |
| B3 CameraFrame 360 | **FIXED** | `competition/adapter.py:150-151` |
| B4 vision UDP wiring | **FIXED** | `mavlink_bridge.py:198`, listener tests |
| B5 camera pitch | **PARTIAL** | Intrinsics only (`race_pipeline.py:139`); PnP transform unchanged |
| Opus F2 multi-gate/tick | **FIXED** | Drain loop `sequencer.py:349-393` + adversarial tests |
| Opus F3 future strut | **FIXED** | `sequencer.py:442-457` |
| Opus F14 strict DQ opening | **FIXED** | `crash_margin` at `sequencer.py:433-435` |
| Opus F9 jpeg_size | **FIXED** | `vision_udp.py:234-238` |
| Opus F5 ILC relative threshold | **FIXED** | `ilc_sections.py:113-115` |
| 8-min VQ1 timeout | **PARTIAL** | Pipeline marks timeout; bench/session omit; off-by-one tick |
| HFoV/VFoV | **FIXED** | `aigp_geometry.py:45-58` documented |
| GateGeometry 1.5 m | **FIXED** | `gate_pnp.py:115-116` |
| Bench crashed vs DQ | **FIXED** | `benchmark.py:451-454` + adversarial test |
| ILC `_pt_to_step` rounding | **FIXED** | `ilc_sections.py:140-143` |

## Things iter-002 got right
- Calibration regression matches NED hover physics; hover-anchored test would have caught the iter-001 sign bug.
- Sequencer multi-gate drain + future-gate strut classification are substantive behaviour changes with dedicated adversarial tests.
- Vision path is no longer dead: real UDP listener, E2E socket test, `get_camera_frame()` returns decoded frames.
- DQ vs crash vs timeout are separate terminal signals on the pipeline and bench honesty surface.
- ILC relative threshold fixes bimodal "all low" collapse on long straights + one turn.

## What I did NOT review
- Full `git show` diff hunks for all six commits (spot-checked files + commit messages).
- `competition/pybullet_adapter.py` camera path, `sim_pybullet/` physics, `runner.py` collapse.
- `control/learned_residual.py` / MLP training artefacts.
- `config/aigp_default.json` geometry fidelity (Opus F7 deferred).
- MAVSDK command-path tests (A12 still absent).
- Performance profiling of ILC / trajectory optimizer at 100 Hz.
