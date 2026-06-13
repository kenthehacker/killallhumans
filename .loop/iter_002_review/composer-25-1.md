# Iter 002 Adversarial Review — Composer 2.5

## Summary

Iter-002 closes most of the iter-001 BLOCKER/MAJOR punch list on paper: calibration physics, pipeline terminal stops, vision→MAVLink wiring, multi-gate crediting, future-gate struts, strict-opening DQ, jpeg_size validation, ILC relative threshold, and adversarial sequencer tests are real improvements. The stack is still not competition-honest end-to-end: PnP still ignores camera tilt in the world transform, standalone `GatePnPEstimator()` defaults remain legacy 640×480, the 8-minute cap is wall-clock-only and never exercised by the bench, and calibration accepts physically absurd fits as long as `k_t > 0`. Treat the 334 passing tests as necessary, not sufficient.

## Findings (ordered by severity)

### F1. Calibration rejects only `k_t ≤ 0`; wrong-physics data can still pass with a large positive bias — [MAJOR]
- **File(s)**: `competition/calibration.py:109-119`, `competition/calibration.py:152-167`, `tests/test_calibration.py:52-91`
- **Issue**: **PARTIAL CLOSE** on iter-001 B1 (Opus F1). The regression equation is now correct (`y = gravity - a`, `X = [u, v]`), and `test_hover_only_samples_recover_g_over_u` anchors hover at ≈21.8. The guard is only `k_t <= 0.0` with no RMSE ceiling or rank check. Feeding samples generated under the *old* wrong convention (`a = -k·u - k_d·v - g`) yields `y = 2g + k·u + …`, which lstsq can fit with a **large positive** `k_t` and low RMSE — the positivity check does not catch convention drift. `read_calibration_json` accepts arbitrary positive floats with no schema or sanity bounds.
- **Repro**: Build 200 samples with `accel_z = -22*u - 0.4*v - 9.81` (old wrong synth). `identify_thrust_drag_ratios` returns `thrust_per_mass >> 22` with small RMSE and no error. Write JSON with `thrust_per_mass_1_per_s2: 1e6`; reload succeeds.
- **Fix sketch**: Assert `rmse < 0.5` (tunable) on identify; reject `k_t` outside a plausible band (e.g. 5–80 1/s²); validate JSON on read. Add `test_wrong_convention_samples_fail_or_high_rmse`.
- **Confidence**: high

### F2. Camera pitch is threaded into intrinsics but never applied in PnP→world — [MAJOR] (iter-001 B5 **PARTIAL**)
- **File(s)**: `race_pipeline.py:134-139`, `estimation/gate_pnp.py:250-267`, `tests/test_camera_geometry.py:129-145`
- **Issue**: Iter-002 B5 wired `PipelineConfig.camera_pitch_offset_rad` → `CameraIntrinsics.pitch_offset_rad` (good). `gate_pose_to_drone_position` still maps `camera_in_gate` with `R_gate_world @ camera_in_gate` only — no `R_body_camera(pitch_offset_rad)`. The horizon test proves trigonometry on intrinsics, not the production code path. EKF PnP updates remain biased for the AIGP +20° mount.
- **Repro**: `GatePnPEstimator(camera=CameraIntrinsics(pitch_offset_rad=math.radians(20)))` vs `0.0` on the same synthetic pose — recovered `drone_world` is identical today.
- **Fix sketch**: Apply body↔camera rotation in `gate_pose_to_drone_position`; regression where ±20° pitch shifts world position by >0.1 m at 5 m range.
- **Confidence**: high

### F3. `GatePnPEstimator()` standalone defaults still legacy 640×480 — [MAJOR] (iter-001 MAJOR **HARD MISS**)
- **File(s)**: `estimation/gate_pnp.py:153-154`, `estimation/gate_pnp.py:115-116`
- **Issue**: `GateGeometry()` now defaults to 1.5 m (iter-002e **CLOSED**). `GatePnPEstimator.__init__` still uses `CameraIntrinsics.from_fov(90.0, 640, 480)` when `camera is None`. Any tool/test constructing a bare estimator gets wrong focal length and principal point for AIGP 360p.
- **Repro**: `GatePnPEstimator()` → `estimator.camera.image_height == 480`, `cy == 240`.
- **Fix sketch**: Default `camera or CameraIntrinsics()` (AIGP dataclass defaults). Update legacy tests to pass explicit 640×480.
- **Confidence**: high

### F4. 8-minute VQ1 cap wired in `RacePipeline` but not in bench or `RaceSession` sequencer state — [MAJOR] (iter-001 8-min item **PARTIAL**)
- **File(s)**: `race_pipeline.py:287-293`, `race_pipeline.py:380-386`, `competition/session.py:36`, `competition/session.py:168-171`, `scripts/benchmark.py` (no `AIGP_VQ1` / `mark_timed_out` references)
- **Issue**: `should_stop` and `_control_callback` honour `is_timed_out` and call `mark_timed_out` on wall-clock overrun (**CLOSED** for live pipeline path). `RaceSession._race_loop` uses a duplicate `MAX_RUN_DURATION_S = 480` and **breaks without** calling `sequencer.mark_timed_out()`, so a session stopped only by the session loop may exit with `is_timed_out == False`. Synthetic/PyBullet benches use `sim_time = step * dt` (or env sim time), never call `mark_timed_out`, and default `--duration 30` — the 480 s contract is untested and invisible in bench JSON.
- **Repro**: `grep mark_timed_out scripts/benchmark.py` → empty. Run synthetic 30 s PASS — says nothing about an 8-minute competition run.
- **Fix sketch**: Import `AIGP_VQ1_MAX_RUN_DURATION_S` in bench; call `seq.mark_timed_out` when `sim_time` exceeds cap; have `RaceSession` invoke a pipeline/sequencer timeout hook; adversarial test at 481 s sim time.
- **Confidence**: high

### F5. Wall-clock timeout can false-DQ a slowed or paused sim — [MAJOR]
- **File(s)**: `race_pipeline.py:296`, `race_pipeline.py:380-386`, `race_pipeline.py:392`
- **Issue**: `_race_start_time` and timeout use `time.monotonic()` (wall clock). `_maybe_replan` also passes wall elapsed as `sim_time`. A debug run with slowed sim time or a paused debugger will hit `vq1_max_run_duration_exceeded` while the vehicle has flown far less than 8 minutes of *sim* time. Competition may be real-time; the bench is not, so metrics diverge.
- **Repro**: Hold breakpoint in `_control_callback` for >480 s wall time with sim frozen — next tick marks timeout.
- **Fix sketch**: Thread competition/sim timestamp from telemetry or `CameraFrame.timestamp_us`; timeout and replanner cooldown on sim clock when available.
- **Confidence**: medium (competition may be wall-clock-only; bench mismatch is certain)

### F6. Synthetic bench `pass_through_margin=1.5` vs PyBullet default `1.0` — platform honesty drift — [MAJOR]
- **File(s)**: `scripts/benchmark.py:317-318`, `gate_sequencing/sequencer.py:78`, `gate_sequencing/tests/test_sequencer_adversarial.py:322-355`
- **Issue**: Synthetic `run_synthetic_benchmark` hard-codes `SequencerConfig(pass_through_margin=1.5)`. PyBullet path builds `SequencerConfig` from `race_config.sequencer_overrides` only (default margin 1.0). Pass/crash/DQ geometry therefore differs between the two bench surfaces that gate iter-002 merges — green synthetic can mask PyBullet failures (or vice versa) on tight race_01 frames.
- **Repro**: Compare `seq.config.pass_through_margin` in synthetic vs sim for default `race_01.json`.
- **Fix sketch**: Single default from track config or AIGP constant; document intentional override per track; matrix test both margins.
- **Confidence**: high

### F7. `VisionUdpListener.start()` is not idempotent — double bind leaks transport — [MINOR]
- **File(s)**: `competition/vision_udp.py:376-383`, `tests/test_vision_udp_listener.py:152-167`
- **Issue**: `stop()` is idempotent (tested). `start()` always calls `create_datagram_endpoint` with no guard if `_transport` is already set. A reconnect path that calls `start()` twice without `stop()` orphans the first socket.
- **Repro**: `await listener.start(); await listener.start()` — second call without closing first transport.
- **Fix sketch**: If `is_listening`, return early or `stop()` then `start()`; test double-start.
- **Confidence**: high

### F8. `latest_frame()` / `pop_latest_frame()` re-decode the same JPEG every poll — [MINOR]
- **File(s)**: `competition/vision_udp.py:257-259`, `competition/vision_udp.py:400-410`, `competition/mavlink_bridge.py:189-198`
- **Issue**: `pop_latest_frame()` returns `_latest_frame` without clearing it (name says "pop", behaviour is "peek"). `VisionUdpListener.latest_frame()` calls `decode_jpeg_to_camera_frame` on every 100 Hz control tick until a new frame completes — wasted CPU and stale frames presented as fresh.
- **Repro**: Complete one frame; call `latest_frame()` 1000 times — 1000 `cv2.imdecode` calls, same `frame_id`.
- **Fix sketch**: Cache decoded `CameraFrame` on completion; clear or version-stamp on consume; rename `peek_latest_frame`.
- **Confidence**: high

### F9. Future-gate DQ/crash loop processes at most one future gate per tick — [MINOR]
- **File(s)**: `gate_sequencing/sequencer.py:419-457`
- **Issue**: After the multi-gate drain on the current target, the `for future_gate in self._gates[self._current_idx + 1:]` loop `break`s after the first future gate that triggers strict-opening DQ or outer-frame crash. A single segment crossing two *future* openings (e.g. g4 and g5 while current is g2) records only the earliest index in the list, not both violations.
- **Repro**: Construct segment crossing g4 and g5 openings with `current_idx` at g2; inspect `_dq_reason` / `_crashes` length.
- **Fix sketch**: Continue scanning remaining future gates (or collect all violations, prefer earliest for DQ semantics).
- **Confidence**: medium (may match competition "first violation wins")

### F10. `enforce_in_order=True` still risks false-DQ on replanner recovery — [MINOR] (iter-001 deferred, now sharper)
- **File(s)**: `gate_sequencing/sequencer.py:413-441`, `race_pipeline.py:388-437`, `planning/dynamic_replanner.py`
- **Issue**: Deferred in iter-001, still present. After a miss, `DynamicReplanner` + `RECOVERY` can fly a path that re-crosses a future gate's **strict opening** legitimately. Sequencer DQ fires on first such crossing; no adversarial test encodes "recovery then re-attempt without DQ".
- **Repro**: Miss g2, replan arc through g4 opening before g2 re-pass — terminal DQ despite recoverable intent.
- **Fix sketch**: DQ only on gates *ahead of* credited progress by more than one index, or whitelist recovery/replan windows; add recovery adversarial test.
- **Confidence**: medium

### F11. `_delivered_ids` sliding window still allows duplicate frame re-emit — [MINOR] (iter-001 deferred)
- **File(s)**: `competition/vision_udp.py:248-253`, `competition/vision_udp.py:181`
- **Issue**: Cap is `max_buffered_frames * 8` (64 with default 8). After ~64 new `frame_id`s, an old ID falls off `_delivered_ids` and a late duplicate chunk can re-complete and overwrite `_latest_frame` — downstream may see a time-travel image.
- **Repro**: Deliver frame_id=1, then 65 distinct IDs, then replay chunks for frame_id=1.
- **Fix sketch**: Use a bounded `deque` + `set` or ring bitmap of recent IDs sized for 30 fps × reassembly timeout at 2× margin.
- **Confidence**: medium

### F12. MAVLink vision socket opens only after connect succeeds; hang defers vision — [NIT]
- **File(s)**: `competition/mavlink_bridge.py:98-121`, `competition/mavlink_bridge.py:137-147`
- **Issue**: **B4 CLOSED** for happy path — `get_camera_frame()` returns listener decode. If `connect()` blocks on `connection_state`, vision never binds. If `connect()` raises after partial setup, `disconnect()` still stops listener (idempotent). Acceptable but worth documenting for competition startup SLO.
- **Repro**: Block MAVSDK connection; observe no UDP :5600 listener.
- **Fix sketch**: Optional parallel vision start with ring buffer; or document "vision requires connected session".
- **Confidence**: high

## Iter-001 punch-list grade (iter-002 scope)

| Item | Grade | Evidence |
|------|-------|----------|
| B1 calibration physics | **CLOSED** (equation); **PARTIAL** (bad-fit guard) | `calibration.py:102-119`, `test_hover_only_samples_recover_g_over_u` |
| B2 pipeline DQ/crash stop | **CLOSED** | `race_pipeline.py:287-293`, `363-369` |
| B3 CameraFrame 360 | **CLOSED** | `competition/adapter.py:150-151` |
| B4 vision UDP wiring | **CLOSED** | `mavlink_bridge.py:189-198`, `test_vision_udp_listener.py` |
| B5 camera pitch | **PARTIAL** | Wired `race_pipeline.py:139`; not used `gate_pnp.py:250-267` |
| Opus F2 multi-gate/tick | **CLOSED** | `sequencer.py:349-393`, `test_segment_crossing_two_gates_credits_both` |
| Opus F3 future strut | **CLOSED** | `sequencer.py:442-457`, `test_future_gate_strut_hit_*` |
| Opus F14 strict DQ opening | **CLOSED** | `sequencer.py:433-435`, `test_dq_uses_strict_opening_when_pass_through_margin_is_lenient` |
| Opus F9 jpeg_size | **CLOSED** | `vision_udp.py:234-238` |
| Opus F5 ILC relative threshold | **CLOSED** | `planning/ilc_sections.py:113-115` |
| 8-min VQ1 timeout | **PARTIAL** | Pipeline yes; bench/session no |
| HFoV/VFoV | **CLOSED** (documented) | `competition/aigp_geometry.py:45-58` |
| GateGeometry 1.2 m default | **CLOSED** | `gate_pnp.py:115-116` |
| Bench crashed vs DQ | **CLOSED** | `scripts/benchmark.py:444-454`, `792-795` |
| Opus F6 ILC step rounding | **CLOSED** | `planning/ilc_sections.py:140-143` |

## Things iter-002 got right

- Calibration regression matches NED hover physics; hover-only test would have caught the original sign bug.
- Sequencer multi-gate drain and strict-opening DQ are implemented with dedicated adversarial tests, not just comments.
- Future-gate strut hits are no longer silent; crash vs DQ semantics are separated on margin.
- Vision UDP is reachable from `MAVLinkBridge.get_camera_frame()` with real-socket listener tests.
- ILC bimodal curvature partitioning uses a relative accel floor instead of the useless 1e-6 absolute collapse.
- Bench honesty split: `crashed` vs `disqualified` are distinct terminal signals in synthetic and sim paths.

## What I did NOT review

- Full `git show` diff hunks for all six commits (read current file state + spot checks only).
- PyBullet physics / `sim_pybullet/` collision manifolds in depth.
- `control/learned_residual.py` MLP paths and clamp ordering (assumed unchanged from iter-001 consensus).
- MAVSDK command integration tests (still deferred A12).
- `config/aigp_default.json` true 1.25× scale claim (deferred; not re-measured).
- DCL binary / live MAVLink flight (no binary in worktree).
- Entire `gate_detection/` CNN runtime and training stack.
