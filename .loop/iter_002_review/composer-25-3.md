# Iter 002 Adversarial Review — Composer 2.5

## Summary

Iter-002 closes most of the iter-001 punch list in unit tests (calibration physics, sequencer DQ/crash/timeout state machine, vision reassembly, multi-gate drain, strict-opening DQ, ILC relative floor, mavlink vision wiring). The green test matrix is believable for those modules. Two regressions undermine “production-ready” claims: **(1)** `RacePipeline` sets `_race_start_time` with `time.time()` but compares it to `time.monotonic()`, so the new 8-minute `mark_timed_out()` path never fires; **(2)** calibration’s positivity guard accepts grossly wrong thrust conventions with near-zero RMSE. Vision wiring is live but `latest_frame()` re-decodes the same JPEG on every poll and `start()` can leak transports on re-bind.

## Iter-001 punch-list grade (iter-002 closure)

| Item | Verdict | Evidence |
|------|---------|----------|
| B1 calibration physics | **CLOSED** | `competition/calibration.py:102-104` uses `y = gravity - a`, `X = [u,v]`; `tests/test_calibration.py:62-91` hover case ≈ g/u ≈ 21.8 |
| B2 DQ/crash termination | **CLOSED** (session); **PARTIAL** (pipeline timeout) | `race_pipeline.py:288-293` `should_stop`; `359-369` early-return on DQ/crash; timeout broken (F1) |
| B3 CameraFrame 360 default | **CLOSED** | `competition/adapter.py:150-151` |
| B4 Vision UDP wiring | **CLOSED** | `competition/mavlink_bridge.py:189-198` delegates to listener |
| B5 Camera tilt wiring | **PARTIAL** | `race_pipeline.py:134-139` sets `pitch_offset_rad`; `gate_pose_to_drone_position` at `estimation/gate_pnp.py:223-266` never applies it (F5) |
| Opus F2 multi-gate/tick | **CLOSED** | `gate_sequencing/sequencer.py:349-393`; tests `test_segment_crossing_two_gates_credits_both` |
| Opus F3 future strut crash | **CLOSED** | `sequencer.py:442-457` |
| Opus F14 strict DQ opening | **CLOSED** | `sequencer.py:433-435` uses `crash_margin` |
| Opus F9 jpeg_size validation | **CLOSED** | `vision_udp.py:234-238`; `tests/test_vision_udp.py:101-148` |
| Opus F5 ILC relative threshold | **CLOSED** | `planning/ilc_sections.py:113-115` |
| 8-min VQ1 timeout | **PARTIAL** | `RaceSession` enforces 480s (`competition/session.py:169`); pipeline `mark_timed_out` dead (F1) |
| HFoV/VFoV | **CLOSED** (documented) | `estimation/gate_pnp.py:58-62` trusts fx=fy=320 / HFoV 90° |
| GateGeometry 1.5 m default | **CLOSED** | `estimation/gate_pnp.py:115-116` |
| Bench crashed≠DQ | **CLOSED** | `scripts/benchmark.py:451-455`; `tests/test_benchmark_adversarial.py:155-198` |
| ILC pt_to_step rounding | **CLOSED** | `planning/ilc_sections.py:140-143` |

## Findings (ordered by severity)

### F1. RacePipeline 8-minute timeout never trips — clock mismatch — [BLOCKER]
- **File(s)**: `race_pipeline.py:296`, `race_pipeline.py:382-386`
- **Issue**: `run()` stores `_race_start_time = time.time()` (Unix epoch), but `_control_callback` computes `elapsed = time.monotonic() - self._race_start_time`. On this machine the difference is ~−1.78×10⁹ s, so `elapsed > AIGP_VQ1_MAX_RUN_DURATION_S` is always false. `mark_timed_out()` is never called from the pipeline; `is_timed_out` early-return at line 373 is dead in live runs. `RaceSession` still stops at 480s via `metrics.elapsed_s` (`competition/session.py:169`), but the sequencer never enters `RaceState.TIMED_OUT`, metrics/replan paths that key off `is_timed_out` are inconsistent, and any consumer of sequencer state alone misreports timeout.
- **Repro**: `python3 -c "import time; print(time.monotonic()-time.time() > 480)"` → `False`. Run `RacePipeline.run()` for >8 min: session exits, `seq.is_timed_out` stays false.
- **Fix sketch**: Use one clock: `_race_start_time = time.monotonic()` everywhere, or pass sim-time from the interface. Add integration test that advances a fake clock >480s and asserts `mark_timed_out` + `is_timed_out`.
- **Confidence**: high

### F2. Calibration positivity guard misses wrong-physics fits — [MAJOR]
- **File(s)**: `competition/calibration.py:109-119`, `tests/test_calibration.py:52-91`
- **Issue**: Guard only rejects `k_t <= 0`. Synthetic samples with the *old* wrong sign (`a = -g - k_t·u`) still yield **positive** `k_t ≈ 65.6` with RMSE ≈ 0 — the fit absorbs the 2g bias into a bogus thrust scale. Hover test catches the original sign-flip bug but not convention swaps that stay positive.
- **Repro**: 100 hover samples with `accel_z = -9.81 - 22*u`; `identify_thrust_drag_ratios` succeeds with `thrust_per_mass ≈ 65.6`, `rmse ≈ 0`.
- **Fix sketch**: Assert `rmse` below a tight bound on clean synth; bound `k_t` to a plausible band (e.g. 5–80 1/s²); optional rank check on `X`; adversarial test with deliberately mis-signed feeder that must raise.
- **Confidence**: high

### F3. `VisionUdpListener.start()` can leak UDP transport — [MAJOR]
- **File(s)**: `competition/vision_udp.py:376-383`, `competition/vision_udp.py:385-390`
- **Issue**: `start()` always calls `create_datagram_endpoint` without closing an existing `_transport`. A second `start()` orphans the first socket (port still bound or fd leak). `stop()` only closes the latest transport.
- **Repro**: `await listener.start(); await listener.start()` — two endpoints; first never closed.
- **Fix sketch**: If `self._transport is not None`, return early or `await stop()` first. Test double-start is idempotent.
- **Confidence**: high

### F4. `latest_frame()` re-decodes the same JPEG every control tick — [MAJOR]
- **File(s)**: `competition/vision_udp.py:254-255`, `competition/vision_udp.py:400-410`, `competition/mavlink_bridge.py:198`
- **Issue**: `pop_latest_frame()` returns `_latest_frame` without clearing it. Each 100 Hz `get_camera_frame()` → `latest_frame()` runs `cv2.imdecode` on the same bytes until a new frame completes. Wastes CPU on the competition hot path and can add jitter to the control loop.
- **Repro**: Feed one complete frame; call `latest_frame()` 1000 times; decode runs 1000 times (profile or spy on `cv2.imdecode`).
- **Fix sketch**: Cache decoded `CameraFrame` keyed by `frame_id`; clear slot after handoff; or rename API to document “decode once per frame_id”.
- **Confidence**: high

### F5. Camera pitch wired on intrinsics but unused in PnP→world — [MAJOR]
- **File(s)**: `race_pipeline.py:134-139`, `estimation/gate_pnp.py:223-266`, `estimation/gate_pnp.py:65-70`
- **Issue**: Iter-002 B5 threads `camera_pitch_offset_rad` into `CameraIntrinsics`. `gate_pose_to_drone_position` still maps camera-in-gate → world with gate orientation only; no `R_pitch(pitch_offset_rad)` despite comments at lines 65-68. Position-only PnP with coincident origins may mask this today, but any orientation or offset work inherits a silent no-op.
- **Repro**: Set `pitch_offset_rad=0.5`; PnP world position unchanged vs 0.0 for same corners.
- **Fix sketch**: Apply body/camera rotation in the world transform chain; extend `tests/test_camera_geometry.py` to assert the *pipeline path* moves projected horizon, not just the formula.
- **Confidence**: high (wiring); medium (runtime impact before competition geometry differs)

### F6. Multi-gate drain cannot credit gate N+2 if current target N was not credited — [MINOR]
- **File(s)**: `gate_sequencing/sequencer.py:338-393`, `gate_sequencing/sequencer.py:413-441`
- **Issue**: The drain `while` loop only runs after a successful pass on the *current* gate. A segment that crosses only g2 and g3 openings while `current_idx=0` (g1 plane missed or classified miss/crash) does not drain-credit g2/g3; future-opening DQ should catch g2. No test covers “only next-next opening, current plane not crossed” on non-collinear layouts. Risk: benign miss classification + permissive path could leave phantom skips without DQ on exotic geometry.
- **Repro**: Craft 3-gate layout where one segment intersects g2/g3 openings but not g1’s plane; assert `gates_passed` and DQ/crash semantics.
- **Fix sketch**: Document invariant; add adversarial geometry test; optionally scan all unpassed gates for plane crossings each tick (heavier).
- **Confidence**: medium

### F7. Future-gate handler processes at most one future gate per tick — [MINOR]
- **File(s)**: `gate_sequencing/sequencer.py:419-457`
- **Issue**: Loop `break`s after first future gate with strict-opening DQ or strut crash. A single long segment grazing two future struts records one crash; the second is ignored until a later tick (may be acceptable but differs from multi-gate pass crediting symmetry).
- **Repro**: One update crossing struts on g2 and g3 while current is g1; only one `_crashes` entry from future branch.
- **Fix sketch**: Continue scanning without `break`, or document single-event-per-tick policy; align dedupe with P1-7 semantics.
- **Confidence**: medium

### F8. Synthetic bench `pass_through_margin=1.5` vs PyBullet default 1.0 — platform drift — [MINOR]
- **File(s)**: `scripts/benchmark.py:317-319`, `scripts/benchmark.py:718-722`, `gate_sequencing/sequencer.py:78`
- **Issue**: Synthetic path hard-codes lenient 1.5; PyBullet uses `SequencerConfig()` defaults (1.0) unless JSON overrides. Pass detection and miss/crash margins differ between bench modes even though DQ now uses strict `crash_margin` (F14 fix). Operators comparing synthetic vs sim gate-pass rates compare different geometric tolerances.
- **Repro**: Same trajectory on synthetic vs sim configs; different `gates_passed` near marginal openings.
- **Fix sketch**: Single default in `aigp_geometry.py` or race config; synthetic uses same `seq_cfg` as PyBullet branch.
- **Confidence**: high

### F9. Synthetic/PyBullet bench never applies 8-minute cap to sim time — [MINOR]
- **File(s)**: `scripts/benchmark.py:422-424`, `scripts/benchmark.py:767-768`, `competition/aigp_geometry.py:78`
- **Issue**: No `AIGP_VQ1_MAX_RUN_DURATION_S` in `benchmark.py`. Loops stop on `duration` CLI (default 30s). Contract for “VQ1 timeout wired” is only true in `RaceSession` / broken pipeline path, not in self-honesty benches.
- **Repro**: `grep AIGP_VQ1_MAX scripts/benchmark.py` → empty.
- **Fix sketch**: `n_steps = min(int(duration/dt), int(480/dt))`; optional adversarial test with tiny duration cap.
- **Confidence**: high

### F10. `read_calibration_json` accepts arbitrary garbage — [MINOR]
- **File(s)**: `competition/calibration.py:153-167`
- **Issue**: No range checks, required keys only via `float()` coercion. Negative `thrust_per_mass_1_per_s2` or NaN loads into planner/controller without error.
- **Repro**: Write JSON with `"thrust_per_mass_1_per_s2": -1`; `read_calibration_json` succeeds.
- **Fix sketch**: Validate positivity and plausible bounds on read; mirror identify-time guard.
- **Confidence**: high

### F11. `_race_start_time` reused as broken replan clock — [MINOR]
- **File(s)**: `race_pipeline.py:392-393`
- **Issue**: `_maybe_replan(sim_time, ...)` uses the same `monotonic() - time.time()` value as F1 — negative and meaningless. Replanner cooldown/progress logic does not see real elapsed race time.
- **Repro**: Log `sim_time` passed to replanner during `run()` — large negative.
- **Fix sketch**: Fix F1; use monotonic delta or telemetry timestamp.
- **Confidence**: high (same root cause as F1)

### F12. `error_received` counts but never recovers stream — [NIT]
- **File(s)**: `competition/vision_udp.py:340-341`
- **Issue**: OS-level UDP errors increment `errors` only; no listener restart or backpressure signal to pipeline.
- **Repro**: Inject `error_received` on protocol; stream stays on dead transport.
- **Fix sketch**: Optional `stop()`+`start()` on sustained errors; expose metric to logs.
- **Confidence**: low (environment-dependent)

### F13. Deferred iter-001 items still open — [NIT]
- **File(s)**: `competition/vision_udp.py:248-253` (`_delivered_ids` cap); `gate_sequencing/sequencer.py:89-95` (`enforce_in_order` recovery); `control/mpc_tracker.py:268-290` (`SimplePositionTracker` ignores `use_residual`)
- **Issue**: Unchanged; with vision now on the hot path, `_delivered_ids` sliding window (~64 frames) can admit duplicate `frame_id` after wrap. Still not production-critical for 30s benches but relevant for 8-minute runs.
- **Confidence**: medium (dup frames); high (others unchanged)

## Things iter-002 got right

- Calibration regression matches NED hover physics; dedicated hover test would have caught the iter-001 sign bug.
- Sequencer terminal semantics: DQ, crash, timeout state machine, strict-opening DQ decoupled from lenient pass margin, future-strut crashes, multi-gate drain with tests.
- Vision: jpeg_size sum validation, mavlink bridge returns listener frames, malformed UDP doesn’t kill the protocol.
- Bench honesty: `crashed` vs `disqualified` split in synthetic and PyBullet result dicts.
- ILC bimodal fix via relative floor; `_pt_to_step` uses `round()`.

## What I did NOT review

- Full `git show` diffs for all six iter-002 commits (spot-checked files above).
- Live MAVSDK/DCL binary, offboard arming, or end-to-end UDP vision with real simulator JPEG stream.
- `competition/session.py` beyond timeout loop; PyBullet env physics; ML residual training path.
- Entire `tests/` suite execution (ran hover calibration test only).
- `planning/racing_line_cache.json` working-tree change (out of iter-002 commit scope).
