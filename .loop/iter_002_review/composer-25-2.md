# Iter 002 Adversarial Review — Composer 2.5

## Summary

Iter-002 genuinely closes most of the iter-001 BLOCKER/MAJOR punch list (calibration physics, pipeline termination, vision bridge wiring, multi-gate drain, future-strut crashes, jpeg_size validation, ILC relative threshold, DQ strict opening, bench DQ/crash split). The 334-test matrix is credible for what was committed. Two regressions stand out: the VQ1 8-minute cap wired in `race_pipeline.py` is effectively dead because `_race_start_time` uses `time.time()` while elapsed checks use `time.monotonic()`, and `VisionUdpListener.start()` is not idempotent (double-bind leaks transports). Several iter-001 deferrals remain production-relevant now that the MAVLink vision path is live.

## Iter-001 punch list — grade

| Item | Verdict | Evidence |
|------|---------|----------|
| B1 calibration physics | **CLOSED** | `competition/calibration.py:102-119` — `y = gravity - a`, `X = [u,v]`, `k_t > 0` guard; `tests/test_calibration.py:63-91` hover test expects ≈ g/u ≈ 21.8 |
| B2 DQ/crash termination | **CLOSED** | `race_pipeline.py:287-294`, `359-369` — `should_stop` and hover early-return honor DQ, timeout, crash |
| B3 CameraFrame 360 | **CLOSED** | `competition/adapter.py:150-151` |
| B4 vision UDP wiring | **CLOSED** | `competition/mavlink_bridge.py:189-198` returns `self._vision_listener.latest_frame()` |
| B5 camera pitch threaded | **CLOSED (wiring only)** | `race_pipeline.py:134-139` sets `pitch_offset_rad`; PnP transform still omits extrinsic rotation (`gate_pnp.py:223-266`) — acceptable per brief note for position-only PnP |
| Opus F2 multi-gate/tick | **CLOSED** | `gate_sequencing/sequencer.py:349-393` drain loop; adversarial tests at `test_sequencer_adversarial.py:210-253` |
| Opus F3 future-strut crash | **CLOSED** | `sequencer.py:442-457`; tests `test_future_gate_strut_hit_*` |
| Opus F14 DQ strict opening | **CLOSED** | `sequencer.py:433-435` uses `crash_margin`; F14 tests with `pass_through_margin=1.5` |
| Opus F9 jpeg_size validation | **CLOSED** | `vision_udp.py:125-128`, `234-238`; matching tests in `tests/test_vision_udp.py` |
| Opus F5 ILC relative threshold | **CLOSED** | `planning/ilc_sections.py:107-115`; `test_bimodal_low_dominant_high_minority_still_partitions` |
| 8-min VQ1 timeout | **PARTIAL / HARD MISS in pipeline** | `RaceState.TIMED_OUT` + `mark_timed_out` wired (`sequencer.py:184-198`); `RaceSession` still enforces 480s via `time.time()` (`session.py:169-171`); **`race_pipeline.py:296` + `382-386` clock-domain bug** — see F1 |
| HFoV/VFoV | **CLOSED** | `competition/aigp_geometry.py` documents 90° H / ~58.7° V; `tests/test_camera_geometry.py` |
| GateGeometry 1.2 m default | **CLOSED** | `estimation/gate_pnp.py:115-116` → `AIGP_GATE_INTERIOR_M` |
| Bench crashed≠DQ | **CLOSED** | `tests/test_benchmark_adversarial.py:155-198` |
| ILC `_pt_to_step` truncation | **CLOSED** | `ilc_sections.py:140-143` uses `round` |

## Findings (ordered by severity)

### F1. VQ1 timeout in `RacePipeline` never fires — `time.time()` vs `time.monotonic()` mismatch — [BLOCKER]
- **File(s)**: `race_pipeline.py:296`, `382-386`, `392-393`
- **Issue**: `run()` sets `self._race_start_time = time.time()` (epoch seconds), but `_control_callback` computes `elapsed = time.monotonic() - self._race_start_time`. On this machine `monotonic() - time.time() ≈ -1.78×10⁹`, so `elapsed > AIGP_VQ1_MAX_RUN_DURATION_S` is never true. `mark_timed_out()` in the pipeline path is dead code. `is_timed_out` in `should_stop` only becomes true if something else calls `mark_timed_out` (nothing does in the MAVLink path today).
- **Repro**: `python3 -c "import time; print(time.monotonic()-time.time())"` → large negative. Grep `race_pipeline.py` for `_race_start_time` — mixed clocks. Iter-002 commit `434801e` claimed the cap was wired; unit tests only call `mark_timed_out()` directly, not the pipeline clock path.
- **Fix sketch**: Use one clock domain: `self._race_start_mono = time.monotonic()` and compare with `time.monotonic() - self._race_start_mono`, or use `time.time()` for both. Add adversarial test that mocks 481s elapsed and asserts `sequencer.is_timed_out` after one `_control_callback`. Align `_maybe_replan`'s `sim_time` (line 392) to the same anchor.
- **Confidence**: high — reproduced numerically; explains why iter-002 timeout tests pass but competition pipeline path does not trip.

### F2. `VisionUdpListener.start()` without guard leaks UDP transport on re-bind — [MAJOR]
- **File(s)**: `competition/vision_udp.py:376-383`, `tests/test_vision_udp_listener.py:152-167`
- **Issue**: `start()` always calls `create_datagram_endpoint` and overwrites `self._transport` without closing an existing transport. Tests cover idempotent `stop()` but not double `start()`. A reconnect or mis-ordered lifecycle leaves the first socket open (port leak / duplicate handlers).
- **Repro**: `await listener.start(); await listener.start()` — second call should either no-op or close the first transport; current code assigns a new transport and orphans the old one.
- **Fix sketch**: At top of `start()`, `if self._transport is not None: return` or `await self.stop()` first. Add `test_listener_double_start_is_idempotent`.
- **Confidence**: high — read lifecycle; no guard in source.

### F3. `latest_frame()` re-decodes the same JPEG on every poll until a new frame arrives — [MAJOR]
- **File(s)**: `competition/vision_udp.py:257-259`, `400-410`, `competition/mavlink_bridge.py:198`
- **Issue**: `pop_latest_frame()` returns `self._latest_frame` without clearing it. `latest_frame()` always runs `decode_jpeg_to_camera_frame(rf)` on that stale `ReassembledFrame`. A 100 Hz control loop pays `cv2.imdecode` every tick (~10 ms class on competition hardware) even when no new UDP frame completed. The iter-002 design note ("decode at most once per tick") understates cost — it's once per tick **per poll**, not once per new frame.
- **Repro**: Call `get_camera_frame()` / `latest_frame()` in a tight loop with a static reassembled frame; CPU scales with poll rate, not camera FPS (30 Hz).
- **Fix sketch**: Track `_last_decoded_frame_id` and return cached `CameraFrame` when `frame_id` unchanged; or clear `_latest_frame` after decode (true pop semantics). Add test asserting decode runs once per `frame_id`.
- **Confidence**: high — control flow read; no cache in listener.

### F4. Calibration positivity guard alone cannot catch wrong-sign physics — large positive `k_t` with tiny RMSE — [MAJOR]
- **File(s)**: `competition/calibration.py:109-119`, `tests/test_calibration.py`
- **Issue**: Guard only rejects `k_t <= 0`. Feeding 50 samples generated with the *old* wrong equation `a = -g + k_t·u` (sign error) yields `k_t ≈ 17.24`, `rmse ≈ 1.8×10⁻¹⁵`, no exception. A mis-wired telemetry feeder or inverted accel convention can pass CI while producing wrong thrust scaling for the planner.
- **Repro**: `identify_thrust_drag_ratios` on `[CalibrationSample(0.5, 0.0, -9.81+22*0.5) for _ in range(50)]` → positive k_t, no error (verified in REPL during review).
- **Fix sketch**: Add RMSE ceiling (e.g. reject if `rmse > 0.5` m/s² on hover-heavy samples), thrust span check (`max(u)-min(u) > ε`), or a golden hover-only fixture asserting `k_t ∈ [18, 25]`. Optionally reject fits where `k_t` deviates from `g/mean(u)` by >X% when vertical accel variance is low.
- **Confidence**: high — REPL repro; hover test only covers correct physics.

### F5. Bench vs competition sequencer margins diverge — honesty drift — [MAJOR]
- **File(s)**: `scripts/benchmark.py:317-320`, `race_pipeline.py:204`, `gate_sequencing/sequencer.py:78`
- **Issue**: Synthetic bench hard-codes `pass_through_margin=1.5`; `RacePipeline` uses `GateSequencer(gates)` with default `1.0` and `enforce_in_order=True`. PyBullet bench builds `SequencerConfig` from track JSON without setting margin unless overridden (defaults `1.0`). A green synthetic bench does not prove the same pass/DQ/crash classification the MAVLink pipeline will see. F14 fixed DQ to use strict opening, but pass-through credit and drain loop still use lenient margin — bench is *more* lenient than production pipeline.
- **Repro**: Compare `benchmark.py:318` vs `race_pipeline.py:204` — no `SequencerConfig` on pipeline path.
- **Fix sketch**: Load sequencer overrides from track config in `RacePipeline.configure()` (mirror PyBullet runner); or set bench to `pass_through_margin=1.0` for "competition honesty" mode. Document which margin the VQ1 binary expects.
- **Confidence**: high — direct config comparison.

### F6. `read_calibration_json` accepts garbage with no schema/range validation — [MINOR]
- **File(s)**: `competition/calibration.py:152-167`
- **Issue**: Any JSON with the four numeric keys loads; negative `thrust_per_mass`, NaN, or absurd RMSE propagate into `CalibrationResult` without checks. Competition could ship a corrupt file at race start.
- **Repro**: Write `{"thrust_per_mass_1_per_s2": -1, "drag_per_mass_1_per_s": 0, "n_samples": 2, "rmse_mps2": 0}` — loads cleanly.
- **Fix sketch**: Validate ranges on read (mirror identify's `k_t > 0`); reject NaN/inf.
- **Confidence**: high.

### F7. `mark_timed_out` and crash in the same control tick — ambiguous terminal semantics — [MINOR]
- **File(s)**: `gate_sequencing/sequencer.py:184-198`, `447-457`, `race_pipeline.py:366-386`
- **Issue**: `update()` can set `_last_event = "crash"` while `_state` stays `RACING`. Later in the same `_control_callback`, wall-time logic may call `mark_timed_out()` (once F1 is fixed), transitioning to `TIMED_OUT` while `last_crash` remains set. `should_stop` triggers on either; metrics may attribute termination inconsistently. `mark_timed_out` does not block on existing crash; `mark_collision` does not exclude `DISQUALIFIED` (only WAITING/COMPLETED/TIMED_OUT at `sequencer.py:521-524`).
- **Repro**: Long run: crash on gate frame, same tick exceeds 480s after F1 fix — both flags true.
- **Fix sketch**: If `last_crash` is set, skip `mark_timed_out`; add `DISQUALIFIED` to `mark_collision` state gate; document precedence in bench JSON (`termination_reason` priority).
- **Confidence**: medium — ordering is plausible; not exercised in tests.

### F8. `GatePnPEstimator()` default camera path still legacy 640×480 — [MINOR]
- **File(s)**: `estimation/gate_pnp.py:153-154`
- **Issue**: Iter-002 fixed `GateGeometry` defaults to AIGP 1.5 m but left `camera or CameraIntrinsics.from_fov(90.0, 640, 480)`. Standalone estimator construction gets wrong resolution and principal point vs AIGP 360p.
- **Repro**: `GatePnPEstimator().camera.image_height` → 480.
- **Fix sketch**: Default to `CameraIntrinsics()` (AIGP module defaults) or `from_fov(90, 640, 360)`.
- **Confidence**: high.

### F9. Multi-gate drain does not advance past a "miss" on the new current target — edge case — [MINOR]
- **File(s)**: `gate_sequencing/sequencer.py:353-393`, `394-401`
- **Issue**: After crediting gate N, the drain loop breaks when the new current gate's plane is crossed outside the lenient opening (strut branch or `break` at 385). A simultaneous *miss* on that gate (plane crossed outside outer frame) is handled only in the first-gate branch (`elif` at 394), not inside the drain loop. Rare fast segment: credit N, cross N+1 plane outside frame entirely → drain breaks without recording miss on N+1; recovery/DQ behaviour may lag one tick.
- **Repro**: Construct segment that credits g1, crosses g2's plane far above outer frame in same tick — inspect `_misses` vs single-gate miss path.
- **Fix sketch**: Mirror the miss `elif` inside the drain loop before `break`, or run miss classification on `nxt` when plane crossed but not opening/outer-strut.
- **Confidence**: low-medium — logic trace; no failing test found.

### F10. `_delivered_ids` sliding window still allows duplicate frame re-emit after ~64 IDs — [MINOR, deferred]
- **File(s)**: `competition/vision_udp.py:248-253`, `195-198`
- **Issue**: Cap is `max_buffered_frames * 8` (64 with default 8). After eviction from `_delivered_ids`, a very late chunk for an old `frame_id` can be reassembled and emitted again → duplicate perception updates. More likely on lossy links during long runs (iter-001 F10 deferral; still relevant with live bridge).
- **Repro**: Deliver frame 1, advance 65 new frame IDs, resend chunk for frame 1 — not in unit tests.
- **Fix sketch**: Use a `set` for delivered IDs with bounded size, or monotonic "last emitted frame_id" drop.
- **Confidence**: medium — design unchanged in iter-002.

### F11. `enforce_in_order` false-DQ on replanner recovery — still deferred, now worse — [MINOR, deferred]
- **File(s)**: `gate_sequencing/sequencer.py:413-441`, `race_pipeline.py:204`, `388-393`
- **Issue**: With vision + replan live, cutting through a future gate opening during recovery still hard-DQs. `RacePipeline` always uses default `enforce_in_order=True` and does not load track `sequencer` overrides. No adversarial test for "recovery arc through gate N+2 opening."
- **Repro**: Same as iter-001 composer-25-2 F5 — REPL still DQ's on future opening.
- **Fix sketch**: Load track config; or suppress DQ in `RaceState.RECOVERY`; add explicit test and policy doc.
- **Confidence**: medium-high for behaviour; policy choice is product-level.

### F12. `SimplePositionTracker` still ignores `use_residual` — [NIT, deferred]
- **File(s)**: `control/mpc_tracker.py:268-290`
- **Issue**: Unchanged from iter-001. Toggling `use_geometric_tracker=False` with `use_residual=True` silently drops ML path.
- **Confidence**: high — no residual branch in `track()`.

### F13. `pop_latest_frame` name implies consumption but is a peek — [NIT]
- **File(s)**: `competition/vision_udp.py:257-259`, `407`
- **Issue**: Docstring says "pop" but implementation peeks; couples with F3 redundant decode.
- **Fix sketch**: Rename to `peek_latest_frame()` or clear after read.
- **Confidence**: high.

## Things iter-002 got right

- **Calibration fix is real and tested**: Correct NED regression plus `test_hover_only_samples_recover_g_over_u` would have caught the original Opus F1 sign bug.
- **Pipeline termination**: DQ, crash, and (intended) timeout are wired into both `should_stop` and hover commands — no more silent fly-through after DQ.
- **Sequencer geometry symmetry**: Future-gate strut hits, strict-opening DQ, and multi-gate drain are implemented with focused adversarial tests — the highest-risk iter-001 MAJORs are addressed in code, not just comments.
- **Vision path integration**: `VisionUdpListener` + E2E UDP test + `MAVLinkBridge.get_camera_frame()` finally connect A11 to the competition surface.
- **Honesty improvements**: Bench no longer sets `crashed=True` on DQ; ILC bimodal partition and jpeg_size validation close silent-failure modes.

## What I did NOT review

- Full `git show` for all six commits line-by-line (read current files + commit messages).
- Live PyBullet / DCL sim runs or `python3 scripts/benchmark.py --mode sim`.
- `gate_detection/`, EKF noise tuning, `DynamicReplanner` geometry, ML residual trainer/weights.
- `competition/session.py` connect/arm/offboard beyond timeout interaction.
- Deferred items: `aigp_default.json` scale fidelity, `vel_scale=0.5` default, adversarial test docstring drift (Opus F12), A12/A16/A17 scaffold files.
