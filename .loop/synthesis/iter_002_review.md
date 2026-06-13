# Iter 002 — Adversarial Review Synthesis

7 substantive reviews returned (2 GPT-5.5 + 5 Composer; Opus crashed at 1.84s — model error, not output). The reviews caught real bugs the previous round missed.

## BLOCKERs (apply iter-003 immediately)

### B1. RacePipeline 8-minute timeout NEVER fires — `time.monotonic() - time.time()` clock mismatch [5/7 reviews, all BLOCKER]
- File: `race_pipeline.py:277` (`self._race_start_time = time.time()`) + my A_c8 patch (`elapsed = time.monotonic() - self._race_start_time`)
- `time.time()` is seconds-since-epoch (~1.78e9); `time.monotonic()` is seconds-since-boot (~1.5e5). The subtraction produces a huge NEGATIVE number that never exceeds 480.
- Net effect: the 8-min VQ1 cap I "wired up" in iter-002c is **dead code**. The sequencer.mark_timed_out call never runs.
- Fix: change `_race_start_time` initialisation to `time.monotonic()` to match the check. Use sim-time when available for paused/sped-up simulators.
- All 5 BLOCKER reports independently identify the same mistake — high confidence.

## MAJORs (apply iter-003 in same patch sequence as B1)

### M1. `VisionUdpListener.start()` is not idempotent — calling twice leaks the first UDP transport [6/7 reviews]
- File: `competition/vision_udp.py` — `start()` overwrites `self._transport` without closing the previous one.
- Real exposure: any retry / reconnect path in MAVLinkBridge would leak.
- Fix: guard with `if self._transport is not None: return` (or close-then-open).

### M2. `latest_frame()` re-decodes the same JPEG on every poll [7/7 reviews, universal MAJOR]
- File: `competition/vision_udp.py::VisionUdpListener.latest_frame`
- `pop_latest_frame()` is a peek (doesn't pop). A 100Hz control loop polling for the freshest frame re-decodes the same bytes 30× per frame between vision updates.
- Fix: cache the decoded `CameraFrame` keyed by frame_id; only decode on frame_id transition.

### M3. `GatePnPEstimator()` standalone defaults still 640×480 [5/7 reviews]
- File: `estimation/gate_pnp.py` (specifically the construction path when `CameraIntrinsics` is built via `from_fov` with defaults)
- Need to verify — my iter-001 A6 fix updated `CameraIntrinsics()` direct constructor defaults; `from_fov(90, 640, 480)` still legitimately returns 640×480. The MAJOR is about callers that use `from_fov` with legacy args, not about the default constructor.
- Fix: investigate the actual code path; if `GatePnPEstimator.__init__` builds its own CameraIntrinsics, change it to AIGP defaults. Otherwise add a regression test confirming the bare path gives AIGP.

### M4. Calibration positivity guard misses wrong-physics with large positive k_t [6/7 reviews]
- File: `competition/calibration.py:115-120`
- Sign-flipped synthetic data absorbed the missing intercept and produced `k_t = 54` instead of true 22 — large positive, my guard says "OK".
- Fix: add an RMSE-relative outlier check. If `rmse > 2.0 * |y|.mean()`, the fit is unreliable; raise.
- Also add an explicit hover sanity check: `k_t > 0.5 * gravity / u_hover_max` (rules out tiny k_t too).

### M5. Wall-clock vs simulation time for the 8-min cap [4/7 reviews MAJOR]
- File: `race_pipeline.py` (my A_c8 patch uses `time.monotonic()`)
- A simulator running at 10× speed under-times-out by 10×; a paused sim could spuriously time out wall-clock-wise.
- Fix: pass sim time to the timeout check. RaceSession already has telemetry timestamps; thread `telem.timestamp_us` into the control callback and check sim elapsed.
- Coupled with B1: fix both in the same patch.

### M6. Bench paths don't enforce 8-min cap [5/7 reviews MAJOR]
- File: `scripts/benchmark.py` — neither `run_synthetic_benchmark` nor the PyBullet path call `seq.mark_timed_out` after the 480s mark.
- Fix: add the check in the synthetic bench loop (`if sim_time > AIGP_VQ1_MAX_RUN_DURATION_S: seq.mark_timed_out(...)`) and mirror in PyBullet path.
- The bench's existing `THRESHOLDS["max_total_time_s"] = 30.0` is a unit-test convenience and unrelated.

### M7. Synthetic bench `pass_through_margin=1.5` vs production / PyBullet default `1.0` [4/7 reviews MAJOR]
- File: `scripts/benchmark.py::run_synthetic_benchmark` (`SequencerConfig(pass_through_margin=1.5, proximity_pass_distance=1.0)`)
- Synthetic and PyBullet now report different pass/DQ behaviour for the same drone trajectory. Platform-honesty drift.
- Fix: align margins — either lower synthetic to 1.0, or document why 1.5 is needed (and add a deviation note in the result dict).

## MINORs (iter-003 or later)
- Multi-strut hits in one segment: only the last is recorded as `last_crash` (gpt-55-1 F3 + composer-25-5 F15)
- Same-segment crash overwritten by DQ (gpt-55-2 F2)
- Multi-gate drain doesn't advance past a "miss" on new current target (composer-25-2 F9)
- `read_calibration_json` has no schema/range validation (4 reviews)
- `enforce_in_order` still false-DQs legitimate replanner recovery (5 reviews — now production-critical with the new RacePipeline termination)
- `_delivered_ids` cap allows duplicate re-emit after 64 frames (multi-review)
- `error_received` counts errors but doesn't restart listener (composer-25-3 F12)
- `pop_latest_frame` name suggests consumption but is a peek (composer-25-2 F13)
- `mark_timed_out` and crash in same tick: ambiguous priority (composer-25-2 F7 + composer-25-5 F3)

## What iter-002 got right (3+ reviews confirmed)
- B1 calibration physics fix (Opus F1 actually fixed; sign positivity guard is necessary but not sufficient — see M4)
- B2 RacePipeline DQ termination correctly wired (modulo the M5 clock issue, which actually breaks B1's twin)
- B3 CameraFrame default 360 — verified
- Multi-gate-per-tick drain loop works for the common 2-gate case
- Future-gate strut crash classification (F3) works
- Vision UDP `jpeg_size` validation
- HFoV vs VFoV constants — correctly disambiguated
- Bench DQ/crash signal split (iter-002e) verified

## Decision: iter-003 patch scope (this round)
Apply B1 + M1 + M2 + M4 in one patch. Defer M3, M5, M6, M7 to iter-003b (need code investigation or larger surface).
