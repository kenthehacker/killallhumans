# Iter 002 Adversarial Review — Composer 2.5 (4/5)

## Summary

Iter-002 closed most of the iter-001 BLOCKER/MAJOR punch list (calibration physics, pipeline DQ/crash stop, vision bridge wiring, multi-gate drain, future-strut crashes, strict DQ opening, `jpeg_size` validation, ILC relative threshold, HFoV documentation, bench `crashed`/`disqualified` split). The tree is materially better, but two items marketed as fixed are only **half-fixed**: the VQ1 8-minute cap is enforced in `RaceSession` via wall `time.time()`, while `RacePipeline`’s `mark_timed_out()` path uses `time.monotonic() - time.time()` and **never trips**; the benchmark still has no 480 s contract. New regressions to prioritize: future-gate scan `break` drops second same-tick strut hits, `VisionUdpListener.start()` can leak transports on re-bind, and `latest_frame()` re-decodes the same JPEG on every control tick.

### Iter-001 punch-list grade (iter-002 bar)

| Item | Verdict |
|------|---------|
| B1 calibration physics | **CLOSED** — `y = gravity - a`, hover test ≈ 21.8 (`competition/calibration.py:102-118`, `tests/test_calibration.py:63-91`) |
| B2 pipeline DQ/crash stop | **CLOSED** — `should_stop` + hover early-return (`race_pipeline.py:287-293`, `359-369`) |
| B3 `CameraFrame` 360 default | **CLOSED** — `competition/adapter.py:150-151` |
| B4 vision UDP → bridge | **CLOSED** — `mavlink_bridge.py:189-198`, listener in `connect()` |
| B5 pitch → intrinsics | **PARTIAL** — threaded (`race_pipeline.py:134-139`); PnP world path still ignores `pitch_offset_rad` (`estimation/gate_pnp.py:223-267`) — acceptable per iter-001 defer, not a new bug |
| Opus F2 multi-gate/tick | **CLOSED** — drain loop `gate_sequencing/sequencer.py:349-393` + tests |
| Opus F3 future strut | **PARTIAL** — crash on future strut works; **same-tick multi-future strut still drops after first `break`** (`sequencer.py:447-457`) |
| Opus F14 strict DQ opening | **CLOSED** — `crash_margin` in DQ branch `sequencer.py:433-435` |
| Opus F9 `jpeg_size` validation | **CLOSED** — `vision_udp.py:125-128`, `234-238` |
| Opus F5 ILC relative threshold | **CLOSED** — `planning/ilc_sections.py:107-115` |
| 8-minute VQ1 timeout | **HARD MISS (partial)** — `RaceSession` OK (`session.py:169`); `RacePipeline.mark_timed_out` broken (`race_pipeline.py:296`, `382-386`); bench uncapped |
| HFoV/VFoV | **CLOSED** — documented `aigp_geometry.py:45-58` |
| `GateGeometry` 1.5 m default | **CLOSED** — `estimation/gate_pnp.py` defaults |
| Bench `crashed` vs DQ | **CLOSED** — `scripts/benchmark.py:451-454`, `test_benchmark_adversarial.py:155+` |
| Opus F6 ILC `round` | **CLOSED** — `ilc_sections.py:140-143` |

Deferred items from iter-001 (`enforce_in_order` recovery false-DQ, `_delivered_ids` cap, `aigp_default` scale, `vel_scale`, `SimplePositionTracker` residual, test docstring drift) remain **open** — not regressions, but still production-relevant with B4 live.

## Findings (ordered by severity)

### F1. `RacePipeline` VQ1 timeout uses `monotonic() - time.time()` — cap never fires — [BLOCKER]
- **File(s)**: `race_pipeline.py:296`, `race_pipeline.py:382-386`, `race_pipeline.py:392-393`
- **Issue**: `_race_start_time` is set with `time.time()` (epoch seconds) but elapsed uses `time.monotonic() - self._race_start_time`. On this host the difference is ≈ −1.78×10⁹ s, so `elapsed > AIGP_VQ1_MAX_RUN_DURATION_S` is never true and `sequencer.mark_timed_out()` is never called from the pipeline path. `should_stop` includes `is_timed_out` (`race_pipeline.py:291`), so the sequencer stays `RACING` for the whole run even when the session hits 480 s. `_maybe_replan` passes the same bogus value as `sim_time`, corrupting replan cooldown timing.
- **Repro**: `python3 -c "import time; print(time.monotonic()-time.time())"` → large negative; run `RacePipeline` > 480 s — `seq.is_timed_out` stays False while `RaceSession._race_loop` exits via `metrics.elapsed_s` (`session.py:169`).
- **Fix sketch**: Store `_race_start_time = time.monotonic()` (or use `time.time()` consistently everywhere). Add adversarial test: mock 481 s elapsed, assert `mark_timed_out` and `is_timed_out`. Align `_maybe_replan` clock with competition sim time when available.
- **Confidence**: high

### F2. Future-gate DQ/crash scan stops after first matching future gate per tick — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:419-457`
- **Issue**: The `for future_gate in self._gates[self._current_idx + 1:]` loop `break`s after the first strict-opening DQ (`441`) or first outer-frame crash (`457`). A single segment that grazes struts on gates N+2 and N+3 in one tick records at most one event; the second future gate is silent until a later tick (may never come if the run terminates).
- **Repro**: Three gates in a line; current `g1`; segment crosses `g2` and `g3` strut annuli (y between opening and outer) in one `update`. Only the lowest-index future gate in the slice gets `last_crash`.
- **Fix sketch**: On crash in the annulus, record all matching future gates (or at least don’t `break` until all futures in the slice are tested); add `test_two_future_strut_hits_same_tick`.
- **Confidence**: high

### F3. `VisionUdpListener.start()` can leak UDP transport if called twice — [MAJOR]
- **File(s)**: `competition/vision_udp.py:376-383`, `competition/vision_udp.py:385-390`
- **Issue**: `start()` always calls `create_datagram_endpoint` and overwrites `self._transport` without closing an existing transport. `stop()` is idempotent, but there is no guard in `start()`. A reconnect path (`connect()` twice without `disconnect()`) orphans the first socket.
- **Repro**: `await listener.start(); await listener.start()` — two endpoints; first never closed (not covered by `test_listener_lifecycle_idempotent_stop`, which only double-stops).
- **Fix sketch**: If `self._transport is not None`, `await self.stop()` first; test double-start.
- **Confidence**: high

### F4. `latest_frame()` re-decodes the same JPEG every poll until a new frame arrives — [MAJOR]
- **File(s)**: `competition/vision_udp.py:257-259`, `competition/vision_udp.py:400-410`, `competition/mavlink_bridge.py:189-198`
- **Issue**: `pop_latest_frame()` returns `self._latest_frame` without clearing it. `latest_frame()` always runs `decode_jpeg_to_camera_frame` on that object. At 100 Hz control with 30 Hz camera, ~70% of ticks pay full `cv2.imdecode` on stale bytes. Under competition CPU budget this competes with EKF/tracker.
- **Repro**: Complete one frame; call `latest_frame()` 100 times — 100 decodes, `receiver.delivered_frames` still 1.
- **Fix sketch**: Track `_last_decoded_frame_id` and skip decode if unchanged; or clear `_latest_frame` on pop and stash last `CameraFrame`; benchmark decode count in integration test.
- **Confidence**: high

### F5. Benchmark still has no 480 s cap or `mark_timed_out` — honesty gap vs competition — [MAJOR]
- **File(s)**: `scripts/benchmark.py` (no `AIGP_VQ1_MAX_RUN_DURATION_S`), `competition/aigp_geometry.py:78`
- **Issue**: Iter-002 wired timeout into `RacePipeline`/`GateSequencer` but not into `run_synthetic_benchmark` / `run_sim_benchmark`. CLI `--duration` can still exceed 480 s; synthetic loop uses `sim_time = step * dt` with no timeout branch. Self-score remains misaligned with VQ1 rules (iter-001 composer-25-4 F5 still applies).
- **Repro**: `grep AIGP_VQ1_MAX_RUN_DURATION_S scripts/benchmark.py` → empty; `--duration 600` runs 10 minutes.
- **Fix sketch**: `duration = min(requested, AIGP_VQ1_MAX_RUN_DURATION_S)`; call `seq.mark_timed_out` when `sim_time` exceeds cap; adversarial test at 481 s sim time.
- **Confidence**: high

### F6. Wall-clock timeout vs simulation time — false timeout / false honesty under paused or accelerated sim — [MAJOR]
- **File(s)**: `race_pipeline.py:380-386`, `competition/session.py:64-65`, `competition/session.py:169`
- **Issue**: VQ1 cap is defined against competition *simulation* duration; both `RaceSession` and (if fixed) `RacePipeline` use wall clock. Paused DCL sim or slow-motion replay would time out while sim time is low; accelerated bench would not time out while sim exceeds 8 minutes.
- **Repro**: Freeze sim UI for 500 s wall time with sim frozen — session exits; sim time unchanged.
- **Fix sketch**: Thread `TelemetryState` / vision `sim_time_ns` into elapsed; document wall-clock as fallback only when sim clock absent.
- **Confidence**: medium (depends on DCL exposing sim clock on MAVLink path)

### F7. Calibration positivity guard does not catch wrong-convention fits with large positive `k_t` — [MINOR]
- **File(s)**: `competition/calibration.py:102-119`, `tests/test_calibration.py:22-49`
- **Issue**: Only `k_t <= 0` raises. Samples generated with the *old* wrong sign (`a = -k_t·u - k_d·v - g`) can still yield a large positive `k_t` under the *new* regression (bias absorbed), with elevated `rmse` but no rejection. `read_calibration_json` accepts arbitrary JSON with no range/schema checks (`calibration.py:153-167`).
- **Repro**: Feed `_synth_samples` built with pre-F1 physics but run through post-F1 `identify_thrust_drag_ratios`; inspect `rmse` vs hover-anchored bound.
- **Fix sketch**: Assert `rmse` below threshold and `k_t` within [g, g/0.2] for normalized thrust; validate JSON keys and sane ranges on load.
- **Confidence**: medium

### F8. Synthetic bench `pass_through_margin=1.5` vs production sequencer default `1.0` — pass-credit drift — [MINOR]
- **File(s)**: `scripts/benchmark.py:317-320`, `gate_sequencing/sequencer.py:78`, `sim_pybullet/runner.py:183`
- **Issue**: F14 fixed DQ to use `crash_margin`; pass-through credit and multi-gate drain still use lenient `pass_through_margin`. Bench uses 1.5; `RacePipeline`/`GateSequencer()` default 1.0. Synthetic PASS paths can credit gates that PyBullet/competition paths would not on the same geometry (especially tight legacy 1.2 m + 0.18 m courses).
- **Repro**: Compare `gates_passed` on identical trajectory with margin 1.0 vs 1.5.
- **Fix sketch**: Align bench margin to 1.0 for honesty, or document explicit “lenient synthetic” profile; split config fields in results JSON.
- **Confidence**: medium

### F9. `mark_timed_out` after crash in same tick: pipeline hovers on crash before timeout check — [MINOR]
- **File(s)**: `race_pipeline.py:346-386`, `gate_sequencing/sequencer.py:184-198`
- **Issue**: `update()` can set `last_crash` without `RaceState.TIMED_OUT`. `_control_callback` returns on `last_crash` before the wall-timeout block, so `mark_timed_out` is skipped that tick. Harmless for termination (session stops on crash via `should_stop`), but metrics never show both signals if a crash happens at t>480 s on the same tick.
- **Repro**: Force crash on final tick after 480 s wall time with F1 fixed — expect only crash in logs, not `timeout_reason`.
- **Fix sketch**: Call `mark_timed_out` before crash early-return, or document crash precedence in honesty schema.
- **Confidence**: medium

### F10. `_delivered_ids` sliding window still allows duplicate `frame_id` after ~64 frames — [NIT]
- **File(s)**: `competition/vision_udp.py:248-253`, `competition/vision_udp.py:195-198`
- **Issue**: Deferred from iter-001 (Opus F10); unchanged. Cap `max_buffered_frames * 8` (64). Late replay of an evicted ID can reassemble twice.
- **Repro**: Deliver 70 unique `frame_id`s, re-send chunks for `frame_id=5`.
- **Fix sketch**: `deque` + `set` membership; test ID reuse policy.
- **Confidence**: low for live AIGP; medium for tests/replay

### F11. Vision `error_received` counts errors but does not restart listener — [NIT]
- **File(s)**: `competition/vision_udp.py:340-341`
- **Issue**: Transient OS errors increment `errors` only. Long outage leaves `get_camera_frame()` returning None with no recovery until reconnect.
- **Repro**: Simulate ICMP port unreachable; observe stuck `None` frames.
- **Fix sketch**: Optional exponential-backoff rebind; expose `errors` in pipeline health log.
- **Confidence**: low

### F12. `enforce_in_order` still false-DQs legitimate replanner recovery — [MINOR, deferred]
- **File(s)**: `gate_sequencing/sequencer.py:89-95`, `413-441`
- **Issue**: Explicitly deferred in iter-001; still default `True`. With B4 live, recovery paths that re-cross a future gate opening will DQ instead of reattempt.
- **Repro**: Miss gate 2, replan loop back through gate 3’s plane inside strict opening before gate 2 is credited.
- **Fix sketch**: DQ only if `future_gate.sequence_index` is less than next expected, or add `recovery_grace` window after `RECOVERY` state.
- **Confidence**: medium

## Things iter-002 got right
- Calibration regression matches NED hover physics; `test_hover_only_samples_recover_g_over_u` would have caught the iter-001 sign bug (`tests/test_calibration.py:63-91`).
- Pipeline termination on DQ, crash, and completion is wired consistently in `should_stop` and hover early-return (`race_pipeline.py:287-293`, `359-377`).
- Vision path is end-to-end: UDP listener → reassembly → `MAVLinkBridge.get_camera_frame()` (`mavlink_bridge.py:111-121`, `189-198`) with real-socket tests (`tests/test_vision_udp_listener.py`).
- Sequencer F2/F3/F14 adversarial tests are concrete and would fail on iter-001 code (`gate_sequencing/tests/test_sequencer_adversarial.py:210-374`).
- `jpeg_size` vs payload-sum validation closes silent corrupt JPEG drops (`competition/vision_udp.py:234-238`, `tests/test_vision_udp.py`).
- ILC bimodal partition test and 5% relative floor address Opus F5 (`planning/ilc_sections.py:107-115`, `tests/test_ilc_sections.py`).
- Bench honesty: `disqualified` no longer sets `crashed=True` (`scripts/benchmark.py:451-454`).

## What I did NOT review
- Full `git show` diffs for all six iter-002 commits (reviewed current tree + commit messages).
- Live `python3 scripts/benchmark.py --mode full` execution or `benchmark_history.jsonl` trends.
- `gate_detection/` beyond intrinsics threading; ML residual training/holdout.
- `sim_pybullet/runner.py` vs `race_pipeline._maybe_replan` line-by-line parity.
- DCL Windows binary / real SITL / MAVSDK command regression (A12 still deferred).
- `config/aigp_default.json` course fidelity (Opus F7 defer).
- All five other iter-002 review agents’ drafts (this file is independent).
