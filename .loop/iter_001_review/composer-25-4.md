# Iter 001 Adversarial Review — Composer 2.5 (4/5)

## Summary
Iter-001 materially improved honesty (sequencer DQ, bench terminal signals, ILC de-magic-ing, vision reassembler tests) but several shipped items are **half-wired**: the competition pipeline still flies through DQ/crash, MAVLink camera is still `None`, and A6 camera tilt exists only as a dataclass field—not in PnP world-frame math. The synthetic bench now *reads* sequencer failures but still rarely *produces* geometric gate-frame crashes. Treat green unit tests as necessary, not sufficient.

## Findings (ordered by severity)

### F1. `RacePipeline` never terminates on DQ or crash — [BLOCKER]
- **File(s)**: `race_pipeline.py:271-273`, `race_pipeline.py:325-336`, `race_pipeline.py:338-343`
- **Issue**: `session.should_stop` only checks `sequencer.is_complete`. After `GateSequencer` sets `RaceState.DISQUALIFIED` (or records `last_crash`), `_control_callback` keeps tracking and sending attitude commands. No `is_disqualified`, `dq_reason`, or `last_crash` checks exist anywhere in `race_pipeline.py`.
- **Repro**: Run `RacePipeline` against any adapter that feeds positions crossing a future gate opening (or triggering P1-6 crash). Sequencer DQ/crash fires; loop continues until wall-clock session timeout or external stop.
- **Fix sketch**: Extend `should_stop` to `is_complete or is_disqualified or last_crash is not None`; in `_control_callback`, return hover/zero-thrust and skip replan when terminal; mirror `scripts/benchmark.py:446-453` / `790-817`.
- **Confidence**: high

### F2. Vision UDP receiver not connected to competition camera path — [BLOCKER]
- **File(s)**: `competition/mavlink_bridge.py:163-167`, `competition/vision_udp.py:144-235`
- **Issue**: A11 shipped `VisionUdpReceiver` + tests, but `MavlinkBridge.get_camera_frame()` still returns `None` with a "will be implemented" comment. The AIGP :5600 JPEG stream cannot reach gate detection / PnP in the only competition entrypoint.
- **Repro**: `await bridge.get_camera_frame()` after connect — always `None`; `RacePipeline._process_detection` never runs with real imagery on MAVLink path.
- **Fix sketch**: Start asyncio datagram listener on `AIGP_CAM_UDP_PORT`, feed packets into `VisionUdpReceiver`, decode via `decode_jpeg_to_camera_frame`, return latest frame from `get_camera_frame()`. Add integration test with mock UDP sender.
- **Confidence**: high

### F3. Camera tilt field not applied in PnP → world position (A6 incomplete) — [BLOCKER]
- **File(s)**: `estimation/gate_pnp.py:214-258`, `tests/test_camera_geometry.py:75-91`
- **Issue**: `CameraIntrinsics.pitch_offset_rad` and comments reference `body_R_camera`, but `gate_pose_to_drone_position()` never applies a body↔camera rotation. `test_horizon_projects_below_image_center_with_upward_tilt` only checks the pinhole formula on intrinsics—it does not exercise PnP recovery under tilt.
- **Repro**: Place a gate at horizon with +20° camera pitch; PnP-derived drone position will be wrong in world NED (systematic lateral/longitudinal bias), while unit tests stay green.
- **Fix sketch**: Apply `R_pitch(pitch_offset_rad)` when mapping camera-frame PnP translation into body/world; add regression test that compares recovered position against known synthetic pose with tilt ≠ 0.
- **Confidence**: high

### F4. `PipelineConfig.camera_pitch_offset_rad` ignored at construction — [MAJOR]
- **File(s)**: `race_pipeline.py:78`, `race_pipeline.py:123-127`, `estimation/gate_pnp.py:68-88`
- **Issue**: `RacePipeline` builds `CameraIntrinsics.from_fov(...)` from width/height/fov only. `config.camera_pitch_offset_rad` is never passed through; non-default pitch in `PipelineConfig` has no effect.
- **Repro**: `PipelineConfig(camera_pitch_offset_rad=0.0)` — intrinsics still carry `AIGP_CAM_PITCH_OFFSET_RAD` via dataclass default on `from_fov()` return value.
- **Fix sketch**: After `from_fov`, set `self.camera.pitch_offset_rad = self.config.camera_pitch_offset_rad`, or add `pitch_offset_rad` parameter to `from_fov` / use `CameraIntrinsics()` AIGP defaults directly.
- **Confidence**: high

### F5. Spec 8-minute run cap not enforced on benchmark / synthetic paths — [MAJOR]
- **File(s)**: `competition/aigp_geometry.py:65`, `scripts/benchmark.py:234`, `scripts/benchmark.py:420`, `competition/session.py:36`, `competition/session.py:169`
- **Issue**: `AIGP_VQ1_MAX_RUN_DURATION_S` is defined but unused outside comments. `run_synthetic_benchmark(duration=30)` and CLI `--duration` default 30s; only `RaceSession._race_loop` enforces 480s—and only when using that session wrapper.
- **Repro**: `python3 scripts/benchmark.py --mode full --duration 600` runs 10 minutes with no spec-aligned cap; self-scoring diverges from competition rules.
- **Fix sketch**: Clamp `duration` to `min(requested, AIGP_VQ1_MAX_RUN_DURATION_S)` in bench entrypoints; add adversarial test that requests 999s and asserts termination ≤ 480s on session path.
- **Confidence**: high

### F6. Synthetic bench still unlikely to exercise geometric gate-frame crash — [MAJOR]
- **File(s)**: `scripts/benchmark.py:422-453`, `tests/test_benchmark_adversarial.py:60-148`
- **Issue**: A7 wires `seq.last_crash` / `is_disqualified` into the result dict, but the kinematic loop follows a precomputed trajectory—it almost never produces a P1-6 plane crossing in the strut annulus. Adversarial bench tests only assert field *presence* or ground crash (`z=0`), not "synthetic sim terminates on geometric frame strike."
- **Repro**: Run full synthetic on `race_01`; expect `crashed=False`, `last_crash_gate=None` even when ILC/tracker are poor—honesty fields exist but stay idle.
- **Fix sketch**: Add a minimal injected course (like `test_sequencer_records_geometric_frame_strike`) inside `run_synthetic_benchmark` regression, or drive positions from a hand-crafted polyline that clips a strut; assert `termination_reason` starts with `crash_gate:`.
- **Confidence**: high

### F7. Vision reassembly trusts `jpeg_size` without validating assembled length — [MAJOR]
- **File(s)**: `competition/vision_udp.py:84-97`, `competition/vision_udp.py:220-225`, `tests/test_vision_udp.py:108-119`
- **Issue**: `_ReassemblyBuf.assemble()` allocates `bytearray(jpeg_size)` from the header but never checks `offset == jpeg_size` or `sum(len(chunk)) == jpeg_size` after concat. Malicious/incorrect headers can cause huge allocations or truncated JPEGs passed to `cv2.imdecode`.
- **Repro**: Feed chunks whose total payload is 100B but `jpeg_size=10_000_000`; frame completes and allocates 10MB buffer. No test covers `jpeg_size` mismatch (only `payload_size` vs datagram length).
- **Fix sketch**: On complete, verify `offset == jpeg_size` (or spec-allowed tolerance); reject frame and increment `dropped_partial_frames` otherwise. Add tests for `jpeg_size` mismatch and `payload_size==0` last chunk.
- **Confidence**: high

### F8. AIGP intrinsics claim 90° VFoV but fx=fy=320 on 360px height implies ~58° VFoV — [MAJOR]
- **File(s)**: `competition/aigp_geometry.py:40-45`, `estimation/gate_pnp.py:58`, `.loop/specs/1_aigp_spec_distill.md:32-33`
- **Issue**: With `fy=320`, `cy=180`, `h=360`, vertical FOV is `2·atan(180/320) ≈ 58.4°`, not 90°. `from_fov(90°, 640, 360)` sets `fx=320` from horizontal FOV, not vertical. Perception-aware lookahead and gate visibility margins derived from "90° VFoV" will be wrong.
- **Repro**: Project a body-forward ray at ±45° elevation; image coverage does not match a 90° vertical cone assumption.
- **Fix sketch**: Reconcile with spec (likely fy derived from VFoV, not HFoV); document which FOV is authoritative; update `AIGP_CAM_VFOV_*` constants and tests accordingly.
- **Confidence**: medium (spec PDF may define this differently, but math on shipped constants is inconsistent)

### F9. Global ILC defaults changed for all non-override tracks — [MAJOR]
- **File(s)**: `config/ilc_defaults.json:5-13`, `sim_pybullet/configs/race_01.json:16-20`, `scripts/benchmark.py:347-356`
- **Issue**: New global defaults use `convergence_threshold=0.002`, `momentum_gamma=0.0`, `max_iterations=5` vs pre-iter race_01 sweep (`0.0005`, `0.2`, `8`). Only `race_01` patches via `ilc_global_overrides`; `aigp_default.json` and any future course inherit the new globals—silent behavior change masked as "de-magic-ing."
- **Repro**: Run synthetic on `aigp_default.json` before/after iter-001; ILC convergence profile shifts without an explicit track override.
- **Fix sketch**: Keep old sweep values as global defaults; put experimental values behind a named profile, or require per-track explicit ILC config for VQ1 placeholder tracks.
- **Confidence**: high

### F10. Benchmark uses `pass_through_margin=1.5` for DQ opening test — [MAJOR]
- **File(s)**: `scripts/benchmark.py:317-320`, `gate_sequencing/sequencer.py:351`, `gate_sequencing/tests/test_sequencer_adversarial.py:48-50`
- **Issue**: Out-of-order DQ calls `_point_in_gate_opening()`, which uses `pass_through_margin` (lenient 1.5 in bench). Adversarial sequencer tests use default margin 1.0. Bench can DQ on paths that graze the *lenient* future-gate credit zone without entering the true 1.5 m opening—false fails during legitimate wide racing lines near upcoming gates.
- **Repro**: Fly within 1.5× half-width of a future gate plane in synthetic bench; DQ fires though geometric opening (1.0×) was never entered.
- **Fix sketch**: Split margins: `dq_opening_margin=1.0` (strict) vs `pass_through_margin` (lenient); or fix bench to `pass_through_margin=1.0` for honesty alignment with competition.
- **Confidence**: medium

### F11. ILC step mapping can drop sections when point times are non-uniform — [MINOR]
- **File(s)**: `planning/ilc_sections.py:125-136`, `planning/ilc_sections.py:141-147`
- **Issue**: `_pt_to_step` uses `int(time/dt)`; if trajectory samples are sparse/irregular, consecutive point runs can map to `e_step <= s_step` and get `continue`d, potentially leaving uncovered step ranges before the final "extend to n_total_steps" fixup (only first/last extended).
- **Repro**: Trajectory with long gaps in `point.time` then burst samples; `derive_section_boundaries` returns sections that don't monotonically cover interior steps (gaps in ILC schedule).
- **Fix sketch**: Map by point index fraction: `step = int(i / len(points) * n_total_steps)` or enforce monotonic `s_step` with merge pass; add test with irregular `point.time`.
- **Confidence**: medium

### F12. `GatePnPEstimator` / `GateGeometry` standalone defaults still legacy 1.2 m / 640×480 — [MINOR]
- **File(s)**: `estimation/gate_pnp.py:106-107`, `estimation/gate_pnp.py:144-145`
- **Issue**: Direct construction `GatePnPEstimator()` still uses `from_fov(90, 640, 480)` and `GateGeometry(1.2, 1.2)`. `RacePipeline` overrides via config, but tests/tools importing defaults hit wrong geometry.
- **Repro**: `GatePnPEstimator().estimate_gate_pose(...)` in a notebook — 20% PnP scale error vs AIGP gates.
- **Fix sketch**: Default `GatePnPEstimator()` to `CameraIntrinsics()` + `GateGeometry(AIGP_GATE_INTERIOR_M, ...)`.
- **Confidence**: high

### F13. Calibration `lstsq` has no rank/degeneracy guard — [MINOR]
- **File(s)**: `competition/calibration.py:90-108`, `tests/test_calibration.py:71-77`
- **Issue**: If thrust commands are constant (hover-only samples) or collinear, `np.linalg.lstsq` can return unstable coefficients (NaN/inf) with no validation. Tests only cover well-conditioned synthetic spread.
- **Repro**: Feed 200 samples with `thrust_normalized=0.5` identical; inspect `thrust_per_mass` for NaN.
- **Fix sketch**: Check `rank(X) == 2` and finite residuals; raise `ValueError` with actionable message; add test.
- **Confidence**: medium

### F14. AIGP drone footprint constants unused in planning/collision — [MINOR]
- **File(s)**: `competition/aigp_geometry.py:28-30`, `planning/racing_line.py`, `gate_sequencing/sequencer.py`
- **Issue**: 280 mm chassis constants are defined but never applied to racing-line clearance, `pass_through_margin`, or crash annulus sizing. PyBullet still uses ~92 mm CF2X model.
- **Repro**: Compare trajectory lateral offset vs 0.14 m half-width drone at 1.5 m gate—no footprint inflation in planner.
- **Fix sketch**: Thread `AIGP_DRONE_WIDTH_M` into racing-line obstacle margin and/or sequencer opening shrink; document sim-vs-AIGP gap until SITL calibration lands.
- **Confidence**: high

### F15. Deferred A12/A16/A17 leave competition path split and unvalidated — [MINOR]
- **File(s)**: `.loop/synthesis/iter_001.md:94-99`, `sim_pybullet/runner.py`, (missing) `scripts/run_pipeline_pybullet.py`, (missing) `scripts/benchmark_matrix.py`
- **Issue**: Iter-001 explicitly deferred runner↔pipeline collapse, MAVLink command tests, and multi-track matrix. `race_pipeline.py` and `sim_pybullet/runner.py` still diverge; iter-002 inherits two autonomy stacks plus no `SET_ATTITUDE_TARGET` regression lock.
- **Repro**: Fix a bug in one path only; bench green on PyBullet runner while MAVLink pipeline regresses.
- **Fix sketch**: Prioritize A16 thin entry + A12 in iter-002 before more ILC tuning; matrix must include `aigp_default.json` once placeholder exists.
- **Confidence**: high

### F16. `_delivered_ids` cap allows late duplicate `frame_id` after ~64 frames — [NIT]
- **File(s)**: `competition/vision_udp.py:172-233`
- **Issue**: Delivered-ID ring is `max_buffered_frames * 8` (64). After 64 newer frames, an old `frame_id` can re-enter `_buffers` and be delivered twice if the sender reuses IDs (uint32 wrap far away, but replay/testing can collide).
- **Repro**: Complete frames 1..70; re-send chunk for frame 5; may be reassembled again.
- **Fix sketch**: Use bounded `collections.deque` + `set` for O(1) membership, or document ID reuse policy; test wrap/reuse.
- **Confidence**: low (competition likely monotonic frame_id)

### F17. Tracker residual clamp composition is correct; `SimplePositionTracker` has no parity — [NIT]
- **File(s)**: `control/mpc_tracker.py:214-245`, `control/mpc_tracker.py:268-331`
- **Issue**: Geometric path applies residual clamp then `max_tilt_rad`—no 0.9 rad leak. `SimplePositionTracker` bypasses residual entirely (expected), but switching `use_geometric_tracker=False` silently drops ML path.
- **Repro**: Toggle tracker mode in config; residual training investment has no effect.
- **Fix sketch**: Document in `PipelineConfig`; or share clamp helper if simple tracker is still used in any bench mode.
- **Confidence**: high (clamp OK); low (operational risk)

## Things iter-001 got right
- Sequencer out-of-order DQ (`enforce_in_order=True` default) with focused adversarial tests in `gate_sequencing/tests/test_sequencer_adversarial.py` — U-turn false-complete pattern is actually closed.
- `competition/aigp_geometry.py` centralizes gate/camera/timing constants; `GateSpec` defaults moved to 1.5 m / 0.6 m border with legacy override path preserved.
- Benchmark honesty surface (`crashed`, `disqualified`, `dq_reason`, `last_crash_gate`, threshold_failures) wired on both synthetic and PyBullet loops.
- ILC wall-clock literals removed from `scripts/benchmark.py` default path; curvature partition + `config/ilc_defaults.json` with explicit `race_01` JSON overrides.
- Vision UDP wire format + reassembly tests are thorough (OOR chunks, GC, duplicates); tracker residual is off-by-default with hard clamps and solid unit contract.

## What I did NOT review
- Full `git show` hunks for all six commits (reviewed current tree state instead).
- `gate_detection/` stack (still predominantly 640×480 assumptions).
- `scripts/train_tracker_residual.py`, holdout eval numbers, and `control/residual_weights.npz` quality.
- Live PyBullet benchmark execution output / `benchmark_history.jsonl` trends.
- `sim_pybullet/runner.py` replan path line-by-line vs `race_pipeline._maybe_replan`.
- DCL Windows binary / real SITL (correctly deferred).
- Deferred A12 MAVLink command tests (files not present).
