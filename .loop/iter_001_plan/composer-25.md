# Iter 001 Plan — Composer 2.5

## Summary

Ship adversarial tests first so iteration 001 has falsifiable bars, then fix sequencer and benchmark honesty (I-1, I-2), centralise AIGP geometry and camera defaults (I-5, I-8), and replace race_01 wall-clock ILC windows with curvature-derived section boundaries plus a multi-track ILC regression harness (I-3, I-4). Land Vision UDP reassembly on port 5600 (I-9) and a thin PyBullet→`RacePipeline` entry path without deleting `runner.py` yet (I-7 deferred to scaffold). Add a tiny sim-trained corner regressor that feeds existing PnP (I-10).

## Actions (ordered)

### A1. Adversarial sequencer tests (out-of-order pass)
- **File(s)**: `tests/test_sequencer_adversarial.py` (new)
- **Change**: Add `test_out_of_order_future_gate_crossing_is_terminal_fail`: build 3-gate line, set `current_idx=0`, step segment that crosses gate-2 plane inside opening before gate-1 is passed; assert `out_of_order` flag / `RaceState` terminal fail and `passed_gate_ids == []`. Add `test_uturn_cannot_credit_skipped_gates`: cross gates 4–6 while current=2, return to gate-2, assert `is_complete` is False and skipped indices recorded.
- **Rationale**: I-11 + I-1 — tests define the bar before the fix lands; mirrors charter gate-order enforcement.
- **Test**: `test_out_of_order_future_gate_crossing_is_terminal_fail`, `test_uturn_cannot_credit_skipped_gates` (must fail on current code, pass after A3)
- **Risk**: Over-strict geometry may false-fail legitimate wide cuts; use opening margin consistent with `pass_through_margin`.
- **Effort**: S

### A2. Adversarial benchmark / crash tests
- **File(s)**: `tests/test_benchmark_adversarial.py` (new), `gate_sequencing/sequencer.py:240-261` (reference for crash zone math)
- **Change**: Add `test_synthetic_frame_crossing_inside_border_aborts_run`: kinematic two-step segment through gate frame strut (between bare opening and outer frame) must set `crashed=True` and `overall_passed=False` once synthetic bench is fixed. Add `test_synthetic_ground_crash_aborts`: z<0.05 path. Add `test_pybullet_gate_contact_aborts` (skip if no pybullet) mirroring `scripts/benchmark.py:746-750`.
- **Rationale**: I-11 + I-2 — benchmark PASS is meaningless until these fail then pass.
- **Test**: `test_synthetic_frame_crossing_inside_border_aborts_run` (fails until A4)
- **Risk**: Flaky numeric thresholds on plane-crossing interpolation; use analytic crossing at gate centre plane.
- **Effort**: S

### A3. Enforce in-order gate passing in sequencer
- **File(s)**: `gate_sequencing/sequencer.py:155-294`, `gate_sequencing/sequencer.py:51-58` (extend `RaceState`)
- **Change**: In `update()`, after building `pos`/`prev_pos`, loop all gates with `sequence_index >= _current_idx` (unpassed). For each, if plane crossed and `_point_in_gate_opening(crossing, gate)` while `gate.sequence_index != _current_idx`, set `_state = RaceState.OUT_OF_ORDER` (new enum), append to `_out_of_order`, return None immediately. Only credit pass on `_current_idx`. Expose `is_failed` / `failure_reason` properties for bench and pipeline.
- **Rationale**: I-1 — closes U-turn false-complete path described in `2_known_issues.md`.
- **Test**: A1 tests pass; existing `gate_sequencing/tests/test_sequencer.py` must stay green
- **Risk**: Legitimate wide racing lines that clip a future gate opening may false-trigger; tune with `pass_through_margin` only on current gate, strict opening on future gates.
- **Effort**: M

### A4. Terminal crash detection in synthetic benchmark
- **File(s)**: `scripts/benchmark.py:409-425`, `scripts/benchmark.py:233-450` (`run_synthetic_benchmark` loop)
- **Change**: Each step, call shared helper `classify_segment_gate_interaction(prev_pos, pos, gate_specs, seq)` that reuses sequencer plane-crossing logic (extract to `gate_sequencing/collision.py` or static methods on `GateSequencer`). On frame-strut crossing, `seq.mark_collision(gate_id, crossing)` and break with `crashed=True`, `termination_reason="crash_gate:..."`. Treat `seq.state == OUT_OF_ORDER` as fail. Set `overall_passed` false when `crashed` or out-of-order.
- **Rationale**: I-2 — synthetic bench matches PyBullet semantics (`benchmark.py:746-750`).
- **Test**: A2 `test_synthetic_frame_crossing_inside_border_aborts_run`
- **Risk**: Duplicate logic drift vs sequencer; mitigate by single shared geometry helper.
- **Effort**: M

### A5. Centralise AIGP gate geometry constants
- **File(s)**: `competition/aigp_geometry.py` (new), `gate_sequencing/sequencer.py:34-39`, `race_pipeline.py:72-74`, `estimation/gate_pnp.py:74-75`, `sim_pybullet/_gate_to_spec.py:24-32`, `sim_pybullet/configs/*.json` (remove overrides only where they hard-code 1.2)
- **Change**: Define `AIGP_GATE_INTERIOR_M = 1.5`, `AIGP_GATE_BORDER_M = 0.6`, `AIGP_GATE_DEPTH_M = 0.26`, `AIGP_DRONE_FOOTPRINT_M = 0.28`. Default `GateSpec` / `GateGeometry` / `PipelineConfig.gate_width|height` / `GateWaypoint` to these. Add `gate_spec_from_config_json(gd)` loader used by benchmark and pipeline. Update `race_01.json` gate_defaults to AIGP values (geometry truth, not course-specific tuning).
- **Rationale**: I-5 — inner 1.5 m per `1_aigp_spec_distill.md`; fixes empty geometric crash zone when border was 0.15 m.
- **Test**: `tests/test_aigp_geometry_defaults.py` — assert `GateSpec().interior_width == 1.5` and `outer_width == 2.7`
- **Risk**: PyBullet gate meshes sized for 1.2 m may need visual scale update in `sim_pybullet/gate_models.py` (separate visual-only change, do not touch physics constants in `sim_pybullet/drone.py`).
- **Effort**: M

### A6. AIGP camera intrinsics and 20° body pitch
- **File(s)**: `race_pipeline.py:67-70`, `estimation/gate_pnp.py:33-56`, `competition/adapter.py:145-146`, `competition/pybullet_adapter.py:129-140`
- **Change**: Set `PipelineConfig.image_height=360`, `camera_fov_h=90`, add `camera_pitch_deg: float = 20.0`. Extend `CameraIntrinsics` with `body_R_camera` (pitch +20° about body Y). Apply in `GatePnPEstimator` world projection and `detect_gate_corners` ray model. Update `CameraFrame` default height to 360; PyBullet adapter decodes FPV at 640×360 when env provides it.
- **Rationale**: I-8 — matches VADR spec fx=fy=320, cy=180 (`1_aigp_spec_distill.md`).
- **Test**: `tests/test_camera_aigp_intrinsics.py` — fx==320, cy==180, pitch rotation maps body +X forward to camera optical axis tilt
- **Risk**: Existing 640×480 detector training images need letterbox or re-export; gate_detection tests use 480 — update fixtures, not detector architecture.
- **Effort**: M

### A7. Curvature-derived ILC section boundaries
- **File(s)**: `planning/ilc_sections.py` (new), `scripts/benchmark.py:318-333`, `planning/trajectory_optimizer.py:184-250`
- **Change**: Add `derive_section_boundaries(trajectory, dt, n_sections=4) -> list` that samples `(t, κ)` along the optimised path, finds curvature peaks (local maxima above median+σ), and maps peaks to step indices for `section_boundaries` tuples `(start, end, alpha, max_corr, cutoff_hz, vel_scale)` with **fixed** hyperparameters from `config/ilc_defaults.yaml` (not race_01 seconds). Replace `inflection_start`/`helix_start` literals in `run_synthetic_benchmark`.
- **Rationale**: I-3 — removes `2.0/dt`, `7.4/dt` race_01 magic (`benchmark.py:322-324`).
- **Test**: `tests/test_ilc_sections.py` — synthetic S-curve trajectory yields boundaries monotonic in arc length; swapping gate order changes boundaries (proves not wall-clock)
- **Risk**: Too few curvature peaks on smooth tracks → fall back to equal-time quartiles (documented fallback, not course names).
- **Effort**: M

### A8. Multi-track ILC regression + externalised defaults
- **File(s)**: `config/ilc_defaults.yaml` (new), `scripts/benchmark.py:344-350`, `tests/test_ilc_regression.py` (new)
- **Change**: Move `convergence_threshold`, `momentum_gamma`, `alpha`, `max_iterations` to YAML. Add `scripts/ilc_regression.py` that runs synthetic bench on ≥3 configs: `race_01.json`, `slalom.json`, `figure8.json` (or generated random gate layouts), asserts ILC reduces mean tracking error on each without increasing crashes. Bench `--mode unit` includes lightweight ILC regression (cap 30s).
- **Rationale**: I-4 — parameters are no longer justified only by race_01 sweep notes at `benchmark.py:344-350`.
- **Test**: `test_ilc_regression_three_tracks` (may `@pytest.mark.slow`)
- **Risk**: Long CI time; gate regression behind env flag `RUN_ILC_REGRESSION=1`.
- **Effort**: M

### A9. Vision UDP receiver (port 5600)
- **File(s)**: `competition/vision_udp.py` (new), `competition/mavlink_bridge.py:163-167`, `competition/session.py:181`
- **Change**: Implement `VisionUdpReceiver` parsing 24-byte LE header (`frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, `sim_time_ns`), reassemble JPEG by `frame_id`, decode to BGR `CameraFrame` at 640×360. Start asyncio datagram task on `connect()`; `get_camera_frame()` returns latest complete frame with sim timestamp. Add unit test with crafted chunked payloads.
- **Rationale**: I-9 — spec requires separate vision UDP (`1_aigp_spec_distill.md:37-38`); bridge currently returns `None` (`mavlink_bridge.py:163-167`).
- **Test**: `tests/test_vision_udp_reassembly.py`
- **Risk**: Endianness / chunk loss; drop incomplete frames, log gap count.
- **Effort**: M

### A10. Wire MAVLink bridge + session to vision receiver
- **File(s)**: `competition/mavlink_bridge.py:74-114`, `competition/mavlink_bridge.py:163-167`, `race_pipeline.py` (detection_active when frame non-None)
- **Change**: Construct `VisionUdpReceiver(port=5600)` in `MAVLinkBridge.__init__`; start/stop with connect/disconnect. Plumb `sim_time_ns` into pipeline clock for EKF predict. Add `scripts/mavlink_smoke.py` that connects to `udp://:14540`, prints telemetry Hz and vision FPS for 5s (manual / CI skip).
- **Rationale**: I-9 — makes DCL sim path runnable without PyBullet.
- **Test**: `test_vision_udp_reassembly.py` + manual smoke script documented in `competition/README.md` snippet
- **Risk**: MAVSDK not installed in CI — smoke marked optional.
- **Effort**: S

### A11. Thin PyBullet harness via RacePipeline (runner collapse scaffold)
- **File(s)**: `scripts/run_pipeline_pybullet.py` (new), `competition/pybullet_adapter.py`, `race_pipeline.py:93-120`, `sim_pybullet/runner.py:152-298` (read-only reference)
- **Change**: New entry: load JSON config → `DroneRaceEnv` + `GPDDrone` → `PyBulletAdapter` → `RacePipeline.configure(gates)` → `RacePipeline.run()`. Wire `env.gate_contact()` into adapter callback that calls `pipeline.sequencer.mark_collision`. **Do not delete** `runner.py` or change default `python -m sim_pybullet.runner` behaviour this iter.
- **Rationale**: I-7 — full collapse of 1089-line `runner.py` ad-hoc stack is **L** effort (duplicate `RacingLine`, detection heuristics, DSLPID). Scaffold proves shared pipeline works; deletion is iter 002+.
- **Test**: `tests/test_pipeline_pybullet_smoke.py` — 5s metadata-mode run, `gates_passed >= 1`, no crash
- **Risk**: NED/ENU frame bugs in adapter; verify against existing `pybullet_adapter` telemetry path.
- **Effort**: M

### A12. Tiny corner regressor (ML) for PnP
- **File(s)**: `gate_detection/ml/corner_net.py` (new), `gate_detection/ml/train_corners.py` (new), `race_pipeline.py` (wire after `detect_gate_corners`), `scripts/export_corner_onnx.py` (new)
- **Change**: MobileNetV2-0.25 backbone, 4×2 output (normalised corner offsets inside gate crop). Training data: `scripts/extract_corner_dataset.py` renders PyBullet FPV crops with GT corners from gate projection (10k–30k crops). Loss: L1 on corners. At runtime: if classical `detect_gate_corners` fails but Phase1 bbox exists, run ONNX regressor on crop → feed PnP. Ship `gate_detection/ml/corner_reg_v1.onnx` <2MB.
- **Rationale**: I-10 — cheapest path with material gain: improves PnP availability without replacing full YOLO stack; aligns with Romero 2025 corner→PnP pattern.
- **Test**: `tests/test_corner_net_inference.py` — synthetic crop → corners within 5 px of GT; pipeline integration test with mocked ONNX
- **Risk**: Sim-to-real gap; gate confidence gating must reject high reprojection error (existing PnP RANSAC).
- **Effort**: M

### A13. Benchmark honesty in overall_passed aggregation
- **File(s)**: `scripts/benchmark.py` (main `run_benchmark` / threshold section), `tests/test_benchmark_adversarial.py`
- **Change**: `overall_passed` requires: no crash, no `OUT_OF_ORDER`, `gates_passed == total_gates`, plus existing tracking thresholds. Add explicit JSON fields `sequencer_failure`, `honesty_checks_passed`. Document in `CLAUDE.md` that synthetic PASS requires adversarial suite green.
- **Rationale**: Charter + I-2 — closes false PASS after A3/A4.
- **Test**: `test_overall_passed_false_on_out_of_order` using mocked benchmark result dict
- **Risk**: May immediately show all benchmarks failing until tuning — expected and desired.
- **Effort**: S

### A14. SITL calibration harness stub (drone dynamics)
- **File(s)**: `competition/calibration.py` (new), `config/drone_calibration.json` (generated, gitignored template committed as `.example`)
- **Change**: Async script: connect MAVLink, hover 3s, step thrust ramp, log thrust→accel, fit `mass`, `max_thrust`, `drag` via least squares; write JSON consumed by `DroneConstraints` / `TrackerConfig` in `RacePipeline`. No PyBullet mass change.
- **Rationale**: I-6 — acknowledges 280 mm drone vs CF2X; unblocks competition-facing gains without sim physics edits.
- **Test**: `tests/test_calibration_fit_synthetic_logs.py` — feed recorded CSV, assert sane mass in [0.5, 5.0] kg
- **Risk**: Cannot fully test without DCL binary; stub stays optional behind `--calibrate`.
- **Effort**: M

## ML choice

**Tiny corner regressor (MobileNetV2-0.25 head)** on Phase1 gate crops, exported to ONNX.

- **Why not full YOLO retrain**: `gate_detection/training/` exists but runner/pipeline do not load ONNX; integrating a 4-corner head on existing bbox is smaller and directly fixes PnP dropout.
- **Why not tracker/EKF residual**: ILC and geometric tracker failures are largely reference/sequencing honesty issues this iter; learned residuals would learn the wrong signal from a lying bench.
- **Data path**: PyBullet FPV (`sim_pybullet` env) → crop by bbox → normalised 128×128 RGB → L1 loss vs projected gate corners.
- **Train/eval**: offline `train_corners.py` 20 epochs; hold out 20% by gate_id; metric mean corner px error <8 on val; deploy ONNX; runtime gate on reprojection error <2 px (PnP already has RANSAC in `gate_pnp.py`).

## What NOT to do this iter

- **Delete or gut `sim_pybullet/runner.py`** — 1089-line parallel autonomy stack; collapse without scaffold risks regressing the only path that reproduces gate-7 helix debugging (PLAN.md reproducer).
- **Modify `sim_pybullet/` physics / drone mass** — charter forbids treating sim as ground truth for AIGP drone; calibration JSON only.
- **Retune ILC alphas for race_01 only** — violates no-magic-numbers rule; only A7/A8 generalised parameter paths.

## Open questions for the synthesiser

1. **Runner collapse depth**: Is A11 (parallel `run_pipeline_pybullet.py`) sufficient for iter 001, or should one agent attempt in-place replacement of `RaceRunner.run()` loop (high conflict risk with `--use-detection` path)?
2. **ML vs ONNX fast path**: If `gate_detection/training/runs/gate_pose_v1` ONNX already has 4 keypoints, should iter 001 wire that instead of training `corner_reg_v1` (faster) vs strict brief wording "tiny CNN"?
3. **Out-of-order strictness**: Should a future-gate crossing **outside** the opening count as out-of-order, or only crossings **inside** the opening (current plan: inside only, outside → existing `miss` on current gate)?
4. **ILC fallback**: Equal-time quartiles vs single global ILC when curvature peaks < `n_sections-1` — prefer simpler global ILC for iter 001 stability?
