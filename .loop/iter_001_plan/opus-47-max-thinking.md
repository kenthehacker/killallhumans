# Iter 001 Plan — Opus 4.7 Max-Thinking

## Summary
Make every bench PASS mean PASS first (adversarial tests + sequencer order
enforcement + synthetic-bench geometric crash), then strip race_01-isms from
the bench and snap GateSpec / CameraIntrinsics / GateGeometry to the
authoritative AIGP numbers (1.5 m opening, 0.6 m border, 0.26 m depth,
640×360 camera tilted 20° up). Land the JPEG-chunk vision-UDP receiver on
:5600 so MAVLink2 ↔ DCL is talkable. Ship a single, cheap, falsifiable ML
piece — a learned tracker residual trained on a 3-track corpus, gated off
by default and bounded by hard clamps so it cannot regress anything.
Defer the `runner.py` ↔ `race_pipeline.py` collapse and the SITL drone
calibration — both are high-blast-radius and we don't have the DCL binary
in-hand to validate against yet.

## Actions (ordered)

### A1. Land adversarial test suite FIRST (the bar for everything else)
- **File(s)**:
  - `tests/test_sequencer_adversarial.py` (new)
  - `tests/test_benchmark_adversarial.py` (new)
  - `tests/test_camera_geometry.py` (new)
  - `tests/test_vision_udp.py` (new)
  - `tests/test_tracker_residual.py` (new, stub until A9 lands)
- **Change**:
  Write the *failing* tests for each downstream fix before any
  implementation lands. Concrete cases the suite must contain:

  Sequencer adversarial (`test_sequencer_adversarial.py`):
  1. `test_out_of_order_dq_on_skip_forward`: build a 5-gate course, current
     target = G1; fly straight through G3's opening (with G1, G2 still
     unpassed). After fix: `seq.is_disqualified is True`,
     `seq.dq_reason == "out_of_order"`. Today: silently no-ops.
  2. `test_u_turn_skip_then_recover_is_still_dq`: pass G1, then fly through
     G3, then U-turn back through G2, then through G3 again. Today:
     `seq.is_complete` becomes True with all gates passed (false-positive
     clean run). After fix: DQ on the first G3 crossing.
  3. `test_far_plane_grazing_does_not_dq`: drone yaws 45° at G2 and its
     trajectory crosses G4's *infinite plane* but the crossing point lies
     outside G4's outer frame. Must not DQ — the gate-frame opening test
     keeps it benign.
  4. `test_aigp_gate_dimensions_apply`: build a GateSpec with default
     constructor → `interior_width == 1.5`, `border_width == 0.6`,
     `depth == 0.26`. Locks the I-5 fix in.

  Synthetic bench adversarial (`test_benchmark_adversarial.py`):
  5. `test_geometric_crash_terminates_synthetic_bench`: inject a kinematic
     trajectory that grazes G3's frame border (segment crossing in the
     1.5 m..2.1 m annulus). Expect `crashed=True` and
     `termination_reason="crash_gate:gate-3"`. Today: silently passes.
  6. `test_floor_clip_no_longer_only_crash`: synthetic run with a
     planar trajectory that never drops below z=0.05 but does scrape a
     gate frame must still report `crashed=True`.
  7. `test_no_race01_magic_constants_in_default_path`: greps the bench
     source for `int(2.0 / dt)`, `int(4.4 / dt)`, `int(7.4 / dt)`,
     `convergence_threshold=0.0005`, `momentum_gamma=0.2` and fails if
     they appear outside a `race_01.json`-keyed override block.

  Camera geometry (`test_camera_geometry.py`):
  8. `test_intrinsics_default_to_aigp`: `CameraIntrinsics()` returns
     `fx=fy=320, cx=320, cy=180, width=640, height=360`.
  9. `test_camera_tilt_propagates_to_pnp`: synthesize a gate directly in
     front of the drone at 4 m range; place the gate so that, with no
     tilt, it projects at image center (cx, cy). Apply the AIGP 20° upward
     tilt to the camera; the gate should now project *below* the image
     center by ~`f·tan(20°) = 116 px`. PnP recovery of drone position must
     remain within 0.05 m of ground truth regardless of tilt.

  Vision UDP (`test_vision_udp.py`):
  10. `test_chunk_reassembly_in_order`: feed 4 in-order chunks of one JPEG
      frame; receiver yields one CameraFrame whose `image.shape ==
      (360, 640, 3)`.
  11. `test_chunk_reassembly_out_of_order`: same payload, chunks delivered
      [2, 0, 3, 1]; must still yield exactly one frame.
  12. `test_partial_frame_timeout`: deliver 3 of 4 chunks then wait
      `> reassembly_timeout_ms`; partial buffer must be dropped, no leak.
  13. `test_duplicate_frame_id_does_not_deadlock`: feed `frame_id=7`
      twice; receiver does not block, returns the first complete frame
      once.

  Tracker residual (`test_tracker_residual.py`, stub for A9):
  14. `test_residual_clamp`: feed a residual model whose raw output is
      `(10.0, 10.0, 10.0)`; tracker output deltas must clamp to
      `(±0.05 rad, ±0.05 rad, ±0.05)`.
  15. `test_residual_off_is_baseline`: with `use_residual=False`, tracker
      output must be byte-identical to today's baseline.

- **Rationale**: I-11. Per charter, the bench is suspect; PASS means
  nothing without an independent adversarial test. These tests *fail*
  on today's main and must turn green as the corresponding fixes land.
- **Test**: this action's deliverable *is* the tests.
- **Risk**: test bar set too high causes downstream fixes to ping-pong;
  mitigated by writing each test against a concrete behavioural claim
  pulled straight from `2_known_issues.md`.
- **Effort**: M

### A2. Sequencer: enforce in-order passing, surface DQ as terminal
- **File(s)**:
  - `gate_sequencing/sequencer.py:155-294` (`update`)
  - `gate_sequencing/sequencer.py:51-58` (`RaceState` — add `DISQUALIFIED`)
  - `gate_sequencing/sequencer.py:60-74` (`SequencerConfig` — add
    `enforce_in_order: bool = True`)
- **Change**:
  1. Add `RaceState.DISQUALIFIED` and a `_dq_reason: Optional[str]`
     field.
  2. In `update()`, after handling the current gate's plane crossing,
     iterate `self._gates[self._current_idx + 1:]` and for each
     still-unpassed gate compute `_plane_was_crossed` *and*
     `_point_in_gate_opening` on the crossing point. If any such gate is
     "passed-through-but-out-of-order", set
     `self._state = RaceState.DISQUALIFIED`,
     `self._dq_reason = f"out_of_order:{gate.gate_id}"`, append to
     `self._crashes` (so downstream `_replanner` / metrics still see a
     terminal event), and return None.
  3. Add `@property is_disqualified` and `@property dq_reason`. Update
     `is_complete` to return `self._current_idx >= len(self._gates) and
     self._state != RaceState.DISQUALIFIED`.
  4. Bench/runner integrators: treat `is_disqualified` exactly like
     `crashed=True` (terminate the loop, mark failure).
- **Rationale**: I-1. The fix in `2_known_issues.md` literally calls for
  this — "track plane crossings of every still-unpassed gate per tick".
  Survey paper (Hanover 2023, autonomous drone racing) explicitly defines
  scoring as ordered gate traversal; AIGP rules echo that.
- **Test**: A1 cases 1, 2, 3.
- **Risk**: existing happy-path replay test
  (`test_race_pipeline_replan_integration.py`) may rely on
  `update()` returning None benignly for far-plane crossings. The
  point-in-opening guard in step (2) avoids that.
- **Effort**: M

### A3. Synthetic bench: terminate on geometric gate-frame crash
- **File(s)**:
  - `scripts/benchmark.py:417-425` (the ground/ceiling-only crash block)
  - `scripts/benchmark.py:296` (where `seq` is built — feed it crash
    callback if needed)
- **Change**: Just *call the sequencer*. Replace the floor/ceiling-only
  block with a call to `seq.update(tuple(pos))` followed by a check on
  `seq.last_crash` and `seq.is_disqualified`. The geometric crash
  classification already exists in
  `gate_sequencing/sequencer.py:240-262` (P1-6 branch). The synthetic
  bench was simply never reading it. Then add `crashed=True` /
  `termination_reason=f"crash_gate:{gate_id}"` and `break` when
  `seq.last_crash` is set, mirroring the PyBullet path
  (`benchmark.py:746-750`).
- **Rationale**: I-2. The detection code is already there; the bench
  ignores it. Zero new geometry; one if-branch.
- **Test**: A1 cases 5, 6.
- **Risk**: synthetic kinematic sim's noise might tip false crashes if
  the trajectory points-of-contact graze the new (larger, 1.5 m) opening
  in iter 28's ILC-corrected reference. Mitigation: A4's ILC clipping
  cap `max_correction_m=0.15` already bounds the worst offset to a
  sub-frame value; the bench should now show *real* crashes that are
  worth fixing.
- **Effort**: S

### A4. Snap GateSpec / GateGeometry / CameraIntrinsics / CameraFrame / PipelineConfig defaults to AIGP geometry
- **File(s)**:
  - `gate_sequencing/sequencer.py:27-48` (GateSpec)
  - `estimation/gate_pnp.py:34-93` (CameraIntrinsics + GateGeometry)
  - `estimation/gate_pnp.py:182-226` (gate_pose_to_drone_position —
    consume camera tilt)
  - `race_pipeline.py:62-91` (PipelineConfig)
  - `planning/trajectory_optimizer.py:43-50` (GateWaypoint defaults)
  - `competition/adapter.py:140-147` (CameraFrame — defaults 640×360)
  - New file: `sim_pybullet/configs/aigp_default.json` (AIGP-geometry
    track; race_01 keeps its 1.2 m legacy values via explicit override)
- **Change**:
  1. `GateSpec` defaults → `interior_width=1.5, interior_height=1.5,
     border_width=0.6, depth=0.26`.
  2. `GateGeometry` defaults → `interior_width_m=1.5,
     interior_height_m=1.5`. Drop `Phase1GateDetector.GATE_WIDTH_METERS
     = 1.0` magic too (`gate_detection/src/phase1_detector.py:52-53`)
     and read from `GateGeometry`.
  3. `CameraIntrinsics` defaults → `fx=320, fy=320, cx=320, cy=180,
     image_width=640, image_height=360`. Add a new field
     `pitch_offset_rad: float = math.radians(20.0)` (positive = camera
     pitched *up* in body frame).
  4. `CameraIntrinsics.from_fov` → keep API but accept an optional
     `vfov_deg` arg for AIGP's square FoV; the existing single-FoV
     formula is fx-only and gets fy wrong by ratio when aspect != 1.
  5. `gate_pose_to_drone_position` currently accepts `drone_orientation`
     but never uses it for the camera-to-body transform — the body↔IMU
     identity claim in `1_aigp_spec_distill.md` is fine but the
     20° upward camera tilt is NOT identity. Fix: apply
     `R_body_camera = R_pitch(camera.pitch_offset_rad)` to `R_gc.T` so
     the recovered drone position lives in body NED, then rotate into
     world by drone yaw/pitch/roll.
  6. `PipelineConfig` defaults → `image_width=640, image_height=360,
     camera_fov_h=90.0, gate_width=1.5, gate_height=1.5,
     camera_pitch_offset_rad=math.radians(20.0)`.
  7. `GateWaypoint` defaults → `width=1.5, height=1.5`.
  8. `CameraFrame` defaults → `width=640, height=360`.
  9. Add `sim_pybullet/configs/aigp_default.json` with 6 representative
     gates at AIGP geometry (don't copy race_01's helix verbatim — the
     point of this config is "a track the AIGP spec actually allows").
  10. Sweep callers: `gate_pnp.GateGeometry(self.config.gate_width,
      self.config.gate_height)` in `race_pipeline.py:120` continues to
      override per-pipeline; `race_01.json` continues to override per
      its `gate_defaults` block. Defaults shift; explicit overrides
      keep race_01 reproducible.
- **Rationale**: I-5, I-8 (and the 20° tilt sub-issue of I-8 that
  `gate_pose_to_drone_position` silently swallows). Per
  `1_aigp_spec_distill.md` §6: "Gate inner-opening 1.5 m" and §3: camera
  is 640×360, fx=fy=320, cx=320, cy=180, tilted 20° upward.
- **Test**: A1 cases 4, 8, 9.
- **Risk**: ILC offsets in `scripts/benchmark.py:317-350` were tuned
  against the 1.2 m-opening race_01 geometry. With the new
  1.5 m default, the ILC corrections may not shrink the larger opening
  margin enough to clear A3's crash test on edge cases. Mitigation:
  race_01.json keeps its 1.2 m override (legacy ground truth); aigp
  defaults apply only to the new aigp_default.json track and any future
  benchmark.
- **Effort**: M

### A5. Remove race_01 magic time-windows; partition ILC by curvature, not wall-clock
- **File(s)**:
  - `scripts/benchmark.py:310-355` (the `section_boundaries` table and
    `compute_ilc_offset_table` invocation)
  - New file: `planning/ilc_section_partition.py` containing
    `partition_by_curvature(trajectory, dt) -> List[Tuple[int, int,
    float, float, float, float]]`.
- **Change**:
  1. Extract `inflection_start = int(2.0/dt)`, `inflection_end =
     int(4.4/dt)`, `helix_start = int(7.4/dt)` — these are wall-clock
     constants tuned to race_01.
  2. Replace with `partition_by_curvature`: walk
     `trajectory.points[i].acceleration` magnitudes, find local maxima
     above `curv_peak_thresh = max(|a|)·0.6`, and split the trajectory
     into sections of equal "curvature class" (low / medium / high).
     Each class maps to a default `(alpha, max_correction_m,
     filter_cutoff_hz, vel_scale)` tuple — *not* per-step magic.
  3. Allow `race_01.json` (and only race_01.json) to opt in to the old
     per-step schedule via an explicit
     `"ilc_section_overrides": {...}` block; the schedule is *additive*
     legacy, not a default.
  4. `convergence_threshold` and `momentum_gamma` become fields on
     `ILCConfig` (defaults preserved). race_01 keeps its sweep-tuned
     values via an override; aigp_default.json gets the gentler
     defaults.
- **Rationale**: I-3, I-4. Hard requirement from charter: "No
  course-specific magic numbers." The curvature-based partition
  generalises to figure8/slalom/grand_tour without re-tuning.
- **Test**: A1 case 7 (greps for magic constants). Also: A10's
  multi-track regression suite must show no track loses > 25% gate
  pass-rate vs its pre-fix baseline.
- **Risk**: curvature classification overfits to high-jerk tracks; for
  smooth tracks (figure8) the partition may degenerate to one global
  section. That's fine — global ILC is the *safe* fallback in
  iter 0 (before iter 28's per-section schedule), per the Bristow &
  Alleyne 2007 baseline cited in `benchmark.py:315`.
- **Effort**: M

### A6. Vision-UDP receiver on :5600 (JPEG chunk reassembly)
- **File(s)**:
  - New file: `competition/vision_udp.py`
  - `competition/mavlink_bridge.py:74-87, 163-167` (wire the receiver
    into `MAVLinkBridge.get_camera_frame`)
  - `competition/adapter.py:140-147` (already touched in A4 for 640×360)
- **Change**:
  1. `VisionUdpReceiver` async coroutine (asyncio Datagram protocol)
     listening on UDP `:5600`. Parses 24-byte header
     (`frame_id u32 LE, chunk_id u16 LE, total_chunks u16 LE,
     jpeg_size u32 LE, payload_size u32 LE, sim_time_ns u64 LE`) per
     `1_aigp_spec_distill.md` §Communication.
  2. Reassembly buffer keyed by `frame_id`; emit a `CameraFrame` once
     all chunks land; drop the buffer after
     `reassembly_timeout_ms = 100` (3 frames at 30 Hz).
  3. JPEG decode via `cv2.imdecode` (already a repo dep) in a thread
     executor so the control loop's 100 Hz pace isn't blocked by
     decode jitter (typical 5-15 ms on ARM/i5 — would chew the budget).
  4. Wire `MAVLinkBridge.get_camera_frame()` to return the latest
     decoded frame (or `None` if no complete frame yet — important so
     `_control_callback` doesn't latch dropout-slowdown).
  5. Add a `port: int = 5600` config field; default to AIGP value.
- **Rationale**: I-9. The bridge today returns `None` from
  `get_camera_frame()` (`mavlink_bridge.py:163-167`); the entire
  perception → EKF correction → trajectory replan branch is dead on
  the competition surface until this lands.
- **Test**: A1 cases 10, 11, 12, 13.
- **Risk**: synchronous JPEG decode on the asyncio thread will starve
  the heartbeat timer. Mitigated by `asyncio.to_thread()`. Also: AIGP
  may use a non-standard JPEG container — if so, the unit test catches
  it on the first SITL test run, not in production.
- **Effort**: M

### A7. Defer `runner.py` ↔ `race_pipeline.py` collapse (I-7) explicitly
- **File(s)**:
  - `PLAN.md` (one-line status update under Phase 5 / "Architecture
    Decision")
  - `sim_pybullet/runner.py:152-170` (single `# TODO(I-7)` block)
- **Change**: a) Add a clear status line saying "collapse deferred
  to iter ≥ 002 — needs the PnP/EKF wiring from race_pipeline.py to
  land first, plus the multi-track regression suite from A10 to be
  green so the collapse can be verified". b) Don't add new behaviour
  to `runner.py` this iter. c) Mark `_target_from_detection` as
  superseded by `RacePipeline._process_detection` so reviewers know
  not to extend the runner path.
- **Rationale**: I-7 is L-effort and the planning brief explicitly
  says "only if cost is low; otherwise leave it for a later iter and
  say so". With the camera config + adversarial tests + sequencer DQ
  + vision-UDP plumbing all churning in iter 001, doing the collapse
  in parallel would mean six concurrent file rewrites in
  `runner.py`. Two-step is safer.
- **Test**: nothing functional; an architecture review check that
  the deferral is documented.
- **Risk**: the two implementations drift further before they're
  merged. Acceptable for one iter.
- **Effort**: S

### A8. Defer DCL/SITL drone calibration (I-6); stub the harness only
- **File(s)**:
  - New file: `competition/calibration.py` containing a `DroneCalibrator`
    class with `identify_mass_thrust(mavlink_bridge, duration_s=10.0)`
    that runs canned step/ramp/hover maneuvres and dumps
    `drone_calibration.json`.
  - `flight_control/types.py` (or wherever the controller config
    lives) — add an optional `calibration_path: Optional[str] = None`
    field that, if set, loads the JSON at config-build time.
- **Change**: implement the *interface* and a numpy-only least-squares
  identifier (no ML). Don't invoke it in CI — there's no DCL binary on
  the worktree. Validation deferred to whichever iter first connects
  the bridge to an actual SITL instance.
- **Rationale**: I-6 names this directly. The MAVLink2 surface in A6
  is the prerequisite; running calibration without a working bridge is
  vapor.
- **Test**: a unit test seeds a known mass/thrust pair, generates
  synthetic step/ramp data, then verifies the identifier recovers the
  parameters within ±10%.
- **Risk**: stub-only is *not enough* for the AIGP runtime — the
  competition drone's mass/thrust will not match our PyBullet CF2X.
  Mitigation: A9's tracker residual buys some robustness without
  needing the calibration to land first.
- **Effort**: S (stub) / L (full validation)

### A9. ML pick — learned tracker residual (tiny MLP, hard-clamped, off-by-default)
- **File(s)**:
  - New file: `control/learned_residual.py` containing
    `TrackerResidualMLP` (numpy-only forward pass, ≤ 1 KB weights).
  - `control/mpc_tracker.py:TrackerConfig` (add
    `use_residual: bool = False`, `residual_weights_path:
    Optional[str] = None`, `residual_clamp_rad: float = 0.05`,
    `residual_thrust_clamp: float = 0.05`).
  - `control/mpc_tracker.py:GeometricTracker.track` (final block —
    if residual enabled, evaluate, clamp, add to roll/pitch/thrust).
  - New file: `scripts/train_tracker_residual.py` (offline trainer).
  - New artifact: `control/residual_weights.npz` (the trained model).
- **Change**:
  - **Model**: 10 → 64 → 3 MLP, `tanh` activation. Inputs:
    `[pos_err_xyz, vel_err_xyz, ref_accel_xyz, gravity_comp_thrust]`
    (10 dims). Outputs: `(delta_roll, delta_pitch, delta_thrust)`
    pre-clamp. ~700 weights. Numpy forward pass; no torch dep at
    inference time.
  - **Training data**: replay the existing benchmark traces across
    `race_01.json`, `figure8.json`, `slalom.json`, `grand_tour.json`
    (held out at training time, used for eval). Each tick logs
    (state, ref, baseline_cmd, *next-tick* position-error reduction).
    Labels: the residual that *would have minimised* the
    leading-tick error, under the small-angle linearisation around
    the baseline cmd.
  - **Training loop**: pure scipy / numpy / no GPU; ≤ 100 epochs;
    L2 weight regulariser; explicit "if residual increases max
    attitude over baseline, penalise" term in the loss.
  - **Inference path**: under `use_residual=True`, the MLP runs at
    100 Hz on the control thread. Forward pass: ~1k flops, < 50 µs.
  - **Hard clamps**: independent of model output, the residual is
    clipped to `±0.05 rad` for attitude and `±0.05` for thrust before
    being added. The clamp is the safety net — even a corrupted
    weight file cannot push commands beyond ±2.9°.
- **Rationale**: I-10. Of the four ML options listed in the brief, the
  tracker residual is the only one that (a) doesn't depend on visual
  data we don't have yet (rules out the tiny CNN corner regressor —
  the YOLOv8n-pose ONNX in `gate_detection/training/` was trained on
  TII-Aerial, not AIGP), (b) has a hard safety clamp that makes its
  failure mode no-worse-than-baseline (rules out learned EKF residual,
  where a bad output corrupts the system of record), (c) trains
  offline from data the benchmark already produces (rules out the
  drone-dynamics regression, which needs SITL access — A8). Research
  backing: NGTC (Pries 2025) shows feedforward residuals beat geometric
  controllers; "Leveling the Playing Field" (Kunapuli 2025) shows the
  feedforward channel is the most impactful single fix. The clamp is
  cribbed from "safe-RL with hard projection" (Berkeley 2024).
- **Test**: A1 cases 14, 15. Plus a synthetic regression: holdout
  `grand_tour.json` avg tracking error must drop by ≥ 3% vs baseline
  with the residual on, and must be byte-identical to baseline with
  the residual off.
- **Risk**: training overfits to PyBullet drone dynamics → useless or
  harmful on the AIGP drone. Mitigated by (i) the hard clamp,
  (ii) `use_residual=False` default, (iii) trained on 3 tracks so it
  doesn't memorise race_01.
- **Effort**: M

### A10. Multi-track regression suite — the "is this fix actually safe?" gate
- **File(s)**:
  - New file: `scripts/benchmark_matrix.py` runs the synthetic + PyBullet
    benches against `race_01.json`, `figure8.json`, `slalom.json`,
    `grand_tour.json`, and `aigp_default.json` (added in A4).
  - New file: `.loop/state/regression_baseline.json` — captures the
    pre-iter-001 metrics so the matrix can compute deltas.
- **Change**: aggregate per-track threshold pass/fail and emit a single
  JSON with `{track: {gate_pass_rate, avg_err, max_err, crashed,
  dq, loop_hz}}`. The PR gate is: no track regresses by more than
  25% on `gate_pass_rate`, none introduces a `crashed=True` that
  wasn't there before, and the new aigp_default.json must reach at
  least 50% gate completion in iter 001 (a low bar; the racing-line
  optimiser hasn't been tuned for it yet).
- **Rationale**: I-4 (over-tuning to one track). Directly addresses
  charter clause "Testbench is suspect" by forcing every iter to
  demonstrate generality across ≥ 4 tracks.
- **Test**: regression matrix is itself the test gate.
- **Risk**: longer CI; mitigated by `--mode quick` flag (3 tracks,
  short duration, ≤ 30 s wall) and `--mode full` (all 5 tracks).
- **Effort**: S

## ML choice
**Learned tracker residual (small MLP, off-by-default, hard-clamped)**
— the cheapest of the four options listed in the brief that also has
a non-vacuous safety story.

Data path:
- Existing `scripts/benchmark.py` synthetic loop already emits a
  controller trace (`controller_trace_summary`, line 561-568); extend
  it to dump the full per-tick `(state, ref, cmd, next-tick err)`
  tuples to `logs/training/<track>_<ts>.npz`.
- Sweep across 3 training tracks: `race_01.json`, `figure8.json`,
  `slalom.json`. Hold out `grand_tour.json` + `aigp_default.json` for
  eval.
- Labels: for each tick, the residual that minimises the *leading*
  position error under the small-angle linearisation of the
  closed-loop dynamics around the baseline command.

Training / eval:
- `scripts/train_tracker_residual.py` — pure numpy/scipy, ≤ 100
  epochs, L2 weight decay, safety regulariser that penalises any
  weight configuration whose `||W·x||_∞` exceeds the clamp on the
  training distribution.
- Eval: holdout tracks must show ≥ 3% avg tracking error reduction at
  no cost in max attitude or thrust.
- Ship the trained weights as `control/residual_weights.npz` (≤ 4 KB).
- Inference: numpy-only forward pass at 100 Hz, < 50 µs/tick.

## What NOT to do this iter
- **Don't swap Phase1GateDetector for the trained YOLOv8n-pose ONNX in
  `gate_detection/training/`.** That ONNX was trained on TII-Aerial
  imagery; AIGP visuals differ. A real swap waits until we have the
  DCL sim binary and can finetune. Until then, the Phase1 HSV pipeline
  is the lesser evil.
- **Don't migrate the geometric tracker to MPCC++.** MPC++-style
  contour control is the right endgame (per PLAN.md Phase 5), but it
  is a controller redesign that should only land *after* iter 001's
  truthful-bench foundation is solid. PLAN.md explicitly says: "no
  controller redesign is accepted unless the same run is already
  stable under a sane target stream".
- **Don't write the SITL calibration end-to-end** (I-6). A8 stubs the
  interface; the actual run needs the DCL binary on the worktree,
  which we don't have. Stub now, validate in iter ≥ 002.
- **Don't tighten the global thresholds in `scripts/benchmark.py:47-56`.**
  They were tightened to "aspirational" already; tightening them
  before the bench is honest just amplifies the false-PASS problem.

## Open questions for the synthesiser
1. **Out-of-order DQ severity.** A2 treats out-of-order pass-through as
   *terminal DQ*. AIGP rules don't formally penalise — "fastest valid
   time" *might* mean "ignore wrong-order runs", *might* mean "DQ".
   Charter §"Hard constraints" says "Gate order is enforced. Skipping a
   gate then coming back to it later is a fail." → terminal. The
   synthesiser should confirm whether other plans agree (and whether to
   gate it behind `SequencerConfig.enforce_in_order=True` for backward
   compatibility on existing tests).
2. **ILC partition strategy.** A5 uses curvature-peak partitioning. An
   alternative is to drop the per-section schedule entirely and rely on
   a single global ILC pass (Bristow & Alleyne 2007 baseline, fewer
   moving parts). Either is acceptable. If the synthesiser sees that
   other plans propose dropping per-section entirely, I'd rather go
   along with that than ship a more complex partition logic that needs
   its own regression coverage.
3. **AIGP camera tilt rotation convention.** A4 step 5 applies
   `R_pitch(+20°)` to map from camera→body. AIGP spec §Coordinate
   frames says body NED has X forward, Y right, Z down, and camera is
   tilted *upward* — so a feature directly ahead at horizon should
   project *below* image center. Confirm the sign matches in the
   chosen rotation convention. (My read: positive pitch = nose up,
   camera optical axis at body pitch + 20°, so world horizon projects
   at `cy + f·tan(20°)` ≈ 296. I'd rather the synthesiser cross-check
   against Composer's plan since this is the most error-prone bit.)
4. **Runner collapse timing.** A7 defers. If another plan thinks the
   collapse is the prerequisite, not a follow-up, the synthesiser
   should resolve before any code lands — once `runner.py` starts
   getting torn down, the half-collapsed state is the worst possible
   place to be for the rest of iter 001's churn.
5. **Where does `aigp_default.json` come from?** A4 step 9 says "6
   representative gates at AIGP geometry"; I don't actually know what
   "representative" looks like until the spec drops the real positions.
   Suggestion: copy race_01's layout *scaled to 1.5 m openings* as a
   placeholder, marked clearly as a placeholder. The synthesiser
   should decide if that's an honest stand-in or if we should leave
   the config blank and only ship the geometry defaults.
