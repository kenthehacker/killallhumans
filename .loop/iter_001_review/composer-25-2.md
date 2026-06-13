# Iter 001 Adversarial Review — Composer 2.5

## Summary

Iter-001 closes the pre-audit issues (I-1/I-2/I-3/I-5) with strong adversarial tests and honest bench wiring, but several synthesis promises are only half-landed: camera tilt exists as a field but is not consumed in PnP→world projection, the vision UDP stack is unit-tested in isolation yet `MAVLinkBridge.get_camera_frame()` still returns `None`, and the ML residual has safety tests but no trainer or shipped weights. The green 310-test matrix is real for what was committed, but it overstates competition-readiness.

## Findings (ordered by severity)

### F1. Camera +20° tilt not applied in PnP world projection — [BLOCKER]
- **File(s)**: `estimation/gate_pnp.py:214-258`, `race_pipeline.py:123-131`, `tests/test_camera_geometry.py:75-91`
- **Issue**: A6 added `CameraIntrinsics.pitch_offset_rad` and a analytic horizon test, but `GatePnPEstimator.gate_pose_to_drone_position()` never applies a body→camera rotation. OpenCV PnP solves in the camera optical frame; without `R_pitch(pitch_offset_rad)` the recovered drone position ignores the AIGP upward-tilted mount. `RacePipeline` still builds intrinsics via `from_fov()` and never threads `PipelineConfig.camera_pitch_offset_rad` into the transform chain. The synthesis explicitly claimed `body_R_camera` in `gate_pose_to_drone_position` — that wiring is absent.
- **Repro**: Run PnP drift correction on a synthetic gate image generated with a 20°-up camera; compare recovered NED position against ground truth with and without extrinsic pitch — error will be systematic, not noise.
- **Fix sketch**: Add `body_R_camera(pitch_offset_rad)` (or equivalent) in `gate_pose_to_drone_position`; unit-test that a known camera-frame translation maps to the expected NED offset under +20° pitch. Wire `RacePipeline` to construct `CameraIntrinsics` with explicit AIGP fx/fy/cx/cy/pitch, not only `from_fov`.
- **Confidence**: high — code path read end-to-end; only the field and a paper formula test exist.

### F2. Vision UDP receiver not integrated into competition path — [MAJOR]
- **File(s)**: `competition/mavlink_bridge.py:163-167`, `competition/vision_udp.py:144-235`, `.loop/state/iter_state.json:58`
- **Issue**: A11 shipped `VisionUdpReceiver` + 15 unit tests, but `MAVLinkBridge.get_camera_frame()` is still a stub returning `None` with a comment that the spec is unreleased. Iter state even notes "mavlink_bridge wiring still pending." The race pipeline therefore cannot consume port-5600 JPEG chunks in a DCL run despite the receiver being production-shaped.
- **Repro**: Instantiate `MAVLinkBridge`, call `get_camera_frame()` — always `None`. Feed valid chunked packets into `VisionUdpReceiver` — frames assemble correctly. The two paths never meet.
- **Fix sketch**: Add an asyncio datagram listener on `AIGP_CAM_UDP_PORT` that feeds `VisionUdpReceiver`, decode via `decode_jpeg_to_camera_frame`, cache latest frame for `get_camera_frame()`. Smoke test: encode 2-chunk JPEG → UDP → bridge returns `CameraFrame` with correct shape.
- **Confidence**: high — grep shows zero imports of `vision_udp` outside tests.

### F3. Tracker residual ML path incomplete (no trainer, no weights) — [MAJOR]
- **File(s)**: `control/learned_residual.py`, `control/mpc_tracker.py:82-124`, `.loop/synthesis/iter_001.md:89`
- **Issue**: A14/A15 landed the MLP, clamps, and off-by-default switch, but `scripts/train_tracker_residual.py` and `control/residual_weights.npz` from the synthesis action list do not exist in the tree. `use_residual=True` without a path falls back to zero-init (safe but useless). The iter-001 ML deliverable is scaffolding only.
- **Repro**: `glob **/train_tracker*` and `**/residual_weights*` → empty. Enable `use_residual=True` with a real path — no artifact to load.
- **Fix sketch**: Ship the trainer reading `controller_trace` from benchmark output; add holdout gate from synthesis (≥3% err drop); commit a small `.npz` or document that ML is explicitly deferred to iter-002 with synthesis updated.
- **Confidence**: high — file search in worktree.

### F4. VQ1 8-minute run cap not enforced in benchmark or `RacePipeline` — [MAJOR]
- **File(s)**: `competition/aigp_geometry.py:65`, `scripts/benchmark.py:234-246,420`, `competition/session.py:36,168-171`
- **Issue**: `AIGP_VQ1_MAX_RUN_DURATION_S` is defined but only `competition/session.py` checks it. `run_synthetic_benchmark(duration=30.0)` and CLI default `--duration 30` never reference the spec cap. A bench PASS at 30s says nothing about an 8-minute competition attempt hanging or over-running.
- **Repro**: `grep AIGP_VQ1_MAX_RUN` in `scripts/benchmark.py` → no matches. Run full bench — no 480s guardrail.
- **Fix sketch**: Default synthetic/sim duration to `min(requested, AIGP_VQ1_MAX_RUN_DURATION_S)`; add adversarial test that a loop exceeding 480s terminates with `timed_out` / `sim_passed=False`.
- **Confidence**: high — constant exists, bench ignores it.

### F5. Default `enforce_in_order=True` can DQ replanner/recovery paths not covered by tests — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:337-355`, `race_pipeline.py:192`, `sim_pybullet/configs/race_01.json:69-71`, `gate_sequencing/tests/test_sequencer.py:698-712`
- **Issue**: DQ scans every future gate's opening on each tick. `RacePipeline` uses `GateSequencer(gates)` with default `enforce_in_order=True` and does not load per-track `sequencer` JSON (race_01 sets `proximity_pass_distance: 1.2` but pipeline ignores it). A dynamic replanner that cuts a corner through gate N+2's opening while recovering to gate N will hard-DQ even if the trajectory is physically valid. Legacy test `test_crossing_non_highlighted_gate_does_not_credit` still passes (`result is None`, `gates_passed==0`) but the mechanism is now `is_disqualified=True` (`out_of_order:G2`) — the test name/doc still describe the pre-fix silent-noop behaviour and do not assert DQ semantics.
- **Repro**: `python3 -c` with `GateSequencer` + two updates through G2 while current is G1 → `dq True, reason out_of_order:G2`. Same steps in `test_crossing_non_highlighted_gate_does_not_credit` — passes without checking `is_disqualified`.
- **Fix sketch**: Add adversarial test for "replanner arc through future opening during RECOVERY" and decide policy: (a) document that strict VQ1 order forbids this, or (b) suppress DQ in `RaceState.RECOVERY`, or (c) load `enforce_in_order` from track config defaulting False for bench until replanner is proven safe.
- **Confidence**: medium-high — behaviour verified in REPL; replanner interaction not simulated in tests.

### F6. Planned A12/A16/A17 not shipped — competition surface still split — [MAJOR]
- **File(s)**: `.loop/synthesis/iter_001.md:86-92,94-99`, `sim_pybullet/runner.py`, `race_pipeline.py`
- **Issue**: Deferred items from the synthesis are still gaps for iter-002: no `tests/test_mavlink_bridge_commands.py`, no `scripts/run_pipeline_pybullet.py`, no `scripts/benchmark_matrix.py`. I-7 (runner vs `RacePipeline` collapse) remains; PyBullet harness and competition pipeline diverge. Charter goal #3 (MAVLink UDP surface) is only partially addressed.
- **Repro**: `glob run_pipeline_pybullet.py benchmark_matrix.py test_mavlink_bridge_commands.py` → missing.
- **Fix sketch**: Prioritize A16 thin PyBullet entry + A12 MAVSDK command contract tests before more ILC tuning; A17 matrix before trusting a single-track green bench.
- **Confidence**: high — files absent from commit list and tree.

### F7. Bench maps disqualification into `crashed=True` — obscures metrics — [MAJOR]
- **File(s)**: `scripts/benchmark.py:446-453,585-622`
- **Issue**: On DQ, synthetic/PyBullet loops set `crashed = True` and `termination_reason = f"disqualified:..."`. Separate `disqualified` field exists, but any consumer that only reads `crashed` cannot distinguish frame strike vs rule violation. Regression dashboards may mis-attribute DQ as physics failure.
- **Repro**: Force out-of-order crossing in synthetic bench — `crashed=True` and `disqualified=True` simultaneously.
- **Fix sketch**: Keep `crashed` for physical impacts only; add `terminated=True` or let `crashed` stay False when only DQ. Update adversarial contract tests accordingly.
- **Confidence**: high — read loop termination block.

### F8. Vision reassembly does not validate `jpeg_size` vs assembled payload — [MAJOR]
- **File(s)**: `competition/vision_udp.py:84-97,220-225`, `tests/test_vision_udp.py`
- **Issue**: `parse_packet` rejects `payload_size != len(payload)` but, on completion, `assemble()` concatenates chunks into a `bytearray(jpeg_size)` without checking `sum(len(chunk)) == jpeg_size` or that assembled length matches. Malicious or buggy sender can declare `jpeg_size=1_000_000` with small chunks → large allocation; or truncate/oversize JPEG vs header. `payload_size==0` is allowed (empty chunk) — not tested.
- **Repro**: Feed `total_chunks=2`, `jpeg_size=10000`, payloads totaling 100 bytes — assembly succeeds with mostly zero padding; `cv2.imdecode` may fail silently downstream.
- **Fix sketch**: After assembly, `assert len(jpeg_bytes) == jpeg_size` (or sum of chunk sizes); reject frame and increment `dropped_partial_frames` on mismatch. Add adversarial tests for zero-length payload and size mismatch.
- **Confidence**: high — `assemble()` has no post-check.

### F9. Calibration accepts degenerate zero-thrust fits without error — [MINOR]
- **File(s)**: `competition/calibration.py:90-108`, `tests/test_calibration.py`
- **Issue**: All-zero thrust samples yield `thrust_per_mass=0`, `drag_per_mass=0` via `lstsq` without `ValueError` or high RMSE flag. Hover identification with no thrust variation is pathological for the model but not rejected.
- **Repro**: `identify_thrust_drag_ratios([CalibrationSample(0.0, v, -9.81) for v in range(10)])` → zeros, no exception.
- **Fix sketch**: Require thrust span ≥ ε or rank check on `X`; return error or mark result invalid in JSON.
- **Confidence**: high — exercised in REPL.

### F10. `GateGeometry` / `GatePnPEstimator` fallback still 1.2 m — [MINOR]
- **File(s)**: `estimation/gate_pnp.py:104-107,144-145`
- **Issue**: `GateSpec` defaults moved to AIGP 1.5 m, but `GateGeometry` and `GatePnPEstimator` fallback `GateGeometry()` still use `interior_width_m=1.2`. Callers constructing `GatePnPEstimator()` without explicit geometry get wrong object points for PnP scale.
- **Repro**: `GatePnPEstimator().gate.interior_width_m` → 1.2.
- **Fix sketch**: Default `GateGeometry` from `AIGP_GATE_INTERIOR_M`; align `RacePipeline.gate_geometry` (already 1.5) with estimator defaults.
- **Confidence**: high.

### F11. `test_synthetic_bench_exposes_honesty_fields` contract is tautological — [MINOR]
- **File(s)**: `tests/test_benchmark_adversarial.py:101-103`
- **Issue**: Assertion `(result["sim_passed"] is False) == terminal or result["sim_passed"] is True` is always true (boolean OR with True). It does not prove `sim_passed=False` whenever `crashed or disqualified`.
- **Repro**: Set `sim_passed=True` with `crashed=True` in a mock dict — assertion still passes.
- **Fix sketch**: Replace with `assert result["sim_passed"] is False` when `result["crashed"] or result["disqualified"]`.
- **Confidence**: high.

### F12. `race_01.json` ILC overrides are still wall-clock helix magic — [MINOR]
- **File(s)**: `sim_pybullet/configs/race_01.json:21-26`, `scripts/benchmark.py:334-340`
- **Issue**: Default path is curvature-derived (good), but race_01 bypasses via `ilc_section_overrides` with steps `[0,200],[200,440],...` — the old helix-tuned windows in disguise. Charter allows explicit track overrides, but A17 multi-track regression was deferred; a new track without overrides gets different ILC personality than race_01, so bench "PASS on race_01" remains course-coupled.
- **Repro**: Load race_01 → overrides used; load `aigp_default.json` → derived sections only.
- **Fix sketch**: iter-002 `benchmark_matrix` must include ≥3 shapes; consider deriving race_01 overrides from curvature peaks and deleting literal step boundaries.
- **Confidence**: high — JSON content inspection.

### F13. Residual clamp order is safe; document fragility for future refactors — [NIT]
- **File(s)**: `control/mpc_tracker.py:214-245`, `tests/test_tracker_residual.py`
- **Issue**: Clamp composition is correct: tilt clamp → residual ±0.05 → tilt clamp again, so 0.85+0.05 cannot leak. No test pins order if someone moves residual before first tilt clamp.
- **Repro**: N/A unless refactor reorders lines.
- **Fix sketch**: Add a one-line comment/test: "residual must run only after first `max_tilt_rad` clamp."
- **Confidence**: high for current code; low for future regression.

## Things iter-001 got right

- **I-1 fix is real**: Out-of-order opening crossings set `RaceState.DISQUALIFIED` with `dq_reason=out_of_order:<id>`; adversarial tests in `gate_sequencing/tests/test_sequencer_adversarial.py` encode U-turn and far-plane cases.
- **I-2 honesty wiring**: Synthetic/PyBullet benches break on `seq.last_crash` and `seq.is_disqualified`; result dict exposes `disqualified`, `dq_reason`, `last_crash_gate`.
- **I-3 default path improved**: `planning/ilc_sections.py` removes `int(2.0/dt)` literals from `benchmark.py`; hyperparameters live in `config/ilc_defaults.json`; `border_width` propagated from track config fixes silent crash-zone drift.
- **Safety story for ML**: Tracker residual hard-clamps at consumer; `use_residual=False` is byte-identical baseline (tested).
- **Vision reassembler quality (in isolation)**: 24-byte LE header, OoO chunks, sim-time GC, duplicate handling — solid unit-test coverage in `tests/test_vision_udp.py`.

## What I did NOT review

- Full `git show` diff for all six commits line-by-line (relied on file reads + commit stats).
- PyBullet sim run end-to-end (`run_sim_benchmark` dynamics, contact manifold vs sequencer agreement).
- `scripts/visual_demo.py`, `sim_pybullet/runner.py` internals, gate_detection training stack.
- `competition/mavlink_bridge.py` send_attitude / offboard / heartbeat implementation beyond `get_camera_frame`.
- Deferred trainer scripts, ONNX paths, EKF noise tuning, replanner `DynamicReplanner` geometry.
- Whether `aigp_default.json` placeholder gates are flyable in sim (config only read, not executed).
