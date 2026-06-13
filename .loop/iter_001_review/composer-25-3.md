# Iter 001 Adversarial Review — Composer 2.5

## Summary

Iter-001 closes several charter gaps in unit tests and the synthetic bench (in-order DQ, honesty fields, curvature ILC, AIGP geometry constants), but the **competition execution path is still half-wired**: vision UDP exists but `mavlink_bridge` never feeds it, and `RacePipeline` / `sim_pybullet/runner` do not terminate on DQ or crash. The sequencer’s in-order rule only watches **future gate openings**, not frame-strut hits on non-current gates — a forward skip can clip future struts without DQ or crash. The bench remains a weak oracle: it checks field presence more than end-to-end gate-frame failure modes.

## Findings (ordered by severity)

### F1. Vision UDP receiver not connected to competition I/O — [BLOCKER]
- **File(s)**: `competition/mavlink_bridge.py:163-167`, `competition/vision_udp.py:144-235`
- **Issue**: A11 shipped `VisionUdpReceiver` + tests, but `MavlinkBridge.get_camera_frame()` still returns `None` with a “spec not yet released” comment. The AIGP JPEG stream on port 5600 is never decoded into `CameraFrame` for the pipeline.
- **Repro**: Run `RacePipeline` against `MavlinkBridge`; perception stays off (`detection_active=False` every tick) regardless of simulator vision traffic.
- **Fix sketch**: Add an asyncio datagram listener on `AIGP_CAM_UDP_PORT`, call `VisionUdpReceiver.feed_packet`, `decode_jpeg_to_camera_frame`, return from `get_camera_frame()`. Wire in iter-002 with A12 command-path tests.
- **Confidence**: high

### F2. `RacePipeline` never stops on DQ or crash — [BLOCKER]
- **File(s)**: `race_pipeline.py:271-273`, `race_pipeline.py:324-336`, `competition/session.py:173-176`
- **Issue**: `session.should_stop` only checks `sequencer.is_complete`. After `is_disqualified` or `last_crash`, the control loop keeps commanding attitude until wall-clock timeout (480 s in `session.py`) or external stop.
- **Repro**: Inject an out-of-order crossing in a MAVLink session harness; observe `RaceSession._race_loop` continues after `seq.is_disqualified` is True.
- **Fix sketch**: `should_stop = lambda: seq.is_complete or seq.is_disqualified or seq.last_crash is not None`; optionally set `RaceState.TIMED_OUT` / log terminal reason in `_control_callback` and return hover/zero thrust.
- **Confidence**: high

### F3. Frame-strut hits on non-current gates are invisible — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:246-306`, `gate_sequencing/sequencer.py:337-355`
- **Issue**: P1-6 crash classification runs only for `self._gates[self._current_idx]`. The out-of-order scan only DQ’s when a **future** gate’s **opening** is crossed. A drone can plane-cross a future gate’s strut annulus (inside outer frame, outside opening) with no `crash`, no `DQ`, and no `miss` on that gate.
- **Repro**: Current target g1; fly segment through g3’s plane at lateral offset in [0.75 m, 1.35 m] for AIGP geometry (strut zone) without entering g3’s opening; `last_crash` stays None and `is_disqualified` stays False.
- **Fix sketch**: In the future-gate loop (or a shared helper), mirror P1-6 for every unpassed gate: opening crossing → DQ; strut annulus → `mark_collision` / crash list; far plane → miss.
- **Confidence**: high

### F4. PyBullet `runner.py` has no DQ handling — [MAJOR]
- **File(s)**: `sim_pybullet/runner.py` (no `is_disqualified` references), `gate_sequencing/sequencer.py:161-168`
- **Issue**: The primary local sim entry path never reads `is_disqualified` / `dq_reason`. Iter-001 honesty work landed in `scripts/benchmark.py` and unit tests, not the runner the team still uses for visual demos and replan integration.
- **Repro**: Reproduce U-turn skip in `sim_pybullet/runner`; run may still report progress as if rules were enforced only in bench JSON.
- **Fix sketch**: After `sequencer.update`, abort run on `is_disqualified` (mirror benchmark PyBullet path at `scripts/benchmark.py:790-792`). Align with deferred A16 `run_pipeline_pybullet.py` scaffold.
- **Confidence**: high

### F5. Tracker residual training artifact missing (A15 incomplete) — [MAJOR]
- **File(s)**: `control/learned_residual.py`, `control/mpc_tracker.py:87-124`, `.loop/synthesis/iter_001.md:89`
- **Issue**: Synthesis promised `scripts/train_tracker_residual.py` and `control/residual_weights.npz`. Neither exists in the tree; only the MLP module + unit tests shipped. Holdout “≥ 3% avg-err drop” gate from the plan is unverifiable.
- **Repro**: `Glob **/train_tracker_residual*` and `residual_weights.npz` → empty.
- **Fix sketch**: Add offline trainer reading benchmark `controller_trace`, ship baseline `.npz`, gate enablement on holdout regression (A17 matrix helps).
- **Confidence**: high

### F6. Synthetic bench overloads `crashed` for disqualification — [MAJOR]
- **File(s)**: `scripts/benchmark.py:450-453`, `scripts/benchmark.py:583-622`
- **Issue**: On DQ, the loop sets `crashed = True` and `termination_reason = f"disqualified:..."`. Downstream consumers that only read `crashed` cannot distinguish rule violations from physical impacts; `no_crash` threshold semantics blur.
- **Repro**: Force DQ in synthetic bench; `result["crashed"]` is True while `result["disqualified"]` is also True — duplicate terminal signaling.
- **Fix sketch**: Keep `crashed` for physical failures only; break on DQ without setting `crashed`; add explicit `terminal_failure` enum in iter-002 JSON schema.
- **Confidence**: high

### F7. 8-minute VQ1 run cap not enforced on bench / sequencer — [MAJOR]
- **File(s)**: `competition/aigp_geometry.py:65`, `scripts/benchmark.py:47-55`, `scripts/benchmark.py:928-929`, `gate_sequencing/sequencer.py:71-72`
- **Issue**: `AIGP_VQ1_MAX_RUN_DURATION_S = 480` is defined but unused in `benchmark.py` (default `--duration 30`, `THRESHOLDS["max_total_time_s"] = 30`). `RaceState.TIMED_OUT` exists but nothing in `GateSequencer.update()` transitions to it.
- **Repro**: Run synthetic bench with `duration=600`; no spec-aligned timeout unless manually passed.
- **Fix sketch**: Default bench/sim duration to `AIGP_VQ1_MAX_RUN_DURATION_S`; set `seq._state = TIMED_OUT` when sim time exceeds cap; expose in honesty JSON.
- **Confidence**: high

### F8. ILC partition can drop sections when point times quantize to duplicate steps — [MAJOR]
- **File(s)**: `planning/ilc_sections.py:125-136`, `tests/test_ilc_sections.py:163-175`
- **Issue**: `_pt_to_step` uses `int(time/dt)`. Non-uniform trajectory sampling can yield `e_step <= s_step`, and those runs are **skipped** (`continue` at line 134-135). Tests only use uniform `time = i * dt_pt`; sparse/refined planners may leave holes before edge extension.
- **Repro**: Build stub trajectory with two points at `t=0.009` and `t=0.011`, `dt=0.01`, high/low class boundary between them; section run may vanish → fallback or shortened coverage not covered by `test_sections_cover_full_step_range_without_gap`.
- **Fix sketch**: Map points to steps with `np.searchsorted` / enforce monotonic `end_step = max(s_step+1, _pt_to_step(e_pt))`; add adversarial test with irregular timestamps.
- **Confidence**: medium (depends on `RaceTrajectory` point density in production)

### F9. Vision reassembly does not validate assembled JPEG size — [MINOR]
- **File(s)**: `competition/vision_udp.py:84-97`, `competition/vision_udp.py:220-225`
- **Issue**: `parse_packet` rejects `payload_size` mismatch vs datagram, but on completion `assemble()` never checks `sum(len(chunk)) == jpeg_size` or `offset == jpeg_size`. Wrong/missing chunks can still produce a buffer passed to `cv2.imdecode` (returns `None` silently in decode helper).
- **Repro**: Feed chunks whose total payload is 50 bytes with `jpeg_size=400`; completion still returns `ReassembledFrame` with undersized bytes.
- **Fix sketch**: After assembly, `if len(jpeg_bytes) != jpeg_size: drop + counter`; optional per-chunk `payload_size==0` policy.
- **Confidence**: high

### F10. `GateGeometry` PnP defaults still legacy 1.2 m — [MINOR]
- **File(s)**: `estimation/gate_pnp.py:106-107`, `race_pipeline.py:128-130`
- **Issue**: `GateSpec` defaults to AIGP 1.5 m, but `GateGeometry` dataclass defaults remain `1.2` m. `RacePipeline` overrides via `PipelineConfig` (1.5), yet any direct `GateGeometry()` or stale import path still builds wrong object points for PnP.
- **Repro**: `GateGeometry()` with no args → 1.2 m square in `object_points`.
- **Fix sketch**: Default `GateGeometry` from `AIGP_GATE_INTERIOR_M`; single test asserting default corners span 1.5 m.
- **Confidence**: high

### F11. `enforce_in_order=True` may DQ legitimate replanner recovery — [MINOR]
- **File(s)**: `gate_sequencing/sequencer.py:89-95`, `planning/dynamic_replanner.py`, `race_pipeline.py:338-343`
- **Issue**: Any future-gate **opening** crossing is terminal. A recovery path that legitimately re-enters a later gate’s opening (e.g. wide reroute on a self-crossing track) is indistinguishable from a skip cheat. No `SequencerConfig` hook on `RacePipeline` (always default `GateSequencer(gates)` at line 192).
- **Repro**: Enable replanner on figure-8 / helix; trajectory cuts through gate N+2 opening before gate N+1 is credited → DQ despite honest recovery intent.
- **Fix sketch**: Document as strict self-test policy; for competition, consider DQ only for gates with `sequence_index` less than current, or add `recovery_grace` window tied to replan epoch.
- **Confidence**: medium

### F12. Calibration `lstsq` lacks rank/degeneracy guards — [MINOR]
- **File(s)**: `competition/calibration.py:90-108`, `tests/test_calibration.py:71-77`
- **Issue**: Tests cover `n < 2` only. All-zero thrust, collinear `(u, v)`, or near-singular `X` can yield NaN/negative `k_t`, `k_d` without validation before writing JSON consumed by controller config.
- **Repro**: `identify_thrust_drag_ratios([CalibrationSample(0,0,-9.81)] * 5)` — may return finite but meaningless coefficients.
- **Fix sketch**: Check `rank(X) == 2`, reject `nan` coeffs, require thrust span `std(u) > ε`; SITL iter-002 adds real telemetry soak test.
- **Confidence**: medium

### F13. Residual clamp composition is correct; `SimplePositionTracker` has no residual path — [NIT]
- **File(s)**: `control/mpc_tracker.py:215-245`, `control/mpc_tracker.py:318-319`
- **Issue**: Geometric tracker applies residual clamp then `max_tilt_rad` — no 0.85+0.05 leak. `SimplePositionTracker.track` has no residual branch (only used when `use_geometric_tracker=False`). Not a bug today but a footgun if someone toggles tracker mode with `use_residual=True`.
- **Repro**: N/A unless config mismatch.
- **Fix sketch**: Document “residual only on `GeometricTracker`”; or share clamp helper.
- **Confidence**: high

### F14. `test_crash_does_not_advance` intentionally uses legacy geometry (not a bug) — [NIT]
- **File(s)**: `gate_sequencing/tests/test_sequencer.py:522-545`
- **Issue**: Brief asked whether AIGP-default border (0.6 m) breaks this test. Test explicitly uses `interior_width=1.2`, `border_width=0.15` and documents that `y=0.7` targets the [0.6, 0.75] strut annulus — still valid for race_01-scale geometry.
- **Repro**: N/A — test is self-consistent.
- **Fix sketch**: Add parallel test case with `border_width=0.18` (race_01.json) and AIGP 0.6 m case in iter-002 matrix.
- **Confidence**: high

## Things iter-001 got right
- **In-order DQ core case** is implemented and covered (`gate_sequencing/sequencer.py:337-355`, `gate_sequencing/tests/test_sequencer_adversarial.py:57-111`).
- **Honesty surface** in synthetic/PyBullet bench JSON (`disqualified`, `dq_reason`, `last_crash_gate`) with contract tests (`tests/test_benchmark_adversarial.py:60-110`).
- **Wall-clock ILC magic removed** from default bench path; race_01 retains explicit `ilc_section_overrides` in JSON (`sim_pybullet/configs/race_01.json:21-26`).
- **AIGP geometry centralised** in `competition/aigp_geometry.py` with camera tilt tests (`tests/test_camera_geometry.py:66-91`).
- **Vision wire format** well tested in isolation (`tests/test_vision_udp.py`, 15 tests per commit message).

## What I did NOT review
- Full `git show` line-by-line for all six commits (relied on commit messages + targeted file reads).
- `scripts/benchmark.py` PyBullet loop beyond DQ/crash/honesty lines (~746-900).
- `planning/trajectory_optimizer.py`, EKF tuning, ILC `compute_ilc_offset_table` internals.
- `gate_detection/` ONNX / Phase1 detector integration.
- Deferred A12 (`test_mavlink_bridge_commands.py`), A16 (`run_pipeline_pybullet.py`), A17 (`benchmark_matrix.py`) — confirmed absent or not searched deeply.
- `sim_pybullet/` physics and contact manifold correctness.
- Whether `config/ilc_defaults.json` global hyperparams (e.g. `convergence_threshold: 0.002`, `momentum_gamma: 0.0`) regress non-race_01 tracks without overrides.
- End-to-end run of `python3 scripts/benchmark.py` (read-only review only).
