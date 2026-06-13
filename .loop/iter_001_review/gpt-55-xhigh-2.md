# Iter 001 Adversarial Review — GPT-5.5 extra-high 2

## Summary
Iter-001 fixed several honesty surfaces in the synthetic path, but the competition-facing path is still not race-ready: the VADR vision stream is implemented as a standalone parser but never connected to `MAVLinkBridge`, and the PnP/camera tilt change is mostly declarative rather than applied. I also found a PyBullet geometry regression where `run_sim_benchmark` silently uses AIGP-thick frames for legacy tracks, plus test gaps that let key crash/vision regressions slip through.

## Findings (ordered by severity)

### F1. VADR vision UDP is not wired into the competition bridge — [BLOCKER]
- **File(s)**: `competition/mavlink_bridge.py:163`, `competition/vision_udp.py:144`
- **Issue**: A11 shipped `VisionUdpReceiver`, but `MAVLinkBridge.get_camera_frame()` still always returns `None` and there is no UDP datagram task started in `connect()` or cancelled in `disconnect()`. On the actual DCL surface, the pipeline will fly with `detection_active=False` forever, so gate detection/PnP never run even though the spec's camera stream is available on port 5600.
- **Repro**: Inspect `MAVLinkBridge.get_camera_frame()`; it still contains the old "spec not yet released" stub. Run `RaceSession`: every loop awaits `interface.get_camera_frame()`, receives `None`, and `RacePipeline._process_detection()` is never called.
- **Fix sketch**: In `MAVLinkBridge.__init__`, own a `VisionUdpReceiver` plus latest decoded `CameraFrame`. In `connect()`, start an asyncio datagram endpoint bound to `AIGP_CAM_UDP_PORT`, feed packets into the receiver, decode complete frames with `decode_jpeg_to_camera_frame`, and return the freshest frame from `get_camera_frame()`. Add a bridge-level async test with a local UDP sender; the current `tests/test_vision_udp.py` only tests the reassembler in isolation.
- **Confidence**: high — the method is an explicit stub and no call site references `VisionUdpReceiver` outside its unit tests.

### F2. AIGP camera tilt and geometry are not actually applied in PnP — [BLOCKER]
- **File(s)**: `estimation/gate_pnp.py:103`, `estimation/gate_pnp.py:144`, `estimation/gate_pnp.py:214`, `tests/test_camera_geometry.py:75`
- **Issue**: The synthesis promised AIGP defaults and `body_R_camera` consumed by `gate_pose_to_drone_position`, but `GatePnPEstimator()` still defaults to `CameraIntrinsics.from_fov(90, 640, 480)` and `GateGeometry()` still defaults to 1.2 m gates. Worse, `gate_pose_to_drone_position()` ignores both `drone_orientation` and `self.camera.pitch_offset_rad`; the 20 degree upward tilt is only a field/comment, not part of the transform.
- **Repro**: `GatePnPEstimator().camera.image_height` is 480 and `.gate.interior_width_m` is 1.2. Calling `gate_pose_to_drone_position()` with the same pose but `drone_orientation=(0,0,0)` vs `(0,0.5236,0)` returns identical positions, proving the orientation/tilt path is dead. The current camera test computes `cy + fy*tan(pitch)` directly, but never exercises PnP conversion.
- **Fix sketch**: Make `GatePnPEstimator` default to `CameraIntrinsics()` and `GateGeometry(AIGP_GATE_INTERIOR_M, AIGP_GATE_INTERIOR_M)`. Add a `CameraIntrinsics.body_R_camera` (or equivalent) and include it, plus the drone/world attitude, when converting camera-frame pose to world/NED. Add a regression that projects a known tilted-camera gate observation through PnP and fails if pitch offset is ignored.
- **Confidence**: high — direct runtime probes and code inspection agree; the tests cover the presence/sign of the field, not the transform.

### F3. PyBullet benchmark uses AIGP-thick frames for legacy tracks — [MAJOR]
- **File(s)**: `scripts/benchmark.py:687`, `sim_pybullet/configs/race_01.json:7`, `sim_pybullet/_gate_to_spec.py:24`
- **Issue**: The synthetic path carefully propagates `border_width_m`, but `run_sim_benchmark()` converts PyBullet gates to `GateSpec` without `border_width` or `depth`. Because `GateSpec` now defaults to AIGP `border_width=0.6`, `race_01.json` gates with `border_width_m=0.18` are geometrically classified with a much larger frame annulus. `seq.last_crash` can now report frame strikes where PyBullet's actual model does not.
- **Repro**: Load `race_01.json`: gate defaults say 1.2 m opening and 0.18 m border. In `run_sim_benchmark._to_specs`, the constructed `GateSpec` only passes `interior_width` and `interior_height`, so the sequencer sees outer half-width `0.6 + 0.6 = 1.2 m` instead of the physical `0.6 + 0.18 = 0.78 m`.
- **Fix sketch**: Reuse `sim_pybullet._gate_to_spec.to_spec()` in `run_sim_benchmark`, or pass `border_width=float(g.config.border_width_m)` and `depth=float(g.config.depth_m)` explicitly. Add a regression asserting the benchmark's `GateSpec.outer_width` matches the loaded `RaceConfig` for `race_01`, `figure8`, and `aigp_default`.
- **Confidence**: high — the helper already exists and does exactly the missing field projection.

### F4. Real pipeline keeps flying after a sequencer DQ/crash — [MAJOR]
- **File(s)**: `race_pipeline.py:271`, `race_pipeline.py:324`, `gate_sequencing/sequencer.py:222`
- **Issue**: Benchmarks stop on `seq.last_crash` and `seq.is_disqualified`, but `RacePipeline.run()` sets `session.should_stop` only to `sequencer.is_complete`. If `_control_callback()` causes an out-of-order DQ, subsequent `sequencer.update()` calls early-return in `RaceState.DISQUALIFIED`, but the session continues sending attitude commands until the 8-minute timeout.
- **Repro**: Trigger an out-of-order pass in the real pipeline path. `GateSequencer` enters `DISQUALIFIED`; `RaceSession._race_loop()` checks only `should_stop`, which only checks completion, so the loop does not stop on the terminal failure state.
- **Fix sketch**: Make the session stop predicate include `sequencer.is_disqualified` and `sequencer.last_crash is not None`, and surface those terminal outcomes in race metrics/logs. Consider returning a safe hover/stop command once before shutdown, but do not keep racing after DQ.
- **Confidence**: medium-high — external collision marking may be handled elsewhere in a future adapter, but the sequencer's new terminal states are definitely not wired into `RacePipeline.run()`.

### F5. Vision reassembly accepts corrupt JPEG sizes — [MAJOR]
- **File(s)**: `competition/vision_udp.py:84`, `competition/vision_udp.py:220`, `tests/test_vision_udp.py:42`
- **Issue**: `parse_packet()` validates each packet's `payload_size`, but the completed frame never checks that `sum(len(chunk)) == jpeg_size`. `_ReassemblyBuf.assemble()` preallocates `jpeg_size` and writes whatever chunks arrived, so a frame whose chunks sum short is emitted with zero padding; inconsistent `jpeg_size` across chunks is also not rejected.
- **Repro**: Feed two chunks with `jpeg_size=10` and payloads `b"abc"` and `b"def"`. `VisionUdpReceiver.feed_packet()` emits a 10-byte frame `b"abcdef\\x00\\x00\\x00\\x00"` instead of dropping or raising. The brief explicitly called out `jpeg_size != sum(chunk payload sizes)` as an adversarial case, but no test covers it.
- **Fix sketch**: Store `jpeg_size` and `sim_time_ns` consistently per frame, reject packets with mismatched frame headers, track total received bytes, and on completion drop/raise if total bytes differ from `jpeg_size`. Add tests for short sum, long sum, zero-byte JPEG, and inconsistent `jpeg_size` on later chunks.
- **Confidence**: high — reproduced with the current receiver.

### F6. Synthetic crash tests do not prove gate-frame crashes terminate the bench — [MINOR]
- **File(s)**: `tests/test_benchmark_adversarial.py:60`, `tests/test_benchmark_adversarial.py:113`, `scripts/benchmark.py:446`
- **Issue**: A2's stated goal was "geometric crash terminates", but the benchmark-level tests only assert honesty fields exist and that a ground crash makes `sim_passed=False`. `test_synthetic_bench_exposes_honesty_fields()` even allows `sim_passed=True` when `terminal=True` because of the `or result["sim_passed"] is True` clause. A regression that removes the `seq.last_crash` termination branch can still pass these tests unless sequencer unit tests happen to catch it separately.
- **Repro**: Delete or bypass the `if seq.last_crash is not None:` block in `run_synthetic_benchmark`; the field-existence test still has no course that forces a gate-frame strike through the benchmark loop, and the terminal failure test uses `z=0` ground contact.
- **Fix sketch**: Add a deterministic injected course or monkeypatch around `GateSequencer.update()`/trajectory sampling that makes `seq.last_crash` non-`None` inside `run_synthetic_benchmark`, then assert `termination_reason.startswith("crash_gate:")`, `crashed is True`, and `sim_passed is False`. Tighten the boolean assertion so terminal failures cannot pass as `sim_passed=True`.
- **Confidence**: medium — constructing a physical course that crashes through the kinematic loop may be fragile, but a monkeypatch/injection test can lock the benchmark wiring directly.

## Things iter-001 got right
- The future-gate scan in `GateSequencer` is a real improvement and catches the original U-turn/skip false-complete pattern.
- `scripts/benchmark.py` now exposes `disqualified`, `dq_reason`, and `last_crash_gate`, which gives the synthesiser a better honesty surface.
- Moving ILC defaults into `config/ilc_defaults.json` and keeping `race_01` overrides explicit is the right direction for course generalization.
- The learned residual is off by default and re-clamped after addition, so the safety story is reasonable for this iteration.

## What I did NOT review
I did not run the full PyBullet benchmark or visual demo. I did not review the deferred MAVLink command tests (A12), the runner-to-pipeline collapse, or every legacy sim config beyond checking their gate default geometry. I also did not audit the full training path for `control/learned_residual.py`; I only reviewed inference safety and tests.
