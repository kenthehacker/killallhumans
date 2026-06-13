# Iter 001 Adversarial Review — GPT-5.5 XHigh 1

## Summary
Iter-001 fixed several real audit items, but the competition path is still not honest: the new UDP vision receiver is not wired into the MAVLink adapter, and `RacePipeline` does not stop on the terminal sequencer states that the benchmark now treats as failures. I also found a few "green test but wrong surface" gaps where defaults or geometry fixes landed in one path but not the actual default/runtime path.

## Findings (ordered by severity)

### F1. Vision UDP receiver is not connected to the competition adapter — [BLOCKER]
- **File(s)**: `competition/mavlink_bridge.py:163`, `competition/vision_udp.py:181`, `race_pipeline.py:320`
- **Issue**: A10/A11 added `VisionUdpReceiver`, but `MAVLinkBridge.get_camera_frame()` still always returns `None`, and `rg` only finds `VisionUdpReceiver` in `competition/vision_udp.py` and its tests. In a real DCL/MAVLink run, `RacePipeline._control_callback()` therefore sets `detection_active = False` every tick and never runs gate detection/PnP. This violates the VQ1 requirement to process the separate UDP JPEG stream on port 5600.
- **Repro**: Instantiate `MAVLinkBridge` and call `get_camera_frame()` after connect: the implementation is still the old placeholder returning `None`. Static repro: `rg "VisionUdpReceiver|vision_udp|decode_jpeg_to_camera_frame"` shows no production caller outside the receiver module.
- **Fix sketch**: In `MAVLinkBridge.connect()`, start an asyncio UDP datagram endpoint on `AIGP_CAM_UDP_PORT`, feed datagrams into `VisionUdpReceiver`, decode completed frames with `decode_jpeg_to_camera_frame()`, and have `get_camera_frame()` return the latest decoded `CameraFrame`. Add a test with a fake datagram transport that sends chunked JPEG bytes and asserts `get_camera_frame()` becomes non-`None`.
- **Confidence**: high — the bridge has an explicit `return None` placeholder and there is no production reference to the new receiver.

### F2. Real pipeline ignores terminal DQ/crash states after sequencing — [BLOCKER]
- **File(s)**: `race_pipeline.py:271`, `race_pipeline.py:325`, `race_pipeline.py:334`, `race_pipeline.py:338`
- **Issue**: The synthetic and PyBullet benchmarks now break on `seq.last_crash` and `seq.is_disqualified`, but the production `RacePipeline` only stops when `sequencer.is_complete`. After `sequencer.update()` it checks completion, then proceeds into replanning/state prediction/control even if the sequencer has just set `RaceState.DISQUALIFIED` or recorded a frame-strut crash. This makes the competition path less honest than the benchmark.
- **Repro**: Configure a pipeline with three gates, feed positions that trigger the sequencer's out-of-order DQ, then call `_control_callback()` again. `session.should_stop` remains false because it only reads `is_complete`, and the callback has no DQ/crash branch before `_maybe_replan()`.
- **Fix sketch**: Extend `session.should_stop` to include `sequencer.is_disqualified` and `sequencer.last_crash is not None`. In `_control_callback()`, immediately return a neutral/hover or `None` command after logging the terminal reason, before replanning. Surface the terminal reason in session metrics so a DQ cannot look like a timeout.
- **Confidence**: high — the terminal checks exist in `scripts/benchmark.py` but not in `race_pipeline.py`.

### F3. PyBullet benchmark still drops per-track gate border width — [MAJOR]
- **File(s)**: `scripts/benchmark.py:687`, `scripts/benchmark.py:693`, `scripts/benchmark.py:814`, `sim_pybullet/env.py:187`
- **Issue**: A9 fixed synthetic `GateSpec` construction to propagate `border_width_m`, but the PyBullet `_to_specs()` path still passes only interior width/height. For `race_01.json`, the env loads `border_width_m=0.18`, while the sequencer sees `GateSpec`'s AIGP default `0.6`. That makes the sequencer's secondary geometric crash zone much larger than the actual PyBullet gate body, so `seq.last_crash` can terminate a run even when `env.gate_contact()` did not report a contact.
- **Repro**: Load `race_01.json`: `DroneRaceEnv.load_config()` preserves `g.config.border_width_m == 0.18`, but `_to_specs()` constructs `GateSpec(... interior_width=1.2, interior_height=1.2)` with default `border_width=0.6`. A crossing at local lateral offset about `1.0m` is outside the physical race_01 frame (`outer half = 0.78m`) but inside the sequencer's mistaken outer frame (`outer half = 1.2m`), so the fallback at `scripts/benchmark.py:814` can fire a false crash.
- **Fix sketch**: Pass `border_width=g.config.border_width_m` and `depth=g.config.depth_m` in `_to_specs()`. Add a PyBullet benchmark unit seam or helper test that asserts every `RaceConfig` gate's geometry is preserved exactly when converted to `GateSpec`.
- **Confidence**: high — direct line inspection shows the synthetic path has the fix and the PyBullet path does not.

### F4. PnP estimator defaults are still legacy 640x480 / 1.2m — [MAJOR]
- **File(s)**: `estimation/gate_pnp.py:103`, `estimation/gate_pnp.py:139`, `estimation/gate_pnp.py:144`, `tests/test_camera_geometry.py:45`
- **Issue**: `CameraIntrinsics()` now defaults to AIGP, but `GatePnPEstimator()` does not use it. Its default constructor still calls `CameraIntrinsics.from_fov(90.0, 640, 480)` and `GateGeometry()` still defaults to `1.2m` gates. The new camera tests only cover `CameraIntrinsics()` and `PipelineConfig`, so the public estimator default remains spec-drifted.
- **Repro**: `GatePnPEstimator().camera.image_height` is `480`, `camera.cy` is `240`, and `GatePnPEstimator().gate.interior_width_m` is `1.2`, despite VADR-TS-002 requiring `360`, `180`, and `1.5`.
- **Fix sketch**: Change `GateGeometry` defaults to import `AIGP_GATE_INTERIOR_M`, and change `GatePnPEstimator.__init__` to use `CameraIntrinsics()` when `camera is None`. Update legacy tests that need 640x480/1.2m to pass those values explicitly.
- **Confidence**: high — I ran the constructor repro and it returned the legacy values.

### F5. Vision reassembly does not validate aggregate JPEG size — [MAJOR]
- **File(s)**: `competition/vision_udp.py:84`, `competition/vision_udp.py:108`, `competition/vision_udp.py:220`, `tests/test_vision_udp.py:64`
- **Issue**: `parse_packet()` validates each packet's `payload_size`, but the completed frame never checks that `sum(len(chunk)) == jpeg_size`. `_ReassemblyBuf.assemble()` starts with `bytearray(self.jpeg_size)`, then slice-assigns each chunk; if chunks sum longer than `jpeg_size`, Python extends the bytearray, and if they sum shorter, the output is padded with zeros. The brief explicitly called out `jpeg_size` mismatch, but tests only cover per-packet payload mismatch.
- **Repro**: Feed two chunks with `jpeg_size=3` and payloads `b"abc"` and `b"def"`; `feed_packet()` emits a `ReassembledFrame` with six bytes `b"abcdef"` instead of rejecting the malformed frame.
- **Fix sketch**: Track cumulative payload bytes per frame and reject/reset the buffer unless it equals `jpeg_size` exactly at completion. Add tests for `sum(payloads) < jpeg_size`, `sum(payloads) > jpeg_size`, and zero-length chunk behavior. Consider a max JPEG size cap to avoid allocating attacker-controlled `jpeg_size` buffers.
- **Confidence**: high — the repro above returns a malformed completed frame.

### F6. Same-tick current-plus-next gate crossing leaks strict ordering — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:308`, `gate_sequencing/sequencer.py:343`, `gate_sequencing/tests/test_sequencer_adversarial.py:57`
- **Issue**: The new out-of-order scan runs after the current-gate pass branch. If one update segment crosses the current gate and the next gate in the same tick, the current gate is credited, `_current_idx` advances, and the future scan starts at `self._current_idx + 1`, skipping the newly-current gate that was also crossed by the same segment. That recreates a smaller version of the U-turn false-complete leak for long telemetry gaps or low-rate updates.
- **Repro**: With gates at x=5, x=10, x=15, call `seq.update((4.5, 0, -2))` then `seq.update((10.5, 0, -2))`. The segment crossed g1 and g2 openings, but the sequencer reports `gates_passed == 1`, current gate `g2`, and no DQ.
- **Fix sketch**: For each update segment, compute all gate-plane crossings with their interpolation parameter `t`, sort by `t`, and process them in physical order. Either credit multiple consecutive gates in order within the same segment, or DQ when a non-current opening is crossed before it can be credited. Add an adversarial test with a long segment crossing two adjacent gate openings.
- **Confidence**: medium — normal 100 Hz / 15 m/s steps are probably too small for race_01 spacing, but telemetry gaps and unknown VQ1 gate spacing make this worth hardening.

### F7. Calibration accepts unidentifiable samples as a perfect zero model — [MAJOR]
- **File(s)**: `competition/calibration.py:90`, `competition/calibration.py:105`, `tests/test_calibration.py:71`
- **Issue**: `DroneCalibrator.identify_thrust_drag_ratios()` ignores the least-squares rank and singular values. Two all-zero/no-excitation samples pass the `n >= 2` guard and return `thrust_per_mass=0.0`, `drag_per_mass=0.0`, `rmse=0.0`. That is not a NaN, but it is worse operationally: a failed calibration can look perfect and write a zero-thrust model if `assumed_mass_kg` is provided.
- **Repro**: `DroneCalibrator().identify_thrust_drag_ratios([CalibrationSample(0,0,-9.81), CalibrationSample(0,0,-9.81)])` returns zero ratios with zero RMSE.
- **Fix sketch**: Require excitation before fitting: at least two distinct thrust levels, enough velocity variation for drag if fitting drag, full-rank design matrix, finite positive `thrust_per_mass`, and plausible bounds. Raise `ValueError` on rank-deficient/no-excitation data and add tests for all-zero thrust, constant thrust/velocity, negative fitted thrust, and NaN inputs.
- **Confidence**: high — the current code discards `_rank` and `_sv`, and the all-zero repro passes.

## Things iter-001 got right
- The sequencer now has a real terminal DQ state and adversarial tests for future-gate opening crossings, which directly addresses the worst iter-0 false-complete pattern.
- Moving AIGP geometry constants into `competition/aigp_geometry.py` is the right direction and makes later spec changes easier to audit.
- The tracker residual is off by default, lazy-loaded, and physically re-clamped after residual application; that is the right safety shape for lightweight ML.
- The ILC refactor removed the hard-coded wall-clock sections from the default synthetic path and put race_01's old tuning into explicit config.
- The UDP reassembler is well factored for unit testing; the missing pieces are production wiring and stricter malformed-frame validation, not a total rewrite.

## What I did NOT review
I did not run the full PyBullet benchmark or visual demo. I did not inspect every historical test file, the full trajectory optimizer implementation, the dynamic replanner internals beyond their interaction points, or MAVSDK command serialization behavior for the deferred A12 command tests. I did not review generated artifacts outside the iter-001 commits except where needed to cross-check current runtime wiring.
