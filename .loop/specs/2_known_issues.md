# Confirmed Issues — Initial Audit (Iter 0, 2026-05-24)

These were found by direct file inspection before any agent ran.
Treat as inputs to the planning round, not as a complete list.

## I-1: Gate sequencer does not enforce in-order passing
File: `gate_sequencing/sequencer.py`
- `update()` only tests the plane-crossing of `self._gates[self._current_idx]` (the *current* target). It records a `miss` if any plane crossing happens outside the current gate's frame.
- **It does not test other gates' planes.** A drone that flies straight through gates 4, 5, 6 while current is 3 generates **no event for those gates** — they're not credited and not flagged as misses.
- Worse: if the drone then returns to gate 3, it gets credited; current becomes 4. The drone is already past gate 4 (which was never credited), so the next plane crossing event after a U-turn could credit gate 4. End state: "all gates passed" with U-turn between them — **counted as a clean run** by `sequencer.is_complete`.
- **Fix direction**: track plane crossings of every still-unpassed gate per tick. If any out-of-order gate is crossed inside its opening, mark it as `out_of_order` (terminal fail) regardless of current target.

## I-2: Crashes are not terminal in the synthetic bench
File: `scripts/benchmark.py:417-425`
- `run_synthetic_benchmark` only terminates on `pos[2] < 0.05` (ground) or `pos[2] > 20.0` (ceiling).
- The platform-agnostic `GateSequencer.mark_collision()` exists and is wired in `sim_pybullet/runner.py`, but **the synthetic bench never calls it** — there is no gate-frame collision check in the synthetic kinematic loop.
- The PyBullet bench has it (`scripts/benchmark.py:746-750`), gated on `env.gate_contact()`.
- **Fix direction**: synthetic bench needs a geometric crash check — any segment from `prev_pos` to `pos` whose plane crossing falls in `[interior, outer_frame]` is a crash; abort the run.

## I-3: Magic numbers tuned to race_01 helix
File: `scripts/benchmark.py:322-333`
```
inflection_start = int(2.0 / dt)    # step 200
inflection_end   = int(4.4 / dt)    # step 440
helix_start      = int(7.4 / dt)    # step 740
```
- These are wall-clock seconds derived from the race_01.json helix at gate-3 / gate-6 / gate-7.
- The whole `section_boundaries` table in lines 325-333 is hand-tuned alpha / cutoff / vel_scale per these time windows.
- **Fix direction**: derive section boundaries from track topology (e.g. curvature spikes along the racing line) rather than wall-clock seconds; or fall back to a single global ILC if a per-section schedule can't be derived robustly.

## I-4: ILC convergence threshold and momentum hand-tuned
File: `scripts/benchmark.py:344-350`
- `convergence_threshold=0.0005`, `momentum_gamma=0.2` are documented as "iter 47-49 best-of-N from a sweep on race_01".
- Highly likely to over-correct on a different track.
- **Fix direction**: park the sweep parameters as defaults and add a regression suite that runs ≥3 synthetic tracks (not all the same shape) so a single course can't dominate the loss.

## I-5: Hard-coded gate dimensions don't match AIGP
- `GateSpec` defaults: `interior_width=1.2`, `interior_height=1.2`, `border_width=0.15`.
- AIGP actual: inner 1.5 m × 1.5 m, frame thickness 0.6 m on each side, depth 0.26 m.
- `race_01.json` gates use 1.2 m as well — local artifact, not the competition.
- **Fix direction**: default `GateSpec` to AIGP geometry. Update any config that overrode the default.

## I-6: Drone footprint and dynamics not validated against AIGP
- PyBullet uses Crazyflie-class dynamics: 1.0 kg, 20 N thrust, ~92 mm chassis.
- AIGP drone: 280 × 280 × 160 mm chassis. Mass / thrust unknown — must be inferred from SITL.
- **Fix direction**: build a calibration harness that connects to the DCL SITL over MAVLink2 UDP, runs a few characterised maneuvres (step, hover, ramp), fits mass / thrust / drag online, and writes a `drone_calibration.json` consumed by the controller config.

## I-7: Two parallel pipelines — runner vs. race_pipeline
- `sim_pybullet/runner.py` runs an ad-hoc detect→target→control chain.
- `race_pipeline.py` runs the proper PnP→EKF→trajectory→tracker stack.
- PLAN.md (2026-04-03) recommended collapsing onto the second; never finished.
- **Fix direction**: complete the collapse. `runner.py` becomes a thin PyBullet harness that calls `RacePipeline.run()`; all autonomy lives in `race_pipeline.py`.

## I-8: Camera config is 640×480, AIGP is 640×360 with 20° upward tilt
- `PipelineConfig` defaults: `image_height: 480`, `camera_fov_h: 90.0`.
- AIGP: 640 × 360, tilted 20° up from body-frame, fx = fy = 320, cx = 320, cy = 180.
- **Fix direction**: update `PipelineConfig` defaults and propagate the 20° pitch tilt into `CameraIntrinsics` / `gate_pnp.py`.

## I-9: MAVLink bridge may not handle the 30 Hz JPEG-chunked vision stream
File: `competition/mavlink_bridge.py`
- The spec defines a *separate* UDP port 5600 with chunked JPEG payload (24-byte header).
- The bridge presently advertises MAVLink-only; verify whether it reassembles JPEG chunks. If not, write a `VisionUdpReceiver` that listens on 5600, reassembles by `frame_id` + `chunk_id`, and feeds the camera frame to the pipeline.

## I-10: No ML model in the stack despite being explicitly asked for
- All gate detection paths are classical (HSV + edge + cluster in `gate_detection/src/`) or YOLOv8n-pose ONNX in `gate_detection/training/` — but the runner doesn't pick up the trained ONNX.
- **Fix direction**: ship at least one of (a) a tiny CNN gate-corner regressor, (b) a learned residual ILC on top of the geometric tracker, (c) a learned model-based predictor for state extrapolation. Pick whichever has the lowest implementation cost and highest robustness gain.

## I-11: Adversarial test coverage near-zero
- `tests/test_race_pipeline_replan_integration.py` covers happy-path replanning.
- No test verifies: out-of-order pass detection, gate-frame crash detection, untracked-gate detection, perception poison rejection, drone-model swap.
- **Fix direction**: ship `tests/test_sequencer_adversarial.py` and `tests/test_benchmark_adversarial.py` BEFORE the main fix lands, so the fix has a concrete bar to clear.
