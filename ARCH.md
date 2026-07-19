# Architecture

## Current System Layout
- `flight_control/`: controller stack (`MPCPlanner`, PID, `TRPYMixer`, target adapters)
- `gate_detection/`: build-3385 VQ2 classical red-gate detector plus historical/general classical detectors; no fused or learned detector is integrated
- `competition/` + `scripts/aigp_vq2_run.py`: current build-3385 camera, `HIGHRES_IMU`, race-status, safety, and staged-control runtime
- `simulation/`: lightweight synthetic field/gate/path/camera environment (no physics)
- `sim_pybullet/`: PyBullet-based closed-loop simulation with real physics, gate sequencing, dual-camera rendering
- `scripts/`: environment bootstrap and demo execution helpers

## Simulation Architecture

### 1. Domain Types (`simulation/model_types.py`)
Defines strongly-typed dataclasses for:
- World objects: `GateConfig`, `Gate`, `FieldConfig`, `Field`
- Camera model: `CameraPose`, `CameraFrame`, `VisibleGateAnnotation`
- Path model: `PathSpec`, `PathPolyline`
- Shared pose/geometry: `Pose3D`

`GateConfig` uses explicit geometric parameters:
- `interior_width_m`, `interior_height_m` (opening)
- `border_width_m` (frame border)
- `depth_m` (small thickness through gate depth axis)

### 2. Scene Construction
- `simulation/gates.py`: builds validated `Gate` objects.
- `simulation/field.py`: assembles a field and provides:
  - gate lookup (`get_gate`)
  - nearest-gate query (`nearest_gate`)
  - coarse visibility prefilter (`visible_gate_prefilter`)
- `simulation/scenarios.py`: loads scene definitions from YAML (`simulation/configs/field_demo.yaml` by default), including gate geometry and per-gate pose (x/y/z + yaw/pitch/roll).

### 3. Path Subsystem (`simulation/pathing.py`)
- Input: spline control points (`PathSpec`)
- Processing: Catmull-Rom interpolation + repeat-point collapse
- Output: sampled polyline + cumulative lengths (`PathPolyline`)

### 4. Camera Subsystem (`simulation/camera.py`)
- Input: `Field`, `CameraPose`
- Steps:
  - radius-based candidate filtering
  - world-to-camera transform using yaw/pitch/roll inverse rotation
  - perspective projection to normalized image plane
  - gate annotation generation and raster overlay
- Output: `CameraFrame` with RGB + metadata (+ optional depth)

### 5. Rendering / Interaction (`simulation/renderer.py`)
- `SimulationViewer` coordinates field/path rendering and camera snapshots.
- Free-roam camera state is represented by `FreeRoamCameraController` and can be toggled independently.
- PyVista integration is optional; if unavailable, snapshot logic still works.
- Gate rendering uses a true 3D frame model (top/bottom/left/right segments) derived from interior/border/depth config, with per-gate rotation applied.

### 6. Integration Edges (`simulation/adapters.py`)
- Adapts detection-like objects to `flight_control` target states.
- Keeps existing subsystems decoupled from simulation internals.

### 7. Example Demo (`simulation/demo.py`)
- Builds a sample field (`3` gates) and spline path.
- Renders two snapshots:
  - primary forward camera
  - free-roam spectator-style camera
- Supports interactive free-roam mode (`--interactive`) using a live PyVista window.
- Writes artifacts to `simulation/example_output/` for quick inspection.

## PyBullet Simulation Architecture (`sim_pybullet/`)

### Overview
Closed-loop simulation with realistic rigid-body physics. Runs the full
autonomy stack: camera → detection → sequencing → planning → control → physics.

### Components
- `sim_pybullet/drone.py` — `QuadrotorDrone`: box-body drone with attitude-level
  inner-loop controller. Accepts normalized (throttle, roll, pitch, yaw). Provides
  FPV and spectator camera images via `pybullet.getCameraImage()`.
- `sim_pybullet/gate_models.py` — Creates gate frame segments as static PyBullet
  bodies. Supports color changes for highlight/dim/reset (gate sequencing visuals).
- `sim_pybullet/env.py` — `DroneRaceEnv`: manages the PyBullet physics client,
  ground plane, gate placement, and drone spawning. Loads race configs from JSON.
  Exposes `gate_contact()` returning the `gate_id` of any current PyBullet contact
  manifold against a gate body (or `None`).
- `sim_pybullet/_gate_to_spec.py` — adapter `to_spec(Gate) -> GateSpec`. The
  sim_pybullet sequencer was collapsed (P2-1, 2026-05-09) into the platform-
  agnostic `gate_sequencing.GateSequencer`; this adapter projects sim Gates
  through.
- `sim_pybullet/runner.py` — `RaceRunner`: the main closed-loop. Ties physics
  stepping, camera rendering, detection, flight control, and HUD display together.
  Exposes `crashed_into_gate` (most recent crash gate_id, or `None`) — backed by
  the sequencer's crash log so it survives a single-tick contact and can be
  inspected after the run ends.
- `planning/dynamic_replanner.py` — `DynamicReplanner`: stateful policy that
  decides *when* to rebuild the racing line (gate collision, missed gate,
  off-track recovery, sustained lateral error) and constructs the new waypoint
  list from the drone's current state. Cooldown prevents replan storms; level
  signals are edge-triggered so the trigger field reads True once per
  perturbation.

### Data Flow (Closed Loop)
1. `pybullet.stepSimulation()` → advance physics
2. Read drone state (position, velocity, orientation)
3. Render FPV camera → RGB image
4. Gate detection (real pipeline or sim metadata)
5. Gate sequencing → target gate selection
   5a. Poll `env.gate_contact()` — first tick of a new contact calls
       `sequencer.mark_collision(gate_id)`. The runner's
       `_last_contact_gate_id` dedupes the persistent PyBullet manifold.
6. Compute lateral_error against the racing line, then run
   `replanner.evaluate(...)` and rebuild `_racing_line` from the drone's
   current position on a positive trigger (`_maybe_replan`).
7. `FlightController.step_trpy()` → `TRPYCommand`
8. `QuadrotorDrone.apply_command()` → forces/torques in PyBullet
9. Render dual-camera HUD display
10. Check gate pass-through → advance sequence
11. Loop until all gates passed or timeout

### Detection Modes
- **Sim metadata** (default): uses known gate positions for fast iteration
- **Real detection** (`--use-detection`): runs actual `gate_detection` pipeline on rendered frames
- **Phase 1** (`--detector phase1`): optimized for highlighted gates in desaturated environment

These modes describe the secondary PyBullet runner. The competition-facing
build-3385 runtime uses `VQ2GateDetector` with the asynchronous VQ2 vision
receiver and gyro-only post-bootstrap attitude estimator.

## Flight Control Architecture

### Control Pipeline
```
TargetState → MPCPlanner → desired velocity/yaw
                              ↓
                         PID controllers (vx, vy, vz, yaw)
                              ↓
                         ControlCommand (ax, ay, az, yaw_rate)
                              ↓
                         TRPYMixer
                              ↓
                         TRPYCommand (throttle, roll, pitch, yaw)
```

### TRPYMixer (`flight_control/mixer.py`)
Converts world-frame accelerations to competition-format controls:
- `throttle` = thrust needed to achieve vertical accel (gravity-compensated)
- `roll` = desired roll angle from lateral accel (body frame)
- `pitch` = desired pitch angle from forward accel (body frame)
- `yaw` = direct pass-through of yaw rate

## Lightweight Simulation (`simulation/`)

(Preserved unchanged — see previous sections above.)

### 1–7: Same as before (domain types, scene construction, etc.)

## Data Flow (Lightweight)
1. Build gates → build field
2. Build path from control points
3. Render/snapshot scene at any camera pose
4. Optionally bridge outputs into flight control target interfaces

## Gate Detection Architecture (`gate_detection/`)

### Detectors
- `GateDetector` — color-agnostic classical pipeline (edge + clustering + HSV)
- `Phase1GateDetector` — saturation/brightness thresholding for VQ1
- `VQ2GateDetector` — low-latency red-gate segmentation used by the build-3385 runtime
- (`FusedGateDetector` was removed — module never landed; the `--detector fused`
  branch was deleted 2026-05-09 with P0-4.)

### Training Pipeline

There is no reproducible learned-detector pipeline in this checkout. The
historical `training/runs/gate_pose_v1/` outputs remain for evidence, but the
dataset and the previously documented extraction/train/validate/export scripts
are absent, and no VQ2 runtime path references those weights. See
`gate_detection/training/README.md` and use the read-only
`scripts/audit_yolo_experiment.py` audit before making any future training
decision. The dataset-manifest schema is prerequisite scaffolding, not a claim
that the missing data has known provenance.

## Tradeoffs
- Two simulation systems: lightweight (fast, testable) + PyBullet (realistic, heavy)
- TRPY mixer is a linear approximation; works for moderate attitudes but degrades at extreme angles
- The production VQ2 detector is deliberately small and build-specific; any
  threshold change needs private replay evidence before a powered trial. The
  historical Phase 1 and YOLO-pose artifacts are not production fallbacks.
- The attitude-control path is single-loop PD (no cascaded angle→rate PID inner
  loop). Acceptable around hover and low-speed manoeuvres; structurally unstable
  beyond hover (P0-5, deferred). Until the cascaded loop ships, target attitudes
  must stay clamped.
- Secondary geometric simulations may use a lenient `pass_through_margin`, but
  the current synthetic benchmark evaluator requires an actual plane crossing
  and keeps `crash_margin` separate. In the build-3385 competition runtime, organizer
  race-status/collision telemetry—not PyBullet geometry—is authoritative.

## Maintenance Rule
Keep this file updated whenever:
- public interfaces change,
- data flow changes,
- a subsystem is added/removed,
- tradeoffs materially shift.
