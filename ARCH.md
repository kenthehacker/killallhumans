# Architecture

## Current System Layout
- `flight_control/`: controller stack (`MPCPlanner`, PID, `TRPYMixer`, target adapters)
- `gate_detection/`: classical CV + YOLOv8-pose fused gate detection, Phase 1 detector
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
- `sim_pybullet/sequencer.py` — `GateSequencer`: tracks gate order, detects
  pass-through events via signed-distance plane crossing, manages gate highlighting.
- `sim_pybullet/runner.py` — `RaceRunner`: the main closed-loop. Ties physics
  stepping, camera rendering, detection, flight control, and HUD display together.

### Data Flow (Closed Loop)
1. `pybullet.stepSimulation()` → advance physics
2. Read drone state (position, velocity, orientation)
3. Render FPV camera → RGB image
4. Gate detection (real pipeline or sim metadata)
5. Gate sequencing → target gate selection
6. `FlightController.step_trpy()` → `TRPYCommand`
7. `QuadrotorDrone.apply_command()` → forces/torques in PyBullet
8. Render dual-camera HUD display
9. Check gate pass-through → advance sequence
10. Loop until all gates passed or timeout

### Detection Modes
- **Sim metadata** (default): uses known gate positions for fast iteration
- **Real detection** (`--use-detection`): runs actual `gate_detection` pipeline on rendered frames
- **Phase 1** (`--detector phase1`): optimized for highlighted gates in desaturated environment

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
- `FusedGateDetector` — classical + YOLOv8n-pose, IoU-matched fusion
- `Phase1GateDetector` — saturation/brightness thresholding for VQ1

### Training Pipeline
- `training/extract_frames.py` — extracts TII dataset frames with YOLO-pose labels
- `training/train.py` — trains YOLOv8n-pose on the extracted dataset
- `training/validate.py` — validation metrics + problem frame analysis
- `training/export.py` — exports to ONNX for deployment

## Tradeoffs
- Two simulation systems: lightweight (fast, testable) + PyBullet (realistic, heavy)
- TRPY mixer is a linear approximation; works for moderate attitudes but degrades at extreme angles
- Phase 1 detector is deliberately simple — will need retuning once we see the actual VQ1 environment

## Maintenance Rule
Keep this file updated whenever:
- public interfaces change,
- data flow changes,
- a subsystem is added/removed,
- tradeoffs materially shift.
