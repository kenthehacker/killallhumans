# Architecture

## Current System Layout
- `flight_control/`: controller stack (`MPCPlanner`, PID, target adapters)
- `gate_detection/`: classical CV gate detection and feature extraction
- `simulation/`: synthetic field/gate/path/camera environment for integration and testing
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

## Data Flow
1. Build gates -> build field
2. Build path from control points
3. Render/snapshot scene at any camera pose
4. Optionally bridge outputs into flight control target interfaces

## Tradeoffs (Current)
- 2.5D kinematic model chosen over rigid-body physics:
  - Pro: fast iteration, deterministic behavior, low complexity
  - Con: lower fidelity for dynamics/collision realism
- Camera rendering is lightweight raster overlay, not photoreal:
  - Pro: fast, testable, dependency-light
  - Con: limited realism for CV stress-testing
- PyVista is optional and not required for core tests:
  - Pro: portability/CI friendliness
  - Con: interactive UX depends on local graphics setup
- Path representation is spline control points only (MVP):
  - Pro: concise racing-line definition
  - Con: no native support yet for constrained optimization paths

## Future Improvements
1. Physics fidelity
   - Add rigid-body dynamics, drag, thrust constraints, and collision checks.
2. Rendering fidelity
   - Add textured meshes, lighting models, and sensor-noise simulation.
3. Camera outputs
   - Add segmentation/depth as first-class products and NumPy-backed buffers.
4. Planning integration
   - Add direct interfaces for sequence-aware gate traversal and trajectory timing.
5. Performance
   - Vectorize projection/raster operations and add batched frame generation.
6. Tooling
   - Extend bootstrap scripts for OS-specific dependency checks and better diagnostics.

## Maintenance Rule
Keep this file updated whenever:
- public interfaces change,
- data flow changes,
- a subsystem is added/removed,
- tradeoffs materially shift.
