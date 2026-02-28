# Simulation Environment MVP Plan

## Goal
Build a simulation environment for autonomous drone-racing development that supports:
1. Configurable gate generation
2. Field generation with gates at arbitrary poses
3. Camera-view rendering from any field position
4. Path display on the field
5. Independent free-roam spectator camera

## Scope (Implemented)
- New `simulation/` Python package with typed APIs.
- 2.5D kinematic world representation (3D coordinates, simplified rendering/projection).
- Optional PyVista-backed scene viewer, with a fallback that still supports programmatic snapshots.
- Thin adapters for interoperability with existing `flight_control` interfaces.
- Unit and smoke tests under `simulation/tests`.

## Deliverables
- `simulation/model_types.py`: dataclasses and validation
- `simulation/gates.py`: configurable gate creation
- `simulation/field.py`: field assembly + spatial helpers
- `simulation/pathing.py`: spline control points -> sampled path polyline
- `simulation/camera.py`: camera projection + RGB/metadata frame output
- `simulation/renderer.py`: scene viewer, free-roam control, snapshots
- `simulation/adapters.py`: bridges to `flight_control`
- `simulation/scenarios.py`: sample field/path builders
- `simulation/demo.py`: reproducible example field + snapshot generator
- `simulation/tests/*`: coverage for each major subsystem

## Success Criteria
- Gates can be generated with validated dimensions and unique IDs.
- Field can hold multiple gates and support nearest/lookup operations.
- Camera API returns RGB frame plus structured visible-gate metadata.
- Path API accepts spline control points and renders sampled trajectory.
- Free-roam camera can be toggled independently of primary camera logic.
- All simulation tests pass without regressions in `flight_control` tests.

## Validation Status
- `simulation/tests`: passing
- `flight_control/tests`: passing

## Assumptions / Defaults
- Right-handed frame with meters/radians.
- MVP favors deterministic typed interfaces over high-fidelity physics.
- Camera API first-class output is `RGB + metadata`.
- Path input is spline control points.

## Change Policy
When architecture or behavior changes:
1. Update this plan with new scope/status.
2. Update `ARCH.md` with structural and interface changes.
3. Update `RUN.md` if commands, deps, or workflows change.
