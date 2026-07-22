# Run Guide

## Active VQ2 workflow (Windows)

Use CPython 3.12 and the exact development lock:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\development-test.lock.txt
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-vq2
```

Run `.\scripts\dev.cmd test-target <paths>` after each edit. The pure-Python
synthetic evaluator is available in the development profile. Actual PyBullet
and visualization runs require the separate
`requirements\legacy-simulation.txt` profile. Benchmark tests run only via the
explicit `test-benchmark` or `test-promotion` tasks; they are not the universal
VQ2 objective. The durable promotion task normally takes 10-13 minutes and
reattaches to an existing exact-candidate run rather than starting a duplicate.
`test-full-non-live` remains a compatibility alias.

`.\scripts\dev.cmd preflight` is passive. Powered FlightSim stages require
explicit user authorization and are never part of a generic test task. Read
`docs/aigp/2026-07-18-vq2-handoff.md` before any live work.

The remaining sections describe legacy visualization and simulation tools.
Install `requirements\legacy-simulation.txt` before using them.

## Generate Example Field Output
Use this to produce a concrete example scene with gates + path and camera snapshots:

```bash
cd /home/john/grand_prix/killallhumans
python3 -m simulation.demo
```

If your shell is not at repo root, this also works:

```bash
python3 /home/john/grand_prix/killallhumans/simulation/demo.py
```

Scene config (YAML) used by default:
- `simulation/configs/field_demo.yaml`
- This file defines:
  - gate geometry (`interior_width_m`, `interior_height_m`, `border_width_m`, `depth_m`)
  - per-gate pose (`x`, `y`, `z`, `yaw`, `pitch`, `roll`)
  - path control points
- It supports any number of gates; each gate can override `config.color`.

Generated files:
- `simulation/example_output/primary_camera_view.ppm`
- `simulation/example_output/free_roam_camera_view.ppm`
- `simulation/example_output/scene_summary.json`

These are snapshot artifacts (still images), not a live viewer.

## Interactive Free-Roam Viewer
To actually move around the field with a free-roam camera:

```bash
cd /home/john/grand_prix/killallhumans
python3 -m simulation.demo --interactive
```

Controls in window:
- Matplotlib fallback controls:
  - `E` / `D`: forward / back
  - `A` / `G`: strafe left / right
  - `Y` / `H`: up / down
  - `L` / `K`: yaw left / right
  - `U` / `J`: pitch up / down
  - `p`: print current camera pose to terminal
  - `q`: close viewer

Requirements:
- A graphical desktop session with either:
  - `DISPLAY` set (X11/XWayland), or
  - `WAYLAND_DISPLAY` set (native Wayland)
- If remote, use X forwarding (`ssh -X`) or equivalent remote desktop setup.
- On headless servers, use snapshot mode (omit `--interactive`).
- In Wayland sessions without `DISPLAY`, the demo automatically falls back to a matplotlib 3D interactive viewer.
- If only non-interactive matplotlib backends are available (e.g. `Agg`), install a GUI backend (Qt/Tk) or use snapshot mode.

## View Demo Output
### 1) Open the generated images directly
Linux:

```bash
xdg-open simulation/example_output/primary_camera_view.ppm
xdg-open simulation/example_output/free_roam_camera_view.ppm
```

macOS:

```bash
open simulation/example_output/primary_camera_view.ppm
open simulation/example_output/free_roam_camera_view.ppm
```

### 2) Inspect scene metadata
```bash
cat simulation/example_output/scene_summary.json
```

### 3) Optional: convert to PNG for easier sharing
If ImageMagick is installed:

```bash
convert simulation/example_output/primary_camera_view.ppm simulation/example_output/primary_camera_view.png
convert simulation/example_output/free_roam_camera_view.ppm simulation/example_output/free_roam_camera_view.png
```

### 4) Python fallback viewer (no desktop opener required)
```bash
python3 - <<'PY'
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for name in ["primary_camera_view.ppm", "free_roam_camera_view.ppm"]:
    path = Path("simulation/example_output") / name
    img = mpimg.imread(path)
    plt.figure(figsize=(10, 4))
    plt.title(name)
    plt.imshow(img)
    plt.axis("off")
plt.show()
PY
```

## Quick Python Usage Example
```python
from simulation import (
    CameraPose,
    Pose3D,
    build_field_from_yaml,
    build_path_from_yaml,
    build_sample_field,
    build_sample_path,
    render_scene,
)

field = build_sample_field()
path = build_sample_path()
# Or explicit YAML loading:
# field = build_field_from_yaml(Path("simulation/configs/field_demo.yaml"))
# path = build_path_from_yaml(Path("simulation/configs/field_demo.yaml"))
viewer = render_scene(field, path)

viewer.set_free_roam(True)
viewer.update_primary_camera(CameraPose(pose=Pose3D(0.0, 0.0, 1.5)))
frame = viewer.snapshot(include_depth=True)
print(len(frame.visible_gates), frame.outside_field)
```

## Core API Entry Points
- `simulation.generate_gate(...)`
- `simulation.generate_field(...)`
- `simulation.build_path(...)`
- `simulation.get_camera_view(...)`
- `simulation.render_scene(...)`

## Notes
- The renderer works even without PyVista; if PyVista is available, scene objects are also constructed in a plotter.
- The camera frame is currently represented as Python lists (RGB tuples + optional depth map) for portability in MVP.

---

## PyBullet Drone Racing Simulation

### Install PyBullet

Use the isolated legacy-simulation profile described in
`docs/development_environment.md`; do not add PyBullet to the normal VQ2
`.venv`. On a Bash host, for example:

```bash
VENV_DIR="$PWD/.venv-legacy" \
REQUIREMENTS_FILE="$PWD/requirements/legacy-simulation.txt" \
bash scripts/setup_venv.sh
source .venv-legacy/bin/activate
```

### Run the Race Simulation
From the repo root:

```bash
# Default: sim-metadata detection (fast, uses known gate positions)
python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json

# With real gate detection pipeline
python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json --use-detection

# With Phase 1 detector (highlighted gates)
python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json --use-detection --detector phase1

# Headless (no PyBullet GUI window, still shows OpenCV HUD)
python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json --no-gui
```

### Controls During Simulation
- `Q` — quit
- `R` — reset simulation

### Simulation Display
- **Left panel**: 1st-person FPV from drone camera (with detection overlay when `--use-detection`)
- **Right panel**: 3rd-person spectator camera (chase view)
- **HUD**: speed, altitude, target gate, gates passed, elapsed time, distance to gate

### Race Configuration
Race configs are JSON files in `sim_pybullet/configs/`. They define:
- Field bounds
- Gate positions, sizes, colors, and sequence order
- Drone start position and heading
- Physics timestep and gravity

See `sim_pybullet/configs/race_01.json` for an example.
