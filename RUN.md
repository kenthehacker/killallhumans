# Run Guide

## Prerequisites
- Python 3.10+
- From repo root: `/home/john/grand_prix/killallhumans`

Optional (for PyVista rendering path):
- `pyvista` and its rendering dependencies

## Quick Start (Brand New Machine)
From repo root, run:

```bash
bash scripts/bootstrap_demo.sh --interactive
```

What this does:
- Creates `.venv/` if missing
- Installs dependencies from `requirements.txt`
- Launches the demo in interactive free-roam mode
- Includes `PyQt6` so matplotlib fallback can open an interactive window on Wayland/X11.

If you only want snapshots:

```bash
bash scripts/bootstrap_demo.sh
```

## Manual Environment Setup
Create a virtual environment and install all required packages:

```bash
bash scripts/setup_venv.sh
```

Then run the demo with the venv Python:

```bash
bash scripts/run_demo.sh --interactive
```

## Run Tests
### Simulation package tests
```bash
bash -lc 'source .venv/bin/activate && python -m unittest discover -s simulation/tests -v'
```

### Existing flight control regression tests
```bash
bash -lc 'source .venv/bin/activate && python -m unittest discover -s flight_control/tests -v'
```

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
  - `W` / `S`: forward / back
  - `A` / `D`: strafe left / right
  - `R` / `F`: up / down
  - `J` / `L`: yaw left / right
  - `I` / `K`: pitch up / down
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
    build_sample_field,
    build_sample_path,
    render_scene,
)

field = build_sample_field()
path = build_sample_path()
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
