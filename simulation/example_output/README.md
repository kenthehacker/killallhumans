# Example Output

This folder contains generated demo outputs for the sample simulation field.

## Regenerate
From repo root:

```bash
python3 -m simulation.demo --output-dir simulation/example_output
```

## Files
- `primary_camera_view.ppm`: forward-facing camera from near the start of the course
- `free_roam_camera_view.ppm`: spectator/free-roam style camera angle
- `scene_summary.json`: gate positions, path stats, and visible-gate summary

`.ppm` files can be opened by most image viewers (or converted with ImageMagick).
