from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

if __package__ in (None, ""):
    import sys

    # Allow direct execution: python3 /abs/path/to/simulation/demo.py
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.renderer import render_scene
    from simulation.scenarios import DEFAULT_SCENE_CONFIG, build_sample_field, build_sample_path
    from simulation.model_types import CameraPose, Pose3D
else:
    from .renderer import render_scene
    from .scenarios import DEFAULT_SCENE_CONFIG, build_sample_field, build_sample_path
    from .model_types import CameraPose, Pose3D


def run_demo(output_dir: Path, interactive: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = DEFAULT_SCENE_CONFIG
    field = build_sample_field(config_path=config_path)
    path = build_sample_path(config_path=config_path)
    viewer = render_scene(field, path)

    if interactive:
        viewer.launch_free_roam()

    primary_pose = CameraPose(
        pose=Pose3D(0.0, 0.0, 1.6, yaw=0.0, pitch=0.0, roll=0.0),
        resolution_width=640,
        resolution_height=360,
    )
    viewer.update_primary_camera(primary_pose)
    primary_frame = viewer.snapshot(include_depth=False)
    primary_file = output_dir / "primary_camera_view.ppm"
    _write_ppm(primary_file, primary_frame.rgb)

    viewer.set_free_roam(True)
    roam_pose = CameraPose(
        pose=Pose3D(12.0, -8.0, 5.0, yaw=0.55, pitch=-0.22, roll=0.0),
        resolution_width=640,
        resolution_height=360,
    )
    viewer.update_primary_camera(roam_pose)
    roam_frame = viewer.snapshot(include_depth=False)
    roam_file = output_dir / "free_roam_camera_view.ppm"
    _write_ppm(roam_file, roam_frame.rgb)

    summary = {
        "field_name": field.config.name,
        "scene_config": str(config_path),
        "gate_count": len(field.gates),
        "gates": [
            {
                "id": gate.gate_id,
                "sequence_index": gate.sequence_index,
                "position": [gate.pose.x, gate.pose.y, gate.pose.z],
                "rotation_rpy": [gate.pose.roll, gate.pose.pitch, gate.pose.yaw],
                "color": gate.config.color,
                "interior_width_m": gate.config.interior_width_m,
                "interior_height_m": gate.config.interior_height_m,
                "border_width_m": gate.config.border_width_m,
                "depth_m": gate.config.depth_m,
            }
            for gate in field.gates
        ],
        "path_point_count": len(path.points),
        "path_total_length_m": round(path.total_length, 3),
        "primary_visible_gates": [annotation.gate_id for annotation in primary_frame.visible_gates],
        "free_roam_visible_gates": [annotation.gate_id for annotation in roam_frame.visible_gates],
        "outputs": {
            "primary_camera": str(primary_file),
            "free_roam_camera": str(roam_file),
        },
    }

    summary_file = output_dir / "scene_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_ppm(path: Path, rgb: List[List[Tuple[int, int, int]]]) -> None:
    if not rgb or not rgb[0]:
        raise ValueError("Cannot write empty image")
    height = len(rgb)
    width = len(rgb[0])
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for row in rgb:
            if len(row) != width:
                raise ValueError("Inconsistent row width in image")
            for r, g, b in row:
                handle.write(bytes((max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b))))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo simulation field snapshots")
    parser.add_argument(
        "--output-dir",
        nargs="?",
        const=Path("simulation/example_output"),
        type=Path,
        default=Path("simulation/example_output"),
        help="Directory where demo outputs are written",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive free-roam viewer window before writing snapshots (requires PyVista).",
    )
    args = parser.parse_args()

    summary = run_demo(args.output_dir, interactive=args.interactive)
    print("Demo generated:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
