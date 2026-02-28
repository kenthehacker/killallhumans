import unittest
import tempfile
from pathlib import Path
import json

from simulation.scenarios import (
    DEFAULT_SCENE_CONFIG,
    build_field_from_yaml,
    build_path_from_yaml,
    build_sample_field,
    build_sample_path,
)


class TestScenarios(unittest.TestCase):
    def test_sample_scene_yaml_exists(self) -> None:
        self.assertTrue(DEFAULT_SCENE_CONFIG.exists())

    def test_build_sample_field_from_yaml(self) -> None:
        field = build_sample_field()
        self.assertEqual(len(field.gates), 3)
        z_values = [gate.pose.z for gate in field.gates]
        yaw_values = [gate.pose.yaw for gate in field.gates]
        self.assertGreater(len(set(z_values)), 1)
        self.assertTrue(any(abs(yaw) > 0.01 for yaw in yaw_values))

    def test_build_sample_path_from_yaml(self) -> None:
        path = build_sample_path()
        self.assertGreater(path.total_length, 0.0)
        self.assertGreater(len(path.points), 20)

    def test_build_custom_yaml_supports_many_gates_colors_and_rotations(self) -> None:
        scene = {
            "field": {
                "name": "custom-many-gates",
                "bounds_min": [0.0, -20.0, 0.0],
                "bounds_max": [50.0, 20.0, 12.0],
            },
            "gate_defaults": {
                "interior_width_m": 1.0,
                "interior_height_m": 1.0,
                "border_width_m": 0.12,
                "depth_m": 0.08,
                "color": "white",
            },
            "gates": [
                {"id": "g1", "pose": {"x": 5, "y": 0, "z": 1.2, "yaw": 0.0}, "config": {"color": "red"}},
                {"id": "g2", "pose": {"x": 10, "y": 1, "z": 1.8, "yaw": 0.2}, "config": {"color": "blue"}},
                {"id": "g3", "pose": {"x": 15, "y": -2, "z": 2.2, "yaw": -0.3}, "config": {"color": "green"}},
                {"id": "g4", "pose": {"x": 20, "y": 3, "z": 2.0, "yaw": 0.4}, "config": {"color": "orange"}},
                {"id": "g5", "pose": {"x": 25, "y": -1, "z": 1.6, "yaw": -0.1}, "config": {"color": "purple"}},
            ],
            "path": {
                "control_points": [
                    [0.0, 0.0, 1.5],
                    [5.0, 0.0, 1.2],
                    [10.0, 1.0, 1.8],
                    [15.0, -2.0, 2.2],
                    [25.0, -1.0, 1.6],
                ],
                "samples_per_segment": 10,
                "closed": False,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "scene.yaml"
            config_path.write_text(json.dumps(scene), encoding="utf-8")

            field = build_field_from_yaml(config_path)
            path = build_path_from_yaml(config_path)

        self.assertEqual(len(field.gates), 5)
        self.assertEqual(field.gates[0].config.color, "red")
        self.assertEqual(field.gates[4].config.color, "purple")
        self.assertTrue(any(abs(g.pose.yaw) > 0.01 for g in field.gates))
        self.assertGreater(path.total_length, 0.0)


if __name__ == "__main__":
    unittest.main()
