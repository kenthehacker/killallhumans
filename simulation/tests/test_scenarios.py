import unittest

from simulation.scenarios import DEFAULT_SCENE_CONFIG, build_sample_field, build_sample_path


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


if __name__ == "__main__":
    unittest.main()
