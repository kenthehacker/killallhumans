import unittest

from simulation.camera import get_camera_view
from simulation.field import generate_field
from simulation.gates import generate_gate
from simulation.model_types import CameraPose, FieldConfig, GateConfig, Pose3D


class TestCameraView(unittest.TestCase):
    def test_gate_visible_in_front_camera(self) -> None:
        gate = generate_gate(GateConfig(), Pose3D(8.0, 0.0, 1.5), "gate-front")
        field = generate_field(FieldConfig(bounds_min=(0.0, -5.0, 0.0), bounds_max=(20.0, 5.0, 5.0)), [gate])

        pose = CameraPose(pose=Pose3D(0.0, 0.0, 1.5), resolution_width=320, resolution_height=240)
        frame = get_camera_view(field, pose, include_depth=True)

        self.assertEqual(len(frame.rgb), 240)
        self.assertEqual(len(frame.rgb[0]), 320)
        self.assertEqual(len(frame.visible_gates), 1)
        self.assertFalse(frame.outside_field)
        self.assertIsNotNone(frame.depth)

    def test_outside_field_flag(self) -> None:
        gate = generate_gate(GateConfig(), Pose3D(8.0, 0.0, 1.5), "gate-front")
        field = generate_field(FieldConfig(bounds_min=(0.0, -5.0, 0.0), bounds_max=(20.0, 5.0, 5.0)), [gate])
        pose = CameraPose(pose=Pose3D(-1.0, 0.0, 1.5))
        frame = get_camera_view(field, pose)
        self.assertTrue(frame.outside_field)


if __name__ == "__main__":
    unittest.main()
