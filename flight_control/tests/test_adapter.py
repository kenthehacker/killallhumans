import math
import unittest

from flight_control.adapter import CameraModel, gate_detection_to_target
from flight_control.types import DroneState


class DummyDetection:
    def __init__(self, normalized_center_x: float, normalized_center_y: float, estimated_distance: float):
        self.normalized_center_x = normalized_center_x
        self.normalized_center_y = normalized_center_y
        self.estimated_distance = estimated_distance


class TestGateAdapter(unittest.TestCase):
    def test_center_gate_forward(self) -> None:
        detection = DummyDetection(0.0, 0.0, 10.0)
        state = DroneState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        target = gate_detection_to_target(detection, state, CameraModel())
        self.assertAlmostEqual(target.position[0], 10.0, delta=1e-3)
        self.assertAlmostEqual(target.position[1], 0.0, delta=1e-3)
        self.assertAlmostEqual(target.position[2], 0.0, delta=1e-3)

    def test_yaw_rotation(self) -> None:
        detection = DummyDetection(0.0, 0.0, 8.0)
        state = DroneState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), math.pi / 2)
        target = gate_detection_to_target(detection, state, CameraModel())
        self.assertAlmostEqual(target.position[0], 0.0, delta=1e-3)
        self.assertAlmostEqual(target.position[1], 8.0, delta=1e-3)

    def test_vertical_offset(self) -> None:
        detection = DummyDetection(0.0, 0.7, 5.0)
        state = DroneState((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0)
        target = gate_detection_to_target(detection, state, CameraModel())
        self.assertLess(target.position[2], state.position[2])


if __name__ == "__main__":
    unittest.main()
