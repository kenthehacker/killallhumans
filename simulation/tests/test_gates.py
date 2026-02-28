import unittest

from simulation.gates import generate_gate
from simulation.model_types import GateConfig, Pose3D


class TestGateGeneration(unittest.TestCase):
    def test_generate_gate_success(self) -> None:
        gate = generate_gate(
            GateConfig(gate_type="square", width_m=1.2, height_m=1.2, color="orange"),
            Pose3D(10.0, 2.0, 1.8),
            gate_id="gate-A",
            sequence_index=1,
        )
        self.assertEqual(gate.gate_id, "gate-A")
        self.assertEqual(gate.pose.position, (10.0, 2.0, 1.8))
        self.assertEqual(gate.sequence_index, 1)

    def test_generate_gate_requires_id(self) -> None:
        with self.assertRaises(ValueError):
            generate_gate(GateConfig(), Pose3D(0.0, 0.0, 0.0), gate_id="")

    def test_gate_config_dimension_validation(self) -> None:
        with self.assertRaises(ValueError):
            GateConfig(width_m=0.0)


if __name__ == "__main__":
    unittest.main()
