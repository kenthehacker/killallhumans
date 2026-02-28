import unittest

from simulation.field import generate_field, get_gate, nearest_gate
from simulation.gates import generate_gate
from simulation.model_types import FieldConfig, GateConfig, Pose3D


class TestField(unittest.TestCase):
    def setUp(self) -> None:
        self.g1 = generate_gate(GateConfig(color="red"), Pose3D(5.0, 0.0, 1.0), "g1")
        self.g2 = generate_gate(GateConfig(color="blue"), Pose3D(15.0, 0.0, 1.0), "g2")

    def test_generate_field(self) -> None:
        field = generate_field(FieldConfig(), [self.g1, self.g2])
        self.assertEqual(len(field.gates), 2)

    def test_generate_field_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            generate_field(FieldConfig(), [self.g1, self.g1])

    def test_get_gate_and_nearest(self) -> None:
        field = generate_field(FieldConfig(), [self.g1, self.g2])
        self.assertEqual(get_gate(field, "g2"), self.g2)
        nearest = nearest_gate(field, (4.0, 0.0, 1.0))
        self.assertEqual(nearest, self.g1)


if __name__ == "__main__":
    unittest.main()
