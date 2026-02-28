import unittest

from simulation.pathing import build_path
from simulation.model_types import PathSpec


class TestPathing(unittest.TestCase):
    def test_build_path_from_spline_points(self) -> None:
        spec = PathSpec(
            control_points=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (3.0, 1.0, 0.0),
            ],
            samples_per_segment=10,
        )
        polyline = build_path(spec)
        self.assertGreater(len(polyline.points), len(spec.control_points))
        self.assertAlmostEqual(polyline.points[0][0], 0.0, delta=1e-6)
        self.assertAlmostEqual(polyline.points[-1][0], 3.0, delta=1e-6)
        self.assertGreater(polyline.total_length, 0.0)

    def test_build_path_rejects_collapsed_input(self) -> None:
        spec = PathSpec(control_points=[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
        with self.assertRaises(ValueError):
            build_path(spec)


if __name__ == "__main__":
    unittest.main()
