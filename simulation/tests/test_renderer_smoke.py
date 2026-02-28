import unittest

from simulation.renderer import render_scene
from simulation.scenarios import build_sample_field, build_sample_path


class TestRendererSmoke(unittest.TestCase):
    def test_viewer_smoke(self) -> None:
        field = build_sample_field()
        path = build_sample_path()
        viewer = render_scene(field, path)
        viewer.set_free_roam(True)
        self.assertTrue(viewer.free_roam.enabled)

        frame = viewer.snapshot()
        self.assertGreater(len(frame.rgb), 0)
        self.assertGreater(len(frame.rgb[0]), 0)


if __name__ == "__main__":
    unittest.main()
