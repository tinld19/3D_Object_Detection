import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class CuboidProjectionTest(unittest.TestCase):
    def test_project_cuboid_returns_eight_finite_points_for_kitti_bottom_centre_box(self):
        from infer_3d import project_cuboid

        points = project_cuboid(
            [2.0, 2.0, 4.0],
            [0.0, 2.0, 10.0],
            0.25,
            [100.0, 0.0, 50.0, 0.0, 0.0, 100.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

        self.assertEqual(8, len(points))
        self.assertTrue(all(math.isfinite(value) for point in points for value in point))

    def test_checkpoint_match_detects_a_different_rfdetr_artifact(self):
        from infer_3d import checkpoint_matches

        with TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.pth", Path(directory) / "second.pth"
            first.touch()
            second.touch()

            self.assertTrue(checkpoint_matches(str(first), first))
            self.assertFalse(checkpoint_matches(str(first), second))


if __name__ == "__main__":
    unittest.main()
