import unittest

import kitti_eval


class TestIoU(unittest.TestCase):
    def test_bev_iou_identical_is_one(self):
        b = (1.0, 5.0, 4.0, 2.0, 0.3)
        self.assertAlmostEqual(kitti_eval.bev_iou(b, b), 1.0)

    def test_bev_iou_disjoint_is_zero(self):
        self.assertEqual(kitti_eval.bev_iou((0, 0, 4, 2, 0), (50, 50, 4, 2, 0)), 0.0)

    def test_bev_iou_half_overlap(self):
        self.assertAlmostEqual(
            kitti_eval.bev_iou((0, 0, 4, 2, 0), (2, 0, 4, 2, 0)), 4 / 12, places=4)

    def test_bev_iou_degenerate_returns_zero(self):
        self.assertEqual(kitti_eval.bev_iou((0, 0, 0, 2, 0), (0, 0, 4, 2, 0)), 0.0)

    def test_iou_3d_identical_is_one(self):
        b = (2.0, 2.0, 4.0, 0.0, 1.0, 10.0, 0.2)
        self.assertAlmostEqual(kitti_eval.iou_3d(b, b), 1.0)

    def test_iou_3d_half_height_overlap(self):
        a = (2.0, 2.0, 4.0, 0.0, 2.0, 10.0, 0.0)
        b = (2.0, 2.0, 4.0, 0.0, 3.0, 10.0, 0.0)
        self.assertAlmostEqual(kitti_eval.iou_3d(a, b), 1 / 3, places=4)


class TestObjectDifficulty(unittest.TestCase):
    def test_easy_boundary(self):
        self.assertEqual(kitti_eval.object_difficulty(41, 0, 0.10), 0)

    def test_easy_exact_thresholds(self):
        self.assertEqual(kitti_eval.object_difficulty(40, 0, 0.15), 0)

    def test_moderate(self):
        self.assertEqual(kitti_eval.object_difficulty(30, 1, 0.20), 1)

    def test_hard(self):
        self.assertEqual(kitti_eval.object_difficulty(26, 2, 0.40), 2)

    def test_below_hard_height_is_ignored(self):
        self.assertIsNone(kitti_eval.object_difficulty(24, 0, 0.0))

    def test_occlusion_above_hard_is_ignored(self):
        self.assertIsNone(kitti_eval.object_difficulty(50, 3, 0.0))

    def test_truncation_above_hard_is_ignored(self):
        self.assertIsNone(kitti_eval.object_difficulty(50, 0, 0.60))


if __name__ == "__main__":
    unittest.main()
