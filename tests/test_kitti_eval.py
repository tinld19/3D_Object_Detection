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


if __name__ == "__main__":
    unittest.main()
