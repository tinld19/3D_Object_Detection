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


class TestMatchImage(unittest.TestCase):
    HIT = (0.0, 10.0, 4.0, 2.0, 0.0)
    FAR = (50.0, 50.0, 4.0, 2.0, 0.0)

    def _match(self, preds, gts):
        return kitti_eval.match_image(
            preds, gts, kitti_eval.bev_iou, 0.7, 0, "Car", kitti_eval.NEIGHBOURS)

    def test_match_counts_tp_and_fp(self):
        gts = [{"box": self.HIT, "class": "Car", "difficulty": 0}]
        preds = [{"score": 0.9, "box": self.HIT}, {"score": 0.4, "box": self.FAR}]
        flags, n = self._match(preds, gts)
        self.assertEqual(flags, [(0.9, True), (0.4, False)])
        self.assertEqual(n, 1)

    def test_match_on_neighbour_is_dontcare(self):
        gts = [{"box": self.HIT, "class": "Van", "difficulty": 0}]
        flags, n = self._match([{"score": 0.9, "box": self.HIT}], gts)
        self.assertEqual(flags, [])
        self.assertEqual(n, 0)

    def test_match_on_other_class_is_fp(self):
        gts = [{"box": self.HIT, "class": "Truck", "difficulty": 0}]
        flags, n = self._match([{"score": 0.9, "box": self.HIT}], gts)
        self.assertEqual(flags, [(0.9, False)])
        self.assertEqual(n, 0)

    def test_match_ignores_too_hard_gt(self):
        gts = [{"box": self.HIT, "class": "Car", "difficulty": None}]
        flags, n = self._match([{"score": 0.9, "box": self.HIT}], gts)
        self.assertEqual(flags, [])
        self.assertEqual(n, 0)

    NEAR = (0.2, 10.0, 4.0, 2.0, 0.0)  # IoU ~0.82 with HIT

    def test_valid_gt_preferred_over_higher_iou_neighbour(self):
        gts = [{"box": self.NEAR, "class": "Car", "difficulty": 0},
               {"box": self.HIT, "class": "Van", "difficulty": 0}]
        flags, n = self._match([{"score": 0.9, "box": self.HIT}], gts)
        self.assertEqual(flags, [(0.9, True)])
        self.assertEqual(n, 1)

    def test_neighbour_gt_not_consumed(self):
        gts = [{"box": self.HIT, "class": "Van", "difficulty": 0}]
        preds = [{"score": 0.9, "box": self.HIT}, {"score": 0.8, "box": self.HIT}]
        flags, n = self._match(preds, gts)
        self.assertEqual(flags, [])
        self.assertEqual(n, 0)

    def test_valid_gt_preferred_over_higher_iou_ignored_car(self):
        gts = [{"box": self.NEAR, "class": "Car", "difficulty": 0},
               {"box": self.HIT, "class": "Car", "difficulty": None}]
        flags, n = self._match([{"score": 0.9, "box": self.HIT}], gts)
        self.assertEqual(flags, [(0.9, True)])
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
