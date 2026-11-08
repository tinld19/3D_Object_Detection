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


class TestAP40(unittest.TestCase):
    def test_ap40_perfect_is_one(self):
        flags = [(1 - i / 100, True) for i in range(5)]
        self.assertAlmostEqual(kitti_eval.ap40(flags, 5), 1.0)

    def test_ap40_half_recall(self):
        self.assertAlmostEqual(kitti_eval.ap40([(0.9, True)], 2), 0.5)

    def test_ap40_fp_before_tp(self):
        self.assertAlmostEqual(
            kitti_eval.ap40([(0.95, False), (0.9, True)], 1), 0.5)

    def test_ap40_no_gt_is_nan(self):
        import math
        self.assertTrue(math.isnan(kitti_eval.ap40([(0.9, True)], 0)))


class TestEvaluate(unittest.TestCase):
    def test_evaluate_shape_and_values(self):
        box = (1.5, 2.0, 4.0, 0.0, 1.5, 10.0, 0.0)
        far = (1.5, 2.0, 4.0, 30.0, 1.5, 10.0, 0.0)
        img1 = {"preds": [{"score": 0.9, "box": box, "class": "Car"}],
                "gts": [{"box": box, "class": "Car", "difficulty": 0}]}
        img2 = {"preds": [{"score": 0.8, "box": far, "class": "Car"}],
                "gts": [{"box": box, "class": "Car", "difficulty": 0}]}
        res = kitti_eval.evaluate(
            [img1, img2], ["Car", "Pedestrian"],
            {"Car": [0.7, 0.5], "Pedestrian": [0.5]}, ["3d", "bev"])
        self.assertEqual(set(res), {"Car", "Pedestrian"})
        self.assertEqual(set(res["Car"]), {"3d", "bev"})
        self.assertEqual(set(res["Car"]["3d"]), {0.7, 0.5})
        self.assertEqual(set(res["Car"]["bev"][0.7]), {"easy", "moderate", "hard"})
        # 1 of 2 GT found at precision 1 -> 20/40 targets -> 0.5
        self.assertAlmostEqual(res["Car"]["3d"][0.7]["easy"], 0.5)
        self.assertAlmostEqual(res["Car"]["bev"][0.5]["hard"], 0.5)
        import math
        self.assertTrue(math.isnan(res["Pedestrian"]["3d"][0.5]["easy"]))


if __name__ == "__main__":
    unittest.main()


class TestFormatting(unittest.TestCase):
    @staticmethod
    def _result(ap):
        cell = {"easy": ap, "moderate": float("nan"), "hard": ap / 2}
        return {"Car": {"3d": {0.7: dict(cell)}, "bev": {0.7: dict(cell)}}}

    def test_format_table_renders_both_blocks_and_nan(self):
        text = kitti_eval.format_table(self._result(80.0), "fusion")
        self.assertIn("3D AP", text)
        self.assertIn("BEV AP", text)
        self.assertIn("Car", text)
        self.assertIn("nan", text.lower())

    def test_format_comparison_builds_markdown(self):
        text = kitti_eval.format_comparison(
            {"rgb": self._result(50.0), "fusion": self._result(60.0)})
        lines = text.splitlines()
        self.assertTrue(any(l.startswith("| rgb") for l in lines))
        self.assertTrue(any(l.startswith("| fusion") for l in lines))
        self.assertIn("---", text)
