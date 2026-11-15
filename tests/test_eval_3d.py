import importlib.util
import tempfile
import unittest
from pathlib import Path

import eval_3d


class TestAssembleGt(unittest.TestCase):
    def test_assemble_gt_handles_label_free_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "000001.txt"
            empty.write_text("")
            dontcare = Path(tmp) / "000002.txt"
            dontcare.write_text(
                "DontCare -1 -1 -10 1 1 50 50 -1 -1 -1 -1000 -1000 -1000 -10\n")
            self.assertEqual(eval_3d.assemble_gt(empty, 1242, 375), [])
            self.assertEqual(eval_3d.assemble_gt(dontcare, 1242, 375), [])

    def test_assemble_gt_sets_difficulty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000003.txt"
            path.write_text(
                "Car 0.00 0 -1.5 100 100 200 160 1.5 1.6 3.9 1.0 1.7 20.0 0.1\n")
            gts = eval_3d.assemble_gt(path, 1242, 375)
            self.assertEqual(len(gts), 1)
            self.assertEqual(gts[0]["difficulty"], 0)
            self.assertEqual(gts[0]["class"], "Car")
            self.assertEqual(gts[0]["box"], (1.5, 1.6, 3.9, 1.0, 1.7, 20.0, 0.1))


@unittest.skipUnless(importlib.util.find_spec("rfdetr"), "rfdetr not installed")
class TestPipeline(unittest.TestCase):
    def test_module_exposes_main(self):
        self.assertTrue(callable(eval_3d.main))
