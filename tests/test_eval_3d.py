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


class TestArchB(unittest.TestCase):
    def _setup(self, tmp):
        import argparse
        import torch
        from PIL import Image
        from mono3d_model import MonoDetr3D

        tmp = Path(tmp)
        for d in ("img", "lab", "cal", "split"):
            (tmp / d).mkdir()
        Image.new("RGB", (64, 32)).save(tmp / "img" / "000001.png")
        (tmp / "lab" / "000001.txt").write_text(
            "Car 0.00 0 -1.5 10 10 40 30 1.5 1.6 3.9 1.0 1.7 20.0 0.1\n")
        (tmp / "cal" / "000001.txt").write_text(
            "P2: 700 0 32 0 0 700 16 0 0 0 1 0\n")
        (tmp / "split" / "val.txt").write_text("000001\n")
        mean_dims = [[1.5, 1.6, 3.9], [1.7, 0.6, 0.8]]
        model = MonoDetr3D(2, mean_dims, feature_mode="fusion", num_queries=4, decoder_layers=1)
        ckpt = tmp / "b.pt"
        torch.save({"arch": "b", "classes": ["Car", "Pedestrian"], "mean_dims": mean_dims,
                    "feature_mode": "fusion", "image_size": 32, "num_queries": 4,
                    "decoder_layers": 1, "state_dict": model.state_dict()}, ckpt)
        return argparse.Namespace(
            three_d_checkpoint=ckpt, rfdetr_checkpoint=tmp / "rf.pth", device="cpu",
            split_dir=tmp / "split", score_thr=0.05, image_dir=tmp / "img",
            label_dir=tmp / "lab", calib_dir=tmp / "cal", limit=None,
            output=tmp / "out" / "eval.json")

    def test_arch_b_dispatch_maps_int_class_to_name_and_writes_arch(self):
        import json
        import torch
        from mono3d_model import MonoDetr3D

        seen = {}

        def fake_predict(self, f0, proposals, p2, image_size, score_thr=0.05, mode=None):
            seen.update(f0=f0, proposals=proposals, p2=p2, image_size=image_size,
                        score_thr=score_thr, mode=mode)
            return [[{"score": 0.9, "class": 0, "box": (1.5, 1.6, 3.9, 1.0, 1.7, 20.0, 0.1)}]]

        feature_fn = lambda x: torch.zeros(1, 256, 4, 4)
        proposal_fn = lambda x: [{"boxes2d": torch.zeros(0, 4)}]
        original = MonoDetr3D.predict
        original_make = eval_3d._make_b_fns
        MonoDetr3D.predict = fake_predict
        eval_3d._make_b_fns = lambda args, dev: (feature_fn, proposal_fn)
        captured = {}
        original_eval = eval_3d.kitti_eval.evaluate
        eval_3d.kitti_eval.evaluate = lambda images, *a, **k: (
            captured.setdefault("images", images), original_eval(images, *a, **k))[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                args = self._setup(tmp)
                eval_3d.evaluate_checkpoint(args)
                data = json.loads(args.output.read_text())
        finally:
            MonoDetr3D.predict = original
            eval_3d._make_b_fns = original_make
            eval_3d.kitti_eval.evaluate = original_eval
        preds = captured["images"][0]["preds"]
        self.assertEqual(preds[0]["class"], "Car")
        self.assertIsInstance(preds[0]["class"], str)
        self.assertEqual(captured["images"][0]["gts"][0]["class"], "Car")
        self.assertEqual(data["arch"], "b")
        self.assertEqual(data["feature_mode"], "fusion")
        self.assertEqual(seen["image_size"], (32, 32))
        self.assertEqual(seen["mode"], "fusion")
        self.assertEqual(len(seen["p2"][0]), 12)
        self.assertAlmostEqual(seen["p2"][0][0], 350.0)  # 700 * 32/64

    def test_arch_b_trainer_checkpoint_with_dict_mean_dims_end_to_end(self):
        import json
        import torch
        import train_mono3d_detr
        from mono3d_model import MonoDetr3D

        def fake_predict(self, f0, proposals, p2, image_size, score_thr=0.05, mode=None):
            return [[{"score": 0.9, "class": 0, "box": (1.5, 1.6, 3.9, 1.0, 1.7, 20.0, 0.1)}]]

        original = MonoDetr3D.predict
        original_make = eval_3d._make_b_fns
        MonoDetr3D.predict = fake_predict
        eval_3d._make_b_fns = lambda args, dev: (
            lambda x: torch.zeros(1, 256, 4, 4), lambda x: [{"boxes2d": torch.zeros(0, 4)}])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                args = self._setup(tmp)
                md = {"Car": [1.5, 1.6, 3.9], "Pedestrian": [1.7, 0.6, 0.8]}
                model = MonoDetr3D(2, [md["Car"], md["Pedestrian"]], feature_mode="fusion",
                                   num_queries=4, decoder_layers=1)
                train_mono3d_detr.save_checkpoint(args.three_d_checkpoint, model, {
                    "classes": ["Car", "Pedestrian"], "mean_dims": md, "feature_mode": "fusion",
                    "image_size": 32, "num_queries": 4, "decoder_layers": 1})
                eval_3d.evaluate_checkpoint(args)
                data = json.loads(args.output.read_text())
        finally:
            MonoDetr3D.predict = original
            eval_3d._make_b_fns = original_make
        self.assertEqual(data["arch"], "b")

    def test_compare_keys_by_arch_and_mode(self):
        import json
        captured = {}
        original = eval_3d.kitti_eval.format_comparison
        eval_3d.kitti_eval.format_comparison = lambda r: captured.setdefault("r", r) and ""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = {"Car": {"3d": {"0.7": {"easy": 1.0}}}}
                a = Path(tmp) / "a.json"
                b = Path(tmp) / "b.json"
                a.write_text(json.dumps({"feature_mode": "fusion", "result": result}))
                b.write_text(json.dumps({"arch": "b", "feature_mode": "fusion", "result": result}))
                eval_3d.compare([a, b])
        finally:
            eval_3d.kitti_eval.format_comparison = original
        self.assertEqual(sorted(captured["r"]), ["a/fusion", "b/fusion"])
