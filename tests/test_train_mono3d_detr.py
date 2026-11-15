import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

import train_mono3d_detr as T
from mono3d_data import Mono3DDetrDataset, collate_mono3d
from mono3d_loss import SetCriterion
from mono3d_model import MonoDetr3D
from tests.test_mono3d_data import CAR, PED, build

FS = (12, 12)


def stub_feature_fn(images):
    return torch.randn(images.shape[0], 256, *FS)


def stub_proposal_fn(images):
    box = torch.tensor([[0.4, 0.5, 0.3, 0.4], [0.7, 0.5, 0.2, 0.4]])
    return [{"boxes2d": box, "scores": torch.tensor([0.9, 0.8]), "classes": torch.tensor([0, 1])}
            for _ in range(images.shape[0])]


def make_root(root):
    build(root, {"000000": [CAR, PED], "000001": [CAR], "000002": [PED]})
    (root / "val.txt").write_text("000002\n")
    return root


class ArgsTest(unittest.TestCase):
    def test_defaults(self):
        a = T.parse_args(["--rfdetr-checkpoint", "c.pth"])
        self.assertEqual(a.split_dir, Path("kitti_imagesets"))
        self.assertEqual(a.batch_size, 8)
        self.assertEqual(a.grad_accum, 1)
        self.assertEqual(a.feature_mode, "fusion")
        self.assertEqual(a.output, Path("runs/mono3d-detr/fusion.pt"))

    def test_output_per_mode(self):
        a = T.parse_args(["--rfdetr-checkpoint", "c.pth", "--feature-mode", "rgb"])
        self.assertEqual(a.output, Path("runs/mono3d-detr/rgb.pt"))

    def test_checkpoint_required(self):
        with self.assertRaises(SystemExit):
            T.parse_args([])


class SaveTest(unittest.TestCase):
    def test_save_checkpoint(self):
        model = MonoDetr3D(2, torch.ones(2, 3), feature_mode="rgb", num_queries=5, decoder_layers=1)
        with TemporaryDirectory() as d:
            path = Path(d) / "sub" / "m.pt"
            T.save_checkpoint(path, model, {"classes": ["Car", "Pedestrian"], "class_ids": {"Car": 0, "Pedestrian": 1},
                                            "mean_dims": {"Car": [1.5, 1.6, 3.9]}, "feature_mode": "rgb"})
            ck = torch.load(path, weights_only=False)
        self.assertEqual(ck["arch"], "b")
        self.assertEqual(ck["classes"], ["Car", "Pedestrian"])
        self.assertIn("Car", ck["mean_dims"])
        self.assertIn("state_dict", ck)


class TrainTest(unittest.TestCase):
    def test_one_step_smoke(self):
        with TemporaryDirectory() as d:
            root = make_root(Path(d))
            ds = Mono3DDetrDataset(root / "image", root / "label", root / "calib", FS, image_size=48)
            loader = DataLoader(ds, batch_size=2, collate_fn=collate_mono3d)
            torch.manual_seed(0)
            model = MonoDetr3D(len(ds.classes), torch.tensor([ds.mean_dims[c] for c in ds.classes]),
                               num_queries=5, decoder_layers=1)
            before = [p.detach().clone() for p in model.parameters()]
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            loss = T.train_epoch(model, SetCriterion(len(ds.classes)), loader, opt,
                                 stub_feature_fn, stub_proposal_fn, "cpu")
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(any(not torch.equal(a, p) for a, p in zip(before, model.parameters())))

    def test_run_training_uses_dataset_mean_dims(self):
        with TemporaryDirectory() as d:
            root = make_root(Path(d))
            out = root / "out" / "m.pt"
            args = T.parse_args(["--rfdetr-checkpoint", "c", "--image-dir", str(root / "image"),
                                 "--label-dir", str(root / "label"), "--calib-dir", str(root / "calib"),
                                 "--split-dir", str(root), "--epochs", "1", "--batch-size", "2",
                                 "--output", str(out), "--device", "cpu"])
            args.image_size, args.num_queries, args.decoder_layers = 48, 5, 1
            (root / "val.txt").write_text("000002\n")
            T.run_training(args, stub_feature_fn, stub_proposal_fn)
            ck = torch.load(out, weights_only=False)
            ds = Mono3DDetrDataset(root / "image", root / "label", root / "calib", FS, image_size=48)
        self.assertEqual(ck["arch"], "b")
        self.assertEqual(ck["mean_dims"], ds.mean_dims)
        self.assertEqual(ck["classes"], ds.classes)
        self.assertEqual(tuple(ck["feature_size"]), FS)


if __name__ == "__main__":
    unittest.main()
