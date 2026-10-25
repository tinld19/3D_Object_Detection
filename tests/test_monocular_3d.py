import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

import monocular_3d


class ForegroundDepthModelTest(unittest.TestCase):
    def test_model_returns_one_3d_prediction_per_object_and_depth_logits(self):
        self.assertTrue(hasattr(monocular_3d, "ForegroundDepth3DRegressor"))
        model = monocular_3d.ForegroundDepth3DRegressor(num_classes=3)
        features = torch.randn(2, 256, 8, 12)
        boxes = [torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.25, 0.25]]), torch.tensor([[0.2, 0.3, 0.4, 0.5]])]
        class_ids = [torch.tensor([0, 1]), torch.tensor([2])]
        geometry = [torch.zeros(2, 8), torch.zeros(1, 8)]

        predictions, logits = model(features, boxes, class_ids, geometry)

        self.assertEqual([(2, 8), (1, 8)], [tuple(item.shape) for item in predictions])
        self.assertEqual((2, 81, 8, 12), tuple(logits.shape))

    def test_depth_loss_backpropagates_for_two_frames_with_different_object_counts(self):
        model = monocular_3d.ForegroundDepth3DRegressor(num_classes=3)
        features = torch.randn(2, 256, 8, 12)
        boxes = [torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.25, 0.25]]), torch.tensor([[0.2, 0.3, 0.4, 0.5]])]
        class_ids = [torch.tensor([0, 1]), torch.tensor([2])]
        geometry = [torch.zeros(2, 8), torch.zeros(1, 8)]

        predictions, depth_logits = model(features, boxes, class_ids, geometry)
        depth_target = torch.zeros(2, 8, 12, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(depth_logits, depth_target, ignore_index=-100)
        loss.backward()

        self.assertIsNotNone(model.depth_predictor[0].weight.grad)


class RFDETRDinoAdapterTest(unittest.TestCase):
    def test_adapter_uses_spatial_backbone_feature(self):
        class Backbone(torch.nn.Module):
            def forward(self, nested):
                return ([SimpleNamespace(tensors=nested.tensors.new_zeros((2, 256, 6, 8)))], None)

        wrapper = SimpleNamespace(
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            model=SimpleNamespace(model=SimpleNamespace(backbone=[Backbone()])),
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "rfdetr.pth"
            checkpoint.touch()
            adapter = monocular_3d.RFDETRDinoAdapter(checkpoint, loader=lambda _path: wrapper)
            features = adapter(torch.rand(2, 3, 384, 384))

        self.assertEqual((2, 256, 6, 8), tuple(features.shape))

    def test_adapter_rejects_missing_checkpoint_and_non_spatial_features(self):
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
            monocular_3d.RFDETRDinoAdapter(Path("missing-rfdetr.pth"))

        class Backbone(torch.nn.Module):
            def forward(self, nested):
                return ([SimpleNamespace(tensors=nested.tensors.new_zeros((1, 128, 6, 8)))], None)

        wrapper = SimpleNamespace(
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            model=SimpleNamespace(model=SimpleNamespace(backbone=[Backbone()])),
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "rfdetr.pth"
            checkpoint.touch()
            adapter = monocular_3d.RFDETRDinoAdapter(checkpoint, loader=lambda _path: wrapper)
            with self.assertRaisesRegex(RuntimeError, "256-channel spatial"):
                adapter(torch.rand(1, 3, 384, 384))


if __name__ == "__main__":
    unittest.main()
