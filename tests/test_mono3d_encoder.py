import unittest

import numpy as np
import torch

import mono3d_encoder


class TrainableEncoderTest(unittest.TestCase):
    def test_shape_and_all_params_get_grad(self):
        enc = mono3d_encoder.TrainableEncoder()
        out = enc(torch.rand(2, 256, 24, 24))
        self.assertEqual((2, 256, 24, 24), tuple(out.shape))
        out.sum().backward()
        params = list(enc.named_parameters())
        self.assertTrue(params)
        for name, p in params:
            self.assertTrue(p.requires_grad, name)
            self.assertIsNotNone(p.grad, name)

    def test_arbitrary_hw(self):
        enc = mono3d_encoder.TrainableEncoder(layers=1)
        self.assertEqual((1, 256, 5, 9), tuple(enc(torch.rand(1, 256, 5, 9)).shape))

    def test_rejects_wrong_channels(self):
        enc = mono3d_encoder.TrainableEncoder()
        with self.assertRaisesRegex(ValueError, "256"):
            enc(torch.rand(1, 128, 6, 6))


class ProposalSourceTest(unittest.TestCase):
    def test_converts_xyxy_pixels_to_normalized_cxcywh(self):
        def predict(image):
            return {
                "xyxy": np.array([[10.0, 20.0, 50.0, 60.0]]),
                "confidence": np.array([0.9]),
                "class_id": np.array([2]),
            }

        src = mono3d_encoder.RFDETRProposalSource("ckpt", predict_fn=predict)
        out = src(torch.rand(1, 3, 100, 200))
        self.assertEqual(1, len(out))
        torch.testing.assert_close(
            out[0]["boxes2d"], torch.tensor([[0.15, 0.4, 0.2, 0.4]])
        )
        torch.testing.assert_close(out[0]["scores"], torch.tensor([0.9]))
        self.assertEqual([2], out[0]["classes"].tolist())
        self.assertFalse(out[0]["boxes2d"].requires_grad)
        for p in src.parameters():
            self.assertFalse(p.requires_grad)

    def test_empty_detections(self):
        src = mono3d_encoder.RFDETRProposalSource(
            "ckpt",
            predict_fn=lambda im: {
                "xyxy": np.zeros((0, 4)),
                "confidence": np.zeros(0),
                "class_id": np.zeros(0, dtype=int),
            },
        )
        out = src(torch.rand(2, 3, 64, 64))
        self.assertEqual((0, 4), tuple(out[1]["boxes2d"].shape))


if __name__ == "__main__":
    unittest.main()
