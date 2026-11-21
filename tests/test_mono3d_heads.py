import math
import unittest

import torch

from mono3d_heads import Mono3DHeads, decode_boxes, depth_ensemble
from mono3d_targets import multibin_decode, ry_from_alpha

P2 = [700.0, 0, 620.0, 0, 0, 700.0, 180.0, 0, 0, 0, 1.0, 0]
IMG_H, IMG_W = 360, 1240


class HeadsTest(unittest.TestCase):
    def setUp(self):
        self.mean = torch.tensor([[1.5, 1.6, 3.9], [1.7, 0.6, 0.8], [1.7, 0.6, 1.8]])
        self.heads = Mono3DHeads(3, bins=12, mean_dims=self.mean)

    def test_forward_shapes(self):
        out = self.heads(torch.randn(1, 50, 256))
        shapes = {"logits": (1, 50, 3), "boxes2d": (1, 50, 4), "center3d": (1, 50, 2),
                  "d_reg": (1, 50), "d_sigma": (1, 50), "dims": (1, 50, 3),
                  "orient_bin": (1, 50, 12), "orient_res": (1, 50, 12)}
        for k, s in shapes.items():
            self.assertEqual(tuple(out[k].shape), s, k)
        self.assertTrue((out["d_reg"] > 0).all())
        self.assertTrue((out["d_sigma"] > 0).all())
        self.assertTrue(((out["boxes2d"] >= 0) & (out["boxes2d"] <= 1)).all())

    def test_dims_are_mean_plus_residual(self):
        for p in self.heads.dims_head.parameters():
            torch.nn.init.zeros_(p)
        out = self.heads(torch.randn(1, 5, 256))
        # zero residual -> per-class mean for every class
        self.assertTrue(torch.allclose(out["dims_all"][0, 0], self.mean))
        # dims = mean of argmax class
        cls = out["logits"].argmax(-1)
        self.assertTrue(torch.allclose(out["dims"][0], self.mean[cls[0]]))


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self.reg = torch.tensor([[20.0, 30.0]])
        self.geo = torch.tensor([[22.0, 28.0]])
        self.sig = torch.tensor([[0.5, 0.5]])

    def test_rgb_ignores_map(self):
        a, _ = depth_ensemble(self.reg, self.geo, torch.tensor([[10.0, 10.0]]), self.sig, "rgb")
        b, _ = depth_ensemble(self.reg, self.geo, torch.tensor([[50.0, 90.0]]), self.sig, "rgb")
        c, _ = depth_ensemble(self.reg, self.geo, None, self.sig, "rgb")
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, c))

    def test_fusion_uses_map(self):
        a, wa = depth_ensemble(self.reg, self.geo, torch.tensor([[10.0, 10.0]]), self.sig, "fusion")
        b, _ = depth_ensemble(self.reg, self.geo, torch.tensor([[50.0, 90.0]]), self.sig, "fusion")
        self.assertFalse(torch.allclose(a, b))
        self.assertIn("map", wa)
        total = wa["reg"] + wa["geo"] + wa["map"]
        self.assertTrue(torch.allclose(total, torch.ones_like(total)))

    def test_fusion_requires_map(self):
        with self.assertRaises(ValueError):
            depth_ensemble(self.reg, self.geo, None, self.sig, "fusion")

    def test_agreeing_sources_return_value(self):
        d = torch.tensor([[25.0]])
        out, _ = depth_ensemble(d, d, d, torch.tensor([[1.0]]), "fusion")
        self.assertTrue(torch.allclose(out, d))


class DecodeTest(unittest.TestCase):
    def _outputs(self, mode):
        x, y, z, h = 1.0, 1.5, 20.0, 1.5
        u, v = 700 * x / z + 620, 700 * y / z + 180
        box_h = 700 * h / z / IMG_H
        bin_idx, res = 5, 0.1
        logits = torch.tensor([[[0.2, 2.0, -1.0]]])
        ob = torch.full((1, 1, 12), -5.0)
        ob[0, 0, bin_idx] = 3.0
        orr = torch.zeros(1, 1, 12)
        orr[0, 0, bin_idx] = res
        out = {"logits": logits,
               "boxes2d": torch.tensor([[[0.5, 0.5, 0.1, box_h]]]),
               "center3d": torch.tensor([[[u / IMG_W, v / IMG_H]]]),
               "d_reg": torch.tensor([[z]]), "d_sigma": torch.tensor([[1.0]]),
               "dims": torch.tensor([[[h, 1.6, 3.9]]]),
               "orient_bin": ob, "orient_res": orr}
        alpha = multibin_decode(bin_idx, res)
        return out, (h, 1.6, 3.9, x, y, z, ry_from_alpha(alpha, x, z))

    def test_decode_known_box(self):
        for mode in ("fusion", "rgb"):
            out, expect = self._outputs(mode)
            dm = torch.full((1, 8, 8), 20.0)
            res = decode_boxes(out, torch.tensor([[[0.5, 0.5]]]), (IMG_H, IMG_W), dm, [P2], mode)
            self.assertEqual(len(res), 1)
            self.assertEqual(len(res[0]), 1)
            d = res[0][0]
            for a, b in zip(d["box"], expect):
                self.assertAlmostEqual(a, b, places=3)
            self.assertEqual(d["class"], 1)
            self.assertAlmostEqual(d["score"], 1 / (1 + math.exp(-2.0)), places=5)


if __name__ == "__main__":
    unittest.main()
