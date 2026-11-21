import math
import unittest

import torch

from mono3d_depth import DepthPredictor, sample_depth_at, soft_argmax_depth
from mono3d_targets import bin_value


def one_hot_logits(i, bins=80, bg=float("-inf"), h=3, w=3):
    x = torch.full((1, bins + 1, h, w), -1e4)
    x[:, i] = 0.0
    x[:, bins] = bg
    return x


class SoftArgmaxTest(unittest.TestCase):
    def test_one_hot_returns_bin_value(self):
        for bg in (float("-inf"), 1e9):
            for i in (0, 10, 79):
                d = soft_argmax_depth(one_hot_logits(i, bg=bg))
                self.assertEqual(tuple(d.shape), (1, 3, 3))
                self.assertTrue(torch.allclose(d, torch.full_like(d, bin_value(i)), atol=1e-3), (bg, i))

    def test_background_excluded(self):
        x = torch.randn(2, 81, 4, 4)
        y = x.clone()
        y[:, 80] = 1e9
        self.assertTrue(torch.allclose(soft_argmax_depth(x), soft_argmax_depth(y)))

    def test_extreme_logits_finite(self):
        x = torch.randn(1, 81, 2, 2) * 1e6
        d = soft_argmax_depth(x)
        self.assertTrue(torch.isfinite(d).all())
        self.assertTrue(((d >= 2.0 - 1e-4) & (d <= 60.0)).all())


class DepthPredictorTest(unittest.TestCase):
    def test_shapes_and_range(self):
        m = DepthPredictor()
        logits, dm, emb = m(torch.randn(1, 256, 24, 24))
        self.assertEqual(tuple(logits.shape), (1, 81, 24, 24))
        self.assertEqual(tuple(dm.shape), (1, 24, 24))
        self.assertEqual(tuple(emb.shape), (1, 256, 24, 24))
        self.assertTrue(torch.isfinite(dm).all())
        self.assertTrue(((dm >= 2.0 - 1e-4) & (dm <= 60.0)).all())


class SampleTest(unittest.TestCase):
    def test_center_of_ramp(self):
        ramp = torch.arange(9, dtype=torch.float32).view(1, 1, 9).expand(2, 9, 9).contiguous()
        ref = torch.full((2, 5, 2), 0.5)
        out = sample_depth_at(ramp, ref)
        self.assertEqual(tuple(out.shape), (2, 5))
        self.assertTrue(torch.allclose(out, torch.full_like(out, 4.0), atol=1e-4))

    def test_ramp_x_position(self):
        ramp = torch.arange(10, dtype=torch.float32).view(1, 1, 10).expand(1, 10, 10).contiguous()
        out = sample_depth_at(ramp, torch.tensor([[[0.25, 0.5]]]))
        self.assertTrue(math.isclose(out.item(), 2.0, abs_tol=1e-4))  # x=0.25*10-0.5=2.0


if __name__ == "__main__":
    unittest.main()
