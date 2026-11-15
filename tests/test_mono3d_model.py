import unittest

import torch

from mono3d_model import MonoDetr3D

P2 = [700.0, 0, 192.0, 0, 0, 700.0, 192.0, 0, 0, 0, 1.0, 0]
SIZE = (384, 384)
MEAN = torch.tensor([[1.5, 1.6, 3.9], [1.7, 0.6, 0.8], [1.7, 0.6, 1.8]])
KEYS = {"logits", "boxes2d", "center3d", "d_reg", "d_sigma", "dims", "depth",
        "ref_points", "valid_mask"}


def make(mode="fusion"):
    torch.manual_seed(0)
    return MonoDetr3D(3, MEAN, feature_mode=mode)


def inputs():
    f0 = torch.randn(1, 256, 24, 24)
    xy = torch.tensor([[50., 80., 150., 200.], [200., 100., 300., 220.]])
    props = [{"boxes2d": xy, "scores": torch.tensor([0.9, 0.8]), "classes": torch.tensor([0, 1])}]
    return f0, props


class ModelTest(unittest.TestCase):
    def test_forward_fusion(self):
        f0, props = inputs()
        out = make().forward(f0, props, P2, SIZE)
        self.assertEqual(len(out["aux"]), 3)
        self.assertEqual(tuple(out["depth_map"].shape), (1, 24, 24))
        for a in out["aux"]:
            self.assertTrue(KEYS <= set(a))
            self.assertEqual(tuple(a["depth"].shape), (1, 50))

    def test_all_modes(self):
        f0, props = inputs()
        for mode in ("rgb", "depth", "fusion"):
            out = make(mode).forward(f0, props, P2, SIZE)
            self.assertEqual(len(out["aux"]), 3, mode)
            if mode == "rgb":
                self.assertIsNone(out["depth_map"])
                self.assertIsNone(out["depth_logits"])
            else:
                self.assertIsNotNone(out["depth_logits"])

    def test_rgb_has_no_depth_predictor(self):
        self.assertIsNone(make("rgb").depth_predictor)
        self.assertIsNotNone(make("fusion").depth_predictor)

    def test_predict(self):
        f0, props = inputs()
        for mode in ("rgb", "fusion"):
            m = make(mode).eval()
            res = m.predict(f0, props, P2, SIZE, score_thr=0.0)
            self.assertEqual(len(res), 1)
            self.assertLessEqual(len(res[0]), 50)
            self.assertGreater(len(res[0]), 0)
            thr = res[0][len(res[0]) // 2]["score"]
            res2 = m.predict(f0, props, P2, SIZE, score_thr=thr)
            self.assertTrue(all(d["score"] >= thr and {"score", "class", "box"} <= set(d)
                                for d in res2[0]))
            self.assertLessEqual(len(res2[0]), len(res[0]))

    def test_backward_trainable(self):
        f0, props = inputs()
        f0.requires_grad_(False)
        m = make()
        out = m.forward(f0, props, P2, SIZE)
        loss = sum(a[k].sum() for a in out["aux"] for k in ("logits", "boxes2d", "center3d", "depth", "dims"))
        loss = loss + out["depth_logits"].sum()
        loss.backward()
        for name, mod in (("encoder", m.encoder), ("decoder", m.decoder), ("heads", m.heads),
                          ("depth", m.depth_predictor)):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in mod.parameters()), name)
        self.assertFalse(any("rfdetr" in n.lower() for n, _ in m.named_modules()))


if __name__ == "__main__":
    unittest.main()
