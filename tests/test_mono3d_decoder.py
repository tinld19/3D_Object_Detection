import unittest

import torch

from mono3d_decoder import DepthAwareDecoder, seed_queries


def props(n):
    boxes = torch.rand(n, 4) * 0.3 + 0.2
    return {"boxes2d": boxes, "scores": torch.rand(n), "classes": torch.zeros(n, dtype=torch.long)}


class SeedQueriesTests(unittest.TestCase):
    def test_three_proposals_padded_to_n(self):
        f = torch.randn(2, 256, 8, 8)
        content, ref, valid = seed_queries([props(3), props(3)], f, num_queries=50)
        self.assertEqual(content.shape, (2, 50, 256))
        self.assertEqual(ref.shape, (2, 50, 2))
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(valid.sum(1).tolist(), [3, 3])

    def test_zero_proposals_all_padded(self):
        f = torch.randn(1, 256, 8, 8)
        empty = {"boxes2d": torch.zeros(0, 4), "scores": torch.zeros(0), "classes": torch.zeros(0, dtype=torch.long)}
        content, ref, valid = seed_queries([empty], f, num_queries=50)
        self.assertEqual(content.shape, (1, 50, 256))
        self.assertFalse(valid.any())

    def test_more_than_n_truncates_by_score(self):
        f = torch.randn(1, 256, 8, 8)
        p = props(70)
        content, ref, valid = seed_queries([p], f, num_queries=50)
        self.assertTrue(valid.all())
        top = p["scores"].topk(50).indices
        self.assertTrue(torch.allclose(ref[0], p["boxes2d"][top][:, :2]))

    def test_jitter_runs(self):
        f = torch.randn(1, 256, 8, 8)
        content, ref, valid = seed_queries([props(10)], f, num_queries=50, jitter=True)
        self.assertEqual(content.shape, (1, 50, 256))
        self.assertLessEqual(valid.sum().item(), 10)
        self.assertTrue(((ref >= 0) & (ref <= 1)).all())


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.f = torch.randn(1, 256, 6, 6)
        self.d = torch.randn(1, 256, 6, 6)
        self.content = torch.randn(1, 50, 256)
        self.ref = torch.rand(1, 50, 2)

    def test_fusion_forward_backward(self):
        dec = DepthAwareDecoder(feature_mode="fusion")
        hs, refs = dec(self.content, self.ref, self.f, self.d)
        self.assertEqual(len(hs), 3)
        self.assertEqual(len(refs), 3)
        self.assertEqual(hs[-1].shape, (1, 50, 256))
        self.assertEqual(refs[0].shape, (1, 50, 2))
        sum(h.sum() for h in hs).backward()
        self.assertIsNotNone(dec.layers[0].depth_cross_attn.in_proj_weight.grad)
        self.assertIsNotNone(dec.layers[0].visual_cross_attn.in_proj_weight.grad)

    def test_rgb_mode_no_depth(self):
        dec = DepthAwareDecoder(feature_mode="rgb")
        hs, _ = dec(self.content, self.ref, self.f, None)
        self.assertEqual(len(hs), 3)
        self.assertFalse(dec.layers[0].use_depth)

    def test_depth_mode_ignores_f(self):
        dec = DepthAwareDecoder(feature_mode="depth").eval()
        self.assertFalse(dec.layers[0].use_visual)
        a, _ = dec(self.content, self.ref, self.f, self.d)
        b, _ = dec(self.content, self.ref, torch.randn_like(self.f), self.d)
        self.assertTrue(torch.allclose(a[-1], b[-1]))

    def test_fusion_uses_f(self):
        dec = DepthAwareDecoder(feature_mode="fusion").eval()
        a, _ = dec(self.content, self.ref, self.f, self.d)
        b, _ = dec(self.content, self.ref, torch.randn_like(self.f), self.d)
        self.assertFalse(torch.allclose(a[-1], b[-1]))


if __name__ == "__main__":
    unittest.main()
