import math
import unittest

import torch

from mono3d_loss import SetCriterion, hungarian_match

C, BINS, N, FH, FW = 3, 12, 8, 4, 6
WEIGHTS = {"class": 2.0, "box": 5.0, "giou": 2.0, "center": 10.0, "depth": 1.0}


def make_targets(gs, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    targets = []
    for g in gs:
        targets.append({
            "labels": torch.randint(0, C, (g,), generator=g_),
            "boxes2d": torch.rand(g, 4, generator=g_) * 0.3 + torch.tensor([0.3, 0.3, 0.1, 0.1]),
            "center3d": torch.rand(g, 2, generator=g_) * 0.5 + 0.2,
            "depth": torch.rand(g, generator=g_) * 40 + 5,
            "dims": torch.rand(g, 3, generator=g_) + 1.0,
            "alpha": torch.rand(g, generator=g_),
            "orient_bin": torch.randint(0, BINS, (g,), generator=g_),
            "orient_res": torch.rand(g, BINS, generator=g_) * 0.2,
        })
    return targets


def make_layer(targets, perfect=True, sigma=1.0, depth_err=0.0, seed=1):
    g_ = torch.Generator().manual_seed(seed)
    B = len(targets)
    out = {
        "logits": torch.full((B, N, C), -10.0),
        "boxes2d": torch.rand(B, N, 4, generator=g_) * 0.3 + torch.tensor([0.6, 0.6, 0.1, 0.1]),
        "center3d": torch.rand(B, N, 2, generator=g_),
        "d_reg": torch.rand(B, N, generator=g_) * 40 + 5,
        "d_sigma": torch.full((B, N), float(sigma)),
        "dims_all": torch.rand(B, N, C, 3, generator=g_) + 1.0,
        "orient_bin": torch.zeros(B, N, BINS),
        "orient_res": torch.zeros(B, N, BINS),
        "ref_points": torch.rand(B, N, 2, generator=g_),
        "valid_mask": torch.ones(B, N, dtype=torch.bool),
    }
    out["depth"] = out["d_reg"].clone()
    out["dims"] = out["dims_all"][:, :, 0]
    for b, t in enumerate(targets):
        for i in range(len(t["labels"])):
            c = int(t["labels"][i])
            out["logits"][b, i, c] = 10.0
            out["boxes2d"][b, i] = t["boxes2d"][i]
            out["center3d"][b, i] = t["center3d"][i]
            out["depth"][b, i] = t["depth"][i] + depth_err
            out["dims_all"][b, i, c] = t["dims"][i]
            out["orient_bin"][b, i, int(t["orient_bin"][i])] = 10.0
            out["orient_res"][b, i] = t["orient_res"][i]
    return out


def add_depthmap(outputs, B, requires_grad=True):
    outputs["depth_logits"] = torch.randn(B, 81, FH, FW, requires_grad=requires_grad)
    outputs["depth_map"] = torch.rand(B, FH, FW) * 40 + 2
    return outputs


def add_dm_targets(targets):
    B = len(targets)
    tgt_map = torch.randint(0, 81, (B, FH, FW))
    mask = torch.rand(B, FH, FW) > 0.3
    for b, t in enumerate(targets):
        t["depth_map_target"] = tgt_map[b]
        t["depth_map_mask"] = mask[b]
    return targets


def wrap(layers, with_depth=True, B=1):
    out = {"aux": layers, "depth_logits": None, "depth_map": None}
    if with_depth:
        add_depthmap(out, B)
    return out


def with_dm(targets):
    add_dm_targets(targets)
    return targets


class MatchTest(unittest.TestCase):
    def test_matches_near_exact_queries(self):
        t = make_targets([2])
        layer = make_layer(t)
        # make the exact queries sit at indices 5 and 2
        for k in ("logits", "boxes2d", "center3d", "depth", "d_sigma"):
            tmp = layer[k][0, 0].clone()
            layer[k][0, 0] = layer[k][0, 5]
            layer[k][0, 5] = tmp
            tmp = layer[k][0, 1].clone()
            layer[k][0, 1] = layer[k][0, 2]
            layer[k][0, 2] = tmp
        (pi, gi), = hungarian_match(layer, t, WEIGHTS)
        pairs = {int(p): int(g) for p, g in zip(pi, gi)}
        self.assertEqual(pairs, {5: 0, 2: 1})

    def test_empty_gt_returns_empty(self):
        t = make_targets([0, 2])
        layer = make_layer(t)
        res = hungarian_match(layer, t, WEIGHTS)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0].numel(), 0)
        self.assertEqual(res[0][1].numel(), 0)
        self.assertEqual(res[1][0].numel(), 2)


class CriterionTest(unittest.TestCase):
    def test_empty_targets_finite_and_backward(self):
        t = with_dm(make_targets([0, 0]))
        layer = make_layer(t)
        layer["logits"].requires_grad_(True)
        out = wrap([layer], B=2)
        losses = SetCriterion(C)(out, t)
        self.assertTrue(torch.isfinite(losses["total"]))
        for k in ("loss_box", "loss_giou", "loss_center", "loss_depth", "loss_dim", "loss_orient"):
            self.assertEqual(float(losses[k]), 0.0, k)
        self.assertGreater(float(losses["loss_class"]), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(layer["logits"].grad)
        self.assertTrue(torch.isfinite(layer["logits"].grad).all())

    def test_empty_targets_rgb_backward(self):
        t = make_targets([0])
        layer = make_layer(t)
        layer["logits"].requires_grad_(True)
        losses = SetCriterion(C)(wrap([layer], with_depth=False), t)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_perfect_predictions_near_zero(self):
        t = with_dm(make_targets([3, 2]))
        out = wrap([make_layer(t)], B=2)
        losses = SetCriterion(C)(out, t)
        self.assertLess(float(losses["loss_box"]), 1e-4)
        self.assertLess(float(losses["loss_giou"]), 1e-4)
        self.assertLess(float(losses["loss_center"]), 1e-4)
        self.assertLess(float(losses["loss_dim"]), 1e-4)
        # sigma=1, error 0 -> NLL = log(1) = 0
        self.assertLess(abs(float(losses["loss_depth"])), 1e-4)
        self.assertLess(float(losses["loss_orient"]), 1e-3)
        self.assertLess(float(losses["loss_class"]), 1e-2)

    def test_loss_depth_laplacian_uncertainty(self):
        t = make_targets([2])
        err = 2.0
        vals = {}
        for s in (0.1, 1.0, 2.0, 8.0):
            layer = make_layer(t, sigma=s, depth_err=err)
            out = wrap([layer], with_depth=False)
            vals[s] = float(SetCriterion(C)(out, t)["loss_depth"])
        # minimum at sigma = sqrt(2)*err ~ 2.83: sigma 2.0 beats both tiny and huge sigma
        self.assertLess(vals[2.0], vals[0.1])
        self.assertLess(vals[2.0], vals[8.0])
        self.assertLess(vals[1.0], vals[0.1])
        self.assertLess(vals[2.0], vals[1.0])

    def test_aux_losses_sum_across_layers(self):
        t = make_targets([2])
        layers = [make_layer(t, perfect=False, seed=s, depth_err=1.0) for s in (1, 2, 3)]
        for l in layers:  # unmatched noise so box loss nonzero
            l["boxes2d"][0, :2] = l["boxes2d"][0, :2] + 0.05
        crit = SetCriterion(C)
        one = crit(wrap(layers[:1], with_depth=False), t)
        three = crit(wrap(layers, with_depth=False), t)
        self.assertGreater(float(three["loss_box"]), float(one["loss_box"]))
        self.assertGreater(float(three["loss_depth"]), float(one["loss_depth"]) + 1e-6)
        self.assertGreater(float(three["total"]), float(one["total"]))
        self.assertAlmostEqual(float(three["loss_class"]) > float(one["loss_class"]), True)

    def test_rgb_mode_depthmap_zero(self):
        t = make_targets([2])
        losses = SetCriterion(C)(wrap([make_layer(t)], with_depth=False), t)
        self.assertEqual(float(losses["loss_depthmap"]), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_missing_depthmap_targets_raise_when_depth_active(self):
        t = make_targets([2])
        with self.assertRaises(ValueError):
            SetCriterion(C)(wrap([make_layer(t)], B=1), t)

    def test_rgb_mode_without_depthmap_targets_ok(self):
        t = make_targets([2])
        losses = SetCriterion(C)(wrap([make_layer(t)], with_depth=False), t)
        self.assertEqual(float(losses["loss_depthmap"]), 0.0)

    def test_depthmap_explicit_kwargs(self):
        t = make_targets([2])
        tgt = torch.randint(0, 81, (1, FH, FW))
        mask = torch.ones(1, FH, FW, dtype=torch.bool)
        losses = SetCriterion(C)(wrap([make_layer(t)], B=1), t, tgt, mask)
        self.assertGreater(float(losses["loss_depthmap"]), 0.0)

    def test_depthmap_active_and_keys(self):
        t = with_dm(make_targets([2]))
        losses = SetCriterion(C)(wrap([make_layer(t)] * 3, B=1), t)
        self.assertGreater(float(losses["loss_depthmap"]), 0.0)
        for k in ("loss_class", "loss_box", "loss_giou", "loss_center", "loss_depth",
                  "loss_dim", "loss_orient", "loss_depthmap", "total"):
            self.assertIn(k, losses)
            self.assertTrue(torch.isfinite(torch.as_tensor(losses[k])))
        losses["total"].backward()


if __name__ == "__main__":
    unittest.main()
