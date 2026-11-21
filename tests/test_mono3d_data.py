import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

import mono3d_targets as T
from mono3d_data import Mono3DDetrDataset, ProposalCache, collate_mono3d, horizontal_flip

W, H = 120, 60
P2 = [100.0, 0, 60.0, 0, 0, 100.0, 30.0, 0, 0, 0, 1.0, 0]
CAR = "Car 0.0 0 -1.5 20 10 60 40 1.5 1.6 3.9 2.0 1.0 20.0 0.8"
PED = "Pedestrian 0.0 0 0.0 80 15 95 50 1.7 0.6 0.8 -3.0 1.2 15.0 -0.4"


def build(root: Path, labels: dict[str, list[str]]) -> Mono3DDetrDataset:
    for name in ("image", "label", "calib"):
        (root / name).mkdir(exist_ok=True)
    for stem, lines in labels.items():
        Image.new("RGB", (W, H), (10, 20, 30)).save(root / "image" / f"{stem}.png")
        (root / "label" / f"{stem}.txt").write_text("\n".join(lines))
        (root / "calib" / f"{stem}.txt").write_text("P2: " + " ".join(map(str, P2)) + "\n")
    return root


def dataset(root, **kw):
    return Mono3DDetrDataset(root / "image", root / "label", root / "calib", feature_size=(24, 24), image_size=48, **kw)


class DatasetTest(unittest.TestCase):
    def test_sample_targets_match_helpers(self):
        with TemporaryDirectory() as d:
            ds = dataset(build(Path(d), {"000000": [CAR, PED]}))
            s = ds[0]
            t = s["targets"]
            self.assertEqual((3, 48, 48), tuple(s["image"].shape))
            self.assertEqual((12,), tuple(s["p2"].shape))
            self.assertEqual((W, H), tuple(s["orig_size"]))
            self.assertFalse(s["flip"])
            self.assertEqual((2, 4), tuple(t["boxes2d"].shape))
            self.assertEqual((2, 2), tuple(t["center3d"].shape))
            self.assertEqual((2, 3), tuple(t["dims"].shape))
            for k in ("labels", "depth", "alpha", "orient_bin", "orient_res"):
                self.assertEqual((2,), tuple(t[k].shape), k)
            self.assertEqual((24, 24), tuple(s["depth_map_target"].shape))
            self.assertEqual((24, 24), tuple(s["depth_map_mask"].shape))
            self.assertEqual(ds.class_ids["Car"], int(t["labels"][0]))
            self.assertEqual([1.5, 1.6, 3.9], [round(v, 4) for v in t["dims"][0].tolist()])
            self.assertAlmostEqual(20.0, float(t["depth"][0]), 4)
            alpha = T.alpha_from_ry(0.8, 2.0, 20.0)
            self.assertAlmostEqual(alpha, float(t["alpha"][0]), 5)
            b, r = T.multibin_target(alpha)
            self.assertEqual(b, int(t["orient_bin"][0]))
            self.assertAlmostEqual(r, float(t["orient_res"][0]), 5)
            u, v = T.project_center(2.0, 1.0, 20.0, P2)
            self.assertAlmostEqual(u / W, float(t["center3d"][0, 0]), 5)
            self.assertAlmostEqual(v / H, float(t["center3d"][0, 1]), 5)
            self.assertAlmostEqual((20 + 60) / 2 / W, float(t["boxes2d"][0, 0]), 5)
            self.assertAlmostEqual(40 / W, float(t["boxes2d"][0, 2]), 5)
            self.assertAlmostEqual(P2[0] * 48 / W, float(s["p2"][0]), 4)

    def test_mean_dims_per_class(self):
        with TemporaryDirectory() as d:
            ds = dataset(build(Path(d), {"a": [CAR, PED], "b": [CAR.replace("1.5 1.6 3.9", "2.5 1.6 4.9")]}))
            self.assertEqual(sorted(ds.mean_dims), ds.classes)
            self.assertEqual([2.0, 1.6, 4.4], [round(x, 4) for x in ds.mean_dims["Car"]])
            self.assertEqual([1.7, 0.6, 0.8], [round(x, 4) for x in ds.mean_dims["Pedestrian"]])

    def test_empty_gt_and_collate(self):
        with TemporaryDirectory() as d:
            ds = dataset(build(Path(d), {"a": [CAR, PED], "b": ["DontCare -1 -1 -10 1 1 5 5 -1 -1 -1 -1000 -1000 -1000 -10"]}))
            e = ds[1]["targets"]
            self.assertEqual((0, 4), tuple(e["boxes2d"].shape))
            self.assertEqual((0, 2), tuple(e["center3d"].shape))
            self.assertEqual((0, 3), tuple(e["dims"].shape))
            for k in ("labels", "depth", "alpha", "orient_bin", "orient_res"):
                self.assertEqual((0,), tuple(e[k].shape), k)
            self.assertEqual(torch.long, e["labels"].dtype)
            batch = collate_mono3d([ds[0], ds[1]])
            self.assertEqual((2, 3, 48, 48), tuple(batch["image"].shape))
            self.assertEqual((2, 12), tuple(batch["p2"].shape))
            self.assertEqual(2, len(batch["targets"]))
            self.assertEqual(2, len(batch["targets"][0]["labels"]))
            self.assertEqual(0, len(batch["targets"][1]["labels"]))
            self.assertEqual((2, 24, 24), tuple(batch["depth_map_target"].shape))
            self.assertEqual((2, 24, 24), tuple(batch["depth_map_mask"].shape))
            self.assertEqual((2, 2), tuple(batch["orig_size"].shape))

    def test_horizontal_flip(self):
        with TemporaryDirectory() as d:
            root = build(Path(d), {"a": [CAR]})
            plain, flipped = dataset(root)[0], dataset(root, flip_prob=1.0)[0]
            self.assertTrue(flipped["flip"])
            tp, tf = plain["targets"], flipped["targets"]
            self.assertAlmostEqual(1 - float(tp["boxes2d"][0, 0]), float(tf["boxes2d"][0, 0]), 5)
            self.assertAlmostEqual(float(tp["boxes2d"][0, 2]), float(tf["boxes2d"][0, 2]), 5)
            self.assertAlmostEqual(1 - float(tp["center3d"][0, 0]), float(tf["center3d"][0, 0]), 5)
            self.assertAlmostEqual(float(tp["center3d"][0, 1]), float(tf["center3d"][0, 1]), 5)
            # Ground truth: mirrored scene x->-x, yaw heading (cos,-sin) mirrored -> ry' = pi - ry.
            alpha = T.alpha_from_ry(math.pi - 0.8, -2.0, 20.0)
            self.assertAlmostEqual(alpha, float(tf["alpha"][0]), 5)
            b, r = T.multibin_target(alpha)
            self.assertEqual(b, int(tf["orient_bin"][0]))
            self.assertAlmostEqual(r, float(tf["orient_res"][0]), 5)
            # Mirrored camera projects mirrored location to the mirrored pixel.
            u, v = T.project_center(-2.0, 1.0, 20.0, flipped["p2"].tolist())
            self.assertAlmostEqual(float(tf["center3d"][0, 0]) * 48, u, 4)
            self.assertAlmostEqual(float(tf["center3d"][0, 1]) * 48, v, 4)
            self.assertTrue(torch.equal(plain["image"].flip(-1), flipped["image"]))
            self.assertTrue(torch.equal(plain["depth_map_target"].flip(-1), flipped["depth_map_target"]))

    def test_horizontal_flip_twice_is_identity(self):
        with TemporaryDirectory() as d:
            s = dataset(build(Path(d), {"a": [CAR, PED]}))[0]
            twice = horizontal_flip(horizontal_flip(s))
            for k in ("boxes2d", "center3d", "alpha", "orient_res"):
                self.assertTrue(torch.allclose(s["targets"][k], twice["targets"][k], atol=1e-5), k)
            self.assertTrue(torch.allclose(s["p2"], twice["p2"], atol=1e-4))


class ProposalCacheTest(unittest.TestCase):
    def test_round_trip_and_miss(self):
        with TemporaryDirectory() as d:
            cache = ProposalCache(Path(d) / "cache")
            self.assertIsNone(cache.get("000001"))
            prop = {"boxes2d": np.random.rand(5, 4).astype(np.float32), "scores": np.random.rand(5).astype(np.float32), "classes": np.arange(5)}
            cache.put("000001", prop)
            got = cache.get("000001")
            self.assertEqual({"boxes2d", "scores", "classes"}, set(got))
            for k in prop:
                np.testing.assert_array_equal(prop[k], got[k])


if __name__ == "__main__":
    unittest.main()
