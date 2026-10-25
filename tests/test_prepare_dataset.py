import json
import struct
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import convert_dataset
import monocular_3d
from monocular_3d import encode_yaw


class ConvertDatasetTest(unittest.TestCase):
    def test_encodes_yaw_as_sine_and_cosine(self):
        self.assertEqual((0.0, -1.0), encode_yaw(-3.141592653589793))

    def test_keeps_3d_metadata_and_skips_dontcare(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            calib_dir = root / "calib"
            output_dir = root / "dataset"
            for directory in (image_dir, label_dir, calib_dir):
                directory.mkdir()
            (image_dir / "000000.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 1242, 375, 8, 2, 0, 0, 0)
            )
            (label_dir / "000000.txt").write_text(
                "Car 0 0 0 10 20 110 220 1.5 1.6 4.0 2.0 1.0 15.0 0.5\n"
                "DontCare 0 0 0 0 0 1 1 0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            (calib_dir / "000000.txt").write_text("P2: 1 0 0 0\n", encoding="utf-8")

            convert_dataset(image_dir, label_dir, calib_dir, output_dir, val_fraction=0, seed=7)

            annotations = json.loads((output_dir / "train" / "_annotations.coco.json").read_text())
            self.assertEqual([{"id": 1, "name": "Car", "supercategory": "Car"}], annotations["categories"])
            self.assertEqual(1, len(annotations["annotations"]))
            annotation = annotations["annotations"][0]
            self.assertEqual([10.0, 20.0, 100.0, 200.0], annotation["bbox"])
            self.assertEqual([1.5, 1.6, 4.0], annotation["dimensions"])
            self.assertEqual([2.0, 1.0, 15.0], annotation["location"])
            self.assertEqual(0.5, annotation["rotation_y"])
            self.assertEqual([1.0, 0.0, 0.0, 0.0], annotations["images"][0]["calibration"]["P2"])

    def test_foreground_depth_target_uses_nearer_overlap_and_background(self):
        self.assertTrue(hasattr(monocular_3d, "foreground_depth_targets"))
        near = {"bbox": [30.0, 30.0, 20.0, 20.0], "location": [0.0, 0.0, 5.0]}
        far = {"bbox": [20.0, 20.0, 60.0, 60.0], "location": [0.0, 0.0, 25.0]}
        target, _ = monocular_3d.foreground_depth_targets([near, far], (100, 100), (10, 10))
        self.assertEqual(monocular_3d.depth_bin(5.0), target[3, 3].item())
        self.assertEqual(80, target[0, 0].item())

    def test_out_of_range_depth_is_ignored(self):
        self.assertTrue(hasattr(monocular_3d, "foreground_depth_targets"))
        too_near = {"bbox": [20.0, 20.0, 20.0, 20.0], "location": [0.0, 0.0, 1.9]}
        _, loss_mask = monocular_3d.foreground_depth_targets([too_near], (100, 100), (10, 10))
        self.assertIsNone(monocular_3d.depth_bin(1.9))
        self.assertIsNone(monocular_3d.depth_bin(60.1))
        self.assertFalse(loss_mask[2, 2].item())

    def test_image_dataset_returns_all_frame_objects_and_clipped_depth_target(self):
        self.assertTrue(hasattr(monocular_3d, "KittiForegroundDepthDataset"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, label_dir, calib_dir = root / "images", root / "labels", root / "calib"
            for directory in (image_dir, label_dir, calib_dir):
                directory.mkdir()
            from PIL import Image

            Image.new("RGB", (100, 100)).save(image_dir / "000000.png")
            (label_dir / "000000.txt").write_text(
                "Car 0 0 0 -10 20 30 60 1 1 1 0 0 10 0\n"
                "Van 0 0 0 50 50 90 90 1 1 1 0 0 20 0\n",
                encoding="utf-8",
            )
            (calib_dir / "000000.txt").write_text("P2: " + " ".join(["1"] * 12), encoding="utf-8")

            dataset = monocular_3d.KittiForegroundDepthDataset(image_dir, label_dir, calib_dir)
            sample = dataset[0]
            target, _ = monocular_3d.foreground_depth_targets(sample["objects"], sample["image_size"], (8, 12))

            self.assertEqual(2, sample["class_ids"].numel())
            self.assertEqual((3, 384, 384), tuple(sample["image"].shape))
            self.assertTrue((target != 80).any())
            self.assertEqual((2, 8), tuple(dataset.raw_targets([0]).shape))


if __name__ == "__main__":
    unittest.main()
