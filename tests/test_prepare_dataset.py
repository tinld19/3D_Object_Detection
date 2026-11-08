import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import _labels, _read_split, convert_dataset
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

            split_dir = root / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("000000\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("", encoding="utf-8")

            convert_dataset(image_dir, label_dir, calib_dir, output_dir, split_dir)

            annotations = json.loads((output_dir / "train" / "_annotations.coco.json").read_text())
            self.assertEqual([{"id": 1, "name": "Car", "supercategory": "Car"}], annotations["categories"])
            self.assertEqual(1, len(annotations["annotations"]))
            annotation = annotations["annotations"][0]
            self.assertEqual([10.0, 20.0, 100.0, 200.0], annotation["bbox"])
            self.assertEqual([1.5, 1.6, 4.0], annotation["dimensions"])
            self.assertEqual([2.0, 1.0, 15.0], annotation["location"])
            self.assertEqual(0.5, annotation["rotation_y"])
            self.assertEqual([1.0, 0.0, 0.0, 0.0], annotations["images"][0]["calibration"]["P2"])

    def test_convert_splits_by_imagesets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, label_dir, calib_dir = root / "images", root / "labels", root / "calib"
            split_dir, output_dir = root / "splits", root / "dataset"
            for directory in (image_dir, label_dir, calib_dir, split_dir):
                directory.mkdir()
            png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 1242, 375, 8, 2, 0, 0, 0)
            for stem in ("000000", "000001", "000002"):
                (image_dir / f"{stem}.png").write_bytes(png)
                (label_dir / f"{stem}.txt").write_text("Car 0 0 0 10 20 110 220 1.5 1.6 4.0 2.0 1.0 15.0 0.5\n", encoding="utf-8")
                (calib_dir / f"{stem}.txt").write_text("P2: 1 0 0 0\n", encoding="utf-8")
            (split_dir / "train.txt").write_text("000000\n000002\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("000001\n", encoding="utf-8")

            convert_dataset(image_dir, label_dir, calib_dir, output_dir, split_dir)

            def names(split):
                data = json.loads((output_dir / split / "_annotations.coco.json").read_text())
                return sorted(Path(image["file_name"]).stem for image in data["images"])

            self.assertEqual(["000000", "000002"], names("train"))
            self.assertEqual(["000001"], names("valid"))

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

    def test_labels_keep_truncation_and_occlusion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "000000.txt"
            label_path.write_text(
                "Car 0.15 1 1.58 539.13 182.36 566.23 206.67 1.41 1.59 4.47 -3.59 2.02 45.01 1.50\n",
                encoding="utf-8",
            )

            labels = _labels(label_path, 1242, 375)

            self.assertEqual(1, len(labels))
            self.assertEqual("Car", labels[0]["class_name"])
            self.assertEqual(0.15, labels[0]["truncation"])
            self.assertEqual(1, labels[0]["occlusion"])


class ReadSplitTest(unittest.TestCase):
    def test_read_split_parses_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "split.txt"
            path.write_text("000000\n000003\n\n000003\n000007")
            self.assertEqual(["000000", "000003", "000007"], _read_split(path))

    def test_read_split_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            _read_split(Path("does-not-exist.txt"))

    def test_committed_chen_split_partitions_7481(self):
        root = Path(__file__).resolve().parent.parent / "kitti_imagesets"
        train = _read_split(root / "train.txt")
        val = _read_split(root / "val.txt")
        self.assertEqual(3712, len(train))
        self.assertEqual(3769, len(val))
        self.assertTrue(set(train).isdisjoint(val))
        self.assertEqual(7481, len(set(train) | set(val)))
        self.assertTrue(all(re.fullmatch(r"\d{6}", i) for i in train + val))


if __name__ == "__main__":
    unittest.main()
