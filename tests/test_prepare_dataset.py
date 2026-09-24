import json
import struct
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import convert_dataset
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


if __name__ == "__main__":
    unittest.main()
