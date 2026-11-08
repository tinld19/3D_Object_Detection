"""Convert KITTI camera labels into the COCO layout accepted by RF-DETR."""

import argparse
import json
import shutil
import struct
from pathlib import Path


IGNORED_CLASSES = {"DontCare"}


def _read_split(path: Path) -> list[str]:
    """Read an ImageSets file into ordered, de-duplicated image-id stems."""
    ids = (line.strip() for line in path.read_text().splitlines())
    return list(dict.fromkeys(i for i in ids if i))


def _image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as image:
        if image.read(8) != b"\x89PNG\r\n\x1a\n" or image.read(4) != b"\x00\x00\x00\r" or image.read(4) != b"IHDR":
            raise ValueError(f"Expected a PNG image with an IHDR header: {path}")
        width, height = struct.unpack(">II", image.read(8))
    return width, height


def _calibration(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, numbers = line.partition(":")
        if separator:
            values[key] = [float(number) for number in numbers.split()]
    return values


def _labels(path: Path, width: int, height: int) -> list[dict]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 15 or fields[0] in IGNORED_CLASSES:
            continue
        left, top, right, bottom = map(float, fields[4:8])
        left, right = max(0.0, left), min(float(width), right)
        top, bottom = max(0.0, top), min(float(height), bottom)
        if right <= left or bottom <= top:
            continue
        labels.append(
            {
                "class_name": fields[0],
                "truncation": float(fields[1]),
                "occlusion": int(float(fields[2])),
                "bbox": [left, top, right - left, bottom - top],
                "dimensions": [float(value) for value in fields[8:11]],
                "location": [float(value) for value in fields[11:14]],
                "rotation_y": float(fields[14]),
            }
        )
    return labels


def _write_split(records: list[dict], categories: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations, images = [], []
    annotation_id = 1
    category_ids = {category["name"]: category["id"] for category in categories}
    for image_id, record in enumerate(records, start=1):
        destination = output_dir / record["image_path"].name
        shutil.copy2(record["image_path"], destination)
        images.append(
            {
                "id": image_id,
                "file_name": destination.name,
                "width": record["width"],
                "height": record["height"],
                "calibration": record["calibration"],
            }
        )
        for label in record["labels"]:
            bbox = label["bbox"]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[label["class_name"]],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "dimensions": label["dimensions"],
                    "location": label["location"],
                    "rotation_y": label["rotation_y"],
                }
            )
            annotation_id += 1
    (output_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}, indent=2), encoding="utf-8"
    )


def convert_dataset(
    image_dir: Path, label_dir: Path, calib_dir: Path, output_dir: Path, split_dir: Path
) -> None:
    """Create RF-DETR COCO `train` and `valid` splits from KITTI files using ImageSets membership."""
    train_stems = set(_read_split(split_dir / "train.txt"))
    val_stems = set(_read_split(split_dir / "val.txt"))
    records = []
    for image_path in sorted(image_dir.glob("*.png")):
        stem = image_path.stem
        if stem not in train_stems and stem not in val_stems:
            continue
        label_path, calib_path = label_dir / f"{stem}.txt", calib_dir / f"{stem}.txt"
        if not label_path.exists() or not calib_path.exists():
            continue
        width, height = _image_size(image_path)
        records.append(
            {
                "stem": stem,
                "image_path": image_path,
                "width": width,
                "height": height,
                "labels": _labels(label_path, width, height),
                "calibration": _calibration(calib_path),
            }
        )
    if not records:
        raise FileNotFoundError("No matching KITTI image, label, and calibration files were found.")
    class_names = sorted({label["class_name"] for record in records for label in record["labels"]})
    categories = [{"id": index, "name": name, "supercategory": name} for index, name in enumerate(class_names, start=1)]
    _write_split([r for r in records if r["stem"] in train_stems], categories, output_dir / "train")
    _write_split([r for r in records if r["stem"] in val_stems], categories, output_dir / "valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/data_object_image_2/training/image_2"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/training/label_2"))
    parser.add_argument("--calib-dir", type=Path, default=Path("data/data_object_calib/training/calib"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--split-dir", type=Path, default=Path("kitti_imagesets"))
    args = parser.parse_args()
    convert_dataset(args.image_dir, args.label_dir, args.calib_dir, args.output_dir, args.split_dir)


if __name__ == "__main__":
    main()
