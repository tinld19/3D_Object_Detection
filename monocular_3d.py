"""Data and model pieces for supervised KITTI monocular 3D box regression."""

import math
from pathlib import Path

from prepare_dataset import _calibration, _image_size, _labels

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    nn = None
    Dataset = object


def encode_yaw(yaw: float) -> tuple[float, float]:
    return round(math.sin(yaw), 12), round(math.cos(yaw), 12)


def _require_torch() -> None:
    if torch is None:
        raise ImportError('Install dependencies first: pip install "rfdetr[train]" Pillow')


if nn is not None:

    class Monocular3DRegressor(nn.Module):
        """Small crop CNN predicting normalised KITTI 3D box parameters."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.class_embedding = nn.Embedding(num_classes, 8)
            self.head = nn.Sequential(nn.Linear(144, 128), nn.ReLU(), nn.Linear(128, 8))

        def forward(self, image, class_id, features):
            return self.head(torch.cat((self.visual(image), self.class_embedding(class_id), features), dim=1))

else:

    class Monocular3DRegressor:
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


class KittiMonocular3DDataset(Dataset):
    """One sample per labelled KITTI 2D object crop."""

    def __init__(self, image_dir: Path, label_dir: Path, calib_dir: Path, image_size: int = 224) -> None:
        _require_torch()
        self.image_size = image_size
        self.samples = []
        for image_path in sorted(image_dir.glob("*.png")):
            label_path, calib_path = label_dir / f"{image_path.stem}.txt", calib_dir / f"{image_path.stem}.txt"
            if not label_path.exists() or not calib_path.exists():
                continue
            width, height = _image_size(image_path)
            calibration = _calibration(calib_path)
            p2 = calibration.get("P2")
            if len(p2 or []) != 12:
                continue
            for label in _labels(label_path, width, height):
                self.samples.append({"image": image_path, "width": width, "height": height, "p2": p2, **label})
        if not self.samples:
            raise FileNotFoundError("No valid KITTI 3D labels were found.")
        self.classes = sorted({sample["class_name"] for sample in self.samples})
        self.class_ids = {name: index for index, name in enumerate(self.classes)}
        self.target_mean = torch.zeros(8)
        self.target_std = torch.ones(8)

    def set_target_normalizer(self, mean, std) -> None:
        self.target_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.target_std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)

    def raw_targets(self):
        return torch.tensor([self._target(sample) for sample in self.samples], dtype=torch.float32)

    @staticmethod
    def _target(sample: dict) -> list[float]:
        return sample["dimensions"] + sample["location"] + list(encode_yaw(sample["rotation_y"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        sample = self.samples[index]
        x, y, width, height = sample["bbox"]
        with Image.open(sample["image"]) as source:
            crop = source.convert("RGB").crop((round(x), round(y), round(x + width), round(y + height)))
            crop = crop.resize((self.image_size, self.image_size))
            image = torch.tensor(bytearray(crop.tobytes()), dtype=torch.float32).view(self.image_size, self.image_size, 3)
        image = image.permute(2, 0, 1).div_(255)
        p2 = sample["p2"]
        features = torch.tensor(
            [x / sample["width"], y / sample["height"], width / sample["width"], height / sample["height"],
             p2[0] / sample["width"], p2[5] / sample["height"], p2[2] / sample["width"], p2[6] / sample["height"]],
            dtype=torch.float32,
        )
        target = (torch.tensor(self._target(sample), dtype=torch.float32) - self.target_mean) / self.target_std
        return image, self.class_ids[sample["class_name"]], features, target
