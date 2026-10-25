"""Data and model pieces for supervised KITTI monocular 3D box regression."""

import math
from pathlib import Path
from types import SimpleNamespace

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


def depth_bin(depth: float, depth_min: float = 2.0, depth_max: float = 60.0, bins: int = 80) -> int | None:
    """Return MonoDETR-style LID bin for a foreground metric depth."""
    if not depth_min <= depth <= depth_max:
        return None
    delta = 2 * (depth_max - depth_min) / (bins * (bins + 1))
    index = math.floor(-0.5 + 0.5 * math.sqrt(1 + 8 * (depth - depth_min) / delta))
    return min(bins - 1, max(0, index))


def foreground_depth_targets(
    objects: list[dict], image_size: tuple[int, int], feature_size: tuple[int, int], depth_min: float = 2.0,
    depth_max: float = 60.0, bins: int = 80,
):
    """Paint object-wise foreground depth bins and their supervision mask."""
    _require_torch()
    image_width, image_height = image_size
    feature_height, feature_width = feature_size
    target = torch.full((feature_height, feature_width), bins, dtype=torch.long)
    loss_mask = torch.ones((feature_height, feature_width), dtype=torch.bool)
    for object_ in sorted(objects, key=lambda item: item["location"][2], reverse=True):
        x, y, width, height = object_["bbox"]
        left = max(0, min(feature_width - 1, math.floor(x * feature_width / image_width)))
        top = max(0, min(feature_height - 1, math.floor(y * feature_height / image_height)))
        right = max(left + 1, min(feature_width, math.ceil((x + width) * feature_width / image_width)))
        bottom = max(top + 1, min(feature_height, math.ceil((y + height) * feature_height / image_height)))
        depth = depth_bin(object_["location"][2], depth_min, depth_max, bins)
        if depth is None:
            loss_mask[top:bottom, left:right] = False
        else:
            target[top:bottom, left:right] = depth
            loss_mask[top:bottom, left:right] = True
    return target, loss_mask


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


    class ForegroundDepth3DRegressor(nn.Module):
        """Fuse frozen DINO features with learned foreground-depth features."""

        def __init__(self, num_classes: int, feature_channels: int = 256, depth_bins: int = 80) -> None:
            super().__init__()
            self.depth_predictor = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1), nn.ReLU(),
            )
            self.depth_logits = nn.Conv2d(feature_channels, depth_bins + 1, 1)
            self.class_embedding = nn.Embedding(num_classes, 8)
            self.head = nn.Sequential(
                nn.Linear(feature_channels * 2 + 16, 256), nn.ReLU(), nn.Linear(256, 8),
            )

        @staticmethod
        def _pool(feature, box):
            _, height, width = feature.shape
            left = max(0, min(width - 1, math.floor(box[0].item() * width)))
            top = max(0, min(height - 1, math.floor(box[1].item() * height)))
            right = max(left + 1, min(width, math.ceil((box[0] + box[2]).item() * width)))
            bottom = max(top + 1, min(height, math.ceil((box[1] + box[3]).item() * height)))
            return torch.nn.functional.adaptive_avg_pool2d(feature[:, top:bottom, left:right], 1).flatten()

        def forward(self, rgb_features, boxes, class_ids, geometry):
            depth_features = self.depth_predictor(rgb_features)
            depth_logits = self.depth_logits(depth_features)
            predictions = []
            for index, object_boxes in enumerate(boxes):
                rgb_pooled = torch.stack([self._pool(rgb_features[index], box) for box in object_boxes])
                depth_pooled = torch.stack([self._pool(depth_features[index], box) for box in object_boxes])
                semantic = self.class_embedding(class_ids[index].to(rgb_features.device))
                predictions.append(self.head(torch.cat((rgb_pooled, depth_pooled, semantic, geometry[index].to(rgb_features.device)), dim=1)))
            return predictions, depth_logits


    class RFDETRDinoAdapter(nn.Module):
        """Expose RF-DETR Nano's frozen, projected DINO feature map."""

        def __init__(self, checkpoint: Path, loader=None) -> None:
            super().__init__()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"RF-DETR checkpoint not found: {checkpoint}")
            wrapper = (loader or self._load)(checkpoint)
            try:
                self.backbone = wrapper.model.model.backbone[0]
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError("RF-DETR Nano checkpoint does not expose a DINO backbone.") from error
            self.register_buffer("mean", torch.tensor(wrapper.means).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(wrapper.stds).view(1, -1, 1, 1))
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        @staticmethod
        def _load(checkpoint: Path):
            try:
                from rfdetr import RFDETRNano
            except ImportError as error:
                raise ImportError("Install rfdetr[train]==1.11.2 before loading an RF-DETR checkpoint.") from error

            model = RFDETRNano.from_checkpoint(checkpoint)
            if not isinstance(model, RFDETRNano):
                raise RuntimeError("The 3D stage supports RF-DETR Nano checkpoints only.")
            return model

        def forward(self, images):
            self.backbone.eval()
            nested = SimpleNamespace(tensors=(images - self.mean) / self.std, no_padding=True)
            with torch.no_grad():
                features, _ = self.backbone(nested)
            feature = features[0].tensors
            if feature.ndim != 4 or feature.shape[1] != 256:
                raise RuntimeError("RF-DETR backbone must return a 256-channel spatial DINO feature map.")
            return feature

else:

    class Monocular3DRegressor:
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class ForegroundDepth3DRegressor:
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class RFDETRDinoAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


class KittiForegroundDepthDataset(Dataset):
    """One sample per KITTI image with all valid foreground objects."""

    def __init__(self, image_dir: Path, label_dir: Path, calib_dir: Path, image_size: int = 384) -> None:
        _require_torch()
        self.image_size = image_size
        self.samples = []
        for image_path in sorted(image_dir.glob("*.png")):
            label_path, calib_path = label_dir / f"{image_path.stem}.txt", calib_dir / f"{image_path.stem}.txt"
            if not label_path.exists() or not calib_path.exists():
                continue
            width, height = _image_size(image_path)
            p2 = _calibration(calib_path).get("P2")
            labels = _labels(label_path, width, height)
            if len(p2 or []) != 12 or not labels:
                continue
            self.samples.append({"image": image_path, "width": width, "height": height, "p2": p2, "labels": labels})
        if not self.samples:
            raise FileNotFoundError("No valid KITTI 3D labels were found.")
        self.classes = sorted({label["class_name"] for sample in self.samples for label in sample["labels"]})
        self.class_ids = {name: index for index, name in enumerate(self.classes)}
        self.target_mean = torch.zeros(8)
        self.target_std = torch.ones(8)

    @staticmethod
    def _target(label: dict) -> list[float]:
        return label["dimensions"] + label["location"] + list(encode_yaw(label["rotation_y"]))

    def set_target_normalizer(self, mean, std) -> None:
        self.target_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.target_std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)

    def raw_targets(self, indices=None):
        samples = self.samples if indices is None else (self.samples[index] for index in indices)
        return torch.tensor([self._target(label) for sample in samples for label in sample["labels"]], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB").resize((self.image_size, self.image_size))
            image = torch.tensor(bytearray(image.tobytes()), dtype=torch.float32).view(self.image_size, self.image_size, 3)
        scale_x, scale_y = self.image_size / sample["width"], self.image_size / sample["height"]
        objects = []
        for label in sample["labels"]:
            x, y, width, height = label["bbox"]
            objects.append({**label, "bbox": [x * scale_x, y * scale_y, width * scale_x, height * scale_y]})
        p2 = sample["p2"]
        boxes = torch.tensor(
            [
                [x / sample["width"], y / sample["height"], width / sample["width"], height / sample["height"]]
                for x, y, width, height in (label["bbox"] for label in sample["labels"])
            ],
            dtype=torch.float32,
        )
        geometry = torch.cat((boxes, torch.tensor(
            [p2[0] / sample["width"], p2[5] / sample["height"], p2[2] / sample["width"], p2[6] / sample["height"]],
            dtype=torch.float32,
        ).expand(len(objects), -1)), dim=1)
        targets = (torch.tensor([self._target(label) for label in sample["labels"]], dtype=torch.float32) - self.target_mean) / self.target_std
        return {
            "image": image.permute(2, 0, 1).div_(255),
            "image_size": (self.image_size, self.image_size),
            "objects": objects,
            "boxes": boxes,
            "class_ids": torch.tensor([self.class_ids[label["class_name"]] for label in sample["labels"]], dtype=torch.long),
            "geometry": geometry,
            "targets": targets,
        }


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
