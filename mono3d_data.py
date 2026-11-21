"""Dataset, collate, flip augmentation and proposal cache for depth-aware Mono3D DETR."""
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mono3d_targets import _wrap_to_pi, alpha_from_ry, multibin_target, project_center, rescale_p2
from monocular_3d import foreground_depth_targets
from prepare_dataset import _calibration, _image_size, _labels, _read_split

TARGET_KEYS = ("labels", "boxes2d", "center3d", "depth", "dims", "alpha", "orient_bin", "orient_res")


def _empty_targets() -> dict:
    return {
        "labels": torch.zeros(0, dtype=torch.long), "boxes2d": torch.zeros(0, 4), "center3d": torch.zeros(0, 2),
        "depth": torch.zeros(0), "dims": torch.zeros(0, 3), "alpha": torch.zeros(0),
        "orient_bin": torch.zeros(0, dtype=torch.long), "orient_res": torch.zeros(0),
    }


def horizontal_flip(sample: dict, bins: int = 12) -> dict:
    """Mirror a sample (returns a new dict). Applying it twice restores the original.

    Mirroring x -> -x maps yaw ry -> pi - ry (heading (cos ry, -sin ry) in x-z), hence
    alpha -> pi - alpha. The P2 principal point becomes W - c_x and its x translation flips sign,
    so a mirrored 3D point projects onto the mirrored pixel.
    """
    t = dict(sample["targets"])
    boxes, center = t["boxes2d"].clone(), t["center3d"].clone()
    boxes[:, 0] = 1.0 - boxes[:, 0]
    center[:, 0] = 1.0 - center[:, 0]
    alpha = torch.tensor([_wrap_to_pi(math.pi - a) for a in t["alpha"].tolist()], dtype=torch.float32)
    bin_res = [multibin_target(a, bins) for a in alpha.tolist()]
    t.update(
        boxes2d=boxes, center3d=center, alpha=alpha,
        orient_bin=torch.tensor([b for b, _ in bin_res], dtype=torch.long),
        orient_res=torch.tensor([r for _, r in bin_res], dtype=torch.float32),
    )
    p2 = sample["p2"].clone()
    width_in = float(sample["image"].shape[-1])
    p2[2] = width_in - p2[2]
    p2[3] = -p2[3]
    out = dict(sample)
    out.update(
        image=sample["image"].flip(-1), p2=p2, targets=t, flip=not sample["flip"],
        depth_map_target=sample["depth_map_target"].flip(-1), depth_map_mask=sample["depth_map_mask"].flip(-1),
    )
    return out


class Mono3DDetrDataset(Dataset):
    """One sample per KITTI image; images with no valid objects yield empty targets."""

    def __init__(
        self, image_dir: Path, label_dir: Path, calib_dir: Path, feature_size: tuple[int, int],
        image_size: int = 384, flip_prob: float = 0.0, split_file: Path | None = None, bins: int = 12,
    ) -> None:
        self.image_size, self.feature_size, self.flip_prob, self.bins = image_size, tuple(feature_size), flip_prob, bins
        stems = _read_split(split_file) if split_file else sorted(p.stem for p in Path(image_dir).glob("*.png"))
        self.samples = []
        for stem in stems:
            image_path = Path(image_dir) / f"{stem}.png"
            label_path, calib_path = Path(label_dir) / f"{stem}.txt", Path(calib_dir) / f"{stem}.txt"
            if not (image_path.exists() and label_path.exists() and calib_path.exists()):
                continue
            width, height = _image_size(image_path)
            p2 = _calibration(calib_path).get("P2")
            if len(p2 or []) != 12:
                continue
            self.samples.append(
                {"stem": stem, "image": image_path, "width": width, "height": height, "p2": p2,
                 "labels": _labels(label_path, width, height)}
            )
        if not self.samples:
            raise FileNotFoundError("No valid KITTI samples were found.")
        self.classes = sorted({o["class_name"] for s in self.samples for o in s["labels"]})
        self.class_ids = {name: i for i, name in enumerate(self.classes)}
        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for s in self.samples:
            for o in s["labels"]:
                acc = sums.setdefault(o["class_name"], [0.0, 0.0, 0.0])
                for i, v in enumerate(o["dimensions"]):
                    acc[i] += v
                counts[o["class_name"]] = counts.get(o["class_name"], 0) + 1
        self.mean_dims = {c: [v / counts[c] for v in sums[c]] for c in self.classes}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        s = self.samples[index]
        w, h = s["width"], s["height"]
        size = self.image_size
        with Image.open(s["image"]) as source:
            pixels = source.convert("RGB").resize((size, size))
            image = torch.tensor(bytearray(pixels.tobytes()), dtype=torch.float32).view(size, size, 3)
        image = image.permute(2, 0, 1) / 255.0
        p2 = torch.tensor(rescale_p2(s["p2"], (w, h), (size, size)), dtype=torch.float32)
        targets = _empty_targets()
        objects = s["labels"]
        if objects:
            rows = {k: [] for k in TARGET_KEYS}
            for o in objects:
                x, y, z = o["location"]
                bx, by, bw, bh = o["bbox"]
                u, v = project_center(x, y, z, s["p2"])
                alpha = alpha_from_ry(o["rotation_y"], x, z)
                b, r = multibin_target(alpha, self.bins)
                rows["labels"].append(self.class_ids[o["class_name"]])
                rows["boxes2d"].append([(bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h])
                rows["center3d"].append([u / w, v / h])
                rows["depth"].append(z)
                rows["dims"].append(o["dimensions"])
                rows["alpha"].append(alpha)
                rows["orient_bin"].append(b)
                rows["orient_res"].append(r)
            targets = {
                k: torch.tensor(rows[k], dtype=torch.long if k in ("labels", "orient_bin") else torch.float32)
                for k in TARGET_KEYS
            }
        depth_target, depth_mask = foreground_depth_targets(objects, (w, h), self.feature_size)
        sample = {
            "stem": s["stem"], "image": image, "p2": p2, "orig_size": torch.tensor([w, h]), "targets": targets,
            "depth_map_target": depth_target, "depth_map_mask": depth_mask, "flip": False,
        }
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            sample = horizontal_flip(sample, self.bins)
        return sample


def collate_mono3d(batch: list[dict]) -> dict:
    return {
        "stem": [b["stem"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "p2": torch.stack([b["p2"] for b in batch]),
        "orig_size": torch.stack([b["orig_size"] for b in batch]),
        "targets": [b["targets"] for b in batch],
        "depth_map_target": torch.stack([b["depth_map_target"] for b in batch]),
        "depth_map_mask": torch.stack([b["depth_map_mask"] for b in batch]),
        "flip": [b["flip"] for b in batch],
    }


class ProposalCache:
    """Per-image RF-DETR proposals (boxes2d[P,4], scores[P], classes[P]) stored as .npz files."""

    KEYS = ("boxes2d", "scores", "classes")

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, stem: str) -> Path:
        return self.cache_dir / f"{stem}.npz"

    def get(self, stem: str) -> dict | None:
        path = self._path(stem)
        if not path.exists():
            return None
        with np.load(path) as data:
            return {k: data[k] for k in self.KEYS}

    def put(self, stem: str, proposals: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._path(stem), **{k: np.asarray(proposals[k]) for k in self.KEYS})
