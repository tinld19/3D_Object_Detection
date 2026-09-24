"""Train the compact supervised monocular 3D regressor on KITTI labels."""

import argparse
import random
from pathlib import Path

from monocular_3d import KittiMonocular3DDataset, Monocular3DRegressor


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/data_object_image_2/training/image_2"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/training/label_2"))
    parser.add_argument("--calib-dir", type=Path, default=Path("data/data_object_calib/training/calib"))
    parser.add_argument("--output", type=Path, default=Path("runs/monocular-3d/best.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("mps", "cpu"))
    args = parser.parse_args()
    try:
        import torch
        from torch.nn import functional as functional
        from torch.utils.data import DataLoader, Subset
    except ImportError as error:
        raise SystemExit('Install dependencies first: pip install "rfdetr[train]" Pillow') from error
    dataset = KittiMonocular3DDataset(args.image_dir, args.label_dir, args.calib_dir)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    valid_count = max(1, round(len(indices) * 0.1))
    train_indices, valid_indices = indices[valid_count:], indices[:valid_count]
    targets = dataset.raw_targets()[train_indices]
    dataset.set_target_normalizer(targets.mean(0), targets.std(0).clamp_min(1e-6))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=args.batch_size, num_workers=args.num_workers)
    selected_device = args.device or device()
    model = Monocular3DRegressor(len(dataset.classes)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, class_ids, features, targets in train_loader:
            optimizer.zero_grad()
            loss = functional.mse_loss(model(images.to(selected_device), class_ids.to(selected_device), features.to(selected_device)), targets.to(selected_device))
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for images, class_ids, features, targets in valid_loader:
                prediction = model(images.to(selected_device), class_ids.to(selected_device), features.to(selected_device))
                losses.append(functional.mse_loss(prediction, targets.to(selected_device)).item())
        valid_loss = sum(losses) / len(losses)
        print(f"epoch={epoch} valid_mse={valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({"state_dict": model.state_dict(), "classes": dataset.classes, "target_mean": dataset.target_mean, "target_std": dataset.target_std}, args.output)


if __name__ == "__main__":
    main()
