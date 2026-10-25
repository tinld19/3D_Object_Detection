"""Fine-tune RF-DETR Nano on a COCO dataset prepared by prepare_dataset.py."""

import argparse
from pathlib import Path


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rfdetr-kitti"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--progress-bar", choices=("rich", "tqdm", "none"), default="rich")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    args = parser.parse_args()
    try:
        from rfdetr import RFDETRNano
    except ImportError as error:
        raise SystemExit('Install dependencies first: pip install "rfdetr[train]"') from error
    model = RFDETRNano()
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        device=args.device or device(),
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        progress_bar=None if args.progress_bar == "none" else args.progress_bar,
    )


if __name__ == "__main__":
    main()
