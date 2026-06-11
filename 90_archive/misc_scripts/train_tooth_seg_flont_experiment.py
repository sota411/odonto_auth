from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


CLASS_FILTER = [0, 1, 2, 7, 8, 9]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a configurable tooth_seg_flont experiment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "dataset_flont" / "dataset_flont.yaml",
        help="Path to the YOLO dataset yaml.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "tooth_recognition_flont" / "tooth_seg_flont_v5" / "weights" / "best.pt",
        help="Initial checkpoint path.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "tooth_recognition_flont",
        help="Directory where training outputs will be written.",
    )
    parser.add_argument("--name", required=True, help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--imgsz", type=int, default=768, help="Input image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. -1 uses AutoBatch.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.05, help="Final learning rate factor.")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer name.")
    parser.add_argument("--momentum", type=float, default=0.937, help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Warmup epochs.")
    parser.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup momentum.")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0, help="Warmup bias lr.")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain.")
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss gain.")
    parser.add_argument("--dfl", type=float, default=1.5, help="Distribution focal loss gain.")
    parser.add_argument("--rle", type=float, default=1.0, help="Segmentation loss gain.")
    parser.add_argument("--mask-ratio", type=int, default=4, help="Mask downsample ratio.")
    parser.add_argument(
        "--overlap-mask",
        default="true",
        choices=["true", "false"],
        help="Whether to use overlapping mask training targets.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout ratio.")
    parser.add_argument("--translate", type=float, default=0.02, help="Translate augmentation.")
    parser.add_argument("--scale", type=float, default=0.15, help="Scale augmentation.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability.")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic augmentation.")
    parser.add_argument("--close-mosaic", type=int, default=0, help="Epochs before mosaic is disabled.")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation.")
    parser.add_argument("--erasing", type=float, default=0.0, help="Random erasing ratio.")
    parser.add_argument("--multi-scale", type=float, default=0.0, help="Enable multi-scale training when > 0.")
    parser.add_argument(
        "--cache",
        default="ram",
        choices=["false", "disk", "ram"],
        help="Dataset cache mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Defaults to CUDA:0 when available, otherwise cpu.",
    )
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save period.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction for quick search.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a minimal 1-epoch smoke test instead of the full training job.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")


def parse_cache(mode: str) -> bool | str:
    if mode == "false":
        return False
    return mode


def parse_bool(mode: str) -> bool:
    return mode == "true"


def compute_best_fitness(results_csv: Path) -> tuple[float, int]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"results.csv is empty: {results_csv}")

    best_fitness = -1.0
    best_epoch = -1
    for row in rows:
        box = float(row["metrics/mAP50(B)"]) * 0.1 + float(row["metrics/mAP50-95(B)"]) * 0.9
        mask = float(row["metrics/mAP50(M)"]) * 0.1 + float(row["metrics/mAP50-95(M)"]) * 0.9
        fitness = (box + mask) / 2.0
        epoch = int(float(row["epoch"]))
        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch
    return best_fitness, best_epoch


def main() -> None:
    args = parse_args()
    ensure_file(args.data, "dataset yaml")
    ensure_file(args.weights, "initial checkpoint")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "ultralytics"))

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("torch and ultralytics must be installed before training.") from exc

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.name}_dry_run" if args.dry_run else args.name
    project_path = args.project.resolve()

    train_kwargs = {
        "data": str(args.data.resolve()),
        "epochs": 1 if args.dry_run else args.epochs,
        "imgsz": min(args.imgsz, 640) if args.dry_run else args.imgsz,
        "batch": 1 if args.dry_run and str(device) == "cpu" else args.batch,
        "device": device,
        "workers": args.workers,
        "project": str(project_path),
        "name": run_name,
        "patience": 1 if args.dry_run else args.patience,
        "classes": CLASS_FILTER,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "rle": args.rle,
        "mask_ratio": args.mask_ratio,
        "overlap_mask": parse_bool(args.overlap_mask),
        "dropout": args.dropout,
        "translate": args.translate,
        "scale": args.scale,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "copy_paste": args.copy_paste,
        "erasing": args.erasing,
        "cos_lr": True,
        "multi_scale": args.multi_scale,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "cache": parse_cache(args.cache),
        "save": True,
        "save_period": args.save_period,
        "plots": not args.dry_run,
        "fraction": 0.02 if args.dry_run else args.fraction,
        "seed": 0,
        "deterministic": True,
        "exist_ok": False,
    }

    print("Starting experiment")
    print(f"- name: {run_name}")
    print(f"- weights: {args.weights.resolve()}")
    print(f"- device: {device}")
    print(f"- cache: {train_kwargs['cache']}")
    print(f"- imgsz: {train_kwargs['imgsz']}")
    print(f"- lr0: {args.lr0}")
    print(f"- lrf: {args.lrf}")
    print(f"- momentum: {args.momentum}")
    print(f"- box: {args.box}")
    print(f"- cls: {args.cls}")
    print(f"- dfl: {args.dfl}")
    print(f"- rle: {args.rle}")
    print(f"- mask_ratio: {args.mask_ratio}")
    print(f"- multi_scale: {args.multi_scale}")
    print(f"- fraction: {train_kwargs['fraction']}")

    model = YOLO(str(args.weights.resolve()))
    model.train(**train_kwargs)

    results_csv = project_path / run_name / "results.csv"
    if results_csv.exists():
        best_fitness, best_epoch = compute_best_fitness(results_csv)
        print(f"- best_fitness: {best_fitness:.6f} (epoch {best_epoch})")
        print(f"- results: {results_csv}")
    else:
        print(f"- results.csv was not found: {results_csv}")


if __name__ == "__main__":
    main()
