from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


CLASS_FILTER = [0, 1, 2, 7, 8, 9]


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Train a targeted v7 candidate to exceed v4 on all metrics with fitness > 0.95."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "datasets" / "dataset_flont" / "dataset_flont.yaml",
        help="Path to the YOLO dataset yaml.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v6_best" / "weights" / "best.pt",
        help="Initial checkpoint path.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments",
        help="Directory where training outputs will be written.",
    )
    parser.add_argument("--name", default="v7_candidate_a", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=18, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    parser.add_argument("--imgsz", type=int, default=896, help="Input image size.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.15, help="Final learning rate factor.")
    parser.add_argument("--box", type=float, default=8.5, help="Box loss gain.")
    parser.add_argument("--rle", type=float, default=1.4, help="Segmentation loss gain.")
    parser.add_argument("--translate", type=float, default=0.02, help="Translate augmentation ratio.")
    parser.add_argument("--scale", type=float, default=0.15, help="Scale augmentation ratio.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip ratio.")
    parser.add_argument("--mosaic", type=float, default=0.1, help="Mosaic augmentation ratio.")
    parser.add_argument("--close-mosaic", type=int, default=6, help="Epoch to disable mosaic.")
    parser.add_argument("--copy-paste", type=float, default=0.1, help="Copy-paste augmentation ratio.")
    parser.add_argument("--erasing", type=float, default=0.05, help="Random erasing ratio.")
    parser.add_argument("--freeze", type=int, default=16, help="Freeze the first N layers.")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save period.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Defaults to CUDA:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a minimal 1-epoch smoke test instead of the full training job.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")


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
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "box": args.box,
        "cls": 0.5,
        "dfl": 1.5,
        "rle": args.rle,
        "mask_ratio": 4,
        "overlap_mask": True,
        "translate": args.translate,
        "scale": args.scale,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "copy_paste": args.copy_paste,
        "erasing": args.erasing,
        "multi_scale": 0.0,
        "freeze": args.freeze,
        "cos_lr": True,
        "cache": "ram",
        "save": True,
        "save_period": args.save_period,
        "plots": not args.dry_run,
        "fraction": 0.02 if args.dry_run else 1.0,
        "seed": 0,
        "deterministic": True,
        "exist_ok": False,
    }

    print("Starting targeted v7 training")
    print(f"- name: {run_name}")
    print(f"- weights: {args.weights.resolve()}")
    print(f"- data: {args.data.resolve()}")
    print(f"- device: {device}")
    print(f"- imgsz: {train_kwargs['imgsz']}")
    print(f"- batch: {train_kwargs['batch']}")
    print(f"- freeze: {args.freeze}")
    print(f"- lr0/lrf: {args.lr0} / {args.lrf}")
    print(f"- box/rle: {args.box} / {args.rle}")
    print(f"- translate/scale/fliplr: {args.translate} / {args.scale} / {args.fliplr}")
    print(f"- mosaic/close_mosaic: {args.mosaic} / {args.close_mosaic}")
    print(f"- copy_paste/erasing: {args.copy_paste} / {args.erasing}")
    print(f"- dry_run: {args.dry_run}")

    model = YOLO(str(args.weights.resolve()))
    model.train(**train_kwargs)

    results_csv = project_path / run_name / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"results.csv was not found: {results_csv}")

    best_fitness, best_epoch = compute_best_fitness(results_csv)
    print(f"- best_fitness: {best_fitness:.6f} (epoch {best_epoch})")
    print(f"- results: {results_csv}")


if __name__ == "__main__":
    main()
