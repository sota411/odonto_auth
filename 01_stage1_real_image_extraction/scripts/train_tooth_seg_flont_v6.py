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
        description="Train the final tooth_seg_flont_v6 configuration."
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
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v4_baseline" / "weights" / "best.pt",
        help="Initial checkpoint path.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments",
        help="Directory where training outputs will be written.",
    )
    parser.add_argument("--name", default="v6_candidate", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--imgsz", type=int, default=832, help="Input image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. -1 uses AutoBatch.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Defaults to CUDA:0 when available, otherwise cpu.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--lr0", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.1, help="Final learning rate factor.")
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
        reader = csv.DictReader(handle)
        rows = list(reader)
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
        "warmup_bias_lr": 0.0,
        "cos_lr": True,
        "close_mosaic": 0,
        "mosaic": 0.2,
        "translate": 0.02,
        "scale": 0.15,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "erasing": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "cache": True,
        "save": True,
        "plots": not args.dry_run,
        "fraction": 0.02 if args.dry_run else 1.0,
        "seed": 0,
        "deterministic": True,
        "exist_ok": False,
    }

    print("Starting v6 training")
    print(f"- weights: {args.weights.resolve()}")
    print(f"- data: {args.data.resolve()}")
    print(f"- device: {device}")
    print(f"- dry_run: {args.dry_run}")
    print(f"- cache: {train_kwargs['cache']}")
    print(f"- imgsz: {train_kwargs['imgsz']}")
    print(f"- classes: {CLASS_FILTER}")

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
