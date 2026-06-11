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
        description="tooth_seg_flont_v4 の best.pt から tooth_seg_flont_v5 を fine-tune します。"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "datasets" / "dataset_flont" / "dataset_flont.yaml",
        help="YOLO dataset yaml のパス",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v4_baseline" / "weights" / "best.pt",
        help="初期重みのパス",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments",
        help="学習結果の出力先",
    )
    parser.add_argument("--name", default="v5_candidate", help="実験名")
    parser.add_argument("--epochs", type=int, default=80, help="最大エポック数")
    parser.add_argument("--patience", type=int, default=20, help="Early Stopping の patience")
    parser.add_argument("--imgsz", type=int, default=768, help="入力画像サイズ")
    parser.add_argument("--batch", type=int, default=-1, help="バッチサイズ。-1 で自動最適化")
    parser.add_argument(
        "--device",
        default=None,
        help="学習デバイス。未指定時は CUDA があれば 0、なければ cpu",
    )
    parser.add_argument("--workers", type=int, default=8, help="データローダ worker 数")
    parser.add_argument("--lr0", type=float, default=5e-4, help="初期学習率")
    parser.add_argument("--lrf", type=float, default=0.1, help="最終学習率比")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="1 epoch / 少量データで疎通確認のみ行います",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} が見つかりません: {path}")


def compute_best_fitness(results_csv: Path) -> tuple[float, int]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"results.csv が空です: {results_csv}")

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
    ensure_file(args.weights, "初期重み")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "ultralytics"))

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("torch / ultralytics が必要です。`uv sync` などで依存を導入してください。") from exc

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
        "mosaic": 0.0,
        "translate": 0.02,
        "scale": 0.15,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "erasing": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "save": True,
        "plots": not args.dry_run,
        "fraction": 0.02 if args.dry_run else 1.0,
        "seed": 0,
        "deterministic": True,
        "exist_ok": False,
    }

    print("v5 学習を開始します。")
    print(f"- weights: {args.weights.resolve()}")
    print(f"- data: {args.data.resolve()}")
    print(f"- device: {device}")
    print(f"- dry_run: {args.dry_run}")
    print(f"- classes: {CLASS_FILTER}")

    model = YOLO(str(args.weights.resolve()))
    model.train(**train_kwargs)

    results_csv = project_path / run_name / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"results.csv は見つかりませんでした: {results_csv}")

    best_fitness, best_epoch = compute_best_fitness(results_csv)
    print(f"- best_fitness: {best_fitness:.6f} (epoch {best_epoch})")
    print(f"- results: {results_csv}")


if __name__ == "__main__":
    main()
