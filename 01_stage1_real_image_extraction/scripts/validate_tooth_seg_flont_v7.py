from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


CLASS_FILTER = [0, 1, 2, 7, 8, 9]
METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
]


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Validate the v7 front-tooth segmentation profile and compare it with v4."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "datasets" / "dataset_flont" / "dataset_flont.yaml",
        help="Path to the dataset yaml.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v7_best" / "weights" / "best.pt",
        help="Checkpoint used for the v7 profile.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v4_baseline" / "results.csv",
        help="Baseline v4 results.csv path.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "02_stage2_capture_matching" / "evaluation" / "validation_runs",
        help="Directory where validation artifacts are written.",
    )
    parser.add_argument(
        "--name",
        default="tooth_seg_flont_v7",
        help="Output directory name under --project.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="Validation device.")
    parser.add_argument("--workers", type=int, default=0, help="Validation dataloader workers.")
    return parser.parse_args()


def compute_fitness(metrics: dict[str, float]) -> float:
    box = metrics["metrics/mAP50(B)"] * 0.1 + metrics["metrics/mAP50-95(B)"] * 0.9
    mask = metrics["metrics/mAP50(M)"] * 0.1 + metrics["metrics/mAP50-95(M)"] * 0.9
    return (box + mask) / 2.0


def load_best_row(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"results.csv is empty: {path}")

    best_row = max(
        rows,
        key=lambda row: (
            float(row["metrics/mAP50(B)"]) * 0.1
            + float(row["metrics/mAP50-95(B)"]) * 0.9
            + float(row["metrics/mAP50(M)"]) * 0.1
            + float(row["metrics/mAP50-95(M)"]) * 0.9
        )
        / 2.0,
    )
    return {key: float(best_row[key]) for key in METRIC_KEYS}


def write_results_csv(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", *METRIC_KEYS, "fitness"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                **{key: f"{metrics[key]:.6f}" for key in METRIC_KEYS},
                "fitness": f"{compute_fitness(metrics):.6f}",
            }
        )


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights.resolve()))
    val_result = model.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        classes=CLASS_FILTER,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=False,
        verbose=False,
    )

    save_dir = Path(val_result.save_dir)
    metrics = {key: float(val_result.results_dict[key]) for key in METRIC_KEYS}
    fitness = compute_fitness(metrics)
    baseline = load_best_row(args.baseline)

    deltas = {key: metrics[key] - baseline[key] for key in METRIC_KEYS}
    all_win = all(delta > 0 for delta in deltas.values())

    write_results_csv(save_dir / "results.csv", metrics)

    payload = {
        "profile": {
            "weights": str(args.weights.resolve()),
            "data": str(args.data.resolve()),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "classes": CLASS_FILTER,
        },
        "metrics": metrics,
        "fitness": fitness,
        "baseline": baseline,
        "deltas_vs_v4_best": deltas,
        "all_metrics_win": all_win,
    }
    (save_dir / "v7_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[save_dir] {save_dir}")
    print(f"[fitness] {fitness:.6f}")
    print(f"[all_metrics_win_vs_v4_best] {all_win}")
    for key in METRIC_KEYS:
        print(
            f"{key}: "
            f"candidate={metrics[key]:.6f} "
            f"baseline={baseline[key]:.6f} "
            f"delta={deltas[key]:+.6f}"
        )


if __name__ == "__main__":
    main()
