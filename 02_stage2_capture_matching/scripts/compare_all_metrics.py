from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class RowMetrics:
    epoch: int
    fitness: float
    metrics: dict[str, float]


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Compare a candidate run against a baseline across all YOLO segmentation metrics."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v4_baseline" / "results.csv",
        help="Baseline results.csv path.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v7_best" / "results.csv",
        help="Candidate results.csv path.",
    )
    return parser.parse_args()


def compute_fitness(row: dict[str, str]) -> float:
    box = float(row["metrics/mAP50(B)"]) * 0.1 + float(row["metrics/mAP50-95(B)"]) * 0.9
    mask = float(row["metrics/mAP50(M)"]) * 0.1 + float(row["metrics/mAP50-95(M)"]) * 0.9
    return (box + mask) / 2.0


def load_best_fitness_row(path: Path) -> RowMetrics:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"results.csv is empty: {path}")

    best_row = max(rows, key=compute_fitness)
    return RowMetrics(
        epoch=int(float(best_row["epoch"])),
        fitness=compute_fitness(best_row),
        metrics={key: float(best_row[key]) for key in METRIC_KEYS},
    )


def main() -> int:
    args = parse_args()
    baseline = load_best_fitness_row(args.baseline)
    candidate = load_best_fitness_row(args.candidate)

    print(f"[baseline] epoch={baseline.epoch} fitness={baseline.fitness:.6f}")
    for key in METRIC_KEYS:
        print(f"  {key}={baseline.metrics[key]:.6f}")

    print(f"[candidate] epoch={candidate.epoch} fitness={candidate.fitness:.6f}")
    all_win = True
    for key in METRIC_KEYS:
        delta = candidate.metrics[key] - baseline.metrics[key]
        status = "WIN" if delta > 0 else "LOSE_OR_TIE"
        if delta <= 0:
            all_win = False
        print(f"  {key}={candidate.metrics[key]:.6f} delta={delta:+.6f} {status}")

    fitness_delta = candidate.fitness - baseline.fitness
    print(f"[fitness] delta={fitness_delta:+.6f}")
    print(f"[all_metrics_win] {all_win}")
    return 0 if all_win else 1


if __name__ == "__main__":
    raise SystemExit(main())
