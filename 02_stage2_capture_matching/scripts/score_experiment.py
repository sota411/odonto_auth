from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BestResult:
    csv_path: Path
    epoch: int
    fitness: float
    box_map50: float
    box_map: float
    mask_map50: float
    mask_map: float


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="YOLO segmentation の official fitness で実験結果を比較します。"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v4_baseline" / "results.csv",
        help="比較基準となる results.csv",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v7_best" / "results.csv",
        help="比較対象となる results.csv",
    )
    return parser.parse_args()


def read_best_result(path: Path) -> BestResult:
    if not path.exists():
        raise RuntimeError(f"results.csv が見つかりません: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"results.csv が空です: {path}")

    best: BestResult | None = None
    for row in rows:
        box_map50 = float(row["metrics/mAP50(B)"])
        box_map = float(row["metrics/mAP50-95(B)"])
        mask_map50 = float(row["metrics/mAP50(M)"])
        mask_map = float(row["metrics/mAP50-95(M)"])
        fitness = ((box_map50 * 0.1 + box_map * 0.9) + (mask_map50 * 0.1 + mask_map * 0.9)) / 2.0
        candidate = BestResult(
            csv_path=path,
            epoch=int(float(row["epoch"])),
            fitness=fitness,
            box_map50=box_map50,
            box_map=box_map,
            mask_map50=mask_map50,
            mask_map=mask_map,
        )
        if best is None or candidate.fitness > best.fitness:
            best = candidate

    assert best is not None
    return best


def print_result(label: str, result: BestResult) -> None:
    print(f"[{label}]")
    print(f"csv: {result.csv_path}")
    print(f"epoch: {result.epoch}")
    print(f"fitness: {result.fitness:.6f}")
    print(f"box:  mAP50={result.box_map50:.5f}, mAP50-95={result.box_map:.5f}")
    print(f"mask: mAP50={result.mask_map50:.5f}, mAP50-95={result.mask_map:.5f}")


def main() -> int:
    args = parse_args()
    baseline = read_best_result(args.baseline)
    candidate = read_best_result(args.candidate)

    print_result("baseline", baseline)
    print_result("candidate", candidate)

    delta = candidate.fitness - baseline.fitness
    print(f"delta: {delta:+.6f}")

    if delta <= 0:
        print("candidate の fitness が baseline を上回っていません。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
