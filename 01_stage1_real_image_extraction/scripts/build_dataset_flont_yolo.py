from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ColorSpec:
    name: str
    class_id: int
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    extra_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = ()


COLOR_SPECS: tuple[ColorSpec, ...] = (
    ColorSpec("R1", 0, (170, 150, 150), (179, 255, 255), (((160, 100, 100), (180, 255, 255)),)),
    ColorSpec("R2", 1, (60, 100, 100), (68, 255, 255)),
    ColorSpec("R3", 2, (24, 150, 150), (28, 255, 255)),
    ColorSpec("R4", 3, (110, 120, 150), (116, 255, 255)),
    ColorSpec("R5", 4, (10, 150, 150), (15, 255, 255)),
    ColorSpec("R6", 5, (140, 150, 100), (146, 255, 255)),
    ColorSpec("R7", 6, (93, 130, 150), (98, 255, 255)),
    ColorSpec("L1", 7, (35, 130, 180), (42, 255, 255)),
    ColorSpec("L2", 8, (0, 50, 180), (5, 100, 255)),
    ColorSpec("L3", 9, (88, 200, 80), (92, 255, 150)),
    ColorSpec("L4", 10, (135, 50, 180), (142, 100, 255)),
    ColorSpec("L5", 11, (15, 150, 100), (18, 255, 180)),
    ColorSpec("L6", 12, (25, 40, 180), (30, 80, 255)),
    ColorSpec("L7", 13, (0, 200, 80), (5, 255, 130)),
)


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description=(
            "90_archive/legacy_preliminary/dataset/*_flont を "
            "YOLO segmentation 形式の dataset_flont に変換します。"
        )
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=repo_root / "90_archive" / "legacy_preliminary" / "dataset" / "train_flont",
        help="旧構成から移した学習用 train_flont ディレクトリ",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=repo_root / "90_archive" / "legacy_preliminary" / "dataset" / "test_flont",
        help="旧構成から移した評価用 test_flont ディレクトリ",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "datasets" / "dataset_flont",
        help="YOLO 形式の出力先ディレクトリ",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="輪郭として採用する最小面積",
    )
    return parser.parse_args()


def relative_png_map(root: Path) -> dict[Path, Path]:
    files = {path.relative_to(root): path for path in root.rglob("*.png")}
    if not files:
        raise RuntimeError(f"PNG が見つかりません: {root}")
    return files


def reset_output_root(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def build_mask(hsv_image: np.ndarray, spec: ColorSpec) -> np.ndarray:
    mask = cv2.inRange(hsv_image, np.array(spec.lower), np.array(spec.upper))
    for lower, upper in spec.extra_ranges:
        mask |= cv2.inRange(hsv_image, np.array(lower), np.array(upper))
    return mask


def contour_to_yolo_line(class_id: int, contour: np.ndarray, width: int, height: int) -> str | None:
    coords = contour.reshape(-1, 2)
    if len(coords) < 3:
        return None

    values: list[str] = [str(class_id)]
    for x, y in coords:
        values.append(f"{x / width:.6f}")
        values.append(f"{y / height:.6f}")
    return " ".join(values)


def validate_pairs(overlay_files: dict[Path, Path], image_files: dict[Path, Path], split_name: str) -> list[Path]:
    overlay_set = set(overlay_files)
    image_set = set(image_files)
    if overlay_set != image_set:
        missing_images = sorted(overlay_set - image_set)
        missing_labels = sorted(image_set - overlay_set)
        details: list[str] = [f"{split_name}: labels_color と images の対応が一致しません。"]
        if missing_images:
            details.append("images 側に不足:")
            details.extend(f"  - {path.as_posix()}" for path in missing_images[:10])
        if missing_labels:
            details.append("labels_color 側に不足:")
            details.extend(f"  - {path.as_posix()}" for path in missing_labels[:10])
        raise RuntimeError("\n".join(details))
    return sorted(overlay_set)


def build_split(split_name: str, split_root: Path, output_root: Path, min_area: float) -> tuple[int, int, Counter[str]]:
    overlay_root = split_root / "labels_color"
    image_root = split_root / "images"
    overlay_files = relative_png_map(overlay_root)
    image_files = relative_png_map(image_root)
    rel_paths = validate_pairs(overlay_files, image_files, split_name)

    instance_counts: Counter[str] = Counter()
    total_instances = 0

    for rel_path in rel_paths:
        overlay_path = overlay_files[rel_path]
        image_path = image_files[rel_path]
        overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if overlay_image is None:
            raise RuntimeError(f"画像を読めません: {overlay_path}")
        if gray_image is None:
            raise RuntimeError(f"画像を読めません: {image_path}")
        if overlay_image.shape[:2] != gray_image.shape[:2]:
            raise RuntimeError(
                f"画像サイズが一致しません: {overlay_path} vs {image_path} "
                f"({overlay_image.shape[:2]} != {gray_image.shape[:2]})"
            )

        height, width = overlay_image.shape[:2]
        hsv = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2HSV)
        yolo_lines: list[str] = []

        for spec in COLOR_SPECS:
            mask = build_mask(hsv, spec)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                line = contour_to_yolo_line(spec.class_id, contour, width, height)
                if line is None:
                    continue
                yolo_lines.append(line)
                instance_counts[spec.name] += 1
                total_instances += 1

        if not yolo_lines:
            raise RuntimeError(f"ラベルが 1 件も生成されませんでした: {overlay_path}")

        safe_name = "_".join(rel_path.parts)
        target_image_path = output_root / split_name / "images" / safe_name
        target_label_path = output_root / split_name / "labels" / f"{Path(safe_name).stem}.txt"

        shutil.copy2(image_path, target_image_path)
        target_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    return len(rel_paths), total_instances, instance_counts


def write_dataset_yaml(output_root: Path) -> Path:
    repo_root = find_project_root(Path(__file__).resolve())
    try:
        dataset_path = output_root.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        dataset_path = output_root.resolve().as_posix()

    dataset_yaml = {
        "path": dataset_path,
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "names": {spec.class_id: spec.name for spec in COLOR_SPECS},
    }
    output_path = output_root / "dataset_flont.yaml"
    output_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    reset_output_root(args.output_root)

    summaries = {
        "train": build_split("train", args.train_root, args.output_root, args.min_area),
        "test": build_split("test", args.test_root, args.output_root, args.min_area),
    }
    yaml_path = write_dataset_yaml(args.output_root)

    print("dataset_flont を再構築しました。")
    for split_name, (image_count, instance_count, class_counts) in summaries.items():
        class_summary = ", ".join(f"{name}={class_counts[name]}" for name in class_counts)
        print(f"- {split_name}: images={image_count}, instances={instance_count}")
        print(f"  classes: {class_summary}")
    print(f"- yaml: {yaml_path}")


if __name__ == "__main__":
    main()
