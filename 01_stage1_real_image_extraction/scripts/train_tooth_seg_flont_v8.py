from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


CLASS_FILTER = [0, 1, 2, 7, 8, 9]
TARGET_CLASS_NAMES = {
    0: "R1",
    1: "R2",
    2: "R3",
    7: "L1",
    8: "L2",
    9: "L3",
}
SUPPORTED_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def blur_limit(value: str) -> int:
    parsed = positive_int(value)
    if parsed < 3 or parsed % 2 == 0:
        raise argparse.ArgumentTypeError("must be an odd integer of at least 3")
    return parsed


def jpeg_quality(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Fine-tune v7_best on a reproducible synthetic and real mixture."
    )
    parser.add_argument(
        "--synthetic-data",
        type=Path,
        required=True,
        help="Synthetic-domain YOLO segmentation dataset YAML.",
    )
    parser.add_argument(
        "--real-data",
        type=Path,
        required=True,
        help="Real-image YOLO segmentation dataset YAML.",
    )
    parser.add_argument(
        "--real-repeat",
        type=positive_int,
        default=2,
        help="Number of times to repeat real train paths in the mixed train split.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root
        / "01_stage1_real_image_extraction"
        / "experiments"
        / "v7_best"
        / "weights"
        / "best.pt",
        help="Initial v7_best checkpoint path.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments",
        help="Directory where training outputs will be written.",
    )
    parser.add_argument("--name", default="v8_real_finetune", help="Experiment name.")
    parser.add_argument("--epochs", type=positive_int, default=40)
    parser.add_argument("--patience", type=positive_int, default=8)
    parser.add_argument("--imgsz", type=positive_int, default=832)
    parser.add_argument("--batch", type=positive_int, default=2)
    parser.add_argument("--workers", type=nonnegative_int, default=4)
    parser.add_argument("--lr0", type=fraction, default=5e-5)
    parser.add_argument("--lrf", type=fraction, default=0.1)
    parser.add_argument("--freeze", type=nonnegative_int, default=16)
    parser.add_argument("--hsv-h", type=fraction, default=0.01)
    parser.add_argument("--hsv-s", type=fraction, default=0.25)
    parser.add_argument("--hsv-v", type=fraction, default=0.2)
    parser.add_argument("--brightness-limit", type=fraction, default=0.15)
    parser.add_argument("--contrast-limit", type=fraction, default=0.15)
    parser.add_argument("--brightness-contrast-prob", type=fraction, default=0.3)
    parser.add_argument("--blur-limit", type=blur_limit, default=5)
    parser.add_argument("--blur-prob", type=fraction, default=0.1)
    parser.add_argument("--jpeg-quality-min", type=jpeg_quality, default=75)
    parser.add_argument("--jpeg-prob", type=fraction, default=0.1)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Defaults to CUDA:0 when available, otherwise cpu.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and print exact kwargs without creating a model.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one actual ML epoch using explicit synthetic-train, real-train, and real-val file lists.",
    )
    return parser.parse_args(argv)


def ensure_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} not found: {resolved}")
    return resolved


def normalize_names(value: object, yaml_path: Path) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(index): str(name) for index, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"dataset names keys must be integers: {yaml_path}") from exc
    raise RuntimeError(f"dataset yaml must define names as a list or mapping: {yaml_path}")


def resolve_dataset_entries(
    root: Path, value: object, split: str, yaml_path: Path
) -> list[Path]:
    entries = [value] if isinstance(value, str) else value
    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(item, str) for item in entries)
    ):
        raise RuntimeError(
            f"dataset '{split}' must be a path or non-empty path list: {yaml_path}"
        )
    paths = [(root / item).resolve() for item in entries]
    missing = next((path for path in paths if not path.exists()), None)
    if missing is not None:
        raise RuntimeError(f"dataset '{split}' path not found: {missing}")
    return paths


def collect_image_paths(entries: list[Path], label: str) -> list[Path]:
    images: list[Path] = []
    for entry in entries:
        if entry.is_dir():
            images.extend(
                path.resolve()
                for path in entry.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            continue
        for line_number, raw_line in enumerate(
            entry.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                raise RuntimeError(f"empty image-list entry: {entry}:{line_number}")
            image_path = (
                (entry.parent / line[2:]).resolve()
                if line.startswith("./")
                else Path(line).expanduser().resolve()
            )
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
            ):
                raise RuntimeError(
                    f"invalid image-list entry: {entry}:{line_number}: {image_path}"
                )
            images.append(image_path)
    images = sorted(images)
    if not images:
        raise RuntimeError(f"{label} contains no supported images")
    if len(images) != len(set(images)):
        raise RuntimeError(f"{label} contains duplicate image paths")
    return images


def validate_dataset_yaml(path: Path, domain: str) -> dict[str, Any]:
    yaml_path = ensure_file(path, f"{domain} dataset yaml")
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"invalid {domain} dataset yaml: {yaml_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"dataset yaml root must be a mapping: {yaml_path}")
    names = normalize_names(payload.get("names"), yaml_path)
    if set(names) != set(range(len(names))):
        raise RuntimeError(f"dataset class IDs must be contiguous from 0: {yaml_path}")
    if "nc" in payload and (
        isinstance(payload["nc"], bool)
        or not isinstance(payload["nc"], int)
        or payload["nc"] != len(names)
    ):
        raise RuntimeError(f"dataset 'nc' must equal names length {len(names)}: {yaml_path}")
    for class_id, expected_name in TARGET_CLASS_NAMES.items():
        if names.get(class_id) != expected_name:
            raise RuntimeError(
                f"dataset class {class_id} must be '{expected_name}', "
                f"got {names.get(class_id)!r}: {yaml_path}"
            )

    root_value = payload.get("path", yaml_path.parent)
    if not isinstance(root_value, (str, Path)):
        raise RuntimeError(f"dataset 'path' must be a path string: {yaml_path}")
    root = Path(root_value).expanduser()
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"dataset root not found: {root}")
    splits = {
        split: resolve_dataset_entries(root, payload.get(split), split, yaml_path)
        for split in ("train", "val")
    }
    images = {
        split: collect_image_paths(paths, f"{domain} {split}")
        for split, paths in splits.items()
    }
    return {
        "path": yaml_path,
        "root": root,
        "names": names,
        "splits": splits,
        "images": images,
    }


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch must be installed to auto-select the training device") from exc
    return "0" if torch.cuda.is_available() else "cpu"


def build_augmentations(args: argparse.Namespace) -> list[object]:
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError("albumentations must be installed before training") from exc
    return [
        A.RandomBrightnessContrast(
            brightness_limit=args.brightness_limit,
            contrast_limit=args.contrast_limit,
            p=args.brightness_contrast_prob,
        ),
        A.Blur(blur_limit=(3, args.blur_limit), p=args.blur_prob),
        A.ImageCompression(
            compression_type="jpeg",
            quality_range=(args.jpeg_quality_min, 100),
            p=args.jpeg_prob,
        ),
    ]


def prepare_training(
    args: argparse.Namespace,
    *,
    device_resolver: Callable[[str | None], str] = resolve_device,
) -> dict[str, Any]:
    synthetic = validate_dataset_yaml(args.synthetic_data, "synthetic")
    real = validate_dataset_yaml(args.real_data, "real")
    if synthetic["names"] != real["names"]:
        raise RuntimeError("synthetic and real dataset class order must match exactly")
    weights_path = ensure_file(args.weights, "initial checkpoint")
    if not SAFE_NAME_PATTERN.fullmatch(args.name):
        raise RuntimeError(f"invalid experiment name: {args.name!r}")

    project_path = args.project.expanduser().resolve()
    if project_path.exists() and not project_path.is_dir():
        raise RuntimeError(f"project path is not a directory: {project_path}")
    run_name = f"{args.name}_dry_run" if args.dry_run else args.name
    output_dir = project_path / run_name
    if output_dir.exists():
        raise RuntimeError(f"output already exists: {output_dir}")

    synthetic_train = [str(path) for path in synthetic["splits"]["train"]]
    real_train = [str(path) for path in real["splits"]["train"]]
    real_val = [str(path) for path in real["splits"]["val"]]
    mixed_dataset = {
        "path": "/",
        "train": synthetic_train + real_train * args.real_repeat,
        "val": real_val[0] if len(real_val) == 1 else real_val,
        "names": synthetic["names"],
    }
    synthetic_count = len(synthetic["images"]["train"])
    real_count = len(real["images"]["train"])
    effective_real_count = real_count * args.real_repeat
    effective_total = synthetic_count + effective_real_count
    mixing = {
        "synthetic_data": str(synthetic["path"]),
        "real_data": str(real["path"]),
        "real_repeat": args.real_repeat,
        "synthetic_train_images": synthetic_count,
        "real_train_images": real_count,
        "effective_real_train_images": effective_real_count,
        "effective_total_train_images": effective_total,
        "effective_real_fraction": effective_real_count / effective_total,
        "real_val_images": len(real["images"]["val"]),
    }
    dry_run_selection = None
    if args.dry_run:
        dry_run_selection = {
            "synthetic_train": synthetic["images"]["train"][:1],
            "real_train": real["images"]["train"][:1],
            "real_val": real["images"]["val"][:1],
        }
        mixing["dry_run"] = {
            key: [str(path) for path in paths]
            for key, paths in dry_run_selection.items()
        }

    device = device_resolver(args.device)
    kwargs = {
        "epochs": 1 if args.dry_run else args.epochs,
        "imgsz": min(args.imgsz, 640) if args.dry_run else args.imgsz,
        "batch": 1 if args.dry_run else args.batch,
        "device": device,
        "workers": args.workers,
        "project": str(project_path),
        "name": run_name,
        "patience": 1 if args.dry_run else args.patience,
        "classes": CLASS_FILTER.copy(),
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "augmentations": build_augmentations(args),
        "degrees": 0.0,
        "translate": 0.02,
        "scale": 0.1,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "freeze": args.freeze,
        "cos_lr": True,
        "cache": False,
        "save": True,
        "plots": not args.dry_run,
        "fraction": 1.0,
        "seed": 0,
        "deterministic": True,
        "exist_ok": False,
    }
    return {
        "weights": weights_path,
        "output_dir": output_dir,
        "kwargs": kwargs,
        "mixed_dataset": mixed_dataset,
        "mixing": mixing,
        "dry_run_selection": dry_run_selection,
    }


def json_ready_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    payload = kwargs.copy()
    payload["augmentations"] = [repr(transform) for transform in kwargs["augmentations"]]
    return payload


def load_yolo_factory() -> Callable[[str], object]:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "ultralytics")
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics must be installed before training") from exc
    return YOLO


def validate_training_outputs(output_dir: Path) -> tuple[Path, Path]:
    results_csv = output_dir / "results.csv"
    if not results_csv.is_file():
        raise RuntimeError(f"results.csv was not found after training: {results_csv}")
    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise RuntimeError(f"results.csv has no epoch column: {results_csv}")
        if not list(reader):
            raise RuntimeError(f"results.csv is empty: {results_csv}")
    best_weights = output_dir / "weights" / "best.pt"
    if not best_weights.is_file() or best_weights.stat().st_size == 0:
        raise RuntimeError(f"best checkpoint was not found after training: {best_weights}")
    return results_csv, best_weights


def write_image_list(path: Path, image_paths: list[Path]) -> None:
    if not image_paths:
        raise RuntimeError(f"image list must contain at least one image: {path}")
    path.write_text("".join(f"{image}\n" for image in image_paths), encoding="utf-8")


def materialize_runtime_dataset(
    prepared: dict[str, Any], directory: Path
) -> dict[str, Any]:
    selection = prepared["dry_run_selection"]
    if selection is None:
        return prepared["mixed_dataset"]
    train_list = directory / "dry_run_train.txt"
    val_list = directory / "dry_run_val.txt"
    write_image_list(
        train_list, selection["synthetic_train"] + selection["real_train"]
    )
    write_image_list(val_list, selection["real_val"])
    return {
        "path": "/",
        "train": str(train_list),
        "val": str(val_list),
        "names": prepared["mixed_dataset"]["names"],
    }


def persist_mixed_dataset(prepared: dict[str, Any], train_kwargs: dict[str, Any]) -> None:
    output_dir = prepared["output_dir"]
    persisted_dataset = prepared["mixed_dataset"]
    selection = prepared["dry_run_selection"]
    if selection is not None:
        train_list = output_dir / "dry_run_train.txt"
        val_list = output_dir / "dry_run_val.txt"
        write_image_list(
            train_list, selection["synthetic_train"] + selection["real_train"]
        )
        write_image_list(val_list, selection["real_val"])
        persisted_dataset = {
            "path": "/",
            "train": str(train_list),
            "val": str(val_list),
            "names": prepared["mixed_dataset"]["names"],
        }
    persisted_yaml = output_dir / "mixed_dataset.yaml"
    persisted_yaml.write_text(
        yaml.safe_dump(persisted_dataset, sort_keys=False), encoding="utf-8"
    )
    persisted_kwargs = train_kwargs.copy()
    persisted_kwargs["data"] = str(persisted_yaml)
    manifest = {
        "mixed_dataset": persisted_dataset,
        "mixing": prepared["mixing"],
        "train_kwargs": json_ready_kwargs(persisted_kwargs),
    }
    (output_dir / "mixed_dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main(
    argv: list[str] | None = None,
    *,
    yolo_factory: Callable[[str], object] | None = None,
    device_resolver: Callable[[str | None], str] = resolve_device,
) -> None:
    args = parse_args(argv)
    prepared = prepare_training(args, device_resolver=device_resolver)
    with tempfile.TemporaryDirectory(prefix="odonto_v8_mixed_") as temp_dir:
        temp_path = Path(temp_dir)
        runtime_dataset = materialize_runtime_dataset(prepared, temp_path)
        mixed_yaml_path = temp_path / "mixed_dataset.yaml"
        mixed_yaml_path.write_text(
            yaml.safe_dump(runtime_dataset, sort_keys=False),
            encoding="utf-8",
        )
        train_kwargs = {"data": str(mixed_yaml_path), **prepared["kwargs"]}
        payload = {
            "mode": "prepare-only"
            if args.prepare_only
            else ("dry-run" if args.dry_run else "train"),
            "weights": str(prepared["weights"]),
            "output_dir": str(prepared["output_dir"]),
            "mixed_dataset": runtime_dataset,
            "mixing": prepared["mixing"],
            "train_kwargs": json_ready_kwargs(train_kwargs),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.prepare_only:
            return

        factory = yolo_factory or load_yolo_factory()
        model = factory(str(prepared["weights"]))
        model.train(**train_kwargs)
        validate_training_outputs(prepared["output_dir"])
        persist_mixed_dataset(prepared, train_kwargs)


if __name__ == "__main__":
    main()
