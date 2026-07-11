from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
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
TAG_COLUMNS = ("view_tag", "lighting_tag", "oral_condition_tag")
IDENTITY_COLUMNS = ("patient_token", "checkup_token", "source_sha256")
REQUIRED_METADATA_COLUMNS = frozenset(
    ("split", "image_name", *IDENTITY_COLUMNS, *TAG_COLUMNS)
)
SUPPORTED_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Compare v7 and v8 on one shared real-image validation split."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the real-image YOLO segmentation dataset YAML.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=WEIGHTS",
        help="Labeled checkpoint; repeat for baseline and candidate.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Finalizer metadata.csv used for strict val-set and condition evaluation.",
    )
    parser.add_argument("--baseline-label", default="v7_zero_shot")
    parser.add_argument("--candidate-label", default="v8")
    parser.add_argument(
        "--project",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments",
        help="Directory where validation outputs will be written.",
    )
    parser.add_argument("--name", default="real_val_comparison")
    parser.add_argument("--imgsz", type=positive_int, default=832)
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and print all planned val kwargs without creating models or files.",
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


def validate_dataset_yaml(path: Path) -> dict[str, Any]:
    yaml_path = ensure_file(path, "dataset yaml")
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"invalid dataset yaml: {yaml_path}") from exc
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
        actual_name = names.get(class_id)
        if actual_name != expected_name:
            raise RuntimeError(
                f"dataset class {class_id} must be '{expected_name}', "
                f"got {actual_name!r}: {yaml_path}"
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
    return {"path": yaml_path, "root": root, "names": names, "splits": splits}


def collect_image_paths(entries: list[Path]) -> list[Path]:
    images: list[Path] = []
    for entry in entries:
        if entry.is_dir():
            images.extend(
                path.resolve()
                for path in entry.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            continue

        lines = entry.read_text(encoding="utf-8").strip().splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
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
        raise RuntimeError("real validation split contains no supported images")
    if len(images) != len(set(images)):
        raise RuntimeError("real validation split contains duplicate image paths")
    return images


def condition_id(tag: str, value: str) -> str:
    digest = hashlib.sha256(f"{tag}\0{value}".encode()).hexdigest()[:12]
    return f"{tag.removesuffix('_tag')}-{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    metadata_path = ensure_file(path, "metadata csv")
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"metadata csv has no header: {metadata_path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"metadata csv has duplicate columns: {metadata_path}")
        missing_columns = sorted(REQUIRED_METADATA_COLUMNS - set(reader.fieldnames))
        if missing_columns:
            raise RuntimeError(
                f"metadata csv missing columns {missing_columns}: {metadata_path}"
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"metadata csv is empty: {metadata_path}")

    seen: set[tuple[str, str]] = set()
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for row_number, row in enumerate(rows, start=2):
        for column in REQUIRED_METADATA_COLUMNS:
            if row[column] is None or not row[column].strip():
                raise RuntimeError(
                    f"metadata value is empty: {metadata_path}:{row_number}:{column}"
                )
            row[column] = row[column].strip()
        if row["split"] not in {"train", "val"}:
            raise RuntimeError(
                f"metadata split must be train or val: {metadata_path}:{row_number}"
            )
        image_name = row["image_name"]
        if Path(image_name).name != image_name:
            raise RuntimeError(
                f"metadata image_name must be a basename: {metadata_path}:{row_number}"
            )
        if not SHA256_PATTERN.fullmatch(row["source_sha256"]):
            raise RuntimeError(
                f"metadata source_sha256 must be 64 lowercase hex characters: "
                f"{metadata_path}:{row_number}"
            )
        identity = (row["split"], image_name)
        if identity in seen:
            raise RuntimeError(
                f"duplicate metadata row for {identity}: {metadata_path}:{row_number}"
            )
        seen.add(identity)
        rows_by_split[row["split"]].append(row)

    image_paths_by_split: dict[str, dict[str, Path]] = {}
    for split in ("train", "val"):
        split_images = collect_image_paths(dataset["splits"][split])
        paths_by_name: dict[str, Path] = {}
        for image_path in split_images:
            if image_path.name in paths_by_name:
                raise RuntimeError(
                    f"real {split} image basenames are not unique: {image_path.name}"
                )
            paths_by_name[image_path.name] = image_path
        metadata_names = {row["image_name"] for row in rows_by_split[split]}
        image_names = set(paths_by_name)
        if metadata_names != image_names:
            missing = sorted(image_names - metadata_names)
            extra = sorted(metadata_names - image_names)
            raise RuntimeError(
                f"metadata {split} image set does not match dataset {split}; "
                f"missing={missing}, extra={extra}"
            )
        image_paths_by_split[split] = paths_by_name

    for split in ("train", "val"):
        for row in rows_by_split[split]:
            image_path = image_paths_by_split[split][row["image_name"]]
            actual_sha256 = sha256_file(image_path)
            if row["source_sha256"] != actual_sha256:
                raise RuntimeError(
                    f"metadata source_sha256 does not match image: "
                    f"split={split}, image_name={row['image_name']}, "
                    f"metadata={row['source_sha256']}, actual={actual_sha256}"
                )

    for column in IDENTITY_COLUMNS:
        train_values = {row[column] for row in rows_by_split["train"]}
        val_values = {row[column] for row in rows_by_split["val"]}
        overlap = sorted(train_values & val_values)
        if overlap:
            raise RuntimeError(
                f"metadata {column} overlaps train and val: {overlap}"
            )

    val_rows = rows_by_split["val"]
    image_paths_by_name = image_paths_by_split["val"]

    conditions = []
    tags: dict[str, list[dict[str, Any]]] = {}
    for tag in TAG_COLUMNS:
        groups: list[dict[str, Any]] = []
        for value in sorted({row[tag] for row in val_rows}):
            names = sorted(row["image_name"] for row in val_rows if row[tag] == value)
            group = {
                "id": condition_id(tag, value),
                "tag": tag,
                "value": value,
                "image_count": len(names),
                "image_names": names,
                "image_paths": [image_paths_by_name[name] for name in names],
            }
            groups.append(group)
            conditions.append(group)
        tags[tag] = groups
    summary = {
        "path": str(metadata_path),
        "train_image_count": len(image_paths_by_split["train"]),
        "train_image_names": sorted(image_paths_by_split["train"]),
        "val_image_count": len(image_paths_by_split["val"]),
        "val_image_names": sorted(image_paths_by_split["val"]),
        "tags": {
            tag: [
                {
                    "value": group["value"],
                    "image_count": group["image_count"],
                    "image_names": group["image_names"],
                }
                for group in groups
            ]
            for tag, groups in tags.items()
        },
    }
    return {"path": metadata_path, "summary": summary, "conditions": conditions}


def parse_model_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not raw_path or not SAFE_NAME_PATTERN.fullmatch(label):
            raise RuntimeError(
                f"model must use LABEL=WEIGHTS with a safe label: {value!r}"
            )
        if label in labels:
            raise RuntimeError(f"duplicate model label: {label}")
        labels.add(label)
        specs.append((label, ensure_file(Path(raw_path), f"{label} checkpoint")))
    return specs


def build_val_kwargs(
    args: argparse.Namespace,
    *,
    data_path: Path,
    project_path: Path,
    label: str,
) -> dict[str, Any]:
    return {
        "data": str(data_path),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "classes": CLASS_FILTER.copy(),
        "project": str(project_path),
        "name": label,
        "exist_ok": False,
        "plots": True,
        "save": False,
        "verbose": False,
    }


def prepare_validation(args: argparse.Namespace) -> dict[str, Any]:
    dataset = validate_dataset_yaml(args.data)
    if not SAFE_NAME_PATTERN.fullmatch(args.name):
        raise RuntimeError(f"invalid output name: {args.name!r}")
    if args.baseline_label == args.candidate_label:
        raise RuntimeError("baseline and candidate labels must differ")
    project_path = args.project.expanduser().resolve()
    if project_path.exists() and not project_path.is_dir():
        raise RuntimeError(f"project path is not a directory: {project_path}")
    output_dir = project_path / args.name
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise RuntimeError(f"output path must be a regular directory: {output_dir}")
    backup = output_dir.parent / f".{output_dir.name}.previous"
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"stale output backup requires recovery: {backup}")

    model_specs = parse_model_specs(args.model)
    model_labels = {label for label, _ in model_specs}
    for role, label in (
        ("baseline", args.baseline_label),
        ("candidate", args.candidate_label),
    ):
        if label not in model_labels:
            raise RuntimeError(f"{role} label is not present in --model: {label}")

    models = [
        {
            "label": label,
            "weights": weights,
            "kwargs": build_val_kwargs(
                args,
                data_path=dataset["path"],
                project_path=output_dir / "overall_runs",
                label=label,
            ),
        }
        for label, weights in model_specs
    ]

    metadata = load_metadata(args.metadata, dataset) if args.metadata is not None else None
    conditions = []
    if metadata is not None:
        for group in metadata["conditions"]:
            condition_dir = output_dir / "condition_datasets" / group["id"]
            dataset_yaml = condition_dir / "dataset.yaml"
            conditions.append(
                {
                    **group,
                    "dataset_yaml": dataset_yaml,
                    "file_list": condition_dir / "images.txt",
                    "models": [
                        {
                            "label": label,
                            "kwargs": build_val_kwargs(
                                args,
                                data_path=dataset_yaml,
                                project_path=output_dir
                                / "condition_runs"
                                / group["id"],
                                label=label,
                            ),
                        }
                        for label, _ in model_specs
                    ],
                }
            )
    return {
        "dataset": dataset,
        "output_dir": output_dir,
        "models": models,
        "metadata": metadata,
        "conditions": conditions,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
    }


def backup_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.previous"


def create_generation_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.generation_", dir=output_dir.parent
        )
    )


def discard_generation(generation_dir: Path) -> None:
    if generation_dir.exists():
        shutil.rmtree(generation_dir)


def publish_generation(generation_dir: Path, output_dir: Path) -> None:
    if not generation_dir.is_dir() or generation_dir.is_symlink():
        raise RuntimeError(f"generation path must be a regular directory: {generation_dir}")
    if generation_dir.parent.resolve() != output_dir.parent.resolve():
        raise RuntimeError("generation and output directories must share the same parent")
    backup = backup_path(output_dir)
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"output backup already exists: {backup}")

    had_previous = output_dir.exists()
    if had_previous:
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise RuntimeError(f"output path must be a regular directory: {output_dir}")
        os.replace(output_dir, backup)
    try:
        os.replace(generation_dir, output_dir)
    except OSError as publish_error:
        if had_previous:
            try:
                os.replace(backup, output_dir)
            except OSError as restore_error:
                raise RuntimeError(
                    f"failed to publish {generation_dir} and restore {backup}"
                ) from restore_error
        raise RuntimeError(f"failed to publish generation: {generation_dir}") from publish_error
    if had_previous:
        shutil.rmtree(backup)


def retarget_to_generation(prepared: dict[str, Any], generation_dir: Path) -> None:
    prepared["generation_dir"] = generation_dir
    for model in prepared["models"]:
        model["kwargs"]["project"] = str(generation_dir / "overall_runs")
    for condition in prepared["conditions"]:
        condition_dir = generation_dir / "condition_datasets" / condition["id"]
        condition["dataset_yaml"] = condition_dir / "dataset.yaml"
        condition["file_list"] = condition_dir / "images.txt"
        condition["public_dataset_yaml"] = (
            prepared["output_dir"]
            / "condition_datasets"
            / condition["id"]
            / "dataset.yaml"
        )
        for model in condition["models"]:
            model["kwargs"]["data"] = str(condition["dataset_yaml"])
            model["kwargs"]["project"] = str(
                generation_dir / "condition_runs" / condition["id"]
            )


def public_generation_path(
    generated_path: Path, generation_dir: Path, output_dir: Path
) -> Path:
    resolved_generation = generation_dir.resolve()
    resolved_path = generated_path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_generation)
    except ValueError as exc:
        raise RuntimeError(
            f"generated path is outside the generation directory: {resolved_path}"
        ) from exc
    return output_dir / relative


def materialize_condition_datasets(prepared: dict[str, Any]) -> None:
    names = prepared["dataset"]["names"]
    for condition in prepared["conditions"]:
        condition_dir = condition["dataset_yaml"].parent
        condition_dir.mkdir(parents=True)
        condition["file_list"].write_text(
            "".join(f"{path}\n" for path in condition["image_paths"]),
            encoding="utf-8",
        )
        payload = {
            "path": str(condition_dir),
            "train": str(condition["file_list"]),
            "val": str(condition["file_list"]),
            "names": names,
        }
        condition["dataset_yaml"].write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def rewrite_condition_datasets_for_publication(prepared: dict[str, Any]) -> None:
    names = prepared["dataset"]["names"]
    for condition in prepared["conditions"]:
        public_yaml = condition["public_dataset_yaml"]
        public_file_list = public_yaml.parent / "images.txt"
        payload = {
            "path": str(public_yaml.parent),
            "train": str(public_file_list),
            "val": str(public_file_list),
            "names": names,
        }
        condition["dataset_yaml"].write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def checked_metric(value: object, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise RuntimeError(f"metric must be finite and between 0 and 1: {label}={parsed}")
    return parsed


def extract_metrics(results_dict: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key not in results_dict:
            raise RuntimeError(f"validation result is missing metric: {key}")
        metrics[key] = checked_metric(results_dict[key], key)
    return metrics


def extract_component(metric: object, class_id: int, component: str) -> dict[str, float] | None:
    class_indices = [int(value) for value in metric.ap_class_index]
    if class_id not in class_indices:
        return None
    result_index = class_indices.index(class_id)
    values = metric.class_result(result_index)
    if len(values) != 4:
        raise RuntimeError(f"unexpected {component} class_result length: {len(values)}")
    return {
        "precision": checked_metric(values[0], f"{component}.precision[{class_id}]"),
        "recall": checked_metric(values[1], f"{component}.recall[{class_id}]"),
        "map50": checked_metric(values[2], f"{component}.map50[{class_id}]"),
        "map50_95": checked_metric(values[3], f"{component}.map50_95[{class_id}]"),
    }


def extract_per_class_metrics(
    val_result: object, *, require_all: bool
) -> list[dict[str, Any]]:
    result_names = normalize_names(val_result.names, Path("val_result.names"))
    missing_names = {
        class_id: expected
        for class_id, expected in TARGET_CLASS_NAMES.items()
        if result_names.get(class_id) != expected
    }
    if missing_names:
        raise RuntimeError(f"validation result class names do not match: {missing_names}")

    rows = []
    missing_classes = []
    for class_id in CLASS_FILTER:
        box = extract_component(val_result.box, class_id, "box")
        mask = extract_component(val_result.seg, class_id, "mask")
        if (box is None) != (mask is None):
            raise RuntimeError(
                f"box/mask evaluated class mismatch for class {class_id}"
            )
        has_ground_truth = box is not None
        if not has_ground_truth:
            missing_classes.append(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": TARGET_CLASS_NAMES[class_id],
                "has_ground_truth": has_ground_truth,
                "box": box,
                "mask": mask,
            }
        )
    if require_all and missing_classes:
        raise RuntimeError(
            f"full validation lacks target classes required for comparison: {missing_classes}"
        )
    return rows


def compute_fitness(metrics: dict[str, float]) -> float:
    box = metrics["metrics/mAP50(B)"] * 0.1 + metrics["metrics/mAP50-95(B)"] * 0.9
    mask = metrics["metrics/mAP50(M)"] * 0.1 + metrics["metrics/mAP50-95(M)"] * 0.9
    return (box + mask) / 2.0


def build_result(
    model: dict[str, Any],
    val_result: object,
    *,
    require_all_classes: bool,
    generation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    metrics = extract_metrics(val_result.results_dict)
    public_save_dir = public_generation_path(
        Path(val_result.save_dir), generation_dir, output_dir
    )
    return {
        "model": model["label"],
        "weights": str(model["weights"]),
        "save_dir": str(public_save_dir),
        "metrics": metrics,
        "fitness": compute_fitness(metrics),
        "per_class": extract_per_class_metrics(
            val_result, require_all=require_all_classes
        ),
    }


def compare_results(
    results: list[dict[str, Any]], baseline_label: str, candidate_label: str
) -> dict[str, Any]:
    results_by_model = {result["model"]: result for result in results}
    baseline = results_by_model[baseline_label]
    candidate = results_by_model[candidate_label]
    baseline_classes = {row["class_id"]: row for row in baseline["per_class"]}
    candidate_classes = {row["class_id"]: row for row in candidate["per_class"]}

    rows = []
    for class_id in CLASS_FILTER:
        baseline_row = baseline_classes[class_id]
        candidate_row = candidate_classes[class_id]
        evaluable = (
            baseline_row["has_ground_truth"] and candidate_row["has_ground_truth"]
        )
        box_delta = (
            candidate_row["box"]["map50"] - baseline_row["box"]["map50"]
            if evaluable
            else None
        )
        mask_delta = (
            candidate_row["mask"]["map50"] - baseline_row["mask"]["map50"]
            if evaluable
            else None
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": TARGET_CLASS_NAMES[class_id],
                "evaluable": evaluable,
                "baseline_box_map50": baseline_row["box"]["map50"]
                if evaluable
                else None,
                "candidate_box_map50": candidate_row["box"]["map50"]
                if evaluable
                else None,
                "box_map50_delta": box_delta,
                "box_map50_improved": box_delta > 0 if box_delta is not None else False,
                "baseline_mask_map50": baseline_row["mask"]["map50"]
                if evaluable
                else None,
                "candidate_mask_map50": candidate_row["mask"]["map50"]
                if evaluable
                else None,
                "mask_map50_delta": mask_delta,
                "mask_map50_improved": mask_delta > 0
                if mask_delta is not None
                else False,
            }
        )
    all_six_evaluable = all(row["evaluable"] for row in rows)
    return {
        "baseline_model": baseline_label,
        "candidate_model": candidate_label,
        "classes": rows,
        "all_six_evaluable": all_six_evaluable,
        "all_six_box_map50_improved": all_six_evaluable
        and all(row["box_map50_improved"] for row in rows),
        "all_six_mask_map50_improved": all_six_evaluable
        and all(row["mask_map50_improved"] for row in rows),
    }


def write_per_class_metrics(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "weights",
        "class_id",
        "class_name",
        "has_ground_truth",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map50_95",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result["per_class"]:
                box = row["box"] or {}
                mask = row["mask"] or {}
                writer.writerow(
                    {
                        "model": result["model"],
                        "weights": result["weights"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "has_ground_truth": row["has_ground_truth"],
                        **{f"box_{key}": value for key, value in box.items()},
                        **{f"mask_{key}": value for key, value in mask.items()},
                    }
                )


def write_comparison(path: Path, comparison: dict[str, Any]) -> None:
    fieldnames = [
        "class_id",
        "class_name",
        "evaluable",
        "baseline_box_map50",
        "candidate_box_map50",
        "box_map50_delta",
        "box_map50_improved",
        "baseline_mask_map50",
        "candidate_mask_map50",
        "mask_map50_delta",
        "mask_map50_improved",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison["classes"])


def write_overall_results(
    output_dir: Path,
    prepared: dict[str, Any],
    results: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": {
            "data": str(prepared["dataset"]["path"]),
            "split": "val",
            "classes": CLASS_FILTER,
        },
        "metadata": prepared["metadata"]["summary"]
        if prepared["metadata"] is not None
        else None,
        "results": results,
        "comparison": comparison,
    }
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    fieldnames = ["model", "weights", "save_dir", *METRIC_KEYS, "fitness"]
    with (output_dir / "overall_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result["model"],
                    "weights": result["weights"],
                    "save_dir": result["save_dir"],
                    **{key: f"{result['metrics'][key]:.6f}" for key in METRIC_KEYS},
                    "fitness": f"{result['fitness']:.6f}",
                }
            )
    write_per_class_metrics(output_dir / "per_class_metrics.csv", results)
    write_comparison(output_dir / "per_class_comparison.csv", comparison)


def write_metadata_outputs(output_dir: Path, prepared: dict[str, Any]) -> None:
    if prepared["metadata"] is None:
        return
    summary = prepared["metadata"]["summary"]
    (output_dir / "metadata_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "metadata_groups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["tag", "value", "image_count", "image_names"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tag, groups in summary["tags"].items():
            for group in groups:
                writer.writerow(
                    {
                        "tag": tag,
                        "value": group["value"],
                        "image_count": group["image_count"],
                        "image_names": json.dumps(group["image_names"]),
                    }
                )


def write_condition_results(
    output_dir: Path,
    prepared: dict[str, Any],
    condition_results: dict[str, list[dict[str, Any]]],
) -> None:
    if not prepared["conditions"]:
        return
    conditions = []
    for condition in prepared["conditions"]:
        results = condition_results[condition["id"]]
        conditions.append(
            {
                "id": condition["id"],
                "tag": condition["tag"],
                "value": condition["value"],
                "image_count": condition["image_count"],
                "image_names": condition["image_names"],
                "dataset_yaml": str(condition["public_dataset_yaml"]),
                "results": results,
                "comparison": compare_results(
                    results,
                    prepared["baseline_label"],
                    prepared["candidate_label"],
                ),
            }
        )
    (output_dir / "condition_metrics.json").write_text(
        json.dumps({"conditions": conditions}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fieldnames = [
        "condition_id",
        "tag",
        "value",
        "image_count",
        "model",
        *METRIC_KEYS,
        "fitness",
    ]
    with (output_dir / "condition_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in conditions:
            for result in condition["results"]:
                writer.writerow(
                    {
                        "condition_id": condition["id"],
                        "tag": condition["tag"],
                        "value": condition["value"],
                        "image_count": condition["image_count"],
                        "model": result["model"],
                        **{
                            key: f"{result['metrics'][key]:.6f}"
                            for key in METRIC_KEYS
                        },
                        "fitness": f"{result['fitness']:.6f}",
                    }
                )


def load_yolo_factory() -> Callable[[str], object]:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "ultralytics")
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics must be installed before validation") from exc
    return YOLO


def preflight_payload(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "prepare-only" if args.prepare_only else "validate",
        "output_dir": str(prepared["output_dir"]),
        "baseline_label": prepared["baseline_label"],
        "candidate_label": prepared["candidate_label"],
        "metadata": prepared["metadata"]["summary"]
        if prepared["metadata"] is not None
        else None,
        "models": [
            {
                "label": item["label"],
                "weights": str(item["weights"]),
                "val_kwargs": item["kwargs"],
            }
            for item in prepared["models"]
        ],
        "conditions": [
            {
                "id": condition["id"],
                "tag": condition["tag"],
                "value": condition["value"],
                "image_count": condition["image_count"],
                "image_names": condition["image_names"],
                "dataset_yaml": str(condition["dataset_yaml"]),
                "models": condition["models"],
            }
            for condition in prepared["conditions"]
        ],
    }


def main(
    argv: list[str] | None = None,
    *,
    yolo_factory: Callable[[str], object] | None = None,
) -> None:
    args = parse_args(argv)
    prepared = prepare_validation(args)
    generation_dir = create_generation_directory(prepared["output_dir"])
    try:
        retarget_to_generation(prepared, generation_dir)
        print(json.dumps(preflight_payload(args, prepared), indent=2, sort_keys=True))
        if args.prepare_only:
            return

        materialize_condition_datasets(prepared)
        factory = yolo_factory or load_yolo_factory()
        overall_results = []
        condition_results = {
            condition["id"]: [] for condition in prepared["conditions"]
        }
        for model_spec in prepared["models"]:
            model = factory(str(model_spec["weights"]))
            overall_val = model.val(**model_spec["kwargs"])
            overall_results.append(
                build_result(
                    model_spec,
                    overall_val,
                    require_all_classes=True,
                    generation_dir=generation_dir,
                    output_dir=prepared["output_dir"],
                )
            )
            for condition in prepared["conditions"]:
                condition_model = next(
                    item
                    for item in condition["models"]
                    if item["label"] == model_spec["label"]
                )
                condition_val = model.val(**condition_model["kwargs"])
                condition_results[condition["id"]].append(
                    build_result(
                        model_spec,
                        condition_val,
                        require_all_classes=False,
                        generation_dir=generation_dir,
                        output_dir=prepared["output_dir"],
                    )
                )

        rewrite_condition_datasets_for_publication(prepared)
        comparison = compare_results(
            overall_results,
            prepared["baseline_label"],
            prepared["candidate_label"],
        )
        write_overall_results(
            generation_dir, prepared, overall_results, comparison
        )
        write_metadata_outputs(generation_dir, prepared)
        write_condition_results(generation_dir, prepared, condition_results)
        publish_generation(generation_dir, prepared["output_dir"])
    finally:
        discard_generation(generation_dir)


if __name__ == "__main__":
    main()
