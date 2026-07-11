from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_authentication import (
    ScoreRecord,
    build_curve,
    compute_auc,
    compute_score_distribution,
    write_score_distribution_plot,
)
from extract_code_features import (
    CropRecord,
    build_hog,
    build_normalized_crop,
    extract_hog_batch,
)
from matching_core import TOOTH_NAMES, score_pair
from output_directory import (
    create_generation_directory,
    discard_generation,
    publish_generation,
)


FDI_TO_TOOTH_INDEX = {11: 0, 12: 1, 13: 2, 21: 3, 22: 4, 23: 5}
REQUIRED_MANIFEST_COLUMNS = (
    "patient_id",
    "case_id",
    "jaw",
    "view_id",
    "azimuth_deg",
    "elevation_deg",
    "camera_position",
    "focal_point",
    "view_up",
    "parallel_scale",
    "image_width",
    "image_height",
    "image_path",
    "label_path",
    "source_path",
    "source_sha256",
)
SCORE_COLUMNS = (
    "query_id",
    "template_id",
    "query_subject_id",
    "template_subject_id",
    "query_session_id",
    "template_session_id",
    "is_genuine",
    "fused_score",
    "pair_id",
    "common_teeth",
    "common_tooth_names",
    "per_tooth_scores",
    "query_image_path",
    "template_image_path",
)
SKIPPED_COLUMNS = (
    "pair_id",
    "is_genuine",
    "query_id",
    "template_id",
    "query_image_path",
    "template_image_path",
    "common_teeth",
    "reason",
)
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class RenderedView:
    patient_id: str
    case_id: str
    view_id: str
    image_reference: str
    label_reference: str
    source_sha256: str
    image: np.ndarray
    labels: np.ndarray

    @property
    def view_key(self) -> str:
        return f"{self.case_id}/{self.view_id}"


@dataclass(frozen=True)
class ViewFeatures:
    embeddings: np.ndarray
    present: np.ndarray


@dataclass(frozen=True)
class ViewPair:
    pair_id: str
    template: RenderedView
    query: RenderedView
    is_genuine: bool


@dataclass(frozen=True)
class ScoredPair:
    pair: ViewPair
    fused_score: float
    per_tooth_scores: dict[str, float]
    common_teeth: tuple[str, ...]


@dataclass(frozen=True)
class SkippedPair:
    pair: ViewPair
    common_teeth: int
    reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HOG matching across rendered views without model inference."
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="Output root created by render_teeth3ds_views.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--min-common-teeth", type=int, default=1)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.render_root.is_dir() or args.render_root.is_symlink():
        raise RuntimeError(
            f"render root must be a regular directory: {args.render_root}"
        )
    render_root = args.render_root.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=False)
    if render_root.is_relative_to(output_dir):
        raise RuntimeError(
            "--output-dir must not be the render root or one of its ancestors: "
            f"render_root={args.render_root}, output_dir={args.output_dir}"
        )
    if not isinstance(args.crop_size, int) or isinstance(args.crop_size, bool):
        raise RuntimeError("--crop-size must be an integer.")
    if args.crop_size <= 0:
        raise RuntimeError("--crop-size must be positive.")
    if not math.isfinite(args.crop_padding) or args.crop_padding < 0.0:
        raise RuntimeError("--crop-padding must be finite and non-negative.")
    if (
        not isinstance(args.min_common_teeth, int)
        or isinstance(args.min_common_teeth, bool)
        or not 1 <= args.min_common_teeth <= len(TOOTH_NAMES)
    ):
        raise RuntimeError(
            f"--min-common-teeth must be an integer from 1 to {len(TOOTH_NAMES)}."
        )


def clean_required(row: Mapping[str, str | None], column: str, row_number: int) -> str:
    value = row[column]
    if value is None:
        raise RuntimeError(f"{column} is missing at manifest row {row_number}.")
    cleaned = value.strip()
    if cleaned == "":
        raise RuntimeError(f"{column} is empty at manifest row {row_number}.")
    return cleaned


def parse_finite_float(value: str, column: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(
            f"{column} must be numeric at manifest row {row_number}: {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise RuntimeError(
            f"{column} must be finite at manifest row {row_number}: {value!r}"
        )
    return parsed


def parse_positive_integer(value: str, column: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{column} must be an integer at manifest row {row_number}: {value!r}"
        ) from error
    if parsed <= 0 or str(parsed) != value:
        raise RuntimeError(
            f"{column} must be a canonical positive integer at manifest row "
            f"{row_number}: {value!r}"
        )
    return parsed


def parse_vector(value: str, column: str, row_number: int) -> tuple[float, float, float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{column} must be a JSON vector at manifest row {row_number}."
        ) from error
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in parsed)
        or any(not math.isfinite(float(item)) for item in parsed)
    ):
        raise RuntimeError(
            f"{column} must contain three finite numbers at manifest row {row_number}."
        )
    return tuple(float(item) for item in parsed)


def resolve_render_path(
    render_root: Path,
    reference: str,
    column: str,
    row_number: int,
) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts or reference in {".", ".."}:
        raise RuntimeError(
            f"{column} must remain within render root at manifest row {row_number}: "
            f"{reference!r}"
        )
    root = render_root.resolve(strict=True)
    try:
        resolved = (render_root / relative).resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"{column} does not reference an existing file at manifest row "
            f"{row_number}: {reference!r}"
        ) from error
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"{column} must remain within render root at manifest row {row_number}: "
            f"{reference!r}"
        )
    if not resolved.is_file():
        raise RuntimeError(
            f"{column} must reference a file at manifest row {row_number}: "
            f"{reference!r}"
        )
    return resolved


def load_png_pair(
    image_path: Path,
    label_path: Path,
    image_reference: str,
    label_reference: str,
    expected_width: int,
    expected_height: int,
    row_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with Image.open(image_path) as image_file:
            if image_file.format != "PNG" or image_file.mode != "RGB":
                raise RuntimeError(
                    f"image_path must be an RGB PNG at manifest row {row_number}: "
                    f"{image_reference!r}"
                )
            image = np.asarray(image_file).copy()
        with Image.open(label_path) as label_file:
            if label_file.format != "PNG" or label_file.mode != "L":
                raise RuntimeError(
                    f"label_path must be a uint8 FDI PNG at manifest row {row_number}: "
                    f"{label_reference!r}"
                )
            labels = np.asarray(label_file).copy()
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to decode PNG files at manifest row {row_number}.") from error

    if image.dtype != np.uint8 or image.shape != (expected_height, expected_width, 3):
        raise RuntimeError(
            f"RGB image shape mismatch at manifest row {row_number}: "
            f"expected={(expected_height, expected_width, 3)}, actual={image.shape}"
        )
    if labels.dtype != np.uint8 or labels.shape != (expected_height, expected_width):
        raise RuntimeError(
            f"RGB/label shape mismatch at manifest row {row_number}: "
            f"image={image.shape[:2]}, label={labels.shape}, "
            f"manifest={(expected_height, expected_width)}"
        )

    unique_labels = {int(value) for value in np.unique(labels)}
    unknown_labels = unique_labels.difference({0, *FDI_TO_TOOTH_INDEX})
    if unknown_labels:
        raise RuntimeError(
            f"label PNG contains unsupported FDI values at manifest row {row_number}: "
            f"{sorted(unknown_labels)}"
        )
    rgb_foreground = np.any(image != 255, axis=2)
    label_foreground = labels != 0
    if not np.array_equal(rgb_foreground, label_foreground):
        raise RuntimeError(
            f"RGB/label foreground mismatch at manifest row {row_number}: "
            f"{image_reference!r}, {label_reference!r}"
        )
    return image, labels


def validate_manifest_header(fieldnames: list[str] | None, manifest_path: Path) -> None:
    if fieldnames is None:
        raise RuntimeError(f"manifest CSV has no header: {manifest_path}")
    duplicates = sorted(name for name, count in Counter(fieldnames).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"manifest CSV has duplicate columns: {duplicates}")
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in fieldnames]
    if missing:
        raise RuntimeError(f"manifest CSV is missing required columns: {missing}")


def load_manifest(render_root: Path) -> list[RenderedView]:
    manifest_path = render_root / "manifest.csv"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"renderer manifest must be a regular file: {manifest_path}")

    views: list[RenderedView] = []
    seen_case_views: dict[tuple[str, str], int] = {}
    case_provenance: dict[str, tuple[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        validate_manifest_header(reader.fieldnames, manifest_path)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise RuntimeError(f"manifest row {row_number} has unexpected extra values.")
            patient_id = clean_required(row, "patient_id", row_number)
            case_id = clean_required(row, "case_id", row_number)
            view_id = clean_required(row, "view_id", row_number)
            jaw = clean_required(row, "jaw", row_number)
            if jaw != "upper":
                raise RuntimeError(
                    f"jaw must be 'upper' for the configured FDI labels at "
                    f"manifest row {row_number}: {jaw!r}"
                )
            clean_required(row, "source_path", row_number)
            for value, column in (
                (patient_id, "patient_id"),
                (case_id, "case_id"),
                (view_id, "view_id"),
            ):
                if not SAFE_ID_PATTERN.fullmatch(value):
                    raise RuntimeError(
                        f"invalid {column} at manifest row {row_number}: {value!r}"
                    )

            case_view = (case_id, view_id)
            if case_view in seen_case_views:
                raise RuntimeError(
                    "duplicate case_id/view_id at manifest row "
                    f"{row_number}; first seen at row {seen_case_views[case_view]}: "
                    f"{case_view}"
                )
            seen_case_views[case_view] = row_number

            source_sha256 = clean_required(row, "source_sha256", row_number).lower()
            if not SHA256_PATTERN.fullmatch(source_sha256):
                raise RuntimeError(
                    f"source_sha256 must contain 64 hexadecimal characters at "
                    f"manifest row {row_number}."
                )
            provenance = (patient_id, source_sha256)
            previous_provenance = case_provenance.setdefault(case_id, provenance)
            if previous_provenance != provenance:
                raise RuntimeError(
                    f"inconsistent patient_id/source_sha256 for case {case_id!r} at "
                    f"manifest row {row_number}."
                )

            parse_finite_float(
                clean_required(row, "azimuth_deg", row_number),
                "azimuth_deg",
                row_number,
            )
            parse_finite_float(
                clean_required(row, "elevation_deg", row_number),
                "elevation_deg",
                row_number,
            )
            for column in ("camera_position", "focal_point", "view_up"):
                parse_vector(clean_required(row, column, row_number), column, row_number)
            parallel_scale = parse_finite_float(
                clean_required(row, "parallel_scale", row_number),
                "parallel_scale",
                row_number,
            )
            if parallel_scale <= 0.0:
                raise RuntimeError(
                    f"parallel_scale must be positive at manifest row {row_number}."
                )
            width = parse_positive_integer(
                clean_required(row, "image_width", row_number),
                "image_width",
                row_number,
            )
            height = parse_positive_integer(
                clean_required(row, "image_height", row_number),
                "image_height",
                row_number,
            )
            image_reference = clean_required(row, "image_path", row_number)
            label_reference = clean_required(row, "label_path", row_number)
            image_path = resolve_render_path(
                render_root, image_reference, "image_path", row_number
            )
            label_path = resolve_render_path(
                render_root, label_reference, "label_path", row_number
            )
            image, labels = load_png_pair(
                image_path,
                label_path,
                image_reference,
                label_reference,
                width,
                height,
                row_number,
            )
            views.append(
                RenderedView(
                    patient_id=patient_id,
                    case_id=case_id,
                    view_id=view_id,
                    image_reference=image_reference,
                    label_reference=label_reference,
                    source_sha256=source_sha256,
                    image=image,
                    labels=labels,
                )
            )

    if not views:
        raise RuntimeError(f"renderer manifest has no data rows: {manifest_path}")
    cases_by_patient: dict[str, set[str]] = defaultdict(set)
    for view in views:
        cases_by_patient[view.patient_id].add(view.case_id)
    repeated_patients = {
        patient_id: sorted(case_ids)
        for patient_id, case_ids in cases_by_patient.items()
        if len(case_ids) != 1
    }
    if repeated_patients:
        raise RuntimeError(
            "rendered evaluation requires one case per patient; "
            f"violations={repeated_patients}"
        )
    views_by_case: dict[str, list[RenderedView]] = defaultdict(list)
    for view in views:
        views_by_case[view.case_id].append(view)
    for case_id, case_views in sorted(views_by_case.items()):
        if len(case_views) < 2:
            raise RuntimeError(
                f"each case must contain at least two views: case={case_id!r}, "
                f"views={len(case_views)}"
            )
    return sorted(
        views,
        key=lambda view: (
            view.case_id,
            view.view_id,
            view.image_reference,
            view.label_reference,
        ),
    )


def generate_pairs(views: Sequence[RenderedView]) -> list[ViewPair]:
    pairs = [
        ViewPair(
            pair_id=f"pair-{index:06d}",
            template=template,
            query=query,
            is_genuine=template.case_id == query.case_id,
        )
        for index, (template, query) in enumerate(combinations(views, 2), start=1)
    ]
    genuine_count = sum(pair.is_genuine for pair in pairs)
    impostor_count = len(pairs) - genuine_count
    if genuine_count == 0 or impostor_count == 0:
        raise RuntimeError(
            "pair generation requires both genuine and impostor classes; "
            f"genuine={genuine_count}, impostor={impostor_count}."
        )
    return pairs


def extract_view_features(
    views: Sequence[RenderedView],
    crop_size: int,
    crop_padding: float,
) -> dict[tuple[str, str], ViewFeatures]:
    hog = build_hog()
    descriptor_size = int(hog.getDescriptorSize())
    features_by_view: dict[tuple[str, str], ViewFeatures] = {}
    for view in views:
        records: list[CropRecord] = []
        for fdi_label, tooth_index in FDI_TO_TOOTH_INDEX.items():
            mask = view.labels == fdi_label
            if not np.any(mask):
                continue
            try:
                crop = build_normalized_crop(
                    view.image,
                    mask,
                    crop_size,
                    crop_padding,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"failed to normalize FDI {fdi_label} in "
                    f"{view.label_reference!r}: {error}"
                ) from error
            records.append(
                CropRecord(
                    checkup_uid=view.case_id,
                    tooth_index=tooth_index,
                    confidence=1.0,
                    photo_reference=view.image_reference,
                    image=crop,
                )
            )

        embeddings = np.zeros((len(TOOTH_NAMES), descriptor_size), dtype=np.float32)
        present = np.zeros(len(TOOTH_NAMES), dtype=np.bool_)
        extracted = extract_hog_batch(records, hog) if records else []
        if len(extracted) != len(records):
            raise RuntimeError(
                f"HOG feature count mismatch for {view.view_key}: "
                f"expected={len(records)}, actual={len(extracted)}"
            )
        for record, feature in zip(records, extracted, strict=True):
            feature_array = np.asarray(feature, dtype=np.float32)
            if feature_array.shape != (descriptor_size,) or not np.all(
                np.isfinite(feature_array)
            ):
                raise RuntimeError(
                    f"invalid HOG feature for {view.view_key}/"
                    f"{TOOTH_NAMES[record.tooth_index]}: {feature_array.shape}"
                )
            embeddings[record.tooth_index] = feature_array
            present[record.tooth_index] = True
        features_by_view[(view.case_id, view.view_id)] = ViewFeatures(
            embeddings=embeddings,
            present=present,
        )
    return features_by_view


def score_pairs(
    pairs: Sequence[ViewPair],
    features_by_view: Mapping[tuple[str, str], ViewFeatures],
    min_common_teeth: int,
) -> tuple[list[ScoredPair], list[SkippedPair]]:
    scored: list[ScoredPair] = []
    skipped: list[SkippedPair] = []
    for pair in pairs:
        template = features_by_view[(pair.template.case_id, pair.template.view_id)]
        query = features_by_view[(pair.query.case_id, pair.query.view_id)]
        common_count = int(np.count_nonzero(template.present & query.present))
        if common_count < min_common_teeth:
            skipped.append(
                SkippedPair(
                    pair=pair,
                    common_teeth=common_count,
                    reason="insufficient_common_teeth",
                )
            )
            continue
        result = score_pair(
            template_embeddings=template.embeddings,
            query_embeddings=query.embeddings,
            template_present=template.present,
            query_present=query.present,
            min_common_teeth=min_common_teeth,
        )
        scored.append(
            ScoredPair(
                pair=pair,
                fused_score=result.fused_score,
                per_tooth_scores=result.per_tooth_scores,
                common_teeth=result.common_teeth,
            )
        )

    genuine_count = sum(record.pair.is_genuine for record in scored)
    impostor_count = len(scored) - genuine_count
    if genuine_count == 0 or impostor_count == 0:
        raise RuntimeError(
            "scored pairs require both genuine and impostor classes; "
            f"genuine={genuine_count}, impostor={impostor_count}."
        )
    return scored, skipped


def to_metric_record(record: ScoredPair, score: float, row_number: int) -> ScoreRecord:
    pair = record.pair
    return ScoreRecord(
        query_id=pair.query.view_key,
        template_id=pair.template.view_key,
        query_subject_id=pair.query.case_id,
        template_subject_id=pair.template.case_id,
        query_session_id=pair.query.view_id,
        template_session_id=pair.template.view_id,
        is_genuine=pair.is_genuine,
        fused_score=score,
        source_row_number=row_number,
    )


def evaluate_metric_records(records: list[ScoreRecord], total_pairs: int) -> dict[str, object]:
    genuine_count = sum(record.is_genuine for record in records)
    impostor_count = len(records) - genuine_count
    result: dict[str, object] = {
        "genuine_count": genuine_count,
        "impostor_count": impostor_count,
        "missing_scores": total_pairs - len(records),
    }
    if genuine_count == 0 or impostor_count == 0:
        result.update(
            {
                "status": "not_evaluated",
                "reason": (
                    "missing_genuine_class"
                    if genuine_count == 0
                    else "missing_impostor_class"
                ),
                "roc_auc": None,
                "genuine_mean": None,
                "genuine_std": None,
                "impostor_mean": None,
                "impostor_std": None,
                "d_prime": None,
            }
        )
        return result

    points = build_curve(records)
    distribution = compute_score_distribution(records)
    result.update(
        {
            "status": "evaluated",
            "roc_auc": compute_auc(points),
            "genuine_mean": distribution.genuine_mean,
            "genuine_std": distribution.genuine_std,
            "impostor_mean": distribution.impostor_mean,
            "impostor_std": distribution.impostor_std,
            "d_prime": distribution.d_prime,
        }
    )
    return result


def build_metrics(scored: Sequence[ScoredPair]) -> tuple[dict[str, object], list[ScoreRecord]]:
    fused_records = [
        to_metric_record(record, record.fused_score, row_number)
        for row_number, record in enumerate(scored, start=2)
    ]
    fused = evaluate_metric_records(fused_records, len(scored))
    per_tooth: dict[str, dict[str, object]] = {}
    for tooth_name in TOOTH_NAMES:
        tooth_records = [
            to_metric_record(record, record.per_tooth_scores[tooth_name], row_number)
            for row_number, record in enumerate(scored, start=2)
            if tooth_name in record.per_tooth_scores
        ]
        per_tooth[tooth_name] = evaluate_metric_records(tooth_records, len(scored))
    return {"fused": fused, "per_tooth": per_tooth}, fused_records


def pair_counts(records: Sequence[ViewPair | ScoredPair | SkippedPair]) -> dict[str, int]:
    genuine = sum(
        record.is_genuine if isinstance(record, ViewPair) else record.pair.is_genuine
        for record in records
    )
    return {
        "total": len(records),
        "genuine": genuine,
        "impostor": len(records) - genuine,
    }


def write_scores_csv(path: Path, scored: Sequence[ScoredPair]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in scored:
            pair = record.pair
            writer.writerow(
                {
                    "query_id": pair.query.view_key,
                    "template_id": pair.template.view_key,
                    "query_subject_id": pair.query.case_id,
                    "template_subject_id": pair.template.case_id,
                    "query_session_id": pair.query.view_id,
                    "template_session_id": pair.template.view_id,
                    "is_genuine": int(pair.is_genuine),
                    "fused_score": f"{record.fused_score:.12g}",
                    "pair_id": pair.pair_id,
                    "common_teeth": len(record.common_teeth),
                    "common_tooth_names": "|".join(record.common_teeth),
                    "per_tooth_scores": json.dumps(
                        record.per_tooth_scores,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "query_image_path": pair.query.image_reference,
                    "template_image_path": pair.template.image_reference,
                }
            )


def write_skipped_csv(path: Path, skipped: Sequence[SkippedPair]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SKIPPED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in skipped:
            pair = record.pair
            writer.writerow(
                {
                    "pair_id": pair.pair_id,
                    "is_genuine": int(pair.is_genuine),
                    "query_id": pair.query.view_key,
                    "template_id": pair.template.view_key,
                    "query_image_path": pair.query.image_reference,
                    "template_image_path": pair.template.image_reference,
                    "common_teeth": record.common_teeth,
                    "reason": record.reason,
                }
            )


def write_metrics_csv(path: Path, metrics: dict[str, object]) -> None:
    fused = metrics["fused"]
    per_tooth = metrics["per_tooth"]
    if not isinstance(fused, dict) or not isinstance(per_tooth, dict):
        raise RuntimeError("internal metric output contract violation.")
    rows = [("fused", fused), *[(name, per_tooth[name]) for name in TOOTH_NAMES]]
    fieldnames = (
        "feature",
        "scope",
        "roc_auc",
        "d_prime",
        "genuine_mean",
        "impostor_mean",
        "genuine_count",
        "impostor_count",
        "missing_scores",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for scope, values in rows:
            writer.writerow(
                {
                    "feature": "hog",
                    "scope": scope,
                    "roc_auc": values["roc_auc"],
                    "d_prime": values["d_prime"],
                    "genuine_mean": values["genuine_mean"],
                    "impostor_mean": values["impostor_mean"],
                    "genuine_count": values["genuine_count"],
                    "impostor_count": values["impostor_count"],
                    "missing_scores": values["missing_scores"],
                }
            )


def write_outputs_atomic(
    output_dir: Path,
    scored: Sequence[ScoredPair],
    skipped: Sequence[SkippedPair],
    summary: dict[str, object],
    metric_records: list[ScoreRecord],
) -> None:
    generation_dir = create_generation_directory(output_dir)
    try:
        write_scores_csv(generation_dir / "scores.csv", scored)
        write_skipped_csv(generation_dir / "skipped_pairs.csv", skipped)
        metrics = summary["metrics"]
        if not isinstance(metrics, dict):
            raise RuntimeError("summary metrics must be a dictionary.")
        write_metrics_csv(generation_dir / "metrics.csv", metrics)
        write_score_distribution_plot(
            generation_dir / "score_distribution.png", metric_records
        )
        (generation_dir / "summary.json").write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        publish_generation(generation_dir, output_dir)
    finally:
        discard_generation(generation_dir)


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    views = load_manifest(args.render_root)
    pairs = generate_pairs(views)
    features_by_view = extract_view_features(
        views,
        args.crop_size,
        args.crop_padding,
    )
    scored, skipped = score_pairs(
        pairs,
        features_by_view,
        args.min_common_teeth,
    )
    metrics, metric_records = build_metrics(scored)
    fused_metrics = metrics["fused"]
    if not isinstance(fused_metrics, dict):
        raise RuntimeError("internal fused metric contract violation.")
    generated_counts = pair_counts(pairs)
    scored_counts = pair_counts(scored)
    skipped_counts = pair_counts(skipped)
    summary: dict[str, object] = {
        "schema_version": 1,
        "feature_name": "hog",
        "manifest_name": "manifest.csv",
        "manifest_rows": len(views),
        "case_count": len({view.case_id for view in views}),
        "view_count": len(views),
        "tooth_names": list(TOOTH_NAMES),
        "fdi_to_tooth_name": {
            str(fdi): TOOTH_NAMES[index]
            for fdi, index in FDI_TO_TOOTH_INDEX.items()
        },
        "pair_rule": (
            "canonical unordered view pairs; same case is genuine, different case "
            "is impostor"
        ),
        "crop_size": args.crop_size,
        "crop_padding": args.crop_padding,
        "min_common_teeth": args.min_common_teeth,
        "pair_counts": {
            "generated": generated_counts,
            "scored": scored_counts,
            "skipped": skipped_counts,
        },
        "genuine_pairs": scored_counts["genuine"],
        "impostor_pairs": scored_counts["impostor"],
        "skipped_by_reason": dict(
            sorted(Counter(record.reason for record in skipped).items())
        ),
        "roc_auc": fused_metrics["roc_auc"],
        "d_prime": fused_metrics["d_prime"],
        "metrics": metrics,
    }
    write_outputs_atomic(
        args.output_dir,
        scored,
        skipped,
        summary,
        metric_records,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(f"scores: {args.output_dir / 'scores.csv'}")
    print(f"skipped: {args.output_dir / 'skipped_pairs.csv'}")
    print(f"summary: {args.output_dir / 'summary.json'}")
    print(f"histogram: {args.output_dir / 'score_distribution.png'}")
    print(
        f"genuine={summary['genuine_pairs']} impostor={summary['impostor_pairs']} "
        f"roc_auc={summary['roc_auc']:.6f} d_prime={summary['d_prime']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
