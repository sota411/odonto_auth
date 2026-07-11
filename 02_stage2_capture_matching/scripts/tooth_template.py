from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from score_code_pairs import EXPECTED_TOOTH_NAMES, FeatureStore, build_feature_store


FORMAT_VERSION = 2
FEATURE_CONTRACT_ARRAYS = (
    "feature_name",
    "segmentation_weights_sha256",
    "segmentation_imgsz",
    "segmentation_conf",
    "segmentation_iou",
    "crop_size",
    "crop_padding",
    "preprocessing_format_version",
)
REQUIRED_TEMPLATE_ARRAYS = (
    "format_version",
    "subject_id",
    "created_at",
    "feature_name",
    "tooth_names",
    "embeddings",
    "present",
    "source_checkup_ids",
    "source_features_sha256",
    "segmentation_weights_sha256",
    "segmentation_imgsz",
    "segmentation_conf",
    "segmentation_iou",
    "crop_size",
    "crop_padding",
    "preprocessing_format_version",
    "registration_image_sha256",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class ExtractionContract:
    feature_name: str
    segmentation_weights_sha256: str
    segmentation_imgsz: int
    segmentation_conf: float
    segmentation_iou: float
    crop_size: int
    crop_padding: float
    preprocessing_format_version: str


@dataclass(frozen=True)
class ToothTemplate:
    format_version: int
    subject_id: str
    created_at: str
    feature_name: str
    tooth_names: tuple[str, ...]
    embeddings: np.ndarray
    present: np.ndarray
    source_checkup_ids: tuple[str, ...]
    source_features_sha256: str
    segmentation_weights_sha256: str
    segmentation_imgsz: int
    segmentation_conf: float
    segmentation_iou: float
    crop_size: int
    crop_padding: float
    preprocessing_format_version: str
    registration_image_sha256: tuple[str, ...]

    @property
    def extraction_contract(self) -> ExtractionContract:
        return ExtractionContract(
            feature_name=self.feature_name,
            segmentation_weights_sha256=self.segmentation_weights_sha256,
            segmentation_imgsz=self.segmentation_imgsz,
            segmentation_conf=self.segmentation_conf,
            segmentation_iou=self.segmentation_iou,
            crop_size=self.crop_size,
            crop_padding=self.crop_padding,
            preprocessing_format_version=self.preprocessing_format_version,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one subject's tooth template from an existing feature NPZ."
    )
    parser.add_argument("--features-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument(
        "--checkup-id",
        action="append",
        required=True,
        help="Source checkup ID. Specify this option at least twice.",
    )
    parser.add_argument(
        "--feature-name",
        required=True,
        help="Expected feature_name stored in the source NPZ.",
    )
    return parser.parse_args(argv)


def clean_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string.")
    cleaned = value.strip()
    if cleaned == "":
        raise RuntimeError(f"{name} must not be empty.")
    if cleaned != value:
        raise RuntimeError(f"{name} must not have leading or trailing whitespace.")
    return cleaned


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"failed to read file for SHA-256: {path}") from exc
    return digest.hexdigest()


def decode_string(value: object, name: str) -> str:
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{name} is not valid UTF-8.") from exc
    elif isinstance(value, str):
        decoded = value
    else:
        raise RuntimeError(f"{name} must be a string.")
    return clean_identifier(decoded, name)


def parse_string_scalar(array: np.ndarray, name: str) -> str:
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise RuntimeError(
            f"{name} must be a scalar string; got shape={array.shape}, dtype={array.dtype}."
        )
    return decode_string(array.item(), name)


def parse_string_vector(array: np.ndarray, name: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise RuntimeError(
            f"{name} must be a one-dimensional string array; "
            f"got shape={array.shape}, dtype={array.dtype}."
        )
    return tuple(
        decode_string(value, f"{name}[{index}]")
        for index, value in enumerate(array.tolist())
    )


def parse_int_scalar(array: np.ndarray, name: str, *, minimum: int) -> int:
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise RuntimeError(
            f"{name} must be a scalar integer; got shape={array.shape}, dtype={array.dtype}."
        )
    value = int(array.item())
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}; got {value}.")
    return value


def parse_float_scalar(
    array: np.ndarray,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if array.shape != () or not np.issubdtype(array.dtype, np.floating):
        raise RuntimeError(
            f"{name} must be a scalar float; got shape={array.shape}, dtype={array.dtype}."
        )
    value = float(array.item())
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite.")
    below_minimum = value <= minimum if minimum_exclusive else value < minimum
    if below_minimum or (maximum is not None and value > maximum):
        interval_start = "(" if minimum_exclusive else "["
        interval_end = "]" if maximum is not None else ")"
        upper = str(maximum) if maximum is not None else "infinity"
        raise RuntimeError(
            f"{name} must be in {interval_start}{minimum}, {upper}{interval_end}; got {value}."
        )
    return value


def parse_sha256_scalar(array: np.ndarray, name: str) -> str:
    value = parse_string_scalar(array, name)
    if SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{name} must be 64 lowercase hex characters.")
    return value


def parse_extraction_contract(arrays: dict[str, np.ndarray]) -> ExtractionContract:
    return ExtractionContract(
        feature_name=parse_string_scalar(arrays["feature_name"], "feature_name"),
        segmentation_weights_sha256=parse_sha256_scalar(
            arrays["segmentation_weights_sha256"],
            "segmentation_weights_sha256",
        ),
        segmentation_imgsz=parse_int_scalar(
            arrays["segmentation_imgsz"],
            "segmentation_imgsz",
            minimum=1,
        ),
        segmentation_conf=parse_float_scalar(
            arrays["segmentation_conf"],
            "segmentation_conf",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        ),
        segmentation_iou=parse_float_scalar(
            arrays["segmentation_iou"],
            "segmentation_iou",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        ),
        crop_size=parse_int_scalar(arrays["crop_size"], "crop_size", minimum=1),
        crop_padding=parse_float_scalar(
            arrays["crop_padding"],
            "crop_padding",
            minimum=0.0,
        ),
        preprocessing_format_version=parse_string_scalar(
            arrays["preprocessing_format_version"],
            "preprocessing_format_version",
        ),
    )


def load_source_extraction_contract(path: Path) -> ExtractionContract:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to load feature NPZ: {path}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise RuntimeError(f"feature input must be an NPZ archive: {path}")
    try:
        missing = [name for name in FEATURE_CONTRACT_ARRAYS if name not in archive.files]
        if missing:
            raise RuntimeError(
                f"feature NPZ is missing required extraction contract arrays {missing}: {path}"
            )
        try:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in FEATURE_CONTRACT_ARRAYS
            }
        except ValueError as exc:
            raise RuntimeError(
                f"failed to read extraction contract from feature NPZ: {path}"
            ) from exc
        return parse_extraction_contract(arrays)
    finally:
        archive.close()


def validate_selection(
    features: FeatureStore,
    subject_id: str,
    checkup_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    selected_ids = tuple(
        clean_identifier(checkup_id, f"checkup_ids[{index}]")
        for index, checkup_id in enumerate(checkup_ids)
    )
    if len(selected_ids) < 2 or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("checkup_ids must contain at least two unique checkup IDs.")

    selected_indices: list[int] = []
    for checkup_id in selected_ids:
        if checkup_id not in features.index_by_checkup_id:
            raise RuntimeError(f"selected checkup was not found in feature NPZ: {checkup_id!r}")
        index = features.index_by_checkup_id[checkup_id]
        actual_subject_id = features.patient_ids[index]
        if actual_subject_id != subject_id:
            raise RuntimeError(
                "feature subject ID mismatch for selected checkup: "
                f"checkup_id={checkup_id!r}, expected={subject_id!r}, "
                f"actual={actual_subject_id!r}"
            )
        selected_indices.append(index)
    return selected_ids, tuple(selected_indices)


def aggregate_embeddings(
    features: FeatureStore,
    selected_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    feature_dimension = features.embeddings.shape[2]
    embeddings = np.zeros(
        (len(EXPECTED_TOOTH_NAMES), feature_dimension),
        dtype=np.float32,
    )
    present = np.zeros(len(EXPECTED_TOOTH_NAMES), dtype=np.bool_)

    for tooth_index, tooth_name in enumerate(EXPECTED_TOOTH_NAMES):
        source_indices = tuple(
            index for index in selected_indices if features.present[index, tooth_index]
        )
        if not source_indices:
            continue
        source_embeddings = features.embeddings[
            np.asarray(source_indices, dtype=np.intp), tooth_index
        ]
        mean_embedding = np.mean(source_embeddings, axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(mean_embedding))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"zero-norm mean embedding for present tooth {tooth_name!r}.")
        embeddings[tooth_index] = (mean_embedding / norm).astype(np.float32)
        present[tooth_index] = True

    if not np.any(present):
        raise RuntimeError("selected checkups contain no tooth embeddings.")
    return embeddings, present


def validate_created_at(value: str) -> str:
    parsed = clean_identifier(value, "created_at")
    try:
        datetime.strptime(parsed, CREATED_AT_FORMAT)
    except ValueError as exc:
        raise RuntimeError(
            f"created_at must use UTC format YYYY-MM-DDTHH:MM:SSZ; got {parsed!r}."
        ) from exc
    return parsed


def current_created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(CREATED_AT_FORMAT)


def build_template(
    features_path: Path,
    subject_id: str,
    checkup_ids: Sequence[str],
    feature_name: str,
    created_at: str | None = None,
) -> ToothTemplate:
    source_path = Path(features_path)
    if not source_path.is_file():
        raise RuntimeError(f"feature NPZ was not found: {source_path}")
    clean_subject_id = clean_identifier(subject_id, "subject_id")
    expected_feature_name = clean_identifier(feature_name, "feature_name")

    source_hash_before = sha256_file(source_path)
    features = build_feature_store(source_path)
    extraction_contract = load_source_extraction_contract(source_path)
    if extraction_contract.feature_name != expected_feature_name:
        raise RuntimeError(
            "feature_name mismatch: "
            f"expected={expected_feature_name!r}, "
            f"actual={extraction_contract.feature_name!r}"
        )
    selected_ids, selected_indices = validate_selection(
        features,
        clean_subject_id,
        checkup_ids,
    )
    embeddings, present = aggregate_embeddings(features, selected_indices)
    registration_image_sha256 = tuple(
        fingerprint.sha256
        for index in selected_indices
        for fingerprint in features.photo_fingerprints[index]
    )
    if not registration_image_sha256:
        raise RuntimeError("selected checkups contain no registration image fingerprints.")
    source_hash_after = sha256_file(source_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"source feature NPZ changed while creating template: {source_path}")

    return ToothTemplate(
        format_version=FORMAT_VERSION,
        subject_id=clean_subject_id,
        created_at=validate_created_at(
            created_at if created_at is not None else current_created_at()
        ),
        feature_name=extraction_contract.feature_name,
        tooth_names=EXPECTED_TOOTH_NAMES,
        embeddings=embeddings,
        present=present,
        source_checkup_ids=selected_ids,
        source_features_sha256=source_hash_before,
        segmentation_weights_sha256=extraction_contract.segmentation_weights_sha256,
        segmentation_imgsz=extraction_contract.segmentation_imgsz,
        segmentation_conf=extraction_contract.segmentation_conf,
        segmentation_iou=extraction_contract.segmentation_iou,
        crop_size=extraction_contract.crop_size,
        crop_padding=extraction_contract.crop_padding,
        preprocessing_format_version=extraction_contract.preprocessing_format_version,
        registration_image_sha256=registration_image_sha256,
    )


def template_arrays(template: ToothTemplate) -> dict[str, np.ndarray]:
    return {
        "format_version": np.asarray(template.format_version, dtype=np.int64),
        "subject_id": np.asarray(template.subject_id),
        "created_at": np.asarray(template.created_at),
        "feature_name": np.asarray(template.feature_name),
        "tooth_names": np.asarray(template.tooth_names),
        "embeddings": np.asarray(template.embeddings, dtype=np.float32),
        "present": np.asarray(template.present, dtype=np.bool_),
        "source_checkup_ids": np.asarray(template.source_checkup_ids),
        "source_features_sha256": np.asarray(template.source_features_sha256),
        "segmentation_weights_sha256": np.asarray(
            template.segmentation_weights_sha256
        ),
        "segmentation_imgsz": np.asarray(template.segmentation_imgsz, dtype=np.int64),
        "segmentation_conf": np.asarray(template.segmentation_conf, dtype=np.float64),
        "segmentation_iou": np.asarray(template.segmentation_iou, dtype=np.float64),
        "crop_size": np.asarray(template.crop_size, dtype=np.int64),
        "crop_padding": np.asarray(template.crop_padding, dtype=np.float64),
        "preprocessing_format_version": np.asarray(
            template.preprocessing_format_version
        ),
        "registration_image_sha256": np.asarray(template.registration_image_sha256),
    }


def atomic_save_template(
    path: Path,
    template: ToothTemplate,
    source_features_path: Path,
) -> ToothTemplate:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **template_arrays(template))
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        verified_template = load_template(
            temporary_path,
            source_features_path=source_features_path,
        )
        final_source_hash = sha256_file(source_features_path)
        if final_source_hash != template.source_features_sha256:
            raise RuntimeError(
                "source feature NPZ changed before publish: "
                f"expected={template.source_features_sha256}, actual={final_source_hash}"
            )
        os.replace(temporary_path, output_path)
        return verified_template
    except OSError as exc:
        raise RuntimeError(f"failed to save template NPZ atomically: {output_path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_template_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to load template NPZ: {path}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise RuntimeError(f"template input must be an NPZ archive: {path}")
    try:
        if len(archive.files) != len(set(archive.files)):
            raise RuntimeError(f"template NPZ contains duplicate arrays: {path}")
        actual_names = set(archive.files)
        expected_names = set(REQUIRED_TEMPLATE_ARRAYS)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise RuntimeError(
                f"template NPZ arrays mismatch: missing={missing}, unexpected={unexpected}"
            )
        try:
            return {
                name: np.array(archive[name], copy=True)
                for name in REQUIRED_TEMPLATE_ARRAYS
            }
        except ValueError as exc:
            raise RuntimeError(f"failed to read template NPZ arrays: {path}") from exc
    finally:
        archive.close()


def validate_format_version(array: np.ndarray) -> int:
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise RuntimeError(
            "format_version must be a scalar integer; "
            f"got shape={array.shape}, dtype={array.dtype}."
        )
    value = int(array.item())
    if value != FORMAT_VERSION:
        raise RuntimeError(
            f"unsupported format_version: expected={FORMAT_VERSION}, actual={value}"
        )
    return value


def parse_template(arrays: dict[str, np.ndarray]) -> ToothTemplate:
    format_version = validate_format_version(arrays["format_version"])
    subject_id = parse_string_scalar(arrays["subject_id"], "subject_id")
    created_at = validate_created_at(
        parse_string_scalar(arrays["created_at"], "created_at")
    )
    extraction_contract = parse_extraction_contract(arrays)
    tooth_names = parse_string_vector(arrays["tooth_names"], "tooth_names")
    if tooth_names != EXPECTED_TOOTH_NAMES:
        raise RuntimeError(
            "tooth_names has an unexpected order; "
            f"expected={EXPECTED_TOOTH_NAMES}, actual={tooth_names}"
        )

    embeddings = arrays["embeddings"]
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(EXPECTED_TOOTH_NAMES)
        or embeddings.shape[1] == 0
        or not np.issubdtype(embeddings.dtype, np.floating)
    ):
        raise RuntimeError(
            "embeddings must have floating shape [6, D] with D > 0; "
            f"got shape={embeddings.shape}, dtype={embeddings.dtype}."
        )
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("embeddings must contain only finite values.")

    present = arrays["present"]
    expected_present_shape = (len(EXPECTED_TOOTH_NAMES),)
    if present.shape != expected_present_shape or present.dtype != np.bool_:
        raise RuntimeError(
            f"present must have shape {expected_present_shape} and bool dtype; "
            f"got shape={present.shape}, dtype={present.dtype}."
        )
    if not np.any(present):
        raise RuntimeError("template must contain at least one present tooth.")
    present_norms = np.linalg.norm(embeddings[present].astype(np.float64), axis=1)
    if not np.allclose(present_norms, 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("present tooth embeddings must be L2-normalized.")
    if not np.all(embeddings[~present] == 0.0):
        raise RuntimeError("absent tooth embeddings must be zero.")

    source_checkup_ids = parse_string_vector(
        arrays["source_checkup_ids"],
        "source_checkup_ids",
    )
    if (
        len(source_checkup_ids) < 2
        or len(set(source_checkup_ids)) != len(source_checkup_ids)
    ):
        raise RuntimeError(
            "source_checkup_ids must contain at least two unique checkup IDs."
        )
    source_hash = parse_sha256_scalar(
        arrays["source_features_sha256"],
        "source_features_sha256",
    )
    registration_image_sha256 = parse_string_vector(
        arrays["registration_image_sha256"],
        "registration_image_sha256",
    )
    if not registration_image_sha256:
        raise RuntimeError("registration_image_sha256 must contain at least one hash.")
    for index, value in enumerate(registration_image_sha256):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise RuntimeError(
                f"registration_image_sha256[{index}] must be 64 lowercase hex characters."
            )

    return ToothTemplate(
        format_version=format_version,
        subject_id=subject_id,
        created_at=created_at,
        feature_name=extraction_contract.feature_name,
        tooth_names=tooth_names,
        embeddings=embeddings,
        present=present,
        source_checkup_ids=source_checkup_ids,
        source_features_sha256=source_hash,
        segmentation_weights_sha256=extraction_contract.segmentation_weights_sha256,
        segmentation_imgsz=extraction_contract.segmentation_imgsz,
        segmentation_conf=extraction_contract.segmentation_conf,
        segmentation_iou=extraction_contract.segmentation_iou,
        crop_size=extraction_contract.crop_size,
        crop_padding=extraction_contract.crop_padding,
        preprocessing_format_version=extraction_contract.preprocessing_format_version,
        registration_image_sha256=registration_image_sha256,
    )


def load_template(
    path: Path,
    source_features_path: Path | None = None,
) -> ToothTemplate:
    template_path = Path(path)
    if not template_path.is_file():
        raise RuntimeError(f"template NPZ was not found: {template_path}")
    template = parse_template(load_template_arrays(template_path))
    if source_features_path is not None:
        source_path = Path(source_features_path)
        actual_hash = sha256_file(source_path)
        if actual_hash != template.source_features_sha256:
            raise RuntimeError(
                "source feature SHA-256 mismatch: "
                f"expected={template.source_features_sha256}, actual={actual_hash}"
            )
        expected = build_template(
            features_path=source_path,
            subject_id=template.subject_id,
            checkup_ids=template.source_checkup_ids,
            feature_name=template.feature_name,
            created_at=template.created_at,
        )
        if (
            expected.source_features_sha256 != template.source_features_sha256
            or expected.extraction_contract != template.extraction_contract
            or expected.registration_image_sha256
            != template.registration_image_sha256
            or expected.tooth_names != template.tooth_names
            or not np.array_equal(expected.embeddings, template.embeddings)
            or not np.array_equal(expected.present, template.present)
        ):
            raise RuntimeError(
                "template content mismatch against the verified source feature NPZ."
            )
    return template


def create_template(
    features_path: Path,
    output_path: Path,
    subject_id: str,
    checkup_ids: Sequence[str],
    feature_name: str,
    created_at: str | None = None,
) -> ToothTemplate:
    source_path = Path(features_path)
    destination_path = Path(output_path)
    if source_path.resolve() == destination_path.resolve():
        raise RuntimeError("features_path and output_path must be different files.")
    template = build_template(
        source_path,
        subject_id,
        checkup_ids,
        feature_name,
        created_at,
    )
    return atomic_save_template(destination_path, template, source_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    template = create_template(
        features_path=args.features_npz,
        output_path=args.output_npz,
        subject_id=args.subject_id,
        checkup_ids=args.checkup_id,
        feature_name=args.feature_name,
    )
    print(
        json.dumps(
            {
                "output_npz": str(args.output_npz),
                "subject_id": template.subject_id,
                "feature_name": template.feature_name,
                "source_checkup_ids": list(template.source_checkup_ids),
                "present_teeth": int(template.present.sum()),
                "embedding_dimension": int(template.embeddings.shape[1]),
                "source_features_sha256": template.source_features_sha256,
                "registration_images": len(template.registration_image_sha256),
                "preprocessing_format_version": (
                    template.preprocessing_format_version
                ),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
