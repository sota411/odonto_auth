from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import stat
import struct
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image, UnidentifiedImageError
from ultralytics.utils.patches import _image_open as _PILLOW_IMAGE_OPEN

from code_photos import sha256_file
from output_directory import create_generation_directory, discard_generation, publish_generation
from prepare_code_annotation_batch import (
    ALL_TOOTH_NAMES,
    ANNOTATED_TOOTH_NAMES,
    SELECTED_SPLITS,
    find_project_root,
    manifest_identity_sha256,
    validate_output_location,
)


ALLOWED_STATUSES = frozenset(("complete", "negative", "excluded"))
SUPPORTED_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
TARGET_CLASS_IDS = frozenset(
    class_id for class_id, name in enumerate(ALL_TOOTH_NAMES) if name in ANNOTATED_TOOTH_NAMES
)
REQUIRED_MANIFEST_COLUMNS = frozenset(
    (
        "split",
        "image_name",
        "patient_token",
        "checkup_token",
        "source_sha256",
        "annotation_status",
        "view_tag",
        "lighting_tag",
        "oral_condition_tag",
    )
)
MAX_ZIP_FILE_BYTES = 128 * 1024 * 1024
MAX_ZIP_ENTRIES = 5_000
MAX_ZIP_ENTRY_BYTES = 64 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_TAG_VALUES = {
    "view_tag": frozenset(
        (
            "frontal",
            "left_lateral",
            "right_lateral",
            "maxillary_occlusal",
            "mandibular_occlusal",
            "other",
        )
    ),
    "lighting_tag": frozenset(("normal", "dark", "overexposed", "reflection")),
    "oral_condition_tag": frozenset(
        ("none", "orthodontic_appliance", "restoration", "missing_tooth", "other")
    ),
}


@dataclass(frozen=True)
class ManifestRow:
    split: str
    image_name: str
    patient_token: str
    checkup_token: str
    source_sha256: str
    annotation_status: str
    view_tag: str
    lighting_tag: str
    oral_condition_tag: str


@dataclass(frozen=True)
class FinalizedImage:
    manifest: ManifestRow
    image_bytes: bytes
    label_text: str
    class_counts: Counter[int]


@dataclass
class ZipBudget:
    max_file_bytes: int
    max_entries: int
    max_entry_bytes: int
    max_total_uncompressed_bytes: int
    total_uncompressed_bytes: int = 0

    def consume_uncompressed(self, byte_count: int, path: Path) -> None:
        self.total_uncompressed_bytes += byte_count
        if self.total_uncompressed_bytes > self.max_total_uncompressed_bytes:
            raise RuntimeError(
                "cumulative ZIP uncompressed size exceeds limit "
                f"{self.max_total_uncompressed_bytes}: {path}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Validate separate CVAT exports and build a split-preserving YOLO dataset."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=(
            repo_root
            / "01_stage1_real_image_extraction"
            / "datasets"
            / "dataset_real"
            / "code_annotation"
        ),
        help="Ignored directory containing the source batch manifest, summary, labels, and ZIPs.",
    )
    parser.add_argument(
        "--train-export",
        type=Path,
        required=True,
        help="CVAT Ultralytics YOLO Segmentation export for the train task.",
    )
    parser.add_argument(
        "--val-export",
        type=Path,
        required=True,
        help="CVAT Ultralytics YOLO Segmentation export for the val task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "01_stage1_real_image_extraction"
            / "datasets"
            / "dataset_real"
            / "dataset_code_real"
        ),
        help="Ignored output directory for the finalized YOLO dataset.",
    )
    parser.add_argument(
        "--max-zip-file-bytes",
        type=int,
        default=MAX_ZIP_FILE_BYTES,
        help="Maximum compressed file size accepted from each input ZIP.",
    )
    parser.add_argument(
        "--max-zip-entries",
        type=int,
        default=MAX_ZIP_ENTRIES,
        help="Maximum central-directory entry count accepted from each input ZIP.",
    )
    parser.add_argument(
        "--max-zip-entry-bytes",
        type=int,
        default=MAX_ZIP_ENTRY_BYTES,
        help="Maximum uncompressed size accepted from one ZIP entry.",
    )
    parser.add_argument(
        "--max-total-uncompressed-bytes",
        type=int,
        default=MAX_TOTAL_UNCOMPRESSED_BYTES,
        help="Maximum cumulative uncompressed size accepted from all input ZIPs.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=MAX_IMAGE_PIXELS,
        help="Maximum decoded pixel count accepted from one source image.",
    )
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} was not found: {path}")


def clean_required(value: str | None, column: str, row_number: int) -> str:
    if value is None:
        raise RuntimeError(f"{column} is missing at manifest row {row_number}.")
    cleaned = value.strip()
    if cleaned == "":
        raise RuntimeError(f"{column} is empty at manifest row {row_number}.")
    return cleaned


def clean_optional(value: str | None, column: str, row_number: int) -> str:
    if value is None:
        raise RuntimeError(f"{column} is missing at manifest row {row_number}.")
    return value.strip()


def validate_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise RuntimeError(f"{label} must be a lowercase hexadecimal SHA-256: {value!r}")
    return normalized


def load_batch_summary(path: Path, manifest_path: Path, batch_zip_paths: dict[str, Path]) -> dict:
    require_file(path, "batch summary")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"batch summary is not valid UTF-8 JSON: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"batch summary root must be an object: {path}")
    try:
        original_manifest_sha256 = document["manifest_sha256"]
        expected_identity_sha256 = document["manifest_identity_sha256"]
        archive_hashes = document["cvat_archive_sha256_by_split"]
    except KeyError as error:
        raise RuntimeError(f"batch summary is missing required key: {error.args[0]}") from error
    if not isinstance(original_manifest_sha256, str):
        raise RuntimeError("batch summary manifest SHA-256 must be a string.")
    validate_sha256(original_manifest_sha256, "batch summary manifest SHA-256")
    if not isinstance(expected_identity_sha256, str):
        raise RuntimeError("batch summary manifest identity SHA-256 must be a string.")
    actual_identity_sha256 = manifest_identity_sha256(manifest_path)
    if expected_identity_sha256 != actual_identity_sha256:
        raise RuntimeError(
            "batch manifest identity SHA-256 does not match summary: "
            f"expected={expected_identity_sha256}, actual={actual_identity_sha256}"
        )
    if not isinstance(archive_hashes, dict):
        raise RuntimeError("batch archive SHA-256 values must be an object.")
    if set(archive_hashes) != set(SELECTED_SPLITS):
        raise RuntimeError("batch archive SHA-256 split keys do not match train/val.")
    for split, zip_path in batch_zip_paths.items():
        expected_hash = archive_hashes[split]
        if not isinstance(expected_hash, str):
            raise RuntimeError(f"batch archive SHA-256 must be a string: {split}")
        actual_hash = sha256_file(zip_path)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"batch {split} ZIP SHA-256 does not match summary: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
    return document


def load_labels_json(path: Path) -> None:
    require_file(path, "CVAT labels JSON")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"CVAT labels are not valid UTF-8 JSON: {path}") from error
    if not isinstance(document, list) or len(document) != len(ALL_TOOTH_NAMES):
        raise RuntimeError("CVAT labels must contain the complete 14-class list.")
    names: list[str] = []
    for index, label in enumerate(document):
        if not isinstance(label, dict):
            raise RuntimeError(f"CVAT label must be an object at index {index}.")
        try:
            name = label["name"]
            shape_type = label["type"]
        except KeyError as error:
            raise RuntimeError(f"CVAT label is missing required key: {error.args[0]}") from error
        if not isinstance(name, str) or not isinstance(shape_type, str):
            raise RuntimeError(f"CVAT label name/type must be strings at index {index}.")
        if shape_type != "polygon":
            raise RuntimeError(f"CVAT label type must be polygon at index {index}: {shape_type!r}")
        names.append(name)
    if tuple(names) != ALL_TOOTH_NAMES:
        raise RuntimeError(
            f"CVAT class names/order do not match: expected={ALL_TOOTH_NAMES}, actual={tuple(names)}"
        )


def load_manifest(path: Path) -> list[ManifestRow]:
    require_file(path, "annotation manifest")
    rows: list[ManifestRow] = []
    seen_names: dict[str, int] = {}
    split_by_patient: dict[str, str] = {}
    split_by_hash: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"annotation manifest has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"annotation manifest has duplicate columns: {path}")
        missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames))
        if missing_columns:
            raise RuntimeError(f"annotation manifest is missing columns: {missing_columns}")
        for row_number, row in enumerate(reader, start=2):
            split = clean_required(row["split"], "split", row_number)
            if split not in SELECTED_SPLITS:
                raise RuntimeError(f"unsupported split at manifest row {row_number}: {split!r}")
            image_name = clean_required(row["image_name"], "image_name", row_number)
            image_path = PurePosixPath(image_name)
            if image_path.name != image_name or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise RuntimeError(f"invalid image_name at manifest row {row_number}: {image_name!r}")
            folded_name = image_name.casefold()
            if folded_name in seen_names:
                raise RuntimeError(
                    f"duplicate image_name at manifest row {row_number}; "
                    f"first seen at row {seen_names[folded_name]}: {image_name!r}"
                )
            seen_names[folded_name] = row_number
            patient_token = clean_required(row["patient_token"], "patient_token", row_number)
            checkup_token = clean_required(row["checkup_token"], "checkup_token", row_number)
            source_sha256 = validate_sha256(
                clean_required(row["source_sha256"], "source_sha256", row_number),
                f"source_sha256 at manifest row {row_number}",
            )
            status = clean_required(row["annotation_status"], "annotation_status", row_number)
            if status == "pending":
                raise RuntimeError(f"annotation status is still pending at manifest row {row_number}.")
            if status not in ALLOWED_STATUSES:
                raise RuntimeError(
                    f"unsupported annotation status at manifest row {row_number}: {status!r}"
                )
            view_tag = clean_optional(row["view_tag"], "view_tag", row_number)
            lighting_tag = clean_optional(row["lighting_tag"], "lighting_tag", row_number)
            oral_condition_tag = clean_optional(
                row["oral_condition_tag"],
                "oral_condition_tag",
                row_number,
            )
            if status != "excluded":
                required_tags = {
                    "view_tag": view_tag,
                    "lighting_tag": lighting_tag,
                    "oral_condition_tag": oral_condition_tag,
                }
                for tag_name, tag_value in required_tags.items():
                    if tag_value == "":
                        raise RuntimeError(
                            f"{tag_name} is empty for included image at manifest row {row_number}."
                        )
                    if tag_value not in ALLOWED_TAG_VALUES[tag_name]:
                        raise RuntimeError(
                            f"{tag_name} has an unsupported value at manifest row {row_number}: "
                            f"{tag_value!r}"
                        )
            if patient_token in split_by_patient and split_by_patient[patient_token] != split:
                raise RuntimeError(f"patient token is assigned to multiple splits: {patient_token!r}")
            split_by_patient[patient_token] = split
            if source_sha256 in split_by_hash and split_by_hash[source_sha256] != split:
                raise RuntimeError(f"image SHA-256 is assigned to multiple splits: {source_sha256}")
            if source_sha256 in split_by_hash:
                raise RuntimeError(f"duplicate image SHA-256 in manifest: {source_sha256}")
            split_by_hash[source_sha256] = split
            rows.append(
                ManifestRow(
                    split=split,
                    image_name=image_name,
                    patient_token=patient_token,
                    checkup_token=checkup_token,
                    source_sha256=source_sha256,
                    annotation_status=status,
                    view_tag=view_tag,
                    lighting_tag=lighting_tag,
                    oral_condition_tag=oral_condition_tag,
                )
            )
    if not rows:
        raise RuntimeError(f"annotation manifest has no rows: {path}")
    for split in SELECTED_SPLITS:
        if not any(row.split == split for row in rows):
            raise RuntimeError(f"annotation manifest has no {split} rows.")
    return rows


def preflight_zip(path: Path, budget: ZipBudget) -> int:
    file_size = path.stat().st_size
    if file_size > budget.max_file_bytes:
        raise RuntimeError(
            f"ZIP file size exceeds limit {budget.max_file_bytes}: {path}: {file_size}"
        )
    if file_size < 22:
        raise RuntimeError(f"ZIP archive is too small to contain an EOCD record: {path}")
    tail_size = min(file_size, 65_557)
    with path.open("rb") as handle:
        handle.seek(file_size - tail_size)
        tail = handle.read(tail_size)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or len(tail) - eocd_offset < 22:
        raise RuntimeError(f"ZIP EOCD record was not found: {path}")
    (
        _signature,
        disk_number,
        directory_disk,
        entries_on_disk,
        total_entries,
        directory_size,
        directory_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    if disk_number != 0 or directory_disk != 0 or entries_on_disk != total_entries:
        raise RuntimeError(f"multi-disk ZIP archives are not supported: {path}")
    if total_entries == 0xFFFF or directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        raise RuntimeError(f"ZIP64 archives are not supported: {path}")
    if eocd_offset + 22 + comment_length != len(tail):
        raise RuntimeError(f"ZIP archive has trailing or inconsistent EOCD data: {path}")
    eocd_absolute_offset = file_size - tail_size + eocd_offset
    if directory_offset + directory_size != eocd_absolute_offset:
        raise RuntimeError(f"ZIP central-directory size or offset is inconsistent: {path}")

    actual_entries = 0
    total_uncompressed_bytes = 0
    consumed_directory_bytes = 0
    with path.open("rb") as handle:
        handle.seek(directory_offset)
        while consumed_directory_bytes < directory_size:
            remaining_bytes = directory_size - consumed_directory_bytes
            if remaining_bytes < 46:
                raise RuntimeError(f"ZIP central-directory record is truncated: {path}")
            header = handle.read(46)
            if len(header) != 46:
                raise RuntimeError(f"ZIP central-directory record cannot be read: {path}")
            fields = struct.unpack("<4s6H3L5H2L", header)
            if fields[0] != b"PK\x01\x02":
                raise RuntimeError(f"ZIP central-directory signature is invalid: {path}")
            compressed_size = fields[8]
            uncompressed_size = fields[9]
            filename_length = fields[10]
            extra_length = fields[11]
            entry_comment_length = fields[12]
            entry_disk = fields[13]
            local_header_offset = fields[16]
            if entry_disk != 0:
                raise RuntimeError(f"multi-disk ZIP entries are not supported: {path}")
            if (
                compressed_size == 0xFFFFFFFF
                or uncompressed_size == 0xFFFFFFFF
                or local_header_offset == 0xFFFFFFFF
            ):
                raise RuntimeError(f"ZIP64 entries are not supported: {path}")
            if uncompressed_size > budget.max_entry_bytes:
                raise RuntimeError(
                    f"ZIP entry size exceeds limit {budget.max_entry_bytes}: {path}"
                )
            variable_size = filename_length + extra_length + entry_comment_length
            record_size = 46 + variable_size
            if record_size > remaining_bytes:
                raise RuntimeError(f"ZIP central-directory record exceeds declared size: {path}")
            handle.seek(variable_size, 1)
            consumed_directory_bytes += record_size
            actual_entries += 1
            if actual_entries > budget.max_entries:
                raise RuntimeError(
                    f"ZIP entry count exceeds limit {budget.max_entries}: "
                    f"{path}: more than {budget.max_entries}"
                )
            total_uncompressed_bytes += uncompressed_size

    if consumed_directory_bytes != directory_size:
        raise RuntimeError(f"ZIP central-directory size does not match records: {path}")
    if actual_entries != total_entries:
        raise RuntimeError(
            f"ZIP EOCD entry count does not match central directory: "
            f"{path}: expected={total_entries}, actual={actual_entries}"
        )
    budget.consume_uncompressed(total_uncompressed_bytes, path)
    return actual_entries


def safe_zip_files(path: Path, budget: ZipBudget) -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo]]:
    require_file(path, "ZIP archive")
    expected_entry_count = preflight_zip(path, budget)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"invalid ZIP archive: {path}") from error
    seen_names: set[str] = set()
    files: list[zipfile.ZipInfo] = []
    try:
        all_entries = archive.infolist()
        if len(all_entries) != expected_entry_count:
            raise RuntimeError(
                f"ZIP central-directory entry count changed during inspection: {path}"
            )
        for info in all_entries:
            name = info.filename
            if "\\" in name:
                raise RuntimeError(f"ZIP entry uses a backslash path: {path}: {name!r}")
            pure_path = PurePosixPath(name)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                raise RuntimeError(f"unsafe ZIP entry path: {path}: {name!r}")
            folded_name = pure_path.as_posix().casefold()
            if folded_name in seen_names:
                raise RuntimeError(f"duplicate ZIP entry: {path}: {name!r}")
            seen_names.add(folded_name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"ZIP symlink entries are not allowed: {path}: {name!r}")
            if info.flag_bits & 0x1:
                raise RuntimeError(f"encrypted ZIP entries are not allowed: {path}: {name!r}")
            if info.file_size > budget.max_entry_bytes:
                raise RuntimeError(
                    f"ZIP entry size exceeds limit {budget.max_entry_bytes}: {path}: {name!r}"
                )
            if not info.is_dir():
                files.append(info)
    except BaseException:
        archive.close()
        raise
    return archive, files


def load_batch_images(
    path: Path,
    split: str,
    manifest_rows: list[ManifestRow],
    budget: ZipBudget,
    max_image_pixels: int,
) -> dict[str, bytes]:
    expected_rows = {row.image_name: row for row in manifest_rows if row.split == split}
    archive, files = safe_zip_files(path, budget)
    try:
        actual_entries: dict[str, zipfile.ZipInfo] = {}
        for info in files:
            pure_path = PurePosixPath(info.filename)
            if len(pure_path.parts) != 2 or pure_path.parts[0] != "images":
                raise RuntimeError(f"unexpected entry in batch {split} ZIP: {info.filename!r}")
            image_name = pure_path.name
            if image_name in actual_entries:
                raise RuntimeError(f"duplicate batch image name: {split}: {image_name!r}")
            actual_entries[image_name] = info
        if set(actual_entries) != set(expected_rows):
            missing = sorted(set(expected_rows) - set(actual_entries))
            extra = sorted(set(actual_entries) - set(expected_rows))
            raise RuntimeError(
                f"batch {split} image set does not match manifest: missing={missing}, extra={extra}"
            )
        images: dict[str, bytes] = {}
        for image_name, info in actual_entries.items():
            image_bytes = archive.read(info)
            actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
            expected_sha256 = expected_rows[image_name].source_sha256
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"batch image SHA-256 does not match manifest for {image_name!r}: "
                    f"expected={expected_sha256}, actual={actual_sha256}"
                )
            try:
                with _PILLOW_IMAGE_OPEN(io.BytesIO(image_bytes)) as image:
                    image_format = image.format
                    if image.width * image.height > max_image_pixels:
                        raise RuntimeError(
                            f"batch image exceeds pixel limit {max_image_pixels}: {image_name!r}"
                        )
                    image.verify()
                with _PILLOW_IMAGE_OPEN(io.BytesIO(image_bytes)) as image:
                    if image.width * image.height > max_image_pixels:
                        raise RuntimeError(
                            f"batch image exceeds pixel limit {max_image_pixels}: {image_name!r}"
                        )
                    image.load()
            except (UnidentifiedImageError, OSError) as error:
                raise RuntimeError(f"batch image cannot be decoded: {image_name!r}") from error
            except ModuleNotFoundError as error:
                if error.name != "pi_heif":
                    raise
                raise RuntimeError(
                    f"batch image cannot be decoded: {image_name!r}"
                ) from error
            expected_formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
            expected_format = expected_formats[Path(image_name).suffix.lower()]
            if image_format != expected_format:
                raise RuntimeError(
                    f"batch image format does not match suffix for {image_name!r}: "
                    f"expected={expected_format}, actual={image_format}"
                )
            images[image_name] = image_bytes
        return images
    finally:
        archive.close()


def normalize_class_names(document: object, path: Path) -> tuple[str, ...]:
    if not isinstance(document, dict):
        raise RuntimeError(f"CVAT data.yaml root must be an object: {path}")
    try:
        names = document["names"]
    except KeyError as error:
        raise RuntimeError(f"CVAT data.yaml is missing names: {path}") from error
    if isinstance(names, list):
        if not all(isinstance(name, str) for name in names):
            raise RuntimeError(f"CVAT data.yaml names list must contain strings: {path}")
        normalized = tuple(names)
    elif isinstance(names, dict):
        normalized_by_id: dict[int, str] = {}
        for raw_id, name in names.items():
            if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
                raise RuntimeError(f"CVAT data.yaml class ID is invalid: {raw_id!r}")
            try:
                class_id = int(raw_id)
            except ValueError as error:
                raise RuntimeError(f"CVAT data.yaml class ID is invalid: {raw_id!r}") from error
            if not isinstance(name, str):
                raise RuntimeError(f"CVAT data.yaml class name must be a string: {class_id}")
            if class_id in normalized_by_id:
                raise RuntimeError(f"CVAT data.yaml has duplicate class ID: {class_id}")
            normalized_by_id[class_id] = name
        if set(normalized_by_id) != set(range(len(ALL_TOOTH_NAMES))):
            raise RuntimeError(f"CVAT data.yaml class IDs do not cover 0..13: {path}")
        normalized = tuple(normalized_by_id[index] for index in range(len(ALL_TOOTH_NAMES)))
    else:
        raise RuntimeError(f"CVAT data.yaml names must be a list or object: {path}")
    if normalized != ALL_TOOTH_NAMES:
        raise RuntimeError(
            f"CVAT class names/order do not match: expected={ALL_TOOTH_NAMES}, actual={normalized}"
        )
    return normalized


def safe_relative_zip_path(base: PurePosixPath, value: str, label: str) -> PurePosixPath:
    if "\\" in value:
        raise RuntimeError(f"{label} uses a backslash path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} is not a safe relative path: {value!r}")
    return base / relative


def load_export_image_names(
    archive: zipfile.ZipFile,
    files: list[zipfile.ZipInfo],
    yaml_entry: zipfile.ZipInfo,
    yaml_document: object,
    split: str,
) -> set[str]:
    if not isinstance(yaml_document, dict):
        raise RuntimeError("CVAT data.yaml root must be an object.")
    if "path" in yaml_document:
        dataset_root_value = yaml_document["path"]
        if not isinstance(dataset_root_value, str):
            raise RuntimeError("CVAT data.yaml path must be a string.")
    else:
        dataset_root_value = "."
    yaml_parent = PurePosixPath(yaml_entry.filename).parent
    dataset_root = safe_relative_zip_path(
        yaml_parent,
        dataset_root_value,
        "CVAT data.yaml path",
    )

    subset_references: list[str] = []
    for key in ("train", "val", "test"):
        if key not in yaml_document:
            continue
        raw_value = yaml_document[key]
        if isinstance(raw_value, str):
            subset_references.append(raw_value)
        elif isinstance(raw_value, list) and all(isinstance(value, str) for value in raw_value):
            subset_references.extend(raw_value)
        else:
            raise RuntimeError(f"CVAT data.yaml {key} must be a string or list of strings.")
    if not subset_references:
        raise RuntimeError("CVAT data.yaml has no train/val/test subset reference.")

    entries_by_name = {PurePosixPath(info.filename).as_posix(): info for info in files}
    image_names: set[str] = set()
    folded_names: set[str] = set()
    for reference in subset_references:
        list_path = safe_relative_zip_path(dataset_root, reference, "CVAT subset reference")
        if list_path.suffix.lower() != ".txt":
            raise RuntimeError(f"CVAT subset reference must point to a .txt list: {reference!r}")
        list_name = list_path.as_posix()
        if list_name not in entries_by_name:
            raise RuntimeError(f"CVAT subset list was not found in export: {list_name!r}")
        try:
            list_text = archive.read(entries_by_name[list_name]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"CVAT subset list is not valid UTF-8: {list_name!r}") from error
        for line_number, raw_line in enumerate(list_text.splitlines(), start=1):
            image_reference = raw_line.strip()
            if image_reference == "":
                continue
            image_path = safe_relative_zip_path(
                PurePosixPath(),
                image_reference,
                f"CVAT subset image at {list_name}:{line_number}",
            )
            image_name = image_path.name
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise RuntimeError(
                    f"CVAT subset image has unsupported suffix at {list_name}:{line_number}: "
                    f"{image_reference!r}"
                )
            folded_name = image_name.casefold()
            if folded_name in folded_names:
                raise RuntimeError(f"duplicate image basename in CVAT subset lists: {image_name!r}")
            folded_names.add(folded_name)
            image_names.add(image_name)
    if not image_names:
        raise RuntimeError(f"CVAT {split} export has no images in its subset lists.")
    return image_names


def load_export_labels(
    path: Path,
    split: str,
    manifest_rows: list[ManifestRow],
    expected_images: dict[str, bytes],
    budget: ZipBudget,
) -> dict[str, str]:
    split_rows = {row.image_name: row for row in manifest_rows if row.split == split}
    expected_stems = {Path(name).stem: name for name in split_rows}
    archive, files = safe_zip_files(path, budget)
    try:
        yaml_entries = [info for info in files if PurePosixPath(info.filename).name == "data.yaml"]
        if len(yaml_entries) != 1:
            raise RuntimeError(f"CVAT export must contain exactly one data.yaml: {path}")
        try:
            yaml_document = yaml.safe_load(archive.read(yaml_entries[0]).decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise RuntimeError(f"CVAT data.yaml is invalid: {path}") from error
        normalize_class_names(yaml_document, path)
        export_image_names = load_export_image_names(
            archive,
            files,
            yaml_entries[0],
            yaml_document,
            split,
        )
        if export_image_names != set(split_rows):
            missing = sorted(set(split_rows) - export_image_names)
            extra = sorted(export_image_names - set(split_rows))
            raise RuntimeError(
                f"CVAT {split} export image set does not match manifest: "
                f"missing={missing}, extra={extra}"
            )

        export_image_entries: dict[str, zipfile.ZipInfo] = {}
        for info in files:
            pure_path = PurePosixPath(info.filename)
            if "images" not in pure_path.parts or pure_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            image_name = pure_path.name
            if image_name in export_image_entries:
                raise RuntimeError(f"duplicate image basename in CVAT export: {image_name!r}")
            export_image_entries[image_name] = info
        if set(export_image_entries) != set(split_rows):
            missing = sorted(set(split_rows) - set(export_image_entries))
            extra = sorted(set(export_image_entries) - set(split_rows))
            raise RuntimeError(
                f"CVAT {split} export image files do not match manifest: "
                f"missing={missing}, extra={extra}"
            )
        for image_name, info in export_image_entries.items():
            actual_sha256 = hashlib.sha256(archive.read(info)).hexdigest()
            expected_sha256 = hashlib.sha256(expected_images[image_name]).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"CVAT export image SHA-256 does not match source batch for {image_name!r}: "
                    f"expected={expected_sha256}, actual={actual_sha256}"
                )

        labels_by_stem: dict[str, str] = {}
        for info in files:
            pure_path = PurePosixPath(info.filename)
            if "labels" not in pure_path.parts or pure_path.suffix.lower() != ".txt":
                continue
            stem = pure_path.stem
            folded_stem = stem.casefold()
            if any(existing.casefold() == folded_stem for existing in labels_by_stem):
                raise RuntimeError(f"duplicate annotation basename in CVAT export: {stem!r}")
            if stem not in expected_stems:
                raise RuntimeError(f"unexpected annotation in {split} export: {stem!r}")
            try:
                labels_by_stem[stem] = archive.read(info).decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(f"annotation is not valid UTF-8: {info.filename}") from error

        finalized: dict[str, str] = {}
        for image_name, row in split_rows.items():
            stem = Path(image_name).stem
            label_text = labels_by_stem.get(stem, "")
            if row.annotation_status == "complete" and label_text.strip() == "":
                raise RuntimeError(f"missing annotation for complete image: {image_name!r}")
            if row.annotation_status == "negative" and label_text.strip() != "":
                raise RuntimeError(f"negative image has a non-empty annotation: {image_name!r}")
            if row.annotation_status != "excluded":
                finalized[image_name] = label_text
        return finalized
    finally:
        archive.close()


def polygon_area(coordinates: list[float]) -> float:
    points = list(zip(coordinates[0::2], coordinates[1::2], strict=True))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]), strict=True)
        )
    ) / 2.0


def validate_label_text(label_text: str, image_name: str) -> tuple[str, Counter[int]]:
    if label_text.strip() == "":
        return "", Counter()
    canonical_lines: list[str] = []
    class_counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(label_text.splitlines(), start=1):
        tokens = raw_line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
            raise RuntimeError(
                f"polygon must contain a class ID and at least 3 coordinate pairs: "
                f"{image_name}:{line_number}"
            )
        try:
            class_id = int(tokens[0])
        except ValueError as error:
            raise RuntimeError(f"class ID is not an integer: {image_name}:{line_number}") from error
        if class_id not in TARGET_CLASS_IDS:
            raise RuntimeError(f"class ID is not an annotation target: {image_name}:{line_number}: {class_id}")
        if class_counts[class_id] != 0:
            raise RuntimeError(f"duplicate class ID in one image: {image_name}: {class_id}")
        try:
            coordinates = [float(token) for token in tokens[1:]]
        except ValueError as error:
            raise RuntimeError(f"polygon coordinate is not numeric: {image_name}:{line_number}") from error
        if not all(math.isfinite(value) for value in coordinates):
            raise RuntimeError(f"polygon coordinate is not finite: {image_name}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise RuntimeError(f"polygon coordinate is outside [0, 1]: {image_name}:{line_number}")
        if polygon_area(coordinates) <= 1e-12:
            raise RuntimeError(f"polygon area is zero: {image_name}:{line_number}")
        canonical_coordinate_tokens = [format(coordinate, ".15g") for coordinate in coordinates]
        canonical_coordinates = [float(token) for token in canonical_coordinate_tokens]
        if polygon_area(canonical_coordinates) <= 1e-12:
            raise RuntimeError(
                f"polygon area becomes zero after normalization: {image_name}:{line_number}"
            )
        canonical_lines.append(" ".join((str(class_id), *canonical_coordinate_tokens)))
        class_counts[class_id] += 1
    return "\n".join(canonical_lines), class_counts


def dataset_path_value(output_dir: Path) -> str:
    return output_dir.resolve().as_posix()


def build_dataset(
    output_dir: Path,
    rows: list[ManifestRow],
    images_by_split: dict[str, dict[str, bytes]],
    labels_by_split: dict[str, dict[str, str]],
    *,
    input_hashes: dict[str, str],
) -> dict[str, object]:
    finalized_by_split: dict[str, list[FinalizedImage]] = {split: [] for split in SELECTED_SPLITS}
    excluded_counts = Counter(row.split for row in rows if row.annotation_status == "excluded")
    negative_counts = Counter(row.split for row in rows if row.annotation_status == "negative")
    instance_counts_by_split: dict[str, Counter[int]] = {
        split: Counter() for split in SELECTED_SPLITS
    }
    for row in rows:
        if row.annotation_status == "excluded":
            continue
        label_text, class_counts = validate_label_text(
            labels_by_split[row.split][row.image_name],
            row.image_name,
        )
        finalized_by_split[row.split].append(
            FinalizedImage(
                manifest=row,
                image_bytes=images_by_split[row.split][row.image_name],
                label_text=label_text,
                class_counts=class_counts,
            )
        )
        instance_counts_by_split[row.split].update(class_counts)

    for split in SELECTED_SPLITS:
        missing_classes = sorted(TARGET_CLASS_IDS - set(instance_counts_by_split[split]))
        if missing_classes:
            missing_names = [ALL_TOOTH_NAMES[class_id] for class_id in missing_classes]
            raise RuntimeError(f"{split} split has no instances for classes: {missing_names}")

    generation_dir = create_generation_directory(output_dir)
    try:
        for split in SELECTED_SPLITS:
            image_dir = generation_dir / "images" / split
            label_dir = generation_dir / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for item in finalized_by_split[split]:
                (image_dir / item.manifest.image_name).write_bytes(item.image_bytes)
                label_path = label_dir / f"{Path(item.manifest.image_name).stem}.txt"
                label_path.write_text(item.label_text, encoding="utf-8")

        dataset_yaml = {
            "path": dataset_path_value(output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(ALL_TOOTH_NAMES)),
        }
        dataset_yaml_path = generation_dir / "dataset_code_real.yaml"
        dataset_yaml_path.write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        metadata_path = generation_dir / "metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = (
                "split",
                "image_name",
                "patient_token",
                "checkup_token",
                "annotation_status",
                "view_tag",
                "lighting_tag",
                "oral_condition_tag",
                "source_sha256",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for split in SELECTED_SPLITS:
                for item in finalized_by_split[split]:
                    row = item.manifest
                    writer.writerow(
                        {
                            "split": row.split,
                            "image_name": row.image_name,
                            "patient_token": row.patient_token,
                            "checkup_token": row.checkup_token,
                            "annotation_status": row.annotation_status,
                            "view_tag": row.view_tag,
                            "lighting_tag": row.lighting_tag,
                            "oral_condition_tag": row.oral_condition_tag,
                            "source_sha256": row.source_sha256,
                        }
                    )

        total_instance_counts = Counter()
        for counts in instance_counts_by_split.values():
            total_instance_counts.update(counts)
        summary: dict[str, object] = {
            "images_by_split": {
                split: len(finalized_by_split[split]) for split in SELECTED_SPLITS
            },
            "negative_images_by_split": {
                split: negative_counts[split] for split in SELECTED_SPLITS
            },
            "excluded_images_by_split": {
                split: excluded_counts[split] for split in SELECTED_SPLITS
            },
            "instances_by_class": {
                ALL_TOOTH_NAMES[class_id]: total_instance_counts[class_id]
                for class_id in sorted(TARGET_CLASS_IDS)
            },
            "instances_by_split_and_class": {
                split: {
                    ALL_TOOTH_NAMES[class_id]: instance_counts_by_split[split][class_id]
                    for class_id in sorted(TARGET_CLASS_IDS)
                }
                for split in SELECTED_SPLITS
            },
            "class_names": list(ALL_TOOTH_NAMES),
            "annotation_classes": list(ANNOTATED_TOOTH_NAMES),
            "input_sha256": input_hashes,
            "dataset_yaml_sha256": sha256_file(dataset_yaml_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
        (generation_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        publish_generation(generation_dir, output_dir)
    finally:
        discard_generation(generation_dir)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    positive_limits = {
        "--max-zip-file-bytes": args.max_zip_file_bytes,
        "--max-zip-entries": args.max_zip_entries,
        "--max-zip-entry-bytes": args.max_zip_entry_bytes,
        "--max-total-uncompressed-bytes": args.max_total_uncompressed_bytes,
        "--max-image-pixels": args.max_image_pixels,
    }
    for option, value in positive_limits.items():
        if value <= 0:
            raise RuntimeError(f"{option} must be positive.")
    zip_budget = ZipBudget(
        max_file_bytes=args.max_zip_file_bytes,
        max_entries=args.max_zip_entries,
        max_entry_bytes=args.max_zip_entry_bytes,
        max_total_uncompressed_bytes=args.max_total_uncompressed_bytes,
    )
    repo_root = find_project_root(Path(__file__).resolve())
    validate_output_location(args.output_dir, repo_root)

    manifest_path = args.batch_dir / "annotation_manifest.csv"
    batch_summary_path = args.batch_dir / "summary.json"
    labels_json_path = args.batch_dir / "cvat_labels.json"
    batch_zip_paths = {
        split: args.batch_dir / f"cvat_{split}_images.zip" for split in SELECTED_SPLITS
    }
    for split, path in batch_zip_paths.items():
        require_file(path, f"batch {split} image ZIP")
    require_file(args.train_export, "train CVAT export")
    require_file(args.val_export, "val CVAT export")

    load_batch_summary(batch_summary_path, manifest_path, batch_zip_paths)
    load_labels_json(labels_json_path)
    rows = load_manifest(manifest_path)
    images_by_split = {
        split: load_batch_images(
            batch_zip_paths[split],
            split,
            rows,
            zip_budget,
            args.max_image_pixels,
        )
        for split in SELECTED_SPLITS
    }
    export_paths = {"train": args.train_export, "val": args.val_export}
    labels_by_split = {
        split: load_export_labels(
            export_paths[split],
            split,
            rows,
            images_by_split[split],
            zip_budget,
        )
        for split in SELECTED_SPLITS
    }
    input_hashes = {
        "manifest": sha256_file(manifest_path),
        "batch_summary": sha256_file(batch_summary_path),
        "cvat_labels": sha256_file(labels_json_path),
        "batch_train_images": sha256_file(batch_zip_paths["train"]),
        "batch_val_images": sha256_file(batch_zip_paths["val"]),
        "train_export": sha256_file(args.train_export),
        "val_export": sha256_file(args.val_export),
    }
    return build_dataset(
        args.output_dir,
        rows,
        images_by_split,
        labels_by_split,
        input_hashes=input_hashes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(f"dataset: {args.output_dir}")
    print(f"yaml: {args.output_dir / 'dataset_code_real.yaml'}")
    print(f"summary: {args.output_dir / 'summary.json'}")
    print(
        f"train_images={summary['images_by_split']['train']} "
        f"val_images={summary['images_by_split']['val']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
