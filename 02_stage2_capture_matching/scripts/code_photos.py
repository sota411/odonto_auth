from __future__ import annotations

import hashlib
import re
from pathlib import Path


PHOTO_DIRECTORY = Path("Images") / "Photographs"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def parse_photographs(value: str, column: str, row_number: int) -> tuple[str, ...]:
    photographs = tuple(dict.fromkeys(item.strip() for item in value.split("|") if item.strip()))
    if not photographs:
        raise RuntimeError(f"{column} has no photograph references at CSV row {row_number}.")
    for reference in photographs:
        reference_path = Path(reference)
        if reference_path.is_absolute() or ".." in reference_path.parts:
            raise RuntimeError(f"unsafe photograph reference at CSV row {row_number}: {reference!r}")
    return photographs


def resolve_photo_reference(images_root: Path, reference: str) -> Path:
    root = images_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"COde images root was not found: {images_root}")
    photo_root = (root / PHOTO_DIRECTORY).resolve()
    try:
        photo_root.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"COde photograph directory escapes the images root: {photo_root}") from exc
    if not photo_root.is_dir():
        raise RuntimeError(f"COde photograph directory was not found: {photo_root}")

    reference_path = Path(reference)
    if reference_path.parts[:2] == PHOTO_DIRECTORY.parts:
        relative_path = Path(*reference_path.parts[2:])
    else:
        relative_path = reference_path
    resolved = (photo_root / relative_path).resolve()
    try:
        resolved.relative_to(photo_root)
    except ValueError as exc:
        raise RuntimeError(
            f"photograph reference escapes the COde photograph directory: {reference!r}"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"photograph was not found: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise RuntimeError(f"{label} must be a 64-character lowercase hexadecimal SHA-256 value.")
    return normalized
