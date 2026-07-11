from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MAX_TARGET_TEETH = 6
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QualityThresholds:
    min_detected_teeth: int
    min_mean_detection_confidence: float
    min_laplacian_variance: float
    min_mean_luminance: float
    max_mean_luminance: float
    max_dark_pixel_fraction: float
    max_bright_pixel_fraction: float
    dark_pixel_threshold: int
    bright_pixel_threshold: int


@dataclass(frozen=True)
class QualityMeasurement:
    detected_teeth: int
    mean_detection_confidence: float
    laplacian_variance: float
    mean_luminance: float
    dark_pixel_fraction: float
    bright_pixel_fraction: float


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    reasons: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate capture quality from an image and segmentation summary."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detected-teeth", type=int, required=True)
    parser.add_argument("--mean-detection-confidence", type=float, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def require_finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise RuntimeError(f"{label} must be finite.")
    return float(value)


def require_integer(value: int, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(f"{label} must be an integer in [{minimum}, {maximum}].")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise RuntimeError(f"{label} must be an integer in [{minimum}, {maximum}].")
    return parsed


def validate_thresholds(thresholds: QualityThresholds) -> None:
    require_integer(
        thresholds.min_detected_teeth,
        "min_detected_teeth",
        0,
        MAX_TARGET_TEETH,
    )
    confidence = require_finite(
        thresholds.min_mean_detection_confidence,
        "min_mean_detection_confidence",
    )
    if not 0.0 <= confidence <= 1.0:
        raise RuntimeError("min_mean_detection_confidence must be in [0, 1].")
    if require_finite(thresholds.min_laplacian_variance, "min_laplacian_variance") < 0.0:
        raise RuntimeError("min_laplacian_variance must be non-negative.")
    minimum_luminance = require_finite(thresholds.min_mean_luminance, "min_mean_luminance")
    maximum_luminance = require_finite(thresholds.max_mean_luminance, "max_mean_luminance")
    if not 0.0 <= minimum_luminance < maximum_luminance <= 255.0:
        raise RuntimeError("mean luminance thresholds must satisfy 0 <= min < max <= 255.")
    for label, value in (
        ("max_dark_pixel_fraction", thresholds.max_dark_pixel_fraction),
        ("max_bright_pixel_fraction", thresholds.max_bright_pixel_fraction),
    ):
        if not 0.0 <= require_finite(value, label) <= 1.0:
            raise RuntimeError(f"{label} must be in [0, 1].")
    require_integer(thresholds.dark_pixel_threshold, "dark_pixel_threshold", 0, 255)
    require_integer(thresholds.bright_pixel_threshold, "bright_pixel_threshold", 0, 255)
    if thresholds.dark_pixel_threshold >= thresholds.bright_pixel_threshold:
        raise RuntimeError("dark_pixel_threshold must be less than bright_pixel_threshold.")


def parse_thresholds(payload: object) -> QualityThresholds:
    if not isinstance(payload, dict):
        raise RuntimeError("quality gate thresholds must be a JSON object.")
    expected = set(QualityThresholds.__dataclass_fields__)
    actual = set(payload)
    if actual != expected:
        raise RuntimeError(
            f"quality gate threshold keys must match exactly: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    try:
        thresholds = QualityThresholds(**payload)
    except TypeError as exc:
        raise RuntimeError("quality gate threshold values have invalid types.") from exc
    validate_thresholds(thresholds)
    return thresholds


def load_thresholds(path: Path, *, require_calibrated: bool = True) -> QualityThresholds:
    if not path.is_file():
        raise RuntimeError(f"quality gate config was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"quality gate config is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("quality gate config root must be a JSON object.")
    expected_root = {"schema_version", "status", "thresholds"}
    if set(payload) != expected_root:
        raise RuntimeError(
            f"quality gate config keys must match exactly: missing={sorted(expected_root - set(payload))}, extra={sorted(set(payload) - expected_root)}"
        )
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise RuntimeError(
            f"quality gate schema_version must be {CONFIG_SCHEMA_VERSION}, got {payload['schema_version']!r}."
        )
    status = payload["status"]
    if status not in {"calibrated", "uncalibrated"}:
        raise RuntimeError("quality gate status must be 'calibrated' or 'uncalibrated'.")
    thresholds = parse_thresholds(payload["thresholds"])
    if require_calibrated and status != "calibrated":
        raise RuntimeError("quality gate configuration is not calibrated.")
    return thresholds


def validate_measurement(measurement: QualityMeasurement) -> None:
    require_integer(measurement.detected_teeth, "detected_teeth", 0, MAX_TARGET_TEETH)
    confidence = require_finite(measurement.mean_detection_confidence, "mean_detection_confidence")
    if not 0.0 <= confidence <= 1.0:
        raise RuntimeError("mean_detection_confidence must be in [0, 1].")
    if require_finite(measurement.laplacian_variance, "laplacian_variance") < 0.0:
        raise RuntimeError("laplacian_variance must be non-negative.")
    luminance = require_finite(measurement.mean_luminance, "mean_luminance")
    if not 0.0 <= luminance <= 255.0:
        raise RuntimeError("mean_luminance must be in [0, 255].")
    for label, value in (
        ("dark_pixel_fraction", measurement.dark_pixel_fraction),
        ("bright_pixel_fraction", measurement.bright_pixel_fraction),
    ):
        if not 0.0 <= require_finite(value, label) <= 1.0:
            raise RuntimeError(f"{label} must be in [0, 1].")


def measure_image_quality(
    image: np.ndarray,
    *,
    detected_teeth: int,
    mean_detection_confidence: float,
    thresholds: QualityThresholds,
) -> QualityMeasurement:
    validate_thresholds(thresholds)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise RuntimeError("image must be a uint8 numpy array.")
    if image.size == 0:
        raise RuntimeError("image must not be empty.")
    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise RuntimeError("image must have shape HxW or HxWx3.")
    measurement = QualityMeasurement(
        detected_teeth=detected_teeth,
        mean_detection_confidence=mean_detection_confidence,
        laplacian_variance=float(cv2.Laplacian(grayscale, cv2.CV_64F).var()),
        mean_luminance=float(grayscale.mean()),
        dark_pixel_fraction=float(np.mean(grayscale <= thresholds.dark_pixel_threshold)),
        bright_pixel_fraction=float(np.mean(grayscale >= thresholds.bright_pixel_threshold)),
    )
    validate_measurement(measurement)
    return measurement


def evaluate_quality(
    measurement: QualityMeasurement,
    thresholds: QualityThresholds,
) -> QualityDecision:
    validate_measurement(measurement)
    validate_thresholds(thresholds)
    reasons: list[str] = []
    if measurement.detected_teeth < thresholds.min_detected_teeth:
        reasons.append("insufficient_detected_teeth")
    if measurement.mean_detection_confidence < thresholds.min_mean_detection_confidence:
        reasons.append("low_detection_confidence")
    if measurement.laplacian_variance < thresholds.min_laplacian_variance:
        reasons.append("blurry")
    if measurement.mean_luminance < thresholds.min_mean_luminance:
        reasons.append("underexposed")
    if measurement.mean_luminance > thresholds.max_mean_luminance:
        reasons.append("overexposed")
    if measurement.dark_pixel_fraction > thresholds.max_dark_pixel_fraction:
        reasons.append("excessive_dark_clipping")
    if measurement.bright_pixel_fraction > thresholds.max_bright_pixel_fraction:
        reasons.append("excessive_bright_clipping")
    return QualityDecision(accepted=not reasons, reasons=tuple(reasons))


def decision_payload(
    measurement: QualityMeasurement,
    decision: QualityDecision,
) -> dict[str, Any]:
    return {
        "accepted": decision.accepted,
        "reasons": list(decision.reasons),
        "measurements": asdict(measurement),
    }


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = load_thresholds(args.config)
    if not args.image.is_file():
        raise RuntimeError(f"input image was not found: {args.image}")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"input image could not be decoded: {args.image}")
    measurement = measure_image_quality(
        image,
        detected_teeth=args.detected_teeth,
        mean_detection_confidence=args.mean_detection_confidence,
        thresholds=thresholds,
    )
    decision = evaluate_quality(measurement, thresholds)
    output = json.dumps(decision_payload(measurement, decision), ensure_ascii=False, sort_keys=True)
    if args.output_json is not None:
        atomic_write_text(args.output_json, output + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
