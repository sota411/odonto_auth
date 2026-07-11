from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from code_photos import normalize_sha256, sha256_file

with redirect_stdout(sys.stderr):
    from extract_code_features import (
        FEATURE_NAMES,
        MODEL_CLASS_TO_TOOTH_INDEX,
        PREPROCESSING_FORMAT_VERSION,
        CropRecord,
        YOLO,
        build_hog,
        build_normalized_crop,
        build_resnet,
        extract_hog_batch,
        extract_resnet_batch,
        prepare_resnet_checkpoint,
        require_file,
        resize_mask,
        torch_device,
    )
from matching_core import TOOTH_NAMES, MatchingResult, score_pair
from tooth_template import ToothTemplate, clean_identifier, load_template


@dataclass(frozen=True)
class ValidatedInputs:
    query_id: str
    query_bgr: np.ndarray
    query_sha256: str
    weights_sha256: str
    template: ToothTemplate
    feature_device: torch.device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match one query image against one per-tooth subject template."
    )
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--query-id",
        help="Identifier emitted in the result JSON. Defaults to the query file stem.",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-query-sha256", required=True)
    parser.add_argument("--feature-name", choices=FEATURE_NAMES, required=True)
    parser.add_argument(
        "--resnet-weights",
        type=Path,
        help="Required local IMAGENET1K_V2 checkpoint for feature-name resnet50.",
    )
    parser.add_argument("--device", required=True, help="Ultralytics device, such as cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--min-common-teeth", type=int, default=1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files, hashes, metadata, image decoding, and device without loading models.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0:
        raise RuntimeError("--imgsz must be positive.")
    if not math.isfinite(args.conf) or not 0.0 < args.conf <= 1.0:
        raise RuntimeError("--conf must be finite and in (0, 1].")
    if not math.isfinite(args.iou) or not 0.0 < args.iou <= 1.0:
        raise RuntimeError("--iou must be finite and in (0, 1].")
    if args.crop_size <= 0:
        raise RuntimeError("--crop-size must be positive.")
    if not math.isfinite(args.crop_padding) or args.crop_padding < 0.0:
        raise RuntimeError("--crop-padding must be finite and non-negative.")
    if not 1 <= args.min_common_teeth <= len(TOOTH_NAMES):
        raise RuntimeError(
            f"--min-common-teeth must be from 1 to {len(TOOTH_NAMES)}."
        )


def verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_query_image(path: Path, expected_sha256: str) -> np.ndarray:
    verified_before = verify_sha256(path, expected_sha256, "query image")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"failed to decode query image as a three-channel image: {path}")
    verified_after = sha256_file(path)
    if verified_after != verified_before:
        raise RuntimeError(f"query image changed while it was being read: {path}")
    return image


def validate_extraction_contract(
    args: argparse.Namespace,
    template: ToothTemplate,
    weights_sha256: str,
) -> None:
    expected_values = {
        "segmentation_weights_sha256": weights_sha256,
        "segmentation_imgsz": args.imgsz,
        "segmentation_conf": args.conf,
        "segmentation_iou": args.iou,
        "crop_size": args.crop_size,
        "crop_padding": args.crop_padding,
        "preprocessing_format_version": PREPROCESSING_FORMAT_VERSION,
    }
    for field_name, expected in expected_values.items():
        actual = getattr(template, field_name)
        if actual != expected:
            raise RuntimeError(
                f"extraction contract mismatch for {field_name}: "
                f"expected={expected!r}, actual={actual!r}"
            )


def validate_inputs(args: argparse.Namespace) -> ValidatedInputs:
    validate_args(args)
    require_file(args.query, "query image")
    require_file(args.template, "template NPZ")
    require_file(args.weights, "segmentation weights")

    expected_weights_sha256 = normalize_sha256(
        args.expected_weights_sha256,
        "--expected-weights-sha256",
    )
    expected_query_sha256 = normalize_sha256(
        args.expected_query_sha256,
        "--expected-query-sha256",
    )
    weights_sha256 = verify_sha256(
        args.weights,
        expected_weights_sha256,
        "segmentation weight",
    )
    query_sha256 = verify_sha256(
        args.query,
        expected_query_sha256,
        "query image",
    )

    template = load_template(args.template)
    if template.tooth_names != TOOTH_NAMES:
        raise RuntimeError(
            "template tooth_names mismatch: "
            f"expected={TOOTH_NAMES}, actual={template.tooth_names}"
        )
    if template.feature_name != args.feature_name:
        raise RuntimeError(
            "feature_name mismatch: "
            f"expected={args.feature_name!r}, actual={template.feature_name!r}"
        )
    validate_extraction_contract(args, template, weights_sha256)
    if query_sha256 in template.registration_image_sha256:
        raise RuntimeError(
            "registered image reuse is forbidden: "
            f"query_sha256={query_sha256}"
        )
    query_bgr = load_query_image(args.query, query_sha256)

    feature_device = torch_device(args.device)
    if args.feature_name == "resnet50":
        if args.resnet_weights is None:
            raise RuntimeError("--resnet-weights is required for feature-name resnet50.")
        require_file(args.resnet_weights, "ResNet50 weights")
        prepare_resnet_checkpoint(args.resnet_weights)

    raw_query_id = args.query_id if args.query_id is not None else args.query.stem
    query_id = clean_identifier(raw_query_id, "query_id")
    return ValidatedInputs(
        query_id=query_id,
        query_bgr=query_bgr,
        query_sha256=query_sha256,
        weights_sha256=weights_sha256,
        template=template,
        feature_device=feature_device,
    )


def validate_model_class_names(model: YOLO) -> None:
    expected = {
        class_id: TOOTH_NAMES[tooth_index]
        for class_id, tooth_index in MODEL_CLASS_TO_TOOTH_INDEX.items()
    }
    try:
        actual = {
            class_id: model.names[class_id]
            for class_id in MODEL_CLASS_TO_TOOTH_INDEX
        }
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"segmentation model does not define the required class IDs: {tuple(expected)}"
        ) from exc
    if actual != expected:
        raise RuntimeError(
            f"unexpected segmentation class names: expected={expected}, actual={actual}"
        )


def select_best_masks(
    result: object,
    image_shape: tuple[int, int],
) -> dict[int, tuple[float, np.ndarray]]:
    boxes = result.boxes
    masks_result = result.masks
    if boxes is None or len(boxes) == 0:
        if masks_result is not None:
            mask_values = masks_result.data.detach().cpu().numpy()
            if len(mask_values) != 0:
                raise RuntimeError("segmentation masks were returned without boxes.")
        return {}
    if masks_result is None:
        raise RuntimeError("segmentation boxes were returned without masks.")

    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy().astype(float)
    masks = masks_result.data.detach().cpu().numpy()
    if classes.ndim != 1 or confidences.ndim != 1 or masks.ndim != 3:
        raise RuntimeError(
            "unexpected segmentation output shapes: "
            f"classes={classes.shape}, confidences={confidences.shape}, masks={masks.shape}"
        )
    if not (len(classes) == len(confidences) == len(masks)):
        raise RuntimeError("segmentation class, confidence, and mask counts must match.")
    if not np.all(np.isfinite(classes)) or not np.all(np.equal(classes, np.floor(classes))):
        raise RuntimeError("segmentation class IDs must be finite integers.")
    if (
        not np.all(np.isfinite(confidences))
        or np.any(confidences < 0.0)
        or np.any(confidences > 1.0)
    ):
        raise RuntimeError("segmentation confidences must be finite values in [0, 1].")

    best_by_tooth: dict[int, tuple[float, np.ndarray]] = {}
    for class_value, confidence, mask in zip(
        classes,
        confidences,
        masks,
        strict=True,
    ):
        class_id = int(class_value)
        if class_id not in MODEL_CLASS_TO_TOOTH_INDEX:
            continue
        tooth_index = MODEL_CLASS_TO_TOOTH_INDEX[class_id]
        if tooth_index in best_by_tooth and confidence <= best_by_tooth[tooth_index][0]:
            continue
        best_by_tooth[tooth_index] = (
            float(confidence),
            resize_mask(mask, image_shape),
        )
    return best_by_tooth


def build_query_embeddings(
    args: argparse.Namespace,
    inputs: ValidatedInputs,
    best_by_tooth: dict[int, tuple[float, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    image_rgb = cv2.cvtColor(inputs.query_bgr, cv2.COLOR_BGR2RGB)
    records = [
        CropRecord(
            checkup_uid=inputs.query_id,
            tooth_index=tooth_index,
            confidence=confidence,
            photo_reference=str(args.query),
            image=build_normalized_crop(
                image_rgb,
                mask,
                args.crop_size,
                args.crop_padding,
            ),
        )
        for tooth_index, (confidence, mask) in sorted(best_by_tooth.items())
    ]

    feature_dimension = inputs.template.embeddings.shape[1]
    query_embeddings = np.zeros(
        (len(TOOTH_NAMES), feature_dimension),
        dtype=np.float32,
    )
    query_present = np.zeros(len(TOOTH_NAMES), dtype=np.bool_)
    if not records:
        return query_embeddings, query_present

    if args.feature_name == "hog":
        features = extract_hog_batch(records, build_hog())
    elif args.feature_name == "resnet50":
        if args.resnet_weights is None:
            raise RuntimeError("--resnet-weights is required for feature-name resnet50.")
        model, transform = build_resnet(inputs.feature_device, args.resnet_weights)
        features = extract_resnet_batch(
            records,
            model,
            transform,
            inputs.feature_device,
        )
    else:
        raise RuntimeError(f"unsupported feature_name: {args.feature_name!r}")

    if len(features) != len(records):
        raise RuntimeError(
            "feature extractor result count mismatch: "
            f"expected={len(records)}, actual={len(features)}"
        )

    for record, feature in zip(records, features, strict=True):
        feature_array = np.asarray(feature, dtype=np.float32)
        if feature_array.shape != (feature_dimension,):
            raise RuntimeError(
                f"{args.feature_name} feature dimension mismatch for "
                f"{TOOTH_NAMES[record.tooth_index]}: "
                f"expected={feature_dimension}, actual={feature_array.shape}"
            )
        if not np.all(np.isfinite(feature_array)):
            raise RuntimeError(
                f"{args.feature_name} produced non-finite values for "
                f"{TOOTH_NAMES[record.tooth_index]}."
            )
        query_embeddings[record.tooth_index] = feature_array
        query_present[record.tooth_index] = True
    return query_embeddings, query_present


def match_query(args: argparse.Namespace, inputs: ValidatedInputs) -> MatchingResult:
    model = YOLO(str(args.weights))
    if sha256_file(args.weights) != inputs.weights_sha256:
        raise RuntimeError(f"segmentation weights changed while loading: {args.weights}")
    validate_model_class_names(model)
    results = tuple(
        model.predict(
            source=inputs.query_bgr,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            retina_masks=True,
            verbose=False,
            stream=False,
        )
    )
    if len(results) != 1:
        raise RuntimeError(
            f"expected one segmentation result for one query image; got {len(results)}."
        )
    best_by_tooth = select_best_masks(results[0], inputs.query_bgr.shape[:2])
    query_embeddings, query_present = build_query_embeddings(
        args,
        inputs,
        best_by_tooth,
    )
    return score_pair(
        template_embeddings=inputs.template.embeddings,
        query_embeddings=query_embeddings,
        template_present=inputs.template.present,
        query_present=query_present,
        min_common_teeth=args.min_common_teeth,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = validate_inputs(args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "query_id": inputs.query_id,
                    "template_subject_id": inputs.template.subject_id,
                    "feature_name": inputs.template.feature_name,
                    "query_sha256": inputs.query_sha256,
                    "weights_sha256": inputs.weights_sha256,
                    "validation_only": True,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0

    result = match_query(args, inputs)
    print(
        json.dumps(
            {
                "query_id": inputs.query_id,
                "template_subject_id": inputs.template.subject_id,
                "per_tooth_scores": result.per_tooth_scores,
                "fused_score": result.fused_score,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
