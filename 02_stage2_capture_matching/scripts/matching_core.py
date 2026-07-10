from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TOOTH_NAMES = ("R1", "R2", "R3", "L1", "L2", "L3")
TOOTH_TYPES = TOOTH_NAMES


@dataclass(frozen=True)
class MatchingResult:
    per_tooth_scores: dict[str, float]
    fused_score: float
    common_teeth: tuple[str, ...]


def masked_square_crop(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int,
    padding_ratio: float = 0.1,
) -> np.ndarray:
    """Extract a zero-masked square crop and resize it to output_size."""
    _validate_image_and_mask(image, mask)
    _validate_output_size(output_size)
    validated_padding_ratio = _validate_padding_ratio(padding_ratio)

    if not np.any(mask):
        raise RuntimeError("mask must contain at least one foreground pixel.")

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    y_indices, x_indices = np.nonzero(mask)
    y_min = int(y_indices.min())
    y_max = int(y_indices.max()) + 1
    x_min = int(x_indices.min())
    x_max = int(x_indices.max()) + 1
    bounding_side = max(y_max - y_min, x_max - x_min)
    square_side = max(
        bounding_side,
        math.ceil(bounding_side * (1.0 + 2.0 * validated_padding_ratio)),
    )

    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0
    square_top = math.floor(center_y - square_side / 2.0)
    square_left = math.floor(center_x - square_side / 2.0)
    square = np.zeros((square_side, square_side, 3), dtype=image.dtype)

    source_top = max(square_top, 0)
    source_left = max(square_left, 0)
    source_bottom = min(square_top + square_side, image.shape[0])
    source_right = min(square_left + square_side, image.shape[1])
    destination_top = source_top - square_top
    destination_left = source_left - square_left
    destination_bottom = destination_top + source_bottom - source_top
    destination_right = destination_left + source_right - source_left
    square[destination_top:destination_bottom, destination_left:destination_right] = (
        masked_image[source_top:source_bottom, source_left:source_right]
    )

    return _resize_square_bilinear(square, output_size)


extract_masked_square = masked_square_crop


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a finite, one-dimensional embedding with unit L2 norm."""
    array = _as_float_array(embedding, "embedding")
    if array.ndim != 1 or array.size == 0:
        raise RuntimeError(
            f"embedding must have shape [D] with D > 0; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError("embedding must contain only finite values.")

    scale = float(np.max(np.abs(array)))
    if scale <= 0.0:
        raise RuntimeError("embedding L2 norm must be positive and finite.")
    scaled = array / scale
    norm = float(np.linalg.norm(scaled))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("embedding L2 norm must be positive and finite.")
    return scaled / norm


l2_normalize = normalize_embedding


def aggregate_embeddings(
    embeddings: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    """Confidence-average view embeddings and return a unit embedding."""
    embedding_matrix = _as_float_array(embeddings, "embeddings")
    confidence_array = _as_float_array(confidences, "confidences")

    if (
        embedding_matrix.ndim != 2
        or embedding_matrix.shape[0] == 0
        or embedding_matrix.shape[1] == 0
    ):
        raise RuntimeError(
            "embeddings must have shape [N, D] with N > 0 and D > 0; "
            f"got {embedding_matrix.shape}."
        )
    if not np.all(np.isfinite(embedding_matrix)):
        raise RuntimeError("embeddings must contain only finite values.")
    if np.any(np.max(np.abs(embedding_matrix), axis=1) <= 0.0):
        raise RuntimeError("each view embedding L2 norm must be positive.")
    if confidence_array.ndim != 1:
        raise RuntimeError(
            f"confidences must have shape [N]; got {confidence_array.shape}."
        )
    if confidence_array.shape[0] != embedding_matrix.shape[0]:
        raise RuntimeError(
            "embedding and confidence counts must match; "
            f"got {embedding_matrix.shape[0]} and {confidence_array.shape[0]}."
        )
    if not np.all(np.isfinite(confidence_array)) or np.any(confidence_array <= 0.0):
        raise RuntimeError("confidences must contain only positive finite values.")

    scaled_confidences = confidence_array / np.max(confidence_array)
    scaled_embeddings = embedding_matrix / np.max(np.abs(embedding_matrix))
    weighted_average = np.average(
        scaled_embeddings,
        axis=0,
        weights=scaled_confidences,
    )
    return normalize_embedding(weighted_average)


aggregate_view_embeddings = aggregate_embeddings


def score_pair(
    template_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    template_present: np.ndarray,
    query_present: np.ndarray,
    min_common_teeth: int = 1,
) -> MatchingResult:
    """Compute per-tooth cosine scores and their unweighted mean."""
    template_array = _validate_tooth_embeddings(
        template_embeddings,
        "template_embeddings",
    )
    query_array = _validate_tooth_embeddings(query_embeddings, "query_embeddings")
    if template_array.shape[1] != query_array.shape[1]:
        raise RuntimeError(
            "template and query embedding dimensions must match; "
            f"got {template_array.shape[1]} and {query_array.shape[1]}."
        )

    template_presence_array = _validate_presence(
        template_present,
        "template_present",
    )
    query_presence_array = _validate_presence(query_present, "query_present")
    _validate_min_common_teeth(min_common_teeth)

    common_mask = template_presence_array & query_presence_array
    common_indices = np.flatnonzero(common_mask)
    if common_indices.size < min_common_teeth:
        raise RuntimeError(
            "common teeth count is below min_common_teeth; "
            f"got {common_indices.size}, required {min_common_teeth}."
        )

    common_template = template_array[common_indices]
    common_query = query_array[common_indices]
    template_scales = np.max(np.abs(common_template), axis=1)
    query_scales = np.max(np.abs(common_query), axis=1)
    if np.any(template_scales <= 0.0) or np.any(query_scales <= 0.0):
        raise RuntimeError("common-tooth embedding L2 norms must be positive.")

    scaled_template = common_template / template_scales[:, None]
    scaled_query = common_query / query_scales[:, None]
    normalized_template = scaled_template / np.linalg.norm(
        scaled_template,
        axis=1,
        keepdims=True,
    )
    normalized_query = scaled_query / np.linalg.norm(
        scaled_query,
        axis=1,
        keepdims=True,
    )
    cosine_scores = np.sum(normalized_template * normalized_query, axis=1)
    normalized_scores = (np.clip(cosine_scores, -1.0, 1.0) + 1.0) / 2.0
    common_tooth_names = tuple(TOOTH_NAMES[index] for index in common_indices)
    per_tooth_scores = {
        tooth_type: float(score)
        for tooth_type, score in zip(
            common_tooth_names,
            normalized_scores,
            strict=True,
        )
    }
    return MatchingResult(
        per_tooth_scores=per_tooth_scores,
        fused_score=float(np.mean(normalized_scores)),
        common_teeth=common_tooth_names,
    )


def compute_matching_score(
    template_embeddings: np.ndarray,
    template_presence: np.ndarray,
    query_embeddings: np.ndarray,
    query_presence: np.ndarray,
    min_common_teeth: int = 1,
) -> MatchingResult:
    return score_pair(
        template_embeddings=template_embeddings,
        query_embeddings=query_embeddings,
        template_present=template_presence,
        query_present=query_presence,
        min_common_teeth=min_common_teeth,
    )


def _validate_image_and_mask(image: np.ndarray, mask: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise RuntimeError("image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise RuntimeError(
            f"image must have shape [H, W, 3] with H, W > 0; got {image.shape}."
        )
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise RuntimeError(f"image must have a real numeric dtype; got {image.dtype}.")
    if not np.all(np.isfinite(image)):
        raise RuntimeError("image must contain only finite values.")

    if not isinstance(mask, np.ndarray):
        raise RuntimeError("mask must be a NumPy array.")
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise RuntimeError(
            f"mask must have shape [H, W] and bool dtype; got {mask.shape}, {mask.dtype}."
        )
    if image.shape[:2] != mask.shape:
        raise RuntimeError(
            "image and mask shapes must match; "
            f"got {image.shape[:2]} and {mask.shape}."
        )


def _validate_output_size(output_size: int) -> None:
    if (
        isinstance(output_size, (bool, np.bool_))
        or not isinstance(output_size, (int, np.integer))
        or output_size <= 0
    ):
        raise RuntimeError(f"output_size must be a positive integer; got {output_size!r}.")


def _validate_padding_ratio(padding_ratio: float) -> float:
    if isinstance(padding_ratio, (bool, np.bool_)):
        raise RuntimeError(
            f"padding_ratio must be a non-negative finite number; got {padding_ratio!r}."
        )
    try:
        validated = float(padding_ratio)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"padding_ratio must be a non-negative finite number; got {padding_ratio!r}."
        ) from exc
    if not math.isfinite(validated) or validated < 0.0:
        raise RuntimeError(
            f"padding_ratio must be a non-negative finite number; got {padding_ratio!r}."
        )
    return validated


def _resize_square_bilinear(image: np.ndarray, output_size: int) -> np.ndarray:
    source_size = image.shape[0]
    if source_size == output_size:
        return image.copy()

    positions = (
        (np.arange(output_size, dtype=np.float64) + 0.5)
        * source_size
        / output_size
        - 0.5
    )
    lower = np.floor(positions).astype(np.intp)
    upper = lower + 1
    weights = positions - lower
    lower = np.clip(lower, 0, source_size - 1)
    upper = np.clip(upper, 0, source_size - 1)

    image_float = image.astype(np.float64)
    top_left = image_float[lower[:, None], lower[None, :]]
    top_right = image_float[lower[:, None], upper[None, :]]
    bottom_left = image_float[upper[:, None], lower[None, :]]
    bottom_right = image_float[upper[:, None], upper[None, :]]
    vertical_weights = weights[:, None, None]
    horizontal_weights = weights[None, :, None]
    top = top_left * (1.0 - horizontal_weights) + top_right * horizontal_weights
    bottom = (
        bottom_left * (1.0 - horizontal_weights) + bottom_right * horizontal_weights
    )
    resized = top * (1.0 - vertical_weights) + bottom * vertical_weights

    if np.issubdtype(image.dtype, np.integer):
        dtype_info = np.iinfo(image.dtype)
        resized = np.clip(np.rint(resized), dtype_info.min, dtype_info.max)
    return resized.astype(image.dtype)


def _as_float_array(value: object, name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric array-like data.") from exc
    if np.iscomplexobj(source):
        raise RuntimeError(f"{name} must contain real values.")
    try:
        return np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{name} must be numeric array-like data.") from exc


def _validate_tooth_embeddings(value: object, name: str) -> np.ndarray:
    array = _as_float_array(value, name)
    if array.ndim != 2 or array.shape[0] != len(TOOTH_NAMES) or array.shape[1] == 0:
        raise RuntimeError(
            f"{name} must have shape [{len(TOOTH_NAMES)}, D] with D > 0; "
            f"got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} must contain only finite values.")
    return array


def _validate_presence(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a bool array-like value.") from exc
    if array.shape != (len(TOOTH_NAMES),) or array.dtype != np.bool_:
        raise RuntimeError(
            f"{name} must have shape [{len(TOOTH_NAMES)}] and bool dtype; "
            f"got {array.shape}, {array.dtype}."
        )
    return array


def _validate_min_common_teeth(min_common_teeth: int) -> None:
    if (
        isinstance(min_common_teeth, (bool, np.bool_))
        or not isinstance(min_common_teeth, (int, np.integer))
        or not 1 <= min_common_teeth <= len(TOOTH_NAMES)
    ):
        raise RuntimeError(
            f"min_common_teeth must be an integer from 1 to {len(TOOTH_NAMES)}; "
            f"got {min_common_teeth!r}."
        )
