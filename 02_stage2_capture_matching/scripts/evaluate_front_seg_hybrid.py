from __future__ import annotations

import atexit
import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_ULTRALYTICS_CONFIG_DIR: Path | None = None
if "YOLO_CONFIG_DIR" not in os.environ:
    _ULTRALYTICS_CONFIG_DIR = Path(tempfile.mkdtemp(prefix="mitou_clean_ultralytics_"))
    os.environ["YOLO_CONFIG_DIR"] = str(_ULTRALYTICS_CONFIG_DIR)
    atexit.register(shutil.rmtree, _ULTRALYTICS_CONFIG_DIR)

from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou, mask_iou


CLASS_FILTER = [0, 1, 2, 7, 8, 9]
CLASS_TO_CONTIGUOUS = {class_id: idx for idx, class_id in enumerate(CLASS_FILTER)}
NAMES = {
    0: "R1",
    1: "R2",
    2: "R3",
    3: "L1",
    4: "L2",
    5: "L3",
}
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


@dataclass
class Detection:
    cls: int
    score: float
    box: np.ndarray
    mask: np.ndarray


@dataclass
class GroundTruth:
    cls: np.ndarray
    boxes: np.ndarray
    masks: np.ndarray


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description="Evaluate single or hybrid front-tooth segmentation models.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "datasets" / "dataset_flont" / "test" / "images",
        help="Validation image directory.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "datasets" / "dataset_flont" / "test" / "labels",
        help="Validation label directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "hybrid"],
        default="single",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "01_stage1_real_image_extraction" / "experiments" / "v7_best" / "weights" / "best.pt",
        help="Weights path for single mode.",
    )
    parser.add_argument(
        "--base-weights",
        type=Path,
        help="Base weights path for hybrid mode.",
    )
    parser.add_argument(
        "--aux-weights",
        type=Path,
        help="Auxiliary weights path for hybrid mode.",
    )
    parser.add_argument("--imgsz", type=int, default=832, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold to fuse base and aux detections.")
    parser.add_argument("--box-alpha", type=float, default=0.7, help="Base box weight when blending matched boxes.")
    parser.add_argument("--score-mode", choices=["base", "max", "mean"], default="max", help="Score fusion mode.")
    parser.add_argument(
        "--mask-source",
        choices=["base", "aux", "union", "intersect"],
        default="aux",
        help="Mask fusion strategy for matched detections.",
    )
    parser.add_argument(
        "--add-unmatched-aux-score",
        type=float,
        default=1.1,
        help="Add unmatched auxiliary detections when score is at least this value. >1 disables the add path.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional path to dump metrics as JSON.",
    )
    return parser.parse_args()


def load_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise RuntimeError(f"image directory not found: {image_dir}")

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not image_paths:
        raise RuntimeError(f"validation images were not found: {image_dir}")
    return image_paths


def polygon_to_mask(width: int, height: int, points: list[float]) -> np.ndarray:
    polygon = [(points[idx], points[idx + 1]) for idx in range(0, len(points), 2)]
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def polygon_to_box(points: list[float]) -> np.ndarray:
    xs = np.asarray(points[0::2], dtype=np.float32)
    ys = np.asarray(points[1::2], dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def load_ground_truth(image_paths: list[Path], label_dir: Path) -> dict[str, GroundTruth]:
    if not label_dir.is_dir():
        raise RuntimeError(f"label directory not found: {label_dir}")

    ground_truth: dict[str, GroundTruth] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise RuntimeError(f"label file not found for image {image_path.name}: {label_path}")

        classes: list[int] = []
        boxes: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.strip().split()
            if not values:
                continue
            class_id = int(float(values[0]))
            if class_id not in CLASS_TO_CONTIGUOUS:
                continue
            coords = [float(value) for value in values[1:]]
            absolute = []
            for idx in range(0, len(coords), 2):
                absolute.append(coords[idx] * width)
                absolute.append(coords[idx + 1] * height)
            mask = polygon_to_mask(width, height, absolute)
            classes.append(CLASS_TO_CONTIGUOUS[class_id])
            boxes.append(polygon_to_box(absolute))
            masks.append(mask.astype(bool))

        if boxes:
            box_array = np.stack(boxes).astype(np.float32)
            mask_array = np.stack(masks).astype(bool)
            cls_array = np.asarray(classes, dtype=np.int64)
        else:
            box_array = np.zeros((0, 4), dtype=np.float32)
            mask_array = np.zeros((0, height, width), dtype=bool)
            cls_array = np.zeros((0,), dtype=np.int64)
        ground_truth[image_path.name] = GroundTruth(cls=cls_array, boxes=box_array, masks=mask_array)
    return ground_truth


def load_predictions(weights: Path, image_paths: list[Path], imgsz: int, conf: float, iou: float, device: str) -> dict[str, list[Detection]]:
    model = YOLO(str(weights))
    predictions: dict[str, list[Detection]] = {}
    for image_path in image_paths:
        result = model.predict(
            source=[str(image_path)],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            retina_masks=True,
            verbose=False,
            stream=False,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
            predictions[image_path.name] = detections
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy() > 0.5
        for box, score, class_id, mask in zip(boxes, scores, classes, masks, strict=True):
            if class_id not in CLASS_TO_CONTIGUOUS:
                continue
            detections.append(
                Detection(
                    cls=CLASS_TO_CONTIGUOUS[class_id],
                    score=float(score),
                    box=box.astype(np.float32),
                    mask=mask.astype(bool),
                )
            )
        predictions[image_path.name] = detections
    return predictions


def pairwise_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    iou = box_iou(torch.from_numpy(box_a[None, :]), torch.from_numpy(box_b[None, :]))
    return float(iou[0, 0].item())


def greedy_match(base: list[Detection], aux: list[Detection], match_iou: float) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    used_aux: set[int] = set()
    for base_idx, base_det in enumerate(base):
        best_aux = None
        best_iou = -1.0
        for aux_idx, aux_det in enumerate(aux):
            if aux_idx in used_aux or aux_det.cls != base_det.cls:
                continue
            iou = pairwise_box_iou(base_det.box, aux_det.box)
            if iou > best_iou:
                best_iou = iou
                best_aux = aux_idx
        if best_aux is not None and best_iou >= match_iou:
            used_aux.add(best_aux)
            matches.append((base_idx, best_aux))
    return matches


def fuse_score(base_score: float, aux_score: float, mode: str) -> float:
    if mode == "base":
        return base_score
    if mode == "mean":
        return (base_score + aux_score) / 2.0
    return max(base_score, aux_score)


def fuse_mask(base_mask: np.ndarray, aux_mask: np.ndarray, source: str) -> np.ndarray:
    if source == "base":
        return base_mask
    if source == "union":
        return np.logical_or(base_mask, aux_mask)
    if source == "intersect":
        return np.logical_and(base_mask, aux_mask)
    return aux_mask


def match_predictions(pred_classes: np.ndarray, true_classes: np.ndarray, iou: torch.Tensor) -> np.ndarray:
    correct = np.zeros((len(pred_classes), len(IOU_THRESHOLDS)), dtype=bool)
    if iou.numel() == 0:
        return correct

    class_mask = true_classes[:, None] == pred_classes
    iou = iou * torch.from_numpy(class_mask).to(iou.device)
    iou_np = iou.cpu().numpy()
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS.cpu().tolist()):
        matches = np.nonzero(iou_np >= threshold)
        if matches[0].size == 0:
            continue
        match_array = np.concatenate((np.stack(matches, axis=1), iou_np[matches[0], matches[1]][:, None]), axis=1)
        if matches[0].size > 1:
            match_array = match_array[match_array[:, 2].argsort()[::-1]]
            match_array = match_array[np.unique(match_array[:, 1], return_index=True)[1]]
            match_array = match_array[match_array[:, 2].argsort()[::-1]]
            match_array = match_array[np.unique(match_array[:, 0], return_index=True)[1]]
        correct[match_array[:, 1].astype(int), threshold_index] = True
    return correct


def hybrid_predictions(
    base_predictions: dict[str, list[Detection]],
    aux_predictions: dict[str, list[Detection]],
    match_iou: float,
    box_alpha: float,
    score_mode: str,
    mask_source: str,
    add_unmatched_aux_score: float,
) -> dict[str, list[Detection]]:
    fused: dict[str, list[Detection]] = {}
    for image_name, base in base_predictions.items():
        aux = aux_predictions[image_name]
        matches = greedy_match(base, aux, match_iou)
        used_base = {base_idx for base_idx, _ in matches}
        used_aux = {aux_idx for _, aux_idx in matches}

        image_detections: list[Detection] = []
        for base_idx, base_det in enumerate(base):
            matched_aux = next((aux_idx for matched_base, aux_idx in matches if matched_base == base_idx), None)
            if matched_aux is None:
                image_detections.append(base_det)
                continue
            aux_det = aux[matched_aux]
            blended_box = base_det.box * box_alpha + aux_det.box * (1.0 - box_alpha)
            blended_mask = fuse_mask(base_det.mask, aux_det.mask, mask_source)
            image_detections.append(
                Detection(
                    cls=base_det.cls,
                    score=fuse_score(base_det.score, aux_det.score, score_mode),
                    box=blended_box.astype(np.float32),
                    mask=blended_mask.astype(bool),
                )
            )

        if add_unmatched_aux_score <= 1.0:
            for aux_idx, aux_det in enumerate(aux):
                if aux_idx in used_aux or aux_det.score < add_unmatched_aux_score:
                    continue
                image_detections.append(aux_det)

        fused[image_name] = image_detections
    return fused


def evaluate_predictions(
    image_paths: list[Path],
    ground_truth: dict[str, GroundTruth],
    predictions: dict[str, list[Detection]],
) -> dict[str, float]:
    stats = {
        "tp_m": [],
        "tp": [],
        "conf": [],
        "pred_cls": [],
        "target_cls": [],
        "target_img": [],
    }

    for image_idx, image_path in enumerate(image_paths):
        gt = ground_truth[image_path.name]
        preds = predictions[image_path.name]

        target_cls = gt.cls
        target_img = np.full((len(target_cls),), image_idx, dtype=np.int64)
        if preds:
            pred_cls = np.asarray([det.cls for det in preds], dtype=np.int64)
            conf = np.asarray([det.score for det in preds], dtype=np.float32)
            pred_boxes = np.stack([det.box for det in preds]).astype(np.float32)
            pred_masks = np.stack([det.mask for det in preds]).astype(bool)
        else:
            pred_cls = np.zeros((0,), dtype=np.int64)
            conf = np.zeros((0,), dtype=np.float32)
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_masks = np.zeros((0, *gt.masks.shape[-2:]), dtype=bool)

        if len(pred_boxes) and len(gt.boxes):
            iou_b = box_iou(torch.from_numpy(gt.boxes), torch.from_numpy(pred_boxes))
            iou_m = mask_iou(
                torch.from_numpy(gt.masks.reshape(len(gt.masks), -1).astype(np.float32)),
                torch.from_numpy(pred_masks.reshape(len(pred_masks), -1).astype(np.float32)),
            )
            tp_b = match_predictions(pred_cls, target_cls, iou_b)
            tp_m = match_predictions(pred_cls, target_cls, iou_m)
        else:
            tp_b = np.zeros((len(pred_cls), 10), dtype=bool)
            tp_m = np.zeros((len(pred_cls), 10), dtype=bool)

        stats["tp_m"].append(tp_m)
        stats["tp"].append(tp_b)
        stats["conf"].append(conf)
        stats["pred_cls"].append(pred_cls)
        stats["target_cls"].append(target_cls)
        stats["target_img"].append(target_img)

    tp_b = np.concatenate(stats["tp"], axis=0) if stats["tp"] else np.zeros((0, 10), dtype=bool)
    tp_m = np.concatenate(stats["tp_m"], axis=0) if stats["tp_m"] else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(stats["conf"], axis=0) if stats["conf"] else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(stats["pred_cls"], axis=0) if stats["pred_cls"] else np.zeros((0,), dtype=np.int64)
    target_cls = np.concatenate(stats["target_cls"], axis=0) if stats["target_cls"] else np.zeros((0,), dtype=np.int64)

    box_stats = ap_per_class(tp_b, conf, pred_cls, target_cls, plot=False, names=NAMES)
    mask_stats = ap_per_class(tp_m, conf, pred_cls, target_cls, plot=False, names=NAMES)
    box_p, box_r, box_ap = box_stats[2], box_stats[3], box_stats[5]
    mask_p, mask_r, mask_ap = mask_stats[2], mask_stats[3], mask_stats[5]
    box = (
        float(box_p.mean()) if box_p.size else 0.0,
        float(box_r.mean()) if box_r.size else 0.0,
        float(box_ap[:, 0].mean()) if box_ap.size else 0.0,
        float(box_ap.mean()) if box_ap.size else 0.0,
    )
    seg = (
        float(mask_p.mean()) if mask_p.size else 0.0,
        float(mask_r.mean()) if mask_r.size else 0.0,
        float(mask_ap[:, 0].mean()) if mask_ap.size else 0.0,
        float(mask_ap.mean()) if mask_ap.size else 0.0,
    )
    fitness = ((box[2] * 0.1 + box[3] * 0.9) + (seg[2] * 0.1 + seg[3] * 0.9)) / 2.0
    return {
        "precision_B": float(box[0]),
        "recall_B": float(box[1]),
        "map50_B": float(box[2]),
        "map5095_B": float(box[3]),
        "precision_M": float(seg[0]),
        "recall_M": float(seg[1]),
        "map50_M": float(seg[2]),
        "map5095_M": float(seg[3]),
        "fitness": float(fitness),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"[{label}]")
    for key, value in metrics.items():
        print(f"  {key}={value:.6f}")


def main() -> None:
    args = parse_args()
    image_paths = load_image_paths(args.image_dir)
    ground_truth = load_ground_truth(image_paths, args.label_dir)

    if args.mode == "single":
        if args.weights is None:
            raise SystemExit("--weights is required in single mode.")
        predictions = load_predictions(args.weights, image_paths, args.imgsz, args.conf, args.iou, args.device)
        metrics = evaluate_predictions(image_paths, ground_truth, predictions)
        print_metrics(args.weights.name, metrics)
    else:
        if args.base_weights is None or args.aux_weights is None:
            raise SystemExit("--base-weights and --aux-weights are required in hybrid mode.")
        base_predictions = load_predictions(args.base_weights, image_paths, args.imgsz, args.conf, args.iou, args.device)
        aux_predictions = load_predictions(args.aux_weights, image_paths, args.imgsz, args.conf, args.iou, args.device)
        base_metrics = evaluate_predictions(image_paths, ground_truth, base_predictions)
        aux_metrics = evaluate_predictions(image_paths, ground_truth, aux_predictions)
        fused_predictions = hybrid_predictions(
            base_predictions=base_predictions,
            aux_predictions=aux_predictions,
            match_iou=args.match_iou,
            box_alpha=args.box_alpha,
            score_mode=args.score_mode,
            mask_source=args.mask_source,
            add_unmatched_aux_score=args.add_unmatched_aux_score,
        )
        fused_metrics = evaluate_predictions(image_paths, ground_truth, fused_predictions)
        print_metrics("base", base_metrics)
        print_metrics("aux", aux_metrics)
        print_metrics("hybrid", fused_metrics)
        metrics = fused_metrics

    if args.dump_json is not None:
        args.dump_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
