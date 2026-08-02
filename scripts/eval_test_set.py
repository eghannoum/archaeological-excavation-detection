"""Final evaluation of all 9 trained models on the held-out test set.

Performs confidence threshold selection via 3-fold CV on the validation set,
then evaluates each model on the held-out test set (~40 images from 10 parent
scenes that were never seen during training or validation).

Outputs
-------
    outputs/test-set/test_results.json   — structured results for all models
    outputs/test-set/test_summary.csv    — tabular summary
    MLflow experiment 'final_test_eval' — all metrics logged

Supported models
----------------
    YOLO:      yolo26n, yolo26s, yolo26m, yolo26l, yolo26x, yolov8m, yolo11m
    Non-YOLO:  faster_rcnn, detr

Usage
-----
    python scripts/eval_test_set.py                         # all 9 models
    python scripts/eval_test_set.py --models yolo26n yolo26m faster_rcnn
    python scripts/eval_test_set.py --device cpu
    python scripts/eval_test_set.py --no-mlflow
    python scripts/eval_test_set.py --skip-threshold-sweep  # use default conf=0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = _PROJECT_ROOT
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_TEST = DATASET_DIR / "images" / "test"
LABELS_TEST = DATASET_DIR / "labels" / "test"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_VAL = DATASET_DIR / "labels" / "val"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test-set"

IMAGE_SIZE = 640
NUM_CLASSES = 1
CLASS_NAMES = ["hole"]
N_FOLDS = 3

# Model registry: name → {type, weights_path}
YOLO_MODELS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x", "yolov8m", "yolo11m"]
NON_YOLO_MODELS = ["faster_rcnn", "detr"]
ALL_MODELS = YOLO_MODELS + NON_YOLO_MODELS

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "yolo26n": {"type": "yolo", "weights": "runs/yolo26n/weights/best.pt"},
    "yolo26s": {"type": "yolo", "weights": "runs/yolo26s/weights/best.pt"},
    "yolo26m": {"type": "yolo", "weights": "runs/yolo26m/weights/best.pt"},
    "yolo26l": {"type": "yolo", "weights": "runs/yolo26l/weights/best.pt"},
    "yolo26x": {"type": "yolo", "weights": "runs/yolo26x/weights/best.pt"},
    "yolov8m": {"type": "yolo", "weights": "runs/yolov8m/weights/best.pt"},
    "yolo11m": {"type": "yolo", "weights": "runs/yolo11m/weights/best.pt"},
    "faster_rcnn": {"type": "faster_rcnn", "weights": "experiments/faster_rcnn/fold_0_best.pt"},
    "detr": {"type": "detr", "weights": "experiments/detr/fold_0_best.pt"},
}

# Known parameters and FLOPs from experiment results / model.info()
_MODEL_COMPLEXITY: dict[str, dict[str, float]] = {
    "yolo26n": {"params_M": 5.29, "flops_G": 8.7},
    "yolo26s": {"params_M": 9.95, "flops_G": 24.6},
    "yolo26m": {"params_M": 42.20, "flops_G": 65.7},
    "yolo26l": {"params_M": 50.70, "flops_G": 86.1},
    "yolo26x": {"params_M": 58.81, "flops_G": 193.4},
    "yolov8m": {"params_M": 49.70, "flops_G": 78.7},
    "yolo11m": {"params_M": 38.80, "flops_G": 67.6},
    "faster_rcnn": {"params_M": 41.50, "flops_G": 134.0},
    "detr": {"params_M": 41.27, "flops_G": 86.0},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_test_set")


# ---------------------------------------------------------------------------
# Scene-ID extraction (same as train_cv.py)
# ---------------------------------------------------------------------------

_SCENE_RE = re.compile(r"^(.+)_[^_]+\.\w+$")


def extract_parent_scene(filename: str) -> str:
    """Extract parent scene identifier from image filename."""
    m = _SCENE_RE.match(filename)
    if m:
        return m.group(1)
    return Path(filename).stem


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format labels → pixel xyxy boxes, scaled to IMAGE_SIZE."""
    boxes = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _cls, cx, cy, w, h = (
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    boxes_arr = np.array(boxes, dtype=np.float32)
    # Scale to IMAGE_SIZE
    scale_x = IMAGE_SIZE / img_w
    scale_y = IMAGE_SIZE / img_h
    boxes_arr[:, 0] *= scale_x
    boxes_arr[:, 1] *= scale_y
    boxes_arr[:, 2] *= scale_x
    boxes_arr[:, 3] *= scale_y
    return boxes_arr


def load_split_data(images_dir: Path, labels_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """Load image paths and GT boxes for a dataset split."""
    import cv2

    data = []
    if not images_dir.exists():
        logger.warning("Images directory not found: %s", images_dir)
        return data

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: %s", img_path)
            continue
        orig_h, orig_w = img.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        gt = load_gt_boxes(label_path, orig_w, orig_h)
        data.append((img_path, gt))

    logger.info("Loaded %d images from %s", len(data), images_dir)
    return data


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes → (N, M) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter_area
    return inter_area / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Per-image AP computation
# ---------------------------------------------------------------------------


def compute_map_at_iou(
    all_preds: list[dict[str, np.ndarray]],
    all_gts: list[np.ndarray],
    iou_threshold: float,
) -> float:
    """Compute mAP at a specific IoU threshold (simplified per-image AP).

    For each image, computes recall-based AP using precision-recall curve.
    """
    aps = []

    for pred, gt in zip(all_preds, all_gts, strict=False):
        n_gt = len(gt)
        scores = pred.get("scores", np.array([]))
        boxes = pred.get("boxes", np.array([]).reshape(0, 4))

        if n_gt == 0:
            aps.append(1.0 if len(scores) == 0 else 0.0)
            continue

        if len(scores) == 0:
            aps.append(0.0)
            continue

        # Sort by score descending
        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]

        # Match predictions to GT
        matched_gt = set()
        tp = np.zeros(len(scores))
        fp = np.zeros(len(scores))

        for i, pred_box in enumerate(boxes):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in range(n_gt):
                if gt_idx in matched_gt:
                    continue
                iou_val = compute_iou_matrix(pred_box.reshape(1, 4), gt[gt_idx : gt_idx + 1])[0, 0]
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1.0
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1.0

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall_curve = tp_cumsum / n_gt

        # AP = area under precision-recall curve (11-point interpolation)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_recall = precision_curve[recall_curve >= t]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall) / 11.0
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def compute_precision_recall_at_threshold(
    all_preds: list[dict[str, np.ndarray]],
    all_gts: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute aggregate precision and recall at a given IoU threshold."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(all_preds, all_gts, strict=False):
        n_gt = len(gt)
        scores = pred.get("scores", np.array([]))
        boxes = pred.get("boxes", np.array([]).reshape(0, 4))

        if n_gt == 0:
            total_fp += len(scores)
            continue
        if len(scores) == 0:
            total_fn += n_gt
            continue

        order = np.argsort(-scores)
        boxes = boxes[order]

        matched_gt = set()
        for pred_box in boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in range(n_gt):
                if gt_idx in matched_gt:
                    continue
                iou_val = compute_iou_matrix(pred_box.reshape(1, 4), gt[gt_idx : gt_idx + 1])[0, 0]
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1
        total_fn += n_gt - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# YOLO inference helpers
# ---------------------------------------------------------------------------


def run_yolo_inference_at_conf(
    model_name: str,
    weights_path: Path,
    data: list[tuple[Path, np.ndarray]],
    conf: float,
    device: str = "auto",
) -> tuple[list[dict[str, np.ndarray]], float]:
    """Run YOLO inference at a specific confidence threshold.

    Returns (per_image_predictions, avg_inference_time_ms).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    all_preds = []
    times = []

    import cv2

    for img_path, _gt in data:
        # Get original image size for coordinate scaling
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            logger.warning("Cannot read image: %s, skipping", img_path)
            all_preds.append(
                {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                }
            )
            continue
        orig_h, orig_w = orig_img.shape[:2]

        t0 = time.time()
        results = model.predict(
            source=str(img_path),
            save=False,
            conf=conf,
            iou=0.5,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.time() - t0) * 1000
        times.append(elapsed_ms)

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            # YOLO returns xyxy in original image pixel space.
            # GT boxes are in IMAGE_SIZE space — scale predictions to match.
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_scores = result.boxes.conf.cpu().numpy()
            scale_x = IMAGE_SIZE / orig_w
            scale_y = IMAGE_SIZE / orig_h
            pred_boxes[:, 0] *= scale_x
            pred_boxes[:, 1] *= scale_y
            pred_boxes[:, 2] *= scale_x
            pred_boxes[:, 3] *= scale_y
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)

        all_preds.append({"boxes": pred_boxes, "scores": pred_scores})

    avg_time = float(np.mean(times)) if times else 0.0
    return all_preds, avg_time


def run_yolo_speed_benchmark(
    weights_path: Path,
    data: list[tuple[Path, np.ndarray]],
    conf: float,
    device: str = "auto",
    warmup: int = 5,
) -> tuple[float, float]:
    """Benchmark YOLO inference speed. Returns (mean_ms, std_ms)."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    times = []

    for i, (img_path, _) in enumerate(data):
        t0 = time.time()
        _ = model.predict(
            source=str(img_path),
            save=False,
            conf=conf,
            iou=0.5,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.time() - t0) * 1000
        if i >= warmup:
            times.append(elapsed_ms)

    mean_ms = float(np.mean(times)) if times else 0.0
    std_ms = float(np.std(times)) if times else 0.0
    return mean_ms, std_ms


# ---------------------------------------------------------------------------
# Faster R-CNN inference
# ---------------------------------------------------------------------------


def run_faster_rcnn_inference_at_conf(
    weights_path: Path,
    data: list[tuple[Path, np.ndarray]],
    conf: float,
    device: str = "auto",
) -> tuple[list[dict[str, np.ndarray]], float]:
    """Run Faster R-CNN inference at a specific confidence threshold."""
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    logger.info("  Loading Faster R-CNN from %s", weights_path)
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)

    all_preds = []
    times = []

    with torch.no_grad():
        for img_path, _gt in data:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(dev)

            t0 = time.time()
            outputs = model(img_tensor)[0]
            elapsed_ms = (time.time() - t0) * 1000
            times.append(elapsed_ms)

            pred_boxes = outputs["boxes"].cpu().numpy()
            pred_scores = outputs["scores"].cpu().numpy()

            # Filter by confidence
            mask = pred_scores >= conf
            all_preds.append(
                {
                    "boxes": pred_boxes[mask],
                    "scores": pred_scores[mask],
                }
            )

    avg_time = float(np.mean(times)) if times else 0.0
    return all_preds, avg_time


def run_faster_rcnn_speed_benchmark(
    weights_path: Path,
    data: list[tuple[Path, np.ndarray]],
    conf: float,
    device: str = "auto",
    warmup: int = 5,
) -> tuple[float, float]:
    """Benchmark Faster R-CNN inference speed."""
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)

    times = []
    with torch.no_grad():
        for i, (img_path, _) in enumerate(data):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(dev)

            t0 = time.time()
            _ = model(img_tensor)
            elapsed_ms = (time.time() - t0) * 1000
            if i >= warmup:
                times.append(elapsed_ms)

    return float(np.mean(times)) if times else 0.0, float(np.std(times)) if times else 0.0


# ---------------------------------------------------------------------------
# DETR inference
# ---------------------------------------------------------------------------


def run_detr_inference_at_conf(
    weights_path: Path,
    data: list[tuple[Path, np.ndarray]],
    conf: float,
    device: str = "auto",
) -> tuple[list[dict[str, np.ndarray]], float]:
    """Run DETR inference at a specific confidence threshold."""
    import cv2
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_detr import create_model, postprocess_detr_output

    logger.info("  Loading DETR from %s", weights_path)
    model = create_model(num_classes=1, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)

    all_preds = []
    times = []

    with torch.no_grad():
        for img_path, _gt in data:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            # Letterbox resize (same as DETRDataset)
            scale = min(IMAGE_SIZE / orig_w, IMAGE_SIZE / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_x = int((IMAGE_SIZE - new_w) / 2.0)
            pad_y = int((IMAGE_SIZE - new_h) / 2.0)

            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            padded = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0, dtype=np.uint8)
            padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = img_resized
            img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(dev)

            t0 = time.time()
            outputs = model(img_tensor)
            elapsed_ms = (time.time() - t0) * 1000
            times.append(elapsed_ms)

            pred = postprocess_detr_output(
                outputs["pred_logits"][0].cpu(),
                outputs["pred_boxes"][0].cpu(),
                score_threshold=conf,
            )

            all_preds.append(
                {
                    "boxes": pred["boxes"].numpy(),
                    "scores": pred["scores"].numpy(),
                }
            )

    avg_time = float(np.mean(times)) if times else 0.0
    return all_preds, avg_time


def run_detr_speed_benchmark(
    weights_path: Path,
    data: list[tuple[Path, np.ndarray]],
    conf: float,
    device: str = "auto",
    warmup: int = 5,
) -> tuple[float, float]:
    """Benchmark DETR inference speed."""
    import cv2
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_detr import create_model

    model = create_model(num_classes=1, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)

    times = []
    with torch.no_grad():
        for i, (img_path, _) in enumerate(data):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            scale = min(IMAGE_SIZE / orig_w, IMAGE_SIZE / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_x = int((IMAGE_SIZE - new_w) / 2.0)
            pad_y = int((IMAGE_SIZE - new_h) / 2.0)

            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            padded = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0, dtype=np.uint8)
            padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = img_resized
            img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(dev)

            t0 = time.time()
            _ = model(img_tensor)
            elapsed_ms = (time.time() - t0) * 1000
            if i >= warmup:
                times.append(elapsed_ms)

    return float(np.mean(times)) if times else 0.0, float(np.std(times)) if times else 0.0


# ---------------------------------------------------------------------------
# Confidence threshold selection (3-fold CV on val set)
# ---------------------------------------------------------------------------


def select_optimal_confidence(
    model_name: str,
    model_type: str,
    weights_path: Path,
    val_data: list[tuple[Path, np.ndarray]],
    device: str = "auto",
    conf_candidates: list[float] | None = None,
) -> float:
    """Select optimal confidence threshold via sweep on val set.

    Uses 3-fold CV: splits val data into 3 scene-based folds, sweeps
    confidence thresholds on each fold, and returns the threshold that
    gives the best mean F1 across folds.

    Falls back to the candidate that gives highest F1 on the full val set
    if scene-based splitting fails.
    """
    if conf_candidates is None:
        conf_candidates = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.70,
            0.80,
            0.90,
        ]

    logger.info("  Threshold selection: sweeping %d candidates on val set", len(conf_candidates))

    # Split val data by parent scene into 3 folds
    scene_groups: dict[str, list[int]] = {}
    for idx, (img_path, _gt) in enumerate(val_data):
        scene = extract_parent_scene(img_path.name)
        scene_groups.setdefault(scene, []).append(idx)

    scenes = list(scene_groups.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(scenes)

    fold_scenes: list[list[str]] = [[] for _ in range(N_FOLDS)]
    for i, scene in enumerate(scenes):
        fold_scenes[i % N_FOLDS].append(scene)

    # For each conf candidate, compute mean F1 across folds
    best_conf = 0.25
    best_f1 = 0.0

    for conf in conf_candidates:
        fold_f1s = []
        for val_fold_idx in range(N_FOLDS):
            val_indices = []
            for s in fold_scenes[val_fold_idx]:
                val_indices.extend(scene_groups[s])

            fold_data = [val_data[i] for i in val_indices]

            # Run inference on this fold
            if model_type == "yolo":
                preds, _ = run_yolo_inference_at_conf(
                    model_name, weights_path, fold_data, conf, device
                )
            elif model_type == "faster_rcnn":
                preds, _ = run_faster_rcnn_inference_at_conf(weights_path, fold_data, conf, device)
            elif model_type == "detr":
                preds, _ = run_detr_inference_at_conf(weights_path, fold_data, conf, device)
            else:
                continue

            gts = [gt for _, gt in fold_data]
            p, r = compute_precision_recall_at_threshold(preds, gts, iou_threshold=0.5)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            fold_f1s.append(f1)

        mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_conf = conf

    logger.info("  Best conf=%.2f (F1=%.4f on val CV)", best_conf, best_f1)
    return best_conf


# ---------------------------------------------------------------------------
# Full model evaluation on test set
# ---------------------------------------------------------------------------


def evaluate_model_on_test(
    model_name: str,
    test_data: list[tuple[Path, np.ndarray]],
    conf_threshold: float,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate a single model on the test set.

    Returns a dict with all metrics: mAP50, mAP50-95, precision, recall, F1,
    inference_time_ms, params_M, flops_G, conf_threshold.
    """
    reg = _MODEL_REGISTRY.get(model_name)
    if reg is None:
        logger.error("Unknown model: %s", model_name)
        return {}

    weights_path = PROJECT_ROOT / reg["weights"]
    if not weights_path.exists():
        logger.error("Weights not found: %s", weights_path)
        return {}

    model_type = reg["type"]

    # --- Inference ---
    logger.info(
        "  Running %s inference on %d test images (conf=%.2f)...",
        model_name,
        len(test_data),
        conf_threshold,
    )

    if model_type == "yolo":
        preds, avg_time_ms = run_yolo_inference_at_conf(
            model_name, weights_path, test_data, conf_threshold, device
        )
    elif model_type == "faster_rcnn":
        preds, avg_time_ms = run_faster_rcnn_inference_at_conf(
            weights_path, test_data, conf_threshold, device
        )
    elif model_type == "detr":
        preds, avg_time_ms = run_detr_inference_at_conf(
            weights_path, test_data, conf_threshold, device
        )
    else:
        return {}

    gts = [gt for _, gt in test_data]

    # --- Speed benchmark (separate pass for accurate measurement) ---
    logger.info("  Benchmarking inference speed...")
    if model_type == "yolo":
        mean_ms, std_ms = run_yolo_speed_benchmark(weights_path, test_data, conf_threshold, device)
    elif model_type == "faster_rcnn":
        mean_ms, std_ms = run_faster_rcnn_speed_benchmark(
            weights_path, test_data, conf_threshold, device
        )
    elif model_type == "detr":
        mean_ms, std_ms = run_detr_speed_benchmark(weights_path, test_data, conf_threshold, device)
    else:
        mean_ms, std_ms = avg_time_ms, 0.0

    # --- Metrics ---
    mAP50 = compute_map_at_iou(preds, gts, iou_threshold=0.5)
    mAP50_95 = compute_map_at_iou_range(preds, gts, iou_min=0.5, iou_max=0.95, step=0.05)
    precision, recall = compute_precision_recall_at_threshold(preds, gts, iou_threshold=0.5)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- Model complexity ---
    complexity = _MODEL_COMPLEXITY.get(model_name, {"params_M": 0.0, "flops_G": 0.0})

    # Also try to get live params count for YOLO
    params_M = complexity["params_M"]
    flops_G = complexity["flops_G"]
    if model_type == "yolo":
        try:
            from ultralytics import YOLO

            m = YOLO(str(weights_path))
            info = m.info(verbose=False)
            if isinstance(info, (tuple, list)) and len(info) >= 2:
                params_M = info[1] / 1e6 if info[1] > 1000 else info[1]
        except Exception:
            pass

    metrics = {
        "model_name": model_name,
        "model_type": model_type,
        "conf_threshold": conf_threshold,
        "iou_threshold": 0.5,
        "mAP50": round(mAP50, 6),
        "mAP50-95": round(mAP50_95, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "F1": round(f1, 6),
        "inference_time_ms": round(mean_ms, 2),
        "inference_time_std_ms": round(std_ms, 2),
        "params_M": round(params_M, 2),
        "flops_G": round(flops_G, 1),
        "n_test_images": len(test_data),
        "weights_path": str(weights_path),
    }

    logger.info(
        "  %s: mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f F1=%.4f time=%.1fms",
        model_name,
        mAP50,
        mAP50_95,
        precision,
        recall,
        f1,
        mean_ms,
    )

    return metrics


def compute_map_at_iou_range(
    all_preds: list[dict[str, np.ndarray]],
    all_gts: list[np.ndarray],
    iou_min: float = 0.5,
    iou_max: float = 0.95,
    step: float = 0.05,
) -> float:
    """Compute mAP averaged over IoU thresholds from iou_min to iou_max."""
    thresholds = np.arange(iou_min, iou_max + step / 2, step)
    maps = []
    for iou_t in thresholds:
        m = compute_map_at_iou(all_preds, all_gts, float(iou_t))
        maps.append(m)
    return float(np.mean(maps)) if maps else 0.0


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_to_mlflow(
    all_results: list[dict[str, Any]],
    tracking_uri: str,
) -> None:
    """Log all model results to MLflow experiment 'final_test_eval'."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("final_test_eval")

        for result in all_results:
            if not result:
                continue

            model_name = result["model_name"]
            tags = {
                "model_name": model_name,
                "model_type": result["model_type"],
                "evaluation": "test_set",
                "data_split": "test",
            }

            _ = mlflow.start_run(run_name=f"test_eval_{model_name}", tags=tags)

            # Log params
            mlflow.log_params(
                {
                    "model/name": model_name,
                    "model/type": result["model_type"],
                    "model/params_M": result["params_M"],
                    "model/flops_G": result["flops_G"],
                    "eval/conf_threshold": result["conf_threshold"],
                    "eval/iou_threshold": result["iou_threshold"],
                    "eval/image_size": IMAGE_SIZE,
                    "eval/n_test_images": result["n_test_images"],
                    "eval/threshold_source": "3fold_cv_sweep",
                }
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "test/mAP50": result["mAP50"],
                    "test/mAP50-95": result["mAP50-95"],
                    "test/precision": result["precision"],
                    "test/recall": result["recall"],
                    "test/F1": result["F1"],
                    "test/inference_time_ms": result["inference_time_ms"],
                    "test/inference_time_std_ms": result["inference_time_std_ms"],
                }
            )

            mlflow.end_run()
            logger.info("  MLflow logged: test_eval_%s", model_name)

    except Exception as e:
        logger.warning("Failed to log to MLflow: %s", e)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def save_results(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save test_results.json and test_summary.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- test_results.json ---
    results_json = {
        "description": "Final test set evaluation for all 9 models",
        "test_set": str(IMAGES_TEST),
        "n_test_images": all_results[0]["n_test_images"] if all_results else 0,
        "image_size": IMAGE_SIZE,
        "iou_threshold": 0.5,
        "threshold_selection": "3-fold CV sweep on validation set",
        "models": all_results,
    }

    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info("Saved: %s", json_path)

    # --- test_summary.csv ---
    csv_path = output_dir / "test_summary.csv"
    fieldnames = [
        "model_name",
        "model_type",
        "conf_threshold",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "F1",
        "inference_time_ms",
        "inference_time_std_ms",
        "params_M",
        "flops_G",
        "n_test_images",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    logger.info("Saved: %s", csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final evaluation of all 9 models on the held-out test set.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model names to evaluate (default: all). Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--skip-threshold-sweep",
        action="store_true",
        help="Skip threshold sweep; use default conf=0.25 for all models",
    )
    parser.add_argument(
        "--default-conf",
        type=float,
        default=0.25,
        help="Default confidence threshold when skipping sweep (default: 0.25)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: outputs/test-set/)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    models_to_eval = args.models if args.models else ALL_MODELS
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    skip_sweep = args.skip_threshold_sweep
    default_conf = args.default_conf

    logger.info("=" * 70)
    logger.info("  FINAL TEST SET EVALUATION — ALL 9 MODELS")
    logger.info("=" * 70)
    logger.info("Models:  %s", models_to_eval)
    logger.info("Device:  %s", device)
    logger.info("Test:    %s", IMAGES_TEST)
    logger.info("Output:  %s", output_dir)
    logger.info(
        "Sweep:   %s", f"SKIPPED (using conf={default_conf:.2f})" if skip_sweep else "ENABLED"
    )
    logger.info("")

    # --- Load test data ---
    logger.info("Loading test set data...")
    test_data = load_split_data(IMAGES_TEST, LABELS_TEST)
    if not test_data:
        logger.error("No test images found at %s", IMAGES_TEST)
        sys.exit(1)
    logger.info(
        "Test set: %d images from %d parent scenes",
        len(test_data),
        len(set(extract_parent_scene(p.name) for p, _ in test_data)),
    )

    # --- Load val data for threshold selection ---
    val_data = []
    if not skip_sweep:
        logger.info("Loading validation set for threshold selection...")
        val_data = load_split_data(IMAGES_VAL, LABELS_VAL)
        if not val_data:
            logger.warning("No val images found — using default conf=%.2f", default_conf)
            skip_sweep = True

    # --- Evaluate each model ---
    all_results: list[dict[str, Any]] = []

    for model_name in models_to_eval:
        logger.info("")
        logger.info("-" * 60)
        logger.info("Evaluating: %s", model_name)
        logger.info("-" * 60)

        reg = _MODEL_REGISTRY.get(model_name)
        if reg is None:
            logger.error("Unknown model: %s — skipping", model_name)
            continue

        weights_path = PROJECT_ROOT / reg["weights"]
        if not weights_path.exists():
            logger.error("Weights not found: %s — skipping", weights_path)
            continue

        start = time.time()

        # --- Threshold selection ---
        if skip_sweep:
            conf_threshold = default_conf
            logger.info("  Using default conf=%.2f", conf_threshold)
        else:
            logger.info("  Running threshold sweep on val set...")
            conf_threshold = select_optimal_confidence(
                model_name=model_name,
                model_type=reg["type"],
                weights_path=weights_path,
                val_data=val_data,
                device=device,
            )

        # --- Test set evaluation ---
        result = evaluate_model_on_test(
            model_name=model_name,
            test_data=test_data,
            conf_threshold=conf_threshold,
            device=device,
        )

        elapsed = time.time() - start
        if result:
            result["elapsed_seconds"] = round(elapsed, 1)
            all_results.append(result)

        logger.info("  Completed in %.1f seconds", elapsed)

    # --- Save results ---
    if all_results:
        save_results(all_results, output_dir)

        # --- MLflow ---
        if not args.no_mlflow:
            from scripts.mlflow_utils import TRACKING_URI

            logger.info("Logging to MLflow...")
            log_to_mlflow(all_results, TRACKING_URI)

        # --- Print summary table ---
        print(f"\n{'=' * 100}")
        print("  TEST SET EVALUATION RESULTS — ALL MODELS")
        print(f"{'=' * 100}")
        header = (
            f"{'Model':<15} {'Type':<12} {'Conf':>5} {'mAP50':>7} {'mAP50-95':>8} "
            f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'Time(ms)':>9} {'Params(M)':>10} {'FLOPs(G)':>9}"
        )
        print(header)
        print("-" * 100)

        for r in all_results:
            print(
                f"{r['model_name']:<15} "
                f"{r['model_type']:<12} "
                f"{r['conf_threshold']:>5.2f} "
                f"{r['mAP50']:>7.4f} "
                f"{r['mAP50-95']:>8.4f} "
                f"{r['precision']:>6.4f} "
                f"{r['recall']:>6.4f} "
                f"{r['F1']:>6.4f} "
                f"{r['inference_time_ms']:>9.1f} "
                f"{r['params_M']:>10.2f} "
                f"{r['flops_G']:>9.1f}"
            )

        print("-" * 100)

        # Find best model
        if all_results:
            best = max(all_results, key=lambda x: x["mAP50"])
            fastest_yolo = min(
                [r for r in all_results if r["model_type"] == "yolo"],
                key=lambda x: x["inference_time_ms"],
                default=None,
            )
            print(f"\n  Best mAP50:     {best['model_name']} ({best['mAP50']:.4f})")
            if fastest_yolo:
                print(
                    f"  Fastest YOLO:   {fastest_yolo['model_name']} ({fastest_yolo['inference_time_ms']:.1f} ms)"
                )

            # Best F1
            best_f1 = max(all_results, key=lambda x: x["F1"])
            print(f"  Best F1:        {best_f1['model_name']} ({best_f1['F1']:.4f})")

        print(f"\n  Results saved to: {output_dir}")
        print(f"{'=' * 100}\n")
    else:
        logger.error("No models were evaluated successfully")


if __name__ == "__main__":
    main()
