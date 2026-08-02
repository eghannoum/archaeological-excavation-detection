"""Compute Precision-Recall curves and F1-Confidence curves for all 9 models.

Runs inference on the val set, computes mAP50/75/50-95, draws PR and F1 plots,
and logs metrics to MLflow.

Usage
-----
    python scripts/eval_pr_curves.py
    python scripts/eval_pr_curves.py --output outputs/pr-curves
    python scripts/eval_pr_curves.py --models yolo26n yolo26m faster_rcnn
    python scripts/eval_pr_curves.py --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.mlflow_utils import (  # noqa: E402  # intentional: import after sys.path setup
    TRACKING_URI,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

IMAGES_VAL = _PROJECT_ROOT / "dataset" / "images" / "val"
LABELS_VAL = _PROJECT_ROOT / "dataset" / "labels" / "val"
IMAGE_SIZE = 640  # inference resolution for all models
NUM_CLASSES = 1  # single class: 'hole'
CLASS_NAMES = ["hole"]

# Model registry: name → weight path(s)
# YOLO models use fold_0_best.pt from experiments/ (single-fold retrained models)
# or weights/best.pt from runs/ (training output).
# Non-YOLO models use fold_0_best.pt from experiments/ (state dicts).
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

# Display order: YOLO family (n→x), then other YOLO, then non-YOLO
ALL_MODEL_NAMES = [
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    "yolov8m",
    "yolo11m",
    "faster_rcnn",
    "detr",
]

# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------


def load_gt_from_yolo_labels(
    label_dir: Path,
    images_dir: Path,
) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    """Load YOLO-format labels → dict of filename stem → (boxes [N,4] xyxy, (img_w, img_h)).

    Label format: ``class_id cx cy w h`` (normalised 0-1).
    Output: ``[x1, y1, x2, y2]`` in **original image pixel** coordinates
    (matching what Ultralytics predict() returns).
    """
    gt: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    if not label_dir.exists():
        return gt

    for txt_path in sorted(label_dir.glob("*.txt")):
        stem = txt_path.stem
        # Find corresponding image to get real dimensions
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        boxes = []
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                _cls, cx, cy, w, h = float(parts[0]), *map(float, parts[1:5])
                x1 = (cx - 0.5 * w) * img_w
                y1 = (cy - 0.5 * h) * img_h
                x2 = (cx + 0.5 * w) * img_w
                y2 = (cy + 0.5 * h) * img_h
                boxes.append([x1, y1, x2, y2])
        gt[stem] = (
            np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            (img_w, img_h),
        )
    return gt


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes [N,4] and [M,4] (xyxy).

    Returns ``[N, M]`` IoU matrix.
    """
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    # Intersection
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions to GT greedily (highest confidence first).

    Returns
    -------
    is_correct : bool array [N_pred] — True if matched to a GT at >= iou_threshold
    matched_gt : bool array [N_gt]   — True if that GT was matched
    """
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    is_correct = np.zeros(n_pred, dtype=bool)
    matched_gt = np.zeros(n_gt, dtype=bool)

    if n_pred == 0 or n_gt == 0:
        return is_correct, matched_gt

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # [N_pred, N_gt]

    # Greedy matching: sort by score descending, match each to best available GT
    order = np.argsort(-pred_scores)
    for idx in order:
        if n_gt == 0:
            break
        # Find best GT for this prediction
        ious = iou_matrix[idx]
        best_gt = -1
        best_iou = iou_threshold - 1e-6  # must exceed threshold
        for gt_idx in range(n_gt):
            if matched_gt[gt_idx]:
                continue
            if ious[gt_idx] > best_iou:
                best_iou = ious[gt_idx]
                best_gt = gt_idx
        if best_gt >= 0:
            is_correct[idx] = True
            matched_gt[best_gt] = True

    return is_correct, matched_gt


# ---------------------------------------------------------------------------
# PR curve computation
# ---------------------------------------------------------------------------


def compute_pr_curve(
    all_confs: np.ndarray,
    all_correct: np.ndarray,
    n_gt_total: int,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve across all images.

    Parameters
    ----------
    all_confs : confidence scores for all predictions across images [N_total]
    all_correct : boolean correctness flags [N_total]
    n_gt_total : total number of ground-truth boxes across all images
    n_points : number of recall points to sample

    Returns
    -------
    recall : [n_points]
    precision : [n_points]
    """
    if len(all_confs) == 0:
        return np.zeros(n_points), np.zeros(n_points)

    # Sort by confidence descending
    order = np.argsort(-all_confs)
    all_correct_sorted = all_correct[order]

    tp_cumsum = np.cumsum(all_correct_sorted).astype(np.float64)
    n_detected = np.arange(1, len(all_confs) + 1, dtype=np.float64)

    recall = tp_cumsum / max(n_gt_total, 1)
    precision = tp_cumsum / n_detected

    # VOC-style interpolation: max precision for recall >= current point
    # Go backwards (from high recall to low) and take max
    interpolated_precision = np.zeros_like(precision)
    interpolated_precision[-1] = precision[-1]
    for i in range(len(precision) - 2, -1, -1):
        interpolated_precision[i] = max(precision[i], interpolated_precision[i + 1])

    # Sample at evenly-spaced recall points
    recall_points = np.linspace(0, 1, n_points)
    sampled_precision = np.zeros(n_points)
    for i, r in enumerate(recall_points):
        mask = recall >= r
        if np.any(mask):
            sampled_precision[i] = np.max(interpolated_precision[mask])

    return recall_points, sampled_precision


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision as area under the interpolated PR curve."""
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz is None:
        # Manual trapezoidal integration fallback
        return float(np.sum((recall[1:] - recall[:-1]) * 0.5 * (precision[1:] + precision[:-1])))
    return float(trapz(precision, recall))


def compute_map_at_iou(
    all_predictions: list[dict[str, Any]],
    gt_by_image: dict[str, tuple[np.ndarray, tuple[int, int]]],
    iou_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Compute AP at a given IoU threshold across all images.

    Returns
    -------
    ap : float
    recall_curve : np.ndarray [n_points]
    precision_curve : np.ndarray [n_points]
    n_gt : int total GT boxes
    """
    # Accumulate all predictions across images
    all_confs_list: list[float] = []
    all_correct_list: list[bool] = []
    n_gt_total = 0

    for pred in all_predictions:
        stem = pred["stem"]
        pred_boxes = pred["boxes"]  # [N, 4] xyxy
        pred_scores = pred["scores"]  # [N]
        gt_entry = gt_by_image.get(stem)
        if gt_entry is not None:
            gt_boxes, _img_dim = gt_entry
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
        n_gt_total += gt_boxes.shape[0]

        if pred_boxes.shape[0] == 0:
            continue

        is_correct, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        all_confs_list.extend(pred_scores.tolist())
        all_correct_list.extend(is_correct.tolist())

    all_confs = (
        np.array(all_confs_list, dtype=np.float32)
        if all_confs_list
        else np.array([], dtype=np.float32)
    )
    all_correct = (
        np.array(all_correct_list, dtype=bool) if all_correct_list else np.array([], dtype=bool)
    )

    recall_curve, precision_curve = compute_pr_curve(all_confs, all_correct, n_gt_total)
    ap = compute_ap(recall_curve, precision_curve)
    return ap, recall_curve, precision_curve, n_gt_total


# ---------------------------------------------------------------------------
# F1-confidence curve
# ---------------------------------------------------------------------------


def compute_f1_confidence(
    all_confs: np.ndarray,
    all_correct: np.ndarray,
    n_gt: int,
    n_thresholds: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute F1 vs confidence threshold curve.

    For each threshold ``t`` in [0, 1]:
    - Predictions with conf >= t are treated as positive
    - Compute precision, recall, F1

    Returns
    -------
    thresholds : [n_thresholds]
    f1_scores : [n_thresholds]
    precisions : [n_thresholds]
    recalls : [n_thresholds]
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    f1s = np.zeros(n_thresholds)
    precs = np.zeros(n_thresholds)
    recs = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        mask = all_confs >= t
        n_pos = mask.sum()
        n_correct = all_correct[mask].sum() if n_pos > 0 else 0
        precision = n_correct / max(n_pos, 1)
        recall = n_correct / max(n_gt, 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s[i] = f1
        precs[i] = precision
        recs[i] = recall

    return thresholds, f1s, precs, recs


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------


def run_yolo_inference(
    weights_path: str,
    image_paths: list[Path],
    conf: float = 0.001,
    device: str = "cpu",
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """Run Ultralytics YOLO inference on a list of images.

    Returns list of dicts: {stem, boxes [N,4] xyxy, scores [N]}
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.predict(
        source=[str(p) for p in image_paths],
        save=False,
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    predictions = []
    for img_path, result in zip(image_paths, results, strict=False):
        stem = img_path.stem
        if result.boxes is not None and len(result.boxes) > 0:
            # Narrow the Tensor | ndarray union before .cpu(); ultralytics always
            # returns tensors here, but the stubs leave the alternative open.
            boxes_xyxy = result.boxes.xyxy
            boxes_conf = result.boxes.conf
            if isinstance(boxes_xyxy, torch.Tensor):
                boxes_xyxy = boxes_xyxy.cpu().numpy()
            if isinstance(boxes_conf, torch.Tensor):
                boxes_conf = boxes_conf.cpu().numpy()
            boxes = np.asarray(boxes_xyxy, dtype=np.float32)
            scores = np.asarray(boxes_conf, dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        predictions.append({"stem": stem, "boxes": boxes, "scores": scores})

    return predictions


# ---------------------------------------------------------------------------
# Faster R-CNN inference
# ---------------------------------------------------------------------------


def run_faster_rcnn_inference(
    weights_path: str,
    image_paths: list[Path],
    device: str = "cpu",
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """Run torchvision Faster R-CNN inference.

    Returns list of dicts: {stem, boxes [N,4] xyxy, scores [N]}
    """
    import torchvision.transforms.functional as F_t
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = fasterrcnn_resnet50_fpn(num_classes=2)  # background + hole
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = F_t.to_tensor(img).to(device)

            outputs = model([img_tensor])[0]
            boxes = outputs["boxes"].cpu().numpy().astype(np.float32)
            scores = outputs["scores"].cpu().numpy().astype(np.float32)
            predictions.append({"stem": img_path.stem, "boxes": boxes, "scores": scores})

    return predictions


# ---------------------------------------------------------------------------
# DETR inference
# ---------------------------------------------------------------------------


def run_detr_inference(
    weights_path: str,
    image_paths: list[Path],
    device: str = "cpu",
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """Run DETR ResNet-50 inference.

    Preprocessing matches the training DETRDataset exactly:
    resize to fit within imgsz×imgsz, pad to imgsz×imgsz, normalise to [0,1].

    The model outputs normalised [0,1] boxes relative to the input tensor size.
    We convert them back to original image coords by undoing the resize+pad.

    Returns list of dicts: {stem, boxes [N,4] xyxy, scores [N]}
    """
    # Import DETR's NestedTensor from the cached hub code
    import importlib.util

    hub_dir = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_detr_main"
    spec = importlib.util.spec_from_file_location("misc", hub_dir / "util" / "misc.py")
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load DETR helper module 'util/misc.py' from the torch hub cache. "
            f"Expected at: {hub_dir / 'util' / 'misc.py'}"
        )
    misc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(misc)
    NestedTensor = misc.NestedTensor

    # Load DETR model with 1 class (hole) + no-object
    model = torch.hub.load(
        "facebookresearch/detr",
        "detr_resnet50",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)  # strict=False for COCO→custom
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            # --- Replicate DETRDataset preprocessing ---
            scale = min(imgsz / orig_w, imgsz / orig_h)
            new_w = orig_w * scale
            new_h = orig_h * scale
            pad_x = (imgsz - new_w) / 2.0
            pad_y = (imgsz - new_h) / 2.0

            resized_img = img.resize((int(new_w), int(new_h)), Image.Resampling.BILINEAR)
            padded_img = Image.new("RGB", (imgsz, imgsz), (0, 0, 0))
            padded_img.paste(resized_img, (int(pad_x), int(pad_y)))

            # Normalize to [0, 1] (matches DETRDataset)
            img_tensor = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1).float() / 255.0

            # Create NestedTensor (mask=0 for valid pixels, 1 for padding)
            mask = torch.zeros(1, imgsz, imgsz, dtype=torch.bool)
            nested = NestedTensor(img_tensor.unsqueeze(0).to(device), mask.to(device))

            outputs = model(nested)

            pred_logits = outputs["pred_logits"][0].cpu()
            pred_boxes = outputs["pred_boxes"][0].cpu()

            probs = torch.nn.functional.softmax(pred_logits, dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)

            # Convert normalised cx,cy,w,h → original image pixel coords
            # Model outputs normalised [0,1] relative to input tensor (imgsz×imgsz)
            cx, cy, w, h = pred_boxes.unbind(-1)
            x1_norm = cx - 0.5 * w
            y1_norm = cy - 0.5 * h
            x2_norm = cx + 0.5 * w
            y2_norm = cy + 0.5 * h

            # [0,1] → padded pixel coords
            x1_padded = x1_norm * imgsz
            y1_padded = y1_norm * imgsz
            x2_padded = x2_norm * imgsz
            y2_padded = y2_norm * imgsz

            # Undo padding + resize → original image coords
            x1_orig = (x1_padded - pad_x) / scale
            y1_orig = (y1_padded - pad_y) / scale
            x2_orig = (x2_padded - pad_x) / scale
            y2_orig = (y2_padded - pad_y) / scale

            boxes_xyxy = torch.stack(
                [
                    torch.tensor(x1_orig).clamp(0, orig_w),
                    torch.tensor(y1_orig).clamp(0, orig_h),
                    torch.tensor(x2_orig).clamp(0, orig_w),
                    torch.tensor(y2_orig).clamp(0, orig_h),
                ],
                dim=-1,
            )

            predictions.append(
                {
                    "stem": img_path.stem,
                    "boxes": boxes_xyxy.numpy().astype(np.float32),
                    "scores": scores.numpy().astype(np.float32),
                }
            )

    return predictions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Okabe-Ito inspired)
_COLORS = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # gray
]


def plot_pr_curve_single(
    recall: np.ndarray,
    precision: np.ndarray,
    ap: float,
    model_name: str,
    optimal_conf: float,
    optimal_f1: float,
    output_path: Path,
) -> None:
    """Plot PR curve for a single model with optimal F1 point marked."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=_COLORS[0], linewidth=2.0, label=f"AP = {ap:.4f}")

    # Mark optimal F1 point
    # Find recall/precision at optimal conf from F1 curve (approximate)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        f"Precision-Recall Curve — {model_name}\n"
        f"Optimal conf={optimal_conf:.3f}, F1={optimal_f1:.4f}",
        fontsize=13,
    )
    ax.set_xlim((0, 1.05))
    ax.set_ylim((0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_path)


def plot_f1_curve_single(
    thresholds: np.ndarray,
    f1_scores: np.ndarray,
    optimal_idx: int,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot F1-confidence curve for a single model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1_scores, color=_COLORS[0], linewidth=2.0)

    # Mark optimal point
    ax.axvline(
        x=thresholds[optimal_idx],
        color=_COLORS[5],
        linestyle="--",
        alpha=0.7,
        label=f"Optimal conf={thresholds[optimal_idx]:.3f}",
    )
    ax.scatter(
        [thresholds[optimal_idx]],
        [f1_scores[optimal_idx]],
        color=_COLORS[4],
        s=100,
        zorder=5,
        edgecolors=_COLORS[5],
        linewidths=2,
        label=f"Max F1={f1_scores[optimal_idx]:.4f}",
    )

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"F1-Confidence Curve — {model_name}", fontsize=13)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_path)


def plot_pr_overlay(
    results: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str,
    output_path: Path,
    top_n: int | None = None,
) -> None:
    """Plot overlaid PR curves for multiple models.

    Parameters
    ----------
    results : dict mapping model_name → (recall, precision, ap)
    title : plot title
    output_path : save path
    top_n : if set, only include top N models by AP
    """
    if top_n is not None:
        # Sort by AP descending, take top N
        sorted_models = sorted(results.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
        results = dict(sorted_models)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, (recall, precision, ap)) in enumerate(results.items()):
        color = _COLORS[i % len(_COLORS)]
        ax.plot(recall, precision, color=color, linewidth=2.0, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim((0, 1.05))
    ax.set_ylim((0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_path)


def plot_f1_overlay(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: Path,
) -> None:
    """Plot overlaid F1-confidence curves.

    Parameters
    ----------
    results : dict mapping model_name → (thresholds, f1_scores)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, (thresholds, f1_scores)) in enumerate(results.items()):
        color = _COLORS[i % len(_COLORS)]
        ax.plot(thresholds, f1_scores, color=color, linewidth=2.0, label=name)

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_model(
    model_name: str,
    model_info: dict[str, Any],
    image_paths: list[Path],
    gt_by_image: dict[str, tuple[np.ndarray, tuple[int, int]]],
    device: str,
) -> dict[str, Any] | None:
    """Evaluate a single model and return metrics + curves data.

    Returns
    -------
    dict with keys: model_name, mAP50, mAP75, mAP50_95, optimal_conf, optimal_f1,
        pr_curve, f1_data, predictions
    or None if model weights not found.
    """
    weights_rel = model_info["weights"]
    weights_path = _PROJECT_ROOT / weights_rel
    if not weights_path.exists():
        logger.warning("Weights not found for %s at %s — skipping", model_name, weights_path)
        return None

    model_type = model_info["type"]
    logger.info("Running inference: %s (%s)", model_name, model_type)
    t0 = time.time()

    # --- Run inference ---
    if model_type == "yolo":
        predictions = run_yolo_inference(
            str(weights_path), image_paths, conf=0.001, device=device, imgsz=IMAGE_SIZE
        )
    elif model_type == "faster_rcnn":
        predictions = run_faster_rcnn_inference(
            str(weights_path), image_paths, device=device, imgsz=IMAGE_SIZE
        )
    elif model_type == "detr":
        predictions = run_detr_inference(
            str(weights_path), image_paths, device=device, imgsz=IMAGE_SIZE
        )
    else:
        logger.error("Unknown model type: %s for %s", model_type, model_name)
        return None

    elapsed = time.time() - t0
    total_preds = sum(p["scores"].shape[0] for p in predictions)
    logger.info(
        "  %s: %d predictions across %d images (%.1fs)",
        model_name,
        total_preds,
        len(image_paths),
        elapsed,
    )

    # --- Compute AP at different IoU thresholds ---
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95
    aps = []
    pr_curves_iou = []

    for iou_t in iou_thresholds:
        ap, recall, precision, n_gt = compute_map_at_iou(predictions, gt_by_image, float(iou_t))
        aps.append(ap)
        pr_curves_iou.append((recall, precision))

    mAP50 = aps[0]  # IoU=0.50
    mAP75_idx = int(round((0.75 - 0.5) / 0.05))  # index 5
    mAP75 = aps[mAP75_idx] if mAP75_idx < len(aps) else 0.0
    mAP50_95 = float(np.mean(aps))

    logger.info("  mAP50=%.4f  mAP75=%.4f  mAP50-95=%.4f", mAP50, mAP75, mAP50_95)

    # --- Compute F1-confidence curve using IoU=0.5 ---
    all_confs_list = []
    all_correct_list = []
    n_gt_total = 0
    for pred in predictions:
        stem = pred["stem"]
        gt_entry = gt_by_image.get(stem)
        gt_boxes = gt_entry[0] if gt_entry is not None else np.zeros((0, 4), dtype=np.float32)
        n_gt_total += gt_boxes.shape[0]
        if pred["scores"].shape[0] > 0:
            is_correct, _ = match_predictions_to_gt(
                pred["boxes"], pred["scores"], gt_boxes, iou_threshold=0.5
            )
            all_confs_list.extend(pred["scores"].tolist())
            all_correct_list.extend(is_correct.tolist())

    all_confs = (
        np.array(all_confs_list, dtype=np.float32)
        if all_confs_list
        else np.array([], dtype=np.float32)
    )
    all_correct = (
        np.array(all_correct_list, dtype=bool) if all_correct_list else np.array([], dtype=bool)
    )

    thresholds, f1_scores, f1_precs, f1_recs = compute_f1_confidence(
        all_confs, all_correct, n_gt_total
    )
    optimal_idx = int(np.argmax(f1_scores))
    optimal_conf = float(thresholds[optimal_idx])
    optimal_f1 = float(f1_scores[optimal_idx])

    logger.info("  Optimal conf=%.3f  F1=%.4f", optimal_conf, optimal_f1)

    # Primary PR curve at IoU=0.5
    pr_recall, pr_precision = pr_curves_iou[0]
    pr_ap = aps[0]

    return {
        "model_name": model_name,
        "mAP50": mAP50,
        "mAP75": mAP75,
        "mAP50_95": mAP50_95,
        "optimal_conf": optimal_conf,
        "optimal_f1": optimal_f1,
        "pr_recall": pr_recall,
        "pr_precision": pr_precision,
        "pr_ap": pr_ap,
        "f1_thresholds": thresholds,
        "f1_scores": f1_scores,
        "total_predictions": total_preds,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PR curves and F1-confidence curves for all models."
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "outputs" / "pr-curves"),
        help="Output directory for plots (default: outputs/pr-curves/)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific models to evaluate (default: all 9). "
        "Choices: " + ", ".join(ALL_MODEL_NAMES),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference: cpu or cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="Models to skip (e.g. --skip-models detr faster_rcnn)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Setup ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve models to evaluate
    skip_set = set(args.skip_models)
    if args.models:
        model_names = [m for m in args.models if m not in skip_set]
    else:
        model_names = [m for m in ALL_MODEL_NAMES if m not in skip_set]

    # Verify model weights exist
    available_models = []
    for name in model_names:
        info = _MODEL_REGISTRY[name]
        wpath = _PROJECT_ROOT / info["weights"]
        if wpath.exists():
            available_models.append(name)
            logger.info("Found weights: %s → %s", name, wpath)
        else:
            logger.warning("Missing weights: %s → %s — will skip", name, wpath)

    if not available_models:
        logger.error("No model weights found. Exiting.")
        sys.exit(1)

    # --- Device ---
    device = args.device
    if device == "cuda" or device.startswith("cuda"):
        if torch.cuda.is_available():
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA not available — falling back to CPU")
            device = "cpu"

    # --- Load ground truth ---
    logger.info("Loading ground truth from %s", LABELS_VAL)
    gt_by_image = load_gt_from_yolo_labels(LABELS_VAL, IMAGES_VAL)
    logger.info("Loaded GT for %d images", len(gt_by_image))

    # --- Collect val images ---
    image_paths = sorted([p for p in IMAGES_VAL.glob("*.png") if p.stem in gt_by_image])
    # Also try jpg
    image_paths.extend(sorted([p for p in IMAGES_VAL.glob("*.jpg") if p.stem in gt_by_image]))
    # Also try jpg with uppercase
    image_paths.extend(sorted([p for p in IMAGES_VAL.glob("*.JPG") if p.stem in gt_by_image]))
    # Deduplicate by stem
    seen_stems = set()
    deduped = []
    for p in image_paths:
        if p.stem not in seen_stems:
            seen_stems.add(p.stem)
            deduped.append(p)
    image_paths = deduped

    logger.info("Found %d val images with GT labels", len(image_paths))
    if len(image_paths) == 0:
        logger.error("No images found in %s", IMAGES_VAL)
        sys.exit(1)

    # --- Evaluate all models ---
    all_results: dict[str, dict[str, Any]] = {}
    for name in available_models:
        info = _MODEL_REGISTRY[name]
        result = evaluate_model(name, info, image_paths, gt_by_image, device)
        if result is not None:
            all_results[name] = result

    if not all_results:
        logger.error("No models evaluated successfully. Exiting.")
        sys.exit(1)

    # --- Generate plots ---
    logger.info("\nGenerating plots...")
    n_gt_total = sum(gt_entry[0].shape[0] for gt_entry in gt_by_image.values())

    # 1. Per-model PR curves
    for name, res in all_results.items():
        plot_pr_curve_single(
            recall=res["pr_recall"],
            precision=res["pr_precision"],
            ap=res["pr_ap"],
            model_name=name,
            optimal_conf=res["optimal_conf"],
            optimal_f1=res["optimal_f1"],
            output_path=output_dir / f"{name}_pr.png",
        )

    # 2. Per-model F1-confidence curves
    for name, res in all_results.items():
        optimal_idx = int(np.argmax(res["f1_scores"]))
        plot_f1_curve_single(
            thresholds=res["f1_thresholds"],
            f1_scores=res["f1_scores"],
            optimal_idx=optimal_idx,
            model_name=name,
            output_path=output_dir / f"{name}_f1.png",
        )

    # 3. Overlaid PR curve — all models
    pr_data_all = {
        name: (res["pr_recall"], res["pr_precision"], res["pr_ap"])
        for name, res in all_results.items()
    }
    plot_pr_overlay(
        pr_data_all,
        title=f"Precision-Recall Curves — All Models\n(val set, {len(image_paths)} images, {n_gt_total} GT boxes)",
        output_path=output_dir / "pr_comparison_all.png",
    )

    # 4. Overlaid PR curve — top 4 by mAP50
    plot_pr_overlay(
        pr_data_all,
        title=f"Precision-Recall Curves — Top 4 Models by mAP50\n(val set, {len(image_paths)} images, {n_gt_total} GT boxes)",
        output_path=output_dir / "pr_comparison_top4.png",
        top_n=4,
    )

    # 5. Overlaid F1-confidence curve — all models
    f1_data_all = {
        name: (res["f1_thresholds"], res["f1_scores"]) for name, res in all_results.items()
    }
    plot_f1_overlay(
        f1_data_all,
        title=f"F1-Confidence Curves — All Models\n(val set, {len(image_paths)} images, IoU=0.5)",
        output_path=output_dir / "f1_confidence_all.png",
    )

    # --- Summary table ---
    sep = "-" * 90
    print()
    print("=" * 90)
    print("  PR Curve & F1 Analysis — Summary")
    print("=" * 90)
    print(f"  {'Model':<16} {'mAP50':>8} {'mAP75':>8} {'mAP50-95':>9} {'OptConf':>9} {'OptF1':>8}")
    print(sep)
    for name in ALL_MODEL_NAMES:
        if name not in all_results:
            print(f"  {name:<16} {'SKIP':>8}")
            continue
        r = all_results[name]
        print(
            f"  {name:<16} {r['mAP50']:>8.4f} {r['mAP75']:>8.4f} {r['mAP50_95']:>9.4f} "
            f"{r['optimal_conf']:>9.3f} {r['optimal_f1']:>8.4f}"
        )
    print(sep)
    print(f"  Images: {len(image_paths)} | GT boxes: {n_gt_total}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 90)

    # --- MLflow logging ---
    if not args.no_mlflow:
        logger.info("\nLogging metrics to MLflow...")
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("pr-curve-evaluation")

        _ = mlflow.start_run(
            run_name="pr_curve_analysis",
            tags={
                "experiment_type": "evaluation",
                "dataset_split": "val",
                "n_images": str(len(image_paths)),
                "n_gt_boxes": str(n_gt_total),
            },
        )

        for name in ALL_MODEL_NAMES:
            if name not in all_results:
                continue
            r = all_results[name]
            prefix = f"{name}/"
            mlflow.log_metrics(
                {
                    f"{prefix}mAP50": r["mAP50"],
                    f"{prefix}mAP75": r["mAP75"],
                    f"{prefix}mAP50-95": r["mAP50_95"],
                    f"{prefix}optimal_conf_threshold": r["optimal_conf"],
                    f"{prefix}optimal_f1": r["optimal_f1"],
                    f"{prefix}total_predictions": r["total_predictions"],
                }
            )

        # Log plots as artifacts
        for png_path in sorted(output_dir.glob("*.png")):
            mlflow.log_artifact(str(png_path), artifact_path="pr-curves")

        mlflow.end_run()
        logger.info("MLflow run complete.")

    logger.info("\nDone. All plots saved to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
