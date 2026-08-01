"""Compute calibration curves, ECE, and MCE for all trained models.

Generates per-model reliability diagrams and a comparison overlay plot saved
to ``docs/calibration/``.  Logs ECE/MCE metrics to MLflow.

Supports all 9 models:
- YOLO family: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x, yolov8m, yolo11m
- Non-YOLO: faster_rcnn, detr

YOLO models are evaluated using the single available best.pt (last CV fold).
Faster R-CNN and DETR are evaluated across all 3 folds for mean±std.

Usage
-----
    python scripts/eval_calibration.py                         # all models
    python scripts/eval_calibration.py --models yolo26n yolo26m  # subset
    python scripts/eval_calibration.py --n-bins 15              # custom bins
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
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
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_VAL = DATASET_DIR / "labels" / "val"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "calibration"

# YOLO models (weights in runs/{model}/weights/best.pt, last fold only)
YOLO_MODELS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x", "yolov8m", "yolo11m"]

# Non-YOLO models (weights in experiments/{model}/fold_*_best.pt)
NON_YOLO_MODELS = ["faster_rcnn", "detr"]

ALL_MODELS = YOLO_MODELS + NON_YOLO_MODELS

N_FOLDS = 3
IMAGE_SIZE = 640  # training image size
IOU_THRESHOLD = 0.5  # for matching predictions to GT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_calibration")


# ---------------------------------------------------------------------------
# GT loading helpers
# ---------------------------------------------------------------------------


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format labels and convert to pixel xyxy boxes.

    Parameters
    ----------
    label_path : Path
        YOLO label file (class cx cy w h, normalized [0,1]).
    img_w, img_h : int
        Original image dimensions (before any resize).

    Returns
    -------
    np.ndarray of shape (N, 4) with columns [x1, y1, x2, y2] in pixel coords.
    Returns empty (0, 4) array if no labels.
    """
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
    return np.array(boxes, dtype=np.float32)


def load_gt_for_val_set() -> list[tuple[Path, np.ndarray]]:
    """Load GT boxes for every image in the val set.

    Returns a list of (image_path, gt_boxes_xyxy) tuples where gt_boxes
    are in 640x640 pixel coordinates (matching the inference resize).
    """
    import cv2

    gt_data = []
    for img_path in sorted(IMAGES_VAL.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue

        # Read original image size
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: %s", img_path)
            continue
        orig_h, orig_w = img.shape[:2]

        # Load GT in original pixel coords, then scale to 640x640
        label_path = LABELS_VAL / (img_path.stem + ".txt")
        gt_orig = load_gt_boxes(label_path, orig_w, orig_h)

        # Scale to 640x640 (same resize as training)
        scale_x = IMAGE_SIZE / orig_w
        scale_y = IMAGE_SIZE / orig_h
        if len(gt_orig) > 0:
            gt_scaled = gt_orig.copy()
            gt_scaled[:, 0] *= scale_x
            gt_scaled[:, 1] *= scale_y
            gt_scaled[:, 2] *= scale_x
            gt_scaled[:, 3] *= scale_y
        else:
            gt_scaled = gt_orig

        gt_data.append((img_path, gt_scaled))

    logger.info("Loaded GT for %d val images", len(gt_data))
    return gt_data


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Parameters
    ----------
    boxes_a : (N, 4) array
    boxes_b : (M, 4) array

    Returns
    -------
    (N, M) IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Expand for broadcasting
    a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union_area = area_a + area_b - inter_area
    iou = inter_area / np.maximum(union_area, 1e-6)
    return iou


# ---------------------------------------------------------------------------
# Core calibration computation
# ---------------------------------------------------------------------------


def compute_calibration(
    all_confs: np.ndarray,
    all_correct: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predictions by confidence and compute accuracy per bin.

    Parameters
    ----------
    all_confs : (K,) confidence scores for all predictions
    all_correct : (K,) binary array (1 if prediction matched a GT box with IoU > threshold)
    n_bins : number of equal-width bins in [0, 1]

    Returns
    -------
    bin_confidences : (n_bins,) mean confidence per bin
    bin_accuracies : (n_bins,) fraction of correct predictions per bin
    bin_counts : (n_bins,) number of predictions per bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins, dtype=np.float64)
    bin_accuracies = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (all_confs >= lo) & (all_confs < hi)
        # Include the right edge in the last bin
        if i == n_bins - 1:
            mask |= (all_confs >= lo) & (all_confs <= hi)

        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = all_confs[mask].mean()
            bin_accuracies[i] = all_correct[mask].mean()
        else:
            # Empty bin: use bin midpoint as representative confidence
            bin_confidences[i] = (lo + hi) / 2.0
            bin_accuracies[i] = 0.0

    return bin_confidences, bin_accuracies, bin_counts


def compute_ece(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """Expected Calibration Error.

    ECE = sum_k (n_k / N) * |acc_k - conf_k|
    """
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    weights = bin_counts / total
    return float(np.sum(weights * np.abs(bin_accuracies - bin_confidences)))


def compute_mce(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """Maximum Calibration Error.

    MCE = max_k |acc_k - conf_k|  (over non-empty bins)
    """
    non_empty = bin_counts > 0
    if not non_empty.any():
        return 0.0
    diffs = np.abs(bin_accuracies[non_empty] - bin_confidences[non_empty])
    return float(diffs.max())


# ---------------------------------------------------------------------------
# YOLO model inference
# ---------------------------------------------------------------------------


def run_yolo_calibration(
    model_name: str,
    weights_path: Path,
    gt_data: list[tuple[Path, np.ndarray]],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run calibration for a YOLO model.

    Returns dict with bin_confidences, bin_accuracies, bin_counts, ece, mce.
    """
    from ultralytics import YOLO

    logger.info("Loading YOLO model: %s from %s", model_name, weights_path)
    model = YOLO(str(weights_path))

    all_confs = []
    all_correct = []

    for img_path, gt_boxes in gt_data:
        # Run inference with very low conf threshold
        results = model.predict(
            source=str(img_path),
            save=False,
            conf=0.001,
            iou=0.5,
            verbose=False,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            pred_boxes = result.boxes.xyxy.cpu().numpy()  # (M, 4)
            pred_confs = result.boxes.conf.cpu().numpy()  # (M,)

            # Compute IoU with all GT boxes
            if len(gt_boxes) > 0:
                iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # (M, G)
                max_iou_per_pred = iou_matrix.max(axis=1)  # (M,)
            else:
                max_iou_per_pred = np.zeros(len(pred_boxes), dtype=np.float32)

            correct = (max_iou_per_pred > IOU_THRESHOLD).astype(np.float32)

            all_confs.append(pred_confs)
            all_correct.append(correct)

    if not all_confs:
        logger.warning("No predictions from YOLO model %s", model_name)
        return {
            "bin_confidences": np.zeros(n_bins),
            "bin_accuracies": np.zeros(n_bins),
            "bin_counts": np.zeros(n_bins, dtype=np.int64),
            "ece": 0.0,
            "mce": 0.0,
            "n_predictions": 0,
        }

    all_confs = np.concatenate(all_confs)
    all_correct = np.concatenate(all_correct)

    bin_conf, bin_acc, bin_cnt = compute_calibration(all_confs, all_correct, n_bins)
    ece = compute_ece(bin_conf, bin_acc, bin_cnt)
    mce = compute_mce(bin_conf, bin_acc, bin_cnt)

    logger.info(
        "  %s: %d predictions, ECE=%.4f, MCE=%.4f",
        model_name,
        len(all_confs),
        ece,
        mce,
    )

    return {
        "bin_confidences": bin_conf,
        "bin_accuracies": bin_acc,
        "bin_counts": bin_cnt,
        "ece": ece,
        "mce": mce,
        "n_predictions": len(all_confs),
    }


# ---------------------------------------------------------------------------
# Non-YOLO model inference (Faster R-CNN)
# ---------------------------------------------------------------------------


def run_faster_rcnn_calibration(
    fold_idx: int,
    weights_path: Path,
    gt_data: list[tuple[Path, np.ndarray]],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run calibration for a Faster R-CNN fold."""
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    logger.info("Loading Faster R-CNN fold %d from %s", fold_idx, weights_path)

    # Recreate model architecture
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        num_classes=2,  # background + hole
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_confs = []
    all_correct = []

    with torch.no_grad():
        for img_path, gt_boxes in gt_data:
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            # Resize to 640x640
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_tensor)[0]

            pred_boxes = outputs["boxes"].cpu().numpy()
            pred_scores = outputs["scores"].cpu().numpy()

            # Compute IoU with GT
            if len(gt_boxes) > 0:
                iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
                max_iou_per_pred = iou_matrix.max(axis=1)
            else:
                max_iou_per_pred = np.zeros(len(pred_boxes), dtype=np.float32)

            correct = (max_iou_per_pred > IOU_THRESHOLD).astype(np.float32)

            all_confs.append(pred_scores)
            all_correct.append(correct)

    if not all_confs:
        return {
            "bin_confidences": np.zeros(n_bins),
            "bin_accuracies": np.zeros(n_bins),
            "bin_counts": np.zeros(n_bins, dtype=np.int64),
            "ece": 0.0,
            "mce": 0.0,
            "n_predictions": 0,
        }

    all_confs = np.concatenate(all_confs)
    all_correct = np.concatenate(all_correct)

    bin_conf, bin_acc, bin_cnt = compute_calibration(all_confs, all_correct, n_bins)
    ece = compute_ece(bin_conf, bin_acc, bin_cnt)
    mce = compute_mce(bin_conf, bin_acc, bin_cnt)

    logger.info(
        "  faster_rcnn fold %d: %d predictions, ECE=%.4f, MCE=%.4f",
        fold_idx,
        len(all_confs),
        ece,
        mce,
    )

    return {
        "bin_confidences": bin_conf,
        "bin_accuracies": bin_acc,
        "bin_counts": bin_cnt,
        "ece": ece,
        "mce": mce,
        "n_predictions": len(all_confs),
    }


# ---------------------------------------------------------------------------
# Non-YOLO model inference (DETR)
# ---------------------------------------------------------------------------


def run_detr_calibration(
    fold_idx: int,
    weights_path: Path,
    gt_data: list[tuple[Path, np.ndarray]],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run calibration for a DETR fold."""
    import cv2
    import torch

    logger.info("Loading DETR fold %d from %s", fold_idx, weights_path)

    # Import the DETR model creation from train_detr.py
    # We need to reconstruct the model with num_classes=1 (no background in DETR output)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_detr import create_model, postprocess_detr_output

    model = create_model(num_classes=1, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_confs = []
    all_correct = []

    with torch.no_grad():
        for img_path, gt_boxes in gt_data:
            # Load and preprocess image (DETR: letterbox resize with padding)
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
            img_tensor = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_tensor)

            # Post-process DETR output
            pred = postprocess_detr_output(
                outputs["pred_logits"][0].cpu(),
                outputs["pred_boxes"][0].cpu(),
                score_threshold=0.001,  # very low for calibration
            )

            pred_boxes = pred["boxes"].numpy()
            pred_scores = pred["scores"].numpy()

            # Compute IoU with GT (both in 640x640 pixel coords)
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
                max_iou_per_pred = iou_matrix.max(axis=1)
            else:
                max_iou_per_pred = np.zeros(len(pred_boxes), dtype=np.float32)

            correct = (max_iou_per_pred > IOU_THRESHOLD).astype(np.float32)

            all_confs.append(pred_scores)
            all_correct.append(correct)

    if not all_confs:
        return {
            "bin_confidences": np.zeros(n_bins),
            "bin_accuracies": np.zeros(n_bins),
            "bin_counts": np.zeros(n_bins, dtype=np.int64),
            "ece": 0.0,
            "mce": 0.0,
            "n_predictions": 0,
        }

    all_confs = np.concatenate(all_confs)
    all_correct = np.concatenate(all_correct)

    bin_conf, bin_acc, bin_cnt = compute_calibration(all_confs, all_correct, n_bins)
    ece = compute_ece(bin_conf, bin_acc, bin_cnt)
    mce = compute_mce(bin_conf, bin_acc, bin_cnt)

    logger.info(
        "  detr fold %d: %d predictions, ECE=%.4f, MCE=%.4f",
        fold_idx,
        len(all_confs),
        ece,
        mce,
    )

    return {
        "bin_confidences": bin_conf,
        "bin_accuracies": bin_acc,
        "bin_counts": bin_cnt,
        "ece": ece,
        "mce": mce,
        "n_predictions": len(all_confs),
    }


# ---------------------------------------------------------------------------
# Weight discovery
# ---------------------------------------------------------------------------


def find_model_weights(model_name: str) -> list[tuple[int, Path]]:
    """Find available weight files for a model.

    Returns a list of (fold_index, weights_path) tuples.
    - YOLO models: single best.pt from runs/{model}/weights/ (fold = -1 for single)
    - Non-YOLO models: fold-specific weights from experiments/{model}/fold_*_best.pt
    """
    weights = []

    if model_name in YOLO_MODELS:
        # YOLO: check runs/{model}/weights/best.pt
        candidate = RUNS_DIR / model_name / "weights" / "best.pt"
        if candidate.exists():
            weights.append((-1, candidate))  # -1 = single model, not fold-specific
            logger.info("Found YOLO weights: %s", candidate)
        else:
            logger.warning("YOLO weights not found: %s", candidate)

    elif model_name in NON_YOLO_MODELS:
        # Non-YOLO: check experiments/{model}/fold_*_best.pt
        for fold_idx in range(N_FOLDS):
            candidate = EXPERIMENTS_DIR / model_name / f"fold_{fold_idx}_best.pt"
            if candidate.exists():
                weights.append((fold_idx, candidate))
                logger.info("Found fold %d weights: %s", fold_idx, candidate)
            else:
                logger.warning("Fold %d weights not found: %s", fold_idx, candidate)

    return weights


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: np.ndarray,
    model_name: str,
    ece: float,
    mce: float,
    output_path: Path,
    fold_idx: int | None = None,
    std_accuracies: np.ndarray | None = None,
) -> None:
    """Plot a single reliability diagram and save to disk.

    Parameters
    ----------
    bin_confidences, bin_accuracies, bin_counts : from compute_calibration()
    model_name : display name
    ece, mce : calibration errors
    output_path : where to save the PNG
    fold_idx : if set, included in the title
    std_accuracies : optional per-bin std across folds for error bars
    """
    n_bins = len(bin_confidences)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

    # Bar chart: gap between accuracy and confidence
    bar_width = 0.8 / n_bins
    bar_positions = bin_confidences - bar_width / 2

    # Fill bars
    for i in range(n_bins):
        if bin_counts[i] > 0:
            color = "#4C72B0"
            ax.bar(
                bar_positions[i],
                bin_accuracies[i],
                width=bar_width,
                alpha=0.6,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

    # Error bars if std available
    if std_accuracies is not None:
        non_empty = bin_counts > 0
        ax.errorbar(
            bin_confidences[non_empty],
            bin_accuracies[non_empty],
            yerr=std_accuracies[non_empty],
            fmt="o",
            color="#C44E52",
            markersize=5,
            capsize=3,
            capthick=1,
            linewidth=1.5,
            label="Mean ± std (3-fold)",
            zorder=5,
        )
    else:
        # Single fold: just scatter points
        non_empty = bin_counts > 0
        ax.scatter(
            bin_confidences[non_empty],
            bin_accuracies[non_empty],
            color="#C44E52",
            s=30,
            zorder=5,
            label="Model",
        )

    # Labels
    title = f"Reliability Diagram — {model_name}"
    if fold_idx is not None and fold_idx >= 0:
        title += f" (fold {fold_idx})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    subtitle = f"ECE = {ece:.4f}  |  MCE = {mce:.4f}"
    ax.text(
        0.5,
        0.92,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="gray",
    )

    ax.set_xlabel("Mean Confidence", fontsize=11)
    ax.set_ylabel("Accuracy (IoU > 0.5)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_comparison(
    all_results: dict[str, dict[str, Any]],
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot overlay of multiple models' calibration curves.

    Parameters
    ----------
    all_results : {model_name: {bin_confidences, bin_accuracies, bin_counts, ece, mce}}
    output_path : where to save the comparison PNG
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(all_results), 1)))

    for idx, (model_name, result) in enumerate(sorted(all_results.items())):
        bin_conf = result["bin_confidences"]
        bin_acc = result["bin_accuracies"]
        bin_cnt = result["bin_counts"]
        ece = result["ece"]

        non_empty = bin_cnt > 0
        if not non_empty.any():
            continue

        color = colors[idx % len(colors)]

        # Line plot connecting bin midpoints
        ax.plot(
            bin_conf[non_empty],
            bin_acc[non_empty],
            "o-",
            color=color,
            linewidth=2,
            markersize=5,
            label=f"{model_name} (ECE={ece:.4f})",
        )

        # Fill area between curve and diagonal
        fill_x = np.concatenate([[0], bin_conf[non_empty], [1]])
        fill_y = np.concatenate([[0], bin_acc[non_empty], [1]])
        diag = fill_x  # perfect calibration diagonal
        ax.fill_between(
            fill_x,
            fill_y,
            diag,
            alpha=0.08,
            color=color,
        )

    ax.set_title(
        "Calibration Comparison — All Models",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Confidence", fontsize=11)
    ax.set_ylabel("Accuracy (IoU > 0.5)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot: %s", output_path)


def plot_top_models(
    all_results: dict[str, dict[str, Any]],
    output_path: Path,
    top_n: int = 4,
    n_bins: int = 10,
) -> None:
    """Plot the best top_n models by lowest ECE."""
    # Sort by ECE (ascending)
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["ece"])
    top_models = dict(sorted_models[:top_n])

    if not top_models:
        logger.warning("No models to plot for top-N comparison")
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    for idx, (model_name, result) in enumerate(top_models.items()):
        bin_conf = result["bin_confidences"]
        bin_acc = result["bin_accuracies"]
        bin_cnt = result["bin_counts"]
        ece = result["ece"]

        non_empty = bin_cnt > 0
        if not non_empty.any():
            continue

        color = colors[idx]

        # Mean ± std if available
        has_std = "std_accuracies" in result
        if has_std:
            std_acc = result["std_accuracies"]
            ax.fill_between(
                bin_conf[non_empty],
                np.maximum(bin_acc[non_empty] - std_acc[non_empty], 0),
                np.minimum(bin_acc[non_empty] + std_acc[non_empty], 1),
                alpha=0.15,
                color=color,
            )

        ax.plot(
            bin_conf[non_empty],
            bin_acc[non_empty],
            "o-",
            color=color,
            linewidth=2,
            markersize=5,
            label=f"{model_name} (ECE={ece:.4f})",
        )

    ax.set_title(
        f"Calibration — Top {len(top_models)} Models (by Lowest ECE)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Confidence", fontsize=11)
    ax.set_ylabel("Accuracy (IoU > 0.5)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved top-%d comparison: %s", top_n, output_path)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_calibration_to_mlflow(
    model_name: str,
    fold_results: list[dict[str, Any]],
    mlflow_experiment: str = "calibration",
) -> None:
    """Log calibration metrics to MLflow.

    Creates/uses a 'calibration' experiment. Logs per-fold ECE/MCE
    and aggregated mean±std.
    """
    try:
        import mlflow

        from scripts.mlflow_utils import TRACKING_URI

        mlflow.set_tracking_uri(TRACKING_URI)
        _ = mlflow.set_experiment(mlflow_experiment)

        # Compute aggregated metrics
        eces = [r["ece"] for r in fold_results if r["ece"] > 0 or r["n_predictions"] > 0]
        mces = [r["mce"] for r in fold_results if r["ece"] > 0 or r["n_predictions"] > 0]

        mean_ece = float(np.mean(eces)) if eces else 0.0
        std_ece = float(np.std(eces)) if len(eces) > 1 else 0.0
        mean_mce = float(np.mean(mces)) if mces else 0.0
        std_mce = float(np.std(mces)) if len(mces) > 1 else 0.0

        fold_label = "single" if len(fold_results) == 1 else f"{len(fold_results)}-fold"

        tags = {
            "model_name": model_name,
            "model_type": "yolo" if model_name in YOLO_MODELS else "non_yolo",
            "evaluation": "calibration",
            "cv_folds": fold_label,
        }

        _ = mlflow.start_run(
            run_name=f"{model_name}_calibration",
            tags=tags,
        )

        # Log params
        mlflow.log_params(
            {
                "model/name": model_name,
                "eval/n_folds": len(fold_results),
                "eval/n_bins": 10,
                "eval/iou_threshold": IOU_THRESHOLD,
                "eval/image_size": IMAGE_SIZE,
                "eval/conf_threshold": 0.001,
            }
        )

        # Log per-fold metrics
        for i, r in enumerate(fold_results):
            fold_prefix = f"fold_{i}" if len(fold_results) > 1 else "eval"
            mlflow.log_metric(f"{fold_prefix}/ece", r["ece"])
            mlflow.log_metric(f"{fold_prefix}/mce", r["mce"])
            mlflow.log_metric(f"{fold_prefix}/n_predictions", r["n_predictions"])

        # Log aggregated metrics
        mlflow.log_metric("ece_mean", mean_ece)
        mlflow.log_metric("ece_std", std_ece)
        mlflow.log_metric("mce_mean", mean_mce)
        mlflow.log_metric("mce_std", std_mce)

        # Log plots as artifacts
        output_dir = OUTPUT_DIR / model_name
        for png_file in output_dir.glob("*.png"):
            mlflow.log_artifact(str(png_file), artifact_path="calibration_plots")

        mlflow.end_run()
        logger.info(
            "MLflow logged %s calibration: ECE=%.4f±%.4f, MCE=%.4f±%.4f",
            model_name,
            mean_ece,
            std_ece,
            mean_mce,
            std_mce,
        )

    except Exception as e:
        logger.warning("Failed to log %s calibration to MLflow: %s", model_name, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_model(
    model_name: str,
    gt_data: list[tuple[Path, np.ndarray]],
    n_bins: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate calibration for a single model across all available folds.

    Returns
    -------
    aggregated : dict with bin_confidences, bin_accuracies, bin_counts, ece, mce,
                 and optionally std_accuracies for multi-fold models
    fold_results : list of per-fold result dicts
    """
    weight_files = find_model_weights(model_name)
    if not weight_files:
        logger.warning("No weights found for %s — skipping", model_name)
        return {}, []

    fold_results = []

    for fold_idx, weights_path in weight_files:
        logger.info(
            "Evaluating %s fold=%s ...", model_name, fold_idx if fold_idx >= 0 else "single"
        )

        if model_name in YOLO_MODELS:
            result = run_yolo_calibration(model_name, weights_path, gt_data, n_bins)
        elif model_name == "faster_rcnn":
            result = run_faster_rcnn_calibration(fold_idx, weights_path, gt_data, n_bins)
        elif model_name == "detr":
            result = run_detr_calibration(fold_idx, weights_path, gt_data, n_bins)
        else:
            logger.warning("Unknown model type: %s", model_name)
            continue

        result["fold_idx"] = fold_idx
        fold_results.append(result)

    if not fold_results:
        return {}, []

    # Aggregate across folds
    if len(fold_results) == 1:
        aggregated = dict(fold_results[0])
    else:
        # Multi-fold: compute mean and std
        all_bin_acc = np.array([r["bin_accuracies"] for r in fold_results])
        all_bin_conf = np.array([r["bin_confidences"] for r in fold_results])
        all_bin_cnt = np.array([r["bin_counts"] for r in fold_results])

        # Use mean bin confidences and accuracies
        mean_bin_conf = all_bin_conf.mean(axis=0)
        mean_bin_acc = all_bin_acc.mean(axis=0)
        mean_bin_cnt = all_bin_cnt.mean(axis=0).astype(np.int64)
        std_bin_acc = all_bin_acc.std(axis=0)

        # Recompute ECE/MCE on aggregated
        ece = compute_ece(mean_bin_conf, mean_bin_acc, mean_bin_cnt)
        mce = compute_mce(mean_bin_conf, mean_bin_acc, mean_bin_cnt)

        aggregated = {
            "bin_confidences": mean_bin_conf,
            "bin_accuracies": mean_bin_acc,
            "bin_counts": mean_bin_cnt,
            "std_accuracies": std_bin_acc,
            "ece": ece,
            "mce": mce,
            "n_predictions": sum(r["n_predictions"] for r in fold_results),
        }

    return aggregated, fold_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute calibration curves, ECE, and MCE for all models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model names to evaluate (default: all). Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of calibration bins (default: 10)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    args = parser.parse_args()

    models_to_eval = args.models if args.models else ALL_MODELS
    n_bins = args.n_bins

    logger.info("=" * 70)
    logger.info("Model Calibration Evaluation")
    logger.info("=" * 70)
    logger.info("Models: %s", models_to_eval)
    logger.info("Bins: %d, IoU threshold: %.1f, image size: %d", n_bins, IOU_THRESHOLD, IMAGE_SIZE)
    logger.info("Output: %s", OUTPUT_DIR)

    # Load GT data once
    logger.info("Loading ground truth labels...")
    gt_data = load_gt_for_val_set()
    if not gt_data:
        logger.error("No validation images found at %s", IMAGES_VAL)
        sys.exit(1)

    # Evaluate each model
    all_model_results = {}
    all_summary = {}

    for model_name in models_to_eval:
        logger.info("")
        logger.info("-" * 50)
        logger.info("Evaluating: %s", model_name)
        logger.info("-" * 50)

        start = time.time()
        aggregated, fold_results = evaluate_model(model_name, gt_data, n_bins)
        elapsed = time.time() - start

        if not aggregated:
            logger.warning("No results for %s", model_name)
            continue

        all_model_results[model_name] = aggregated

        # Save per-model plots
        model_output_dir = OUTPUT_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        if len(fold_results) == 1:
            # Single fold: plot without error bars
            plot_reliability_diagram(
                aggregated["bin_confidences"],
                aggregated["bin_accuracies"],
                aggregated["bin_counts"],
                model_name,
                aggregated["ece"],
                aggregated["mce"],
                model_output_dir / f"{model_name}_calibration.png",
            )
        else:
            # Multi-fold: plot with error bars
            plot_reliability_diagram(
                aggregated["bin_confidences"],
                aggregated["bin_accuracies"],
                aggregated["bin_counts"],
                model_name,
                aggregated["ece"],
                aggregated["mce"],
                model_output_dir / f"{model_name}_calibration.png",
                std_accuracies=aggregated.get("std_accuracies"),
            )

            # Also plot individual folds
            for fr in fold_results:
                if fr["fold_idx"] >= 0:
                    plot_reliability_diagram(
                        fr["bin_confidences"],
                        fr["bin_accuracies"],
                        fr["bin_counts"],
                        model_name,
                        fr["ece"],
                        fr["mce"],
                        model_output_dir / f"{model_name}_fold{fr['fold_idx']}_calibration.png",
                        fold_idx=fr["fold_idx"],
                    )

        # Log to MLflow
        if not args.no_mlflow:
            log_calibration_to_mlflow(model_name, fold_results)

        # Collect summary
        fold_label = "single" if len(fold_results) == 1 else f"{len(fold_results)}-fold"
        all_summary[model_name] = {
            "ece": aggregated["ece"],
            "mce": aggregated["mce"],
            "n_predictions": aggregated["n_predictions"],
            "folds": fold_label,
            "elapsed_seconds": round(elapsed, 1),
        }

    # Save comparison plots
    if all_model_results:
        plot_comparison(
            all_model_results,
            OUTPUT_DIR / "comparison_all_models.png",
            n_bins,
        )
        plot_top_models(
            all_model_results,
            OUTPUT_DIR / "comparison_top4_models.png",
            top_n=4,
            n_bins=n_bins,
        )

    # Print summary table
    print(f"\n{'=' * 70}")
    print("CALIBRATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'ECE':>8} {'MCE':>8} {'#Pred':>8} {'Folds':>10} {'Time':>8}")
    print("-" * 70)
    for model_name, summary in sorted(all_summary.items()):
        print(
            f"{model_name:<15} "
            f"{summary['ece']:>8.4f} "
            f"{summary['mce']:>8.4f} "
            f"{summary['n_predictions']:>8d} "
            f"{summary['folds']:>10s} "
            f"{summary['elapsed_seconds']:>7.1f}s"
        )
    print("=" * 70)
    print(f"\nPlots saved to: {OUTPUT_DIR}")

    # Save summary JSON
    summary_path = OUTPUT_DIR / "calibration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
