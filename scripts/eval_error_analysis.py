"""Error analysis for the best-performing object detection model.

Identifies failure modes: FP vs FN breakdown, size-based error distribution,
class confusion patterns, and confidence calibration by class.

Usage
-----
    # Default: analyze faster_rcnn on test set
    python scripts/eval_error_analysis.py

    # Explicit model and weights
    python scripts/eval_error_analysis.py --model faster_rcnn --weights experiments/faster_rcnn/fold_0_best.pt

    # YOLO model
    python scripts/eval_error_analysis.py --model yolo11m --weights runs/yolo11m/weights/best.pt

    # Custom thresholds
    python scripts/eval_error_analysis.py --iou-threshold 0.5 --conf-threshold 0.25

Outputs
-------
    outputs/error-analysis/
        error_type_pie.png
        size_error_bars.png
        confusion_heatmap.png
        confidence_hist_by_class.png
        error_summary_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROJECT_ROOT = _PROJECT_ROOT
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
TEST_IMAGES = DATASET_DIR / "images" / "test"
TEST_LABELS = DATASET_DIR / "labels" / "test"

# ---------------------------------------------------------------------------
# Class names (single class: hole)
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "hole"}
CLASS_ID_TO_NAME = {0: "hole"}
NUM_CLASSES = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Error analysis for the best-performing detection model"
    )
    parser.add_argument(
        "--model",
        default="faster_rcnn",
        choices=[
            "faster_rcnn",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
            "yolo26n",
            "yolo11m",
            "yolov8m",
            "detr",
        ],
        help="Model to analyze (default: faster_rcnn)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to model weights. If not provided, auto-detect from experiments/ or runs/",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "error-analysis"),
        help="Output directory for plots and report",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to GT (default: 0.5)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help="Minimum confidence for predictions (default: 0.05)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLO models (default: 640)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu, cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--significance-report",
        default=str(PROJECT_ROOT / "outputs" / "significance" / "significance_report.json"),
        help="Path to significance report JSON for auto-selecting best model",
    )
    return parser.parse_args()


def extract_parent_scene(filename: str) -> str:
    """Extract parent scene identifier from image filename."""
    m = re.compile(r"^(.+)_[^_]+\.\w+$").match(filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def resolve_weights(model_name: str, weights_arg: str | None) -> Path:
    """Find model weights from argument or auto-detect."""
    if weights_arg:
        p = Path(weights_arg)
        if p.exists():
            return p
        # Try relative to project root
        p = PROJECT_ROOT / weights_arg
        if p.exists():
            return p
        raise FileNotFoundError(f"Weights not found: {weights_arg}")

    if model_name == "faster_rcnn":
        # Use the best fold by mAP50 from results.json
        results_path = PROJECT_ROOT / "experiments" / "faster_rcnn" / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            folds = data.get("fold_results", [])
            if folds:
                best_fold = max(folds, key=lambda x: x["metrics"].get("val/mAP50", 0))
                fold_idx = best_fold["fold"]
                weights = PROJECT_ROOT / "experiments" / "faster_rcnn" / f"fold_{fold_idx}_best.pt"
                if weights.exists():
                    print(
                        f"Selected fold {fold_idx} (mAP50={best_fold['metrics']['val/mAP50']:.4f})"
                    )
                    return weights
        # Fallback to fold_0
        return PROJECT_ROOT / "experiments" / "faster_rcnn" / "fold_0_best.pt"

    if model_name == "detr":
        return PROJECT_ROOT / "experiments" / "detr" / "fold_0_best.pt"

    # YOLO models: runs/{model}/weights/best.pt
    candidate = PROJECT_ROOT / "runs" / model_name / "weights" / "best.pt"
    if candidate.exists():
        return candidate

    # Fallback: check other locations
    for loc in [
        PROJECT_ROOT / "runs" / "train" / f"{model_name}-hole" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / f"{model_name}" / "weights" / "best.pt",
    ]:
        if loc.exists():
            return loc

    raise FileNotFoundError(
        f"Cannot find weights for {model_name}. " f"Please specify --weights explicitly."
    )


# ---------------------------------------------------------------------------
# YOLO label loading
# ---------------------------------------------------------------------------


def load_yolo_labels(label_path: Path) -> list[dict[str, Any]]:
    """Load YOLO-format labels: class_id cx cy w h (normalized).

    Returns list of dicts with keys: class_id, bbox (x1y1x2y2 normalized).
    """
    boxes: list[dict[str, Any]] = []
    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append(
                {
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                    "area": w * h,
                }
            )
    return boxes


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes (normalized coords)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Inference: YOLO models
# ---------------------------------------------------------------------------


def run_yolo_inference(
    model_path: Path,
    image_paths: list[Path],
    imgsz: int = 640,
    conf_threshold: float = 0.05,
    device: str = "cpu",
) -> dict[str, list[dict[str, Any]]]:
    """Run YOLO inference on test images.

    Returns dict mapping image filename -> list of predictions.
    Each prediction: {class_id, bbox [x1y1x2y2 norm], confidence}.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    all_preds: dict[str, list[dict[str, Any]]] = {}

    for img_path in image_paths:
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            imgsz=imgsz,
            device=device,
            save=False,
            verbose=False,
        )
        result = results[0]
        preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box, cls_id, conf in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf, strict=False
            ):
                # Normalize to [0,1]
                img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                preds.append(
                    {
                        "class_id": int(cls_id),
                        "bbox": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
                        "confidence": float(conf),
                    }
                )
        all_preds[img_path.name] = preds

    return all_preds


# ---------------------------------------------------------------------------
# Inference: Faster R-CNN
# ---------------------------------------------------------------------------


def run_fasterrcnn_inference(
    model_path: Path,
    image_paths: list[Path],
    conf_threshold: float = 0.05,
    device: str = "cpu",
    image_size: int = 640,
) -> dict[str, list[dict[str, Any]]]:
    """Run Faster R-CNN inference on test images.

    Returns dict mapping image filename -> list of predictions.
    Each prediction: {class_id, bbox [x1y1x2y2 norm], confidence}.
    """
    import torch
    from PIL import Image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Create model and load state dict
    model = fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1
    )  # +1 for background
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds: dict[str, list[dict[str, Any]]] = {}

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img_resized = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_tensor)[0]

            preds = []
            scores = outputs["scores"].cpu().numpy()
            boxes = outputs["boxes"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()

            for score, box, label in zip(scores, boxes, labels, strict=False):
                if score < conf_threshold:
                    continue
                # box is in resized image pixel coords; normalize
                x1 = float(box[0]) / image_size
                y1 = float(box[1]) / image_size
                x2 = float(box[2]) / image_size
                y2 = float(box[3]) / image_size
                # label 1 = hole (class 0 in our scheme), since Faster R-CNN uses 1-indexed
                class_id = int(label) - 1
                preds.append(
                    {
                        "class_id": class_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(score),
                    }
                )
            all_preds[img_path.name] = preds

    return all_preds


# ---------------------------------------------------------------------------
# Inference: DETR
# ---------------------------------------------------------------------------


def run_detr_inference(
    model_path: Path,
    image_paths: list[Path],
    conf_threshold: float = 0.05,
    device: str = "cpu",
    image_size: int = 640,
) -> dict[str, list[dict[str, Any]]]:
    """Run DETR inference on test images.

    Returns dict mapping image filename -> list of predictions.
    """
    import torch
    import torchvision
    from PIL import Image

    # Load DETR model from torchvision
    model = torchvision.models.detection.detr_resnet50(weights=None, num_classes=NUM_CLASSES + 1)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds: dict[str, list[dict[str, Any]]] = {}

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_tensor)[0]

            preds = []
            scores = outputs["scores"].cpu().numpy()
            boxes = outputs["boxes"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()

            for score, box, label in zip(scores, boxes, labels, strict=False):
                if score < conf_threshold:
                    continue
                x1 = float(box[0]) / image_size
                y1 = float(box[1]) / image_size
                x2 = float(box[2]) / image_size
                y2 = float(box[3]) / image_size
                class_id = int(label) - 1
                preds.append(
                    {
                        "class_id": class_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(score),
                    }
                )
            all_preds[img_path.name] = preds

    return all_preds


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_errors(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Classify predictions and GT into error categories.

    Returns dict with:
        - true_positives: predictions matched to GT with IoU >= threshold, correct class
        - false_positives: predictions not matched (IoU < threshold with all GT)
        - false_negatives: GT objects not matched by any prediction
        - localization_errors: predictions matched but IoU in [0.1, threshold) with correct class
        - class_confusion: predictions matched with IoU >= threshold but wrong class
        - duplicates: multiple predictions matched to same GT
        - background_fp: predictions with IoU < 0.1 with any GT
        - missed: GT with no prediction having IoU >= 0.5
    """
    matched_gt = set()
    gt_to_preds: dict[int, list[tuple[float, dict]]] = defaultdict(list)

    # Classify each prediction
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # Check class
            if pred["class_id"] == ground_truths[best_gt_idx]["class_id"]:
                pred["error_type"] = "correct"
                pred["matched_gt_idx"] = best_gt_idx
                pred["iou"] = best_iou
                matched_gt.add(best_gt_idx)
                gt_to_preds[best_gt_idx].append((best_iou, pred))
            else:
                pred["error_type"] = "class_confusion"
                pred["matched_gt_idx"] = best_gt_idx
                pred["iou"] = best_iou
        elif best_iou >= 0.1:
            # Localization error
            pred["error_type"] = "localization"
            pred["matched_gt_idx"] = best_gt_idx
            pred["iou"] = best_iou
        else:
            # Background FP
            pred["error_type"] = "background_fp"
            pred["matched_gt_idx"] = -1
            pred["iou"] = best_iou

    # Find false negatives (missed GT)
    false_negatives = []
    for gt_idx, gt in enumerate(ground_truths):
        if gt_idx not in matched_gt:
            gt["error_type"] = "missed"
            gt["matched_pred_idx"] = -1
            false_negatives.append(gt)
        else:
            gt["error_type"] = "matched"
            gt["matched_pred_idx"] = gt_idx

    # Detect duplicates (multiple predictions matched to same GT)
    duplicates = []
    for _, pred_list in gt_to_preds.items():
        if len(pred_list) > 1:
            for _, pred in pred_list[1:]:  # Skip first (the TP)
                pred["error_type"] = "duplicate"
                duplicates.append(pred)

    # Separate predictions by type
    tp = [p for p in predictions if p.get("error_type") == "correct"]
    fp = [
        p
        for p in predictions
        if p.get("error_type") in ("localization", "class_confusion", "background_fp", "duplicate")
    ]

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": false_negatives,
        "localization_errors": [p for p in predictions if p.get("error_type") == "localization"],
        "class_confusion": [p for p in predictions if p.get("error_type") == "class_confusion"],
        "background_fp": [p for p in predictions if p.get("error_type") == "background_fp"],
        "duplicates": duplicates,
        "missed": false_negatives,
    }


# ---------------------------------------------------------------------------
# Size analysis
# ---------------------------------------------------------------------------


def get_size_category(bbox_area: float) -> str:
    """Categorize object by relative area in image.

    - small: <1% of image area
    - medium: 1-10% of image area
    - large: >10% of image area
    """
    if bbox_area < 0.01:
        return "small"
    elif bbox_area < 0.10:
        return "medium"
    else:
        return "large"


def compute_size_analysis(
    all_errors: dict[str, Any],
    all_gt: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute error rates by object size category."""
    size_stats = {
        "small": {"total": 0, "matched": 0, "missed": 0},
        "medium": {"total": 0, "matched": 0, "missed": 0},
        "large": {"total": 0, "matched": 0, "missed": 0},
    }

    for gt in all_gt:
        size = get_size_category(gt["area"])
        size_stats[size]["total"] += 1
        if gt.get("error_type") == "missed":
            size_stats[size]["missed"] += 1
        else:
            size_stats[size]["matched"] += 1

    # Compute miss rates
    result = {}
    for size, stats in size_stats.items():
        total = stats["total"]
        missed = stats["missed"]
        result[size] = {
            "total": total,
            "matched": stats["matched"],
            "missed": missed,
            "miss_rate": missed / total if total > 0 else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Confusion analysis
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    all_predictions: dict[str, list[dict]],
    all_ground_truths: dict[str, list[dict]],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.05,
) -> dict[str, dict[str, int]]:
    """Build confusion matrix showing GT class vs predicted class.

    Returns nested dict: confusion_matrix[gt_class][pred_class] = count.
    Includes "background" row for FN and "background" column for FP.
    """
    classes = sorted(set(gt["class_id"] for gts in all_ground_truths.values() for gt in gts))
    class_names_list = [CLASS_ID_TO_NAME.get(c, str(c)) for c in classes]

    # Initialize matrix
    cm: dict[str, dict[str, int]] = {}
    for gt_class in class_names_list + ["background"]:
        cm[gt_class] = {}
        for pred_class in class_names_list + ["background"]:
            cm[gt_class][pred_class] = 0

    for img_name in all_ground_truths:
        gts = all_ground_truths[img_name]
        preds = all_predictions.get(img_name, [])

        # Filter preds by conf threshold
        filtered_preds = [p for p in preds if p.get("confidence", 1.0) >= conf_threshold]

        matched_gt = set()

        for pred in filtered_preds:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            pred_class = CLASS_ID_TO_NAME.get(pred["class_id"], str(pred["class_id"]))

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_class = CLASS_ID_TO_NAME.get(
                    gts[best_gt_idx]["class_id"], str(gts[best_gt_idx]["class_id"])
                )
                matched_gt.add(best_gt_idx)
                if pred_class == gt_class:
                    cm[gt_class][pred_class] += 1  # TP
                else:
                    cm[gt_class][pred_class] += 1  # Class confusion
            else:
                cm["background"][pred_class] += 1  # FP (no matching GT)

        # Count FN (unmatched GT)
        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt:
                gt_class = CLASS_ID_TO_NAME.get(gt["class_id"], str(gt["class_id"]))
                cm[gt_class]["background"] += 1  # FN

    return cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_error_type_pie(all_errors: dict[str, Any], output_dir: Path) -> None:
    """Plot pie chart of error types (FP vs FN breakdown)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Count error types
    error_counts = {
        "True Positives": len(all_errors["true_positives"]),
        "Localization Errors": len(all_errors["localization_errors"]),
        "Background FPs": len(all_errors["background_fp"]),
        "Missed Objects (FN)": len(all_errors["missed"]),
        "Class Confusion": len(all_errors["class_confusion"]),
        "Duplicates": len(all_errors["duplicates"]),
    }

    # Filter out zero counts
    labels = []
    sizes = []
    for label, count in error_counts.items():
        if count > 0:
            labels.append(f"{label}\n({count})")
            sizes.append(count)

    if not sizes:
        print("WARNING: No errors to plot")
        return

    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#3498db", "#95a5a6"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all categories
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(sizes)],
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax1.set_title("Prediction Breakdown", fontsize=13, fontweight="bold")

    # Right: FP vs FN only
    fp_count = len(all_errors["false_positives"])
    fn_count = len(all_errors["false_negatives"])
    tp_count = len(all_errors["true_positives"])

    right_labels = []
    right_sizes = []
    right_colors = []
    if tp_count > 0:
        right_labels.append(f"Correct\n({tp_count})")
        right_sizes.append(tp_count)
        right_colors.append("#2ecc71")
    if fp_count > 0:
        right_labels.append(f"False Positives\n({fp_count})")
        right_sizes.append(fp_count)
        right_colors.append("#e74c3c")
    if fn_count > 0:
        right_labels.append(f"False Negatives\n({fn_count})")
        right_sizes.append(fn_count)
        right_colors.append("#9b59b6")

    if right_sizes:
        ax2.pie(
            right_sizes,
            labels=right_labels,
            autopct="%1.1f%%",
            colors=right_colors,
            startangle=90,
            textprops={"fontsize": 10},
        )
    ax2.set_title("FP vs FN Overview", fontsize=13, fontweight="bold")

    fig.suptitle("Error Type Distribution", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "error_type_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: error_type_pie.png")


def plot_size_error_bars(
    size_analysis: dict[str, dict],
    output_dir: Path,
) -> None:
    """Plot error rates by object size category."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ["small", "medium", "large"]
    miss_rates = [size_analysis[c]["miss_rate"] for c in categories]
    totals = [size_analysis[c]["total"] for c in categories]
    missed_counts = [size_analysis[c]["missed"] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(categories, miss_rates, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, rate, total, missed in zip(bars, miss_rates, totals, missed_counts, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{rate:.1%}\n({missed}/{total})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, max(miss_rates) * 1.3 if miss_rates else 1.0)
    ax.set_ylabel("Miss Rate (False Negative Rate)", fontsize=12)
    ax.set_xlabel("Object Size Category", fontsize=12)
    ax.set_title("Error Rate by Object Size", fontsize=14, fontweight="bold")

    # Add legend explanation
    size_defs = [
        "Small: <1% of image area",
        "Medium: 1-10% of image area",
        "Large: >10% of image area",
    ]
    legend_text = "\n".join(size_defs)
    ax.text(
        0.98,
        0.95,
        legend_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "size_error_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: size_error_bars.png")


def plot_confusion_heatmap(
    confusion_matrix: dict[str, dict[str, int]],
    output_dir: Path,
) -> None:
    """Plot class confusion heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get all classes
    gt_classes = list(confusion_matrix.keys())
    pred_classes = list(set(pc for row in confusion_matrix.values() for pc in row))
    pred_classes.sort()

    # Build matrix
    matrix_rows = []
    for gt_cls in gt_classes:
        row = []
        for pred_cls in pred_classes:
            row.append(confusion_matrix[gt_cls].get(pred_cls, 0))
        matrix_rows.append(row)

    matrix = np.array(matrix_rows, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap
    try:
        import seaborn as sns

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            xticklabels=pred_classes,
            yticklabels=gt_classes,
            ax=ax,
            cbar_kws={"label": "Count"},
            linewidths=0.5,
            linecolor="white",
        )
    except ImportError:
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pred_classes)))
        ax.set_xticklabels(pred_classes, rotation=45, ha="right")
        ax.set_yticks(range(len(gt_classes)))
        ax.set_yticklabels(gt_classes)
        plt.colorbar(im, ax=ax, label="Count")
        for i in range(len(gt_classes)):
            for j in range(len(pred_classes)):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=10)

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Ground Truth Class", fontsize=12)
    ax.set_title("Confusion Matrix (Detection)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_heatmap.png")


def plot_confidence_histograms(
    all_predictions: dict[str, list[dict]],
    all_ground_truths: dict[str, list[dict]],
    iou_threshold: float = 0.5,
    output_dir: Path | None = None,
) -> None:
    """Plot confidence distributions split by correct/incorrect per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = Path.cwd()

    # Separate confidences by correctness
    correct_confs = defaultdict(list)
    incorrect_confs = defaultdict(list)

    for img_name, preds in all_predictions.items():
        gts = all_ground_truths.get(img_name, [])

        for pred in preds:
            cls_name = CLASS_ID_TO_NAME.get(pred["class_id"], str(pred["class_id"]))

            # Find best matching GT
            best_iou = 0.0
            best_gt_class = None
            for gt in gts:
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_class = CLASS_ID_TO_NAME.get(gt["class_id"], str(gt["class_id"]))

            if best_iou >= iou_threshold and cls_name == best_gt_class:
                correct_confs[cls_name].append(pred["confidence"])
            else:
                incorrect_confs[cls_name].append(pred["confidence"])

    # Get all class names
    all_classes = sorted(set(list(correct_confs.keys()) + list(incorrect_confs.keys())))

    if not all_classes:
        print("WARNING: No predictions to plot confidence histograms")
        return

    n_classes = len(all_classes)
    fig, axes = plt.subplots(1, max(n_classes, 1), figsize=(6 * max(n_classes, 1), 5))
    if n_classes == 1:
        axes = [axes]

    for ax, cls_name in zip(axes, all_classes, strict=False):
        correct = correct_confs.get(cls_name, [])
        incorrect = incorrect_confs.get(cls_name, [])

        bins = np.linspace(0, 1, 30)

        if correct:
            ax.hist(
                correct,
                bins=bins,
                alpha=0.6,
                color="#2ecc71",
                label=f"Correct ({len(correct)})",
                density=True,
            )
        if incorrect:
            ax.hist(
                incorrect,
                bins=bins,
                alpha=0.6,
                color="#e74c3c",
                label=f"FP ({len(incorrect)})",
                density=True,
            )

        ax.set_xlabel("Confidence Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Class: {cls_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Confidence Calibration by Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_hist_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: confidence_hist_by_class.png")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(args: argparse.Namespace) -> None:
    """Run full error analysis pipeline."""

    print("=" * 60)
    print("  Error Analysis — Best Model")
    print("=" * 60)

    # Resolve weights
    weights_path = resolve_weights(args.model, args.weights)
    print(f"\nModel:      {args.model}")
    print(f"Weights:    {weights_path}")
    print(f"IoU thr:    {args.iou_threshold}")
    print(f"Conf thr:   {args.conf_threshold}")
    print(f"Device:     {args.device}")

    # Collect test images
    if not TEST_IMAGES.exists():
        print(f"ERROR: Test images directory not found: {TEST_IMAGES}")
        sys.exit(1)

    image_paths = sorted(TEST_IMAGES.glob("*.png")) + sorted(TEST_IMAGES.glob("*.jpg"))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"ERROR: No images found in {TEST_IMAGES}")
        sys.exit(1)

    print(f"\nTest set:   {len(image_paths)} images")
    print(f"Labels dir: {TEST_LABELS}")

    # Load ground truth
    all_ground_truths: dict[str, list[dict]] = {}
    all_gt_flat: list[dict] = []
    for img_path in image_paths:
        label_path = TEST_LABELS / (img_path.stem + ".txt")
        gt = load_yolo_labels(label_path)
        all_ground_truths[img_path.name] = gt
        for g in gt:
            g["image"] = img_path.name
        all_gt_flat.extend(gt)

    total_gt = len(all_gt_flat)
    print(f"GT objects: {total_gt}")

    # Run inference
    print("\nRunning inference...")
    if args.model in ("yolo26s", "yolo26m", "yolo26l", "yolo26x", "yolo26n", "yolo11m", "yolov8m"):
        all_predictions = run_yolo_inference(
            weights_path,
            image_paths,
            imgsz=args.imgsz,
            conf_threshold=args.conf_threshold,
            device=args.device,
        )
    elif args.model == "faster_rcnn":
        all_predictions = run_fasterrcnn_inference(
            weights_path,
            image_paths,
            conf_threshold=args.conf_threshold,
            device=args.device,
        )
    elif args.model == "detr":
        all_predictions = run_detr_inference(
            weights_path,
            image_paths,
            conf_threshold=args.conf_threshold,
            device=args.device,
        )
    else:
        print(f"ERROR: Unknown model type: {args.model}")
        sys.exit(1)

    total_preds = sum(len(p) for p in all_predictions.values())
    print(f"Predictions: {total_preds}")

    # Classify errors per image
    all_errors_combined: dict[str, Any] = {
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
        "localization_errors": [],
        "class_confusion": [],
        "background_fp": [],
        "duplicates": [],
        "missed": [],
    }

    for img_name in sorted(all_ground_truths.keys()):
        gts = all_ground_truths[img_name]
        preds = all_predictions.get(img_name, [])

        # Reset error types for this image
        for g in gts:
            g.pop("error_type", None)

        errors = classify_errors(preds, gts, iou_threshold=args.iou_threshold)

        for key in all_errors_combined:
            all_errors_combined[key].extend(errors[key])

    # Summary counts
    n_tp = len(all_errors_combined["true_positives"])
    n_fp = len(all_errors_combined["false_positives"])
    n_fn = len(all_errors_combined["false_negatives"])

    print("\nResults:")
    print(f"  True Positives:  {n_tp}")
    print(f"  False Positives: {n_fp}")
    print(f"  False Negatives: {n_fn}")
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (total_gt) if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1:              {f1:.4f}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_error_type_pie(all_errors_combined, output_dir)
    plot_size_error_bars(
        compute_size_analysis(all_errors_combined, all_gt_flat),
        output_dir,
    )

    # Confusion matrix (for single class, mostly TP/FN/FP)
    confusion = compute_confusion_matrix(
        all_predictions,
        all_ground_truths,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
    )
    plot_confusion_heatmap(confusion, output_dir)
    plot_confidence_histograms(
        all_predictions,
        all_ground_truths,
        iou_threshold=args.iou_threshold,
        output_dir=output_dir,
    )

    # Size analysis
    size_analysis = compute_size_analysis(all_errors_combined, all_gt_flat)

    # Error type percentages
    total_errors = n_fp + n_fn  # Total errors
    error_type_counts = {
        "localization": len(all_errors_combined["localization_errors"]),
        "class_confusion": len(all_errors_combined["class_confusion"]),
        "background_fp": len(all_errors_combined["background_fp"]),
        "missed": len(all_errors_combined["missed"]),
        "duplicates": len(all_errors_combined["duplicates"]),
    }
    error_type_pcts = {}
    for k, v in error_type_counts.items():
        error_type_pcts[k] = round(v / total_errors, 4) if total_errors > 0 else 0.0

    # Build summary report
    report = {
        "best_model": args.model,
        "weights_path": str(weights_path),
        "test_images": len(image_paths),
        "total_gt_objects": total_gt,
        "total_predictions": total_preds,
        "thresholds": {
            "iou": args.iou_threshold,
            "confidence": args.conf_threshold,
        },
        "overall_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "false_positives": {
            "count": n_fp,
            "rate": round(n_fp / total_preds, 4) if total_preds > 0 else 0.0,
        },
        "false_negatives": {
            "count": n_fn,
            "rate": round(n_fn / total_gt, 4) if total_gt > 0 else 0.0,
        },
        "error_types": {
            "localization": {
                "count": error_type_counts["localization"],
                "pct": error_type_pcts["localization"],
            },
            "class_confusion": {
                "count": error_type_counts["class_confusion"],
                "pct": error_type_pcts["class_confusion"],
            },
            "background_fp": {
                "count": error_type_counts["background_fp"],
                "pct": error_type_pcts["background_fp"],
            },
            "missed": {"count": error_type_counts["missed"], "pct": error_type_pcts["missed"]},
            "duplicates": {
                "count": error_type_counts["duplicates"],
                "pct": error_type_pcts["duplicates"],
            },
        },
        "size_analysis": size_analysis,
        "confusion_matrix": confusion,
    }

    # Save report
    report_path = output_dir / "error_summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  Saved: error_summary_report.json")

    # Generate findings text
    findings = generate_findings(args.model, report, size_analysis, all_errors_combined)

    # Print findings
    print("\n" + "=" * 60)
    print(findings)
    print("=" * 60)

    # Save findings alongside report
    findings_path = output_dir / "key_findings.md"
    with open(findings_path, "w") as f:
        f.write(findings)
    print("\n  Saved: key_findings.md")
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


def generate_findings(
    model_name: str,
    report: dict,
    size_analysis: dict,
    all_errors: dict,
) -> str:
    """Generate human-readable key findings text."""
    n_tp = len(all_errors["true_positives"])
    n_fp = len(all_errors["false_positives"])
    n_fn = len(all_errors["false_negatives"])
    total_gt = report["total_gt_objects"]
    total_preds = report["total_predictions"]

    # Identify worst size category
    worst_size = max(size_analysis, key=lambda k: size_analysis[k]["miss_rate"])
    worst_rate = size_analysis[worst_size]["miss_rate"]

    # Identify dominant error type
    error_types = report["error_types"]
    dominant_error = max(error_types, key=lambda k: error_types[k]["count"])
    dominant_count = error_types[dominant_error]["count"]

    # Build findings
    findings = f"""## Error Analysis Findings — {model_name}

### Summary
- **Total GT objects**: {total_gt}
- **Total predictions**: {total_preds}
- **Correct detections (TP)**: {n_tp} ({n_tp/total_gt:.1%} of GT)
- **False Positives**: {n_fp}
- **False Negatives**: {n_fn}
- **Precision**: {report['overall_metrics']['precision']:.4f}
- **Recall**: {report['overall_metrics']['recall']:.4f}
- **F1 Score**: {report['overall_metrics']['f1_score']:.4f}

### Most Common Errors
1. **{dominant_error.replace('_', ' ').title()}** ({dominant_count} instances, {error_types[dominant_error]['pct']:.1%} of errors)
"""

    # Sort error types by count
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1]["count"], reverse=True)
    for i, (err_type, err_data) in enumerate(sorted_errors[1:], 2):
        if err_data["count"] > 0:
            findings += f"{i}. **{err_type.replace('_', ' ').title()}** ({err_data['count']} instances, {err_data['pct']:.1%})\n"

    findings += f"""
### Size-Based Analysis
- **Small objects** (<1% of image): {size_analysis['small']['miss_rate']:.1%} miss rate ({size_analysis['small']['missed']}/{size_analysis['small']['total']})
- **Medium objects** (1-10%): {size_analysis['medium']['miss_rate']:.1%} miss rate ({size_analysis['medium']['missed']}/{size_analysis['medium']['total']})
- **Large objects** (>10%): {size_analysis['large']['miss_rate']:.1%} miss rate ({size_analysis['large']['missed']}/{size_analysis['large']['total']})
- **Worst category**: {worst_size} objects ({worst_rate:.1%} miss rate)

### Key Findings
"""

    if size_analysis["small"]["miss_rate"] > 0.3:
        findings += "- **Critical**: Small object detection is severely impaired — models struggle with objects <1% of image area\n"
    if size_analysis["medium"]["miss_rate"] > 0.2:
        findings += "- **Warning**: Medium object detection has notable miss rates — consider augmentations targeting mid-size objects\n"
    if size_analysis["large"]["miss_rate"] < 0.1:
        findings += "- **Good**: Large objects are detected reliably (>90% recall)\n"

    if error_types["localization"]["count"] > 0:
        findings += f"- **Localization**: {error_types['localization']['count']} predictions had IoU in [0.1, 0.5) — boxes are roughly correct but imprecise\n"
    if error_types["background_fp"]["count"] > 0:
        findings += f"- **Background FPs**: {error_types['background_fp']['count']} predictions with IoU < 0.1 — likely false alarms on similar-looking terrain\n"

    findings += """
### Recommendations
1. **Small-object augmentation**: Apply mosaic, random crop+resize, or Copy-Paste augmentation to improve small object detection
2. **Confidence threshold tuning**: Consider raising conf threshold if background FP rate is high
3. **Data imbalance**: If large objects dominate recall but small objects fail, add more small-object training examples
4. **Model ensemble**: Combine Faster R-CNN (better localization) with YOLO (better recall) for complementary strengths
"""
    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Need to import torchvision inside the main function to avoid import errors
    # if torchvision is not installed (for YOLO-only paths)
    try:
        import torchvision
    except ImportError:
        print("WARNING: torchvision not available. Faster R-CNN and DETR analysis will fail.")
        print("Install with: pip install torchvision")

    args = parse_args()
    run_analysis(args)
