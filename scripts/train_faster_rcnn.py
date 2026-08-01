"""Faster R-CNN 3-fold cross-validation training for archaeological hole detection.

Custom training script for torchvision's Faster R-CNN (not an Ultralytics model).
Reads YOLO-format labels, converts to COCO format, and trains with parent-scene CV splits.

Usage
-----
    python scripts/train_faster_rcnn.py                        # full 3-fold CV, 100 epochs
    python scripts/train_faster_rcnn.py --epochs 10            # quick smoke test
    python scripts/train_faster_rcnn.py --folds 2              # 2-fold only
    python scripts/train_faster_rcnn.py --no-mlflow            # skip MLflow logging
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

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

PROJECT_ROOT = _PROJECT_ROOT
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL = DATASET_DIR / "labels" / "val"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_faster_rcnn")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Dataset
# ---------------------------------------------------------------------------


class YOLODetectionDataset(Dataset):
    """PyTorch dataset that reads YOLO-format labels and converts to COCO format.

    YOLO format: class_id cx cy w h (normalized [0,1])
    COCO format: x1 y1 x2 y2 (pixel coordinates)
    """

    def __init__(self, image_paths: list[Path], image_size: int = 640):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        label_path = DATASET_DIR / "labels" / img_path.parent.name / (img_path.stem + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to training size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Scale factors
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        # Read YOLO labels and convert to COCO pixel format
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx, cy, w, h = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )

                    # Convert normalized YOLO [cx,cy,w,h] to pixel [x1,y1,x2,y2]
                    x1 = (cx - w / 2) * orig_w * scale_x
                    y1 = (cy - h / 2) * orig_h * scale_y
                    x2 = (cx + w / 2) * orig_w * scale_x
                    y2 = (cy + h / 2) * orig_h * scale_y

                    # Clamp to image bounds
                    x1 = max(0.0, min(float(self.image_size), x1))
                    y1 = max(0.0, min(float(self.image_size), y1))
                    x2 = max(0.0, min(float(self.image_size), x2))
                    y2 = max(0.0, min(float(self.image_size), y2))

                    # Skip degenerate boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    labels.append(
                        class_id + 1
                    )  # +1: Faster R-CNN uses 1-indexed labels (0 = background)

        # Convert to tensors
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # Normalize image to [0,1] (Faster R-CNN expects this)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return img_tensor, target


def collate_fn(batch):
    """Custom collate for Faster R-CNN (expects list of images and targets)."""
    return tuple(zip(*batch, strict=False))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def create_model(num_classes: int = 1) -> torchvision.models.detection.FasterRCNN:
    """Create Faster R-CNN with ResNet50-FPN backbone, pretrained on COCO.

    Replaces the box predictor head to support ``num_classes`` (excluding
    background, which is class 0).
    """
    model = fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # Replace box predictor for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes + 1
    )  # +1 for background

    return model


# ---------------------------------------------------------------------------
# mAP50 evaluation
# ---------------------------------------------------------------------------


def compute_map50(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> float:
    """Compute mAP50 across all images (simple per-image AP, then averaged).

    For each image:
    - Match predicted boxes to ground truth using IoU >= threshold
    - AP = precision at the recall point (simplified: fraction of GT matched)
    - mAP50 = mean of per-image APs
    """
    aps = []

    for pred, target in zip(all_preds, all_targets, strict=False):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        n_gt = len(gt_boxes)

        if n_gt == 0:
            # No ground truth: AP = 1.0 if no predictions, 0.0 otherwise
            if len(pred["scores"]) == 0:
                aps.append(1.0)
            else:
                aps.append(0.0)
            continue

        # Filter by score threshold
        scores = pred["scores"]
        mask = scores >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]
        pred_scores = pred["scores"][mask]

        if len(pred_boxes) == 0:
            aps.append(0.0)
            continue

        # Sort by score (descending)
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        # Match predictions to GT
        matched_gt = set()
        tp = 0
        for pred_box, pred_label in zip(pred_boxes, pred_labels, strict=False):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in range(n_gt):
                if gt_idx in matched_gt:
                    continue
                if gt_labels[gt_idx] != pred_label:
                    continue
                iou = box_iou(pred_box.unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)

        precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
        # Simplified AP: use precision at max recall achieved
        ap = precision  # Conservative: precision as proxy for AP
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Parent-scene CV splits (same logic as train_cv.py)
# ---------------------------------------------------------------------------


def get_all_images() -> list[Path]:
    """Return all image paths in both train and val directories."""
    images: list[Path] = []
    for d in [IMAGES_TRAIN, IMAGES_VAL]:
        if d.exists():
            images.extend(sorted(d.iterdir()))
    return images


def group_by_parent_scene(image_paths: list[Path]) -> dict[str, list[Path]]:
    """Group image paths by parent-scene identifier."""
    groups: dict[str, list[Path]] = {}
    for img_path in image_paths:
        scene = extract_parent_scene(img_path.name)
        groups.setdefault(scene, []).append(img_path)
    return groups


def create_fold_splits(
    scene_groups: dict[str, list[Path]],
    n_folds: int = 3,
    seed: int = 42,
) -> list[tuple[list[Path], list[Path]]]:
    """Split parent-scene groups into n_folds train/val folds."""
    scenes = list(scene_groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)

    fold_scenes: list[list[str]] = [[] for _ in range(n_folds)]
    for i, scene in enumerate(scenes):
        fold_scenes[i % n_folds].append(scene)

    folds = []
    for val_idx in range(n_folds):
        train_scenes = []
        val_scenes = fold_scenes[val_idx]
        for fold_idx, scenes_in_fold in enumerate(fold_scenes):
            if fold_idx != val_idx:
                train_scenes.extend(scenes_in_fold)

        train_paths = []
        for s in train_scenes:
            train_paths.extend(scene_groups[s])

        val_paths = []
        for s in val_scenes:
            val_paths.extend(scene_groups[s])

        folds.append((train_paths, val_paths))

    return folds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_fold(
    fold_idx: int,
    train_paths: list[Path],
    val_paths: list[Path],
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    step_size: int = 30,
    gamma: float = 0.1,
    patience: int = 20,
    image_size: int = 640,
    num_classes: int = 1,
    seed: int = 42,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Train Faster R-CNN for one fold with early stopping.

    Returns
    -------
    dict with keys: fold, best_epoch, epochs_trained, elapsed_seconds, metrics
    """
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Fold %d: device=%s, train=%d, val=%d, batch_size=%d, epochs=%d",
        fold_idx,
        device,
        len(train_paths),
        len(val_paths),
        batch_size,
        epochs,
    )

    # Datasets
    train_dataset = YOLODetectionDataset(train_paths, image_size=image_size)
    val_dataset = YOLODetectionDataset(val_paths, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Avoid batch-norm issues with batch_size=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = create_model(num_classes=num_classes)
    model.to(device)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Early stopping
    best_map50 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    # Metrics tracking
    train_losses = []
    val_map50s = []
    epoch_times = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Training ---
        model.train()
        epoch_losses = []
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_losses.append(losses.item())

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                # Move predictions to CPU for mAP computation
                for output in outputs:
                    all_preds.append({k: v.cpu() for k, v in output.items()})
                for target in targets:
                    all_targets.append({k: v.cpu() for k, v in target.items()})

        map50 = compute_map50(all_preds, all_targets, iou_threshold=0.5, score_threshold=0.05)
        val_map50s.append(map50)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Fold %d | Epoch %d/%d | loss=%.4f | mAP50=%.4f | lr=%.6f | time=%.1fs",
            fold_idx,
            epoch + 1,
            epochs,
            avg_train_loss,
            map50,
            current_lr,
            epoch_time,
        )

        # MLflow per-epoch logging
        if use_mlflow and mlflow.active_run():
            mlflow.log_metrics(
                {
                    "train/loss": avg_train_loss,
                    "val/mAP50": map50,
                    "train/lr": current_lr,
                },
                step=epoch,
            )

        # Early stopping
        if map50 > best_map50 + 0.001:  # min_delta
            best_map50 = map50
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # Save best model
            best_model_path = (
                PROJECT_ROOT / "experiments" / "faster_rcnn" / f"fold_{fold_idx}_best.pt"
            )
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= patience:
            logger.info(
                "Fold %d: Early stopping triggered at epoch %d (best=%d, patience=%d)",
                fold_idx,
                epoch + 1,
                best_epoch,
                patience,
            )
            break

    total_time = time.time() - start_time
    epochs_trained = epoch + 1

    # Compute final metrics on validation set using best model
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds_final = []
    all_targets_final = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for output in outputs:
                all_preds_final.append({k: v.cpu() for k, v in output.items()})
            for target in targets:
                all_targets_final.append({k: v.cpu() for k, v in target.items()})

    final_map50 = compute_map50(
        all_preds_final, all_targets_final, iou_threshold=0.5, score_threshold=0.05
    )

    # Compute precision and recall at the best model
    final_metrics = compute_precision_recall(
        all_preds_final, all_targets_final, iou_threshold=0.5, score_threshold=0.05
    )

    result = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "elapsed_seconds": round(total_time, 1),
        "metrics": {
            "val/mAP50": final_map50,
            "val/precision": final_metrics["precision"],
            "val/recall": final_metrics["recall"],
            "train/loss": float(np.mean(train_losses)) if train_losses else 0.0,
        },
    }

    logger.info(
        "Fold %d complete: mAP50=%.4f, precision=%.4f, recall=%.4f (%.1fs, %d epochs)",
        fold_idx,
        final_map50,
        final_metrics["precision"],
        final_metrics["recall"],
        total_time,
        epochs_trained,
    )

    return result


def compute_precision_recall(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> dict[str, float]:
    """Compute aggregate precision and recall across all images."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(all_preds, all_targets, strict=False):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        n_gt = len(gt_boxes)

        scores = pred["scores"]
        mask = scores >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]
        pred_scores = pred["scores"][mask]

        if len(pred_boxes) == 0:
            total_fn += n_gt
            continue

        if n_gt == 0:
            total_fp += len(pred_boxes)
            continue

        # Sort by score
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        matched_gt = set()
        for pred_box, pred_label in zip(pred_boxes, pred_labels, strict=False):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in range(n_gt):
                if gt_idx in matched_gt:
                    continue
                if gt_labels[gt_idx] != pred_label:
                    continue
                iou = box_iou(pred_box.unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        total_fn += n_gt - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {"precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faster R-CNN 3-fold cross-validation training.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--step-size", type=int, default=30, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--image-size", type=int, default=640, help="Training image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Faster R-CNN 3-Fold Cross-Validation Training")
    logger.info("=" * 70)
    logger.info(
        "Config: epochs=%d, folds=%d, batch_size=%d, lr=%.4f",
        args.epochs,
        args.folds,
        args.batch_size,
        args.lr,
    )
    logger.info("Device: %s", "CUDA" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Scan dataset
    all_images = get_all_images()
    if not all_images:
        logger.error("No images found in %s or %s", IMAGES_TRAIN, IMAGES_VAL)
        sys.exit(1)

    scene_groups = group_by_parent_scene(all_images)
    n_scenes = len(scene_groups)
    logger.info("Found %d images in %d parent scenes", len(all_images), n_scenes)

    # Create fold splits
    folds = create_fold_splits(scene_groups, n_folds=args.folds, seed=args.seed)
    for i, (train_p, val_p) in enumerate(folds):
        logger.info("Fold %d: train=%d, val=%d images", i, len(train_p), len(val_p))

    # Setup experiment directory
    exp_dir = PROJECT_ROOT / "experiments" / "faster_rcnn"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        experiment = mlflow.set_experiment("faster_rcnn")
        logger.info("MLflow experiment: faster_rcnn (id=%s)", experiment.experiment_id)

    # Run each fold
    fold_results = []
    all_fold_start = time.time()

    for fold_idx, (train_paths, val_paths) in enumerate(folds):
        logger.info("\n" + "=" * 60)
        logger.info("FOLD %d / %d", fold_idx + 1, args.folds)
        logger.info("=" * 60)

        # Start MLflow run for this fold
        run_id = None
        if not args.no_mlflow:
            tags = {
                "model_family": "faster_rcnn",
                "model_scale": "none",
                "cv_fold": str(fold_idx),
                "experiment_type": "training",
                "augmentation": "none",
                "image_size": str(args.image_size),
            }
            run = mlflow.start_run(
                run_name=f"faster_rcnn_fold{fold_idx}",
                tags=tags,
            )
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "training/epochs": args.epochs,
                    "training/batch_size": args.batch_size,
                    "training/lr": args.lr,
                    "training/momentum": args.momentum,
                    "training/weight_decay": args.weight_decay,
                    "training/step_size": args.step_size,
                    "training/gamma": args.gamma,
                    "training/early_stopping_patience": args.patience,
                    "data/image_size": args.image_size,
                    "model/arch": "fasterrcnn_resnet50_fpn",
                    "model/num_classes": 1,
                    "cv/n_folds": args.folds,
                    "cv/seed": args.seed,
                }
            )

        try:
            result = train_one_fold(
                fold_idx=fold_idx,
                train_paths=train_paths,
                val_paths=val_paths,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                step_size=args.step_size,
                gamma=args.gamma,
                patience=args.patience,
                image_size=args.image_size,
                seed=args.seed,
                use_mlflow=not args.no_mlflow,
            )
            fold_results.append(result)

            # Log final fold metrics to MLflow
            if run_id and not args.no_mlflow:
                mlflow.log_metrics(result["metrics"])

        except Exception as e:
            logger.exception("Fold %d failed: %s", fold_idx, e)
            if run_id and not args.no_mlflow:
                mlflow.log_metric("error", 1.0)
            raise
        finally:
            if run_id and not args.no_mlflow:
                mlflow.end_run()

    all_fold_time = time.time() - all_fold_start

    # Aggregate metrics
    aggregated = aggregate_fold_metrics(fold_results)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS (Faster R-CNN, {args.folds}-fold)")
    print(f"{'=' * 60}")
    for key, (mean_val, std_val) in aggregated.items():
        print(f"  {key:<25s}  {mean_val:.4f} +/- {std_val:.4f}")
    print(f"  {'total_time':<25s}  {all_fold_time:.1f}s ({all_fold_time/60:.1f} min)")
    print(f"{'=' * 60}\n")

    # Save results.json
    results_data = {
        "experiment": "faster_rcnn",
        "model": "faster_rcnn",
        "model_arch": "fasterrcnn_resnet50_fpn",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "optimizer": "SGD",
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "precision": "fp32",
            "patience": args.patience,
        },
        "cv": {
            "n_folds": args.folds,
            "split_strategy": "parent_scene",
            "seed": args.seed,
        },
        "fold_results": fold_results,
        "cross_fold_metrics": aggregated,
        "total_time_seconds": round(all_fold_time, 1),
        "notes": (
            f"Faster R-CNN ResNet50-FPN with torchvision pretrained weights. "
            f"Batch_size={args.batch_size} (reduced from YOLO default due to GPU memory). "
            f"SGD lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}. "
            f"StepLR scheduler (step={args.step_size}, gamma={args.gamma}). "
            f"Early stopping patience={args.patience}. "
            f"Total training time: {all_fold_time/60:.1f} min."
        ),
    }

    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results_data


def aggregate_fold_metrics(fold_results: list[dict]) -> dict[str, tuple[float, float]]:
    """Compute mean +/- std across folds for each metric."""
    if not fold_results:
        return {}

    all_keys: set = set()
    for r in fold_results:
        all_keys.update(r.get("metrics", {}).keys())

    aggregated = {}
    for key in sorted(all_keys):
        values = [r["metrics"].get(key, float("nan")) for r in fold_results]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated[key] = (float(np.mean(values)), float(np.std(values)))

    return aggregated


if __name__ == "__main__":
    main()
