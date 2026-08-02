"""DETR 3-fold cross-validation training for archaeological hole detection.

Uses torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).
Custom training loop with Hungarian matching, set criterion, and AdamW optimizer.
Transformer-based end-to-end detection (no NMS needed).

Usage
-----
    python scripts/train_detr.py                        # full 3-fold CV, 100 epochs
    python scripts/train_detr.py --epochs 10            # quick smoke test
    python scripts/train_detr.py --folds 2              # 2-fold only
    python scripts/train_detr.py --batch-size 2         # reduce if OOM
    python scripts/train_detr.py --no-mlflow            # skip MLflow logging
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, generalized_box_iou

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
logger = logging.getLogger("train_detr")


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
# Scene-ID extraction (same as train_cv.py / train_faster_rcnn.py)
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


class DETRDataset(Dataset):
    """PyTorch dataset for DETR: reads YOLO-format labels, resizes with padding.

    YOLO format: class_id cx cy w h (normalized [0,1])
    DETR targets: labels (0-indexed), boxes as [cx, cy, w, h] normalized [0,1]

    Images are resized to fit within ``image_size x image_size`` while
    maintaining aspect ratio, then zero-padded to the target size.  Box
    coordinates are transformed to match the resized + padded image.
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

        # Compute resize scale to fit within image_size x image_size
        scale = min(self.image_size / orig_w, self.image_size / orig_h)
        new_w = orig_w * scale
        new_h = orig_h * scale
        pad_x = (self.image_size - new_w) / 2.0
        pad_y = (self.image_size - new_h) / 2.0

        # Resize image
        resized_img = img.resize((int(new_w), int(new_h)), Image.Resampling.BILINEAR)

        # Pad to image_size x image_size with zeros (black)
        padded_img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        padded_img.paste(resized_img, (int(pad_x), int(pad_y)))

        # Normalize to [0,1]
        img_tensor = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1).float() / 255.0

        # Read YOLO labels and transform to padded image coordinates
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

                    # Transform YOLO coords to resized + padded image
                    # Original pixel: cx_px = cx * orig_w, cy_px = cy * orig_h
                    # After resize: cx_px * scale, cy_px * scale
                    # After pad: + pad_x, + pad_y
                    # Normalized to image_size: / image_size
                    cx_new = (cx * orig_w * scale + pad_x) / self.image_size
                    cy_new = (cy * orig_h * scale + pad_y) / self.image_size
                    w_new = (w * orig_w * scale) / self.image_size
                    h_new = (h * orig_h * scale) / self.image_size

                    # Clamp to [0, 1]
                    cx_new = max(0.0, min(1.0, cx_new))
                    cy_new = max(0.0, min(1.0, cy_new))
                    w_new = max(0.0, min(1.0, w_new))
                    h_new = max(0.0, min(1.0, h_new))

                    # Skip degenerate boxes
                    if w_new <= 0.001 or h_new <= 0.001:
                        continue

                    boxes.append([cx_new, cy_new, w_new, h_new])
                    labels.append(class_id)  # 0-indexed for DETR

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {"labels": labels_tensor, "boxes": boxes_tensor}
        return img_tensor, target


def collate_fn(batch):
    """Custom collate: returns (image_tensor_list, target_list) for DETR."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------


class HungarianMatcher(nn.Module):
    """Computes an optimal bipartite matching between predictions and ground truth.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
    the one-to-one assignment that minimizes the total matching cost.

    Cost = cost_class * CE_class + cost_bbox * L1 + cost_giou * GIoU
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be zero"

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        pred_logits : [batch_size, num_queries, num_classes + 1]
        pred_boxes : [batch_size, num_queries, 4] (cx, cy, w, h) normalized
        targets : list of dicts with 'labels' [num_gt] and 'boxes' [num_gt, 4]

        Returns
        -------
        List of (pred_indices, gt_indices) tuples per sample in the batch.
        """
        batch_size, num_queries, num_classes = pred_logits.shape

        # Remove no-object class for cost computation
        pred_scores = pred_logits.softmax(-1)[:, :, :-1]  # [B, Q, C]

        indices = []
        for b in range(batch_size):
            target_labels = targets[b]["labels"]  # [num_gt]
            target_boxes = targets[b]["boxes"]  # [num_gt, 4]
            num_gt = len(target_labels)

            if num_gt == 0:
                # No GT: empty matching
                indices.append(
                    (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
                )
                continue

            # Classification cost: -prob[gt_class] for each query-GT pair
            # pred_scores[b]: [Q, C], target_labels: [num_gt]
            cost_class = -pred_scores[b][:, target_labels]  # [Q, num_gt]

            # L1 cost
            cost_bbox = torch.cdist(pred_boxes[b], target_boxes, p=1)  # [Q, num_gt]

            # GIoU cost (need xyxy format)
            pred_xyxy = self._cxcywh_to_xyxy(pred_boxes[b])
            gt_xyxy = self._cxcywh_to_xyxy(target_boxes)
            cost_giou = -generalized_box_iou(pred_xyxy, gt_xyxy)  # [Q, num_gt]

            # Total cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            # Hungarian matching
            C = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append(
                (torch.tensor(row_ind, dtype=torch.int64), torch.tensor(col_ind, dtype=torch.int64))
            )

        return indices

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Set Criterion (DETR Loss)
# ---------------------------------------------------------------------------


class GIoULoss(nn.Module):
    """Generalized IoU loss: 1 - GIoU.  Input boxes in xyxy format."""

    def __init__(self, reduction: str = "none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_boxes : [N, 4] (x1, y1, x2, y2)
        target_boxes : [N, 4] (x1, y1, x2, y2)

        Returns
        -------
        [N] loss per box pair (1 - GIoU)
        """
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        # Areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        )

        # Union
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # Enclosing box
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        # GIoU
        giou = iou - (enc_area - union_area) / torch.clamp(enc_area, min=1e-6)

        loss = 1.0 - giou
        return loss


class SetCriterion(nn.Module):
    """DETR set-based loss using Hungarian matching.

    Computes three losses:
    1. Classification: cross-entropy on matched queries
    2. L1 box regression: on matched queries only
    3. GIoU: on matched queries only

    Unmatched queries are supervised with the no-object class (eos_coef weight).
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes (excluding no-object).
        matcher : HungarianMatcher
        weight_dict : dict mapping loss names to weights
        eos_coef : float
            Weight for no-object class in classification loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.empty_weight = torch.tensor([1.0, eos_coef])  # [class_weight, no_object_weight]

        # Loss functions
        self.loss_ce = nn.CrossEntropyLoss(weight=self.empty_weight)
        self.loss_bbox = nn.L1Loss(reduction="none")
        self.loss_giou = GIoULoss(reduction="none")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        outputs : dict with 'pred_logits' [B, Q, C+1] and 'pred_boxes' [B, Q, 4]
        targets : list of dicts with 'labels' and 'boxes'

        Returns
        -------
        dict of weighted losses
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # Compute losses
        losses = {}
        losses.update(self._loss_labels(pred_logits, targets, indices))
        losses.update(self._loss_boxes(pred_boxes, targets, indices))

        return losses

    def _loss_labels(
        self,
        pred_logits: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Classification loss (cross-entropy)."""
        batch_size, num_queries, _ = pred_logits.shape

        # Target class for each query: fill with no-object class (num_classes)
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # no-object class index
            dtype=torch.int64,
            device=pred_logits.device,
        )

        # Fill matched queries with their GT class
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][gt_idx]

        # Cross-entropy loss (on all queries)
        # Move empty_weight to same device
        self.empty_weight = self.empty_weight.to(pred_logits.device)
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # [B, C+1, Q]
            target_classes,  # [B, Q]
            weight=self.empty_weight,
        )

        return {"loss_ce": loss_ce}

    def _loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Box regression losses (L1 + GIoU) on matched pairs only."""
        num_matched = sum(len(pred_idx) for pred_idx, _ in indices)

        if num_matched == 0:
            return {
                "loss_bbox": torch.tensor(0.0, device=pred_boxes.device),
                "loss_giou": torch.tensor(0.0, device=pred_boxes.device),
            }

        # Gather matched predictions and targets
        pred_boxes_matched_list: list[torch.Tensor] = []
        target_boxes_matched_list: list[torch.Tensor] = []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_matched_list.append(pred_boxes[b, pred_idx])
                target_boxes_matched_list.append(targets[b]["boxes"][gt_idx])

        pred_boxes_matched = torch.cat(pred_boxes_matched_list, dim=0)
        target_boxes_matched = torch.cat(target_boxes_matched_list, dim=0)

        # L1 loss
        loss_bbox = self.loss_bbox(pred_boxes_matched, target_boxes_matched).mean()

        # GIoU loss
        pred_xyxy = HungarianMatcher._cxcywh_to_xyxy(pred_boxes_matched)
        target_xyxy = HungarianMatcher._cxcywh_to_xyxy(target_boxes_matched)
        loss_giou = self.loss_giou(pred_xyxy, target_xyxy).mean()

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}


# ---------------------------------------------------------------------------
# mAP50 evaluation
# ---------------------------------------------------------------------------


def compute_map50(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> float:
    """Compute mAP50 across all images.

    For each image:
    - Match predicted boxes to ground truth using IoU >= threshold
    - AP = fraction of GT matched by confident predictions
    - mAP50 = mean of per-image APs
    """
    aps = []

    for pred, target in zip(all_preds, all_targets, strict=False):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        n_gt = len(gt_boxes)

        if n_gt == 0:
            if len(pred["scores"]) == 0:
                aps.append(1.0)
            else:
                aps.append(0.0)
            continue

        # Filter by score
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
        ap = precision
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


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
# DETR output post-processing for evaluation
# ---------------------------------------------------------------------------


def postprocess_detr_output(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    score_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Convert DETR output to filtered predictions for mAP evaluation.

    Parameters
    ----------
    pred_logits : [Q, num_classes+1]
    pred_boxes : [Q, 4] (cx, cy, w, h) normalized [0,1]
    score_threshold : minimum confidence to keep a prediction

    Returns
    -------
    dict with 'boxes' [N, 4] in xyxy pixel coords, 'labels' [N], 'scores' [N]
    """
    # Get class probabilities (exclude no-object class)
    probs = F.softmax(pred_logits, dim=-1)  # [Q, C+1]
    scores, labels = probs[:, :-1].max(dim=-1)  # [Q], [Q] (class 0 = hole)

    # Filter by score
    mask = scores >= score_threshold
    scores = scores[mask]
    labels = labels[mask]
    boxes_cxcywh = pred_boxes[mask]  # [N, 4]

    if len(boxes_cxcywh) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    # Convert cx,cy,w,h -> x1,y1,x2,y2 pixel coordinates (image_size=640)
    image_size = 640
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w) * image_size
    y1 = (cy - 0.5 * h) * image_size
    x2 = (cx + 0.5 * w) * image_size
    y2 = (cy + 0.5 * h) * image_size
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    return {"boxes": boxes_xyxy, "labels": labels, "scores": scores}


def targets_to_pixel_coords(
    targets: list[dict[str, torch.Tensor]],
    image_size: int = 640,
) -> list[dict[str, torch.Tensor]]:
    """Convert target boxes from normalized cx,cy,w,h to pixel xyxy."""
    converted = []
    for t in targets:
        boxes = t["boxes"]
        if len(boxes) == 0:
            converted.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": t["labels"],
                }
            )
            continue
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * image_size
        y1 = (cy - 0.5 * h) * image_size
        x2 = (cx + 0.5 * w) * image_size
        y2 = (cy + 0.5 * h) * image_size
        converted.append(
            {
                "boxes": torch.stack([x1, y1, x2, y2], dim=-1),
                "labels": t["labels"],
            }
        )
    return converted


# ---------------------------------------------------------------------------
# Parent-scene CV splits (same as train_cv.py / train_faster_rcnn.py)
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
# Model loading
# ---------------------------------------------------------------------------


def create_model(
    num_classes: int = 1,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Load DETR ResNet-50 from torch.hub with custom num_classes.

    The pretrained checkpoint is for 91 COCO classes, but we need ``num_classes``
    (typically 1).  We load the model with the target num_classes first (random
    class head), then load the pretrained state dict with ``strict=False`` to
    skip the class_embed size mismatch — the backbone, transformer, and bbox head
    weights are all compatible.

    Parameters
    ----------
    num_classes : int
        Number of detection classes (excluding no-object).
    pretrained : bool
        Load COCO-pretrained backbone+transformer weights.

    Returns
    -------
    DETR model with class head adapted to num_classes.
    """
    # Step 1: create model with target num_classes (random class head)
    logger.info("Loading DETR ResNet-50 from torch.hub (num_classes=%d)...", num_classes)
    model = torch.hub.load(
        "facebookresearch/detr",
        "detr_resnet50",
        pretrained=False,
        num_classes=num_classes,
    )

    # Step 2: load COCO-pretrained weights, skipping class_embed (different num_classes)
    if pretrained:
        logger.info("Loading COCO-pretrained weights (skipping class head)...")
        state_dict = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
            map_location="cpu",
        )
        # Remove class_embed keys (91+1 classes vs our 1+1) to avoid size mismatch
        pretrained_dict = {k: v for k, v in state_dict["model"].items() if "class_embed" not in k}
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        logger.info(
            "Loaded pretrained weights. Missing: %d keys (class_embed), Unexpected: %d keys",
            len(missing),
            len(unexpected),
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "DETR loaded: %.1fM parameters, num_queries=100, num_classes=%d",
        n_params / 1e6,
        num_classes,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_fold(
    fold_idx: int,
    train_paths: list[Path],
    val_paths: list[Path],
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_drop: int = 200,
    patience: int = 20,
    image_size: int = 640,
    num_classes: int = 1,
    seed: int = 42,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Train DETR for one fold with early stopping.

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

    # --- Datasets ---
    train_dataset = DETRDataset(train_paths, image_size=image_size)
    val_dataset = DETRDataset(val_paths, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Model ---
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    # --- Loss components ---
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
    )

    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    # Add auxiliary loss weights (one per decoder layer, layers 0..4 of 6 total)
    num_aux = 5  # DETR has 6 decoder layers, aux losses for first 5
    for i in range(num_aux):
        weight_dict.update(
            {
                f"loss_ce_{i}": 1.0,
                f"loss_bbox_{i}": 5.0,
                f"loss_giou_{i}": 2.0,
            }
        )

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
    )
    criterion.to(device)

    # --- Optimizer and scheduler ---
    # Separate learning rates for backbone and transformer
    backbone_params = []
    transformer_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "transformer" in name:
            transformer_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},  # backbone: 10x lower LR
        {"params": transformer_params, "lr": lr},
        {"params": other_params, "lr": lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # lr_drop: reduce LR by 10x at epoch lr_drop
    def lr_lambda(epoch):
        return 0.1 ** (1 if epoch >= lr_drop else 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Early stopping ---
    best_map50 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = PROJECT_ROOT / "experiments" / "detr" / f"fold_{fold_idx}_best.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Metrics tracking ---
    train_losses = []
    val_map50s = []
    epoch_times = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ===== Training =====
        model.train()
        criterion.train()
        epoch_losses = []

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass — model returns predictions dict
            outputs = model(images)

            # Compute losses
            loss_dict = criterion(outputs, targets)

            # Weight and sum all losses
            total_loss = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict
                if k in criterion.weight_dict
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_losses.append(total_loss.item())

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        train_losses.append(avg_train_loss)

        # ===== Validation =====
        model.eval()
        all_preds = []
        all_targets_px = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                # Process each sample in the batch
                for b in range(len(images)):
                    pred = postprocess_detr_output(
                        outputs["pred_logits"][b].cpu(),
                        outputs["pred_boxes"][b].cpu(),
                        score_threshold=0.05,
                    )
                    all_preds.append(pred)

                # Convert targets to pixel coords for mAP
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                targets_px = targets_to_pixel_coords(targets_cpu, image_size=image_size)
                all_targets_px.extend(targets_px)

        map50 = compute_map50(all_preds, all_targets_px, iou_threshold=0.5, score_threshold=0.05)
        val_map50s.append(map50)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Logging
        current_lr = optimizer.param_groups[1]["lr"]  # transformer LR
        logger.info(
            "Fold %d | Epoch %d/%d | loss=%.4f | mAP50=%.4f | lr=%.2e | time=%.1fs",
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
        if map50 > best_map50 + 0.001:
            best_map50 = map50
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= patience:
            logger.info(
                "Fold %d: Early stopping at epoch %d (best=%d, patience=%d)",
                fold_idx,
                epoch + 1,
                best_epoch,
                patience,
            )
            break

    total_time = time.time() - start_time
    epochs_trained = epoch + 1

    # ===== Final evaluation with best model =====
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
            for b in range(len(images)):
                pred = postprocess_detr_output(
                    outputs["pred_logits"][b].cpu(),
                    outputs["pred_boxes"][b].cpu(),
                    score_threshold=0.05,
                )
                all_preds_final.append(pred)
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            all_targets_final.extend(targets_to_pixel_coords(targets_cpu, image_size=image_size))

    final_map50 = compute_map50(
        all_preds_final, all_targets_final, iou_threshold=0.5, score_threshold=0.05
    )
    final_metrics = compute_precision_recall(
        all_preds_final, all_targets_final, iou_threshold=0.5, score_threshold=0.05
    )

    # Peak VRAM
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
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
        "peak_vram_gb": round(peak_vram_gb, 2),
    }

    logger.info(
        "Fold %d complete: mAP50=%.4f, precision=%.4f, recall=%.4f (peak_vram=%.2fGB, %.1fs, %d epochs)",
        fold_idx,
        final_map50,
        final_metrics["precision"],
        final_metrics["recall"],
        peak_vram_gb,
        total_time,
        epochs_trained,
    )

    return result


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DETR ResNet-50 3-fold cross-validation training.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr-drop", type=int, default=200, help="Epoch to drop LR by 10x")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--image-size", type=int, default=640, help="Training image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return parser.parse_args(argv)


def main() -> dict[str, Any]:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("DETR ResNet-50 3-Fold Cross-Validation Training")
    logger.info("=" * 70)
    logger.info(
        "Config: epochs=%d, folds=%d, batch_size=%d, lr=%.2e, weight_decay=%.2e",
        args.epochs,
        args.folds,
        args.batch_size,
        args.lr,
        args.weight_decay,
    )
    logger.info("Device: %s", "CUDA" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / (1024**3))

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
    exp_dir = PROJECT_ROOT / "experiments" / "detr"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        experiment = mlflow.set_experiment("detr")
        logger.info("MLflow experiment: detr (id=%s)", experiment.experiment_id)

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
                "model_family": "detr",
                "model_scale": "none",
                "cv_fold": str(fold_idx),
                "experiment_type": "training",
                "augmentation": "none",
                "image_size": str(args.image_size),
            }
            run = mlflow.start_run(
                run_name=f"detr_fold{fold_idx}",
                tags=tags,
            )
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "training/epochs": args.epochs,
                    "training/batch_size": args.batch_size,
                    "training/lr": args.lr,
                    "training/weight_decay": args.weight_decay,
                    "training/lr_drop": args.lr_drop,
                    "training/early_stopping_patience": args.patience,
                    "data/image_size": args.image_size,
                    "model/arch": "detr_resnet50",
                    "model/num_classes": 1,
                    "model/num_queries": 100,
                    "model/aux_loss": True,
                    "cv/n_folds": args.folds,
                    "cv/seed": args.seed,
                    "detr/loss_ce_weight": 1.0,
                    "detr/loss_bbox_weight": 5.0,
                    "detr/loss_giou_weight": 2.0,
                    "detr/eos_coef": 0.1,
                    "detr/matcher_cost_class": 1.0,
                    "detr/matcher_cost_bbox": 5.0,
                    "detr/matcher_cost_giou": 2.0,
                }
            )

        # Reset VRAM tracking per fold
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            result = train_one_fold(
                fold_idx=fold_idx,
                train_paths=train_paths,
                val_paths=val_paths,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_drop=args.lr_drop,
                patience=args.patience,
                image_size=args.image_size,
                seed=args.seed,
                use_mlflow=not args.no_mlflow,
            )
            fold_results.append(result)

            # Log final fold metrics to MLflow
            if run_id and not args.no_mlflow:
                mlflow.log_metrics(result["metrics"])
                mlflow.log_metric("peak_vram_gb", result.get("peak_vram_gb", 0.0))

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
    print(f"CROSS-VALIDATION RESULTS (DETR, {args.folds}-fold)")
    print(f"{'=' * 60}")
    for key, (mean_val, std_val) in aggregated.items():
        print(f"  {key:<25s}  {mean_val:.4f} +/- {std_val:.4f}")
    print(f"  {'total_time':<25s}  {all_fold_time:.1f}s ({all_fold_time/60:.1f} min)")
    peak_vram = [r.get("peak_vram_gb", 0) for r in fold_results]
    if peak_vram:
        print(f"  {'peak_vram_gb':<25s}  {max(peak_vram):.2f} GB")
    print(f"{'=' * 60}\n")

    # Save results.json
    results_data = {
        "experiment": "detr",
        "model": "detr",
        "model_arch": "detr_resnet50",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_drop": args.lr_drop,
            "precision": "fp32",
            "patience": args.patience,
            "num_queries": 100,
            "aux_loss": True,
        },
        "detr_loss": {
            "loss_ce_weight": 1.0,
            "loss_bbox_weight": 5.0,
            "loss_giou_weight": 2.0,
            "eos_coef": 0.1,
            "matcher_cost_class": 1.0,
            "matcher_cost_bbox": 5.0,
            "matcher_cost_giou": 2.0,
        },
        "cv": {
            "n_folds": args.folds,
            "split_strategy": "parent_scene",
            "seed": args.seed,
        },
        "fold_results": fold_results,
        "cross_fold_metrics": {k: {"mean": v[0], "std": v[1]} for k, v in aggregated.items()},
        "total_time_seconds": round(all_fold_time, 1),
        "notes": (
            f"DETR ResNet-50 from torch.hub (facebookresearch/detr), pretrained on COCO 91-class. "
            f"Class head replaced for 1-class detection (hole). "
            f"Batch_size={args.batch_size} (transformer-based, VRAM-heavy). "
            f"AdamW lr={args.lr}, weight_decay={args.weight_decay}, lr_drop={args.lr_drop}. "
            f"Hungarian matcher with L1+GIoU+CE costs. "
            f"Set criterion: CE (eos_coef=0.1) + L1*5 + GIoU*2. "
            f"Auxiliary losses from intermediate decoder layers. "
            f"Early stopping patience={args.patience}. "
            f"Total training time: {all_fold_time/60:.1f} min."
        ),
    }

    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results_data


if __name__ == "__main__":
    main()
