# Image Size Ablation Study — YOLO26m

> **Model:** YOLO26m  
> **Experiment:** `yolo26m` (MLflow experiment ID: 7)  
> **Method:** Single-fold (fold 0), 100 epochs, early stopping (patience=20)  
> **Device:** RTX 5070 Laptop (8 GB VRAM)  
> **Date:** 2026-07-16

---

## 1. Hyperparameters (held constant)

All three runs share identical HPO-best parameters from trial #49 (mAP50 = 0.3803 @ 30 epochs).

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW |
| lr | 0.0004701 |
| momentum | 0.8514 |
| weight_decay | 0.0007860 |
| warmup_epochs | 3.0 |
| cos_lr | False |
| mosaic | 0.5 |
| mixup | 0.5 |
| degrees | 0.0 |
| hsv_h | 0.015 |
| hsv_s | 0.0 |
| hsv_v | 0.3 |

Batch size was adjusted per image size to fit in VRAM.

---

## 2. Results

| | 320 px | 640 px | 1280 px |
|---|--------|--------|---------|
| **Batch size** | 32 | 8 | 2 |
| **Epochs trained** | 100 | 58 (best @ 38) | 46 (best @ 26) |
| **mAP50** | 0.2899 | 0.3514 | **0.3707** |
| **mAP50-95** | 0.0806 | 0.1034 | **0.1169** |
| **Precision** | **0.4670** | 0.4375 | 0.4237 |
| **Recall** | 0.3246 | 0.3887 | **0.4282** |
| **Inference speed** | **1.3 ms/img** | 5.0 ms/img | 22.0 ms/img |
| **Training time** | ~9 min | ~20 min | ~68 min |
| **Peak VRAM** | ~7.0 GB | ~8.2 GB | ~6.7 GB |
| **MLflow run** | `85a43fea` | `304f6119` | `41514331` |

---

## 3. Analysis

### 3.1 Accuracy vs. Image Size

- **mAP50 improves monotonically** with image size: 320 → 640 (+21.2%) → 1280 (+5.5%).
- **mAP50-95 follows the same trend**: 320 → 640 (+28.3%) → 1280 (+13.0%).
- The largest gain comes from 320 → 640; diminishing returns from 640 → 1280.

### 3.2 Precision–Recall Trade-off

- **Precision decreases** slightly as image size grows (0.467 → 0.424), meaning smaller images produce fewer false positives.
- **Recall increases** significantly with image size (0.325 → 0.428), meaning larger images detect more true positives.
- This is expected: larger images preserve small-object detail, improving recall at the cost of slightly more false positives.

### 3.3 Speed

- **Inference latency** scales roughly quadratically with image dimension: 320→640 is ~4×, 640→1280 is ~4.4×.
- **320 px is 17× faster** than 1280 px for inference (1.3 ms vs 22 ms).
- **Training time** also scales roughly with total pixel count per batch: 320 px (batch 32) finishes in 9 min vs 1280 px (batch 2) in 68 min.

### 3.4 VRAM

- No OOM at 1280 px with batch=2 — peak VRAM was only 6.7 GB (below the 8 GB budget).
- 640 px with batch=8 used the most VRAM (8.2 GB) because the larger batch size dominates memory more than image size alone.

### 3.5 Early Stopping

| Image size | Stopped epoch | Best epoch |
|------------|---------------|------------|
| 320 px | 100 (full run) | ~80 |
| 640 px | 58 | 38 |
| 1280 px | 46 | 26 |

Larger images converge faster (fewer epochs to best score) but each epoch takes longer.

---

## 4. Recommendation

| Priority | Image Size | Rationale |
|----------|-----------|-----------|
| **Best accuracy** | 1280 px | Highest mAP50 (0.371) and recall (0.428), acceptable speed (~22 ms inference) |
| **Best balance** | 640 px | Strong mAP50 (0.351) with 4× faster inference than 1280 px; good default |
| **Fastest inference** | 320 px | 1.3 ms inference, but significant accuracy loss (−17% mAP50 vs 640) |

**For production deployment:** 640 px offers the best speed/accuracy trade-off.  
**For maximum recall on small archaeological features:** 1280 px is preferred.

---

## 5. MLflow Tags

All runs are tagged under MLflow experiment `yolo26m`:

| Tag | Value |
|-----|-------|
| `experiment_type` | `ablation` |
| `ablation.study` | `imgsz` |
| `model_family` | `yolo` |
| `task` | `detection` |
| `image_size` | `320` / `640` / `1280` |
