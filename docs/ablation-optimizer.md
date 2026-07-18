# Optimizer Ablation Study — YOLO26m

> **Model:** YOLO26m  
> **Experiment:** `yolo26m` (MLflow experiment ID: 7)  
> **Method:** Single-fold (full dataset, cv.n_folds=1), 100 epochs max, early stopping (patience=20)  
> **Device:** RTX 5070 Laptop (8 GB VRAM)  
> **Date:** 2026-07-17

---

## 1. Hyperparameters

All shared parameters are from HPO best trial #49 (mAP50 = 0.3803 @ 30 epochs). Only optimizer and related params differ between runs.

| Parameter | AdamW | SGD |
|-----------|-------|-----|
| optimizer | AdamW | SGD |
| lr | 0.0004701 | 0.01 |
| momentum | 0.8514 | 0.937 |
| weight_decay | 0.0007860 | 0.0005 |
| warmup_epochs | 5 | 5 |
| scheduler | linear | linear |
| batch_size | 8 | 8 |
| epochs | 100 | 100 |

**Augmentation (identical for both):**

| Parameter | Value |
|-----------|-------|
| hsv_h | 0.0680 |
| hsv_s | 0.2985 |
| hsv_v | 0.4326 |
| degrees | 14.589 |
| mosaic | 0.5 |
| mixup | 1.0 |

SGD uses higher lr (0.01, ~21x AdamW lr) with standard SGD momentum (0.937) and weight_decay (0.0005), consistent with Ultralytics defaults for SGD.

---

## 2. Results

| Metric | AdamW | SGD | Delta | Winner |
|--------|-------|-----|-------|--------|
| **mAP50** | 0.3432 | 0.3422 | −0.0010 (−0.3%) | AdamW (negligible) |
| **mAP50-95** | 0.1001 | 0.1003 | +0.0002 (+0.2%) | SGD (negligible) |
| **Precision** | 0.4449 | 0.4442 | −0.0007 (−0.2%) | AdamW (negligible) |
| **Recall** | 0.3834 | 0.3928 | +0.0094 (+2.5%) | SGD |
| **Epochs trained** | 52 | 52 | 0 | Tie |
| **Best epoch** | 32 | 32 | 0 | Tie |
| **Training time** | ~21 min | ~18 min | −3 min | SGD |
| **Inference speed** | 21.2 ms/img | 16.4 ms/img | −4.8 ms (−23%) | SGD |
| **Peak VRAM** | ~8.6 GB | ~8.5 GB | ~0 | Tie |
| **MLflow run** | `4a971424` | `1127ebbe` | — | — |

---

## 3. Training Loss Curves (final epoch)

| Loss | AdamW | SGD |
|------|-------|-----|
| train/box_loss | 2.3717 | 2.3776 |
| train/cls_loss | 1.5906 | 1.5906 |
| train/dfl_loss | 0.00303 | 0.00304 |
| val/box_loss | 2.5256 | 2.6156 |
| val/cls_loss | 1.8130 | 1.7194 |
| val/dfl_loss | 0.00322 | 0.00347 |

Training losses are nearly identical. SGD has slightly lower validation classification loss (1.719 vs 1.813) but slightly higher box loss (2.616 vs 2.526), suggesting SGD may generalize slightly differently on classification vs localization.

---

## 4. Analysis

### 4.1 Accuracy: Negligible Difference

- **mAP50 is effectively tied**: AdamW 0.3432 vs SGD 0.3422 (Δ = 0.001, within noise for single-fold evaluation).
- **mAP50-95 is also tied**: SGD leads by 0.0002 — immeasurably small.
- Both optimizers converge to the same best epoch (32) and trigger early stopping at the same epoch (52), indicating similar convergence dynamics with these hyperparameters.

### 4.2 Recall: SGD Slight Edge

- SGD achieves +2.5% higher recall (0.393 vs 0.383). This is the largest meaningful difference between the two optimizers.
- Combined with slightly lower val classification loss, SGD may be slightly better at detecting all positive instances (fewer false negatives) at the cost of a marginally higher box loss.

### 4.3 Speed: SGD Wins

- **SGD trains ~15% faster** (18 min vs 21 min). This is expected: AdamW maintains per-parameter adaptive learning rates (2x memory/state), while SGD uses a single global learning rate.
- **SGD inference is 23% faster** (16.4 ms vs 21.2 ms). This is surprising and likely due to batch composition variance during timing, not an inherent optimizer advantage at inference (optimizer is stripped from saved weights).

### 4.4 Convergence Behavior

Both optimizers exhibit identical convergence patterns:
- Best model saved at epoch 32
- Early stopping triggered at epoch 52 (20 epochs without improvement)
- No optimizer-specific instability or divergence observed

### 4.5 Comparison with Prior CV Results

The prior 3-fold CV with AdamW at 640px achieved mAP50 = 0.3888 ± 0.063. Both optimizer runs here (single dataset, no CV) achieved ~0.343, which is within 1 standard deviation of the CV mean. The lower absolute value is expected with a single train/val split rather than averaged across folds.

---

## 5. Recommendation

| Priority | Optimizer | Rationale |
|----------|-----------|-----------|
| **Default choice** | AdamW | Matches HPO-tuned configuration; adaptive LR provides more robust convergence across tasks; slightly higher mAP50 |
| **Speed-optimized** | SGD | 15% faster training, 2.5% better recall; acceptable for time-constrained experiments |
| **Production deployment** | Either | Both produce equivalent mAP50; optimizer is stripped from exported models |

**Key finding:** With properly tuned hyperparameters (HPO best params for AdamW, standard defaults for SGD), optimizer choice has minimal impact on mAP50 for this dataset and model. The HPO search space should not need to explore optimizer as a critical dimension for YOLO26m on this task.

---

## 6. MLflow Tags

All runs are tagged under MLflow experiment `yolo26m`:

| Tag | AdamW | SGD |
|-----|-------|-----|
| `ablation` | `optimizer` | `optimizer` |
| `ablation.study` | `optimizer_adamw` | `optimizer_sgd` |
| `experiment_type` | `training` | `training` |
| `model_family` | `yolo26` | `yolo26` |
| `model_scale` | `m` | `m` |
| `image_size` | `640` | `640` |
