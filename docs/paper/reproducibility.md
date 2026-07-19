# Reproducibility Instructions

Step-by-step guide to reproducing all results reported in the paper.

---

## Prerequisites

**Hardware:**
- CPU: Intel processor (all training and inference ran on CPU; no GPU acceleration was used despite RTX 5070 being present in the system)
- RAM: 16GB minimum recommended

**Software:**
- Python 3.12+
- PyTorch 2.11+ (CPU or CUDA)
- Ultralytics 8.4.72

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages used by evaluation scripts (install as needed):
```bash
pip install scipy scikit-learn matplotlib seaborn pandas
```

---

## 1. Dataset Preparation

The dataset consists of 432 satellite image tiles (640x640 pixels) with bounding box annotations for the single class "hole."

```bash
# Extract the dataset
cd data/
# data.zip contains input_images/ and output_annotations_notebook/
unzip data.zip
```

**Data split** (parent-scene level, no spatial leakage):

| Split | Tiles | Purpose |
|-------|------:|---------|
| Train | 352 | Model training |
| Validation | 400 | Hyperparameter tuning, confidence threshold selection |
| Test | 40 | Final held-out evaluation |

Splits were created at the parent-scene level: all tiles from the same satellite pass are assigned to the same split to prevent spatial autocorrelation from inflating performance estimates.

---

## 2. Training All Models

### YOLO Models (7 variants)

All YOLO models train via `scripts/train.py` with Hydra config management:

```bash
# Train each YOLO variant
for model in yolo26n yolo26s yolo26m yolo26l yolo26x yolov8m yolo11m; do
  python scripts/train.py model=${model}
done
```

Default training configuration (from `configs/default.yaml`):
- Epochs: 100
- Image size: 640x640
- Optimizer: AdamW
- Augmentation: Ultralytics default (Mosaic, MixUp, HSV)
- Confidence threshold: Optimized per model on validation set

### Faster R-CNN

```bash
python scripts/train_faster_rcnn.py
```

Uses ResNet-50-FPN backbone, trained for 100 epochs.

### DETR

```bash
python scripts/train_detr.py
```

Uses ResNet-50 backbone with 6-layer transformer decoder and 100 object queries, trained for 100 epochs.

### 3-Fold Cross-Validation

```bash
python scripts/train_cv.py
```

Trains all models across 3 folds at the parent-scene level. Produces 27 model weight files (9 models x 3 folds) saved under `experiments/`.

---

## 3. Cross-Validation Evaluation

After CV training completes:

```bash
# Statistical significance testing with bootstrap
python scripts/eval_statistical_significance.py
```

This script:
- Loads CV predictions from all 3 folds for each model
- Computes paired bootstrap tests (1,000 iterations) for all 36 model pairs
- Applies Bonferroni correction for multiple comparisons
- Assigns significance groupings (a, b, c) via hierarchical clustering
- Outputs `docs/significance/significance_report.json`

---

## 4. Test Set Evaluation

```bash
# Evaluate all 9 models on held-out test set
python scripts/eval_test_set.py
```

This script:
- Loads the 40-image test set
- Runs inference with each model at its optimized confidence threshold
- Computes mAP50, mAP50-95, Precision, Recall, F1, and inference time
- Outputs `docs/test-set/test_results.json`

---

## 5. Error Analysis

```bash
python scripts/eval_error_analysis.py
```

Analyzes detection errors by category (background FP, missed, localization) and by object size (small < 1%, medium 1-10%, large > 10% of image area). Output: `docs/error-analysis/key_findings.md`.

---

## 6. Additional Evaluations

```bash
# Precision-recall curves
python scripts/eval_pr_curves.py

# Confidence calibration analysis
python scripts/eval_calibration.py
```

---

## 7. Ablation Studies

All ablations use YOLO26-M as the base model, varying one factor at a time.

### Image Size Ablation

```bash
# Trains YOLO26-M at 320, 640 (baseline), and 1280 pixels
python scripts/train.py model=yolo26m data.image_size=320
python scripts/train.py model=yolo26m data.image_size=640
python scripts/train.py model=yolo26m data.image_size=1280
```

### Optimizer Ablation

```bash
# AdamW (baseline) vs SGD
python scripts/train.py model=yolo26m training.optimizer=adamw
python scripts/train.py model=yolo26m training.optimizer=sgd
```

### Augmentation Ablation

```bash
# Ultralytics default (baseline) vs Light vs Heavy
python scripts/train.py model=yolo26m augmentation=ultralytics_base
python scripts/train.py model=yolo26m augmentation=light
python scripts/train.py model=yolo26m augmentation=heavy
```

---

## 8. Generating Tables and Figures

```bash
# Generate all metrics tables (docs/tables/)
python scripts/generate_metrics_tables.py

# Generate all ablation tables (docs/tables/ablation/)
python scripts/generate_ablation_tables.py

# Generate all publication figures (docs/figures/)
python scripts/generate_publication_figures.py
```

---

## 9. File Structure After Full Reproduction

```
docs/
  paper/
    paper.md
    reproducibility.md
  tables/
    main_results.md
    cv_results.md
    ablation/
      augmentation.md
      image_size.md
      optimizer.md
  figures/
    fig_mAP50_comparison.png
    fig_speed_accuracy.png
    fig_ablation_summary.png
    fig_error_breakdown.png
    fig_calibration.png
    fig_pr_curves.png
  error-analysis/
    key_findings.md
  significance/
    significance_report.json
  test-set/
    test_results.json

models/
  yolo26n.pt
  yolo26s.pt
  yolo26m.pt
  yolo26l.pt
  yolo26x.pt
  yolov8m.pt
  yolo11m.pt

experiments/
  yolo26n/   (3 fold checkpoints)
  yolo26s/
  yolo26m/
  yolo26l/
  yolo26x/
  yolov8m/
  yolo11m/
  faster_rcnn/
  detr/
```

---

## Reproducibility Checklist

| Criterion | Status | Details |
|-----------|:------:|---------|
| Dataset splits provided | Done | Train 352 / Val 40 / Test 40 at parent-scene level |
| 3-fold cross-validation | Done | Parent-scene level splits prevent spatial leakage |
| All hyperparameters documented | Done | 100 epochs, default augmentation, per-model conf thresholds |
| Statistical significance tests | Done | Paired bootstrap, 1,000 iterations, Bonferroni correction |
| Model weights saved | Done | 7 YOLO weights in `models/`, plus CV checkpoints in `experiments/` |
| Evaluation scripts provided | Done | 8 scripts in `scripts/` covering all evaluation stages |
| Bootstrap confidence intervals | Done | 95% CIs for all CV means in `significance_report.json` |
| Significance groupings | Done | Pairwise p-values and group assignments reported |
| Ablation studies | Done | Image size, optimizer, augmentation with single-factor variation |
| Error analysis | Done | Size-based breakdown and error category classification |

---

## Notes

- All inference times are CPU-only. GPU inference would be faster but the relative ordering of models remains the same.
- DETR requires longer training convergence; 100 epochs may be insufficient. The poor performance reported (mAP50 = 0.006 test / 0.066 CV) likely reflects under-training rather than inherent architectural limitation.
- Confidence thresholds were optimized on the validation set: YOLO26-X (0.25), YOLO26-M (0.15), YOLO26-S (0.20), YOLO11-M (0.25), YOLOv8-M (0.30), Faster R-CNN (0.70), YOLO26-N (0.15), YOLO26-L (0.25), DETR (0.90).
- The config system uses Hydra with YAML files under `configs/`. Override any parameter via CLI, e.g., `python scripts/train.py model=yolo26x training.epochs=200`.
