# Automated Detection of Unauthorized Archaeological Excavations

Detection of unauthorized archaeological excavation holes ("looting holes") from satellite and aerial imagery. The project benchmarks nine object detection architectures — YOLO26 (N/S/M/L/X), YOLOv8-M, YOLO11-M, Faster R-CNN, and DETR — on a curated dataset of 432 satellite image tiles, using a Hydra-based training pipeline with MLflow tracking, 3-fold parent-scene cross-validation, Optuna hyperparameter optimization, and ablation studies.

## Features

- **Hydra configuration** — every training, data, augmentation, and experiment setting is a YAML override; no hardcoded experiment variants.
- **MLflow experiment tracking** — per-run parameters, per-epoch metrics, and nested HPO runs logged automatically.
- **3-fold parent-scene cross-validation** — splits group tiles by satellite pass to prevent spatial leakage (`scripts/train_cv.py`).
- **Hyperparameter optimization** — 12-parameter Optuna study (TPE sampler + median pruner) targeting `val/mAP50` (`scripts/hpo.py`).
- **Ablation studies** — image size (320/640/1280), optimizer (AdamW/SGD), and augmentation (none/light/heavy) vs. the YOLO26-M baseline.
- **9-architecture benchmark** — YOLO26 N/S/M/L/X, YOLOv8-M, YOLO11-M, Faster R-CNN, and DETR with paired bootstrap significance testing.
- **Reproducible dataset pipeline** — deterministic COCO-to-YOLO conversion with parent-scene splitting (`scripts/coco_to_yolo.py`).
- **Batch inference & dashboard** — CLI inference to images/CSV/YOLO labels plus a Gradio web dashboard.

## Installation

Requires **Python 3.10+** (developed on 3.12). CPU-only inference is supported; GPU acceleration is optional.

```bash
# Clone the repository
git clone https://github.com/eghannoum/archaeological-excavation-detection.git
cd STI-Unauthorized-Archaeological-Excavations

# (Recommended) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate

# Install core dependencies (CPU-portable torch)
pip install -r requirements.txt
```

**GPU (CUDA) installation** — the default `requirements.txt` installs the portable PyTorch wheel. For a CUDA 12.8-pinned build, install `requirements-gpu.txt` first, then the core requirements:

```bash
pip install -r requirements-gpu.txt
pip install -r requirements.txt
```

**Developer tooling** (linting, testing, pre-commit):

```bash
pip install -r requirements-dev.txt
```

## Quickstart

The fastest path from a fresh clone to running detections. This assumes you have a small batch of imagery in `data/sample/` — any folder of `.jpg`/`.png` tiles works as `--source`. Neither the full dataset nor trained weights ship with this repository (see [Dataset](#dataset) and [Model weights](#model-weights)); the quickstart trains a tiny model and runs it on your sample images.

```bash
# 1. Inspect the resolved configuration (dry run, no training)
python scripts/train.py experiment=yolo26n --info

# 2. Smoke-test training (1 epoch) — checkpoint lands in runs/yolo26n/weights/
python scripts/train.py experiment=yolo26n training.epochs=1

# 3. Run inference on sample imagery
python scripts/inference.py --model runs/yolo26n/weights/best.pt \
    --source data/sample/ --save-img --save-csv
```

Outputs: annotated images in `runs/inference/images/` and a `runs/inference/detections.csv` with per-box class, confidence, and pixel coordinates.

## Dataset

The dataset comprises **432 satellite image tiles (640x640)** with bounding-box annotations for a single class, **`hole`**, at the parent-scene level:

| Split | Tiles | Purpose |
|-------|------:|---------|
| Train | 352 | Model training |
| Validation | 40 | Hyperparameter tuning, confidence threshold selection |
| Test | 40 | Final held-out evaluation |

All tiles from the same satellite pass are kept in the same split to prevent spatial autocorrelation from inflating performance estimates.

### Raw data layout

The source imagery and annotations live under `data/` (not distributed, see below):

```
data/
├── input_images/                          # 108 raw satellite scene images (JPG)
├── output_annotations_notebook/
│   └── combined_coco_split.json           # COCO-format annotations (432 images, 21,950 boxes)
└── output_splits_notebook/                # 432 quadrant tiles (_tl/_tr/_bl/_br, PNG)
```

### Building the YOLO dataset

The `dataset/` tree (images/labels per split plus `data.yaml`) is **generated and gitignored**. Rebuild it from the raw COCO annotations with `scripts/coco_to_yolo.py`, which splits deterministically at the parent-scene level (seed 42, default 10% validation / 10% test):

```bash
python scripts/coco_to_yolo.py \
    --coco-path data/output_annotations_notebook/combined_coco_split.json \
    --image-dir data/output_splits_notebook \
    --output-dir dataset \
    --include-test-in-yaml
```

Useful flags: `--dry-run` (preview the split without writing), `--validate-only` (check COCO structure and bbox containment), `--val-split`/`--test-split`/`--seed`, `--overwrite` (delete and recreate an existing `--output-dir`).

> **Note:** the full dataset is **not distributed** with this repository. To reproduce the reported results you must supply your own annotated imagery in the layout above. `data/sample/` is the designated quickstart location for a small batch of unlabeled tiles to smoke-test the pipeline.

## Training

### YOLO models (YOLO26 N/S/M/L/X, YOLOv8-M, YOLO11-M)

All YOLO variants train through the Hydra entrypoint:

```bash
python scripts/train.py experiment=yolo26m
```

Available experiment configs (`configs/experiment/`):

| Config | Architecture | Config | Architecture |
|--------|--------------|--------|--------------|
| `yolo26n` | YOLO26-Nano | `yolo26x` | YOLO26-XLarge |
| `yolo26s` | YOLO26-Small | `yolov8` | YOLOv8-M |
| `yolo26m` | YOLO26-Medium | `yolo11` | YOLO11-M |
| `yolo26l` | YOLO26-Large | | |

Defaults: 100 epochs, 640x640 input, AdamW, Ultralytics default augmentation, cosine schedule, fp16, early stopping. Override anything via Hydra CLI:

```bash
# Inspect the resolved config without training
python scripts/train.py experiment=yolo26n --info

# Adjust hyperparameters
python scripts/train.py experiment=yolo26m training.epochs=200 training.batch_size=8
python scripts/train.py experiment=yolo26m data.image_size=1280
python scripts/train.py experiment=yolo26m augmentation=heavy      # none | light | ultralytics_base
python scripts/train.py experiment=yolo26m ablation=optimizer_sgd  # ablation presets
```

Checkpoints and per-epoch metrics are written under `runs/{experiment_name}/` and logged to MLflow.

### Cross-validation

Parent-scene 3-fold CV, with per-fold training launched as subprocesses and metrics aggregated from MLflow:

```bash
python scripts/train_cv.py experiment=yolo26m
python scripts/train_cv.py experiment=yolo26m --folds 3 --epochs 100   # explicit
python scripts/train_cv.py experiment=yolo26m --no-cleanup              # keep fold data
```

### Hyperparameter optimization

Optuna study over 12 hyperparameters (LR, momentum, weight decay, optimizer, warmup, cosine schedule, HSV, rotation, mosaic, mixup), target `val/mAP50`:

```bash
python scripts/hpo.py                          # 50 trials (TPE + median pruner)
python scripts/hpo.py --dry-run                # print the search space and exit
python scripts/hpo.py --n-trials 25            # override trial count
python scripts/hpo.py --n-jobs 2               # parallel trials
python scripts/hpo.py experiment=yolo26n       # different base experiment
```

### Faster R-CNN and DETR

Standalone CV trainers with their own CLI (defaults: 3 folds, 100 epochs):

```bash
python scripts/train_faster_rcnn.py            # ResNet-50-FPN backbone
python scripts/train_faster_rcnn.py --epochs 50 --batch-size 2

python scripts/train_detr.py                   # ResNet-50 + transformer decoder
python scripts/train_detr.py --epochs 50 --folds 2 --no-mlflow
```

## Evaluation

Evaluate a trained checkpoint on the test or validation split:

```bash
python scripts/evaluate.py \
    --model models/yolo26m.pt \
    --data dataset/data.yaml \
    --split test \
    --output runs/eval
```

Writes `results.json` (mAP50, mAP50-95, precision, recall, F1) plus a confusion matrix to the output directory. Paper-specific analyses (PR curves, calibration, error analysis, significance testing) are documented in `docs/paper/reproducibility.md`.

## Inference

```bash
python scripts/inference.py --model <weights.pt> --source <img|dir|glob> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | Path to trained weights (e.g. `models/yolo26m.pt` or `runs/yolo26n/weights/best.pt`) |
| `--source` | required | Single image file, directory of images, or glob pattern |
| `--output` | `runs/inference` | Output directory |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |
| `--save-img` | off | Save annotated images to `<output>/images/` |
| `--save-txt` | off | Save YOLO-format labels to `<output>/labels/` |
| `--save-csv` | on* | Save `detections.csv` (default when no other save flag is set) |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, or a specific GPU id; auto = first available CUDA device, else CPU |

Examples:

```bash
python scripts/inference.py --model models/yolo26m.pt --source data/sample/ --save-img --save-csv
python scripts/inference.py --model models/yolo26n.pt --source "runs/inference/**/*.png" --device 0
```

## Dashboard

A Gradio web app with three tabs (model metrics, interactive inference, batch inference):

```bash
python scripts/dashboard.py
```

Serves on `http://localhost:7860` by default. Override the port with the `GRADIO_PORT` environment variable.

## Model weights

**Weights are not distributed** with this repository — train to reproduce. After training, checkpoints appear under:

- `runs/{experiment_name}/weights/best.pt` — single-run YOLO checkpoints
- `experiments/{model}/` — 3-fold CV checkpoints for all nine models
- `models/` — the seven YOLO weights used for the final benchmark evaluation

## Results

Performance on the held-out test set (40 tiles at 640px input; times are CPU inference per image). Full tables and significance groupings: `docs/tables/main_results.md`.

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Time (ms) | Params (M) | FLOPs (G) |
|-------|------:|---------:|----------:|-------:|----:|----------:|-----------:|----------:|
| YOLO26-X | 0.437 | 0.140 | 0.431 | 0.603 | 0.503 | 370.2 | 58.8 | 193.4 |
| YOLO26-M | 0.406 | 0.130 | 0.410 | 0.459 | 0.433 | 152.2 | 42.2 | 65.7 |
| YOLO26-S | 0.376 | 0.123 | 0.486 | 0.521 | 0.503 | 71.2 | 9.9 | 24.6 |
| YOLO11-M | 0.369 | 0.124 | 0.487 | 0.529 | 0.507 | 160.0 | 38.8 | 67.6 |
| YOLOv8-M | 0.368 | 0.116 | 0.406 | 0.513 | 0.453 | 151.6 | 49.7 | 78.7 |
| Faster R-CNN | 0.354 | 0.101 | 0.441 | 0.471 | 0.456 | 46.0 | 41.5 | 134.0 |
| YOLO26-N | 0.329 | 0.111 | 0.295 | 0.490 | 0.368 | 41.1 | 5.3 | 8.7 |
| YOLO26-L | 0.307 | 0.094 | 0.290 | 0.458 | 0.355 | 189.1 | 50.7 | 86.1 |
| DETR | 0.006 | 0.001 | 0.011 | 0.016 | 0.013 | 25.0 | 41.3 | 86.0 |

Key findings: all YOLO variants and Faster R-CNN form a statistically indistinguishable top tier; YOLO26-X is best by mAP50 while YOLO11-M reaches the best F1; DETR performs near chance and is not recommended for this task. Small objects (< 1% of image area) drive most missed detections. See `docs/paper/paper.md` for the full analysis.

## Project structure

```
STI-Unauthorized-Archaeological-Excavations/
├── configs/             Hydra config system (default, experiment, model, training,
│                        data, augmentation, cv, hpo, ablation)
├── scripts/             CLI entrypoints: train, train_cv, hpo, evaluate, inference,
│                        dashboard, coco_to_yolo, analysis/visualization utilities
├── paper/               Publication scripts (tables, figures, significance tests)
├── docs/                Paper, reproducibility guide, ablations, analysis reports,
│                        tables, figures, deployment guide
├── data/                Raw source imagery + COCO annotations (not distributed)
├── dataset/             Generated YOLO dataset — reproducible, gitignored
├── models/              Trained YOLO weights (gitignored — train to reproduce)
├── runs/                Ultralytics training/inference outputs (gitignored)
├── experiments/         CV fold checkpoints and results (gitignored)
├── mlruns/              MLflow tracking store (gitignored)
├── pyproject.toml       Package metadata + ruff/black/isort/mypy/pytest config
├── requirements.txt     Core dependencies (CPU-portable)
├── requirements-gpu.txt Optional CUDA-pinned torch build
└── requirements-dev.txt Development tooling
```

## Documentation

- `docs/paper/paper.md` — benchmark paper (9 architectures, methodology, results)
- `docs/paper/reproducibility.md` — step-by-step instructions to reproduce every result
- `docs/tables/` — main results, CV results, and ablation tables (Markdown + LaTeX)
- `docs/ablation-{augmentation,imgsz,optimizer}.md` — ablation study write-ups
- `docs/hpo-analysis.md`, `docs/dataset-analysis.md`, `docs/gpu-benchmark.md` — analysis reports
- `docs/DEPLOYMENT.md` — ONNX/TensorRT export and serving notes

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This work builds on open-source software: [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO), [PyTorch](https://pytorch.org/), [MLflow](https://mlflow.org/), [Optuna](https://optuna.org/), [Albumentations](https://albumentations.ai/), and [Hydra](https://hydra.cc/).
