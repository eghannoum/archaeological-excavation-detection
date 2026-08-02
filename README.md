# Automated Detection of Unauthorized Archaeological Excavations

Detection of unauthorized archaeological excavation holes ("looting holes") from satellite and aerial imagery. The project benchmarks nine object-detection architectures (YOLO26 N/S/M/L/X, YOLOv8-M, YOLO11-M, Faster R-CNN, and DETR) on a curated dataset of 432 satellite tiles, using a Hydra-based pipeline with MLflow tracking and 3-fold parent-scene cross-validation.

## Features

- Hydra configuration: every training, data, augmentation, and experiment setting is a YAML override.
- MLflow experiment tracking: per-run parameters, per-epoch metrics, and nested HPO runs logged automatically.
- 3-fold parent-scene cross-validation: tiles are grouped by satellite pass to prevent spatial leakage (`scripts/train_cv.py`).
- 9-architecture benchmark with Optuna hyperparameter optimization and ablation studies (image size, optimizer, augmentation).
- Batch inference CLI (images, CSV, YOLO labels) plus a Gradio dashboard.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/eghannoum/archaeological-excavation-detection.git
python -m venv .venv
pip install -r requirements.txt
```

- GPU (CUDA): `pip install -r requirements-gpu.txt` first, then `requirements.txt`.
- Dev tooling: `pip install -r requirements-dev.txt`.

## Quickstart

```bash
python scripts/train.py experiment=yolo26n --info
python scripts/train.py experiment=yolo26n training.epochs=1
python scripts/inference.py --model runs/yolo26n/weights/best.pt --source data/sample/ --save-img --save-csv
```

Outputs land in `runs/inference/`.

## Training

```bash
python scripts/train.py experiment=yolo26m
```

- Cross-validation: `python scripts/train_cv.py experiment=yolo26m`
- Hyperparameter optimization: `python scripts/hpo.py` (Windows: `scripts/run_hpo.ps1 -NTrials 50`)
- Faster R-CNN: `python scripts/train_faster_rcnn.py`
- DETR: `python scripts/train_detr.py`

## Evaluation

```bash
python scripts/evaluate.py --model models/yolo26m.pt --data dataset/data.yaml --split test
```

Writes `results.json` plus a confusion matrix; analysis scripts write to gitignored `outputs/`.

## Inference & Dashboard

```bash
python scripts/inference.py --model models/yolo26m.pt --source data/sample/ --save-img --save-csv
python scripts/dashboard.py
```

The Gradio dashboard serves on `http://localhost:7860` by default.

## Results

Performance on the held-out test set (40 tiles at 640px input; times are CPU inference per image).

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

Key findings: all YOLO variants and Faster R-CNN form a statistically indistinguishable top tier; YOLO26-X is best by mAP50 while YOLO11-M reaches the best F1; DETR performs near chance and is not recommended for this task.

## Project structure

```
configs/   Hydra configs (experiment, model, training, data, augmentation, cv, hpo, ablation)
scripts/   CLI entrypoints (train, train_cv, hpo, evaluate, inference, dashboard, analysis)
tests/     pytest suite
data/      raw imagery + COCO annotations (not distributed; data/sample/ for quickstart)
```

Generated artifacts (`dataset/`, `models/`, `runs/`, `experiments/`, `mlruns/`, `outputs/`) are gitignored and regenerable.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Built on [Ultralytics](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [MLflow](https://mlflow.org/), [Optuna](https://optuna.org/), [Albumentations](https://albumentations.ai/), and [Hydra](https://hydra.cc/).
