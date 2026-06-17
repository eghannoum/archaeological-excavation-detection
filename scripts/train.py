"""Train YOLO26 on the archaeological hole detection dataset.

NOTE: RTX 5070 (Blackwell sm_120) not yet supported by PyTorch CUDA builds.
Training uses CPU with YOLO26n (nano model) for practical training times.
For GPU acceleration, use PyTorch ≥ 2.8 with CUDA ≥ 12.8 once available.
"""

from pathlib import Path
import sys
import os
import time
import torch
from ultralytics import YOLO

DATA_YAML = Path("dataset/data.yaml")
PROJECT = "runs/train"
NAME = "yolo26n-hole"
MODEL_NAME = "yolo26n.pt"
EPOCHS = 50
PATIENCE = 15


def main() -> None:
    # --- Checks ---
    if not DATA_YAML.exists():
        sys.exit(f"ERROR: {DATA_YAML} not found — run from project root")

    # Detect device
    DEVICE_STR = "cpu"
    BATCH = 8
    IMGSZ = 320  # smaller for CPU training speed
    WORKERS = min(4, os.cpu_count() or 4)

    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {gpu} ({mem:.1f} GB)")
            test = torch.rand(2, 2).cuda()
            test = test @ test
            DEVICE_STR = "cuda:0"
            BATCH = 16
            IMGSZ = 640
            print(f"CUDA works — using GPU (batch={BATCH})")
        except Exception as e:
            print(f"CUDA available but GPU compute failed: {e}")
            print("Falling back to CPU training")

    if DEVICE_STR == "cpu":
        print("Training on CPU (multi-core)")
        print(f"  Model:    {MODEL_NAME} (nano — optimized for CPU)")
        print(f"  CPU cores: {os.cpu_count()}")

    # --- Load model ---
    print(f"\nLoading YOLO26m pretrained weights...")
    model = YOLO(MODEL_NAME)
    print(f"Model loaded — task: {model.task}")

    # --- Train ---
    print(f"\nStarting training...")
    print(f"  Data:     {DATA_YAML}")
    print(f"  Imgsz:    {IMGSZ}")
    print(f"  Batch:    {BATCH}")
    print(f"  Epochs:   {EPOCHS} (patience={PATIENCE})")
    print(f"  Device:   {DEVICE_STR}")
    print(f"  Output:   {PROJECT}/{NAME}/\n")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE_STR,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        warmup_epochs=3,
        save=True,
        save_period=10,
        val=True,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")
    print(f"Results:    {PROJECT}/{NAME}/")
    print(f"{'='*60}")

    # Print final metrics
    if hasattr(results, "metrics"):
        m = results.metrics
        print(f"\nFinal Validation Metrics:")
        print(f"  mAP50:    {m.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {m.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision: {m.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall:   {m.get('metrics/recall(B)', 'N/A')}")


if __name__ == "__main__":
    main()
