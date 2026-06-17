"""Evaluate trained YOLO26m on archaeological hole detection test set."""

import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO26m hole detection model"
    )
    parser.add_argument(
        "--model",
        default="runs/train/yolo26n-hole/weights/best.pt",
        help="Path to trained model weights (default: runs/train/yolo26n-hole/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Dataset config YAML (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--output",
        default="runs/eval",
        help="Output directory for evaluation results (default: runs/eval)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate on: test, val (default: test)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)
    output_dir = Path(args.output)
    split = args.split

    # --- Checks ---
    if not model_path.exists():
        sys.exit(
            f"ERROR: Model not found at {model_path}\n"
            f"       Train a model first with: python scripts/train.py"
        )

    if not data_yaml.exists():
        sys.exit(f"ERROR: Dataset config not found at {data_yaml}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        device = "cpu"
        print("WARNING: CUDA not available — running on CPU")

    # --- Load model ---
    print(f"\nLoading model from {model_path}...")
    model = YOLO(str(model_path))
    print(f"Model loaded — task: {model.task}")

    # --- Evaluate ---
    print(f"\nEvaluating on split='{split}'...")
    print(f"  Data:   {data_yaml}")
    print(f"  Device: {device}\n")

    results = model.val(
        data=str(data_yaml),
        split=split,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    # --- Extract metrics ---
    metrics = {}

    # Primary source: results_dict (present in DetMetrics returned by model.val())
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        metrics["mAP50"] = rd.get("metrics/mAP50(B)")
        metrics["mAP50-95"] = rd.get("metrics/mAP50-95(B)")
        metrics["precision"] = rd.get("metrics/precision(B)")
        metrics["recall"] = rd.get("metrics/recall(B)")
    else:
        # Fallback: stats() method
        try:
            stats = results.stats()
            if isinstance(stats, dict):
                metrics["mAP50"] = stats.get("metrics/mAP50(B)") or stats.get("map50")
                metrics["mAP50-95"] = stats.get("metrics/mAP50-95(B)") or stats.get("map")
                metrics["precision"] = stats.get("metrics/precision(B)") or stats.get("precision")
                metrics["recall"] = stats.get("metrics/recall(B)") or stats.get("recall")
        except Exception:
            pass

    # Compute F1
    p = metrics.get("precision")
    r = metrics.get("recall")
    if p is not None and r is not None and (p + r) > 0:
        metrics["f1_score"] = round(2 * p * r / (p + r), 6)
    else:
        metrics["f1_score"] = "N/A"

    # --- Per-class metrics ---
    per_class = {}
    if hasattr(results, "ap_class_index") and hasattr(results, "maps"):
        maps = results.maps
        for cls_idx, cls_id in enumerate(results.ap_class_index):
            class_name = results.names.get(cls_id, str(cls_id)) if hasattr(results, "names") else str(cls_id)
            per_class[class_name] = {
                "mAP50": float(maps[cls_idx]) if cls_idx < len(maps) and maps[cls_idx] is not None else None,
            }

    # --- Save results.json ---
    results_dict = {
        "model": str(model_path),
        "data": str(data_yaml),
        "split": split,
        "metrics": {k: round(v, 6) if v is not None else None for k, v in metrics.items()},
        "per_class": per_class,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nMetrics saved to {results_json_path}")

    # --- Confusion matrix ---
    cm_path = None
    if hasattr(results, "save_dir"):
        candidate = Path(results.save_dir) / "confusion_matrix.png"
        if candidate.exists():
            cm_path = candidate

    # If confusion matrix not found in results, check output_dir
    if cm_path is None:
        candidate = output_dir / "confusion_matrix.png"
        if candidate.exists():
            cm_path = candidate

    # If still not found, try to generate one via model.val's built-in confusion matrix
    if cm_path is None and hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            cm = results.confusion_matrix
            plt.figure(figsize=(8, 7))
            sns.heatmap(
                cm.matrix if hasattr(cm, "matrix") else np.array(cm),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=results.names if hasattr(results, "names") else ["hole"],
                yticklabels=results.names if hasattr(results, "names") else ["hole"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix — YOLO26m Hole Detection")
            cm_out = output_dir / "confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_out, dpi=150)
            plt.close()
            cm_path = cm_out
            print(f"Confusion matrix saved to {cm_path}")
        except ImportError:
            print("WARNING: matplotlib/seaborn not available — skipping confusion matrix plot")
    else:
        print(f"Confusion matrix available at {cm_path}")

    if cm_path is not None:
        if cm_path != output_dir / "confusion_matrix.png":
            # Copy/rename to output dir
            import shutil
            dest = output_dir / "confusion_matrix.png"
            shutil.copy2(str(cm_path), str(dest))
            print(f"Confusion matrix copied to {dest}")

    # --- Print summary ---
    sep = "─" * 45
    print()
    print("=" * 45)
    print("  YOLO26n Hole Detection — Test Set Evaluation")
    print("=" * 45)
    print(f"{'Metric':<20} {'Value':<10}")
    print(sep)
    print(f"{'mAP50':<20} {metrics.get('mAP50', 'N/A'):<10}")
    print(f"{'mAP50-95':<20} {metrics.get('mAP50-95', 'N/A'):<10}")
    print(f"{'Precision':<20} {metrics.get('precision', 'N/A'):<10}")
    print(f"{'Recall':<20} {metrics.get('recall', 'N/A'):<10}")
    print(f"{'F1-Score':<20} {metrics.get('f1_score', 'N/A'):<10}")
    print(sep)

    # Per-class summary if multiple classes
    if len(per_class) > 1:
        print("\nPer-Class Metrics:")
        for cls_name, cls_metrics in per_class.items():
            print(f"  {cls_name}: {cls_metrics}")


if __name__ == "__main__":
    main()
