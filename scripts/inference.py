"""Batch inference pipeline for YOLO26m hole detection on satellite imagery."""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26m hole detection inference on satellite imagery"
    )
    parser.add_argument(
        "--model",
        default="runs/train/yolo26m-hole/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image file, directory of images, or glob pattern",
    )
    parser.add_argument(
        "--output",
        default="runs/inference",
        help="Output directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--save-img",
        action="store_true",
        help="Save annotated images",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format .txt labels",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save detection CSV (default: True if no other save flag)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="GPU device ID, or 'cpu' for CPU (default: cpu since CUDA unavailable on Blackwell)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save model visualization features",
    )
    return parser.parse_args()


def collect_images(source: str) -> list[Path]:
    """Resolve --source into a list of image paths."""
    src = Path(source)

    if src.is_file():
        return [src]

    if src.is_dir():
        # Recursively gather common image extensions
        exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
        images: list[Path] = []
        for ext in exts:
            images.extend(src.rglob(ext))
        # Deduplicate and sort
        return sorted(set(images))

    # Treat as glob pattern
    import glob as glob_mod

    matched = sorted(glob_mod.glob(source, recursive=True))
    paths = [Path(p) for p in matched if Path(p).suffix.lower() in (".png", ".jpg", ".jpeg")]
    return paths


def main() -> None:
    args = parse_args()

    # --- Resolve paths ---
    model_path = Path(args.model)
    output_dir = Path(args.output)

    # --- Check model ---
    if not model_path.exists():
        alt_dir = Path("runs/train")
        available = sorted(alt_dir.glob("*/weights/best.pt")) if alt_dir.exists() else []
        print(f"ERROR: Model not found at {model_path.resolve()}")
        if available:
            print("Available checkpoints:")
            for ckpt in available:
                print(f"  {ckpt}")
        else:
            print("No checkpoints found under runs/train/.")
        sys.exit(1)

    # --- Collect images ---
    image_paths = collect_images(args.source)
    if not image_paths:
        print(f"WARNING: No images found matching --source '{args.source}'")
        sys.exit(0)

    print(f"\nFound {len(image_paths)} image(s) from: {args.source}\n")

    # --- Default save-csv if no other save flag is set ---
    if not args.save_img and not args.save_txt and not args.save_csv:
        args.save_csv = True

    # --- Create output directories ---
    if args.save_img:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from {model_path} ...")
    model = YOLO(str(model_path))
    print(f"Model loaded — task: {model.task}\n")

    # --- Run inference ---
    all_detections: list[dict] = []
    processed = 0
    total_detections = 0

    for img_path in image_paths:
        try:
            # Verify image is loadable before passing to model
            with Image.open(img_path) as _img:
                _img.load()
        except Exception:
            print(f"  SKIP (unreadable): {img_path.name}")
            continue

        results = model.predict(
            str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            visualize=args.visualize,
            save=False,  # we save ourselves
        )

        result = results[0]
        n_detections = len(result.boxes)
        processed += 1
        total_detections += n_detections
        print(f"  {img_path.name}: {n_detections} detections")

        # --- Save annotated image ---
        if args.save_img:
            annotated = result.plot()
            save_path = output_dir / "images" / img_path.name
            Image.fromarray(annotated[..., ::-1]).save(save_path)  # BGR -> RGB

        # --- Save YOLO labels ---
        if args.save_txt and result.boxes is not None:
            txt_path = output_dir / "labels" / f"{img_path.stem}.txt"
            with open(txt_path, "w") as f:
                for box, cls_id, conf in zip(
                    result.boxes.xywhn, result.boxes.cls, result.boxes.conf
                ):
                    f.write(f"{int(cls_id)} {box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f} {conf:.6f}\n")

        # --- Collect detections for CSV ---
        if args.save_csv and result.boxes is not None:
            names = model.names
            for box, cls_id, conf in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            ):
                all_detections.append({
                    "filename": img_path.name,
                    "class_id": int(cls_id),
                    "class_name": names[int(cls_id)],
                    "confidence": float(conf),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                })

        # Save model visualizations (ultralytics saves these automatically)
        if args.visualize:
            # Ultralytics saves visualizations to the current working directory
            # under the runs/ directory by default; we just log that it was enabled.
            pass

    # --- Save CSV ---
    if args.save_csv and all_detections:
        csv_path = output_dir / "detections.csv"
        fieldnames = ["filename", "class_id", "class_name", "confidence",
                      "x1", "y1", "x2", "y2"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detections)
        print(f"\n  CSV saved: {csv_path}")

    # --- Summary ---
    mean_conf = (
        sum(d["confidence"] for d in all_detections) / len(all_detections)
        if all_detections
        else 0.0
    )

    print(f"\n{'=' * 40}")
    print("Inference Complete")
    print(f"{'=' * 40}")
    print(f"Source:        {args.source}")
    print(f"Images:        {processed} processed")
    print(f"Total Dets:    {total_detections}")
    print(f"Mean Conf:     {mean_conf:.3f}")
    print(f"Output:        {output_dir.resolve()}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
