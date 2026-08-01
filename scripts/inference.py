"""Production-grade inference CLI for YOLO hole detection on satellite imagery.

Usage:
    python scripts/inference.py --source path/to/image.png
    python scripts/inference.py --source path/to/images/
    python scripts/inference.py --source "data/*.jpg"
    python scripts/inference.py --source path/to/video.mp4
    python scripts/inference.py --source webcam            # webcam uses source "0"

Supports single images, folders, glob patterns, videos (.mp4/.avi/.mov/.mkv),
and live webcam input. Images are processed in batches; videos are processed
frame-by-frame with an annotated video written to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger("inference")

CSV_FIELDNAMES = ["filename", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_BATCH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO hole detection inference on satellite imagery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="runs/train/yolo26m-hole/weights/best.pt",
        help=(
            "Path to trained model weights. If missing, searches "
            "runs/train/*/weights/best.pt, runs/*/weights/best.pt and "
            "experiments/*/fold_0_best.pt as fallbacks."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Image file, directory, glob pattern, video file (.mp4/.avi/.mov/.mkv), "
            "or webcam (use '0' or 'webcam')."
        ),
    )
    parser.add_argument("--output", default="runs/inference", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--batch",
        type=_positive_int,
        default=DEFAULT_BATCH,
        help="Images per batched predict call",
    )
    parser.add_argument("--save-img", action="store_true", help="Save annotated images")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format .txt labels")
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save detection CSV (default: True if no other save flag)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device: 'cpu', 'cuda', 'cuda:0', etc. "
            "(default: auto — cuda:0 if available, else cpu)"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging")
    return parser.parse_args()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def collect_images(source: str) -> list[Path]:
    """Resolve ``--source`` into a sorted list of image paths.

    Handles single image files, directories (searched recursively), and glob
    patterns. Returns an empty list (never raises) for missing sources, empty
    directories, or video/webcam inputs.
    """
    src = Path(source)

    if src.is_file():
        return [src] if src.suffix.lower() in IMAGE_EXTS else []

    if src.is_dir():
        images = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return sorted(set(images))

    matched = glob.glob(source, recursive=True)
    images = [Path(p) for p in matched if Path(p).suffix.lower() in IMAGE_EXTS]
    return sorted(set(images))


def resolve_device(requested: str | None) -> str:
    """Resolve the torch device string for inference.

    Returns ``requested`` when provided; otherwise auto-selects ``cuda:0`` if a
    CUDA GPU is available and falls back to ``cpu``.
    """
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def write_detections_csv(detections: list[dict], path: Path) -> None:
    """Write ``detections`` to ``path`` using the standard CSV schema.

    Schema: filename,class_id,class_name,confidence,x1,y1,x2,y2
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(detections)


def _setup_logging(verbose: bool) -> None:
    """Configure the module logger; safe to call multiple times."""
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False


def _resolve_model_path(requested: str) -> Path:
    """Resolve the model checkpoint, searching known checkpoint dirs as fallbacks.

    When ``requested`` does not exist, the first available checkpoint from
    runs/train/*/weights/best.pt, runs/*/weights/best.pt, then
    experiments/*/fold_0_best.pt is used.
    """
    requested_path = Path(requested)
    if requested_path.is_file():
        return requested_path

    candidates: list[Path] = []
    for pattern in (
        "runs/train/*/weights/best.pt",
        "runs/*/weights/best.pt",
        "experiments/*/fold_0_best.pt",
    ):
        candidates.extend(sorted(Path(".").glob(pattern)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)

    if unique:
        logger.warning(
            "Model %s not found; falling back to %s", requested_path.resolve(), unique[0]
        )
        return unique[0]

    raise FileNotFoundError(
        f"Model not found at {requested_path.resolve()}. Searched "
        "runs/train/*/weights/best.pt, runs/*/weights/best.pt and "
        "experiments/*/fold_0_best.pt."
    )


def _verify_image(path: Path) -> bool:
    """Return True when ``path`` is a readable image, False otherwise."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _predict_batch(
    model: YOLO, paths: list[Path], args: argparse.Namespace, device: str
) -> list[Any]:
    """Predict over ``paths`` in one batch, degrading to per-image on failure.

    Returns a list aligned with ``paths``; entries are ``None`` when a
    per-image retry also failed.
    """
    try:
        return list(
            model.predict(
                [str(p) for p in paths],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                batch=len(paths),
                save=False,
                verbose=False,
            )
        )
    except Exception as exc:  # noqa: BLE001 - fall back to per-image predict
        logger.warning("Batch predict failed (%s); retrying individually", exc)
        results: list[Any] = []
        for path in paths:
            try:
                results.append(
                    model.predict(
                        str(path),
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=device,
                        save=False,
                        verbose=False,
                    )[0]
                )
            except Exception as inner:  # noqa: BLE001
                logger.warning("SKIP (predict failed): %s (%s)", path.name, inner)
                results.append(None)
        return results


def _extract_frame_detections(result: Any, frame_label: str, model: YOLO) -> list[dict]:
    """Convert one frame's boxes into CSV rows labelled ``frame_label``."""
    detections: list[dict] = []
    boxes = result.boxes
    if boxes is None:
        return detections
    names = model.names
    for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf, strict=False):
        detections.append(
            {
                "filename": frame_label,
                "class_id": int(cls_id),
                "class_name": names[int(cls_id)],
                "confidence": float(conf),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        )
    return detections


def _run_image_inference(
    model: YOLO,
    image_paths: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    save_csv: bool,
) -> tuple[int, int, list[dict]]:
    """Run batched inference over ``image_paths``.

    Returns ``(processed, total_detections, detections)``.
    """
    if args.save_img:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    valid: list[Path] = []
    for path in image_paths:
        if _verify_image(path):
            valid.append(path)
        else:
            logger.warning("SKIP (unreadable): %s", path.name)

    processed = 0
    total_detections = 0
    all_detections: list[dict] = []

    for start in range(0, len(valid), args.batch):
        batch = valid[start : start + args.batch]
        results = _predict_batch(model, batch, args, device)
        for path, result in zip(batch, results, strict=False):
            if result is None:
                continue
            boxes = result.boxes
            n_detections = 0 if boxes is None else len(boxes)
            processed += 1
            total_detections += n_detections
            logger.info("%s: %d detection(s)", path.name, n_detections)

            if args.save_img and boxes is not None:
                annotated = result.plot()
                save_path = output_dir / "images" / path.name
                Image.fromarray(annotated[..., ::-1]).save(save_path)  # BGR -> RGB

            if args.save_txt and boxes is not None:
                txt_path = output_dir / "labels" / f"{path.stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    for box, cls_id, conf in zip(boxes.xywhn, boxes.cls, boxes.conf, strict=False):
                        f.write(
                            f"{int(cls_id)} {box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f} {conf:.6f}\n"
                        )

            if save_csv and boxes is not None:
                names = model.names
                for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf, strict=False):
                    all_detections.append(
                        {
                            "filename": path.name,
                            "class_id": int(cls_id),
                            "class_name": names[int(cls_id)],
                            "confidence": float(conf),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }
                    )

    return processed, total_detections, all_detections


def _run_frame_loop(
    model: YOLO,
    capture_source: str | int,
    out_video_path: Path,
    frame_label_prefix: str,
    args: argparse.Namespace,
    device: str,
    save_csv: bool,
) -> tuple[int, int, list[dict]]:
    """Process frames one at a time via OpenCV (used for videos and webcam).

    Returns ``(processed, total_detections, detections)``.
    """
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source: {capture_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer: cv2.VideoWriter | None = None
    processed = 0
    total_detections = 0
    all_detections: list[dict] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                save=False,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(annotated)
            processed += 1
            n_detections = 0 if result.boxes is None else len(result.boxes)
            total_detections += n_detections
            logger.info("frame %04d: %d detection(s)", processed, n_detections)
            if save_csv and result.boxes is not None:
                all_detections.extend(
                    _extract_frame_detections(
                        result, f"{frame_label_prefix}_frame_{processed:04d}", model
                    )
                )
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                logger.info("Capture stopped by user (ESC)")
                break
    except KeyboardInterrupt:
        logger.info("Capture interrupted by user")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    return processed, total_detections, all_detections


def _run_video_stream(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    save_csv: bool,
) -> tuple[int, int, list[dict]]:
    """Process a video file with a streaming predictor.

    Falls back to :func:`_run_frame_loop` when streaming inference fails before
    any frame was processed. Returns ``(processed, total_detections, detections)``.
    """
    video_stem = video_path.stem
    out_video = output_dir / "videos" / f"{video_stem}_annotated.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer: cv2.VideoWriter | None = None
    processed = 0
    total_detections = 0
    all_detections: list[dict] = []

    try:
        results = model.predict(
            source=str(video_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            stream=True,
            save=False,
            verbose=False,
        )
        for idx, result in enumerate(results):
            frame = result.plot()  # BGR ndarray
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(frame)
            processed += 1
            n_detections = 0 if result.boxes is None else len(result.boxes)
            total_detections += n_detections
            logger.info("frame %04d: %d detection(s)", idx, n_detections)
            if save_csv and result.boxes is not None:
                all_detections.extend(
                    _extract_frame_detections(result, f"{video_stem}_frame_{idx:04d}", model)
                )
    except Exception as exc:  # noqa: BLE001 - fall back to a frame loop
        logger.warning("Streaming inference failed at frame %d (%s)", processed, exc)
        if processed == 0:
            logger.info("Falling back to frame-by-frame processing")
            return _run_frame_loop(
                model, str(video_path), out_video, video_stem, args, device, save_csv
            )
        logger.warning("Keeping the %d frame(s) processed so far", processed)
    finally:
        if writer is not None:
            writer.release()

    logger.info("Annotated video saved: %s", out_video)
    return processed, total_detections, all_detections


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Run inference for ``args.source`` and return a summary dict.

    ``args`` must provide: model, source, output, conf, iou, imgsz, batch,
    device, save_img, save_txt, save_csv. Returns a dict with keys:
    mode, processed, total_detections, mean_confidence, output_dir, csv_path.
    """
    model_path = _resolve_model_path(args.model)
    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv = args.save_csv
    if not args.save_img and not args.save_txt and not save_csv:
        save_csv = True

    logger.info("Loading model from %s", model_path)
    model = YOLO(str(model_path))
    logger.info("Model loaded — task: %s, device: %s", model.task, device)

    summary: dict[str, Any] = {
        "mode": "image",
        "processed": 0,
        "total_detections": 0,
        "mean_confidence": 0.0,
        "output_dir": str(output_dir.resolve()),
        "csv_path": None,
    }

    source = args.source
    if source == "0" or source.lower() == "webcam":
        summary["mode"] = "webcam"
        processed, total_detections, detections = _run_frame_loop(
            model,
            0,
            output_dir / "videos" / "webcam_annotated.mp4",
            "webcam",
            args,
            device,
            save_csv,
        )
    elif Path(source).suffix.lower() in VIDEO_EXTS:
        summary["mode"] = "video"
        processed, total_detections, detections = _run_video_stream(
            model, Path(source), output_dir, args, device, save_csv
        )
    else:
        image_paths = collect_images(source)
        if not image_paths:
            logger.warning("No images found matching --source '%s'", source)
            return summary
        logger.info("Found %d image(s) from: %s", len(image_paths), source)
        processed, total_detections, detections = _run_image_inference(
            model, image_paths, output_dir, args, device, save_csv
        )

    summary["processed"] = processed
    summary["total_detections"] = total_detections
    summary["mean_confidence"] = (
        sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0
    )

    if save_csv:
        if detections:
            csv_path = output_dir / "detections.csv"
            write_detections_csv(detections, csv_path)
            summary["csv_path"] = str(csv_path.resolve())
            logger.info("CSV saved: %s", csv_path.resolve())
        else:
            logger.info("No detections to write to CSV")

    logger.info(
        "Inference complete — mode=%s, processed=%d, detections=%d, "
        "mean_confidence=%.3f, output=%s",
        summary["mode"],
        summary["processed"],
        summary["total_detections"],
        summary["mean_confidence"],
        summary["output_dir"],
    )
    return summary


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    _setup_logging(args.verbose)
    try:
        run_inference(args)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
