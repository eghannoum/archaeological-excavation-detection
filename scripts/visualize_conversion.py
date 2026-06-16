#!/usr/bin/env python3
"""
Overlay YOLO-format bounding boxes on images for visual QA.

Reads images and their corresponding YOLO .txt label files, draws axis-aligned
bounding boxes in red, and saves the overlaid image for human inspection.

Usage:
    python scripts/visualize_conversion.py \\
        --image-dir dataset/images/train \\
        --label-dir dataset/labels/train \\
        --output-dir dataset/debug \\
        --count 10 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ANNOTATIONS_PER_IMAGE = 20
RECT_OUTLINE_COLOR = "red"
RECT_OUTLINE_WIDTH = 3
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay YOLO-format bounding boxes on images for visual QA. "
            "Reads images and their corresponding .txt label files, draws "
            "bounding boxes, and saves the result."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --image-dir dataset/images/train --label-dir dataset/labels/train\n"
            "  %(prog)s --image-dir dataset/images/val --label-dir dataset/labels/val --count 10\n"
        ),
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Path to directory containing source images.",
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        type=Path,
        help="Path to directory containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/debug"),
        help="Output directory for debug images (default: dataset/debug).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of sample images to visualize (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible image selection (default: 42).",
    )

    return parser


# ---------------------------------------------------------------------------
# YOLO parsing helpers
# ---------------------------------------------------------------------------


def parse_yolo_label(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse a single YOLO label line.

    Expected format: ``<class_id> <cx> <cy> <w> <h>`` where cx, cy, w, h are
    normalized to [0, 1].

    Returns:
        ``(class_id, cx, cy, w, h)`` on success, or ``None`` on malformed input.
    """
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except (ValueError, IndexError):
        return None
    return class_id, cx, cy, w, h


def denormalize_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO bbox to pixel ``(x1, y1, x2, y2)``.

    Args:
        cx, cy, w, h: Normalized coordinates in [0, 1].
        img_w, img_h: Image dimensions in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` in pixel space.
    """
    px_cx = cx * img_w
    px_cy = cy * img_h
    px_w = w * img_w
    px_h = h * img_h

    x1 = px_cx - px_w / 2.0
    y1 = px_cy - px_h / 2.0
    x2 = px_cx + px_w / 2.0
    y2 = px_cy + px_h / 2.0

    # Clip to image bounds
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def find_images(image_dir: Path) -> list[Path]:
    """List all image files in *image_dir* sorted by name.

    Returns only files with extensions in ``VALID_IMAGE_EXTS``.
    """
    images: list[Path] = []
    for entry in sorted(image_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VALID_IMAGE_EXTS:
            images.append(entry)
    return images


def find_matching_images(
    image_dir: Path, label_dir: Path
) -> list[tuple[Path, Path]]:
    """Find image-label pairs where both files exist.

    Returns:
        List of ``(image_path, label_path)`` tuples, sorted by image filename.
    """
    images = find_images(image_dir)
    pairs: list[tuple[Path, Path]] = []
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def visualize_one(
    img_path: Path,
    label_path: Path,
    output_dir: Path,
) -> int:
    """Overlay bboxes from *label_path* on *img_path* and save to *output_dir*.

    Args:
        img_path: Path to the source image.
        label_path: Path to the YOLO .txt label file.
        output_dir: Directory where the debug image is saved.

    Returns:
        Number of annotations rendered.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"WARNING: Could not open {img_path.name}: {exc}", file=sys.stderr)
        return 0
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # Read label file
    with open(label_path, encoding="utf-8") as f:
        lines = f.readlines()

    rendered = 0
    for line in lines:
        parsed = parse_yolo_label(line)
        if parsed is None:
            continue
        class_id, cx, cy, w, h = parsed

        # Denormalize and convert to x1, y1, x2, y2
        x1, y1, x2, y2 = denormalize_bbox(cx, cy, w, h, img_w, img_h)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=RECT_OUTLINE_COLOR,
            width=RECT_OUTLINE_WIDTH,
        )
        rendered += 1

        if rendered >= MAX_ANNOTATIONS_PER_IMAGE:
            print(
                f"  WARNING: {img_path.name} has >{MAX_ANNOTATIONS_PER_IMAGE} "
                f"annotations — capping at {MAX_ANNOTATIONS_PER_IMAGE} for visibility"
            )
            break

    # Save output
    output_path = output_dir / f"{img_path.stem}_debug.png"
    img.save(output_path)

    return rendered


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- PIL availability ---
    if Image is None or ImageDraw is None:
        print(
            "ERROR: PIL (Pillow) is required. Install with: pip install Pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Resolve paths ---
    image_dir: Path = args.image_dir.resolve()
    label_dir: Path = args.label_dir.resolve()
    output_dir: Path = args.output_dir

    # --- Validate source directories exist ---
    if not image_dir.is_dir():
        print(f"ERROR: --image-dir '{image_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not label_dir.is_dir():
        print(f"ERROR: --label-dir '{label_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # --- Find matching image-label pairs ---
    pairs = find_matching_images(image_dir, label_dir)
    if not pairs:
        print(
            f"ERROR: No images with matching label files found in "
            f"'{image_dir}' / '{label_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_total = len(pairs)
    n_requested = args.count
    if n_total < n_requested:
        print(
            f"WARNING: Only {n_total} valid image(s) found with label files, "
            f"but --count={n_requested}. Proceeding with {n_total} image(s)."
        )

    # --- Random sample ---
    random.seed(args.seed)
    sample_size = min(n_requested, n_total)
    sampled_pairs = random.sample(pairs, sample_size)

    # --- Ensure output directory exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Process each image ---
    print(f"Visualizing {sample_size}/{n_total} images with YOLO overlays...")
    print(f"  Output dir: {output_dir.resolve()}")
    print()

    for img_path, label_path in sampled_pairs:
        n_anns = visualize_one(img_path, label_path, output_dir)
        print(f"  {img_path.name}: {n_anns} annotation(s) rendered")

    print()
    print(f"Done — debug images saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
