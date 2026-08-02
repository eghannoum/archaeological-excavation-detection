#!/usr/bin/env python3
"""Visualize augmentations for archaeological hole detection dataset.

Generates side-by-side comparison grids showing original images alongside
light and heavy augmented versions with bounding boxes drawn.
Also validates bounding box integrity after augmentation.

Usage:
    python scripts/visualize_augmentations.py --samples 5 --output outputs/augmentation-samples/
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path so scripts.augmentation can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.augmentation import (  # noqa: E402  # intentional: import after sys.path setup
    get_pipeline,
)
from scripts.yolo_utils import denormalize_bbox, load_yolo_labels  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = {
    "original": (0, 255, 0),  # green
    "light": (255, 100, 0),  # orange (RGB)
    "heavy": (220, 0, 0),  # red (RGB)
}

ASPECT_RATIO_TOLERANCE = 0.3
"""Minimum allowed `min(ar_out/ar_in, ar_in/ar_out)` before flagging."""

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize augmentations for archaeological hole detection dataset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample images to visualize (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "augmentation-samples"),
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[list[float]],
    color: tuple[int, int, int],
    line_width: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image copy (in-place on the copy)."""
    img = image.copy()
    h, w = img.shape[:2]
    for cx, cy, bw, bh in bboxes:
        x1, y1, x2, y2 = denormalize_bbox(cx, cy, bw, bh, w, h)
        cv2.rectangle(
            img,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            line_width,
        )
    return img


def best_aspect_ratio_match(ar1: float, ar2: float) -> float:
    """Return the smaller of ar2/ar1 or ar1/ar2 (closer to 1.0 = better)."""
    if ar1 <= 0 or ar2 <= 0:
        return 0.0
    return min(ar2 / ar1, ar1 / ar2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_bboxes(
    input_bboxes: list[list[float]],
    output_bboxes: list[list[float]],
    sample_name: str,
    mode: str,
) -> dict:
    """Validate bbox integrity after augmentation.

    Checks
    ------
    1.  Count preservation — ``len(input) == len(output)``
    2.  Center coordinates within [0, 1] after augmentation
    3.  Aspect-ratio preservation (within ``ASPECT_RATIO_TOLERANCE``)

    Returns a dict with pass/fail booleans and detailed diagnostics.
    """
    results: dict = {
        "sample": sample_name,
        "mode": mode,
        "count_pass": True,
        "range_pass": True,
        "aspect_pass": True,
        "details": {},
    }
    in_count = len(input_bboxes)
    out_count = len(output_bboxes)
    results["details"]["input_count"] = in_count
    results["details"]["output_count"] = out_count

    # -- 1. Count -----------------------------------------------------------
    if in_count != out_count:
        results["count_pass"] = False
        diff = in_count - out_count
        results["details"]["count_diff"] = diff
        logger.warning(
            "%s [%s] Bbox count MISMATCH: %d in → %d out (Δ %+d)",
            sample_name,
            mode,
            in_count,
            out_count,
            diff,
        )
    else:
        logger.info(
            "%s [%s] Bbox count OK: %d → %d",
            sample_name,
            mode,
            in_count,
            out_count,
        )

    # -- 2. Center range ----------------------------------------------------
    out_of_range: list[tuple[int, float, float]] = []
    for i, bbox in enumerate(output_bboxes):
        cx, cy = bbox[0], bbox[1]
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            out_of_range.append((i, cx, cy))
    if out_of_range:
        results["range_pass"] = False
        results["details"]["out_of_range"] = out_of_range
        logger.warning(
            "%s [%s] %d bbox(es) have centers outside [0, 1]",
            sample_name,
            mode,
            len(out_of_range),
        )
    else:
        logger.info("%s [%s] All bbox centers in [0, 1]: OK", sample_name, mode)

    # -- 3. Aspect ratio ----------------------------------------------------
    min_change = 1.0
    aspect_changes: list[float] = []
    if in_count > 0 and out_count > 0:
        n = min(in_count, out_count)
        for i in range(n):
            ar_in = _aspect_ratio(input_bboxes[i])
            ar_out = _aspect_ratio(output_bboxes[i])
            match = best_aspect_ratio_match(ar_in, ar_out)
            aspect_changes.append(match)
            if match < min_change:
                min_change = match

    if aspect_changes:
        results["details"]["min_aspect_ratio_change"] = round(min_change, 4)
        results["details"]["mean_aspect_ratio_change"] = round(float(np.mean(aspect_changes)), 4)
        if min_change < ASPECT_RATIO_TOLERANCE:
            results["aspect_pass"] = False
            bad_count = sum(1 for c in aspect_changes if c < ASPECT_RATIO_TOLERANCE)
            results["details"]["bad_aspect_count"] = bad_count
            logger.warning(
                "%s [%s] Aspect ratio min match: %.3f — %d bbox(es) below " "tolerance %.1f",
                sample_name,
                mode,
                min_change,
                bad_count,
                ASPECT_RATIO_TOLERANCE,
            )
        else:
            logger.info(
                "%s [%s] Aspect ratios OK (min match: %.3f)",
                sample_name,
                mode,
                min_change,
            )

    return results


def _aspect_ratio(bbox: list[float]) -> float:
    """Aspect ratio *w / h* for a YOLO bbox [cx, cy, w, h]."""
    w, h = bbox[2], bbox[3]
    return w / h if h > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = PROJECT_ROOT / "dataset" / "images" / "train"
    label_dir = PROJECT_ROOT / "dataset" / "labels" / "train"

    # -- Collect image paths ------------------------------------------------
    image_paths = sorted(img_dir.glob("*.png"))
    if not image_paths:
        logger.error("No training images found in %s", img_dir)
        sys.exit(1)

    n_samples = min(args.samples, len(image_paths))
    selected = random.sample(image_paths, n_samples)
    logger.info(
        "Selected %d / %d training images for visualization",
        n_samples,
        len(image_paths),
    )

    # -- Initialise pipelines -----------------------------------------------
    light_pipeline = get_pipeline("light")
    heavy_pipeline = get_pipeline("heavy")
    if light_pipeline is None or heavy_pipeline is None:
        logger.error("Augmentation pipeline initialisation failed — aborting.")
        sys.exit(1)

    logger.info("Light pipeline: %s", light_pipeline)
    logger.info("Heavy pipeline: %s", heavy_pipeline)

    # -- Data storage for individual figures --------------------------------
    all_results: list[dict] = []

    # Per-row data so we can save individual sample figures
    rows_data: list[dict] = []

    # -- Build grid ---------------------------------------------------------
    ncols = 3
    fig, axes = plt.subplots(
        n_samples,
        ncols,
        figsize=(16, 5 * n_samples),
        constrained_layout=True,
    )
    if n_samples == 1:
        axes = axes.reshape(1, ncols)

    for row_idx, img_path in enumerate(selected):
        label_path = label_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            logger.warning("Label not found for %s — skipping row", img_path.name)
            continue

        # Load image (BGR → keep for cv2 drawing)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Could not read %s — skipping row", img_path.name)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # Load labels
        parsed_labels = load_yolo_labels(label_path)
        input_bboxes = [[cx, cy, w, h] for _, cx, cy, w, h in parsed_labels]
        class_labels = [cls_id for cls_id, _, _, _, _ in parsed_labels]
        n_bboxes = len(input_bboxes)
        logger.info(
            "Row %d: %s (%d×%d, %d bboxes)",
            row_idx + 1,
            img_path.name,
            img_w,
            img_h,
            n_bboxes,
        )

        # Line width proportional to image diagonal
        diag = math.sqrt(img_w**2 + img_h**2)
        line_width = max(2, int(diag / 180))

        # Apply augmentations
        light_result = light_pipeline(image=img_rgb, bboxes=input_bboxes, class_labels=class_labels)
        heavy_result = heavy_pipeline(image=img_rgb, bboxes=input_bboxes, class_labels=class_labels)

        light_img = light_result["image"]
        heavy_img = heavy_result["image"]
        light_bboxes = light_result["bboxes"]
        heavy_bboxes = heavy_result["bboxes"]

        # Validate
        for mode, obboxes in [("light", light_bboxes), ("heavy", heavy_bboxes)]:
            v = validate_bboxes(input_bboxes, obboxes, img_path.stem, mode)
            all_results.append(v)

        # Draw
        orig_drawn = draw_bboxes(img_rgb, input_bboxes, COLORS["original"], line_width)
        light_drawn = draw_bboxes(light_img, light_bboxes, COLORS["light"], line_width)
        heavy_drawn = draw_bboxes(heavy_img, heavy_bboxes, COLORS["heavy"], line_width)

        # Store for individual figures
        rows_data.append(
            {
                "stem": img_path.stem,
                "orig": orig_drawn,
                "light": light_drawn,
                "heavy": heavy_drawn,
                "n_orig": n_bboxes,
                "n_light": len(light_bboxes),
                "n_heavy": len(heavy_bboxes),
            }
        )

        # Plot row
        titles = [
            f"Original\n{n_bboxes} bboxes",
            f"Light Augmentation\n{len(light_bboxes)} bboxes",
            f"Heavy Augmentation\n{len(heavy_bboxes)} bboxes",
        ]
        imgs = [orig_drawn, light_drawn, heavy_drawn]

        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx])
            ax.set_title(titles[col_idx], fontsize=10, fontweight="bold")
            ax.axis("off")

        # Add filename label to the left of each row
        axes[row_idx, 0].set_ylabel(
            img_path.stem, fontsize=8, fontweight="bold", rotation=0, labelpad=12
        )

    # -- Save combined grid -------------------------------------------------
    fig.suptitle(
        "Augmentation Comparison: Original vs Light vs Heavy\n"
        "Green = original bboxes  ·  Orange = light aug  ·  Red = heavy aug",
        fontsize=13,
        fontweight="bold",
        y=1.005,
    )
    grid_path = output_dir / "augmentation_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    logger.info("Saved combined grid → %s", grid_path)
    plt.close(fig)

    # -- Save individual sample figures -------------------------------------
    for rd in rows_data:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        imgs = [rd["orig"], rd["light"], rd["heavy"]]
        titles = [
            f"Original ({rd['n_orig']} bboxes)",
            f"Light Augmentation ({rd['n_light']} bboxes)",
            f"Heavy Augmentation ({rd['n_heavy']} bboxes)",
        ]
        for col_idx in range(3):
            ax = axes2[col_idx]
            ax.imshow(imgs[col_idx])
            ax.set_title(titles[col_idx], fontsize=12, fontweight="bold")
            ax.axis("off")

        fig2.suptitle(rd["stem"], fontsize=14, fontweight="bold", y=1.02)
        individual_path = output_dir / f"{rd['stem']}_augmentation.png"
        fig2.savefig(individual_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("Saved individual grid → %s", individual_path)

    # -- Validation summary -------------------------------------------------
    logger.info("")
    logger.info("=" * 62)
    logger.info("  BBOX INTEGRITY — VALIDATION SUMMARY")
    logger.info("=" * 62)

    summary_data: dict[str, dict] = {}
    for mode in ("light", "heavy"):
        mode_results = [r for r in all_results if r["mode"] == mode]
        total = len(mode_results)
        count_ok = sum(1 for r in mode_results if r["count_pass"])
        range_ok = sum(1 for r in mode_results if r["range_pass"])
        aspect_ok = sum(1 for r in mode_results if r["aspect_pass"])

        summary_data[mode] = {
            "total": total,
            "count_ok": count_ok,
            "range_ok": range_ok,
            "aspect_ok": aspect_ok,
        }

        logger.info("")
        logger.info("  ┌─ %s (%d samples) ─────────────────────", mode.upper(), total)
        logger.info(
            "  │  Count preservation  …  %2d / %-2d  %s",
            count_ok,
            total,
            "✓" if count_ok == total else f"({total - count_ok} FAIL)",
        )
        logger.info(
            "  │  Center in [0, 1]    …  %2d / %-2d  %s",
            range_ok,
            total,
            "✓" if range_ok == total else f"({total - range_ok} FAIL)",
        )
        logger.info(
            "  │  Aspect ratio (tol)  …  %2d / %-2d  %s",
            aspect_ok,
            total,
            "✓" if aspect_ok == total else f"({total - aspect_ok} FAIL)",
        )
        logger.info("  └──────────────────────────────────────────")

    fail_any = any(
        not r["count_pass"] or not r["range_pass"] or not r["aspect_pass"] for r in all_results
    )
    if fail_any:
        logger.warning("")
        logger.warning("  ⚠  Some validation checks FAILED — see per-sample logs above.")
    else:
        logger.info("")
        logger.info("  ✓  All validation checks passed for all samples.")

    logger.info("")
    logger.info("  Figures saved to: %s", output_dir.resolve())
    logger.info("=" * 62)

    # Return summary for use by callers / tests
    return summary_data  # type: ignore[return-value]


if __name__ == "__main__":
    main()
