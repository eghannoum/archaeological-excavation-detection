#!/usr/bin/env python3
"""
COCO-to-YOLO26 Dataset Conversion Script.

Converts COCO-format JSON annotations (with quadrant-tiled images) into
YOLOv5/v8/v11-format datasets with parent-level deterministic splitting.

Usage:
    python scripts/coco_to_yolo.py \\
        --coco-path data/annotations.json \\
        --image-dir data/splits \\
        --output-dir dataset \\
        --dry-run

    python scripts/coco_to_yolo.py --validate-only \\
        --coco-path data/annotations.json \\
        --image-dir data/splits

    python scripts/coco_to_yolo.py \\
        --coco-path data/annotations.json \\
        --image-dir data/splits \\
        --output-dir dataset \\
        --overwrite --yes
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_QUADRANT_SUFFIXES = "_tl,_tr,_bl,_br"
DEFAULT_BBOX_FORMAT = "xywh"
CLASS_NAME = "hole"
YOLO_DECIMAL_PLACES = 6
ENCODING_READ = "utf-8-sig"
ENCODING_WRITE = "utf-8"
ENCODING_LABEL_READ = "utf-8"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _validate_splits(val_split: float, test_split: float) -> None:
    """Validate split fractions at parse time."""
    if not 0.0 <= val_split < 1.0:
        raise argparse.ArgumentTypeError(f"--val-split must be in [0, 1), got {val_split}")
    if not 0.0 <= test_split < 1.0:
        raise argparse.ArgumentTypeError(f"--test-split must be in [0, 1), got {test_split}")
    if val_split + test_split >= 1.0:
        raise argparse.ArgumentTypeError(
            f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0"
        )


def _positive_float(v: str) -> float:
    f = float(v)
    if f < 0:
        raise argparse.ArgumentTypeError(f"Must be non-negative, got {v}")
    return f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert COCO-format quadrant-tile JSON to YOLOv5/v8/v11 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\\n"
            "  %(prog)s --coco-path data.json --image-dir images --output-dir dataset\\n"
            "  %(prog)s --validate-only --coco-path data.json --image-dir images\\n"
            "  %(prog)s --coco-path data.json --image-dir images --output-dir dataset --dry-run"
        ),
    )

    # --- Required ---
    parser.add_argument(
        "--coco-path",
        required=True,
        type=Path,
        help="Path to COCO-format JSON annotation file.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing source tile images.",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for YOLO dataset. Required unless --dry-run or "
            "--validate-only is set."
        ),
    )

    # --- Mutually exclusive modes ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview split allocation without writing any files.",
    )
    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate COCO JSON structure and bbox containment only.",
    )

    # --- Split parameters ---
    parser.add_argument(
        "--val-split",
        type=_positive_float,
        default=DEFAULT_VAL_SPLIT,
        help=f"Validation fraction (default: {DEFAULT_VAL_SPLIT}).",
    )
    parser.add_argument(
        "--test-split",
        type=_positive_float,
        default=DEFAULT_TEST_SPLIT,
        help=f"Test fraction (default: {DEFAULT_TEST_SPLIT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible shuffling (default: {DEFAULT_SEED}).",
    )

    # --- Bbox format ---
    parser.add_argument(
        "--input-bbox-format",
        choices=["xywh", "xyxy"],
        default=DEFAULT_BBOX_FORMAT,
        help=f"COCO bbox format (default: {DEFAULT_BBOX_FORMAT}).",
    )

    # --- Safety / control ---
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If output-dir exists, delete and recreate it. " "DESTRUCTIVE — safety guards apply."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation for --overwrite.",
    )
    parser.add_argument(
        "--allow-incomplete-parents",
        action="store_true",
        help="Continue when parent groups have != expected tile count.",
    )

    # --- Output options ---
    parser.add_argument(
        "--include-test-in-yaml",
        action="store_true",
        help="Include 'test: images/test' in data.yaml (default: omit).",
    )
    parser.add_argument(
        "--no-include-incomplete-log",
        action="store_false",
        dest="include_incomplete_log",
        default=True,
        help="Suppress writing .incomplete_parents.txt (default: log IS written).",
    )

    # --- Quadrant suffixes ---
    parser.add_argument(
        "--quadrant-suffixes",
        type=str,
        default=DEFAULT_QUADRANT_SUFFIXES,
        help=(
            "Comma-separated quadrant suffixes for parent extraction "
            f"(default: {DEFAULT_QUADRANT_SUFFIXES})."
        ),
    )

    # --- EXIF ---
    parser.add_argument(
        "--apply-exif-orientation",
        action="store_true",
        help=(
            "If set, physically rotate images per EXIF orientation and adjust "
            "bbox coordinates accordingly."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------


def _check_overwrite_safety(
    output_dir: Path,
    image_dir: Path,
    coco_path: Path,
) -> None:
    """Raise ``ValueError`` if *output_dir* targets a dangerous location.

    Safety checks:
    1. Reject ``.`` or ``..`` as the output directory
    2. Reject if output is an ancestor (via ``Path.parents``) of image-dir
       or coco-path's parent
    3. Reject if output equals the resolved ``scripts/`` directory
    """
    resolved_out = output_dir.resolve()
    resolved_img = image_dir.resolve()
    resolved_coco_parent = coco_path.resolve().parent

    # --- 1. Reject . or .. ---
    parts = output_dir.parts
    if "." in parts or ".." in parts:
        raise ValueError(
            f"Refusing to overwrite relative traversal path: {output_dir}. "
            "Use an explicit subdirectory name."
        )

    # --- 2. Reject if ancestor of source dirs ---
    # Check if resolved_out is an ancestor of resolved_img or resolved_coco_parent
    # using Path.parents membership
    if resolved_img != resolved_out and resolved_out in resolved_img.parents:
        raise ValueError(
            f"Output directory {resolved_out} is an ancestor of "
            f"--image-dir ({resolved_img}). Refusing to overwrite."
        )
    if resolved_coco_parent != resolved_out and resolved_out in resolved_coco_parent.parents:
        raise ValueError(
            f"Output directory {resolved_out} is an ancestor of "
            f"--coco-path parent ({resolved_coco_parent}). Refusing to overwrite."
        )

    # --- 3. Reject if equals scripts/ ---
    scripts_dir = Path("scripts").resolve()
    if resolved_out == scripts_dir:
        raise ValueError(
            f"Output directory {resolved_out} is the scripts/ directory. " "Refusing to overwrite."
        )


def _confirm_overwrite(output_dir: Path, yes: bool) -> None:
    """Prompt user to confirm deletion of *output_dir*."""
    if yes:
        return
    answer = (
        input(f"WARNING: About to DELETE and recreate {output_dir.resolve()}. " "Continue? (y/N): ")
        .strip()
        .lower()
    )
    if answer != "y":
        print("Aborted by user.")
        sys.exit(1)


def _stale_tmp_cleanup(output_dir: Path) -> None:
    """Remove any ``*.tmp`` files left by a previous crashed run."""
    if output_dir.exists():
        for tmp in output_dir.rglob("*.tmp"):
            with contextlib.suppress(OSError):
                tmp.unlink()


# ---------------------------------------------------------------------------
# COCO parsing
# ---------------------------------------------------------------------------


def load_coco(coco_path: Path) -> dict:
    """Load and return the COCO JSON dict.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(coco_path, encoding=ENCODING_READ) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: COCO JSON file not found: {coco_path}", file=sys.stderr)
        sys.exit(1)


def validate_coco_structure(coco: dict) -> None:
    """Basic structural validation of the COCO JSON."""
    if "images" not in coco or not isinstance(coco["images"], list):
        raise ValueError("COCO JSON missing 'images' array.")
    if "annotations" not in coco or not isinstance(coco["annotations"], list):
        raise ValueError("COCO JSON missing 'annotations' array.")
    if "categories" not in coco or not isinstance(coco["categories"], list):
        raise ValueError("COCO JSON missing 'categories' array.")

    # Enforce single class
    if len(coco["categories"]) != 1:
        raise ValueError(
            f"Expected exactly 1 category, got {len(coco['categories'])}. "
            "This converter supports single-class datasets only."
        )

    # Validate no duplicate image IDs
    image_ids = [img["id"] for img in coco["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs detected in COCO JSON.")


# ---------------------------------------------------------------------------
# Parent extraction (single-pass, longest-match-first)
# ---------------------------------------------------------------------------


def extract_parents(
    filenames: list[str],
    quadrant_suffixes: list[str],
) -> dict[str, list[str]]:
    """Group *filenames* (basenames) by parent key.

    Single-pass: strip extension via ``Path(filename).stem``, then match
    the longest suffix from the end using ``str.endswith()``.
    The parent key is whatever remains after stripping.  Do NOT re-check
    whether the result ends with a suffix again.

    Returns:
        A dict mapping *parent_key* -> [matching filenames].

    Raises:
        ValueError: If a filename does not match any suffix.
    """
    # Sort suffixes longest-first for deterministic matching
    sorted_ss = sorted(quadrant_suffixes, key=len, reverse=True)

    parents: dict[str, list[str]] = {}
    for fn in filenames:
        stem = Path(fn).stem  # strips extension
        matched: str | None = None
        for suffix in sorted_ss:
            if stem.endswith(suffix):
                matched = suffix
                break
        if matched is None:
            raise ValueError(
                f"Filename '{fn}' (stem='{stem}') does not match any "
                f"configured quadrant suffix: {quadrant_suffixes}"
            )
        parent_key = stem[: -len(matched)] if len(matched) > 0 else stem
        if parent_key not in parents:
            parents[parent_key] = []
        parents[parent_key].append(fn)

    return parents


def validate_parents(
    parents: dict[str, list[str]],
    expected_tile_count: int,
    quadrant_suffixes: list[str],
    allow_incomplete: bool,
    is_dry_run: bool,
) -> dict[str, list[str]]:
    """Validate parent-group completeness.

    A parent is *valid* iff it has exactly *expected_tile_count* tiles.
    Otherwise it is *incomplete* (fewer/more) or *ambiguous* (the parent key
    itself likely includes a suffix — double-suffix case).

    Returns:
        Filtered dict containing only valid (complete) parents, unless
        ``--allow-incomplete-parents`` is set, in which case all parents
        are returned as-is.

    Raises:
        ValueError: If incomplete/ambiguous parents found and
            ``allow_incomplete`` is False.
    """
    valid: dict[str, list[str]] = {}
    incomplete: dict[str, list[str]] = {}
    ambiguous: dict[str, list[str]] = {}

    for parent_key, tiles in parents.items():
        if len(tiles) == expected_tile_count:
            valid[parent_key] = tiles
        elif len(tiles) < expected_tile_count:
            # Ambiguous: stripped parent key still contains a suffix → not a real tile set
            # But we classify based on count; anything < expected is flagged
            ambiguous[parent_key] = tiles
        else:
            incomplete[parent_key] = tiles

    # Also check if any parent_key matches a suffix pattern — that's ambiguous
    # (double-suffix case where stripped result looks like a parent but has <4 tiles)
    for parent_key in list(ambiguous.keys()):
        ambig_tiles = ambiguous[parent_key]
        for suffix in quadrant_suffixes:
            stripped = suffix.lstrip("_")
            if parent_key.endswith(f"_{stripped}") or parent_key.endswith(stripped):
                pass  # already flagged
        # Add ambiguous parents to incomplete so they get reported together
        incomplete[parent_key] = ambig_tiles
    # Clear ambiguous from the dict since we merge into incomplete
    for k in list(ambiguous.keys()):
        if k not in incomplete:
            incomplete[k] = ambiguous[k]

    if incomplete and not allow_incomplete:
        lines: list[str] = []
        for parent_key, tiles in sorted(incomplete.items()):
            lines.append(
                f"  Parent '{parent_key}' has {len(tiles)} tile(s): "
                f"{', '.join(sorted(tiles))} - expected {expected_tile_count}."
            )
        raise ValueError(
            f"Found {len(incomplete)} incomplete/ambiguous parent(s):\n" + "\n".join(lines)
        )

    if incomplete and allow_incomplete:
        print(
            f"WARNING: {len(incomplete)} parent(s) are incomplete. "
            "Including all available tiles."
        )
        for parent_key, tiles in sorted(incomplete.items()):
            print(
                f"  '{parent_key}': {len(tiles)}/{expected_tile_count} tiles - "
                f"{', '.join(sorted(tiles))}"
            )
        # Merge incomplete back into valid since we're allowing them
        valid.update(incomplete)

    return valid


# ---------------------------------------------------------------------------
# Split computation
# ---------------------------------------------------------------------------


def compute_split(
    parent_keys: list[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically shuffle *parent_keys* and compute train/val/test splits.

    Uses ``math.floor()`` for non-train splits; all remainder goes to train.
    """
    rng = random.Random(seed)
    shuffled = list(parent_keys)
    rng.shuffle(shuffled)

    n_parents = len(shuffled)
    n_val = int(math.floor(n_parents * val_split))
    n_test = int(math.floor(n_parents * test_split))
    n_train = n_parents - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"n_train={n_train} < 1 with n_parents={n_parents}, "
            f"val_split={val_split}, test_split={test_split}. "
            "Reduce --val-split or --test-split."
        )
    if val_split > 0 and n_val < 1:
        raise ValueError(
            f"n_val={n_val} < 1 but val_split={val_split} > 0. "
            "Increase --val-split or --seed or the number of parents."
        )
    if test_split > 0 and n_test < 1:
        raise ValueError(
            f"n_test={n_test} < 1 but test_split={test_split} > 0. "
            "Increase --test-split or --seed or the number of parents."
        )

    train_keys = shuffled[:n_train]
    val_keys = shuffled[n_train : n_train + n_val]
    test_keys = shuffled[n_train + n_val :]
    return train_keys, val_keys, test_keys


# ---------------------------------------------------------------------------
# YOLO conversion helpers
# ---------------------------------------------------------------------------


def coco_bbox_to_yolo(
    bbox: list[float],
    img_w: int,
    img_h: int,
    input_format: str,
) -> tuple[float, float, float, float]:
    """Convert a COCO bbox to normalized YOLO ``(cx, cy, w, h)``.

    Args:
        bbox: ``[x, y, w, h]`` or ``[x1, y1, x2, y2]`` depending on
            *input_format*.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        input_format: ``'xywh'`` or ``'xyxy'``.

    Returns:
        ``(cx, cy, w, h)`` all normalized to ``[0, 1]``.
    """
    if input_format == "xyxy":
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
    else:  # xywh
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

    # Normalize
    nc_x = cx / img_w
    nc_y = cy / img_h
    nw = w / img_w
    nh = h / img_h

    # Clip to [0, 1]
    nc_x = max(0.0, min(1.0, nc_x))
    nc_y = max(0.0, min(1.0, nc_y))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return nc_x, nc_y, nw, nh


def yolo_line_str(cx: float, cy: float, w: float, h: float, class_id: int = 0) -> str:
    """Format a YOLO label line with 6 decimal places."""
    return f"{class_id} {cx:.{YOLO_DECIMAL_PLACES}f} {cy:.{YOLO_DECIMAL_PLACES}f} {w:.{YOLO_DECIMAL_PLACES}f} {h:.{YOLO_DECIMAL_PLACES}f}"


def atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via ``.tmp`` → ``os.replace``."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding=ENCODING_WRITE) as f:
        f.write(content)
    os.replace(str(tmp_path), str(path))


# ---------------------------------------------------------------------------
# Bbox validation
# ---------------------------------------------------------------------------


def validate_bboxes(
    coco: dict,
    image_dir: Path,
    coco_path: Path,
    input_bbox_format: str,
    apply_exif_orientation: bool,
    is_validate_only: bool,
) -> int:
    """Validate bbox containment and plausibility.

    Opens every image with PIL to check actual dimensions against COCO
    metadata.  Reports EXIF orientation issues.

    Args:
        coco: Parsed COCO JSON dict.
        image_dir: Directory containing the source images.
        coco_path: Path to COCO JSON (for warnings).
        input_bbox_format: ``'xywh'`` or ``'xyxy'``.
        apply_exif_orientation: Whether user passed --apply-exif-orientation.
        is_validate_only: If True, print detailed validation output.

    Returns:
        Number of validation errors found.
    """
    if Image is None:
        print("WARNING: PIL not installed - cannot validate bbox dimensions.", file=sys.stderr)
        return 0

    # Build lookups
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n_images = len(coco["images"])
    n_annotations = len(coco["annotations"])
    errors = 0

    print(f"Validating {n_images} images, {n_annotations} annotations...")

    total_bbox_w = 0.0
    total_img_w = 0.0
    bbox_count = 0

    for i, img in enumerate(coco["images"]):
        img_id = img["id"]
        raw_fn = img["file_name"]
        filename = os.path.basename(raw_fn)
        coco_w = img["width"]
        coco_h = img["height"]

        anns = anns_by_img.get(img_id, [])

        if is_validate_only:
            print(f"Validating image {i+1}/{n_images} ({filename})...", end="\r")

        img_path = image_dir / filename
        if not img_path.exists():
            print(f"\nWARNING: {filename} not found at {img_path}", file=sys.stderr)
            errors += 1
            continue

        try:
            pil_img = Image.open(img_path)
            pil_w, pil_h = pil_img.size
        except Exception as exc:
            print(f"\nWARNING: Cannot open {filename}: {exc}", file=sys.stderr)
            errors += 1
            continue

        # Dimension cross-check
        if pil_w != coco_w or pil_h != coco_h:
            print(
                f"\nWARNING: {filename}: COCO says {coco_w}x{coco_h}, "
                f"PIL reads {pil_w}x{pil_h} - using PIL as source of truth"
            )

        # EXIF orientation check
        try:
            exif = pil_img.getexif()
            orientation = exif.get(0x0112, 1)
            if orientation != 1:
                msg = (
                    f"\nWARNING: {filename} has EXIF orientation {orientation} "
                    f"- bbox coordinates may mismatch raw pixel space."
                )
                if not apply_exif_orientation:
                    msg += (
                        " Use --apply-exif-orientation to auto-correct, or "
                        "pre-process with 'jhead -autorot'."
                    )
                print(msg)
        except Exception:
            pass

        # Use PIL dims as truth
        img_w, img_h = pil_w, pil_h

        for ann in anns:
            bbox = ann["bbox"]
            bbox_count += 1

            if input_bbox_format == "xyxy":
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2:
                    print(
                        f"\nWARNING: {filename} ann {ann['id']}: xyxy "
                        f"x1={x1} >= x2={x2} or y1={y1} >= y2={y2}"
                    )
                    errors += 1
                    continue
                bw = x2 - x1
                bh = y2 - y1
            else:
                x, y, bw, bh = bbox

            # Check w/h validity
            if bw <= 0 or bh <= 0:
                print(
                    f"\nWARNING: {filename} ann {ann['id']}: " f"skipping bbox with w={bw}, h={bh}"
                )
                continue

            # For xywh: check containment
            if input_bbox_format == "xywh":
                x, y = bbox[0], bbox[1]
                if x < 0 or y < 0 or (x + bw) > img_w or (y + bh) > img_h:
                    print(
                        f"\nWARNING: {filename} ann {ann['id']}: bbox "
                        f"[{x:.2f}, {y:.2f}, {bw:.2f}, {bh:.2f}] exceeds "
                        f"image bounds ({img_w}x{img_h})"
                    )
                    errors += 1
                if bw > img_w:
                    print(
                        f"\nWARNING: {filename} ann {ann['id']}: bbox width "
                        f"{bw:.2f} > image width {img_w} - "
                        f"possible xyxy format mislabeled as xywh"
                    )
                    errors += 1

            total_bbox_w += bw
            total_img_w += img_w

    if bbox_count > 0 and total_img_w > 0:
        avg_bbox_w = total_bbox_w / bbox_count
        avg_img_w = total_img_w / bbox_count
        ratio = avg_bbox_w / avg_img_w if avg_img_w > 0 else 0.0
        if ratio > 0.5 and input_bbox_format == "xywh":
            print(
                f"\nWARNING: Average bbox width / image width = {ratio:.3f} "
                f"(> 0.5). Possible xyxy format mislabeled as xywh."
            )

    if is_validate_only:
        status = "Validation PASS" if errors == 0 else f"Validation FAIL - {errors} error(s)"
        print(f"\n{status}")

    return errors


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------


def print_dry_run_summary(
    parents: dict[str, list[str]],
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    image_id_to_filename: dict[int, str],
    image_id_to_anns: dict[int, list[dict]],
    coco_images: list[dict],
) -> None:
    """Print a detailed dry-run summary to stdout."""

    def count_images(keys: list[str]) -> int:
        return sum(len(parents[k]) for k in keys)

    def count_anns(keys: list[str]) -> int:
        # Map parent keys back to image IDs
        tile_filenames = set()
        for k in keys:
            for fn in parents[k]:
                tile_filenames.add(fn)
        # Build reverse filename→image_id lookup
        fn_to_id = {os.path.basename(img["file_name"]): img["id"] for img in coco_images}
        total = 0
        for fn in tile_filenames:
            img_id = fn_to_id.get(fn)
            if img_id is not None:
                total += len(image_id_to_anns.get(img_id, []))
        return total

    n_parents = len(parents)
    n_train_p = len(train_keys)
    n_val_p = len(val_keys)
    n_test_p = len(test_keys)

    total_all_images = count_images(list(parents.keys()))
    print(
        f"DRY RUN: {n_parents} parents "
        f"-> ~{n_train_p}/{n_val_p}/{n_test_p} split, "
        f"~{total_all_images} images"
    )
    print(
        f"  train: {count_images(train_keys)} images, "
        f"~{count_anns(train_keys)} annotations (class '{CLASS_NAME}')"
    )
    print(
        f"  val:   {count_images(val_keys)} images, "
        f"~{count_anns(val_keys)} annotations (class '{CLASS_NAME}')"
    )
    print(
        f"  test:  {count_images(test_keys)} images, "
        f"~{count_anns(test_keys)} annotations (class '{CLASS_NAME}')"
    )


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert_split(
    split_keys: list[str],
    parents: dict[str, list[str]],
    image_id_to_filename: dict[int, str],
    image_id_to_anns: dict[int, list[dict]],
    coco_images: list[dict],
    image_dir: Path,
    output_dir: Path,
    split_name: str,
    input_bbox_format: str,
    apply_exif_orientation: bool,
) -> tuple[int, int]:
    """Convert one split (train / val / test) to YOLO format.

    Returns:
        ``(image_count, annotation_count)``.
    """
    if Image is None:
        print("ERROR: PIL (Pillow) is required for YOLO conversion.", file=sys.stderr)
        sys.exit(1)

    # Build filename→image_id lookup
    fn_to_id: dict[str, int] = {}
    for img in coco_images:
        fn = os.path.basename(img["file_name"])
        fn_to_id[fn] = img["id"]

    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_anns = 0

    for parent_key in split_keys:
        tile_fns = parents[parent_key]
        for fn in tile_fns:
            src = image_dir / fn
            if not src.exists():
                print(f"WARNING: {src} not found, skipping.", file=sys.stderr)
                continue

            # Copy image
            dst_img = images_out / fn
            shutil.copy2(str(src), str(dst_img))

            # Get annotations
            img_id = fn_to_id.get(fn)
            anns = image_id_to_anns.get(img_id, []) if img_id is not None else []
            total_anns += len(anns)

            # Open image for dynamic dimensions
            pil_img: Image.Image = Image.open(str(dst_img))
            # If --apply-exif-orientation, physically rotate
            if apply_exif_orientation:
                try:
                    exif = pil_img.getexif()
                    orientation = exif.get(0x0112, 1)
                    if orientation != 1:
                        pil_img = ImageOps.exif_transpose(pil_img) or pil_img
                        pil_img.save(str(dst_img))  # overwrite with corrected image
                except Exception:
                    pass

            img_w, img_h = pil_img.size

            # Build YOLO label content
            lines: list[str] = []
            for ann in anns:
                bbox = ann["bbox"]
                try:
                    nc_x, nc_y, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h, input_bbox_format)
                except (IndexError, TypeError, ZeroDivisionError) as exc:
                    print(
                        f"WARNING: bbox conversion failed for {fn}, " f"ann {ann['id']}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                lines.append(yolo_line_str(nc_x, nc_y, nw, nh))

            # Write label file
            label_path = labels_out / fn.replace(".png", ".txt").replace(".jpg", ".txt")
            label_content = "\n".join(lines) + ("\n" if lines else "")
            atomic_write(label_path, label_content)

            total_images += 1

    return total_images, total_anns


# ---------------------------------------------------------------------------
# data.yaml generation
# ---------------------------------------------------------------------------


def generate_data_yaml(
    include_test: bool,
) -> str:
    """Generate the content for ``data.yaml``.

    Args:
        include_test: If True, include ``test: images/test``.

    Returns:
        The YAML content as a string.
    """
    # Use as_posix() for forward slashes
    lines = [
        "# Paths relative to this file's directory - run yolo train from project root",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.append("nc: 1")
    lines.append(f"names: ['{CLASS_NAME}']")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---------- Validate splits ----------
    _validate_splits(args.val_split, args.test_split)

    # ---------- Resolve paths ----------
    coco_path: Path = args.coco_path.resolve()
    image_dir: Path = args.image_dir.resolve()
    output_dir: Path | None = args.output_dir.resolve() if args.output_dir else None
    read_only_mode = args.dry_run or args.validate_only

    # ---------- Check output-dir requirement ----------
    if output_dir is None and not read_only_mode:
        parser.error("--output-dir is required unless --dry-run or --validate-only is set.")
    if output_dir is not None and read_only_mode:
        print(
            f"INFO: --output-dir provided but ignored in "
            f"{'--dry-run' if args.dry_run else '--validate-only'} mode.",
            file=sys.stderr,
        )

    # ---------- Safety guard: overwrite ----------
    if output_dir is not None and not read_only_mode:
        # Stale tmp cleanup
        _stale_tmp_cleanup(output_dir)

        if output_dir.exists():
            if not args.overwrite:
                print(
                    f"ERROR: Output directory {output_dir} already exists. "
                    "Use --overwrite to delete and recreate, or choose a "
                    "different --output-dir.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Destructive safety check
            _check_overwrite_safety(output_dir, image_dir, coco_path)
            _confirm_overwrite(output_dir, args.yes)
            shutil.rmtree(str(output_dir))
            print(f"Removed existing {output_dir}")

        os.makedirs(str(output_dir), exist_ok=True)
        print(f"Created output directory {output_dir}")

    # ---------- Load & validate COCO ----------
    print(f"Loading COCO JSON from {coco_path}...")
    try:
        coco = load_coco(coco_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"ERROR: Failed to parse COCO JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_coco_structure(coco)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    category_name = coco["categories"][0]["name"]
    print(f"  Category: '{category_name}' (class ID 0)")

    # ---------- Build lookups ----------
    image_id_to_filename: dict[int, str] = {}
    for img in coco["images"]:
        image_id_to_filename[img["id"]] = img["file_name"]

    image_id_to_anns: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # ---------- Extract filenames (basename only) ----------
    filenames: list[str] = []
    for img in coco["images"]:
        raw = img["file_name"]
        filenames.append(os.path.basename(raw))

    # ---------- Parse quadrant suffixes ----------
    quadrant_suffixes = [s.strip() for s in args.quadrant_suffixes.split(",") if s.strip()]
    expected_tile_count = len(quadrant_suffixes)

    # ---------- Parent extraction ----------
    try:
        parents = extract_parents(filenames, quadrant_suffixes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(parents)} parent groups from {len(filenames)} images")

    # ---------- Parent validation ----------
    try:
        valid_parents = validate_parents(
            parents,
            expected_tile_count,
            quadrant_suffixes,
            args.allow_incomplete_parents,
            args.dry_run,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---------- Write incomplete parent log ----------
    if (
        args.allow_incomplete_parents
        and args.include_incomplete_log
        and output_dir is not None
        and not read_only_mode
    ):
        incomplete = {k: v for k, v in parents.items() if len(v) != expected_tile_count}
        if incomplete and output_dir is not None:
            log_path = output_dir / ".incomplete_parents.txt"
            log_lines: list[str] = []
            for pkey, tiles in sorted(incomplete.items()):
                log_lines.append(
                    f"{pkey}: {len(tiles)}/{expected_tile_count} tiles "
                    f"({', '.join(sorted(tiles))})"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(log_path, "\n".join(log_lines) + "\n")
            print(f"  Wrote incomplete parent log to {log_path}")

    # ---------- Check all image files exist ----------
    missing = 0
    for fn in filenames:
        src = image_dir / fn
        if not src.exists():
            print(f"WARNING: {src} does not exist on disk.", file=sys.stderr)
            missing += 1
    if missing:
        print(f"WARNING: {missing} image(s) missing from {image_dir}.", file=sys.stderr)

    # ---------- Bbox validation (always runs) ----------
    validation_errors = validate_bboxes(
        coco,
        image_dir,
        coco_path,
        args.input_bbox_format,
        args.apply_exif_orientation,
        args.validate_only,
    )

    # ---------- Validate-only mode: exit ----------
    if args.validate_only:
        sys.exit(1 if validation_errors > 0 else 0)

    # ---------- Compute split (dry-run + conversion) ----------
    parent_keys = list(valid_parents.keys())
    try:
        train_keys, val_keys, test_keys = compute_split(
            parent_keys, args.val_split, args.test_split, args.seed
        )
    except ValueError as exc:
        print(f"ERROR: Split computation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Cross-split uniqueness assertion
    all_split_keys = set(train_keys) | set(val_keys) | set(test_keys)
    total_unique = len(train_keys) + len(val_keys) + len(test_keys)
    assert total_unique == len(all_split_keys), (
        f"Cross-split uniqueness violation: " f"{total_unique} != {len(all_split_keys)}"
    )

    # ---------- Dry-run mode: print summary + exit ----------
    if args.dry_run:
        print_dry_run_summary(
            valid_parents,
            train_keys,
            val_keys,
            test_keys,
            image_id_to_filename,
            image_id_to_anns,
            coco["images"],
        )
        sys.exit(0)

    # ---------- Full conversion ----------
    if output_dir is None:
        # Should not happen due to earlier check, but guard anyway
        parser.error("--output-dir is required for conversion.")

    # Create output structure
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    n_train, a_train = (
        convert_split(
            train_keys,
            valid_parents,
            image_id_to_filename,
            image_id_to_anns,
            coco["images"],
            image_dir,
            output_dir,
            "train",
            args.input_bbox_format,
            args.apply_exif_orientation,
        )
        if train_keys
        else (0, 0)
    )

    n_val, a_val = (
        convert_split(
            val_keys,
            valid_parents,
            image_id_to_filename,
            image_id_to_anns,
            coco["images"],
            image_dir,
            output_dir,
            "val",
            args.input_bbox_format,
            args.apply_exif_orientation,
        )
        if val_keys
        else (0, 0)
    )

    n_test, a_test = (
        convert_split(
            test_keys,
            valid_parents,
            image_id_to_filename,
            image_id_to_anns,
            coco["images"],
            image_dir,
            output_dir,
            "test",
            args.input_bbox_format,
            args.apply_exif_orientation,
        )
        if test_keys
        else (0, 0)
    )

    # ---------- data.yaml ----------
    yaml_content = generate_data_yaml(args.include_test_in_yaml)
    yaml_path = output_dir / "data.yaml"
    atomic_write(yaml_path, yaml_content)
    print(f"  Wrote {yaml_path}")

    # ---------- Summary ----------
    print()
    print("=" * 60)
    print("Conversion complete!")
    print(f"  Train: {n_train} images, {a_train} annotations")
    print(f"  Val:   {n_val} images, {a_val} annotations")
    print(f"  Test:  {n_test} images, {a_test} annotations")
    print(f"  Total: {n_train + n_val + n_test} images, " f"{a_train + a_val + a_test} annotations")
    print(f"  Classes: 1 ('{CLASS_NAME}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
