"""Shared helpers for YOLO-format labels and bounding boxes.

Provides parsing, validation and denormalisation utilities used across the
analysis and visualization scripts in this project.  YOLO labels are stored
as one annotation per line: ``<class_id> <cx> <cy> <w> <h>`` with all four
box values normalised to [0, 1] relative to the image.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Tolerance (in normalised units) allowed for label-smoothing edge cases
# where a box coordinate sits marginally outside [0, 1].
_EPS = 1e-3


def parse_yolo_label(
    line: str,
) -> tuple[int, float, float, float, float] | None:
    """Parse a single YOLO label line.

    Expected format: ``<class_id> <cx> <cy> <w> <h>`` with exactly five
    whitespace-separated fields.

    Parameters
    ----------
    line : str
        A single line from a YOLO label file.

    Returns
    -------
    tuple[int, float, float, float, float] or None
        ``(class_id, cx, cy, w, h)`` on success, or ``None`` when the line is
        malformed (wrong field count or non-numeric values).
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None
    return class_id, cx, cy, w, h


def load_yolo_labels(
    path: Path,
) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file into a list of parsed annotations.

    Lines that fail :func:`parse_yolo_label` (blank or malformed) are
    skipped.

    Parameters
    ----------
    path : Path
        Path to the ``.txt`` label file.

    Returns
    -------
    list[tuple[int, float, float, float, float]]
        One ``(class_id, cx, cy, w, h)`` tuple per valid line.
    """
    labels: list[tuple[int, float, float, float, float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_yolo_label(line)
            if parsed is not None:
                labels.append(parsed)
    return labels


def denormalize_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert a normalised YOLO bbox to pixel coordinates.

    Parameters
    ----------
    cx, cy, w, h : float
        Normalised centre and size in [0, 1].
    img_w, img_h : int
        Image dimensions in pixels.

    Returns
    -------
    tuple[float, float, float, float]
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

    return x1, y1, x2, y2


def validate_bbox(cx: float, cy: float, w: float, h: float) -> bool:
    """Check that a normalised bbox has plausible values.

    A bbox is valid when the centre coordinates lie in [0, 1] and the width
    and height are positive and no larger than 1.  A small epsilon
    (``1e-3``) is tolerated for label smoothing, which can push values
    marginally outside [0, 1].

    Parameters
    ----------
    cx, cy, w, h : float
        Normalised bbox values.

    Returns
    -------
    bool
        ``True`` when the bbox is valid, ``False`` otherwise.
    """
    lo = -_EPS
    hi = 1.0 + _EPS
    return lo <= cx <= hi and lo <= cy <= hi and 0.0 < w <= hi and 0.0 < h <= hi


def image_dims(path: Path) -> tuple[int, int]:
    """Return the ``(width, height)`` of an image in pixels.

    Parameters
    ----------
    path : Path
        Path to the image file (any format Pillow can open).

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` of the image.
    """
    with Image.open(path) as img:
        return img.size
