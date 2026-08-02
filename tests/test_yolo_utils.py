"""Tests for the shared YOLO label / bbox helpers in ``scripts/yolo_utils.py``.

The module is extracted from three duplicated copies of the same parsing logic
(``dataset_analysis.py``, ``visualize_augmentations.py``,
``visualize_conversion.py``).
"""

from PIL import Image

from scripts import yolo_utils

# ---------------------------------------------------------------------------
# parse_yolo_label
# ---------------------------------------------------------------------------


def test_parse_yolo_label_valid_line():
    assert yolo_utils.parse_yolo_label("0 0.5 0.5 0.25 0.25") == (0, 0.5, 0.5, 0.25, 0.25)


def test_parse_yolo_label_blank_line_returns_none():
    assert yolo_utils.parse_yolo_label("") is None
    assert yolo_utils.parse_yolo_label("   ") is None


def test_parse_yolo_label_malformed_line_returns_none():
    assert yolo_utils.parse_yolo_label("0 0.5 not-a-number 0.25 0.25") is None
    assert yolo_utils.parse_yolo_label("a b c d e") is None


def test_parse_yolo_label_wrong_field_count_returns_none():
    assert yolo_utils.parse_yolo_label("0 0.5 0.5") is None
    assert yolo_utils.parse_yolo_label("0 0.5 0.5 0.25 0.25 0.9") is None


# ---------------------------------------------------------------------------
# load_yolo_labels
# ---------------------------------------------------------------------------


def test_load_yolo_labels_mixed_valid_and_invalid(tmp_path):
    label_file = tmp_path / "labels.txt"
    label_file.write_text(
        "0 0.5 0.5 0.25 0.25\n"
        "1 0.3 0.4 0.1 0.2\n"
        "not a valid line\n"
        "2 0.1 0.1 0.1 0.1\n"
        "\n"
        "0 0.9 0.9 0.05 0.05\n",
        encoding="utf-8",
    )
    labels = yolo_utils.load_yolo_labels(label_file)
    assert labels == [
        (0, 0.5, 0.5, 0.25, 0.25),
        (1, 0.3, 0.4, 0.1, 0.2),
        (2, 0.1, 0.1, 0.1, 0.1),
        (0, 0.9, 0.9, 0.05, 0.05),
    ]


def test_load_yolo_labels_empty_file(tmp_path):
    label_file = tmp_path / "empty.txt"
    label_file.write_text("", encoding="utf-8")
    assert yolo_utils.load_yolo_labels(label_file) == []


# ---------------------------------------------------------------------------
# denormalize_bbox
# ---------------------------------------------------------------------------


def test_denormalize_bbox_known_values():
    x1, y1, x2, y2 = yolo_utils.denormalize_bbox(0.5, 0.5, 0.5, 0.5, 100, 200)
    assert (x1, y1, x2, y2) == (25.0, 50.0, 75.0, 150.0)


def test_denormalize_bbox_full_image_span():
    x1, y1, x2, y2 = yolo_utils.denormalize_bbox(0.5, 0.5, 1.0, 1.0, 640, 480)
    assert (x1, y1, x2, y2) == (0.0, 0.0, 640.0, 480.0)


def test_denormalize_bbox_integer_dims():
    x1, y1, x2, y2 = yolo_utils.denormalize_bbox(0.25, 0.75, 0.5, 0.5, 64, 32)
    assert (x1, y1, x2, y2) == (0.0, 16.0, 32.0, 32.0)


# ---------------------------------------------------------------------------
# validate_bbox
# ---------------------------------------------------------------------------


def test_validate_bbox_in_range_is_valid():
    assert yolo_utils.validate_bbox(0.5, 0.5, 0.25, 0.25) is True
    assert yolo_utils.validate_bbox(0.0, 0.0, 1.0, 1.0) is True


def test_validate_bbox_out_of_range_is_invalid():
    assert yolo_utils.validate_bbox(1.5, 0.5, 0.25, 0.25) is False
    assert yolo_utils.validate_bbox(0.5, -0.1, 0.25, 0.25) is False
    assert yolo_utils.validate_bbox(0.5, 0.5, 1.2, 0.25) is False


def test_validate_bbox_zero_width_or_height_is_invalid():
    assert yolo_utils.validate_bbox(0.5, 0.5, 0.0, 0.25) is False
    assert yolo_utils.validate_bbox(0.5, 0.5, 0.25, 0.0) is False


# ---------------------------------------------------------------------------
# image_dims
# ---------------------------------------------------------------------------


def test_image_dims_returns_width_height(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (64, 32), color="red").save(image_path)
    assert yolo_utils.image_dims(image_path) == (64, 32)


def test_image_dims_accepts_str_path(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (10, 20), color="blue").save(image_path)
    assert yolo_utils.image_dims(str(image_path)) == (10, 20)
