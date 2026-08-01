"""Unit tests for helper functions in ``scripts/inference.py``.

The module-level ``importorskip`` guards the ``ultralytics`` dependency, so the
whole file skips gracefully when it is not installed (e.g. on minimal CI
images). No model weights or real images are required — everything uses
``tmp_path`` fixtures.
"""

import csv
from pathlib import Path

import pytest

# NOTE: import order matters on Windows. Importing the `scripts` package first
# loads the mlflow -> pandas -> pyarrow chain; pyarrow's native DLL must be
# loaded BEFORE torch/ultralytics native DLLs, otherwise the pytest process
# crashes with an access violation inside pyarrow.lib (Windows DLL conflict).
import scripts  # noqa: E402,F401  (imported for its side effects)

pytest.importorskip("ultralytics")

from scripts import inference  # noqa: E402

CSV_FIELDNAMES = ["filename", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]


def _make_image_dir(tmp_path: Path) -> Path:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.png").write_bytes(b"")
    (img_dir / "b.jpg").write_bytes(b"")
    (img_dir / "c.jpeg").write_bytes(b"")
    (img_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    return img_dir


# ---------------------------------------------------------------------------
# collect_images
# ---------------------------------------------------------------------------


def test_collect_images_single_file(tmp_path):
    image = tmp_path / "single.jpg"
    image.write_bytes(b"")
    assert inference.collect_images(str(image)) == [image]


def test_collect_images_directory(tmp_path):
    img_dir = _make_image_dir(tmp_path)
    names = [p.name for p in inference.collect_images(str(img_dir))]
    assert names == ["a.png", "b.jpg", "c.jpeg"]


def test_collect_images_glob(tmp_path):
    _make_image_dir(tmp_path)
    found = inference.collect_images(str(tmp_path / "imgs" / "*.jpg"))
    assert [p.name for p in found] == ["b.jpg"]


def test_collect_images_empty_glob_returns_empty_list(tmp_path):
    assert inference.collect_images(str(tmp_path / "does_not_exist_*.png")) == []


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert inference.resolve_device(None) == "cpu"


def test_resolve_device_accepts_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert inference.resolve_device(None) in ("cuda:0", "cpu")


# ---------------------------------------------------------------------------
# write_detections_csv
# ---------------------------------------------------------------------------


def _sample_detections() -> list[dict]:
    return [
        {
            "filename": "a.jpg",
            "class_id": 0,
            "class_name": "hole",
            "confidence": 0.91,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 3.0,
            "y2": 4.0,
        },
        {
            "filename": "b.jpg",
            "class_id": 0,
            "class_name": "hole",
            "confidence": 0.44,
            "x1": 5.0,
            "y1": 6.0,
            "x2": 7.0,
            "y2": 8.0,
        },
    ]


def test_write_detections_csv_header_and_rows(tmp_path):
    csv_path = tmp_path / "detections.csv"
    detections = _sample_detections()
    inference.write_detections_csv(detections, csv_path)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_FIELDNAMES
        rows = list(reader)

    assert len(rows) == 2
    for row, expected in zip(rows, detections, strict=True):
        assert row == {key: str(value) for key, value in expected.items()}


def test_write_detections_csv_empty_writes_header_only(tmp_path):
    csv_path = tmp_path / "detections.csv"
    inference.write_detections_csv([], csv_path)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_FIELDNAMES
        assert list(reader) == []
