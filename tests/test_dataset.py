"""Dataset layout validation.

Two groups of tests:

1. **Real dataset** (``dataset/``) — the tree is gitignored and rebuilt via
   ``scripts/coco_to_yolo.py``, so these tests skip gracefully when absent
   (e.g. on CI after a fresh clone).
2. **Synthetic fixture** — label/image parity and bbox range checks against
   the ``yolo_dataset`` fixture, which always runs and requires no data.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS = ("train", "val", "test")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _load_data_yaml(dataset_root: Path) -> dict:
    with open(dataset_root / "data.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), f"{dataset_root / 'data.yaml'} is not a mapping"
    return cfg


def _image_stems(split_dir: Path) -> list[str]:
    return sorted(p.stem for p in split_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def _label_stems(split_dir: Path) -> list[str]:
    return sorted(p.stem for p in split_dir.glob("*.txt"))


# Skip marker for the gitignored, reproducible dataset/ tree.
real_dataset = pytest.mark.skipif(
    not (DATASET_DIR / "data.yaml").is_file(),
    reason="dataset/ is gitignored and absent (rebuild with scripts/coco_to_yolo.py)",
)


# ---------------------------------------------------------------------------
# Real dataset/ tree (skips when absent)
# ---------------------------------------------------------------------------


@real_dataset
def test_dataset_data_yaml_has_required_keys():
    cfg = _load_data_yaml(DATASET_DIR)
    assert {"train", "val", "test", "nc", "names"} <= set(cfg)


@real_dataset
def test_dataset_data_yaml_nc_matches_names():
    cfg = _load_data_yaml(DATASET_DIR)
    assert isinstance(cfg["nc"], int) and cfg["nc"] > 0
    assert isinstance(cfg["names"], list)
    assert len(cfg["names"]) == cfg["nc"]


@real_dataset
def test_dataset_split_dirs_exist():
    for split in SPLITS:
        assert (DATASET_DIR / "images" / split).is_dir(), f"missing images/{split}"
        assert (DATASET_DIR / "labels" / split).is_dir(), f"missing labels/{split}"


@real_dataset
def test_dataset_label_image_parity():
    for split in SPLITS:
        images = _image_stems(DATASET_DIR / "images" / split)
        labels = _label_stems(DATASET_DIR / "labels" / split)
        assert images, f"no images found in images/{split}"
        assert images == labels, f"label/image stem mismatch in split '{split}'"


@real_dataset
def test_dataset_label_bbox_ranges():
    cfg = _load_data_yaml(DATASET_DIR)
    nc = cfg["nc"]
    for split in SPLITS:
        for label_file in sorted((DATASET_DIR / "labels" / split).glob("*.txt")):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                assert len(parts) == 5, f"{label_file.name}: expected 5 fields, got {len(parts)}"
                cls_id = int(parts[0])
                assert 0 <= cls_id < nc, f"{label_file.name}: class {cls_id} out of range"
                for coord in map(float, parts[1:]):
                    assert 0.0 <= coord <= 1.0, f"{label_file.name}: coord {coord} outside [0, 1]"


# ---------------------------------------------------------------------------
# Synthetic fixture (always runs — no real data required)
# ---------------------------------------------------------------------------


def test_fixture_data_yaml_has_required_keys(yolo_dataset):
    cfg = _load_data_yaml(yolo_dataset)
    assert {"train", "val", "test", "nc", "names"} <= set(cfg)
    assert cfg["nc"] == 1
    assert cfg["names"] == ["hole"]


def test_fixture_label_image_parity(yolo_dataset):
    for split in SPLITS:
        images = _image_stems(yolo_dataset / "images" / split)
        labels = _label_stems(yolo_dataset / "labels" / split)
        assert images, f"fixture created no images for split '{split}'"
        assert images == labels, f"label/image stem mismatch in fixture split '{split}'"


def test_fixture_label_bbox_ranges(yolo_dataset):
    for split in SPLITS:
        for label_file in (yolo_dataset / "labels" / split).glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                assert len(parts) == 5, f"{label_file.name}: expected 5 fields, got {len(parts)}"
                cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                assert cls_id == 0
                assert 0.0 <= cx <= 1.0
                assert 0.0 <= cy <= 1.0
                assert 0.0 < w <= 1.0
                assert 0.0 < h <= 1.0
