"""Shared fixtures for the test suite.

Every fixture is synthetic and self-contained: it builds a tiny YOLO-format
dataset under ``tmp_path`` and never touches the real (gitignored)
``dataset/``, a GPU, or the network.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

# Make the project root importable no matter how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("train", "val", "test")


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """Create a minimal YOLO-format dataset (images, labels, data.yaml).

    Returns the dataset root directory. Images are tiny 64x64 RGB JPEGs with
    one valid normalized bbox label each; ``data.yaml`` mirrors the layout of
    the real ``dataset/`` tree.
    """
    root = tmp_path / "dataset"
    counts = {"train": 3, "val": 1, "test": 1}
    for split in SPLITS:
        img_dir = root / "images" / split
        lab_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i in range(counts[split]):
            image = Image.new("RGB", (64, 64), color=(20 + 40 * i, 80, 160))
            image.save(img_dir / f"img_{i:02d}.jpg", format="JPEG")
            (lab_dir / f"img_{i:02d}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['hole']\n",
        encoding="utf-8",
    )
    return root
