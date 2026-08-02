"""Unit tests for ``scripts/model_registry.py`` (the best-model registry).

Pure-stdlib tests: no weights, no torch/ultralytics. The module's path
constants are monkeypatched onto ``tmp_path`` in every test, so the real
``models/`` directory is never touched and the suite runs cleanly on CI.
"""

import csv
import json
from pathlib import Path

import pytest

# NOTE: import order matters on Windows. Importing the `scripts` package first
# loads the mlflow -> pandas -> pyarrow chain; pyarrow's native DLL must be
# loaded BEFORE torch/ultralytics native DLLs, otherwise the pytest process
# crashes with an access violation inside pyarrow.lib (Windows DLL conflict).
import scripts  # noqa: F401  (imported for its side effects)
from scripts import model_registry  # noqa: E402


@pytest.fixture
def registry_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sandbox the registry module under ``tmp_path``.

    The three module-level path constants are all monkeypatched so every test
    writes only under ``tmp_path`` (never the real ``models/`` dir).
    """
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "models" / "best.json")
    monkeypatch.setattr(model_registry, "BEST_PT_PATH", tmp_path / "models" / "best.pt")
    return tmp_path, model_registry.REGISTRY_PATH, model_registry.BEST_PT_PATH


def _write_results_csv(run_dir: Path, rows: list[tuple[str, str]]) -> None:
    """Write an Ultralytics-style ``results.csv`` (header + per-epoch rows)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "metrics/mAP50(B)"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# update_best_model
# ---------------------------------------------------------------------------


def test_update_best_model_creates_registry_and_copies_checkpoint(registry_env):
    tmp_path, registry_path, best_pt_path = registry_env
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"weights-v1")

    updated = model_registry.update_best_model(checkpoint, {"mAP50": 0.5}, experiment="yolo26m")

    assert updated is True
    assert registry_path.exists()
    assert best_pt_path.exists()
    assert best_pt_path.read_bytes() == b"weights-v1"
    entry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert entry["model"] == str(checkpoint)
    assert entry["best_pt"] == str(best_pt_path)
    assert entry["experiment"] == "yolo26m"
    assert entry["metrics"] == {"mAP50": 0.5}
    assert "updated_at" in entry


def test_update_best_model_does_not_overwrite_when_score_lower(registry_env):
    tmp_path, registry_path, best_pt_path = registry_env
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"weights-v1")
    assert model_registry.update_best_model(checkpoint, {"mAP50": 0.5}) is True

    content_before = best_pt_path.read_bytes()
    mtime_before = best_pt_path.stat().st_mtime
    registry_before = registry_path.read_text(encoding="utf-8")

    worse = tmp_path / "worse.pt"
    worse.write_bytes(b"weights-worse")
    updated = model_registry.update_best_model(worse, {"mAP50": 0.3})

    assert updated is False
    assert best_pt_path.read_bytes() == content_before
    assert best_pt_path.stat().st_mtime == mtime_before
    assert registry_path.read_text(encoding="utf-8") == registry_before


def test_update_best_model_overwrites_when_score_higher(registry_env):
    tmp_path, registry_path, best_pt_path = registry_env
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"weights-v1")
    assert model_registry.update_best_model(checkpoint, {"mAP50": 0.5}) is True

    better = tmp_path / "better.pt"
    better.write_bytes(b"weights-v2")
    updated = model_registry.update_best_model(better, {"mAP50": 0.7}, experiment="yolo26x")

    assert updated is True
    assert best_pt_path.read_bytes() == b"weights-v2"
    entry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert entry["model"] == str(better)
    assert entry["experiment"] == "yolo26x"
    assert entry["metrics"]["mAP50"] == 0.7


def test_update_best_model_is_noop_without_usable_score(registry_env):
    tmp_path, registry_path, best_pt_path = registry_env
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"weights-v1")

    updated = model_registry.update_best_model(checkpoint, {"precision": 0.9})

    assert updated is False
    assert not registry_path.exists()
    assert not best_pt_path.exists()


# ---------------------------------------------------------------------------
# get_best
# ---------------------------------------------------------------------------


def test_get_best_returns_none_for_missing_registry(registry_env):
    assert model_registry.get_best() is None


def test_get_best_returns_parsed_dict_for_existing_registry(registry_env):
    _, registry_path, _ = registry_env
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({"model": "runs/train/exp/weights/best.pt", "metrics": {"mAP50": 0.5}}),
        encoding="utf-8",
    )

    entry = model_registry.get_best()

    assert entry == {"model": "runs/train/exp/weights/best.pt", "metrics": {"mAP50": 0.5}}


def test_get_best_returns_none_for_corrupt_registry(registry_env):
    _, registry_path, _ = registry_env
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("not json{", encoding="utf-8")

    assert model_registry.get_best() is None


# ---------------------------------------------------------------------------
# scan_runs
# ---------------------------------------------------------------------------


def test_scan_runs_picks_highest_mAP50_checkpoint(registry_env):
    tmp_path, _, _ = registry_env
    run_a = tmp_path / "runs" / "exp_a"
    run_b = tmp_path / "runs" / "exp_b"
    (run_a / "weights").mkdir(parents=True)
    (run_b / "weights").mkdir(parents=True)
    (run_a / "weights" / "best.pt").write_bytes(b"a")
    (run_b / "weights" / "best.pt").write_bytes(b"b")
    _write_results_csv(run_a, [("0", "0.4"), ("1", "0.55")])
    _write_results_csv(run_b, [("0", "0.3"), ("1", "0.75")])

    best = model_registry.scan_runs()

    assert best is not None
    assert best["model"] == str(run_b / "weights" / "best.pt")
    assert best["metrics"] == {"mAP50": 0.75}
    assert best["experiment"] == "exp_b"


def test_scan_runs_returns_none_when_no_outputs_exist(registry_env):
    assert model_registry.scan_runs() is None


# ---------------------------------------------------------------------------
# resolve_best
# ---------------------------------------------------------------------------


def test_resolve_best_returns_registry_path_when_model_exists(registry_env):
    tmp_path, _, _ = registry_env
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"weights-v1")
    assert model_registry.update_best_model(checkpoint, {"mAP50": 0.5}) is True

    resolved = model_registry.resolve_best()

    assert resolved == checkpoint
    assert resolved.is_file()


def test_resolve_best_falls_back_to_scan_runs_when_registry_missing(registry_env):
    tmp_path, _, _ = registry_env
    run_dir = tmp_path / "runs" / "exp_b"
    (run_dir / "weights").mkdir(parents=True)
    _write_results_csv(run_dir, [("0", "0.75")])

    resolved = model_registry.resolve_best()

    assert resolved == run_dir / "weights" / "best.pt"


def test_resolve_best_falls_back_to_scan_runs_when_registry_stale(registry_env):
    tmp_path, registry_path, _ = registry_env
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({"model": str(tmp_path / "missing.pt"), "metrics": {"mAP50": 0.9}}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "exp_b"
    (run_dir / "weights").mkdir(parents=True)
    _write_results_csv(run_dir, [("0", "0.4")])

    resolved = model_registry.resolve_best()

    assert resolved == run_dir / "weights" / "best.pt"


def test_resolve_best_raises_when_nothing_exists(registry_env):
    with pytest.raises(FileNotFoundError, match="python scripts/train.py"):
        model_registry.resolve_best()
