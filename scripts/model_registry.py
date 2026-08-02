"""Best-model registry: where the current best checkpoint and its metrics live.

Purpose
-------
Training and evaluation write the best checkpoint they produce to
``models/best.pt`` and record its provenance + metrics in ``models/best.json``
(see :func:`update_best_model`). Inference, evaluation, and the dashboard read
that registry to pick a default model when ``--model auto`` is requested (see
:func:`resolve_best`).

Who updates it
--------------
- ``scripts/train.py`` -- registers the best checkpoint of a training run.
- ``scripts/evaluate.py`` -- registers the evaluated checkpoint after writing
  ``runs/eval/results.json``.

Who reads it
------------
- ``scripts/inference.py`` -- ``--model auto`` -> ``resolve_best()``.
- ``scripts/evaluate.py`` -- ``--model auto`` -> ``resolve_best()``.
- ``scripts/dashboard.py`` -- default model -> ``resolve_best()``.

Fallback chain (:func:`resolve_best`)
-------------------------------------
1. ``models/best.json`` -- registry entry whose ``model`` path still exists.
2. :func:`scan_runs` -- highest mAP50 among ``runs/*/results.csv``,
   ``runs/eval/results.json``, and (unscored, newest first)
   ``experiments/*/fold_0_best.pt``.
3. ``FileNotFoundError`` with training guidance.

This module imports only the standard library, so consumers may import it
without torch/ultralytics installed (CI import guard).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Absolute path to the project root (two levels up from ``scripts/``)."""

REGISTRY_PATH = PROJECT_ROOT / "models" / "best.json"
"""Path to the best-model registry JSON."""

BEST_PT_PATH = PROJECT_ROOT / "models" / "best.pt"
"""Path to the copied best checkpoint."""

logger = logging.getLogger(__name__)


def _extract_score(metrics: dict) -> float | None:
    """Return the best available score (mAP50 > mAP50-95 > f1_score), or ``None``.

    Non-finite values (NaN/inf) are treated like missing keys: they are
    skipped, so a poisoned metric can never enter the registry.
    """
    for key in ("mAP50", "mAP50-95", "f1_score"):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        return value
    return None


def get_best() -> dict | None:
    """Return the parsed ``models/best.json`` entry, or ``None`` if missing/corrupt."""
    if not REGISTRY_PATH.exists():
        return None
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read registry %s: %s", REGISTRY_PATH, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Registry %s is not a JSON object; ignoring it", REGISTRY_PATH)
        return None
    return data


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically via a temp file + ``os.replace``."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def update_best_model(
    checkpoint_path: Path | str,
    metrics: dict,
    experiment: str | None = None,
) -> bool:
    """Register ``checkpoint_path`` as the best model if its score beats the current best.

    Score precedence: ``metrics["mAP50"]``, then ``mAP50-95``, then ``f1_score``.
    On update, the checkpoint is copied to ``BEST_PT_PATH`` and ``best.json`` is
    written atomically (temp file + ``os.replace``). Returns ``True`` if the
    registry was updated, ``False`` otherwise (no usable score key, or the new
    score does not beat the current best).
    """
    score = _extract_score(metrics)
    if score is None:
        logger.warning(
            "No usable score key (mAP50/mAP50-95/f1_score) in metrics %s; skipping update",
            metrics,
        )
        return False
    if not math.isfinite(score):
        logger.warning(
            "Non-finite score %r in metrics %s; skipping update",
            score,
            metrics,
        )
        return False

    current = get_best()
    current_score = _extract_score(current.get("metrics")) if current else None
    if current is not None and current_score is not None and score <= current_score:
        logger.info(
            "Current best score %.4f >= new score %.4f; keeping existing best",
            current_score,
            score,
        )
        return False

    BEST_PT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, BEST_PT_PATH)

    _atomic_write_json(
        REGISTRY_PATH,
        {
            "model": str(checkpoint_path),
            "best_pt": str(BEST_PT_PATH),
            "experiment": experiment,
            "metrics": metrics,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    logger.info("Registered new best model (score=%.4f) from %s", score, checkpoint_path)
    return True


def _max_csv_mAP50(csv_path: Path) -> float | None:
    """Parse the max ``metrics/mAP50(B)`` value (fallback ``mAP50(B)``) from an Ultralytics CSV."""
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            column = (
                "metrics/mAP50(B)"
                if "metrics/mAP50(B)" in (reader.fieldnames or [])
                else "mAP50(B)"
            )
            values = []
            for row in reader:
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    continue
                values.append(value)
            return max(values) if values else None
    except OSError as exc:
        logger.warning("Could not parse results CSV %s: %s", csv_path, exc)
        return None


def _mtime(path: Path) -> float:
    """Return the file's mtime, or ``0.0`` if it cannot be stat'ed."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def scan_runs() -> dict | None:
    """Return the highest-mAP50 trained model found under ``runs/`` / ``experiments/``.

    Sources, in order: ``runs/*/results.csv`` (per-run Ultralytics CSV, max of
    column ``metrics/mAP50(B)`` with fallback ``mAP50(B)``), ``runs/eval/results.json``
    (``metrics.mAP50``), and ``experiments/*/fold_0_best.pt`` (unscored; used only
    when nothing scored exists, newest mtime wins). Returns ``None`` if nothing
    is found.
    """
    candidates: list[dict] = []
    for csv_path in sorted((PROJECT_ROOT / "runs").glob("*/results.csv")):
        mAP50 = _max_csv_mAP50(csv_path)
        if mAP50 is None:
            continue
        candidates.append(
            {
                "model": str(csv_path.parent / "weights" / "best.pt"),
                "metrics": {"mAP50": mAP50},
                "experiment": csv_path.parent.name,
            }
        )

    eval_json = PROJECT_ROOT / "runs" / "eval" / "results.json"
    if eval_json.exists():
        try:
            data = json.loads(eval_json.read_text(encoding="utf-8"))
            mAP50 = (data.get("metrics") or {}).get("mAP50")
            if mAP50 is not None:
                candidates.append(
                    {
                        "model": str(data.get("model", eval_json)),
                        "metrics": {"mAP50": float(mAP50)},
                        "experiment": "eval",
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("Could not parse eval results %s: %s", eval_json, exc)

    unscored = [
        {
            "model": str(pt_path),
            "metrics": {},
            "experiment": pt_path.parent.name,
        }
        for pt_path in sorted((PROJECT_ROOT / "experiments").glob("*/fold_0_best.pt"))
    ]

    if candidates:
        return max(candidates, key=lambda c: c["metrics"]["mAP50"])
    if unscored:
        return max(unscored, key=lambda c: (_mtime(Path(c["model"])), c["model"]))
    return None


def resolve_best() -> Path:
    """Resolve the best available model: registry first, then a mAP50 scan, else raise."""
    best = get_best()
    if best is not None:
        model = best.get("model")
        if model and Path(model).is_file():
            return Path(model)

    scanned = scan_runs()
    if scanned is not None and scanned.get("model"):
        return Path(scanned["model"])

    raise FileNotFoundError(
        "No trained model found. Train one first: python scripts/train.py "
        "experiment=yolo26m (searched models/best.json, runs/*/results.csv, "
        "runs/eval/results.json, experiments/*/fold_0_best.pt)"
    )
