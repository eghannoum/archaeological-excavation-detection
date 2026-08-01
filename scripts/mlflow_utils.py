"""MLflow experiment tracking utilities for the archaeological hole detection pipeline.

Provides helpers to initialise MLflow runs with structured taxonomy tags, log
Hydra/OmegaConf configs as flattened params, log metrics with optional step,
and log model checkpoints as artifacts — all via **local SQLite tracking**
(no MLflow server).

Usage
-----
    from scripts.mlflow_utils import init_mlflow, finish_mlflow

    run_id = init_mlflow(cfg)
    # ... training loop, calling log_metrics() periodically ...
    finish_mlflow(run_id, metrics={"val/mAP50": 0.85}, model_path="outputs/best.pt")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Module-level state & constants
# ---------------------------------------------------------------------------

# Resolve the mlruns/ directory once at import time.
_MLRUNS_DIR = Path(__file__).resolve().parent.parent / "mlruns"
_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite-backed tracking URI stored inside mlruns/.
# MLflow 3.14.0 deprecated the legacy filesystem store; SQLite is the
# recommended local-only replacement (no server required).
_DB_PATH = _MLRUNS_DIR / "mlflow.db"
# Windows-safe sqlite URI:  sqlite:///C:/.../mlruns/mlflow.db
TRACKING_URI = f"sqlite:///{_DB_PATH.as_posix()}"

# Tell MLflow where to store artifacts (under mlruns/) so that
# ``mlflow.log_artifact()`` knows where to copy files.
os.environ.setdefault("MLFLOW_ARTIFACT_URI", _MLRUNS_DIR.as_uri())

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def init_experiment(experiment_name: str, tags: dict[str, str]) -> str:
    """Create or retrieve an MLflow experiment, set tags, return experiment ID.

    Args:
        experiment_name: Unique name for the experiment.
        tags: Key-value pairs to attach at the experiment level.

    Returns:
        The numeric experiment ID as a string.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name)

    for key, value in tags.items():
        mlflow.set_experiment_tag(key, value)

    return experiment.experiment_id


def log_config(cfg: Any) -> None:
    """Flatten and log a Hydra/OmegaConf configuration as MLflow params.

    Two representations are logged:

    1. ``full_config_yaml`` – the **complete** config as a single YAML string
       (human-readable, good for full reproducibility).
    2. Individual leaf-value params (e.g. ``training/lr``, ``data/image_size``)
       so they are **searchable / filterable** in the MLflow UI.
    """
    # --- full YAML dump (human-readable) ---
    yaml_str = OmegaConf.to_yaml(cfg)
    mlflow.log_param("full_config_yaml", yaml_str)

    # --- flattened leaf-value params (for filtering) ---
    flat = _flatten_config(cfg)
    # Keep only truthy values that are useful for search.
    # 0, False, and empty strings are still logged so that they can be filtered.
    filtered: dict[str, Any] = {}
    for k, v in flat.items():
        if v is None:
            continue  # skip null leaves
        if isinstance(v, (list, dict)) and not v:
            continue  # skip empty containers
        filtered[k] = v

    # Coerce non-primitive types to string so MLflow accepts them.
    for k, v in filtered.items():
        if not isinstance(v, (str, int, float, bool)):
            filtered[k] = str(v)

    mlflow.log_params(filtered)


def log_metrics(metrics_dict: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics, optionally at a specific step / epoch.

    Args:
        metrics_dict: Mapping of metric name → scalar value.
        step: Step or epoch number (0-indexed).  ``None`` means *current step*.
    """
    mlflow.log_metrics(metrics_dict, step=step)


def log_artifact(path: str) -> None:
    """Log a local file or directory as an MLflow artifact.

    Args:
        path: Absolute or relative filesystem path to the artifact.

    Raises:
        FileNotFoundError: The path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    mlflow.log_artifact(str(p))


def log_model_checkpoint(path: str, name: str = "model") -> None:
    """Log a model weights file as an MLflow artifact under ``checkpoints/``.

    Args:
        path: Path to the checkpoint file (e.g. ``outputs/yolo26m/weights/best.pt``).
        name: Logical sub-directory name inside the artifact store.

    Raises:
        FileNotFoundError: The checkpoint file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    mlflow.log_artifact(str(p), artifact_path=name)


def get_run_metrics(run_id: str) -> dict[str, float]:
    """Retrieve the latest value of every metric logged under a run.

    Args:
        run_id: MLflow run UUID.

    Returns:
        ``{metric_name: latest_value}`` dictionary.
    """
    client = mlflow.MlflowClient()
    run: Run = client.get_run(run_id)
    return dict(run.data.metrics)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_config(
    cfg: Any,
    *,
    parent_key: str = "",
    sep: str = "/",
) -> dict[str, Any]:
    """Recursively flatten an OmegaConf / DictConfig to a key-value dictionary.

    Leaf values that are lists are converted to comma-separated strings.
    Nested dicts produce dotted keys (e.g. ``training/lr``).

    Args:
        cfg: The resolved config object.
        parent_key: Internal – used during recursion.
        sep: Separator between key levels (default ``/`` for MLflow readability).

    Returns:
        Flattened dictionary.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    items: dict[str, Any] = {}

    def _recurse(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                if isinstance(v, dict):
                    _recurse(v, new_key)
                elif isinstance(v, (list, tuple)):
                    items[new_key] = ",".join(str(x) for x in v) if v else None
                else:
                    items[new_key] = v
        elif isinstance(obj, list):
            items[parent_key or "list"] = str(obj) if obj else None
        else:
            items[parent_key or "value"] = obj

    _recurse(resolved, parent_key)
    return items


def _extract_model_family(model_name: str) -> str:
    r"""Extract the model *family* string from a model name.

    Examples::

        yolo26n  →  yolo26
        yolov8m  →  yolov8
        yolo11m  →  yolo11
        faster_rcnn  →  faster_rcnn
        detr     →  detr
    """
    # Match: yolo<NUMBER>[nmslx]  OR  yolov<NUMBER>[nmslx]
    m = re.match(r"^(yolo\d+|yolov\d+)([nmslx])$", model_name)
    if m:
        return m.group(1)
    # Also match bare family names without scale (fallback)
    m = re.match(r"^(yolo\d+|yolov\d+)$", model_name)
    if m:
        return m.group(1)
    return model_name


def _extract_model_scale(model_name: str) -> str:
    r"""Extract the YOLO scale letter from a model name.

    Returns ``"n"``, ``"s"``, ``"m"``, ``"l"``, or ``"x"`` for YOLO variants;
    ``"m"`` for bare family names (e.g. ``"yolov8"`` → ``"m"``);
    ``"none"`` for non-YOLO models.
    """
    # Match: yolo<NUMBER>[nmslx]  OR  yolov<NUMBER>[nmslx]
    m = re.match(r"^(?:yolo\d+|yolov\d+)([nmslx])$", model_name)
    if m:
        return m.group(1)
    # Bare family name → default to "m"
    if re.match(r"^(yolo\d+|yolov\d+)$", model_name):
        return "m"
    return "none"


def _extract_augmentation_mode(cfg: Any) -> str:
    """Determine the augmentation mode string from the config."""
    try:
        aug = cfg.augmentation
        if hasattr(aug, "mode") and aug.mode:
            return str(aug.mode)
        if hasattr(aug, "enabled") and not aug.enabled:
            return "none"
        return "ultralytics"
    except Exception:
        return "unknown"


def _extract_cv_fold(cfg: Any) -> int:
    """Return the current cross-validation fold (0 when CV is disabled)."""
    try:
        n_folds = cfg.cv.n_folds
        if n_folds and n_folds > 1:
            fold = getattr(cfg, "fold", None)
            if fold is not None:
                return int(fold)
            return 0  # first fold of a multi-fold run
    except Exception:
        pass
    return 0


def _extract_experiment_type(cfg: Any) -> str:
    """Classify the run into one of: ``hpo``, ``ablation``, ``evaluation``, ``training``.

    Priority order: HPO → ablation → evaluation → training.
    """
    try:
        if OmegaConf.is_dict(cfg.hpo) and cfg.hpo.get("enabled", False):
            return "hpo"
    except Exception:
        pass

    try:
        if cfg.ablation is not None:
            return "ablation"
    except Exception:
        pass

    try:
        exp = cfg.experiment
        if hasattr(exp, "tags") and exp.tags:
            for tag in exp.tags:
                lowered = str(tag).lower().strip()
                if lowered in ("hpo", "ablation", "evaluation", "training"):
                    return lowered
    except Exception:
        pass

    return "training"


def _build_taxonomy_tags(cfg: Any) -> dict[str, str]:
    """Construct the 6 mandatory MLflow taxonomy tags from a Hydra config.

    Returns
    -------
    dict with keys:
        ``model_family``, ``model_scale``, ``cv_fold``,
        ``experiment_type``, ``augmentation``, ``image_size``
    """
    model_name: str = str(cfg.model.get("name", "unknown"))

    tags = {
        "model_family": _extract_model_family(model_name),
        "model_scale": _extract_model_scale(model_name),
        "cv_fold": str(_extract_cv_fold(cfg)),
        "experiment_type": _extract_experiment_type(cfg),
        "augmentation": _extract_augmentation_mode(cfg),
        "image_size": str(cfg.data.get("image_size", "640")),
    }

    # Optionally attach the experiment description as an MLflow note.
    try:
        desc = cfg.experiment.description
        if desc:
            tags["mlflow.note.content"] = str(desc)
    except Exception:
        pass

    return tags


# ---------------------------------------------------------------------------
# High-level pipeline integration helpers
# ---------------------------------------------------------------------------


def init_mlflow(cfg: Any) -> str:
    """Initialise a complete MLflow run from a resolved Hydra/OmegaConf config.

    Steps
    -----
    1. Set tracking URI to ``sqlite:///mlruns/mlflow.db`` (local DB, no server).
    2. Create or retrieve an experiment named after ``cfg.experiment.name``.
    3. Start a run and attach all 6 mandatory taxonomy tags.
    4. Flatten and log the full Hydra config as MLflow params.
    5. Return the run ID for later use with :func:`finish_mlflow`.

    Args:
        cfg: A resolved Hydra ``DictConfig`` (the output of ``@hydra.main``).

    Returns:
        MLflow run ID (UUID string).
    """
    # --- experiment name ---
    try:
        experiment_name = str(cfg.experiment.name)
    except Exception:
        experiment_name = "archaeological-hole-detection"

    # --- descriptive run name ---
    try:
        model_name = str(cfg.model.name)
        aug_mode = _extract_augmentation_mode(cfg)
        img_size = cfg.data.image_size
        run_name = f"{model_name}_{aug_mode}_{img_size}"
    except Exception:
        run_name = str(cfg.model.get("name", "model"))

    # --- boot MLflow ---
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    tags = _build_taxonomy_tags(cfg)
    run: Run = mlflow.start_run(run_name=run_name, tags=tags)
    run_id = run.info.run_id

    log_config(cfg)

    return run_id


def finish_mlflow(
    run_id: str,
    metrics: dict[str, float] | None = None,
    model_path: str | None = None,
) -> None:
    """Finalise an MLflow run: log final metrics, model checkpoint, and close.

    Uses ``MlflowClient`` to log data without needing to re-activate the run
    (the run is still active after :func:`init_mlflow`).

    Args:
        run_id: The run ID returned by :func:`init_mlflow`.
        metrics: Final metrics to log (e.g. ``{"val/mAP50": 0.85}``).
        model_path: Path to a model weight file to log as artifact.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    if metrics:
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)

    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        client.log_artifact(run_id, str(p), artifact_path="model")

    # End the currently active run (started by init_mlflow).
    mlflow.end_run()
