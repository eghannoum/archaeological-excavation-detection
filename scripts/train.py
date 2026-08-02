"""Main training entrypoint for the archaeological hole detection ML pipeline.

Usage
-----
    python scripts/train.py experiment=yolo26n training.epochs=100
    python scripts/train.py --info                          # dry-run: print config
    python scripts/train.py experiment=yolo26n ++info=true  # same dry-run via Hydra
    python scripts/train.py experiment=yolo26n training.epochs=1   # 1-epoch smoke test
    python scripts/train.py experiment=yolo26n augmentation=heavy
    python scripts/train.py experiment=yolo26m ablation=optimizer_sgd

Integrates Hydra config (configs/default.yaml) + MLflow tracking (scripts.mlflow_utils).
"""

from __future__ import annotations

import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so ``from scripts.xxx`` imports work
# regardless of whether the user runs ``python scripts/train.py`` or
# ``python -m scripts.train``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from scripts.mlflow_utils import finish_mlflow, init_mlflow  # noqa: E402
from scripts.model_registry import update_best_model  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = _PROJECT_ROOT
"""Absolute path to the project root (two levels up from scripts/)."""
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
"""Path to the Ultralytics-format dataset YAML."""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across torch, numpy, and Python's ``random``."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: Any) -> Any:
    """Resolve the ``training.device`` config value into an Ultralytics device.

    ``auto`` uses the first CUDA device when available, otherwise CPU.
    Explicit values (``cuda:0``, ``cuda:1``, ``cpu``) pass through unchanged.
    """
    device_str = str(device).strip().lower()
    if device_str == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device_str


def _build_ultralytics_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Translate the resolved Hydra config into Ultralytics ``model.train()`` kwargs.

    Handles paths, hyper-parameters, augmentation overrides, and Ultralytics
    training flags.  When cross-validation is active (``cv.n_folds > 1``), the
    fold-specific data YAML at ``dataset/fold_N/data.yaml`` is used instead of
    the default ``dataset/data.yaml``.
    """
    # --- Determine data YAML path (handle CV fold) ---
    n_folds = getattr(cfg, "cv", {}).get("n_folds", 1)
    fold = getattr(cfg, "fold", 0) if n_folds > 1 else 0

    if n_folds > 1:
        data_yaml = PROJECT_ROOT / "dataset" / f"fold_{fold}" / "data.yaml"
        if not data_yaml.exists():
            logger.warning(
                "Fold data YAML not found at %s — falling back to default %s",
                data_yaml,
                DATA_YAML,
            )
            data_yaml = DATA_YAML
    else:
        data_yaml = DATA_YAML

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": cfg.training.epochs,
        "batch": cfg.training.batch_size,
        "imgsz": cfg.data.image_size,
        "workers": cfg.training.workers,
        "device": _resolve_device(cfg.training.get("device", "auto")),
        "pretrained": cfg.model.pretrained,
        "optimizer": cfg.training.optimizer,
        "lr0": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "momentum": cfg.training.momentum,
        "warmup_epochs": cfg.training.scheduler.warmup_epochs,
        "cos_lr": cfg.training.scheduler.name == "cosine",
        "val": True,
        "amp": cfg.training.precision == "fp16",
        "save": True,
        "exist_ok": True,
        "project": str(PROJECT_ROOT / "runs"),
        "name": cfg.experiment.name,
    }

    # -- Early stopping --
    es = cfg.training.get("early_stopping", None)
    if es and es.get("enabled", False):
        kwargs["patience"] = es.patience
        # Ultralytics uses 'patience' for early stopping

    # -- Ultralytics augmentation overrides --
    aug = cfg.get("augmentation", None)
    if aug and aug.get("enabled", False):
        ultralytics_aug = getattr(aug, "ultralytics", None)
        if ultralytics_aug is not None:
            for k, v in OmegaConf.to_container(ultralytics_aug, resolve=True).items():
                if k not in kwargs:
                    kwargs[k] = v

        # -- Albumentations integration (light/heavy mode) --
        mode = aug.get("mode", "ultralytics")
        if mode in ("light", "heavy"):
            try:
                from scripts.augmentation import get_pipeline

                pipeline = get_pipeline(mode)
                if pipeline is not None:
                    kwargs["augmentations"] = pipeline.transforms
                    logger.info(
                        "Albumentations pipeline loaded (mode=%s, %d transforms)",
                        mode,
                        len(pipeline.transforms),
                    )
            except Exception as exc:
                logger.warning("Could not load Albumentations pipeline: %s", exc)

    # Remove None values to avoid confusing Ultralytics
    return {k: v for k, v in kwargs.items() if v is not None}


def _sanitize_metric_name(name: str) -> str:
    """Replace characters in metric names that MLflow rejects.

    MLflow metric names may only contain alphanumerics, underscores (``_``),
    dashes (``-``), periods (``.``), spaces (`` ``) and slashes (``/``).
    This function strips or replaces other characters (e.g. parentheses).
    """
    return re.sub(r"[^\w\-./ ]", "", name).strip()


def _make_epoch_callback(run_id: str):
    """Return an Ultralytics ``on_train_epoch_end`` callback that logs per-epoch
    metrics to the active MLflow run.

    The callback is invoked by Ultralytics after each training epoch.
    ``trainer.epoch`` (0-indexed) and ``trainer.metrics`` are used for the
    MLflow step and metric values respectively.
    """

    def _on_train_epoch_end(trainer) -> None:
        if not hasattr(trainer, "metrics") or not trainer.metrics:
            return
        step = getattr(trainer, "epoch", None)
        # Sanitize metric names for MLflow compatibility
        sanitized = {_sanitize_metric_name(k): v for k, v in trainer.metrics.items()}
        mlflow.log_metrics(sanitized, step=step)

    return _on_train_epoch_end


def _find_best_model(results, experiment_name: str) -> Path | None:
    """Locate the best model checkpoint produced by an Ultralytics training run.

    Search order:
    1. ``results.save_dir / weights / best.pt`` (when available)
    2. ``runs/detect/{experiment_name}/weights/best.pt`` (detection task default)
    3. ``runs/{experiment_name}/weights/best.pt`` (generic)
    4. Most recently modified ``weights/best.pt`` in the ``runs/`` tree.
    """
    # 1. Results save_dir
    if hasattr(results, "save_dir") and results.save_dir:
        candidate = Path(str(results.save_dir)) / "weights" / "best.pt"
        if candidate.exists():
            return candidate.resolve()

    # 2. Detection task subdirectory
    candidate = PROJECT_ROOT / "runs" / "detect" / experiment_name / "weights" / "best.pt"
    if candidate.exists():
        return candidate.resolve()

    # 3. Generic runs subdirectory
    candidate = PROJECT_ROOT / "runs" / experiment_name / "weights" / "best.pt"
    if candidate.exists():
        return candidate.resolve()

    # 4. Fallback: newest best.pt in runs/
    best_pts = sorted(
        (PROJECT_ROOT / "runs").rglob("weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if best_pts:
        return best_pts[0].resolve()

    return None


def _extract_final_metrics(results) -> dict[str, float]:
    """Extract a flat dict of final validation metrics from Ultralytics results.

    Tries multiple access patterns to support different Ultralytics versions:
    1. ``results.results_dict`` (preferred — flat dict with metric values)
    2. ``results.metrics`` (Ultralytics namespace/object with ``.get()``)
    3. ``results.metrics.results_dict`` (nested access)

    Returns keys like ``val/mAP50``, ``val/mAP50-95`` etc.  Returns an empty
    dict when no metrics can be extracted.
    """
    metrics: dict[str, float] = {}

    # Collect raw metrics from whichever access pattern works
    raw: dict = {}

    # Pattern 1: results.results_dict (common in Ultralytics 8.x)
    if hasattr(results, "results_dict") and results.results_dict:
        raw = dict(results.results_dict)

    # Pattern 2: results.metrics as dict-like
    elif hasattr(results, "metrics"):
        m = results.metrics
        if hasattr(m, "get"):
            # Try to get known keys
            for k in [
                "mAP50(B)",
                "mAP50-95(B)",
                "precision(B)",
                "recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(B)",
                "metrics/recall(B)",
            ]:
                v = m.get(k, None)
                if v is not None:
                    raw[k] = float(v)
        # Pattern 3: results.metrics.results_dict
        if not raw and hasattr(m, "results_dict") and m.results_dict:
            raw = dict(m.results_dict)

    # Map Ultralytics metric keys to our canonical names
    key_map = {
        "mAP50(B)": "val/mAP50",
        "mAP50-95(B)": "val/mAP50-95",
        "precision(B)": "val/precision",
        "recall(B)": "val/recall",
        "metrics/mAP50(B)": "val/mAP50",
        "metrics/mAP50-95(B)": "val/mAP50-95",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
    }
    for old_key, new_key in key_map.items():
        val = raw.get(old_key)
        if val is not None:
            metrics[new_key] = float(val)

    return metrics


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def train_yolo(cfg: DictConfig) -> tuple[Any, Path | None]:
    """Train an Ultralytics YOLO model (YOLO26, YOLOv8, YOLO11).

    Returns
    -------
    results
        The object returned by ``YOLO.train()`` (has ``.metrics`` and
        sometimes ``.save_dir``).
    best_path
        Absolute ``Path`` to ``best.pt``, or ``None`` if not found.
    """
    weights_path = PROJECT_ROOT / "models" / f"{cfg.model.name}.pt"
    if not weights_path.exists():
        logger.warning(
            "Pre-downloaded weights not found at %s — "
            "Ultralytics will download them automatically (name=%s)",
            weights_path,
            cfg.model.name,
        )
        weights_str: str = cfg.model.name
    else:
        weights_str = str(weights_path)

    logger.info("Loading YOLO model from: %s", weights_str)
    model = YOLO(weights_str)
    logger.info("Model loaded — task: %s, model: %s", model.task, cfg.model.name)

    kwargs = _build_ultralytics_kwargs(cfg)
    logger.debug("Ultralytics train kwargs: %s", kwargs)

    # Register per-epoch MLflow callback
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    if run_id:
        model.add_callback("on_train_epoch_end", _make_epoch_callback(run_id))

    results = model.train(**kwargs)
    best_path = _find_best_model(results, cfg.experiment.name)

    return results, best_path


def train_model(cfg: DictConfig) -> tuple[Any, Path | None]:
    """Factory: dispatch to the correct training function based on model type.

    Routing logic
    -------------
    * ``faster_rcnn``, ``detr`` → not trainable here; a clear message directs
      the user to the dedicated standalone trainers.
    * Everything else (``yolo26*``, ``yolov8*``, ``yolo11*``) → :func:`train_yolo`
    """
    model_name: str = str(cfg.model.name)
    if model_name in ("faster_rcnn", "detr"):
        trainer = "train_faster_rcnn.py" if model_name == "faster_rcnn" else "train_detr.py"
        logger.error(
            "%s is not trained by this script — run scripts/%s instead "
            "(reference config: experiment=%s)",
            model_name,
            trainer,
            model_name,
        )
        raise SystemExit(1)
    return train_yolo(cfg)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint — invoked by Hydra.

    The ``@hydra.main`` decorator:
    * Composes the config from ``configs/default.yaml`` + any overrides.
    * Changes Hydra's output directory to ``outputs/{config_name}/``.
    * Resolves ``${hydra:runtime.cwd}`` to the project root.
    """
    # --- Dry-run / info mode -------------------------------------------------
    if cfg.get("info", False) or cfg.get("dry_run", False):
        print(OmegaConf.to_yaml(cfg))
        return

    # --- Reproducibility -----------------------------------------------------
    set_seeds(cfg.training.get("seed", 42))

    # --- Device --------------------------------------------------------------
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info("CUDA available — %d device(s). Using: %s", device_count, device_name)
    else:
        logger.warning("CUDA NOT available — training will fall back to CPU")
        logger.warning(
            "Expected CUDA 12.8 + RTX 5070. " "Check driver and PyTorch-CUDA compatibility."
        )

    # --- Dataset check -------------------------------------------------------
    if not DATA_YAML.exists():
        logger.error("Dataset YAML not found: %s", DATA_YAML)
        logger.error("Run this script from the project root directory.")
        sys.exit(1)
    logger.info("Dataset YAML: %s", DATA_YAML)

    # --- Load augmentation config from YAML if using custom transforms ------
    # (Ultralytics handles its own augmentation internally; no action needed.)

    # --- MLflow initialisation -----------------------------------------------
    run_id = init_mlflow(cfg)
    logger.info("MLflow run started: %s", run_id)

    # --- Train ---------------------------------------------------------------
    final_metrics: dict[str, float] = {}
    model_path_str: str | None = None

    try:
        results, best_path = train_model(cfg)

        final_metrics = _extract_final_metrics(results)
        logger.info("Final validation metrics: %s", final_metrics)

        if best_path:
            model_path_str = str(best_path)
            logger.info("Best model checkpoint: %s", model_path_str)
        else:
            logger.warning("Could not locate best.pt — model artifact not logged")

        if best_path and final_metrics:
            try:
                # _extract_final_metrics() returns val/-prefixed keys (val/mAP50, ...)
                # to match the MLflow convention; the registry scores bare metric
                # names (mAP50, mAP50-95, f1_score) — normalize before registering.
                registry_metrics = {
                    key.removeprefix("val/"): value for key, value in final_metrics.items()
                }
                updated = update_best_model(
                    best_path, registry_metrics, experiment=cfg.experiment.name
                )
                logger.info("Best-model registry updated=%s (best=%s)", updated, best_path)
            except Exception as exc:  # noqa: BLE001 - registry must never crash training
                logger.warning("Could not update best-model registry: %s", exc)

    except Exception:
        logger.exception("Training failed")
        final_metrics["error"] = 1.0
        raise

    finally:
        # Always finalise the MLflow run (log whatever metrics we have)
        finish_mlflow(run_id, metrics=final_metrics, model_path=model_path_str)
        logger.info("MLflow run finished: %s", run_id)

    # --- Summary -------------------------------------------------------------
    if final_metrics:
        logger.info("Training complete — final validation metrics:")
        for key, value in final_metrics.items():
            logger.info("  %-20s  %.4f", key, value)
        if model_path_str:
            logger.info("  Best model:           %s", model_path_str)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Support --info / --dry-run flags for quick config inspection.
    # Hydra's @hydra.main does not recognise bare flags, so we translate
    # them into Hydra-compatible config overrides before calling main().
    if "--info" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--info"]
        sys.argv.append("++info=true")
    if "--dry-run" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--dry-run"]
        sys.argv.append("++dry_run=true")

    main()
