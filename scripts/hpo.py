"""Hyperparameter optimisation (HPO) via Optuna for YOLO26m archaeological hole detection.

Usage
-----
    python scripts/hpo.py                              # Run 50-trial HPO study
    python scripts/hpo.py --dry-run                    # Print search space & exit
    python scripts/hpo.py --n-jobs 2                   # Parallel trials (multi-GPU)
    python scripts/hpo.py --n-trials 25                # Override trial count
    python scripts/hpo.py experiment=yolo26n           # Use a different experiment

Architecture
------------
+---------------------------+--------------------------------------------------+
| Component                | Role                                             |
+---------------------------+--------------------------------------------------+
| :func:`run_study`        | Orchestrates the Optuna study + MLflow parent run |
| :func:`objective`        | Suggests params, trains via subprocess, returns   |
|                          | val/mAP50 (or ``-inf`` on failure)                |
| :func:`_suggest_params`  | Maps Optuna ``suggest_*`` calls to the config     |
|                          | search space (12 hyper-parameters)                |
| :func:`_build_overrides` | Translates suggested params to Hydra CLI args     |
| Subprocess ``train.py``  | Each trial runs ``scripts/train.py`` to do the    |
|                          | actual YOLO26m training                           |
+---------------------------+--------------------------------------------------+

MLflow experiment layout
------------------------
Two-level hierarchy (when ``n_jobs=1``):
    ``{experiment}-hpo/``  ← MLflow experiment
    └── ``study-{name}``   ← parent run (tags: ``experiment_type=hpo``, …)
        ├── ``trial-0``    ← nested run (params, intermediate/final metrics)
        ├── ``trial-1``    ← nested run
        └── …              ← (pruned/failed trials still logged)

Integration with ``configs/hpo/default.yaml``
----------------------------------------------
* :ref:`Sampler <sampler>` — ``TPESampler(seed=42, n_startup_trials=5, n_ei_candidates=24)``
* :ref:`Pruner <pruner>` — ``MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)``
* Direction — :attr:`~optuna.study.StudyDirection.MAXIMIZE`
* Target metric — ``val/mAP50``
"""

from __future__ import annotations

import io
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from scripts.xxx`` imports work
# regardless of invocation method.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.mlflow_utils import TRACKING_URI  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = _PROJECT_ROOT
"""Absolute path to the project root."""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
# Each entry: (name, type, range/values, default_value, config_override_key)
# The ``config_override_key`` tells _build_overrides() how to emit a Hydra
# CLI override string.

SEARCH_SPACE: List[Tuple[str, str, Any, Any, str]] = [
    # ── Optimiser hyper-parameters ──────────────────────────────────
    ("lr",           "log-uniform", [1e-4, 1e-2],      0.001,  "training.lr"),
    ("momentum",     "uniform",     [0.8, 0.95],        0.937,  "training.momentum"),
    ("weight_decay", "log-uniform", [1e-5, 1e-3],      0.0005, "training.weight_decay"),
    ("optimizer",    "categorical", ["AdamW", "SGD"],   "AdamW","training.optimizer"),

    # ── LR schedule ─────────────────────────────────────────────────
    ("warmup_epochs","int",         [0, 5],             3,      "training.scheduler.warmup_epochs"),
    ("cos_lr",       "categorical", [True, False],      True,   "__special__cos_lr"),

    # ── Ultralytics colour-space augmentation ───────────────────────
    ("hsv_h",        "uniform",     [0.0, 0.1],         0.015,  "augmentation.ultralytics.hsv_h"),
    ("hsv_s",        "uniform",     [0.0, 0.9],         0.7,    "augmentation.ultralytics.hsv_s"),
    ("hsv_v",        "uniform",     [0.0, 0.9],         0.4,    "augmentation.ultralytics.hsv_v"),

    # ── Ultralytics geometric augmentation ──────────────────────────
    ("degrees",      "uniform",     [0.0, 45.0],        0.0,    "augmentation.ultralytics.degrees"),

    # ── Ultralytics mosaic & mixup ──────────────────────────────────
    ("mosaic",       "categorical", [0.0, 0.5, 1.0],    1.0,    "augmentation.ultralytics.mosaic"),
    ("mixup",        "categorical", [0.0, 0.5, 1.0],    0.0,    "augmentation.ultralytics.mixup"),
]

# Human-readable header for dry-run mode
_DRY_RUN_TABLE_HEADER = (
    "Hyperparameter             Type            Range/Values                  Default\n"
    "------------------------------------------------------------------------------------------------------------------------"
)

# Regex to extract model family and scale from model name (e.g. "yolo26m" -> ("yolo26", "m")).
_MODEL_NAME_RE = re.compile(r"^(yolo\d+|yolov\d+)([nmslx])$")


def _parse_model_name(model_name: str) -> Tuple[str, str]:
    """Extract (family, scale) from a YOLO model name.

    Examples::
        "yolo26m"  ->  ("yolo26", "m")
        "yolov8x"  ->  ("yolov8", "x")
        "yolo11n"  ->  ("yolo11", "n")
        "unknown"  ->  ("unknown", "m")   # fallback
    """
    m = _MODEL_NAME_RE.match(model_name)
    if m:
        return m.group(1), m.group(2)
    return model_name, "m"


# Regex to extract mAP50 from Ultralytics per-epoch validation lines.
# Typical output:
#                          all        352       3876      0.856      0.782      0.845      0.612
_MAP50_RE = re.compile(
    # Handles \r and ANSI escape sequences from tqdm progress bars
    r"(?:\x1b\[K|\r)?\s*all\s+\d+\s+\d+\s+"
    r"([\d.]+(?:e[+-]?\d+)?)\s+"  # 1 — Precision
    r"([\d.]+(?:e[+-]?\d+)?)\s+"  # 2 — Recall
    r"([\d.]+(?:e[+-]?\d+)?)\s+"  # 3 — mAP50  ←  target
    r"([\d.]+(?:e[+-]?\d+)?)"     # 4 — mAP50-95
)

# Post-hoc regex: match "val/mAP50: 0.1234" in full output after proc.wait()
_FINAL_MAP50_RE = re.compile(r"val/mAP50[:\s]+([\d.]+)")

# Regex to detect per-epoch progress lines (e.g. " 29/30 …").
_EPOCH_RE = re.compile(r"^\s*(\d+)/(\d+)")


# ---------------------------------------------------------------------------
# Helper: suggest params
# ---------------------------------------------------------------------------


def _suggest_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Suggest one hyper-parameter value for each entry in :data:`SEARCH_SPACE`.

    Returns a flat dict suitable for :func:`_build_overrides`.
    """
    params: Dict[str, Any] = {}
    for name, stype, range_or_values, _default, _override_key in SEARCH_SPACE:
        if stype == "log-uniform":
            lo, hi = range_or_values
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        elif stype == "uniform":
            lo, hi = range_or_values
            params[name] = trial.suggest_float(name, lo, hi)
        elif stype == "int":
            lo, hi = range_or_values
            params[name] = trial.suggest_int(name, lo, hi)
        elif stype == "categorical":
            params[name] = trial.suggest_categorical(name, range_or_values)
        else:
            raise ValueError(f"Unknown search space type: {stype}")
    return params


def _format_range(stype: str, range_or_values: Any) -> str:
    """Pretty-print a search-space range for dry-run output."""
    if stype == "log-uniform":
        lo, hi = range_or_values
        return f"[{lo:.0e}, {hi:.0e}]"
    if stype == "uniform":
        lo, hi = range_or_values
        return f"[{lo}, {hi}]"
    if stype == "int":
        lo, hi = range_or_values
        return f"[{lo}, {hi}]"
    if stype == "categorical":
        return str(list(range_or_values))
    return str(range_or_values)


def _get_default_value(name: str) -> Any:
    """Return the default value for a parameter by name."""
    for n, _stype, _rng, default, _key in SEARCH_SPACE:
        if n == name:
            return default
    raise KeyError(f"Parameter '{name}' not found in SEARCH_SPACE")


# ---------------------------------------------------------------------------
# Helper: build Hydra override list
# ---------------------------------------------------------------------------


def _build_overrides(params: Dict[str, Any], trial_epochs: int) -> List[str]:
    """Translate a flat ``{param_name: value}`` dict into Hydra CLI overrides.

    The returned list is meant to be appended to a ``python scripts/train.py
    experiment=yolo26m …`` command.

    Special handling
    ----------------
    * ``cos_lr`` → ``training.scheduler.name``  (``True`` → ``"cosine"``,
      ``False`` → ``"linear"``)
    """
    overrides: List[str] = [
        f"training.lr={params['lr']}",
        f"training.momentum={params['momentum']}",
        f"training.weight_decay={params['weight_decay']}",
        f"training.optimizer={params['optimizer']}",
        f"training.scheduler.warmup_epochs={params['warmup_epochs']}",
        f"training.scheduler.name={'cosine' if params['cos_lr'] else 'linear'}",
        f"augmentation.ultralytics.hsv_h={params['hsv_h']}",
        f"augmentation.ultralytics.hsv_s={params['hsv_s']}",
        f"augmentation.ultralytics.hsv_v={params['hsv_v']}",
        f"augmentation.ultralytics.degrees={params['degrees']}",
        f"augmentation.ultralytics.mosaic={params['mosaic']}",
        f"augmentation.ultralytics.mixup={params['mixup']}",
        f"training.epochs={trial_epochs}",
        "training.batch_size=8",
    ]
    return overrides


# ---------------------------------------------------------------------------
# Helper: parse mAP50 from training subprocess stdout
# ---------------------------------------------------------------------------


def _parse_map50_line(line: str) -> Optional[float]:
    """Extract ``val/mAP50`` from a single stdout line if it contains one.

    Returns ``None`` when the line does not carry a validation metric.
    """
    m = _MAP50_RE.search(line)
    if m:
        return float(m.group(3))
    return None


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------


def _get_cfg_str(cfg: DictConfig, key_path: str, fallback: str = "yolo26m") -> str:
    """Get a string value from config with a fallback for missing mandatory values."""
    try:
        val = OmegaConf.select(cfg, key_path)
        return str(val) if val is not None else fallback
    except Exception:
        return fallback


def _print_search_space(cfg: DictConfig) -> None:
    """Print the search space table plus current HPO/MLflow config to stdout."""
    exp_name = _get_cfg_str(cfg, "experiment.name", "yolo26m")
    model_name = _get_cfg_str(cfg, "model.name", "yolo26m")
    model_family, model_scale = _parse_model_name(model_name)

    print()
    print("=" * 80)
    print("                     OPTUNA HYPERPARAMETER SEARCH SPACE")
    print("=" * 80)
    print()
    print(_DRY_RUN_TABLE_HEADER)
    for name, stype, range_or_values, default, _override_key in SEARCH_SPACE:
        display_name = name.ljust(26)
        display_type = stype.ljust(16)
        display_range = _format_range(stype, range_or_values).ljust(28)
        print(f"  {display_name}{display_type}{display_range}{default}")
    print()

    print("-" * 80)
    print("  OPTUNA CONFIGURATION")
    print("-" * 80)
    hpo = cfg.hpo
    print(f"  Study name:            {hpo.study_name}")
    print(f"  Direction:             {hpo.direction}")
    print(f"  Target metric:         {hpo.target_metric}")
    print(f"  n_trials:              {hpo.n_trials}")
    print(f"  Trial epochs:          {hpo.get('trial_epochs', 30)}")
    print(f"  Sampler:               TPESampler(seed={hpo.sampler.seed}, "
          f"n_startup_trials={hpo.sampler.n_startup_trials}, "
          f"n_ei_candidates={hpo.sampler.n_ei_candidates})")
    print(f"  Pruner:                MedianPruner(n_startup_trials={hpo.pruner.n_startup_trials}, "
          f"n_warmup_steps={hpo.pruner.n_warmup_steps}, "
          f"interval_steps={hpo.pruner.interval_steps})")
    print(f"  Storage:               {hpo.storage or 'in-memory'}")
    print("  Batch size:            8 (training — conservative per GPU benchmark)")
    print()

    print("-" * 80)
    print("  MLFLOW CONFIGURATION")
    print("-" * 80)
    print(f"  Tracking URI:          {TRACKING_URI}")
    print(f"  Experiment:            {exp_name}-hpo")
    print(f"  Study tags:            experiment_type=hpo, "
          f"model_family={model_family}, model_scale={model_scale}")
    print()

    print("-" * 80)
    print("  EXAMPLE TRIAL COMMAND")
    print("-" * 80)
    example_params = {name: _get_default_value(name) for name, _, _, _, _ in SEARCH_SPACE}
    example_overrides = _build_overrides(example_params, trial_epochs=hpo.get("trial_epochs", 30))
    cmd = (
        f"  python scripts/train.py experiment={exp_name} \\\n"
        + " \\\n".join(f"    {o}" for o in example_overrides)
    )
    print(cmd)
    print()

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Best-params exporter
# ---------------------------------------------------------------------------


def _export_best_params(study: optuna.study.Study, cfg: DictConfig) -> Path:
    """Write the best trial's hyper-parameters to ``configs/best_hparams.yaml``.

    The file contains both the raw Optuna params and the Hydra override keys
    that can be copy-pasted into a command line.
    """
    best = study.best_trial
    params = best.params

    # Build the Hydra-friendly "config_overrides" block
    scheduler_name = "cosine" if params.get("cos_lr", True) else "linear"

    overrides_block = {
        "training": {
            "lr": params.get("lr", 0.001),
            "momentum": params.get("momentum", 0.937),
            "weight_decay": params.get("weight_decay", 0.0005),
            "optimizer": params.get("optimizer", "AdamW"),
            "batch_size": 8,
            "scheduler": {
                "warmup_epochs": params.get("warmup_epochs", 3),
                "name": scheduler_name,
            },
        },
        "augmentation": {
            "ultralytics": {
                "hsv_h": params.get("hsv_h", 0.015),
                "hsv_s": params.get("hsv_s", 0.7),
                "hsv_v": params.get("hsv_v", 0.4),
                "degrees": params.get("degrees", 0.0),
                "mosaic": params.get("mosaic", 1.0),
                "mixup": params.get("mixup", 0.0),
            },
        },
    }

    export = {
        "best_trial_number": best.number,
        "best_val_mAP50": round(best.value, 4) if best.value is not None else None,
        "hyperparameters": dict(params),
        "config_overrides": overrides_block,
        "cli_override_example": " ".join(
            _build_overrides(dict(params), trial_epochs=cfg.hpo.get("trial_epochs", 30))
        ),
    }

    output_path = PROJECT_ROOT / "configs" / "best_hparams.yaml"
    OmegaConf.save(
        OmegaConf.create(export),
        output_path,
    )
    logger.info("Best hyper-parameters exported to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Objective function (per trial)
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    trial_epochs: int,
    study_run_id: Optional[str],
) -> float:
    """Optuna objective — train YOLO26m with suggested hyper-parameters.

    Steps
    -----
    1. :func:`_suggest_params` — sample hyper-parameters from the search space.
    2. Create an MLflow **nested run** under the parent study run (when
       ``study_run_id`` is provided, i.e. single-threaded mode).
    3. Build Hydra CLI overrides via :func:`_build_overrides`.
    4. Launch ``scripts/train.py`` as a subprocess.
    5. Stream stdout in real-time; parse ``val/mAP50`` per epoch for pruning.
    6. Log final results to MLflow, mark trial status.
    7. Return final mAP50 or ``float('-inf')`` on failure / ``optuna.TrialPruned``.

    Parameters
    ----------
    trial:
        The Optuna trial object (used for ``suggest_*`` and ``report``).
    cfg:
        The composed Hydra config (read-only; trials override via CLI).
    trial_epochs:
        Number of training epochs per trial (reduced from full 100).
    study_run_id:
        MLflow run ID of the parent study run.  When ``None`` (multi-threaded
        mode) trial-level runs are created as independent runs tagged with
        ``hpo_study_{study_name}``.
    """
    start_time = time.time()

    # --- 1. Suggest hyper-parameters ----------------------------------------
    params = _suggest_params(trial)

    # --- 2. MLflow trial run -------------------------------------------------
    parent_active = study_run_id is not None and (
        mlflow.active_run() is not None
        and mlflow.active_run().info.run_id == study_run_id
    )

    if parent_active:
        # Nested run — parent run is active in this thread (n_jobs=1 mode)
        mlflow.start_run(
            nested=True,
            run_name=f"trial-{trial.number}",
        )
        mlflow.log_params({f"params/{k}": v for k, v in params.items()})
        mlflow.set_tag("trial_status", "running")
    else:
        # Independent run — n_jobs>1 mode (separate thread/process)
        experiment_name = f"{cfg.experiment.name}-hpo"
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(
            run_name=f"trial-{trial.number}",
            tags={
                "experiment_type": "hpo",
                "model_family": _parse_model_name(cfg.model.name)[0],
                "model_scale": _parse_model_name(cfg.model.name)[1],
                "trial_number": str(trial.number),
                "trial_status": "running",
            },
        )
        mlflow.log_params({f"params/{k}": v for k, v in params.items()})

    # --- 3. Build CLI overrides ---------------------------------------------
    overrides = _build_overrides(params, trial_epochs=trial_epochs)
    experiment_override = cfg.experiment.name

    cmd = [
        sys.executable,
        "scripts/train.py",
        f"experiment={experiment_override}",
        "++dry_run=false",
        "++info=false",
    ] + [f"++{o}" for o in overrides]

    logger.info("Trial %3d — starting subprocess (epochs=%d)", trial.number, trial_epochs)
    logger.debug("  cmd: %s", " ".join(cmd))

    # --- 4. Launch subprocess & stream stdout --------------------------------
    final_mAP = float("-inf")
    epoch_counter = 0
    trial_pruned = False
    returncode = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )

        # 4a. Stream stdout, parse per-epoch mAP50, report for pruning
        _hpo_stdout = proc.stdout
        if sys.platform == "win32":
            _hpo_stdout = io.TextIOWrapper(proc.stdout.buffer, encoding="utf-8", errors="replace")
        _hpo_lines: List[str] = []  # buffer for post-hoc extraction
        for line in _hpo_stdout:
            _hpo_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()

            # Track current epoch number
            epoch_m = _EPOCH_RE.match(line)
            if epoch_m:
                epoch_counter = int(epoch_m.group(1))

            # Parse validation mAP50
            current_map = _parse_map50_line(line)
            if current_map is not None and epoch_counter > 0:
                final_mAP = max(final_mAP, current_map)
                # Report to Optuna for MedianPruner
                trial.report(current_map, step=epoch_counter)

                # Check if we should prune
                if trial.should_prune():
                    logger.info(
                        "Trial %3d — pruned at epoch %d (mAP50=%.4f)",
                        trial.number,
                        epoch_counter,
                        current_map,
                    )
                    proc.terminate()
                    proc.wait()
                    trial_pruned = True
                    raise optuna.TrialPruned()

        proc.wait()
        returncode = proc.returncode

        # Post-hoc: if streaming parsing didn't capture mAP50, try full-output extraction
        if final_mAP == float("-inf") or final_mAP <= 0:
            _full_output = "\n".join(_hpo_lines[-200:])  # last 200 lines only
            _post_m = _FINAL_MAP50_RE.search(_full_output)
            if _post_m:
                final_mAP = max(final_mAP, float(_post_m.group(1)))

        if returncode != 0 and not trial_pruned:
            logger.warning("Trial %d failed — return code %d", trial.number, returncode)
            final_mAP = float("-inf")

    except optuna.TrialPruned:
        raise  # Re-raise for Optuna to handle

    except FileNotFoundError:
        logger.error(
            "Trial %d — subprocess failed: 'scripts/train.py' not found. "
            "Run this script from the project root.",
            trial.number,
        )
        final_mAP = float("-inf")

    except Exception as exc:
        logger.exception("Trial %d — unexpected error: %s", trial.number, exc)
        final_mAP = float("-inf")

    # --- 5. Log results & duration ------------------------------------------
    duration_secs = time.time() - start_time

    try:
        if trial_pruned:
            mlflow.log_metric("duration_secs", round(duration_secs, 1))
            mlflow.set_tag("trial_status", "pruned")
        elif final_mAP == float("-inf"):
            mlflow.log_metric("duration_secs", round(duration_secs, 1))
            mlflow.set_tag("trial_status", "failed")
            mlflow.set_tag("return_code", str(returncode))
        else:
            mlflow.log_metrics({
                "val/mAP50": final_mAP,
                "duration_secs": round(duration_secs, 1),
            })
            mlflow.set_tag("trial_status", "completed")
            mlflow.set_tag("return_code", str(returncode))
            logger.info(
                "Trial %3d — mAP50=%.4f  duration=%.0fs",
                trial.number,
                final_mAP,
                duration_secs,
            )
    except Exception:
        logger.exception("Trial %d — MLflow logging failed", trial.number)

    # --- 6. End trial MLflow run --------------------------------------------
    try:
        mlflow.end_run()  # ends the trial-level run (nested or independent)
    except Exception:
        pass

    return final_mAP


# ---------------------------------------------------------------------------
# Study orchestrator
# ---------------------------------------------------------------------------


def run_study(
    cfg: DictConfig,
    *,
    dry_run: bool = False,
    n_jobs: int = 1,
    n_trials: Optional[int] = None,
) -> Optional[optuna.study.Study]:
    """Run a full Optuna hyper-parameter optimisation study.

    Parameters
    ----------
    cfg:
        The composed Hydra config.
    dry_run:
        When ``True``, print the search space and exit without training.
    n_jobs:
        Number of parallel trials.  **Default 1.**  When >1, MLflow trial runs
        are independent (not nested) since the parent run is not accessible in
        worker threads/processes.
    n_trials:
        Override the trial count from config (handy for quick smoke tests).
    """
    # ------------------------------------------------------------------
    # Dry-run mode — just print and return
    # ------------------------------------------------------------------
    if dry_run:
        _print_search_space(cfg)
        return None

    # ------------------------------------------------------------------
    # Resolve configuration values
    # ------------------------------------------------------------------
    study_name: str = cfg.hpo.study_name
    target_n_trials: int = n_trials if n_trials is not None else cfg.hpo.n_trials
    trial_epochs: int = cfg.hpo.get("trial_epochs", 30)
    experiment_name: str = cfg.experiment.name

    model_name: str = cfg.model.name  # e.g. "yolo26m"
    model_family, model_scale = _parse_model_name(model_name)

    logger.info("=" * 60)
    logger.info("HPO Study: %s", study_name)
    logger.info("  Trials:            %d", target_n_trials)
    logger.info("  Trial epochs:      %d", trial_epochs)
    logger.info("  Parallel (n_jobs): %d", n_jobs)
    logger.info("  Experiment:        %s", experiment_name)
    logger.info("  Model:             %s (family=%s, scale=%s)", model_name, model_family, model_scale)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # MLflow — study-level parent run
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(str(TRACKING_URI))
    mlflow.set_experiment(f"{experiment_name}-hpo")

    study_tags = {
        "experiment_type": "hpo",
        "model_family": model_family,
        "model_scale": model_scale,
        "study_name": study_name,
    }
    study_run = mlflow.start_run(
        run_name=f"study-{study_name}",
        tags=study_tags,
    )
    study_run_id = study_run.info.run_id
    logger.info("MLflow study run: %s", study_run_id)

    # Log search space metadata on the parent run
    mlflow.log_param("n_trials", target_n_trials)
    mlflow.log_param("trial_epochs", trial_epochs)
    mlflow.log_param("n_jobs", n_jobs)
    mlflow.log_param("sampler", f"TPESampler(seed={cfg.hpo.sampler.seed})")
    mlflow.log_param("pruner", f"MedianPruner(n_startup={cfg.hpo.pruner.n_startup_trials})")
    mlflow.log_param("search_space", OmegaConf.to_yaml(
        {name: {"type": stype, "range": _format_range(stype, rng)}
         for name, stype, rng, _default, _key in SEARCH_SPACE}
    ))

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    sampler = TPESampler(
        seed=cfg.hpo.sampler.seed,
        n_startup_trials=cfg.hpo.sampler.n_startup_trials,
        n_ei_candidates=cfg.hpo.sampler.n_ei_candidates,
    )
    pruner = MedianPruner(
        n_startup_trials=cfg.hpo.pruner.n_startup_trials,
        n_warmup_steps=cfg.hpo.pruner.n_warmup_steps,
        interval_steps=cfg.hpo.pruner.interval_steps,
    )

    storage = cfg.hpo.get("storage", None)
    load_if_exists = cfg.hpo.get("load_if_exists", False)

    study = optuna.create_study(
        study_name=study_name,
        direction=cfg.hpo.direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=load_if_exists,
    )

    # Attach helpful metadata on the study object itself
    study.set_user_attr("study_run_id", study_run_id)
    study.set_user_attr("config_yaml", OmegaConf.to_yaml(cfg))

    logger.info(
        "Optuna study created — %s (sampler=%s, pruner=%s)",
        study_name,
        type(sampler).__name__,
        type(pruner).__name__,
    )

    # ------------------------------------------------------------------
    # Optimise
    # ------------------------------------------------------------------
    try:
        study.optimize(
            lambda trial: objective(
                trial,
                cfg=cfg,
                trial_epochs=trial_epochs,
                study_run_id=study_run_id if n_jobs == 1 else None,
            ),
            n_trials=target_n_trials,
            n_jobs=n_jobs,
            callbacks=[_study_progress_callback],
        )
    except KeyboardInterrupt:
        logger.warning("Study interrupted by user — saving partial results")

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    _log_study_results(study, study_run_id)

    # ------------------------------------------------------------------
    # Export best params
    # ------------------------------------------------------------------
    if study.best_trial and study.best_trial.value is not None:
        _export_best_params(study, cfg)

    # ------------------------------------------------------------------
    # Cleanup MLflow
    # ------------------------------------------------------------------
    try:
        mlflow.end_run()  # end the study-level run
    except Exception:
        pass

    return study


def _study_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log study-wide progress after each completed trial."""
    if trial.value is not None:
        logger.info(
            "Study progress — trial %3d/%d  best mAP50=%.4f  (trial %d)",
            trial.number + 1,
            study._stop_flag if hasattr(study, "_stop_flag") else "?",
            study.best_value,
            study.best_trial.number,
        )
    else:
        logger.info(
            "Study progress — trial %3d   (pruned/failed)",
            trial.number + 1,
        )


def _log_study_results(study: optuna.study.Study, study_run_id: str) -> None:
    """Log aggregate study results to the parent MLflow run."""
    completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == TrialState.FAIL])

    logger.info("-" * 50)
    logger.info("STUDY RESULTS  --  %s", study.study_name)
    logger.info("  Completed: %d  |  Pruned: %d  |  Failed: %d", completed, pruned, failed)
    if study.best_trial and study.best_trial.value is not None:
        logger.info("  Best trial:  #%d  (mAP50=%.4f)", study.best_trial.number, study.best_trial.value)
        for key, value in study.best_trial.params.items():
            logger.info("    %s: %s", key, value)
    logger.info("-" * 50)

    # Log to MLflow parent run
    try:
        client = mlflow.MlflowClient()
        _study_metrics = {
            "study/completed_trials": completed,
            "study/pruned_trials": pruned,
            "study/failed_trials": failed,
            "study/total_trials": len(study.trials),
        }
        for _k, _v in _study_metrics.items():
            client.log_metric(study_run_id, _k, _v)
        if study.best_trial and study.best_trial.value is not None:
            client.log_metric(study_run_id, "study/best_mAP50", study.best_trial.value)
            client.log_metric(study_run_id, "study/best_trial_number", study.best_trial.number)
    except Exception as e:
        logger.warning("Failed to log study results to MLflow: %s", e)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

# Module-level globals for CLI args parsed *before* @hydra.main processes argv.
_HPO_DRY_RUN = False
_HPO_N_JOBS = 1
_HPO_N_TRIALS: Optional[int] = None


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """HPO entrypoint — invoked by ``@hydra.main`` with the composed config."""
    # Configure logging (Hydra may have changed the root logger)
    logging.basicConfig(
        level=getattr(logging, cfg.experiment.get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_study(
        cfg,
        dry_run=_HPO_DRY_RUN,
        n_jobs=_HPO_N_JOBS,
        n_trials=_HPO_N_TRIALS,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Strip HPO-specific CLI args *before* @hydra.main processes sys.argv.
    # Unrecognised args are left for Hydra to interpret as config overrides.
    _hpo_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--dry-run":
            _HPO_DRY_RUN = True
        elif sys.argv[i] == "--n-jobs" and i + 1 < len(sys.argv):
            i += 1
            _HPO_N_JOBS = int(sys.argv[i])
        elif sys.argv[i] == "--n-trials" and i + 1 < len(sys.argv):
            i += 1
            _HPO_N_TRIALS = int(sys.argv[i])
        else:
            _hpo_argv.append(sys.argv[i])
        i += 1
    sys.argv = _hpo_argv

    main()
