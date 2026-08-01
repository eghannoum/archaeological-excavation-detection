"""K-fold cross-validation wrapper for the archaeological hole detection pipeline.

Splits the dataset at the **parent-scene** level to prevent spatial leakage
between train and validation folds.  For each fold it creates a temporary
dataset YAML, launches ``scripts/train.py`` as a subprocess, then aggregates
cross-fold metrics (mean ± std).

Usage
-----
    python scripts/train_cv.py experiment=yolo26n
    python scripts/train_cv.py experiment=yolo26n training.epochs=50
    python scripts/train_cv.py experiment=yolo26n ++cv.n_folds=5

The script does **not** use ``@hydra.main`` — it composes the config manually
via the Hydra compose API so it can programmatically run multiple training
loops without CWD changes interfering.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_cv")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Absolute path to the project root."""
DATASET_DIR = PROJECT_ROOT / "dataset"
"""Dataset root directory."""
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"

# ---------------------------------------------------------------------------
# Scene-ID extraction
# ---------------------------------------------------------------------------

# Regex to extract the parent-scene prefix from image filenames.
# Examples of expected patterns:
#   013cc13e-img_16_bl.png   →  "013cc13e-img_16"
#   M_51_B_season2_2022_8_13_0.jpg →  "M_51_B_season2_2022_8_13"
#
# The pattern strips the last underscore-separated component (quadrant /
# index) and the extension.
_SCENE_RE = re.compile(r"^(.+)_[^_]+\.\w+$")


def extract_parent_scene(filename: str) -> str:
    """Extract the **parent scene** identifier from an image filename.

    The parent scene groups all sub-images that originate from the same
    archaeological aerial photograph.  By splitting at this level we prevent
    spatial leakage — sub-images from the same parent scene never appear in
    both train and validation folds.

    Parameters
    ----------
    filename : str
        Image filename (e.g. ``"013cc13e-img_16_bl.png"``).

    Returns
    -------
    str
        Parent scene identifier (e.g. ``"013cc13e-img_16"``).

    Examples
    --------
    >>> extract_parent_scene("013cc13e-img_16_bl.png")
    '013cc13e-img_16'
    >>> extract_parent_scene("M_51_B_season2_2022_8_13_0.jpg")
    'M_51_B_season2_2022_8_13'
    """
    m = _SCENE_RE.match(filename)
    if m:
        return m.group(1)
    # Fallback: return filename without extension (shouldn't happen with
    # well-formed names — keep the pipeline robust).
    return Path(filename).stem


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------


def get_all_images() -> list[Path]:
    """Return all image paths in both ``train`` and ``val`` directories."""
    images: list[Path] = []
    for d in [IMAGES_TRAIN, IMAGES_VAL]:
        if d.exists():
            images.extend(sorted(d.iterdir()))
    return images


def group_by_parent_scene(
    image_paths: list[Path],
) -> dict[str, list[Path]]:
    """Group image paths by their parent-scene identifier.

    Returns
    -------
    dict
        ``{parent_scene: [list of image Paths]}``
    """
    groups: dict[str, list[Path]] = {}
    for img_path in image_paths:
        scene = extract_parent_scene(img_path.name)
        groups.setdefault(scene, []).append(img_path)
    return groups


# ---------------------------------------------------------------------------
# Fold split
# ---------------------------------------------------------------------------


def create_fold_splits(
    scene_groups: dict[str, list[Path]],
    n_folds: int,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[list[Path], list[Path]]]:
    """Split parent-scene groups into ``n_folds`` train/val folds.

    Each fold uses ``n_folds - 1`` scene groups for training and 1 held-out
    group for validation.  All images belonging to a parent scene are kept
    together to avoid spatial leakage.

    Parameters
    ----------
    scene_groups
        ``{scene_id: [image_paths]}`` mapping.
    n_folds
        Number of folds.
    shuffle
        Randomize scene order before folding.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    list of (train_paths, val_paths) tuples, one per fold.
    """
    scenes = list(scene_groups.keys())
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(scenes)

    # Split scenes into n_folds roughly equal-sized groups
    fold_scenes: list[list[str]] = [[] for _ in range(n_folds)]
    for i, scene in enumerate(scenes):
        fold_scenes[i % n_folds].append(scene)

    folds: list[tuple[list[Path], list[Path]]] = []
    for val_idx in range(n_folds):
        train_scenes: list[str] = []
        val_scenes = fold_scenes[val_idx]
        for fold_idx, scenes_in_fold in enumerate(fold_scenes):
            if fold_idx != val_idx:
                train_scenes.extend(scenes_in_fold)

        train_paths: list[Path] = []
        for s in train_scenes:
            train_paths.extend(scene_groups[s])

        val_paths: list[Path] = []
        for s in val_scenes:
            val_paths.extend(scene_groups[s])

        folds.append((train_paths, val_paths))

    return folds


# ---------------------------------------------------------------------------
# Fold data YAML writer
# ---------------------------------------------------------------------------


def _write_image_list(file_path: Path, image_paths: list[Path]) -> None:
    """Write a list of absolute image paths to a text file, one per line.

    Ultralytics reads these files when the data YAML ``train:`` / ``val:``
    keys point to ``.txt`` files.  Labels are resolved by replacing the
    ``images`` segment in the path with ``labels`` and the extension with
    ``.txt``.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for img_path in image_paths:
            f.write(img_path.resolve().as_posix() + "\n")


def write_fold_data_yaml(
    fold_dir: Path,
    train_paths: list[Path],
    val_paths: list[Path],
    nc: int = 1,
    names: list[str] | None = None,
) -> Path:
    """Create a fold-specific data YAML and accompanying image-list text files.

    Parameters
    ----------
    fold_dir
        Output directory for the fold (created if it doesn't exist).
    train_paths
        Image paths for the training set.
    val_paths
        Image paths for the validation set.
    nc
        Number of classes (default 1 — hole).
    names
        Class name list (default ``["hole"]``).

    Returns
    -------
    Path
        Absolute path to the generated ``data.yaml``.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    data_yaml = fold_dir / "data.yaml"

    _write_image_list(train_txt, train_paths)
    _write_image_list(val_txt, val_paths)

    if names is None:
        names = ["hole"]

    yaml_content = {
        "train": train_txt.resolve().as_posix(),
        "val": val_txt.resolve().as_posix(),
        "nc": nc,
        "names": names,
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        OmegaConf.save(yaml_content, f)

    logger.info("Wrote fold data YAML: %s", data_yaml)
    logger.info("  Train images: %d", len(train_paths))
    logger.info("  Val images:   %d", len(val_paths))

    return data_yaml.resolve()


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------


def query_fold_metrics(
    experiment_name: str,
    fold: int,
) -> dict[str, float]:
    """Query MLflow for the final metrics logged by the fold training run.

    This function uses the ``MlflowClient`` to search for runs matching the
    given experiment name and ``cv_fold`` tag, then returns the latest metric
    values.

    Parameters
    ----------
    experiment_name
        MLflow experiment name (default: ``cfg.experiment.name``).
    fold
        The fold number (used to filter by ``cv_fold`` tag).

    Returns
    -------
    dict
        ``{metric_name: latest_value}``.
    """
    try:
        import mlflow
        from mlflow.entities import Run

        from scripts.mlflow_utils import TRACKING_URI

        # Use the shared tracking URI — same SQLite DB as scripts/train.py
        mlflow.set_tracking_uri(TRACKING_URI)

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("MLflow experiment '%s' not found", experiment_name)
            return {}

        runs: list[Run] = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.cv_fold = '{fold}'",
            order_by=["attributes.start_time desc"],
            max_results=1,
        )
        if not runs:
            logger.warning("No MLflow run found for fold %d", fold)
            return {}

        return dict(runs[0].data.metrics)
    except Exception as e:
        logger.warning("Failed to query MLflow metrics for fold %d: %s", fold, e)
        return {}


def aggregate_fold_metrics(
    all_fold_metrics: list[dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Compute mean ± std across folds for each metric.

    Parameters
    ----------
    all_fold_metrics
        List of per-fold metric dicts (one entry per fold).

    Returns
    -------
    dict
        ``{metric_name: (mean, std)}``.
    """
    if not all_fold_metrics:
        return {}

    # Collect all metric keys
    all_keys: set = set()
    for m in all_fold_metrics:
        all_keys.update(m.keys())

    aggregated: dict[str, tuple[float, float]] = {}
    for key in sorted(all_keys):
        values = [m.get(key, float("nan")) for m in all_fold_metrics]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated[key] = (float(np.mean(values)), float(np.std(values)))
    return aggregated


# ---------------------------------------------------------------------------
# CV entrypoint
# ---------------------------------------------------------------------------


def run_cv(
    experiment_override: str,
    extra_overrides: list[str] | None = None,
    *,
    n_folds: int = 3,
    cleanup: bool = True,
    epochs: int = 100,
    cv_seed: int = 42,
    cv_shuffle: bool = True,
) -> dict[str, tuple[float, float]]:
    """Run k-fold cross-validation.

    Parameters
    ----------
    experiment_override
        Experiment name override (e.g. ``"yolo26n"``).
    extra_overrides
        Additional Hydra overrides to pass to each training run.
    n_folds
        Number of folds (default 3).
    cleanup
        Remove fold data files after completion.
    epochs
        Number of training epochs (overrides config).
    cv_seed
        RNG seed for the parent-scene fold split (from ``cv.random_seed``).
    cv_shuffle
        Whether to randomize scene order before folding (from ``cv.shuffle``).

    Returns
    -------
    dict
        ``{metric_name: (mean, std)}`` aggregated across folds.
    """
    # --- 1. Parse dataset into parent-scene groups ---
    logger.info("Scanning dataset images...")
    all_images = get_all_images()
    if not all_images:
        logger.error("No images found in %s or %s", IMAGES_TRAIN, IMAGES_VAL)
        return {}

    scene_groups = group_by_parent_scene(all_images)
    n_scenes = len(scene_groups)
    n_images = len(all_images)
    logger.info(
        "Found %d images grouped into %d parent scenes (%d folds)",
        n_images,
        n_scenes,
        n_folds,
    )

    if n_scenes < n_folds:
        logger.error(
            "Number of parent scenes (%d) is less than n_folds (%d) — "
            "cannot create valid CV splits.",
            n_scenes,
            n_folds,
        )
        return {}

    # --- 2. Create fold splits ---
    folds = create_fold_splits(
        scene_groups,
        n_folds=n_folds,
        shuffle=cv_shuffle,
        seed=cv_seed,
    )

    # --- 3. Run training for each fold ---
    train_script = PROJECT_ROOT / "scripts" / "train.py"
    if not train_script.exists():
        logger.error("Training script not found: %s", train_script)
        return {}

    fold_results: list[dict[str, float]] = []
    base_overrides = extra_overrides or []

    for fold_idx, (train_paths, val_paths) in enumerate(folds):
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold_idx + 1, n_folds)
        logger.info("  Train images: %d  |  Val images: %d", len(train_paths), len(val_paths))
        logger.info("=" * 60)

        # Create fold data YAML
        fold_dir = DATASET_DIR / f"fold_{fold_idx}"
        write_fold_data_yaml(fold_dir, train_paths, val_paths)

        # Build subprocess command
        cmd = [
            sys.executable,
            str(train_script),
            f"experiment={experiment_override}",
            f"training.epochs={epochs}",
            f"++cv.n_folds={n_folds}",
            f"++fold={fold_idx}",
            "++dry_run=false",
        ] + base_overrides

        logger.info("Launching fold %d: %s", fold_idx, " ".join(cmd))

        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # pass output through to console
            text=True,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            logger.error(
                "Fold %d FAILED (return code=%d) — skipping metric aggregation for this fold",
                fold_idx,
                result.returncode,
            )
            fold_results.append({})
            continue

        logger.info(
            "Fold %d completed in %.1f s (return code=%d)", fold_idx, elapsed, result.returncode
        )

        # Collect metrics from MLflow
        fold_metrics = query_fold_metrics(experiment_override, fold_idx)
        logger.info("Fold %d metrics: %s", fold_idx, fold_metrics)
        fold_results.append(fold_metrics)

    # --- 4. Aggregate ---
    aggregated = aggregate_fold_metrics(fold_results)

    # --- 5. Print summary ---
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS ({n_folds}-fold, experiment={experiment_override})")
    print(f"{'=' * 60}")
    if aggregated:
        for key, (mean_val, std_val) in aggregated.items():
            print(f"  {key:<25s}  {mean_val:.4f} ± {std_val:.4f}")
    else:
        print("  No metrics aggregated — check MLflow run logs.")
    print(f"{'=' * 60}\n")

    # --- 6. Cleanup ---
    if cleanup:
        for fold_idx in range(n_folds):
            fold_dir = DATASET_DIR / f"fold_{fold_idx}"
            if fold_dir.exists():
                shutil.rmtree(fold_dir)
                logger.info("Cleaned up: %s", fold_dir)

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    We take the first non-``--`` argument as the experiment name override for
    convenience (positional), plus optional ``--folds`` / ``--epochs`` flags.
    All remaining unknown arguments are forwarded as Hydra overrides to each
    fold's ``train.py`` invocation.
    """
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation for archaeological hole detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        type=str,
        nargs="?",
        default="yolo26n",
        help="Experiment name (e.g. yolo26n, yolo26m, yolov8m) [default: yolo26n]",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Number of CV folds [default: from config, typically 3]",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs per fold [default: from config, typically 100]",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep fold data files after completion",
    )
    # Use parse_known_args so known flags are consumed and remaining args
    # (Hydra overrides) are captured without interference.
    args, remaining = parser.parse_known_args(argv)
    args.hydra_overrides = remaining
    return args


def main() -> None:
    """CLI entrypoint for cross-validation."""
    args = parse_args()

    # Strip "experiment=" prefix if present (user may pass Hydra-style)
    experiment_name: str = args.experiment
    if experiment_name.startswith("experiment="):
        experiment_name = experiment_name[len("experiment=") :]
        logger.info("Stripped 'experiment=' prefix — using experiment='%s'", experiment_name)

    # Build extra Hydra overrides list (excluding the consumed args)
    extra_overrides: list[str] = list(args.hydra_overrides)

    n_folds = args.folds  # may be None — will use compose API default
    epochs = args.epochs

    # Initialise Hydra to read default config values (n_folds, epochs, etc.)
    # without using @hydra.main (which would change CWD).
    configs_dir = PROJECT_ROOT / "configs"
    try:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(configs_dir)):
            cfg = compose(
                config_name="default",
                overrides=[f"experiment={experiment_name}"] + extra_overrides,
                return_hydra_config=False,
            )
    except Exception as e:
        logger.warning(
            "Hydra compose failed (%s) — using CLI defaults for n_folds/epochs",
            e,
        )
        cfg = None

    if n_folds is None:
        try:
            n_folds = cfg.cv.n_folds if cfg else 3
        except Exception:
            n_folds = 3
        logger.info("Using n_folds=%d", n_folds)

    if epochs is None:
        try:
            epochs = cfg.training.epochs if cfg else 100
        except Exception:
            epochs = 100
        logger.info("Using epochs=%d", epochs)

    cv_seed = getattr(cfg.cv, "random_seed", 42) if cfg else 42
    cv_shuffle = getattr(cfg.cv, "shuffle", True) if cfg else True
    logger.info("Using cv random_seed=%d, shuffle=%s", cv_seed, cv_shuffle)

    # Run CV
    run_cv(
        experiment_override=experiment_name,
        extra_overrides=extra_overrides,
        n_folds=n_folds,
        cleanup=not args.no_cleanup,
        epochs=epochs,
        cv_seed=cv_seed,
        cv_shuffle=cv_shuffle,
    )


if __name__ == "__main__":
    main()
