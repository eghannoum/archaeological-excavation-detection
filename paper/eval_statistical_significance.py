"""Compute confidence intervals and pairwise significance tests for all 9 models.

Reads 3-fold CV results from experiments/*/results.json, computes:
- 95% bootstrap confidence intervals for mAP50 and mAP50-95
- Pairwise bootstrap significance tests (p-values)
- Generates a significance matrix heatmap saved to docs/significance/
- Logs all metrics to MLflow

Usage
-----
    python paper/eval_statistical_significance.py                           # all models
    python paper/eval_statistical_significance.py --models yolo26n yolo26m  # subset
    python paper/eval_statistical_significance.py --n-bootstrap 5000        # custom
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = _PROJECT_ROOT
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "significance"

ALL_MODELS = [
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "yolov8m", "yolo11m", "faster_rcnn", "detr",
]

# Models that lack mAP50-95 in their results.json
MODELS_WITHOUT_MAP95 = {"detr"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_significance")


# ---------------------------------------------------------------------------
# Robust results.json parser
# ---------------------------------------------------------------------------


def extract_fold_metrics(data: dict) -> Tuple[List[float], List[float]]:
    """Extract per-fold mAP50 and mAP50-95 from a results.json dict.

    Handles all known format variations across the 9 models:

    - ``fold_results[].metrics.val/mAP50``  (yolo26n, yolo26s, faster_rcnn, detr)
    - ``fold_results[].metrics.mAP50``      (yolo26l, yolo26x)
    - ``folds[].metrics.mAP50``             (yolo26m)
    - ``folds[].mAP50``                     (yolov8m, yolo11m)

    Returns
    -------
    map50_values : list of float
    map95_values : list of float  (may be empty if not available)
    """
    map50: List[float] = []
    map95: List[float] = []

    # Strategy 1: fold_results[].metrics.*
    fold_results = data.get("fold_results", [])
    if fold_results:
        for fold in fold_results:
            metrics = fold.get("metrics", {})
            # Try val/ prefix first
            m50 = metrics.get("val/mAP50", metrics.get("mAP50"))
            m95 = metrics.get("val/mAP50-95", metrics.get("mAP50-95"))
            if m50 is not None:
                map50.append(float(m50))
            if m95 is not None:
                map95.append(float(m95))

    # Strategy 2: folds[].metrics.mAP50
    if not map50:
        folds = data.get("folds", [])
        if folds:
            for fold in folds:
                # Nested metrics dict
                if "metrics" in fold and isinstance(fold["metrics"], dict):
                    m50 = fold["metrics"].get("mAP50")
                    m95 = fold["metrics"].get("mAP50-95")
                else:
                    # Flat keys (yolov8m, yolo11m)
                    m50 = fold.get("mAP50")
                    m95 = fold.get("mAP50-95")
                if m50 is not None:
                    map50.append(float(m50))
                if m95 is not None:
                    map95.append(float(m95))

    # Strategy 3: cross_fold or cross_fold_metrics already has summary
    # We still prefer per-fold values for bootstrapping.
    # If no per-fold found, try to reconstruct from cross_fold summary
    if not map50:
        for key in ("cross_fold", "cross_fold_metrics"):
            summary = data.get(key, {})
            if "mAP50" in summary:
                val = summary["mAP50"]
                if isinstance(val, dict):
                    mean_val = val.get("mean")
                else:
                    mean_val = val
                if mean_val is not None:
                    # We can't bootstrap from a single mean; log a warning
                    logger.warning(
                        "No per-fold data for mAP50 — using cross_fold mean=%.4f only",
                        mean_val,
                    )
                    map50.append(float(mean_val))
            if "mAP50-95" in summary:
                val = summary["mAP50-95"]
                if isinstance(val, dict):
                    mean_val = val.get("mean")
                else:
                    mean_val = val
                if mean_val is not None:
                    map95.append(float(mean_val))

    return map50, map95


def load_all_results(
    models: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load results.json for all (or selected) models.

    Returns
    -------
    dict
        ``{model_name: {"map50": [fold0, fold1, fold2], "map95": [...]}}``
    """
    model_list = models or ALL_MODELS
    all_results: Dict[str, Dict[str, Any]] = {}

    for model_name in model_list:
        results_path = EXPERIMENTS_DIR / model_name / "results.json"
        if not results_path.exists():
            logger.warning("results.json not found for %s — skipping", model_name)
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        map50, map95 = extract_fold_metrics(data)

        if not map50:
            logger.warning("No mAP50 values extracted for %s — skipping", model_name)
            continue

        all_results[model_name] = {
            "map50": map50,
            "map95": map95,
        }
        logger.info(
            "Loaded %s: mAP50=%s, mAP50-95=%s",
            model_name,
            [f"{v:.4f}" for v in map50],
            [f"{v:.4f}" for v in map95] if map95 else "N/A",
        )

    return all_results


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------


def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array of per-fold metric values
    n_bootstrap : number of bootstrap resamples
    confidence : confidence level (default 0.95)
    seed : RNG seed for reproducibility

    Returns
    -------
    mean, ci_lower, ci_upper, std_of_bootstrap_means
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = sample.mean()

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    mean_val = float(values.mean())
    boot_std = float(bootstrap_means.std())

    return mean_val, ci_lower, ci_upper, boot_std


# ---------------------------------------------------------------------------
# Pairwise bootstrap significance test
# ---------------------------------------------------------------------------


def paired_bootstrap_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test for difference in means.

    For each bootstrap iteration:
    1. Resample fold indices with replacement (same indices for both models)
    2. Compute mean difference
    3. p-value = 2 * min(P(diff >= 0), P(diff <= 0))

    Parameters
    ----------
    values_a, values_b : per-fold metric arrays (same length)
    n_bootstrap : number of resamples
    seed : RNG seed

    Returns
    -------
    p-value (two-sided)
    """
    rng = np.random.default_rng(seed)
    n = len(values_a)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        diff = values_a[indices].mean() - values_b[indices].mean()
        diffs[i] = diff

    prop_ge_zero = (diffs >= 0).mean()
    prop_le_zero = (diffs <= 0).mean()
    p_value = 2.0 * min(prop_ge_zero, prop_le_zero)

    return float(p_value)


# ---------------------------------------------------------------------------
# Significance matrix computation
# ---------------------------------------------------------------------------


def compute_significance_matrix(
    all_results: Dict[str, Dict[str, List[float]]],
    metric: str = "map50",
    n_bootstrap: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute pairwise p-value matrix and mean-difference matrix.

    Parameters
    ----------
    all_results : ``{model: {"map50": [...], "map95": [...]}}``
    metric : which metric to test ("map50" or "map95")
    n_bootstrap : bootstrap iterations for each test

    Returns
    -------
    p_matrix : (n_models, n_models) p-values
    diff_matrix : (n_models, n_models) mean differences
    model_names : list of model names (ordered)
    """
    model_names = sorted(all_results.keys())
    n_models = len(model_names)
    p_matrix = np.ones((n_models, n_models), dtype=np.float64)
    diff_matrix = np.zeros((n_models, n_models), dtype=np.float64)

    for i in range(n_models):
        vals_i = np.array(all_results[model_names[i]][metric])
        for j in range(n_models):
            if i == j:
                p_matrix[i, j] = 1.0
                diff_matrix[i, j] = 0.0
            elif j > i:
                vals_j = np.array(all_results[model_names[j]][metric])
                p_val = paired_bootstrap_test(vals_i, vals_j, n_bootstrap=n_bootstrap, seed=42)
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
                diff_matrix[i, j] = vals_i.mean() - vals_j.mean()
                diff_matrix[j, i] = vals_j.mean() - vals_i.mean()
            # j < i already filled

    return p_matrix, diff_matrix, model_names


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------


def plot_significance_heatmap(
    p_matrix: np.ndarray,
    diff_matrix: np.ndarray,
    model_names: List[str],
    metric_name: str = "mAP50",
    output_path: Optional[Path] = None,
) -> None:
    """Generate a significance matrix heatmap.

    Lower triangle: p-values (colored by significance, red = significant)
    Upper triangle: mean differences (model_i - model_j)
    Diagonal: model names
    """
    n = len(model_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Build annotation matrix: lower = p-values, upper = diffs
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = model_names[i]
            elif i > j:
                # Lower triangle: p-value
                p = p_matrix[i, j]
                annot[i, j] = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
                if p < 0.05:
                    annot[i, j] += "*"
            else:
                # Upper triangle: mean difference
                d = diff_matrix[i, j]
                annot[i, j] = f"{d:+.4f}"

    # Mask upper triangle for p-value coloring
    # We use p_matrix values for the heatmap colors in the lower triangle
    # and set the upper triangle to NaN for color mapping
    display_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                display_matrix[i, j] = 0.5  # neutral for diagonal
            elif i > j:
                display_matrix[i, j] = p_matrix[i, j]
            else:
                display_matrix[i, j] = np.nan  # no color for upper

    # Custom colormap: green (not significant) -> yellow -> red (significant)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    # Plot heatmap
    mask_upper = np.zeros_like(display_matrix, dtype=bool)
    mask_upper[np.triu_indices(n, k=0)] = True  # mask diagonal + upper

    sns.heatmap(
        display_matrix,
        mask=~mask_upper,  # show lower triangle
        cmap=cmap,
        vmin=0,
        vmax=0.1,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "p-value (lower triangle)", "shrink": 0.6},
        ax=ax,
    )

    # Add the upper triangle (mean differences) as a second layer
    upper_display = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i < j:
                upper_display[i, j] = diff_matrix[i, j]

    # No mask for upper — show all upper triangle cells
    mask_lower = np.zeros_like(upper_display, dtype=bool)
    mask_lower[np.tril_indices(n, k=-1)] = True
    mask_lower[np.diag_indices(n)] = True

    # Custom colormap for differences (blue-white-red)
    diff_cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        upper_display,
        mask=mask_lower,
        cmap=diff_cmap,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Mean difference (upper triangle)", "shrink": 0.6},
        ax=ax,
        alpha=0.85,
    )

    ax.set_title(
        f"Pairwise Significance Tests — {metric_name}\n"
        f"Lower: p-values (green=not sig, red=sig, * = p<0.05)  |  "
        f"Upper: mean difference (blue=row<col, red=row>col)",
        fontsize=11,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap: %s", output_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------


def print_summary_table(
    all_results: Dict[str, Dict[str, List[float]]],
    n_bootstrap: int = 1000,
) -> None:
    """Print a formatted summary table with CIs and significance vs best."""
    # Sort by mean mAP50 descending
    model_stats = []
    for model_name, data in all_results.items():
        vals50 = np.array(data["map50"])
        mean50, ci_lo50, ci_hi50, _ = compute_bootstrap_ci(vals50, n_bootstrap=n_bootstrap)

        vals95 = data.get("map95", [])
        if vals95:
            vals95_arr = np.array(vals95)
            mean95, ci_lo95, ci_hi95, _ = compute_bootstrap_ci(vals95_arr, n_bootstrap=n_bootstrap)
        else:
            mean95, ci_lo95, ci_hi95 = None, None, None

        model_stats.append({
            "name": model_name,
            "mean50": mean50,
            "ci50": (ci_lo50, ci_hi50),
            "mean95": mean95,
            "ci95": (ci_lo95, ci_hi95) if ci_lo95 is not None else None,
            "n_folds": len(vals50),
        })

    # Sort by mAP50 descending
    model_stats.sort(key=lambda x: x["mean50"], reverse=True)

    # Find best model
    best = model_stats[0]

    # Print table
    sep = "=" * 110
    thin_sep = "-" * 110
    print(f"\n{sep}")
    print("STATISTICAL SIGNIFICANCE ANALYSIS — 95% Bootstrap Confidence Intervals")
    print(sep)
    header = (
        f"{'Model':<15} "
        f"{'mAP50':>8} {'95% CI':>18} "
        f"{'mAP50-95':>10} {'95% CI':>18} "
        f"{'Folds':>5} "
        f"{'Sig vs Best?':>15}"
    )
    print(header)
    print(thin_sep)

    for s in model_stats:
        ci50_str = f"[{s['ci50'][0]:.4f}, {s['ci50'][1]:.4f}]"
        if s["mean95"] is not None and s["ci95"] is not None:
            ci95_str = f"[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"
            map95_str = f"{s['mean95']:.4f}"
        else:
            ci95_str = "N/A"
            map95_str = "N/A"

        # Test vs best model
        if s["name"] == best["name"]:
            sig_str = "(BEST)"
        else:
            p = paired_bootstrap_test(
                np.array(all_results[s["name"]]["map50"]),
                np.array(all_results[best["name"]]["map50"]),
                n_bootstrap=n_bootstrap,
            )
            if p < 0.05:
                sig_str = f"p={p:.4f} **"
            else:
                sig_str = f"p={p:.4f} ns"

        print(
            f"{s['name']:<15} "
            f"{s['mean50']:>8.4f} {ci50_str:>18} "
            f"{map95_str:>10} {ci95_str:>18} "
            f"{s['n_folds']:>5d} "
            f"{sig_str:>15}"
        )

    print(thin_sep)
    print(f"  * = significant at p<0.05 (paired bootstrap test, {n_bootstrap} resamples)")
    print(f"  ns = not significant")
    print(f"  n_folds = {model_stats[0]['n_folds']} per model (3-fold CV)")
    print(sep)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_to_mlflow(
    all_results: Dict[str, Dict[str, List[float]]],
    n_bootstrap: int = 1000,
) -> None:
    """Log significance analysis results to MLflow."""
    try:
        import mlflow
        from scripts.mlflow_utils import TRACKING_URI

        mlflow.set_tracking_uri(TRACKING_URI)
        experiment = mlflow.set_experiment("statistical_significance")

        tags = {
            "analysis_type": "pairwise_significance",
            "n_models": str(len(all_results)),
            "n_bootstrap": str(n_bootstrap),
            "cv_folds": "3",
        }

        run = mlflow.start_run(
            run_name="significance_analysis",
            tags=tags,
        )

        # Log params
        mlflow.log_params({
            "n_models": len(all_results),
            "n_bootstrap": n_bootstrap,
            "cv_folds": 3,
            "confidence_level": 0.95,
            "significance_threshold": 0.05,
            "models": ", ".join(sorted(all_results.keys())),
        })

        # Log per-model metrics
        for model_name, data in all_results.items():
            prefix = f"model/{model_name}"
            vals50 = np.array(data["map50"])
            mean50, ci_lo50, ci_hi50, boot_std50 = compute_bootstrap_ci(
                vals50, n_bootstrap=n_bootstrap
            )

            mlflow.log_metric(f"{prefix}/mAP50_mean", mean50)
            mlflow.log_metric(f"{prefix}/mAP50_ci_lower", ci_lo50)
            mlflow.log_metric(f"{prefix}/mAP50_ci_upper", ci_hi50)
            mlflow.log_metric(f"{prefix}/mAP50_boot_std", boot_std50)

            for i, v in enumerate(vals50):
                mlflow.log_metric(f"{prefix}/mAP50_fold{i}", v)

            if data.get("map95"):
                vals95 = np.array(data["map95"])
                mean95, ci_lo95, ci_hi95, boot_std95 = compute_bootstrap_ci(
                    vals95, n_bootstrap=n_bootstrap
                )
                mlflow.log_metric(f"{prefix}/mAP50-95_mean", mean95)
                mlflow.log_metric(f"{prefix}/mAP50-95_ci_lower", ci_lo95)
                mlflow.log_metric(f"{prefix}/mAP50-95_ci_upper", ci_hi95)
                mlflow.log_metric(f"{prefix}/mAP50-95_boot_std", boot_std95)

        # Log pairwise p-values
        p_matrix, diff_matrix, model_names = compute_significance_matrix(
            all_results, metric="map50", n_bootstrap=n_bootstrap
        )
        for i, name_i in enumerate(model_names):
            for j, name_j in enumerate(model_names):
                if i < j:
                    mlflow.log_metric(
                        f"pval/{name_i}_vs_{name_j}",
                        p_matrix[i, j],
                    )
                    mlflow.log_metric(
                        f"diff/{name_i}_vs_{name_j}",
                        diff_matrix[i, j],
                    )

        # Log heatmaps as artifacts
        for png_file in OUTPUT_DIR.glob("*.png"):
            mlflow.log_artifact(str(png_file), artifact_path="significance_heatmaps")

        mlflow.end_run()
        logger.info("Logged significance analysis to MLflow")

    except Exception as e:
        logger.warning("Failed to log to MLflow: %s", e)


# ---------------------------------------------------------------------------
# Save significance report
# ---------------------------------------------------------------------------


def save_significance_report(
    all_results: Dict[str, Dict[str, List[float]]],
    n_bootstrap: int = 1000,
) -> None:
    """Save a JSON report with all computed metrics and p-values."""
    model_stats = {}
    for model_name, data in all_results.items():
        vals50 = np.array(data["map50"])
        mean50, ci_lo50, ci_hi50, boot_std50 = compute_bootstrap_ci(
            vals50, n_bootstrap=n_bootstrap
        )
        entry: Dict[str, Any] = {
            "mAP50_mean": round(mean50, 4),
            "mAP50_ci_95": [round(ci_lo50, 4), round(ci_hi50, 4)],
            "mAP50_boot_std": round(boot_std50, 4),
            "mAP50_folds": [round(v, 4) for v in vals50],
        }
        if data.get("map95"):
            vals95 = np.array(data["map95"])
            mean95, ci_lo95, ci_hi95, boot_std95 = compute_bootstrap_ci(
                vals95, n_bootstrap=n_bootstrap
            )
            entry["mAP50-95_mean"] = round(mean95, 4)
            entry["mAP50-95_ci_95"] = [round(ci_lo95, 4), round(ci_hi95, 4)]
            entry["mAP50-95_boot_std"] = round(boot_std95, 4)
            entry["mAP50-95_folds"] = [round(v, 4) for v in vals95]

        model_stats[model_name] = entry

    # Pairwise p-values
    p_matrix, diff_matrix, model_names = compute_significance_matrix(
        all_results, metric="map50", n_bootstrap=n_bootstrap
    )
    pairwise = {}
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i < j:
                pairwise[f"{name_i}_vs_{name_j}"] = {
                    "p_value": round(float(p_matrix[i, j]), 6),
                    "mean_diff": round(float(diff_matrix[i, j]), 4),
                    "significant_at_0.05": bool(p_matrix[i, j] < 0.05),
                }

    report = {
        "n_bootstrap": n_bootstrap,
        "significance_threshold": 0.05,
        "model_stats": model_stats,
        "pairwise_comparisons": pairwise,
    }

    report_path = OUTPUT_DIR / "significance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report: %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute confidence intervals and pairwise significance tests.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model names to evaluate (default: all). Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    args = parser.parse_args()

    models_to_eval = args.models if args.models else ALL_MODELS
    n_bootstrap = args.n_bootstrap

    logger.info("=" * 70)
    logger.info("Statistical Significance Analysis")
    logger.info("=" * 70)
    logger.info("Models: %s", models_to_eval)
    logger.info("Bootstrap resamples: %d", n_bootstrap)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("")

    # --- Load results ---
    all_results = load_all_results(models_to_eval)
    if not all_results:
        logger.error("No results loaded — exiting")
        sys.exit(1)

    logger.info("Loaded %d models", len(all_results))

    # --- Compute and print summary table ---
    print_summary_table(all_results, n_bootstrap=n_bootstrap)

    # --- Generate heatmaps ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # mAP50 heatmap
    p_matrix_50, diff_matrix_50, model_names_50 = compute_significance_matrix(
        all_results, metric="map50", n_bootstrap=n_bootstrap
    )
    plot_significance_heatmap(
        p_matrix_50,
        diff_matrix_50,
        model_names_50,
        metric_name="mAP50",
        output_path=OUTPUT_DIR / "significance_matrix_mAP50.png",
    )

    # mAP50-95 heatmap (only if data available)
    models_with_95 = {
        k: v for k, v in all_results.items() if v.get("map95")
    }
    if len(models_with_95) >= 2:
        p_matrix_95, diff_matrix_95, model_names_95 = compute_significance_matrix(
            models_with_95, metric="map95", n_bootstrap=n_bootstrap
        )
        plot_significance_heatmap(
            p_matrix_95,
            diff_matrix_95,
            model_names_95,
            metric_name="mAP50-95",
            output_path=OUTPUT_DIR / "significance_matrix_mAP50-95.png",
        )

    # --- Save JSON report ---
    save_significance_report(all_results, n_bootstrap=n_bootstrap)

    # --- Log to MLflow ---
    if not args.no_mlflow:
        log_to_mlflow(all_results, n_bootstrap=n_bootstrap)

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE — outputs saved to %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
