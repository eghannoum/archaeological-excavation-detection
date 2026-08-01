"""Generate publication-ready LaTeX and markdown ablation tables.

Queries MLflow for ablation experiment results (Image Size, Optimizer,
Augmentation) and formats them into LaTeX and markdown tables saved to
``docs/tables/ablation/``.

Baseline reference values (yolo26m, 3-fold CV mean):
    mAP50=0.404, mAP50-95=0.117, P=0.482, R=0.460

Usage
-----
    python paper/generate_ablation_tables.py
    python paper/generate_ablation_tables.py --baseline-mAP50 0.3888
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

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
OUTPUT_DIR = PROJECT_ROOT / "docs" / "tables" / "ablation"
TEST_RESULTS_PATH = PROJECT_ROOT / "docs" / "test-set" / "test_results.json"
YOLO26M_RESULTS_PATH = PROJECT_ROOT / "experiments" / "yolo26m" / "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_tables")


# ---------------------------------------------------------------------------
# MLflow query helpers
# ---------------------------------------------------------------------------


def _get_mlflow_client():
    """Create and return an MLflow client with local SQLite tracking."""
    import mlflow
    from scripts.mlflow_utils import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.MlflowClient()


def get_ablation_runs(client) -> Dict[str, List[Dict[str, Any]]]:
    """Query MLflow for all ablation-tagged runs across all experiments.

    Returns
    -------
    dict
        ``{"image_size": [...], "optimizer": [...], "augmentation": [...]}``
        Each entry is a list of dicts with keys:
        ``condition, mAP50, mAP50-95, precision, recall, image_size, optimizer,
        augmentation``
    """
    exps = client.search_experiments()
    all_runs: List[Dict[str, Any]] = []

    for exp in exps:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for r in runs:
            metrics = r.data.metrics
            tags = r.data.tags
            params = r.data.params

            m50 = metrics.get("val/mAP50")
            if m50 is None:
                continue  # skip runs without validation metrics

            entry = {
                "experiment": exp.name,
                "run_name": r.info.run_name,
                "run_id": r.info.run_id,
                "mAP50": float(m50),
                "mAP50-95": float(metrics.get("val/mAP50-95", 0)),
                "precision": float(metrics.get("val/precision", 0)),
                "recall": float(metrics.get("val/recall", 0)),
                "experiment_type": tags.get("experiment_type", ""),
                "image_size": tags.get("image_size", params.get("data/image_size", "")),
                "optimizer": params.get("training/optimizer", ""),
                "lr": params.get("training/lr", ""),
                "augmentation": tags.get("augmentation", params.get("augmentation/mode", "")),
            }
            # Compute F1 from precision and recall
            p, r = entry["precision"], entry["recall"]
            entry["F1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            all_runs.append(entry)

    # --- Categorize runs ---
    image_size_runs: List[Dict[str, Any]] = []
    optimizer_runs: List[Dict[str, Any]] = []
    augmentation_runs: List[Dict[str, Any]] = []

    for run in all_runs:
        # Image size ablation: tagged as ablation, varies image_size
        if run["experiment_type"] == "ablation":
            image_size_runs.append(run)
        # Optimizer ablation: found by optimizer param variation
        elif run["optimizer"] in ("SGD", "Adam") and run["image_size"] == "640":
            optimizer_runs.append(run)
        # Augmentation ablation: found by augmentation mode variation in yolo26m
        elif (
            run["experiment"] == "yolo26m"
            and run["augmentation"] in ("heavy", "light", "none")
            and run["image_size"] == "640"
            and run["optimizer"] == "AdamW"
        ):
            augmentation_runs.append(run)

    # De-duplicate: keep best mAP50 per unique condition
    def _best_per_condition(runs_list, key):
        seen = {}
        for r in runs_list:
            cond = r[key]
            if cond not in seen or r["mAP50"] > seen[cond]["mAP50"]:
                seen[cond] = r
        return list(seen.values())

    image_size_runs = _best_per_condition(image_size_runs, "image_size")
    optimizer_runs = _best_per_condition(optimizer_runs, "optimizer")
    augmentation_runs = _best_per_condition(augmentation_runs, "augmentation")

    return {
        "image_size": sorted(image_size_runs, key=lambda x: int(x["image_size"])),
        "optimizer": sorted(optimizer_runs, key=lambda x: x["mAP50"], reverse=True),
        "augmentation": sorted(augmentation_runs, key=lambda x: x["mAP50"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------


def load_baseline(default_mAP50=0.404, default_mAP95=0.117,
                  default_P=0.482, default_R=0.460) -> Dict[str, float]:
    """Load baseline metrics. Try yolo26m CV results, fall back to defaults."""
    # Try loading from yolo26m cross-fold results
    if YOLO26M_RESULTS_PATH.exists():
        with open(YOLO26M_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cf = data.get("cross_fold", {})
        m50 = cf.get("mAP50", {}).get("mean")
        m95 = cf.get("mAP50-95", {}).get("mean")
        p = cf.get("precision", {}).get("mean")
        r = cf.get("recall", {}).get("mean")
        if all(v is not None for v in [m50, m95, p, r]):
            logger.info("Loaded baseline from cross-fold results: mAP50=%.4f", m50)
            return {
                "mAP50": float(m50),
                "mAP50-95": float(m95),
                "precision": float(p),
                "recall": float(r),
            }

    # Fallback to task-specified defaults
    logger.info("Using default baseline values")
    return {
        "mAP50": default_mAP50,
        "mAP50-95": default_mAP95,
        "precision": default_P,
        "recall": default_R,
    }


def _compute_f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _delta_str(val: float, baseline: float) -> str:
    """Format delta with sign, or '-' for baseline."""
    if baseline == 0:
        return "-"
    d = val - baseline
    return f"{d:+.3f}"


# ---------------------------------------------------------------------------
# Table formatting: Image Size
# ---------------------------------------------------------------------------


def format_image_size_latex(rows: List[Dict], baseline: Dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of input image resolution on YOLO26-M detection performance.}",
        r"\label{tab:ablation-image-size}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Image Size & mAP50 & mAP50--95 & P & R & $\Delta$mAP50 \\",
        r"\midrule",
    ]

    # Add baseline row first (640) — use displayed mAP50 as reference
    bl_row = next((r for r in rows if r["image_size"] == "640"), None)
    ref_m50 = bl_row["mAP50"] if bl_row else baseline["mAP50"]
    if bl_row:
        lines.append(
            f"640 (baseline) & {bl_row['mAP50']:.3f} & "
            f"{bl_row['mAP50-95']:.3f} & {bl_row['precision']:.3f} & "
            f"{bl_row['recall']:.3f} & -- \\\\"
        )
    else:
        lines.append(
            f"640 (baseline) & {baseline['mAP50']:.3f} & "
            f"{baseline['mAP50-95']:.3f} & {baseline['precision']:.3f} & "
            f"{baseline['recall']:.3f} & -- \\\\"
        )

    # Other rows — delta relative to the displayed baseline row
    for row in rows:
        if row["image_size"] == "640":
            continue
        delta = row["mAP50"] - ref_m50
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{row['image_size']} & {row['mAP50']:.3f} & "
            f"{row['mAP50-95']:.3f} & {row['precision']:.3f} & "
            f"{row['recall']:.3f} & {sign}{delta:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_image_size_markdown(rows: List[Dict], baseline: Dict) -> str:
    header = (
        "| Image Size | mAP50 | mAP50-95 | P | R | F1 | ΔmAP50 |\n"
        "|:-----------|------:|---------:|--:|--:|--:|-------:|"
    )
    lines = [header]

    # Baseline row first — use displayed mAP50 as reference
    bl_row = next((r for r in rows if r["image_size"] == "640"), None)
    ref_m50 = bl_row["mAP50"] if bl_row else baseline["mAP50"]
    if bl_row:
        f1 = _compute_f1(bl_row["precision"], bl_row["recall"])
        lines.append(
            f"| **640 (baseline)** | **{bl_row['mAP50']:.3f}** | "
            f"{bl_row['mAP50-95']:.3f} | {bl_row['precision']:.3f} | "
            f"{bl_row['recall']:.3f} | {f1:.3f} | **--** |"
        )

    for row in rows:
        if row["image_size"] == "640":
            continue
        delta = row["mAP50"] - ref_m50
        sign = "+" if delta >= 0 else ""
        f1 = _compute_f1(row["precision"], row["recall"])
        lines.append(
            f"| {row['image_size']} | {row['mAP50']:.3f} | "
            f"{row['mAP50-95']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {f1:.3f} | {sign}{delta:.3f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table formatting: Optimizer
# ---------------------------------------------------------------------------


def format_optimizer_latex(rows: List[Dict], baseline: Dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of optimizer choice on YOLO26-M detection performance.}",
        r"\label{tab:ablation-optimizer}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Optimizer & LR & Scheduler & mAP50 & mAP50--95 & P & R & $\Delta$mAP50 \\",
        r"\midrule",
    ]

    # Find AdamW baseline row — use displayed mAP50 as reference
    adamw_row = next((r for r in rows if r["optimizer"] == "AdamW"), None)
    ref_m50 = adamw_row["mAP50"] if adamw_row else baseline["mAP50"]
    if adamw_row:
        lines.append(
            f"AdamW (baseline) & 4.7e-4 & Linear & "
            f"{adamw_row['mAP50']:.3f} & {adamw_row['mAP50-95']:.3f} & "
            f"{adamw_row['precision']:.3f} & {adamw_row['recall']:.3f} & -- \\\\"
        )
    else:
        lines.append(
            f"AdamW (baseline) & 4.7e-4 & Linear & "
            f"{baseline['mAP50']:.3f} & {baseline['mAP50-95']:.3f} & "
            f"{baseline['precision']:.3f} & {baseline['recall']:.3f} & -- \\\\"
        )

    # Other rows — delta relative to the displayed baseline row
    lr_map = {"SGD": "1.0e-2", "Adam": "1.0e-3"}
    sched_map = {"SGD": "Cosine", "Adam": "Cosine"}
    for row in rows:
        if row["optimizer"] == "AdamW":
            continue
        delta = row["mAP50"] - ref_m50
        sign = "+" if delta >= 0 else ""
        lr_display = lr_map.get(row["optimizer"], row.get("lr", "?"))
        sched = sched_map.get(row["optimizer"], "Linear")
        lines.append(
            f"{row['optimizer']} & {lr_display} & {sched} & "
            f"{row['mAP50']:.3f} & {row['mAP50-95']:.3f} & "
            f"{row['precision']:.3f} & {row['recall']:.3f} & {sign}{delta:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_optimizer_markdown(rows: List[Dict], baseline: Dict) -> str:
    header = (
        "| Optimizer | LR | mAP50 | mAP50-95 | P | R | F1 | ΔmAP50 |\n"
        "|:----------|---:|------:|---------:|--:|--:|--:|-------:|"
    )
    lines = [header]

    lr_map = {"SGD": "1.0e-2", "Adam": "1.0e-3"}

    # AdamW baseline row first — use displayed mAP50 as reference
    adamw_row = next((r for r in rows if r["optimizer"] == "AdamW"), None)
    ref_m50 = adamw_row["mAP50"] if adamw_row else baseline["mAP50"]
    if adamw_row:
        f1 = _compute_f1(adamw_row["precision"], adamw_row["recall"])
        lines.append(
            f"| **AdamW (baseline)** | 4.7e-4 | **{adamw_row['mAP50']:.3f}** | "
            f"{adamw_row['mAP50-95']:.3f} | {adamw_row['precision']:.3f} | "
            f"{adamw_row['recall']:.3f} | {f1:.3f} | **--** |"
        )

    for row in rows:
        if row["optimizer"] == "AdamW":
            continue
        delta = row["mAP50"] - ref_m50
        sign = "+" if delta >= 0 else ""
        f1 = _compute_f1(row["precision"], row["recall"])
        lr_display = lr_map.get(row["optimizer"], row.get("lr", "?"))
        lines.append(
            f"| {row['optimizer']} | {lr_display} | {row['mAP50']:.3f} | "
            f"{row['mAP50-95']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {f1:.3f} | {sign}{delta:.3f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table formatting: Augmentation
# ---------------------------------------------------------------------------


def format_augmentation_latex(rows: List[Dict], baseline: Dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of data augmentation strategy on YOLO26-M detection performance.}",
        r"\label{tab:ablation-augmentation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Augmentation & mAP50 & mAP50--95 & P & R & $\Delta$mAP50 \\",
        r"\midrule",
    ]

    # Find baseline augmentation row — use displayed mAP50 as reference
    aug_rows = {r["augmentation"]: r for r in rows}
    bl_aug = aug_rows.get("ultralytics")
    ref_m50 = bl_aug["mAP50"] if bl_aug else baseline["mAP50"]

    # Define display order: baseline first, then others
    aug_order = ["ultralytics", "light", "heavy", "none"]
    aug_labels = {
        "ultralytics": "Ultralytics (baseline)",
        "light": "Light",
        "heavy": "Heavy",
        "none": "None",
    }

    for aug_key in aug_order:
        if aug_key not in aug_rows:
            continue
        row = aug_rows[aug_key]
        label = aug_labels[aug_key]
        is_baseline = aug_key == "ultralytics"

        if is_baseline:
            lines.append(
                f"{label} & {row['mAP50']:.3f} & {row['mAP50-95']:.3f} & "
                f"{row['precision']:.3f} & {row['recall']:.3f} & -- \\\\"
            )
        else:
            delta = row["mAP50"] - ref_m50
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"{label} & {row['mAP50']:.3f} & {row['mAP50-95']:.3f} & "
                f"{row['precision']:.3f} & {row['recall']:.3f} & {sign}{delta:.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_augmentation_markdown(rows: List[Dict], baseline: Dict) -> str:
    header = (
        "| Augmentation | mAP50 | mAP50-95 | P | R | F1 | ΔmAP50 |\n"
        "|:-------------|------:|---------:|--:|--:|--:|-------:|"
    )
    lines = [header]

    aug_order = ["ultralytics", "light", "heavy", "none"]
    aug_labels = {
        "ultralytics": "Ultralytics (baseline)",
        "light": "Light",
        "heavy": "Heavy",
        "none": "None",
    }
    aug_rows = {r["augmentation"]: r for r in rows}

    # Reference mAP50 from the displayed baseline row
    bl_aug = aug_rows.get("ultralytics")
    ref_m50 = bl_aug["mAP50"] if bl_aug else baseline["mAP50"]

    for aug_key in aug_order:
        if aug_key not in aug_rows:
            continue
        row = aug_rows[aug_key]
        label = aug_labels[aug_key]
        is_baseline = aug_key == "ultralytics"
        f1 = _compute_f1(row["precision"], row["recall"])

        if is_baseline:
            lines.append(
                f"| **{label}** | **{row['mAP50']:.3f}** | "
                f"{row['mAP50-95']:.3f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {f1:.3f} | **--** |"
            )
        else:
            delta = row["mAP50"] - ref_m50
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {label} | {row['mAP50']:.3f} | "
                f"{row['mAP50-95']:.3f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {f1:.3f} | {sign}{delta:.3f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_table(name: str, latex: str, markdown: str) -> None:
    """Save LaTeX and markdown table files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUTPUT_DIR / f"{name}.tex"
    md_path = OUTPUT_DIR / f"{name}.md"

    tex_path.write_text(latex, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    logger.info("Saved: %s", tex_path)
    logger.info("Saved: %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation study tables (LaTeX + Markdown).",
    )
    parser.add_argument(
        "--baseline-mAP50",
        type=float,
        default=0.404,
        help="Baseline mAP50 for delta calculations (default: 0.404)",
    )
    parser.add_argument(
        "--baseline-mAP95",
        type=float,
        default=0.117,
        help="Baseline mAP50-95 (default: 0.117)",
    )
    parser.add_argument(
        "--baseline-P",
        type=float,
        default=0.482,
        help="Baseline precision (default: 0.482)",
    )
    parser.add_argument(
        "--baseline-R",
        type=float,
        default=0.460,
        help="Baseline recall (default: 0.460)",
    )
    args = parser.parse_args()

    baseline = load_baseline(
        default_mAP50=args.baseline_mAP50,
        default_mAP95=args.baseline_mAP95,
        default_P=args.baseline_P,
        default_R=args.baseline_R,
    )
    logger.info("Baseline: mAP50=%.3f, mAP50-95=%.3f, P=%.3f, R=%.3f",
                baseline["mAP50"], baseline["mAP50-95"],
                baseline["precision"], baseline["recall"])

    # --- Query MLflow ---
    try:
        client = _get_mlflow_client()
        ablation_data = get_ablation_runs(client)
    except Exception as e:
        logger.error("Failed to connect to MLflow: %s", e)
        sys.exit(1)

    generated = 0

    # --- Image Size Ablation ---
    img_rows = ablation_data.get("image_size", [])
    if img_rows:
        logger.info("Image Size Ablation: %d conditions found", len(img_rows))
        for r in img_rows:
            logger.info("  %s px: mAP50=%.4f", r["image_size"], r["mAP50"])
        save_table(
            "image_size",
            format_image_size_latex(img_rows, baseline),
            format_image_size_markdown(img_rows, baseline),
        )
        generated += 2
    else:
        logger.warning("Image Size ablation: no MLflow data found — skipping")

    # --- Optimizer Ablation ---
    opt_rows = ablation_data.get("optimizer", [])
    # Always include AdamW baseline in optimizer table
    if img_rows:
        bl_640 = next((r for r in img_rows if r["image_size"] == "640"), None)
        if bl_640 and not any(r["optimizer"] == "AdamW" for r in opt_rows):
            opt_rows.append(bl_640)

    if opt_rows:
        logger.info("Optimizer Ablation: %d conditions found", len(opt_rows))
        for r in opt_rows:
            logger.info("  %s (lr=%s): mAP50=%.4f",
                        r["optimizer"], r.get("lr", "?"), r["mAP50"])
        save_table(
            "optimizer",
            format_optimizer_latex(opt_rows, baseline),
            format_optimizer_markdown(opt_rows, baseline),
        )
        generated += 2
    else:
        logger.warning("Optimizer ablation: no MLflow data found — skipping")

    # --- Augmentation Ablation ---
    aug_rows = ablation_data.get("augmentation", [])
    # Include ultralytics baseline if not already present
    if aug_rows and not any(r["augmentation"] == "ultralytics" for r in aug_rows):
        # Find best ultralytics run from all yolo26m 640 runs
        try:
            exp = client.get_experiment_by_name("yolo26m")
            all_runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.image_size = '640'",
            )
            best_ultra = None
            for r in all_runs:
                if r.data.params.get("augmentation/mode") == "ultralytics":
                    m50 = r.data.metrics.get("val/mAP50")
                    if m50 and (best_ultra is None or m50 > best_ultra["mAP50"]):
                        metrics = r.data.metrics
                        p = float(metrics.get("val/precision", 0))
                        rc = float(metrics.get("val/recall", 0))
                        best_ultra = {
                            "experiment": "yolo26m",
                            "run_name": r.info.run_name,
                            "run_id": r.info.run_id,
                            "mAP50": float(m50),
                            "mAP50-95": float(metrics.get("val/mAP50-95", 0)),
                            "precision": p,
                            "recall": rc,
                            "F1": _compute_f1(p, rc),
                            "experiment_type": "",
                            "image_size": "640",
                            "optimizer": "AdamW",
                            "lr": r.data.params.get("training/lr", ""),
                            "augmentation": "ultralytics",
                        }
            if best_ultra:
                aug_rows.insert(0, best_ultra)
        except Exception:
            pass

    if aug_rows:
        logger.info("Augmentation Ablation: %d conditions found", len(aug_rows))
        for r in aug_rows:
            logger.info("  %s: mAP50=%.4f", r["augmentation"], r["mAP50"])
        save_table(
            "augmentation",
            format_augmentation_latex(aug_rows, baseline),
            format_augmentation_markdown(aug_rows, baseline),
        )
        generated += 2
    else:
        logger.warning("Augmentation ablation: no MLflow data found — skipping")

    # --- Summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Generated %d table files in %s", generated, OUTPUT_DIR)
    if generated == 0:
        logger.warning("No ablation tables generated — all MLflow experiments missing")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
