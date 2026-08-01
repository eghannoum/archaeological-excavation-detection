"""
Generate publication-ready LaTeX and Markdown tables from experiment results.

Reads:
  - docs/test-set/test_results.json  (held-out test metrics)
  - docs/significance/significance_report.json  (CV means, CIs, pairwise p-values)

Writes:
  - docs/tables/main_results.tex / .md
  - docs/tables/cv_results.tex  / .md
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TEST_JSON = ROOT / "docs" / "test-set" / "test_results.json"
SIG_JSON = ROOT / "docs" / "significance" / "significance_report.json"
OUT_DIR = ROOT / "docs" / "tables"

# Display labels (ordered by mAP50 descending for the main table)
DISPLAY_ORDER = [
    "yolo26x",
    "yolo26m",
    "yolo26s",
    "yolo11m",
    "yolov8m",
    "faster_rcnn",
    "yolo26n",
    "yolo26l",
    "detr",
]

DISPLAY_LABELS: dict[str, str] = {
    "yolo26x": "YOLO26-X",
    "yolo26m": "YOLO26-M",
    "yolo26s": "YOLO26-S",
    "yolo26l": "YOLO26-L",
    "yolo26n": "YOLO26-N",
    "yolo11m": "YOLO11-M",
    "yolov8m": "YOLOv8-M",
    "faster_rcnn": "Faster R-CNN",
    "detr": "DETR",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(val: float, decimals: int) -> str:
    return f"{val:.{decimals}f}"


def compute_significance_groups(
    pairwise: dict[str, dict], models: list[str]
) -> dict[str, str]:
    """Return {model: group_letter} using connected-component labelling.

    Models whose pairwise p-value >= 0.05 are placed in the same group.
    """
    # Build adjacency for non-significant pairs
    adj: dict[str, set[str]] = {m: set() for m in models}
    for key, info in pairwise.items():
        if not info.get("significant_at_0.05", True):
            a, b = key.split("_vs_")
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)

    # BFS connected components
    visited: set[str] = set()
    components: list[list[str]] = []
    # Order components by best mean mAP50 so letter assignment is stable
    for model in models:
        if model in visited:
            continue
        queue = [model]
        comp: list[str] = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)

    # Sort components by best mAP50 mean (descending) for letter ordering
    # (passed from caller via models ordering)
    model_order = {m: i for i, m in enumerate(models)}
    components.sort(key=lambda c: min(model_order[m] for m in c))

    letters = "abcdefghijklmnopqrstuvwxyz"
    groups: dict[str, str] = {}
    for idx, comp in enumerate(components):
        letter = letters[idx] if idx < len(letters) else str(idx)
        for m in comp:
            groups[m] = letter
    return groups


def std_from_folds(folds: list[float]) -> float:
    """Population std from 3 fold values."""
    return statistics.pstdev(folds)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------
def build_test_lookup(test_data: dict) -> dict[str, dict]:
    return {m["model_name"]: m for m in test_data["models"]}


def build_cv_lookup(sig_data: dict) -> dict[str, dict]:
    return sig_data["model_stats"]


def make_main_rows(
    test_lookup: dict[str, dict],
) -> list[tuple[str, list[str]]]:
    """Return rows: (label, [mAP50, mAP50-95, P, R, F1, Time, Params, FLOPs])."""
    # First pass: find best / second-best for each column
    # Columns: mAP50, mAP50-95, P, R, F1  (higher is better)
    #          Time, Params, FLOPs          (lower is better)
    metric_keys = [
        ("mAP50", True),
        ("mAP50-95", True),
        ("precision", True),
        ("recall", True),
        ("F1", True),
        ("inference_time_ms", False),
        ("params_M", False),
        ("flops_G", False),
    ]

    raw: dict[str, list[float]] = {}
    for model_id in DISPLAY_ORDER:
        d = test_lookup[model_id]
        raw[model_id] = [d[k] for k, _ in metric_keys]

    rows: list[tuple[str, list[str]]] = []
    for model_id in DISPLAY_ORDER:
        vals = raw[model_id]
        fmt_vals: list[str] = []
        for ci, (key, higher_better) in enumerate(metric_keys):
            v = vals[ci]
            if key in ("mAP50", "mAP50-95", "precision", "recall", "F1"):
                fmt_vals.append(fmt(v, 3))
            else:
                fmt_vals.append(fmt(v, 1))
        rows.append((DISPLAY_LABELS[model_id], fmt_vals))

    # Determine best / second-best per column
    best: list[str] = [""] * len(metric_keys)
    second: list[str] = [""] * len(metric_keys)
    for ci, (key, higher_better) in enumerate(metric_keys):
        all_models_sorted = sorted(
            DISPLAY_ORDER,
            key=lambda m: raw[m][ci],
            reverse=higher_better,
        )
        best_val = raw[all_models_sorted[0]][ci]
        second_val = raw[all_models_sorted[1]][ci]
        best[ci] = all_models_sorted[0]
        # Find second-best distinct value
        for m in all_models_sorted[1:]:
            if raw[m][ci] != best_val:
                second[ci] = m
                second_val = raw[m][ci]
                break

    # Apply bold/italic
    styled_rows: list[tuple[str, list[str]]] = []
    for (label, vals) in rows:
        styled: list[str] = []
        for ci, v in enumerate(vals):
            model_id = DISPLAY_ORDER[rows.index((label, vals))]
            if model_id == best[ci]:
                styled.append(f"\\textbf{{{v}}}")
            elif model_id == second[ci]:
                styled.append(f"\\textit{{{v}}}")
            else:
                styled.append(v)
        styled_rows.append((label, styled))

    return styled_rows


def make_main_rows_md(
    test_lookup: dict[str, dict],
) -> list[tuple[str, list[str]]]:
    """Markdown version — no LaTeX markup, just raw values."""
    metric_keys = [
        ("mAP50", True),
        ("mAP50-95", True),
        ("precision", True),
        ("recall", True),
        ("F1", True),
        ("inference_time_ms", False),
        ("params_M", False),
        ("flops_G", False),
    ]

    raw: dict[str, list[float]] = {}
    for model_id in DISPLAY_ORDER:
        d = test_lookup[model_id]
        raw[model_id] = [d[k] for k, _ in metric_keys]

    rows: list[tuple[str, list[str]]] = []
    for model_id in DISPLAY_ORDER:
        vals = raw[model_id]
        fmt_vals: list[str] = []
        for ci, (key, _) in enumerate(metric_keys):
            v = vals[ci]
            if key in ("mAP50", "mAP50-95", "precision", "recall", "F1"):
                fmt_vals.append(fmt(v, 3))
            else:
                fmt_vals.append(fmt(v, 1))
        rows.append((DISPLAY_LABELS[model_id], fmt_vals))

    # Best / second-best markers
    best_models: list[str] = [""] * len(metric_keys)
    second_models: list[str] = [""] * len(metric_keys)
    for ci, (key, higher_better) in enumerate(metric_keys):
        all_sorted = sorted(
            DISPLAY_ORDER,
            key=lambda m: raw[m][ci],
            reverse=higher_better,
        )
        best_models[ci] = all_sorted[0]
        for m in all_sorted[1:]:
            if raw[m][ci] != raw[all_sorted[0]][ci]:
                second_models[ci] = m
                break

    styled_rows: list[tuple[str, list[str]]] = []
    for idx, (label, vals) in enumerate(rows):
        model_id = DISPLAY_ORDER[idx]
        styled: list[str] = []
        for ci, v in enumerate(vals):
            if model_id == best_models[ci]:
                styled.append(f"**{v}**")
            elif model_id == second_models[ci]:
                styled.append(f"*{v}*")
            else:
                styled.append(v)
        styled_rows.append((label, styled))

    return styled_rows


def generate_main_latex(rows: list[tuple[str, list[str]]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Object detection performance comparison on the held-out test set. "
        "Metrics: mean Average Precision (mAP) at IoU thresholds 0.5 and 0.5:0.95, "
        "Precision (P), Recall (R), F1-score, inference speed, params, FLOPs. "
        "\\textbf{Bold} indicates best in each column, \\textit{italics} second best.}",
        "\\label{tab:main_results}",
        "\\small",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Model & mAP50$\\uparrow$ & mAP50-95$\\uparrow$ & P$\\uparrow$ & R$\\uparrow$ & F1$\\uparrow$ & Time(ms)$\\downarrow$ & Params(M)$\\downarrow$ & FLOPs(G)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for label, vals in rows:
        line = label + " & " + " & ".join(vals) + " \\\\"
        lines.append(line)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def generate_main_markdown(rows: list[tuple[str, list[str]]]) -> str:
    header = "| Model | mAP50↑ | mAP50-95↑ | P↑ | R↑ | F1↑ | Time(ms)↓ | Params(M)↓ | FLOPs(G)↓ |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    body_lines = []
    for label, vals in rows:
        body_lines.append("| " + label + " | " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body_lines) + "\n"


def make_cv_rows(
    cv_lookup: dict[str, dict],
    pairwise: dict[str, dict],
) -> list[tuple[str, list[str]]]:
    """CV rows: (label, [mAP50 mean±std, mAP50-95 mean±std, Group])."""
    groups = compute_significance_groups(pairwise, DISPLAY_ORDER)

    rows: list[tuple[str, list[str]]] = []
    for model_id in DISPLAY_ORDER:
        stats = cv_lookup.get(model_id, {})
        # mAP50
        m50_mean = stats.get("mAP50_mean", 0)
        folds_50 = stats.get("mAP50_folds", [])
        m50_std = std_from_folds(folds_50) if folds_50 else 0

        # mAP50-95
        m5095_mean = stats.get("mAP50-95_mean")
        folds_5095 = stats.get("mAP50-95_folds", [])
        has_m5095 = m5095_mean is not None and folds_5095
        m5095_std = std_from_folds(folds_5095) if has_m5095 else 0
        if m5095_mean is None:
            m5095_mean = 0

        group_letter = groups.get(model_id, "")

        m5095_cell = (
            f"{fmt(m5095_mean, 4)}$\\pm${fmt(m5095_std, 4)}"
            if has_m5095
            else "N/A"
        )

        vals = [
            f"{fmt(m50_mean, 4)}$\\pm${fmt(m50_std, 4)}",
            m5095_cell,
            group_letter,
        ]
        rows.append((DISPLAY_LABELS[model_id], vals))

    return rows


def make_cv_rows_md(
    cv_lookup: dict[str, dict],
    pairwise: dict[str, dict],
) -> list[tuple[str, list[str]]]:
    """Markdown CV rows: mean±std for mAP50, mAP50-95, and group letter."""
    groups = compute_significance_groups(pairwise, DISPLAY_ORDER)

    rows: list[tuple[str, list[str]]] = []
    for model_id in DISPLAY_ORDER:
        stats = cv_lookup.get(model_id, {})
        m50_mean = stats.get("mAP50_mean", 0)
        folds_50 = stats.get("mAP50_folds", [])
        m50_std = std_from_folds(folds_50) if folds_50 else 0

        m5095_mean = stats.get("mAP50-95_mean")
        folds_5095 = stats.get("mAP50-95_folds", [])
        has_m5095 = m5095_mean is not None and folds_5095
        m5095_std = std_from_folds(folds_5095) if has_m5095 else 0
        if m5095_mean is None:
            m5095_mean = 0

        group_letter = groups.get(model_id, "")

        m5095_cell = (
            f"{fmt(m5095_mean, 4)} ± {fmt(m5095_std, 4)}"
            if has_m5095
            else "N/A"
        )

        vals = [
            f"{fmt(m50_mean, 4)} ± {fmt(m50_std, 4)}",
            m5095_cell,
            group_letter,
        ]
        rows.append((DISPLAY_LABELS[model_id], vals))

    return rows


def generate_cv_latex(rows: list[tuple[str, list[str]]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{3-fold cross-validation results with 95\\% confidence intervals "
        "and statistical significance groupings. Models sharing a letter "
        "are not significantly different (paired bootstrap test, $p<0.05$).}",
        "\\label{tab:cv_results}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & mAP50 (mean$\\pm$std) & mAP50-95 (mean$\\pm$std) & Group \\\\",
        "\\midrule",
    ]
    for label, vals in rows:
        line = label + " & " + " & ".join(vals) + " \\\\"
        lines.append(line)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def generate_cv_markdown(rows: list[tuple[str, list[str]]]) -> str:
    header = "| Model | mAP50 (mean ± std) | mAP50-95 (mean ± std) | Group |"
    sep = "|---|---|---|---|"
    body_lines = []
    for label, vals in rows:
        body_lines.append("| " + label + " | " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body_lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_data = load_json(TEST_JSON)
    sig_data = load_json(SIG_JSON)

    test_lookup = build_test_lookup(test_data)
    cv_lookup = build_cv_lookup(sig_data)

    # --- Main results table ---
    main_rows = make_main_rows(test_lookup)
    main_rows_md = make_main_rows_md(test_lookup)

    main_tex = generate_main_latex(main_rows)
    main_md = generate_main_markdown(main_rows_md)

    (OUT_DIR / "main_results.tex").write_text(main_tex, encoding="utf-8")
    (OUT_DIR / "main_results.md").write_text(main_md, encoding="utf-8")

    # --- CV results table ---
    cv_rows = make_cv_rows(cv_lookup, sig_data["pairwise_comparisons"])
    cv_rows_md = make_cv_rows_md(cv_lookup, sig_data["pairwise_comparisons"])

    cv_tex = generate_cv_latex(cv_rows)
    cv_md = generate_cv_markdown(cv_rows_md)

    (OUT_DIR / "cv_results.tex").write_text(cv_tex, encoding="utf-8")
    (OUT_DIR / "cv_results.md").write_text(cv_md, encoding="utf-8")

    print(f"Generated 4 files in {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
