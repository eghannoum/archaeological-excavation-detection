"""Gradio dashboard for YOLO26m archaeological hole detection system.

Tabs:
  1. Model Metrics — confusion matrix, metrics table, training curves
  2. Interactive Inference — upload image, detect, view results
  3. Batch Inference — configure and run batch processing

Launch:
    python scripts/dashboard.py
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# Ensure the project root is on sys.path so ``from scripts.xxx`` imports work
# regardless of whether the user runs ``python scripts/dashboard.py`` or
# ``python -m scripts.dashboard``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from ultralytics import YOLO

    UltralyticsYOLO: type[YOLO] | None = YOLO
except ImportError:
    UltralyticsYOLO = None

from scripts.model_registry import resolve_best  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
TRAIN_DIR = RUNS_DIR / "train" / "yolo26m-hole"
EVAL_DIR = RUNS_DIR / "eval"
DATASET_TEST_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

BEST_PT = TRAIN_DIR / "weights" / "best.pt"
LAST_PT = TRAIN_DIR / "weights" / "last.pt"
CONFUSION_MATRIX = EVAL_DIR / "confusion_matrix.png"
RESULTS_JSON = EVAL_DIR / "results.json"
TRAIN_CSV = TRAIN_DIR / "results.csv"

DEFAULT_BATCH_OUTPUT = RUNS_DIR / "inference"


# ---------------------------------------------------------------------------
# Model cache (lazy-loaded, thread-safe)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model_instance = None


def _resolve_model_path() -> Path | None:
    """Return the best available model path, or None."""
    try:
        return resolve_best()
    except FileNotFoundError:
        # No registry/scan result yet — fall back to the historical
        # train-output locations so the dashboard still renders.
        if BEST_PT.exists():
            return BEST_PT
        if LAST_PT.exists():
            return LAST_PT
        return None


def load_model() -> YOLO | None:
    """Lazy-load and cache the YOLO model. Returns None on failure."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:
            return _model_instance
        if UltralyticsYOLO is None:
            raise ImportError("ultralytics is not installed — run: pip install ultralytics")

        model_path = _resolve_model_path()
        if model_path is None:
            return None
        try:
            _model_instance = UltralyticsYOLO(str(model_path))
        except Exception:
            _model_instance = None
    return _model_instance


def unload_model() -> None:
    """Force reload on next call (useful after training)."""
    global _model_instance
    with _model_lock:
        _model_instance = None


# ---------------------------------------------------------------------------
# Tab 1 — Model Metrics
# ---------------------------------------------------------------------------


def _load_metrics_table() -> pd.DataFrame | None:
    """Load evaluation metrics from results.json into a DataFrame."""
    if not RESULTS_JSON.exists():
        return None
    try:
        with open(RESULTS_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    row = {
        "Metric": {
            "mAP50": "mAP@50",
            "mAP50-95": "mAP@50:95",
            "Precision": "Precision",
            "Recall": "Recall",
            "F1-Score": "F1-Score",
        },
        "Value": {
            "mAP50": data.get("mAP50", data.get("map50", "N/A")),
            "mAP50-95": data.get("mAP50-95", data.get("map", "N/A")),
            "Precision": data.get("Precision", data.get("precision", "N/A")),
            "Recall": data.get("Recall", data.get("recall", "N/A")),
            "F1-Score": data.get("F1-Score", data.get("f1", "N/A")),
        },
    }
    return pd.DataFrame(row)


def _load_training_curve() -> Path | None:
    """Return path to training loss chart if available, else None."""
    if not TRAIN_CSV.exists():
        return None

    # Build a simple loss curve image from the CSV using matplotlib (if available)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: N812
    except ImportError:
        return None

    # Parse CSV
    rows: list[dict[str, float]] = []
    with open(TRAIN_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v})

    if not rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    fig.suptitle("Training Curves", color="#e6edf3", fontsize=13, fontweight="bold")

    epochs = [r.get("epoch", i) for i, r in enumerate(rows)]

    # Left: Losses
    ax = axes[0]
    ax.set_facecolor("#161b22")
    for key, label, color in [
        ("train/box_loss", "Box Loss", "#e94560"),
        ("train/cls_loss", "Cls Loss", "#4ecdc4"),
        ("train/dfl_loss", "DFL Loss", "#ffd166"),
    ]:
        vals = [r.get(key) for r in rows if key in r]
        if vals:
            ax.plot(epochs[: len(vals)], vals, label=label, color=color, linewidth=1.5)
    ax.set_title("Training Loss", color="#e6edf3", fontsize=10)
    ax.set_xlabel("Epoch", color="#a0a0b0", fontsize=9)
    ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")
    ax.tick_params(colors="#a0a0b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a2a4a")

    # Right: Metrics
    ax = axes[1]
    ax.set_facecolor("#161b22")
    for key, label, color in [
        ("metrics/mAP50(B)", "mAP@50", "#4ecdc4"),
        ("metrics/mAP50-95(B)", "mAP@50:95", "#e94560"),
        ("metrics/precision(B)", "Precision", "#ffd166"),
        ("metrics/recall(B)", "Recall", "#8884d8"),
    ]:
        vals = [r.get(key) for r in rows if key in r]
        if vals:
            ax.plot(epochs[: len(vals)], vals, label=label, color=color, linewidth=1.5)
    ax.set_title("Validation Metrics", color="#e6edf3", fontsize=10)
    ax.set_xlabel("Epoch", color="#a0a0b0", fontsize=9)
    ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")
    ax.tick_params(colors="#a0a0b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a2a4a")

    plt.tight_layout()
    chart_path = TRAIN_DIR / "_dashboard_curves.png"
    fig.savefig(str(chart_path), dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return chart_path if chart_path.exists() else None


def build_metrics_tab() -> None:
    """Render the Model Metrics tab."""
    confusion_path = CONFUSION_MATRIX if CONFUSION_MATRIX.exists() else None
    metrics_df = _load_metrics_table()
    curve_path = _load_training_curve()
    has_any_data = confusion_path or metrics_df is not None or curve_path

    if not has_any_data:
        gr.Markdown(
            "## No Results Found\n\n"
            "No evaluation results or training data are available yet.\n\n"
            "**Run one of the following first:**\n"
            "- `python scripts/train.py` to train the model\n"
            "- Evaluate with the evaluation script to generate metrics"
        )
        return

    # --- Confusion Matrix ---
    if confusion_path:
        gr.Markdown("### Confusion Matrix")
        gr.Image(
            value=str(confusion_path),
            label="Confusion Matrix",
            show_label=False,
            height=480,
        )
    else:
        gr.Markdown("### Confusion Matrix\n*Not available — run evaluation to generate.*")

    # --- Metrics Table ---
    gr.Markdown("### Performance Metrics")
    if metrics_df is not None:
        gr.DataFrame(
            value=metrics_df,
            label="Metrics",
            show_label=False,
            row_count=5,
            column_count=2,
        )
    else:
        gr.Markdown("*No metrics file found.*")

    # --- Training Curves ---
    if curve_path:
        gr.Markdown("### Training History")
        gr.Image(
            value=str(curve_path),
            label="Training Curves",
            show_label=False,
            height=360,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Interactive Inference
# ---------------------------------------------------------------------------


def run_inference(
    image: Image.Image,
    conf_threshold: float,
    progress: gr.Progress | None = None,
) -> tuple[Image.Image | None, str, pd.DataFrame | None]:
    """Run YOLO detection on a single image and return annotated result."""
    if image is None:
        return None, "Please upload an image first.", None

    if progress is None:
        progress = gr.Progress()

    progress(0.0, desc="Loading model…")
    try:
        model = load_model()
    except ImportError as exc:
        return None, f"**Error:** {exc}", None

    if model is None:
        return (
            None,
            (
                "**Model not found.**\n\n"
                f"Expected at `{BEST_PT}` or `{LAST_PT}`.\n\n"
                "Run `python scripts/train.py` first to train the model."
            ),
            None,
        )

    progress(0.3, desc="Running detection…")
    try:
        results = model(image, conf=conf_threshold, verbose=False)
    except Exception as exc:
        return None, f"**Inference failed:** {exc}", None

    progress(0.7, desc="Rendering results…")

    result = results[0]
    annotated = result.plot()
    annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR → RGB

    # Build detections table
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        data = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if model.names else str(cls_id)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            data.append(
                [cls_name, f"{conf:.3f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"]
            )
        detections_df = pd.DataFrame(
            data,
            columns=["Class", "Confidence", "x1", "y1", "x2", "y2"],
        )
        count = len(data)
        summary = f"**{count} hole{'s' if count != 1 else ''} detected**  "
        summary += f"· Avg confidence: {sum(float(d[1]) for d in data) / count:.3f}"
    else:
        detections_df = pd.DataFrame(columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])
        summary = "**No holes detected** — try lowering the confidence threshold."

    progress(1.0, desc="Done")
    return annotated_pil, summary, detections_df


def build_inference_tab() -> None:
    """Render the Interactive Inference tab."""
    gr.Markdown(
        "Upload a satellite or aerial image to detect potential "
        "unauthorized archaeological excavations."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Input Image",
                height=400,
            )
            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
                info="Lower values detect more holes but may increase false positives.",
            )
            detect_btn = gr.Button("Detect", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_output = gr.Image(
                type="pil",
                label="Detection Result",
                height=400,
            )

    summary_output = gr.Markdown()
    detections_table = gr.DataFrame(
        label="Detections",
        row_count=10,
        column_widths=["Class", "Confidence", "x1", "y1", "x2", "y2"],
    )

    # Example images
    if DATASET_TEST_DIR.exists():
        png_files = sorted(DATASET_TEST_DIR.glob("*.png"))
        if png_files:
            gr.Markdown("### Quick Test Examples")
            gr.Examples(
                examples=[[str(p)] for p in png_files[:8]],
                inputs=[image_input],
                label="Select a test image",
            )

    detect_btn.click(
        fn=run_inference,
        inputs=[image_input, conf_slider],
        outputs=[image_output, summary_output, detections_table],
    )


# ---------------------------------------------------------------------------
# Tab 3 — Batch Inference
# ---------------------------------------------------------------------------


def run_batch(
    source_path: str,
    output_path: str,
    conf_threshold: float,
    progress: gr.Progress | None = None,
) -> tuple[str, str | None]:
    """Run detection on all images in a directory."""
    src = Path(source_path) if source_path else None

    if src is None or not src.is_dir():
        return "**Error:** Source path is not a valid directory.", None

    if progress is None:
        progress = gr.Progress()

    out = Path(output_path).resolve() if output_path else DEFAULT_BATCH_OUTPUT.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Gather image files
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")
    image_files: list[Path] = []
    for ext in extensions:
        image_files.extend(sorted(src.glob(ext)))
        image_files.extend(sorted(src.glob(ext.upper())))

    if not image_files:
        return "**Error:** No supported image files found in the source directory.", None

    progress(0.0, desc="Loading model…")
    try:
        model = load_model()
    except ImportError as exc:
        return f"**Error:** {exc}", None

    if model is None:
        return (
            "**Model not found.** Train the model first with `python scripts/train.py`.",
            None,
        )

    total = len(image_files)
    total_detections = 0
    confidences: list[float] = []
    log_lines: list[str] = []

    for i, img_path in enumerate(image_files, 1):
        pct = i / total
        progress(pct, desc=f"Processing {img_path.name} ({i}/{total})")

        try:
            img = Image.open(img_path).convert("RGB")
            results = model(img, conf=conf_threshold, verbose=False)
            result = results[0]

            # Save annotated image
            annotated = result.plot()
            annotated_pil = Image.fromarray(annotated[..., ::-1])
            out_path = out / f"annotated_{img_path.stem}.png"
            annotated_pil.save(str(out_path), quality=92)

            # Collect stats
            boxes = result.boxes
            n_det = len(boxes) if boxes is not None else 0
            total_detections += n_det
            if boxes is not None:
                confidences.extend(float(b.conf[0]) for b in boxes)

            log_lines.append(f"{img_path.name}: {n_det} hole(s) → {out_path.name}")

        except Exception as exc:
            log_lines.append(f"{img_path.name}: ERROR — {exc}")

    # Summary
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    summary = (
        f"**Batch Inference Complete**\n\n"
        f"- **Total images processed:** {total}\n"
        f"- **Total detections:** {total_detections}\n"
        f"- **Average confidence:** {avg_conf:.3f}\n"
        f"- **Images with detections:** {len(confidences) > 0}\n"
        f"- **Output directory:** `{out}`"
    )

    # Write log
    log_path = out / "_batch_log.txt"
    with contextlib.suppress(OSError):
        log_path.write_text("\n".join(log_lines), encoding="utf-8")

    progress(1.0, desc="Done")
    return summary, str(out)


def open_output_folder(folder_path: str | None) -> None:
    """Open the output folder in the platform file explorer."""
    if not folder_path or not Path(folder_path).exists():
        return
    startfile = getattr(os, "startfile", None)
    if startfile is not None:
        startfile(folder_path)
        return
    if sys.platform == "darwin":
        subprocess.run(["open", folder_path], check=False)
    else:
        subprocess.run(["xdg-open", folder_path], check=False)


def build_batch_tab() -> None:
    """Render the Batch Inference tab."""
    gr.Markdown(
        "Run hole detection on a directory of images. "
        "Annotated results will be saved to the output folder."
    )

    with gr.Row():
        source_input = gr.Textbox(
            label="Source Image Directory",
            placeholder="e.g., dataset/images/test",
            value="dataset/images/test",
        )
        output_input = gr.Textbox(
            label="Output Directory",
            placeholder="e.g., runs/inference",
            value=str(DEFAULT_BATCH_OUTPUT),
        )

    conf_slider = gr.Slider(
        minimum=0.05,
        maximum=0.95,
        value=0.25,
        step=0.05,
        label="Confidence Threshold",
    )

    run_btn = gr.Button("Run Batch Inference", variant="primary", size="lg")
    summary_output = gr.Markdown()
    output_path_state = gr.State()

    run_btn.click(
        fn=run_batch,
        inputs=[source_input, output_input, conf_slider],
        outputs=[summary_output, output_path_state],
    )

    open_btn = gr.Button("Open Output Folder", variant="secondary", size="sm", visible=True)
    open_btn.click(
        fn=open_output_folder,
        inputs=[output_path_state],
        outputs=[],
    )


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --primary-h: 210;
    --primary-s: 55%;
    --primary-l: 35%;
    --neutral-h: 35;
    --neutral-s: 20%;
    --neutral-l: 45%;
}
footer { display: none !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto; }
h1, h2, h3 { font-weight: 600; letter-spacing: 0.01em; }
h1 { font-size: 1.6rem !important; border-bottom: 1px solid #2a2a4a; padding-bottom: 0.5rem; }
h3 { font-size: 1.1rem !important; color: #c9d1d9; }
button.primary { background: linear-gradient(135deg, #1e3a5f 0%, #2a5a8f 100%) !important; border: 1px solid #3a6a9f !important; }
button.primary:hover { background: linear-gradient(135deg, #2a5a8f 0%, #3a7abf 100%) !important; }
"""


def create_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(
        title="Hole Detection Dashboard \u2014 YOLO26m",
        fill_height=False,
    ) as demo:
        gr.Markdown(
            "# Unauthorized Archaeological Excavation Detection\n"
            "**YOLO26m** \u2014 Satellite & aerial imagery analysis "
            "for cultural heritage protection"
        )

        with gr.Tab("Model Metrics"):
            build_metrics_tab()

        with gr.Tab("Interactive Inference"):
            build_inference_tab()

        with gr.Tab("Batch Inference"):
            build_batch_tab()

        # Footer
        model_path = _resolve_model_path()
        model_path_text = "*none — run training first*" if model_path is None else f"`{model_path}`"
        gr.Markdown(
            "---\n"
            f"**Model path:** {model_path_text}  \u00b7  "
            f"**Data:** `dataset/data.yaml`  \u00b7  "
            "Single class detection: **hole**"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

demo = create_demo()

if __name__ == "__main__":
    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#e8f0fe",
            c100="#c2d7f7",
            c200="#9bbcf0",
            c300="#6f9ee6",
            c400="#4a82db",
            c500="#1e3a5f",
            c600="#1a3352",
            c700="#162a45",
            c800="#122238",
            c900="#0e192b",
            c950="#0a101e",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f0eb",
            c100="#e0d6c9",
            c200="#cbbca8",
            c300="#b3a085",
            c400="#a08b6a",
            c500="#8b6f47",
            c600="#7a623e",
            c700="#695435",
            c800="#58462c",
            c900="#473923",
            c950="#362b1a",
        ),
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_md,
    )
    port = int(os.environ.get("GRADIO_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=CUSTOM_CSS,
    )
