# Deployment Guide

This document covers exporting trained models to optimized formats, running
inference on CPU and GPU, deploying the Gradio dashboard, and serving options.

## 1. Model export

The Ultralytics pipeline (installed via `requirements.txt`) exports trained
weights to ONNX, TensorRT, and other formats. Export any checkpoint produced
by training (e.g. `models/yolo26n.pt` or `runs/yolo26m/weights/best.pt`).

### ONNX

```bash
# CLI
yolo export model=models/yolo26n.pt format=onnx imgsz=640

# Python API (equivalent)
python -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt').export(format='onnx', imgsz=640)"
```

The exported file is written next to the source weights (e.g.
`models/yolo26n.onnx`). Use `opset=17` if you need a specific ONNX opset, and
`dynamic=True` for dynamic input shapes (at the cost of some latency).

### TensorRT

```bash
# Requires an NVIDIA GPU and the `tensorrt` Python package
yolo export model=models/yolo26n.pt format=engine device=0 imgsz=640
```

`format=engine` produces a TensorRT engine optimized for the exact GPU it was
built on; engines are not portable across GPU models or driver versions.
`half=True` enables FP16 inference (up to ~2x throughput on modern GPUs).

### Other formats

`format=torchscript`, `format=tflite`, and `format=openvino` are supported by
Ultralytics as well. For edge/CPU deployments, OpenVINO (`format=openvino`)
typically gives the best CPU throughput without an NVIDIA GPU.

## 2. CPU and GPU inference

### Repo inference script

`scripts/inference.py` runs a trained model over an image file, a directory of
images, or a glob pattern:

```bash
# CPU (default)
python scripts/inference.py --model models/yolo26n.pt \
    --source data/sample/ --save-img --save-csv

# GPU — pass a device id
python scripts/inference.py --model models/yolo26n.pt \
    --source data/sample/ --save-img --device 0
```

The exported formats are loaded the same way — point `--model` at the `.onnx`
or `.engine` file. A default `--device cpu` keeps the script dependency-free;
use `--imgsz` to trade throughput against small-object recall.

### Video and webcam

The repository script is image-based. For video files or webcam streams, use
the Ultralytics CLI that ships with the installed dependency:

```bash
# Video file
yolo predict model=models/yolo26n.pt source=video.mp4 conf=0.25 save=True

# Webcam (device 0)
yolo predict model=models/yolo26n.pt source=0 conf=0.25 show=True

# Folder of images (batch, GPU)
yolo predict model=models/yolo26n.pt source=data/sample/ device=0 save=True
```

### Throughput notes

- Batch inference: Ultralytics batches natively when `source` is a directory.
  Larger batches amortize GPU launch overhead; CPU users benefit less.
- Precision: FP16 (`half=True` in export, or `amp` during training) roughly
  halves TensorRT memory bandwidth without meaningful mAP change on this task.
- Input size: the models were trained at 640x640. Larger `--imgsz` (e.g. 1280)
  improves small-hole recall at the cost of latency (see the image-size
  ablation in `docs/ablation-imgsz.md`); smaller sizes are faster but miss
  more small objects.

## 3. Dashboard deployment

The Gradio dashboard (`scripts/dashboard.py`) binds `0.0.0.0` on port 7860 by
default:

```bash
# Default port
python scripts/dashboard.py

# Custom port
GRADIO_PORT=8080 python scripts/dashboard.py
```

Notes:

- The dashboard expects trained checkpoints under `runs/{experiment}/weights/`
  and evaluation artifacts under `runs/eval/` (produced by `scripts/evaluate.py`).
- For a public endpoint behind a reverse proxy (nginx/Caddy), proxy to
  `http://127.0.0.1:7860` and enable WebSocket upgrade headers — Gradio uses
  WebSockets for live progress.
- For a temporary shareable link, launch with `share=True` in
  `scripts/dashboard.py`; this routes through a Gradio-hosted tunnel and is not
  recommended for sensitive data.

### Containerized

The repository `Dockerfile` builds a CPU image that installs dependencies and
runs `scripts/inference.py --help` by default. To serve the dashboard instead,
override the command at runtime:

```bash
docker build -t hole-detection .
docker run --rm -p 7860:7860 hole-detection python scripts/dashboard.py
```

## 4. Serving suggestions

For production serving at scale, prefer an inference server over per-request
process spawns:

- **ONNX Runtime** — export with `format=onnx` and serve via
  `onnxruntime` (CPU/GPU) or `onnxruntime-gpu`. Simple, dependency-light,
  good for batch REST endpoints.
- **TensorRT + Triton** — export with `format=engine`, then deploy via NVIDIA
  Triton Inference Server with the TensorRT backend for multi-model,
  high-concurrency GPU serving. Recommended when throughput matters.
- **Ultralytics FastAPI pattern** — load the model once into a module-level
  `YOLO` object (as `scripts/dashboard.py` already does) and call `model(...)`
  per request; this avoids re-loading weights on every call.

For all options, consider a preprocessing step that tiles large satellite
scenes into 640x640 inference tiles (matching the training distribution) and
re-assembles detections afterward. Geo-referenced output (GeoJSON) can be
produced from the `detections.csv` that `scripts/inference.py` writes, given
per-tile georeferencing metadata.
