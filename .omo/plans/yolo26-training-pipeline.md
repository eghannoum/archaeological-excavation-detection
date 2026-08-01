# YOLO26 Training + Inference + Evaluation + Dashboard Pipeline

## TL;DR
Train a YOLO26m model on the prepared archaeological hole dataset (352 train images, single class "hole"), build an evaluation pipeline, batch inference script, and a Gradio visualization dashboard.

## Dependencies
```
Phase 1 (Install) → Phase 2 (Train) → (Phase 3 Evaluate, Phase 4 Inference, Phase 5 Dashboard)
Phases 3-5 script writing can start after Phase 1; full testing requires Phase 2 output.
```

## Execution Strategy
```
Wave 1 (Sequential):
├── Phase 1: Install ML stack (PyTorch + Ultralytics + Gradio)

Wave 2 (Sequential — needs Phase 1):
├── Phase 2: Train YOLO26m model

Wave 3 (Parallel — scripts written after Phase 1, run after Phase 2):
├── Phase 3: Evaluation pipeline (test set metrics)
├── Phase 4: Inference script (batch processing)
├── Phase 5: Gradio dashboard

Wave 4 (Parallel):
├── F1-F4: Final verification wave
```

## TODOs

- [x] 1. Install ML stack (PyTorch + Ultralytics + Gradio + dependencies)

  **What to do**:
  - Install PyTorch with CUDA support:
    ```powershell
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
  - Install Ultralytics: `python -m pip install ultralytics`
  - Install Gradio for dashboard: `python -m pip install gradio`
  - Install support libs: `python -m pip install pandas matplotlib seaborn`
  - Verify: `python -c "import torch; import ultralytics; import gradio; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, Ultralytics {ultralytics.__version__}, Gradio {gradio.__version__}')"`
  - Verify GPU: `python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"`
  - Create `runs/` directory for model outputs

  **Must NOT do**:
  - Do not install CPU-only torch (needs CUDA for training speed)

- [x] 2. Train YOLO26m model

  **What to do**:
  Create `scripts/train.py` that:
  - Uses Ultralytics YOLO API: `from ultralytics import YOLO`
  - Loads pretrained `yolo26m.pt` weights
  - Trains on `dataset/data.yaml` with relative paths (run from project root)
  - Training params:
    - `imgsz=640` (balanced for RTX 5070 8GB)
    - `batch=16` (adjust down to 8 if OOM)
    - `epochs=100` (with early stopping patience=20)
    - `workers=4`
    - `device=0` (GPU)
    - `patience=20` (early stopping)
    - `save=True`, `save_period=10`
    - `project='runs/train'`, `name='yolo26m-hole'`
  - Run training with: `python scripts/train.py`
  - Capture training curves and metrics
  - Save best model to `runs/train/yolo26m-hole/weights/best.pt`

  **Must NOT do**:
  - Do not use CPU (will be impractically slow)
  - Do not modify the dataset

- [x] 3. Build evaluation pipeline

  **What to do**:
  Create `scripts/evaluate.py` that:
  - Loads the trained model from `runs/train/yolo26m-hole/weights/best.pt`
  - Runs `model.val(data='dataset/data.yaml', split='test')` on the test set
  - Extracts: mAP50, mAP50-95, precision, recall, F1-score
  - Generates and saves:
    - `runs/eval/confusion_matrix.png`
    - `runs/eval/results.json` (all metrics serialized)
    - `runs/eval/val_batch*.jpg` (sample predictions on test images)
  - Prints summary table of all metrics
  - Handles model not found with clear error message

- [x] 4. Build inference pipeline

  **What to do**:
  Create `scripts/inference.py` that:
  - Argparse: `--model` (path to .pt), `--source` (image/dir), `--output` (dir), `--conf` (default 0.25), `--save-txt`, `--save-img`, `--visualize`
  - Supports: single image, directory of images, glob patterns
  - For each image:
    - Run `model.predict()` with configured confidence threshold
    - Save annotated image to `--output` dir
    - Save detections as CSV (filename, class, confidence, x1, y1, x2, y2)
    - Print per-image detection summary
  - Prints aggregate summary (total images, total detections, mean confidence)
  - Default model: `runs/train/yolo26m-hole/weights/best.pt`
  - Example: `python scripts/inference.py --source new_images/ --output detections/ --save-img --save-txt`

- [x] 5. Build Gradio visualization dashboard

  **What to do**:
  Create `scripts/dashboard.py` that:
  - Uses Gradio Blocks/Interface
  - **Tab 1: Model Metrics**
    - Display confusion matrix image from `runs/eval/confusion_matrix.png`
    - Display metrics table (mAP50, mAP50-95, precision, recall, F1)
    - Show training loss curves (from `runs/train/yolo26m-hole/results.csv`)
  - **Tab 2: Interactive Inference**
    - Upload image widget
    - Run inference on uploaded image
    - Display annotated result with bounding boxes
    - Show detection table (class, confidence, coordinates)
    - Adjustable confidence slider (0.1-0.9)
  - **Tab 3: Batch Inference**
    - Select source directory
    - Configure output directory
    - Run and display results summary
  - Run with: `python scripts/dashboard.py`
  - Default port: 7860 (Gradio default)
  - Handle missing model/results gracefully with placeholder messages

---

## Final Verification Wave

- [x] F1. **Training Verification** — Verify model trained successfully
  - Check `runs/train/yolo26m-hole/weights/best.pt` exists
  - Check training completed without OOM/error
  - Verify `runs/train/yolo26m-hole/results.csv` has loss curves
  - Evidence: `.omo/evidence/f1-training.txt`

- [x] F2. **Evaluation Verification** — Verify evaluation pipeline
  - Run `python scripts/evaluate.py` and check exit code 0
  - Verify `runs/eval/results.json` exists with valid metrics
  - Verify mAP50 > 0 (model learned something)
  - Evidence: `.omo/evidence/f2-eval.txt`

- [x] F3. **Inference Verification** — Verify inference works
  - Run inference on a test image: `python scripts/inference.py --source dataset/images/test/ --output runs/inference-test --save-img --conf 0.25`
  - Verify annotated images contain bounding boxes
  - Verify detection CSV is valid
  - Evidence: `.omo/evidence/f3-inference.txt`

- [x] F4. **Dashboard Verification** — Verify dashboard launches
  - Verify `python scripts/dashboard.py` passes syntax check
  - Verify Gradio app structure loads without import errors
  - Evidence: `.omo/evidence/f4-dashboard.txt`
