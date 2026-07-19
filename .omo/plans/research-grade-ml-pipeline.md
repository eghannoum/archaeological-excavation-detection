# Research-Grade ML Pipeline: Archaeological Hole Detection with YOLO26

## TL;DR

> **Quick Summary**: Upgrade the existing YOLO archaeological hole detection pipeline to a publication-quality research-grade ML system with full model comparison suite (YOLO26 n/s/m/l/x + YOLOv8/v11 + Faster R-CNN + DETR), 3-fold cross-validation, heavy augmentations, Hydra+MLflow infrastructure, systematic hyperparameter optimization, deep evaluation, and paper preparation.
>
> **Deliverables**:
> - Working GPU (CUDA on RTX 5070 Blackwell) with verified sm_120 support
> - Hydra config system + MLflow experiment tracking across 40+ training runs
> - Heavy augmentation pipeline (Albumentations, Mosaic, MixUp, CutMix)
> - Hyperparameter optimization (50 Optuna trials on YOLO26m)
> - Systematic training: 5 YOLO26 variants × 3-fold CV + 3 comparison detectors × 3-fold CV + 3 ablation studies
> - Deep evaluation: calibration curves, error analysis, statistical significance
> - Paper-ready metrics tables and publication-quality figures
>
> **Estimated Effort**: XL (40+ training runs, 7 phases)
> **Parallel Execution**: YES — multi-wave with GPU as bottleneck
> **Critical Path**: GPU fix → Hydra infra → HPO → YOLO26m training → ablations → deep eval → paper

---

## Context

### Original Request
Upgrade the existing YOLO archaeological hole detection project to a publication-quality research-grade ML pipeline. The current baseline is a YOLO26m model (21.7M params) trained on CPU at 640px, achieving mAP50=0.404 / mAP50-95=0.117 / Precision=0.482 / Recall=0.460 / F1=0.471 on a held-out test set of 40 images. The dataset has 432 images (352 train + 40 val + 40 test) across 108 parent scenes, single class "hole", 640px satellite tiles.

The RTX 5070 Laptop GPU (Blackwell sm_120, 8GB VRAM) is currently unused because PyTorch 2.4.1+cpu doesn't support sm_120.

### Interview Summary
**Key Decisions**:
- **Publication venue**: TBD — build the best possible pipeline now, decide venue later
- **Model comparison**: Full comparison suite — YOLO26 n/s/m/l/x + YOLOv8/v11 + Faster R-CNN + DETR
- **Dataset strategy**: Heavy augmentation to multiply effective dataset (Mosaic, MixUp, CutMix, Albumentations)
- **Cross-validation**: 3-fold split at parent-scene level for statistical rigor
- **Experiment budget**: Exhaustive — 40+ training runs with systematic ablations
- **ML infrastructure**: Hydra configs + MLflow experiment tracking
- **GPU strategy**: Fix local CUDA first (reinstall PyTorch with cu128 wheels for sm_120 support)
- **Timeline**: No deadline — iterative improvement
- **Test strategy**: Agent-Executed QA (no unit tests — research pipeline). Verify GPU works, verify MLflow logs, verify training completes, verify metrics.

**Research Findings**:
- PyTorch 2.7+ with CUDA 12.8 wheels (`--index-url https://download.pytorch.org/whl/cu128`) includes sm_120 (Blackwell) support
- MLflow 3.14.0 and Optuna 4.9.0 are ALREADY installed — skip installation
- Only yolo26n.pt (5.3MB) and yolo26m.pt (42.2MB) weights exist on disk — s/l/x need downloading
- NVIDIA driver 592.27, CUDA 13.1 runtime (WDDM mode, not TCC — expect 5-15% training overhead)
- RTX 5070 has 8GB VRAM — YOLO26l/x may require reduced batch size or gradient accumulation
- Windows path handling needed for Hydra/MLflow compat

### Momus Review
**Verdict**: APPROVE — ready for execution. 5 minor issues found (all MEDIUM/LOW), all resolved:
1. [MEDIUM] **Task 3**: Removed l-variant YOLOv8/v11 downloads (only train m variants) ✅
2. [MEDIUM] **Task 20**: Flipped to torchvision Faster R-CNN as primary (Detectron2 has no Windows/cu128 wheels) ✅
3. [LOW] **Task 21**: Standardized DETR to 100 epochs (consistent with all models) ✅
4. [LOW] **Task 4**: Removed StructuredConfig/pydantic requirement (plain Hydra YAML sufficient) ✅
5. [LOW] **Task 9**: Added specific Albumentations integration approach (subclass YOLODataset) ✅

**Strengths noted**: GPU validation as Phase 0 gate, every task has QA scenarios with commands, test set sealed, OOM risks identified, commit strategy planned, statistical rigor acknowledged.

### Metis Review
**Identified Gaps** (addressed):
- Publication venue → Resolved: TBD, build best pipeline
- Dataset size → Resolved: heavy augmentation strategy
- Comparison detectors → Resolved: full suite (YOLOv8/v11 + FRCNN + DETR)
- Experiment budget → Resolved: exhaustive 40+ runs
- GPU assumption → Noted: must verify cu128 + sm_120 as Phase 0 gate

---

## Work Objectives

### Core Objective
Upgrade the YOLO archaeological hole detection project to a publication-quality research-grade ML pipeline with systematic model comparison, 3-fold CV, heavy augmentations, full experiment tracking, and deep evaluation — producing paper-ready results and publication-quality figures.

### Concrete Deliverables
- [ ] Working GPU environment (sm_120 CUDA support verified)
- [ ] Hydra config system (`configs/`) with structured YAML for all experiments
- [ ] MLflow project with all 40+ experiments logged
- [ ] Heavy augmentation pipeline integrated with training
- [ ] HPO results (optuna study with best hyperparams for YOLO26m)
- [ ] 15 trained YOLO26 models (5 variants × 3 folds) with metrics
- [ ] 9 trained comparison models (3 detectors × 3 folds)
- [ ] Ablation study results (image size, optimizer, augmentation)
- [ ] Calibration curves + PR curves + F1-confidence curves for all models
- [ ] Statistical significance tests with confidence intervals
- [ ] Error analysis report on best model
- [ ] Paper-ready metrics tables (model × metric × mean±std) with highlighting
- [ ] 4+ publication-quality figures
- [ ] Complete paper draft with abstract, intro, methods, results, discussion
- [ ] Reproducibility instructions (`requirements.txt` + README)

### Definition of Done
- [ ] All 40+ training runs complete with metrics logged to MLflow
- [ ] Best model achieves ≥50% relative mAP50 improvement over CPU baseline (or documented ceiling)
- [ ] Paper-ready metrics table with mean±std across folds
- [ ] All verification commands pass (see final checklist)
- [ ] git commit with all code changes + README

### Must Have
- Phase 0 (GPU validation) must pass before any GPU training begins
- All training runs must log to MLflow with full hyperparameter config
- 3-fold cross-validation split at parent-scene level (no data leakage)
- Fixed seeds for reproducibility (`torch.manual_seed`, `np.random.seed`)
- Test set (40 images) sealed — NO evaluation on test set until Phase 5
- All 5 YOLO26 variants (n/s/m/l/x) trained and compared
- At least 2 comparison detectors (YOLOv8/v11 + one non-YOLO)
- Heavy augmentation pipeline (Albumentations + Ultralytics built-ins)
- Confidence intervals on all primary metrics
- Reproducibility checklist: same config + seed → identical metrics

### Must NOT Have (Guardrails)
- No Docker containerization (unless explicitly requested later)
- No new data collection (use existing 432 images with augmentations)
- No production deployment or real-time inference
- No model pruning, quantization, or optimization for speed
- No web-based visualization beyond existing Gradio dashboard
- No experiment exceeding the defined budget without re-authorization
- No test set evaluation before Phase 5 (sealed until final evaluation)
- No hyperparameter tuning based on test set metrics
- No comparing against non-detection architectures (ViT, etc.)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (research pipeline)
- **Automated tests**: None — research pipeline, not production code
- **Agent-Executed QA**: PRIMARY verification method for all tasks

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **GPU Verification**: `python -c "import torch; t=torch.rand(2,2).cuda(); print(t @ t)"` — prints 2×2 matrix
- **Training Verification**: `yolo train model=yolo26n.pt data=dataset/data.yaml epochs=1 imgsz=640 device=0` — exits 0
- **MLflow Verification**: `mlflow experiments list` — shows experiment
- **Metrics Verification**: Read MLflow run metrics JSON — numeric values present
- **Config Verification**: `python -c "import yaml; cfg=yaml.safe_load(open('configs/default.yaml')); print(cfg)"`

---

## Execution Strategy

### Parallel Execution Waves

> GPU is the bottleneck resource — most training tasks run sequentially or in small groups.
> Infrastructure, data, and evaluation tasks maximize CPU parallelism.

```
Wave 0 (GPU - critical path, sequential):
├── Task 1: PyTorch cu128 reinstall + CUDA/sm_120 verification [quick]
├── Task 2: Install dependencies + model weights [quick]
└── Task 3: Weight download + verification [quick]

Wave 1 (Infrastructure - partially parallel):
├── Task 4: Hydra config system [unspecified-high]
├── Task 5: MLflow experiment tracking integration [unspecified-high]
├── Task 6: Directory structure + helper scripts [unspecified-high]
└── Task 7: GPU benchmark + memory profiling [quick]

Wave 2 (Data Pipeline):
├── Task 8: Dataset analysis report [unspecified-high]
├── Task 9: Albumentations pipeline (light + heavy configs) [unspecified-high]
└── Task 10: Augmentation visualization + validation [visual-engineering]

Wave 3 (Hyperparameter Optimization - GPU bottleneck):
├── Task 11: Optuna HPO integration for YOLO26m [unspecified-high]
├── Task 12: Run 50 HPO trials (GPU-bound, sequential) [deep]
└── Task 13: Export best hyperparameters [quick]

Wave 4 (YOLO26 Systematic Training - GPU bottleneck, sequential):
├── Task 14: YOLO26n 3-fold CV training [deep]
├── Task 15: YOLO26s 3-fold CV training [deep]
├── Task 16: YOLO26m 3-fold CV training (with HPO best params) [deep]
├── Task 17: YOLO26l 3-fold CV training [deep]
└── Task 18: YOLO26x 3-fold CV training [deep]

Wave 5 (Comparison Detectors):
├── Task 19: YOLOv8/v11 3-fold CV training [deep]
├── Task 20: Faster R-CNN 3-fold CV training [deep]
└── Task 21: DETR 3-fold CV training [deep]

Wave 6 (Ablation Studies - partially parallel, GPU bottleneck):
├── Task 22: Image size ablation (320 vs 640 vs 1280) [deep]
├── Task 23: Optimizer ablation (AdamW vs SGD) [deep]
└── Task 24: Augmentation strategy ablation (none vs light vs heavy) [deep]

Wave 7 (Deep Evaluation - CPU parallel):
├── Task 25: Calibration curves + confidence analysis [unspecified-high]
├── Task 26: PR curves + F1-confidence curves [unspecified-high]
├── Task 27: Error analysis on best model [unspecified-high]
├── Task 28: Statistical significance testing [unspecified-high]
└── Task 29: Final test set evaluation [unspecified-high]

Wave 8 (Paper Preparation):
├── Task 30: Metrics tables (model × metric × mean±std) [writing]
├── Task 31: Publication-quality figures [visual-engineering]
├── Task 32: Ablation study tables [writing]
└── Task 33: Paper draft + reproducibility instructions [writing]

Wave FINAL (Verification - 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix
- **1**: - - 2, 3
- **2**: 1 - 3
- **3**: 2 - 4-7
- **4**: 3 - 6, 8, 11
- **5**: 3 - 6, 11, 13
- **6**: 4, 5 - 8, 9
- **7**: 3 - 12
- **8**: 6 - 9
- **9**: 6, 8 - 10, 11
- **10**: 9 - 12
- **11**: 4, 5, 9 - 12
- **12**: 1, 7, 10, 11 - 13, 14-18
- **13**: 12 - 16
- **14**: 12 - -
- **15**: 12 - -
- **16**: 12, 13 - -
- **17**: 12 - -
- **18**: 12 - -
- **19**: 2, 10, 13 - -
- **20**: 2, 10 - -
- **21**: 2, 10 - -
- **22**: 14-18 - 25-29
- **23**: 14-18 - 25-29
- **24**: 14-18 - 25-29
- **25**: 14-24 - 27, 28, 29
- **26**: 14-24 - 27, 28
- **27**: 25, 26 - 28
- **28**: 27 - 29
- **29**: 25, 28 - 30-33
- **30**: 29 - 33
- **31**: 29 - 33
- **32**: 29 - 33
- **33**: 30-32 - F1-F4

### Agent Dispatch Summary
- **Wave 0**: 3 tasks — T1 → `quick`, T2 → `quick`, T3 → `quick`
- **Wave 1**: 4 tasks — T4 → `unspecified-high`, T5 → `unspecified-high`, T6 → `unspecified-high`, T7 → `quick`
- **Wave 2**: 3 tasks — T8 → `unspecified-high`, T9 → `unspecified-high`, T10 → `visual-engineering`
- **Wave 3**: 3 tasks — T11 → `unspecified-high`, T12 → `deep`, T13 → `quick`
- **Wave 4-6**: 11 tasks — T14-T18, T19-T21, T22-T24 all `deep`
- **Wave 7**: 5 tasks — T25-T29 → `unspecified-high`
- **Wave 8**: 4 tasks — T30-T32 → `writing`/`visual-engineering`, T33 → `writing`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. **Install PyTorch cu128 + Verify GPU/CUDA/sm_120 Support**

  **What to do**:
  - Reinstall PyTorch with CUDA 12.8+ wheels: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
  - Verify CUDA is available and sm_120 is in arch list
  - If cu128 wheels work: proceed. If not, try nightly: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`
  - If nightly fails: document the failure and fall back to CPU-only for development (cloud GPU for final runs)
  - Pin PyTorch version in `requirements.txt`
  - Record GPU details: compute capability, VRAM, driver version, CUDA runtime

  **Must NOT do**:
  - Do NOT install from pytorch.org default (CPU-only) — must use cu128 index
  - Do NOT proceed to GPU training tasks if this task fails

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Well-defined installation and verification steps, no deep reasoning needed
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: NO (critical path)
  - **Blocks**: Tasks 2, 3, 12, 14-24

  **Acceptance Criteria**:
  - [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
  - [ ] `python -c "import torch; print('sm_120' in torch.cuda.get_arch_list())"` → `True`
  - [ ] `python -c "import torch; t=torch.rand(2,2).cuda(); print(t @ t)"` → prints 2×2 matrix
  - [ ] `nvidia-smi` shows CUDA version ≥ 12.8

  **QA Scenarios**:

  ```
  Scenario: GPU CUDA availability
    Tool: Bash
    Preconditions: PyTorch cu128 installed
    Steps:
      1. Run: python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0))"
      2. Parse output
    Expected Result: True, list contains "sm_120", device name contains "RTX 5070" or similar
    Evidence: .omo/evidence/task-1-cuda-verify.txt

  Scenario: CUDA tensor operation (failure case)
    Tool: Bash
    Preconditions: PyTorch cu128 installed
    Steps:
      1. Run: python -c "import torch; t=torch.rand(2,2).cuda(); print(t @ t)"
    Expected Result: Prints 2×2 tensor without error
    Evidence: .omo/evidence/task-1-tensor-ops.txt
  ```

  **Evidence to Capture**:
  - [ ] `nvidia-smi` output
  - [ ] CUDA tensor operation result
  - [ ] `torch.cuda.get_arch_list()` showing sm_120

  **Commit**: YES
  - Message: `env(pytorch): reinstall with cu128 wheels for Blackwell sm_120 support`
  - Files: `requirements.txt`
  - Pre-commit: `python -c "import torch; assert torch.cuda.is_available()"`

- [x] 2. **Install Dependencies + Detection Frameworks**

  **What to do**:
  - Install Ultralytics: `pip install ultralytics`
  - Install Detectron2 (for Faster R-CNN): wheels at `https://github.com/facebookresearch/detectron2` — use pre-built for PyTorch 2.7+
  - Install DETR dependencies (torchvision deps already satisfied, DETR is pure torchvision-based)
  - Install Albumentations: `pip install albumentations`
  - Pin ALL package versions in `requirements.txt`
  - Verify each import works: `ultralytics`, `detectron2` (if installed), `albumentations`

  **Must NOT do**:
  - Do NOT install packages that conflict with PyTorch cu128 version
  - Do NOT attempt to compile Detectron2 from source if no pre-built wheel exists

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Package installation with version pinning, straightforward

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1)
  - **Blocks**: Tasks 3, 20, 21
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] `python -c "import ultralytics; print(ultralytics.__version__)"` → version printed
  - [ ] `python -c "import albumentations; print(albumentations.__version__)"` → version printed
  - [ ] (If Detectron2) `python -c "import detectron2; print(detectron2.__version__)"` → version printed
  - [ ] `requirements.txt` has ALL pinned versions

  **QA Scenarios**:

  ```
  Scenario: Verify imports work
    Tool: Bash
    Preconditions: Task 1 completed, packages installed
    Steps:
      1. Run: python -c "import ultralytics; print('ultralytics OK')"
      2. Run: python -c "import albumentations; print('albumentations OK')"
    Expected Result: Both print OK
    Evidence: .omo/evidence/task-2-imports.txt

  Scenario: Verify YOLO model loads
    Tool: Bash
    Preconditions: yolo26n.pt exists
    Steps:
      1. Run: python -c "from ultralytics import YOLO; m=YOLO('yolo26n.pt'); print('model loaded')"
    Expected Result: Model loads without error
    Evidence: .omo/evidence/task-2-yolo-load.txt
  ```

  **Evidence to Capture**:
  - [ ] Package version listing
  - [ ] Import verification output
  - [ ] `requirements.txt` file

  **Commit**: YES (with Task 1)
  - Message: `env: install ultralytics + albumentations dependencies`
  - Files: `requirements.txt`

- [x] 3. **Download Model Weights + Verify**

  **What to do**:
  - Check which weights exist on disk: `ls models/*.pt`
  - Download missing YOLO26 variants: `yolo26s.pt` (22.5M), `yolo26l.pt` (52.9M), `yolo26x.pt` (99.1M)
  - Download YOLOv8/v11 weights for comparison: `yolov8m.pt`, `yolov11m.pt`
  - For DETR: note that it uses `torch.hub.load('facebookresearch/detr', 'detr_resnet50')` — no separate weight download
  - For Faster R-CNN: note that it uses `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` — downloads on first use
  - Verify each weight file loads correctly
  - Organize weights in `models/` directory

  **Must NOT do**:
  - Do NOT download redundant variants (e.g., don't download yolo26m if it already exists)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple download + verify operations

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 2)
  - **Blocks**: Tasks 4, 5, 6, 7, 14-19
  - **Blocked By**: Task 2

  **Acceptance Criteria**:
  - [ ] All 5 YOLO26 weights exist: `models/yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`
  - [ ] YOLOv8/v11 weights exist: `models/yolov8m.pt`, `models/yolov11m.pt`
  - [ ] Each weight loads: `python -c "from ultralytics import YOLO; YOLO('models/...')"` passes

  **QA Scenarios**:

  ```
  Scenario: Verify all YOLO26 weights
    Tool: Bash
    Preconditions: Weights downloaded
    Steps:
      1. Run: for v in n s m l x; do python -c "from ultralytics import YOLO; YOLO('models/yolo26$v.pt')" && echo "$v OK"; done
    Expected Result: All 5 variants print "OK"
    Evidence: .omo/evidence/task-3-weights-yolo26.txt

  Scenario: Verify comparison detector weights
    Tool: Bash
    Preconditions: Weights downloaded
    Steps:
      1. Run: for v in v8m v8l v11m v11l; do python -c "from ultralytics import YOLO; YOLO('models/yolo$v.pt')" && echo "$v OK"; done
    Expected Result: All 4 variants print "OK"
    Evidence: .omo/evidence/task-3-weights-compare.txt
  ```

  **Evidence to Capture**:
  - [ ] File listing of `models/` with sizes
  - [ ] Weight load verification output

  **Commit**: YES (with Task 2)
  - Files: (git-lfs or dvc if weights tracked) or just note in README
  - Pre-commit: verification script runs

- [x] 4. **Create Hydra Config System**

  **What to do**:
  - Create `configs/` directory with structured Hydra YAML configs
  - `configs/default.yaml`: top-level config with `defaults` list
  - `configs/data/default.yaml`: dataset paths, image size, augmentations
  - `configs/model/default.yaml`: model architecture, pretrained weights path
  - `configs/training/default.yaml`: optimizer, LR, batch size, epochs, seed
  - `configs/evaluation/default.yaml`: confidence threshold, IoU thresholds
  - `configs/experiment/default.yaml`: experiment name, tags, description
  - Create `configs/experiment/` overrides for each experiment type:
    - `yolo26n.yaml`, `yolo26s.yaml`, `yolo26m.yaml`, `yolo26l.yaml`, `yolo26x.yaml`
    - `yolov8.yaml`, `yolov11.yaml`
    - `faster_rcnn.yaml`, `detr.yaml`
  - Create `configs/ablation/` configs for ablation studies:
    - `imgsz320.yaml`, `imgsz640.yaml`, `imgsz1280.yaml`
    - `optimizer_adamw.yaml`, `optimizer_sgd.yaml`
    - `augmentation_none.yaml`, `augmentation_light.yaml`, `augmentation_heavy.yaml`
  - Create `configs/hpo/default.yaml`: Optuna config (n_trials, sampler, pruner)
  - Configs use plain Hydra YAML with OmegaConf's built-in interpolation — no StructuredConfig or pydantic needed for a research pipeline
  - Add `--config-dir` CLI entrypoint via `scripts/train.py` (refactored)
  - Test: `python train.py --config-name=default data.image_size=640` loads correctly

  **Must NOT do**:
  - Do NOT hardcode paths in configs — use `${hydra:runtime.cwd}` or relative paths
  - Do NOT create redundant nested defaults (keep flat where possible)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding Hydra's composition system and creating ~20 structured YAML files with proper inheritance
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 5)
  - **Parallel Group**: Wave 1 (with Tasks 5, 6, 7)
  - **Blocks**: Tasks 6, 8, 11
  - **Blocked By**: Task 3

  **Acceptance Criteria**:
  - [ ] `python train.py --help` shows all config groups
  - [ ] `python train.py --config-name=default` resolves without error
  - [ ] `python train.py model=yolo26m data.image_size=640` overrides correctly
  - [ ] Config structure documented in README

  **QA Scenarios**:

  ```
  Scenario: Verify hydra config loads
    Tool: Bash
    Preconditions: Hydra configs created
    Steps:
      1. Run: python scripts/train.py --config-name=default --info (dry-run flag)
    Expected Result: Config printed without error, shows composed values
    Evidence: .omo/evidence/task-4-config-load.txt

  Scenario: Verify config override works
    Tool: Bash
    Preconditions: Configs created
    Steps:
      1. Run: python scripts/train.py --config-name=default data.image_size=320 --info
      2. Check image_size=320 in output
    Expected Result: Override applied correctly
    Evidence: .omo/evidence/task-4-config-override.txt
  ```

  **Evidence to Capture**:
  - [ ] Hydra config tree listing
  - [ ] Config load verification output

  **Commit**: YES
  - Message: `feat(configs): add Hydra structured config system for research experiments`
  - Files: `configs/**`, `scripts/train.py` (refactored)

- [x] 5. **Set Up MLflow Experiment Tracking**

  **What to do**:
  - Verify MLflow is already installed (v3.14.0 confirmed)
  - Create `scripts/mlflow_utils.py`: helper module for logging
    - `log_config(cfg)`: log Hydra config as MLflow params
    - `log_metrics(metrics_dict, step)`: log metrics with step
    - `log_artifact(path)`: log file/dir as artifact
    - `log_model_checkpoint(path)`: log model weights as artifact
    - `init_experiment(experiment_name, tags)`: create/get experiment
  - Set up MLflow tracking URI to local `mlruns/` directory
  - Configure file-based tracking (no server needed)
  - Create experiment tagging taxonomy:
    - `model_family`: yolov26, yolov8, yolov11, faster_rcnn, detr
    - `model_scale`: n, s, m, l, x
    - `cv_fold`: 0, 1, 2
    - `experiment_type`: hpo, training, ablation, evaluation
    - `augmentation`: none, light, heavy
    - `image_size`: 320, 640, 1280
  - Add MLflow logging to `scripts/train.py`
  - Test: run 1-epoch training → verify MLflow run created with params + metrics + artifacts

  **Must NOT do**:
  - Do NOT use MLflow server mode (local file tracking only)
  - Do NOT log redundant params (avoid logging both Hydra config and per-param)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding MLflow API and designing a structured experiment taxonomy

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 4)
  - **Parallel Group**: Wave 1 (with Tasks 4, 6, 7)
  - **Blocks**: Tasks 6, 11, 13
  - **Blocked By**: Task 3

  **Acceptance Criteria**:
  - [ ] `mlflow experiments list` shows the experiment
  - [ ] After 1-epoch training, MLflow run exists with params + metrics + artifacts
  - [ ] `mlflow runs list --experiment-id X` shows the run
  - [ ] Artifact includes model checkpoint (best.pt)

  **QA Scenarios**:

  ```
  Scenario: Verify MLflow logging after 1-epoch training
    Tool: Bash
    Preconditions: MLflow utils created, yolo26n.pt exists
    Steps:
      1. Run: python scripts/train.py --config-name=default model=yolo26n epochs=1
      2. Run: mlflow runs list --experiment-id 1
    Expected Result: MLflow run exists with params, metrics, artifacts
    Evidence: .omo/evidence/task-5-mlflow-run.txt

  Scenario: Verify MLflow artifacts saved
    Tool: Bash
    Preconditions: Previous step completed
    Steps:
      1. Run: mlflow artifacts list --run-id <run-id>
      2. Check for model checkpoint in artifacts
    Expected Result: Artifacts directory contains model weights
    Evidence: .omo/evidence/task-5-mlflow-artifacts.txt
  ```

  **Evidence to Capture**:
  - [ ] MLflow experiment listing
  - [ ] MLflow run with params + metrics
  - [ ] Artifact directory listing

  **Commit**: YES (with Task 4)
  - Message: `feat(mlflow): add experiment tracking with structured taxonomy`
  - Files: `scripts/mlflow_utils.py`

- [x] 6. **Create Directory Structure + Helper Scripts**

  **What to do**:
  - Create top-level research directory structure:
    ```
    configs/           (from Task 4)
    models/            (model weights)
    scripts/           (training, evaluation, helpers)
      train.py         (refactored Hydra+MLflow version)
      evaluate.py      (refactored for MLflow logging)
      inference.py     (batch inference)
      dashboard.py     (Gradio visualization)
      hpo.py           (Optuna hyperparameter optimization)
      train_cv.py      (k-fold CV wrapper)
      dataset_analysis.py  (dataset statistics)
      paper_metrics.py     (summary tables from MLflow)
    mlruns/            (MLflow tracking data — gitignored)
    experiments/       (per-experiment run outputs — gitignored)
    notebooks/         (analysis notebooks)
    outputs/           (Hydra outputs — gitignored)
    runs/              (Ultralytics runs — gitignored)
    .omo/evidence/     (QA evidence)
    ```
  - Create `scripts/train.py` entrypoint that:
    - Loads Hydra config
    - Initializes MLflow run
    - Sets random seeds
    - Trains model (Ultralytics API or custom training loop)
    - Logs metrics + model to MLflow
  - Create `scripts/train_cv.py` for 3-fold CV training:
    - Creates fold datasets from parent-scene groups
    - Runs train.py for each fold with `cv_fold=0/1/2` override
    - Aggregates cross-fold metrics
  - Add `scripts/__init__.py` and document usage in README
  - Add `mlruns/`, `experiments/`, `outputs/`, `runs/`, `working/` to `.gitignore`

  **Must NOT do**:
  - Do NOT include large files (weights, datasets) in git

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding the existing codebase structure and refactoring train.py while integrating Hydra + MLflow

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 7)
  - **Parallel Group**: Wave 1 (with Tasks 4, 5, 7)
  - **Blocks**: Tasks 8, 9, 11
  - **Blocked By**: Tasks 4, 5

  **Acceptance Criteria**:
  - [ ] Directory structure matches specification
  - [ ] `python scripts/train.py --config-name=default epochs=1` runs without error
  - [ ] `python scripts/train_cv.py --config-name=default model=yolo26n n_folds=3 epochs=1` runs without error
  - [ ] MLflow run created with experiment tag `model_family=yolo26`
  - [ ] `.gitignore` updated with all generated directories

  **QA Scenarios**:

  ```
  Scenario: Verify 1-epoch training end-to-end
    Tool: Bash
    Preconditions: Tasks 4, 5 complete
    Steps:
      1. Run: python scripts/train.py --config-name=default model=yolo26n epochs=1
    Expected Result: Training completes, exits 0, MLflow run created
    Evidence: .omo/evidence/task-6-train-e2e.txt

  Scenario: Verify 3-fold CV wrapper
    Tool: Bash
    Preconditions: train_cv.py created
    Steps:
      1. Run: python scripts/train_cv.py --config-name=default model=yolo26n n_folds=3 epochs=1
    Expected Result: All 3 folds complete, metrics aggregated
    Evidence: .omo/evidence/task-6-cv-wrapper.txt
  ```

  **Evidence to Capture**:
  - [ ] Directory listing
  - [ ] Training script output
  - [ ] CV wrapper output

  **Commit**: YES (with Tasks 4, 5)
  - Message: `feat(scripts): refactor training with Hydra+MLflow, add CV wrapper`
  - Files: `scripts/*.py`, `scripts/__init__.py`, `.gitignore`

- [x] 7. **GPU Benchmark + Memory Profiling**

  **What to do**:
  - Run GPU benchmark to establish baseline:
    - PyTorch matmul benchmark (matrix size sweep)
    - YOLO26n inference benchmark (100 warmup, 100 timed, batch=1/8/16)
    - YOLO26m inference benchmark
  - Profile VRAM usage for each YOLO26 variant at 640px:
    - `yolo26n`: batch=16, 32, 64
    - `yolo26s`: batch=16, 32
    - `yolo26m`: batch=8, 16
    - `yolo26l`: batch=4, 8
    - `yolo26x`: batch=2, 4
  - Record results to `docs/gpu-benchmark.md`
  - Determine optimal batch sizes for each variant
  - Note WDDM overhead compared to expected training throughput
  - Create `scripts/gpu_benchmark.py` for repeatable measurement

  **Must NOT do**:
  - Do NOT run training during this task (inference/forward-pass only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Well-defined measurement tasks with clear parameters

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 6)
  - **Parallel Group**: Wave 1 (with Tasks 4, 5, 6)
  - **Blocks**: Task 12 (HPO uses benchmark results for batch config)
  - **Blocked By**: Task 3

  **Acceptance Criteria**:
  - [ ] Benchmark results saved to `docs/gpu-benchmark.md`
  - [ ] Optimal batch sizes documented for each YOLO26 variant
  - [ ] VRAM usage recorded for each config
  - [ ] `scripts/gpu_benchmark.py` committed

  **QA Scenarios**:

  ```
  Scenario: Verify GPU benchmark runs
    Tool: Bash
    Preconditions: GPU verified, models downloaded
    Steps:
      1. Run: python scripts/gpu_benchmark.py --quick
    Expected Result: Benchmark completes, prints throughput numbers
    Evidence: .omo/evidence/task-7-benchmark.txt

  Scenario: Verify VRAM profiling
    Tool: Bash
    Preconditions: GPU verified
    Steps:
      1. Run: python scripts/gpu_benchmark.py --vram-only
    Expected Result: VRAM usage for each model variant printed
    Evidence: .omo/evidence/task-7-vram.txt
  ```

  **Evidence to Capture**:
  - [ ] Benchmark output
  - [ ] `docs/gpu-benchmark.md` file
  - [ ] VRAM profiling results

  **Commit**: YES (with Tasks 4-6)
  - Message: `docs: add GPU benchmark results with optimal batch sizes`
  - Files: `scripts/gpu_benchmark.py`, `docs/gpu-benchmark.md`

- [x] 8. **Dataset Analysis Report**

  **What to do**:
  - Create `scripts/dataset_analysis.py`:
    - Load dataset from `dataset/` (train/val/test splits)
    - For each split: count images, annotations, bboxes per image
    - Bounding box statistics: width, height, area, aspect ratio distribution
    - Class distribution (single class "hole" but check for imbalances)
    - Image-level stats: resolution, file size, channels
    - Parent-scene analysis: images per parent, coverage patterns
  - Generate report saved to `docs/dataset-analysis.md`
    - Histograms (bbox size, bboxes per image)
    - Summary statistics table
    - Notable patterns (e.g., small holes vs large excavations)
  - Key metric: annotation quality indicators
    - Bbox edge padding (are annotations tight or loose?)
    - Size distribution (are there many tiny bboxes that models struggle with?)
    - Per-image annotation count variance

  **Must NOT do**:
  - Do NOT modify any dataset files
  - Do NOT run analysis on test set (sealed) — only do basic counting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires data analysis and statistical reporting skills

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 6)
  - **Blocks**: Task 9
  - **Blocked By**: Task 6

  **Acceptance Criteria**:
  - [ ] `docs/dataset-analysis.md` created with full statistics
  - [ ] All histograms and plots generated
  - [ ] Parent-scene distribution documented

  **QA Scenarios**:

  ```
  Scenario: Verify dataset analysis runs
    Tool: Bash
    Preconditions: Task 6 complete, dataset exists
    Steps:
      1. Run: python scripts/dataset_analysis.py --output docs/dataset-analysis.md
      2. Check: output file exists and has content
    Expected Result: Report generated with statistics
    Evidence: .omo/evidence/task-8-analysis.txt
  ```

  **Evidence to Capture**:
  - [ ] Report file
  - [ ] Statistics output

  **Commit**: YES
  - Message: `docs: add dataset analysis report with bbox statistics`
  - Files: `scripts/dataset_analysis.py`, `docs/dataset-analysis.md`

- [x] 9. **Create Albumentations Pipeline (Light + Heavy Configs)**

  **What to do**:
  - Create `scripts/augmentation.py` with Albumentations transforms:
  - **Light augmentation** (`configs/augmentation/light.yaml`):
    - HorizontalFlip (p=0.5)
    - RandomBrightnessContrast (p=0.3)
    - HueSaturationValue (p=0.2)
    - Blur (p=0.2)
  - **Heavy augmentation** (`configs/augmentation/heavy.yaml`):
    - All light transforms at higher probabilities
    - RandomRotate90 (p=0.5)
    - ShiftScaleRotate (p=0.5)
    - RandomGamma (p=0.3)
    - CLAHE (p=0.2)
    - CoarseDropout (p=0.3)
    - ISONoise (p=0.2)
    - Cutout (p=0.2)
  - Integrate with Ultralytics training (two approaches, choose one during implementation):
    - **Option A (preferred)**: Subclass `ultralytics.data.dataset.YOLODataset` and override `get_transforms()` to chain Albumentations transforms after Ultralytics' built-in pipeline
    - **Option B (simpler)**: Apply Albumentations in a custom `Dataset` wrapper that pre-processes images before passing to Ultralytics dataloader
    - Note: Ultralytics' callback system fires AFTER augmentations are already applied, so a custom Callback approach WON'T work for modifying training inputs
    - Configure via Hydra: `augmentation=light` or `augmentation=heavy`
  - Create `configs/augmentation/ultralytics_base.yaml` for Ultralytics built-in settings:
    - `hsv_h`, `hsv_s`, `hsv_v` (color jitter)
    - `degrees` (rotation)
    - `translate`, `scale`, `shear`
    - `perspective`, `flipud`, `fliplr`
    - `mosaic`, `mixup`, `copy_paste`
    - `close_mosaic` (epoch to disable mosaic)

  **Must NOT do**:
  - Do NOT apply augmentations to test set
  - Do NOT modify original dataset images

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires Albumentations API knowledge and Ultralytics callback integration

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 6, 8)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Tasks 6, 8

  **Acceptance Criteria**:
  - [ ] Augmentation pipeline loads with Albumentations
  - [ ] Light and heavy configs apply transforms without error
  - [ ] Augmented images + bboxes visually verified (Task 10)
  - [ ] Integration test: train 1 epoch with heavy augmentations

  **QA Scenarios**:

  ```
  Scenario: Verify augmentation pipeline loads
    Tool: Bash
    Preconditions: augmentation.py created
    Steps:
      1. Run: python scripts/augmentation.py --config configs/augmentation/heavy.yaml
    Expected Result: Pipeline loads without error
    Evidence: .omo/evidence/task-9-aug-load.txt

  Scenario: Verify augmentation applies to image+bboxes
    Tool: Bash
    Preconditions: augmentation.py, sample image
    Steps:
      1. Run: python -c "from scripts.augmentation import get_pipeline; p=get_pipeline('heavy'); print(p)"
    Expected Result: Pipeline description printed with all transforms
    Evidence: .omo/evidence/task-9-aug-pipeline.txt
  ```

  **Evidence to Capture**:
  - [ ] Augmentation config files
  - [ ] Pipeline load verification

  **Commit**: YES (with Task 8)
  - Message: `feat(augmentation): add Albumentations pipeline with light/heavy configs`
  - Files: `scripts/augmentation.py`, `configs/augmentation/*.yaml`

- [x] 10. **Augmentation Visualization + Validation**

  **What to do**:
  - Create `scripts/visualize_augmentations.py`:
    - Load 5 sample images from training set
    - Apply both light and heavy augmentation pipelines
    - Generate side-by-side comparison grid: original | light | heavy
    - Draw bounding boxes on all images
    - Save to `docs/augmentation-samples/`
  - Validate bbox integrity:
    - Check no bboxes are clipped out of image bounds
    - Check bbox aspect ratios are preserved correctly
    - Check no annotations are lost due to augmentation
  - Generate `docs/augmentation-samples/README.md` with examples

  **Must NOT do**:
  - Do NOT use test set images for visualization

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Requires creating side-by-side comparison visualizations with matplotlib/PIL

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 9)
  - **Blocks**: Tasks 12, 14-24
  - **Blocked By**: Task 9

  **Acceptance Criteria**:
  - [ ] `docs/augmentation-samples/` contains 5+ comparison grids
  - [ ] Bboxes visible and correctly positioned on augmented images
  - [ ] No bboxes clipped or lost (validation check passes)
  - [ ] README with augmentation descriptions

  **QA Scenarios**:

  ```
  Scenario: Verify augmentation visualization
    Tool: Bash
    Preconditions: Task 9 complete
    Steps:
      1. Run: python scripts/visualize_augmentations.py --n-samples 5 --output docs/augmentation-samples/
      2. Check: output directory has 5+ image files
    Expected Result: Grid comparison images created
    Evidence: .omo/evidence/task-10-viz.txt
  ```

  **Evidence to Capture**:
  - [ ] Sample comparison images
  - [ ] Bbox validation output

  **Commit**: YES (with Task 9)
  - Message: `docs: add augmentation visualization samples with bbox validation`
  - Files: `scripts/visualize_augmentations.py`, `docs/augmentation-samples/*`

- [x] 11. **Optuna HPO Integration**

  **What to do**:
  - Create `scripts/hpo.py` for Optuna hyperparameter optimization:
    - Optuna study configuration (TPE sampler, Median pruner)
    - Search space:
      - `lr` (learning rate): log-uniform [1e-4, 1e-2]
      - `momentum`: uniform [0.8, 0.95]
      - `weight_decay`: log-uniform [1e-5, 1e-3]
      - `optimizer`: categorical ['AdamW', 'SGD']
      - `warmup_epochs`: int [0, 5]
      - `cos_lr`: categorical [True, False]
      - `hsv_h`: uniform [0.0, 0.1]
      - `hsv_s`: uniform [0.0, 0.9]
      - `hsv_v`: uniform [0.0, 0.9]
      - `degrees`: uniform [0.0, 45.0]
      - `mosaic`: categorical [0.0, 0.5, 1.0]
      - `mixup`: categorical [0.0, 0.5, 1.0]
    - Run YOLO26m for reduced epochs (N epochs, TBD based on benchmark)
    - Log each trial to MLflow with parent run linking
    - Prune trials that underperform after 30% of epochs
    - Save best trial params to `configs/best_hparams.yaml`
  - Create `configs/hpo/default.yaml`:
    - `n_trials`: 50
    - `n_startup_trials`: 5 (random exploration)
    - `n_ei_candidates`: 24
    - `timeout_minutes`: 720 (12 hours max)
    - `prune_after_epoch_pct`: 0.3
    - `target_metric`: 'metrics/mAP50(B)'
    - `direction`: 'maximize'
  - Add `scripts/train.py` integration: when called from HPO, accept Optuna params as overrides to Hydra config

  **Must NOT do**:
  - Do NOT run HPO trials yet (deferred to Task 12)
  - Do NOT search batch size (determined by GPU memory benchmarks from Task 7)
  - Do NOT use test set for HPO (use val set from CV fold)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires Optuna API knowledge and careful search space design

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 4, 5, 9)
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 4, 5, 9

  **Acceptance Criteria**:
  - [ ] `python scripts/hpo.py --dry-run` prints search space without running
  - [ ] Optuna study configuration defined with all hyperparameters
  - [ ] MLflow integration: each trial creates a nested run
  - [ ] Best params export to YAML works in dry-run mode

  **QA Scenarios**:

  ```
  Scenario: Verify HPO script dry-run
    Tool: Bash
    Preconditions: Task 4, 5, 9 complete
    Steps:
      1. Run: python scripts/hpo.py --dry-run
    Expected Result: Prints search space, config, no training happens
    Evidence: .omo/evidence/task-11-hpo-dryrun.txt
  ```

  **Evidence to Capture**:
  - [ ] HPO config file
  - [ ] Dry-run output showing search space

  **Commit**: YES
  - Message: `feat(hpo): add Optuna hyperparameter optimization for YOLO26m`
  - Files: `scripts/hpo.py`, `configs/hpo/*.yaml`

- [x] 12. **Run 50 Optuna HPO Trials (GPU-Bound)**

  **What to do**:
  - Run `python scripts/hpo.py` with 50 trials on YOLO26m
  - Each trial: train YOLO26m for N epochs with suggested hyperparams
  - Use reduced epochs for speed (e.g., 50-75% of full training)
  - Monitor and log:
    - Each trial's validation mAP50
    - Trial duration
    - VRAM usage (detect OOM and skip candidates)
  - Prune unpromising trials early (after 30% epochs if below median of completed trials)
  - Handle GPU OOM: if YOLO26m OOMs at suggested batch size, halve batch and retry
  - Save best hyperparameters to `configs/best_hparams.yaml`
  - Log study to MLflow with:
    - Study-level: search space, best value, best params
    - Per-trial: params, metrics, duration, status (COMPLETE/PRUNED/FAIL)
  - Expected duration: ~4-8 hours GPU time (50 trials × ~5-10 min each)

  **Must NOT do**:
  - Do NOT adjust batch size during HPO (fixed per benchmark results)
  - Do NOT use test set for HPO validation

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Long-running GPU task requiring autonomous failure recovery and progress monitoring

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU-bound, sequential trials)
  - **Blocks**: Tasks 13, 14-18
  - **Blocked By**: Tasks 1, 7, 10, 11

  **Acceptance Criteria**:
  - [ ] 50 Optuna trials completed (or 12-hour timeout reached)
  - [ ] `configs/best_hparams.yaml` saved with best hyperparameters
  - [ ] MLflow parent run + 50 child trials logged
  - [ ] Study visualization (plot_optimization_history, plot_parallel_coordinate)
  - [ ] Best mAP50 ≥ 5% relative improvement over default params (or documented ceiling)

  **QA Scenarios**:

  ```
  Scenario: Verify HPO study started and progressing
    Tool: Bash
    Preconditions: Task 11 complete, GPU available
    Steps:
      1. Run: python scripts/hpo.py (with --n-trials 50)
      2. After 2 trials: check mlflow runs list
      3. Monitor: check progress via MLflow UI or CLI
    Expected Result: Trials logged to MLflow, progress visible
    Evidence: .omo/evidence/task-12-hpo-progress.txt

  Scenario: Verify best params exported
    Tool: Bash
    Preconditions: HPO completed
    Steps:
      1. Check: cat configs/best_hparams.yaml
    Expected Result: YAML file with best hyperparameters present
    Evidence: .omo/evidence/task-12-best-hparams.yaml
  ```

  **Evidence to Capture**:
  - [ ] HPO study output (best value + params)
  - [ ] `configs/best_hparams.yaml`
  - [ ] MLflow study summary

  **Commit**: YES
  - Message: `feat(hpo): complete 50-trial Optuna study for YOLO26m`
  - Files: `configs/best_hparams.yaml`, HPO results

- [x] 13. **Export Best Hyperparameters + Analysis**

  **What to do**:
  - Extract best trial from Optuna study
  - Save best params to `configs/best_hparams.yaml`
  - Generate HPO analysis report:
    - Parameter importance plot (which hyperparams matter most?)
    - Optimization history (learning curve across trials)
    - Parallel coordinate plot (parameter interactions)
    - Slice plot (single-parameter sensitivity)
  - Apply best params to `configs/model/yolo26m.yaml` as defaults
  - Also create derivative configs for other YOLO variants (n/s/l/x) with appropriate scaling:
    - Smaller models (n/s): may benefit from higher LR
    - Larger models (l/x): may need lower LR, more regularization
  - Save analysis to `docs/hpo-analysis.md`

  **Must NOT do**:
  - Do NOT change YOLO26l/x params without verification

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Well-defined analysis and export task

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 12)
  - **Blocks**: Task 16 (YOLO26m training uses best params)
  - **Blocked By**: Task 12

  **Acceptance Criteria**:
  - [ ] `configs/best_hparams.yaml` with best trial params
  - [ ] `docs/hpo-analysis.md` with parameter importance + plots
  - [ ] YOLO26 variant configs updated with scaled params

  **QA Scenarios**:

  ```
  Scenario: Verify best params config
    Tool: Bash
    Preconditions: Task 12 complete
    Steps:
      1. Run: python -c "import yaml; cfg=yaml.safe_load(open('configs/best_hparams.yaml')); print(cfg)"
    Expected Result: Valid YAML with hyperparameters
    Evidence: .omo/evidence/task-13-params.txt
  ```

  **Evidence to Capture**:
  - [ ] Best params YAML
  - [ ] HPO analysis report

  **Commit**: YES (with Task 12)
  - Message: `docs: add HPO analysis with parameter importance`
  - Files: `configs/best_hparams.yaml`, `docs/hpo-analysis.md`

- [x] 14. **YOLO26n 3-Fold CV Training** (nano, 3.2M params)

  **What to do**:
  - Train YOLO26n with 3-fold cross-validation:
    - `python scripts/train_cv.py model=yolo26n n_folds=3 epochs=100 imgsz=640`
  - Use Ultralytics built-in augmentations (Mosaic, MixUp, etc.) + Albumentations heavy from Task 9
  - Use default/reasonable hyperparameters for the first run (HPO from Task 12 is for YOLO26m, only apply if transferable)
  - Log each fold to MLflow with tags: `model_family=yolo26`, `model_scale=n`, `cv_fold=0/1/2`
  - Log per-fold metrics: mAP50, mAP50-95, Precision, Recall, F1
  - Save best checkpoint per fold to MLflow artifact
  - Compute cross-fold statistics: mean ± std for all metrics
  - Save aggregated results to `experiments/yolo26n/results.json`
  - Expected: best batch size from Task 7, ~1-2 hours total GPU time

  **Must NOT do**:
  - Do NOT run on test set
  - Do NOT change model architecture

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Long-running GPU training requiring monitoring and failure recovery

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU sequential — one fold at a time)
  - **Blocks**: Tasks 22, 23, 24 (ablations depend on YOLO26 results)
  - **Blocked By**: Tasks 1, 3, 10, 12

  **Acceptance Criteria**:
  - [ ] 3 folds complete → 3 MLflow runs
  - [ ] Per-fold metrics logged in MLflow
  - [ ] Cross-fold mean ± std computed
  - [ ] `experiments/yolo26n/results.json` saved

  **QA Scenarios**:

  ```
  Scenario: Verify YOLO26n CV training
    Tool: Bash
    Preconditions: GPU, model weights, augmentations ready
    Steps:
      1. Run: python scripts/train_cv.py model=yolo26n n_folds=3 epochs=100
      2. Check: mlflow runs list --experiment-id X (3 runs exist)
      3. Check: experiments/yolo26n/results.json exists
    Expected Result: Training completes, 3 MLflow runs, results saved
    Evidence: .omo/evidence/task-14-yolo26n.txt
  ```

  **Evidence to Capture**:
  - [ ] Training logs (one per fold)
  - [ ] Cross-fold metrics
  - [ ] MLflow run IDs

  **Commit**: NO (will commit all YOLO26 results together)

- [x] 15. **YOLO26s 3-Fold CV Training** (small, 11.4M params)

  **What to do**: Same as Task 14 with `model_scale=s`
  - Training may be ~2x slower than YOLO26n
  - Use batch size from Task 7 benchmarks
  - Log to MLflow with appropriate tags

  **Acceptance Criteria**: Same as Task 14 but for YOLO26s
  **Evidence**: `.omo/evidence/task-15-yolo26s.txt`

  **Parallelization**: NO (GPU sequential)
  **Blocked By**: Tasks 1, 3, 10, 12
  **Blocks**: Tasks 22, 23, 24

- [x] 16. **YOLO26m 3-Fold CV Training** (medium, 21.7M params) — **WITH HPO BEST PARAMS**

  **What to do**: Same as Task 14 with `model_scale=m`, PLUS apply HPO best params from Task 13:
  - `python scripts/train_cv.py model=yolo26m n_folds=3 epochs=100 imgsz=640 hparams=configs/best_hparams.yaml`
  - This is the KEY experiment — best params found by Optuna
  - Expected to outperform the default-hparams baseline significantly
  - Log HPO params used for reproducibility

  **Acceptance Criteria**: Same as Task 14, plus HPO params logged in MLflow
  **Evidence**: `.omo/evidence/task-16-yolo26m.txt`

  **Parallelization**: NO (GPU sequential)
  **Blocked By**: Tasks 1, 3, 10, 12, 13
  **Blocks**: Tasks 22, 23, 24

- [x] 17. **YOLO26l 3-Fold CV Training** (large, 52.9M params)

  **What to do**: Same as Task 14 with `model_scale=l`
  - CRITICAL: 8GB VRAM may OOM at batch=16 — use batch size from Task 7 (likely batch=4-8)
  - If OOM occurs: implement automatic batch halving with gradient accumulation
  - Training expected 3-4x slower than YOLO26n

  **Acceptance Criteria**: Same as Task 14, plus OOM recovery documented
  **Evidence**: `.omo/evidence/task-17-yolo26l.txt`

  **Parallelization**: NO (GPU sequential)
  **Blocked By**: Tasks 1, 3, 10, 12

- [x] 18. **YOLO26x 3-Fold CV Training** (x-large, 99.1M params)

  **What to do**: Same as Task 17 with `model_scale=x`
  - HIGH OOM RISK: 8GB VRAM may force batch=2-4
  - If OOM at batch=2: try imgsz=512 reduction
  - If still OOM: document as "VRAM-limited, use gradient accumulation or smaller input"
  - Training expected 5-6x slower than YOLO26n

  **Acceptance Criteria**: Same as Task 17, plus any imgsz reduction documented
  **Evidence**: `.omo/evidence/task-18-yolo26x.txt`

  **Parallelization**: NO (GPU sequential)
  **Blocked By**: Tasks 1, 3, 10, 12

- [x] 19. **YOLOv8/v11 3-Fold CV Training**

  **What to do**:
  - Train YOLOv8m and YOLOv11m with 3-fold CV each:
    - `python scripts/train_cv.py model=yolov8m n_folds=3 epochs=100`
    - `python scripts/train_cv.py model=yolov11m n_folds=3 epochs=100`
  - Use same dataset, augmentations, image size (640px) as YOLO26 experiments
  - Use reasonable default hyperparameters per model
  - Log each to MLflow with tags: `model_family=yolov8` (or `yolov11`), `model_scale=m`
  - Expected: similar training time to YOLO26m (~1hr per fold)
  - Save cross-fold results to `experiments/yolov8m/` and `experiments/yolov11m/`
  - This provides YOLO architecture evolution comparison (v8 → v11 → v26)

  **Must NOT do**:
  - Do NOT use YOLO26-specific augmentation/HPO configs (use model-appropriate defaults)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Long-running multi-model training with same infrastructure as YOLO26

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU-bound)
  - **Blocked By**: Tasks 2, 3, 10, 13

  **Acceptance Criteria**:
  - [ ] 6 MLflow runs (v8m × 3 folds + v11m × 3 folds)
  - [ ] Cross-fold metrics for each model
  - [ ] Results comparable with YOLO26m counterpart

  **QA Scenarios**:

  ```
  Scenario: Verify YOLOv8m training
    Tool: Bash
    Preconditions: GPU, weights exist
    Steps:
      1. Run: python scripts/train_cv.py model=yolov8m n_folds=3 epochs=100
      2. Check mlflow runs
    Expected Result: 3 runs created with metrics
    Evidence: .omo/evidence/task-19-yolov8.txt

  Scenario: Verify YOLOv11m training
    Tool: Bash
    Preconditions: GPU, weights exist
    Steps:
      1. Run: python scripts/train_cv.py model=yolov11m n_folds=3 epochs=100
    Expected Result: 3 runs created with metrics
    Evidence: .omo/evidence/task-19-yolov11.txt
  ```

  **Evidence to Capture**:
  - [ ] Training logs
  - [ ] Cross-fold metrics

  **Commit**: NO (commit with Wave 4 results)

- [x] 20. **Faster R-CNN 3-Fold CV Training**

  **What to do**:
  - Use torchvision's built-in Faster R-CNN (no Detectron2 needed — torchvision is already a PyTorch dependency):
    - `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` — PRIMARY approach
    - Detectron2 is NOT available as pre-built wheel for Windows + cu128, so skip Detectron2 entirely
  - Create `scripts/train_faster_rcnn.py`:
    - Data loader that reads YOLO-format labels and converts to COCO format
    - Faster R-CNN training loop with same CV splits
    - Standard hyperparameters: LR=0.005, momentum=0.9, weight_decay=0.0005
    - Train for 100 epochs with StepLR decay
  - Create `configs/model/faster_rcnn.yaml` with model-specific config
  - Train 3-fold CV: `python scripts/train_faster_rcnn.py --config-name=faster_rcnn n_folds=3`
  - Log to MLflow with tag `model_family=faster_rcnn`
  - Expected: slower per-epoch than YOLO (Faster R-CNN is two-stage)

  **Must NOT do**:
  - Do NOT modify YOLO dataset format (read YOLO format, convert in loader)
  - Do NOT use custom backbone (ResNet50-FPN standard)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding Faster R-CNN API and custom training loop

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU-bound)
  - **Blocked By**: Tasks 2, 10

  **Acceptance Criteria**:
  - [ ] 3 MLflow runs (3 folds)
  - [ ] Cross-fold metrics (mAP50, mAP50-95, Precision, Recall)
  - [ ] Results comparable with YOLO26m

  **QA Scenarios**:

  ```
  Scenario: Verify Faster R-CNN training
    Tool: Bash
    Preconditions: Detectron2 installed
    Steps:
      1. Run: python scripts/train_faster_rcnn.py --config-name=faster_rcnn n_folds=3 epochs=100
    Expected Result: 3 folds complete, metrics logged
    Evidence: .omo/evidence/task-20-faster-rcnn.txt
  ```

  **Evidence to Capture**:
  - [ ] Training logs
  - [ ] Cross-fold metrics

  **Commit**: YES
  - Message: `feat(faster-rcnn): add Faster R-CNN training with 3-fold CV`
  - Files: `scripts/train_faster_rcnn.py`, `configs/model/faster_rcnn.yaml`

- [x] 21. **DETR 3-Fold CV Training**

  **What to do**:
  - Use `torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)`
  - Create `scripts/train_detr.py`:
    - Data loader converts YOLO format to DETR's COCO format expectations
    - DETR training loop: 100 epochs (consistent with all other models for fair comparison; the COCO-standard 300 epochs is for 118k images, disproportionate for 352-image dataset)
    - Hyperparameters: LR=1e-4, weight_decay=1e-4, lr_drop=200
    - Use hungarian matcher, set criterion (bce + giou + l1)
  - Create `configs/model/detr.yaml` with model-specific config
  - Train 3-fold CV: `python scripts/train_detr.py --config-name=detr n_folds=3`
  - Log to MLflow with tag `model_family=detr`
  - Expected: different training dynamics (no NMS needed, end-to-end)

  **Must NOT do**:
  - Do NOT hardcode number of queries (use DETR default 100)
  - Do NOT use non-standard DETR architecture modifications

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires DETR-specific knowledge (transformer-based detector, hungarian matching)

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU-bound)
  - **Blocked By**: Tasks 2, 10

  **Acceptance Criteria**:
  - [ ] 3 MLflow runs (3 folds)
  - [ ] Cross-fold metrics
  - [ ] Comparison results with other detectors

  **QA Scenarios**:

  ```
  Scenario: Verify DETR training
    Tool: Bash
    Preconditions: torch.hub can load DETR
    Steps:
      1. Run: python scripts/train_detr.py --config-name=detr n_folds=3 epochs=100
    Expected Result: 3 folds complete, metrics logged
    Evidence: .omo/evidence/task-21-detr.txt
  ```

  **Evidence to Capture**:
  - [ ] Training logs
  - [ ] Cross-fold metrics

  **Commit**: YES
  - Message: `feat(detr): add DETR training with 3-fold CV`
  - Files: `scripts/train_detr.py`, `configs/model/detr.yaml`

- [x] 22. **Image Size Ablation (320 vs 640 vs 1280)**

  **What to do**:
  - Train YOLO26m at 3 image sizes (1 fold only — 1-fold from CV is sufficient for ablation comparison):
    - 320px: `python scripts/train.py model=yolo26m imgsz=320 epochs=100`
    - 640px: `python scripts/train.py model=yolo26m imgsz=640 epochs=100`
    - 1280px: `python scripts/train.py model=yolo26m imgsz=1280 epochs=100`
  - Use same hyperparameters (from Task 13) except image size
  - Log to MLflow with tag `ablation=imgsz`, value 320/640/1280
  - Compare: does larger input improve mAP50-95? Does smaller hurt mAP50?
  - Trade-off analysis: speed vs accuracy
  - Expected: 1280px may OOM on 8GB VRAM at reasonable batch sizes — use batch=1-2

  **Must NOT do**:
  - Do NOT change other hyperparameters between runs
  - Do NOT use test set

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Structured ablation experiment requiring rigorous control

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU-bound)
  - **Blocked By**: Tasks 14-18 (YOLO26 results for comparison)
  - **Wait For**: All YOLO26 experiments complete

  **Acceptance Criteria**:
  - [ ] 3 MLflow runs (320, 640, 1280)
  - [ ] Metrics compared in table
  - [ ] Speed-accuracy trade-off documented

  **QA Scenarios**:

  ```
  Scenario: Verify image size ablation
    Tool: Bash
    Preconditions: YOLO26m trained
    Steps:
      1. Run 3 training runs at different imgsz
      2. Compare metrics
    Expected Result: Metrics table showing imgsz effect
    Evidence: .omo/evidence/task-22-imgsz-ablation.txt
  ```

  **Evidence to Capture**: [Training logs, metrics comparison]

  **Commit**: NO (commit with all ablation results)

- [x] 23. **Optimizer Ablation (AdamW vs SGD)**

  **What to do**:
  - Train YOLO26m with 2 optimizers (1 fold each):
    - AdamW: `python scripts/train.py model=yolo26m optimizer=AdamW epochs=100`
    - SGD: `python scripts/train.py model=yolo26m optimizer=SGD epochs=100`
  - Use default/HPO-tuned learning rates for each optimizer
  - Log to MLflow with tag `ablation=optimizer`
  - Compare convergence speed and final mAP

  **Acceptance Criteria**: Same pattern as Task 22

  **Evidence**: `.omo/evidence/task-23-optimizer-ablation.txt`

  **Parallelization**: NO (GPU-bound)
  **Blocked By**: Tasks 14-18

- [x] 24. **Augmentation Strategy Ablation (None vs Light vs Heavy)**

  **What to do**:
  - Train YOLO26m with 3 augmentation strategies (1 fold each):
    - None: minimal augmentations (only flip)
    - Light: from Task 9 `configs/augmentation/light.yaml`
    - Heavy: from Task 9 `configs/augmentation/heavy.yaml`
  - Compare overfitting behavior (train loss vs val loss gap)
  - Log to MLflow with tag `ablation=augmentation`
  - This is CRITICAL for the paper — shows augmentation necessity for small dataset

  **Acceptance Criteria**: Same pattern as Task 22, plus overfitting analysis

  **Evidence**: `.omo/evidence/task-24-augmentation-ablation.txt`

  **Parallelization**: NO (GPU-bound)
  **Blocked By**: Tasks 14-18

- [x] 25. **Calibration Curves + Confidence Analysis**

  **What to do**:
  - Create `scripts/eval_calibration.py`:
    - For each trained model, compute calibration curve:
      - Bin predictions by confidence (0-0.1, 0.1-0.2, ..., 0.9-1.0)
      - For each bin: compute accuracy (is IoU > threshold?)
      - Plot: confidence vs accuracy — perfectly calibrated = diagonal
    - Compute Expected Calibration Error (ECE)
    - Compute Maximum Calibration Error (MCE)
    - Generate per-model calibration plots saved to `docs/calibration/`
  - Log calibration metrics to MLflow for each model
  - Compare calibration across model scales (n vs m vs l vs x)
  - Key insight: are models overconfident? (typical for small datasets)
  - Error bars from 3-fold CV

  **Must NOT do**:
  - Do NOT use test set (use val folds)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires calibration curve computation and plotting

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 26, 27)
  - **Parallel Group**: Wave 7 (with Tasks 26, 27, 28)
  - **Blocked By**: Tasks 14-24 (all training results)

  **Acceptance Criteria**:
  - [ ] Calibration curves for all models
  - [ ] ECE / MCE metrics logged to MLflow
  - [ ] Calibration plots saved

  **QA Scenarios**:

  ```
  Scenario: Verify calibration analysis
    Tool: Bash
    Preconditions: Training results exist
    Steps:
      1. Run: python scripts/eval_calibration.py --models yolo26n,yolo26m --output docs/calibration/
    Expected Result: Calibration plots and metrics generated
    Evidence: .omo/evidence/task-25-calibration.txt
  ```

  **Evidence to Capture**: [Calibration plots, ECE/MCE values]

  **Commit**: YES (with Task 26)
  - Message: `feat(eval): add calibration analysis and confidence metrics`
  - Files: `scripts/eval_calibration.py`, `docs/calibration/*`

- [x] 26. **PR Curves + F1-Confidence Curves**

  **What to do**:
  - Create `scripts/eval_pr_curves.py`:
    - For each model, compute Precision-Recall curve across IoU thresholds (0.5:0.05:0.95)
    - Compute F1-confidence curve (F1 score as function of confidence threshold)
    - Find optimal confidence threshold for each model (max F1)
    - Compute mAP50, mAP50-95, mAP75
    - Generate PR plots overlaid (all models on same plot for comparison)
    - Generate F1-confidence plot (all models on same plot)
  - Save to `docs/pr-curves/`
  - Log optimal thresholds to MLflow
  - Key insight: which model has best precision-recall trade-off?

  **Acceptance Criteria**: Same pattern as Task 25

  **Evidence**: `.omo/evidence/task-26-pr-curves.txt`

  **Parallelization**: YES (with Tasks 25, 27, 28)

  **Commit**: YES (with Task 25)

- [x] 27. **Error Analysis on Best Model**

  **What to do**:
  - Create `scripts/eval_error_analysis.py`:
    - For the best-performing model (overall mAP50 winner):
    - Run inference on validation set (all 3 folds)
    - Categorize 100-200 representative failure cases:
      - False Positives by type:
        - Background confusion (detecting holes where none exist)
        - Object confusion (detecting non-hole objects)
        - Localization error (IoU between 0.1 and 0.5)
      - False Negatives by type:
        - Small objects (bbox area < 1% of image)
        - Occluded/overlapping objects
        - Low contrast (hole similar to background)
      - Duplicate detections (multiple bboxes for same hole)
    - Generate error distribution pie chart / bar chart
    - Save 10-20 annotated failure examples to `docs/error-analysis/`
    - Document failure modes that might be addressed with:
      - Better data (more diverse examples)
      - Architecture changes
      - Post-processing (NMS tuning, confidence thresholding)

  **Must NOT do**:
  - Do NOT use test set for error analysis
  - Do NOT tune model based on error analysis (document only)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires structured error analysis with categorization

  **Acceptance Criteria**:
  - [ ] Error taxonomy defined and populated
  - [ ] 10-20 annotated failure images saved
  - [ ] Error distribution documented

  **Evidence**: `.omo/evidence/task-27-error-analysis.txt`

  **Parallelization**: YES (with Tasks 25, 26, 28)
  - **Blocked By**: Tasks 25, 26 (needs calibration+PR curves first)

- [x] 28. **Statistical Significance Testing**

  **What to do**:
  - Create `scripts/eval_statistical_significance.py`:
    - For each model's 3-fold CV results:
    - Compute 95% confidence intervals for mAP50, mAP50-95 using:
      - Bootstrapping (recommended: 1000 bootstrap samples)
      - Normal approximation (if distribution is Gaussian enough)
    - Perform pairwise significance tests between models:
      - YOLO26n vs YOLO26m: is the improvement statistically significant?
      - Best YOLO26 vs YOLOv8m: is YOLO26 truly better?
      - Best YOLO26 vs Faster R-CNN: cross-paradigm comparison
      - Use: paired bootstrap test or Wilcoxon signed-rank test
    - Generate significance matrix (heatmap): p-values for all model pairs
    - Report: which differences are significant at p<0.05?
    - Key insight: many metric differences may NOT be significant with only 3 folds

  **Must NOT do**:
  - Do NOT claim significance without p-value evidence
  - Do NOT use test set for significance testing

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires statistical knowledge (bootstrapping, hypothesis testing)

  **Acceptance Criteria**:
  - [ ] Confidence intervals for all models
  - [ ] Pairwise significance test matrix
  - [ ] Documented which differences are significant

  **Evidence**: `.omo/evidence/task-28-significance.txt`

  **Parallelization**: YES (with Tasks 25, 26, 27)

- [x] 29. **Final Test Set Evaluation** (sealed until now)

  **What to do**:
  - **IMPORTANT**: The 40-image test set has been SEALED until this task
  - Run final evaluation of ALL trained models on the held-out test set:
    - `python scripts/evaluate.py --test-set --models yolo26n,yolo26s,yolo26m,yolo26l,yolo26x,yolov8m,yolov11m,faster_rcnn,detr`
  - For each model: compute mAP50, mAP50-95, Precision, Recall, F1
  - Generate final comparison table: model | mAP50 | mAP50-95 | Precision | Recall | F1
  - Highlight best overall model
  - Save to `docs/test-set-results.md`
  - Generate per-model test inference visualizations (5-10 images each)
  - Save annotated test images to `docs/test-set-predictions/`
  - Log test metrics to MLflow as separate experiment type

  **Must NOT do**:
  - Do NOT go back and retrain based on test results
  - Do NOT modify any models after seeing test metrics
  - Do NOT share test metrics until paper is ready

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Final evaluation requires systematic comparison and documentation

  **Parallelization**: NO (depends on Tasks 25-28)
  - **Blocked By**: Tasks 25, 28

  **Acceptance Criteria**:
  - [ ] Final metrics table saved to `docs/test-set-results.md`
  - [ ] All models evaluated on identical test set
  - [ ] Best model identified and documented
  - [ ] Test prediction visualizations saved

  **QA Scenarios**:

  ```
  Scenario: Verify test set evaluation
    Tool: Bash
    Preconditions: All training complete, test set untouched
    Steps:
      1. Run: python scripts/evaluate.py --test-set
      2. Check: docs/test-set-results.md exists
    Expected Result: Final metrics table generated
    Evidence: .omo/evidence/task-29-test-eval.txt
  ```

  **Evidence to Capture**:
  - [ ] Final metrics table
  - [ ] Test set predictions (annotated images)
  - [ ] Per-model metrics

  **Commit**: YES
  - Message: `feat(eval): final test set evaluation for all models`
  - Files: `docs/test-set-results.md`, `docs/test-set-predictions/*`

- [x] 30. **Create Metrics Tables (Model × Metric × Mean±Std)**

  **What to do**:
  - Create `scripts/paper_metrics.py`:
    - Read all MLflow runs from completed experiments
    - Generate primary results table:
      | Model | Params | mAP50 | mAP50-95 | Precision | Recall | F1 |
      |-------|--------|-------|----------|-----------|--------|-----|
      | YOLO26n | 3.2M | 0.XXX±0.XXX | 0.XXX±0.XXX | ... | ... | ... |
      | YOLO26s | 11.4M | ... | ... | ... | ... | ... |
      | YOLO26m | 21.7M | ... | ... | ... | ... | ... |
      | YOLO26l | 52.9M | ... | ... | ... | ... | ... |
      | YOLO26x | 99.1M | ... | ... | ... | ... | ... |
      | YOLOv8m | 25.9M | ... | ... | ... | ... | ... |
      | YOLOv11m | 20.0M | ... | ... | ... | ... | ... |
      | Faster R-CNN | 41.5M | ... | ... | ... | ... | ... |
      | DETR | 41.3M | ... | ... | ... | ... | ... |
    - Bold the best value per column
    - Include: mean ± std across 3 folds
    - Include: parameter count, FLOPs estimate
  - Generate ablation result table (separate):
    | Ablation | Variant | mAP50 |
    |----------|---------|-------|
    | Image Size | 320 | ... |
    | Image Size | 640 | ... |
    | Image Size | 1280 | ... |
    | Optimizer | AdamW | ... |
    | Optimizer | SGD | ... |
    | Augmentation | None | ... |
    | Augmentation | Light | ... |
    | Augmentation | Heavy | ... |
  - Save to `docs/paper-metrics.md`

  **Must NOT do**:
  - Do NOT cherry-pick best fold (report all folds)
  - Do NOT fabricate FLOPs if can't compute (mark as N/A)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Requires structured data compilation and tables

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 31, 32)
  - **Parallel Group**: Wave 8 (with Tasks 31, 32, 33)
  - **Blocked By**: Task 29

  **Acceptance Criteria**:
  - [ ] Primary metrics table with all 9 models
  - [ ] Ablation study table
  - [ ] Mean±std format across folds

  **QA Scenarios**:

  ```
  Scenario: Verify metrics table generation
    Tool: Bash
    Preconditions: Task 29 complete
    Steps:
      1. Run: python scripts/paper_metrics.py --output docs/paper-metrics.md
    Expected Result: Tables generated with all model metrics
    Evidence: .omo/evidence/task-30-metrics-table.txt
  ```

  **Evidence to Capture**: [Metrics table file]

  **Commit**: YES (with Tasks 31, 32)
  - Message: `docs: add paper-ready metrics tables with all models`
  - Files: `scripts/paper_metrics.py`, `docs/paper-metrics.md`

- [x] 31. **Generate Publication-Quality Figures**

  **What to do**:
  - Create `scripts/paper_figures.py`:
    - Generate 4-5 publication-quality figures:
    - **Figure 1**: Dataset samples with annotation examples (4-6 panel mosaic)
    - **Figure 2**: Model comparison PR curves (all models overlaid, test set)
    - **Figure 3**: Calibration curves comparison (best 3 models)
    - **Figure 4**: Example detections on test set (best model, 4-6 panels: easy vs hard cases)
    - **Figure 5** (optional): Ablation study bar chart
  - Use matplotlib with publication styling:
    - Font: Times New Roman or similar serif
    - DPI: 300
    - Figure size: column-width (3.5in) or page-width (7in)
    - Consistent color scheme (colorblind-friendly)
    - Legend outside plot area
  - Save to `docs/paper-figures/`

  **Must NOT do**:
  - Do NOT use matplotlib defaults (ugly, not publication-quality)
  - Do NOT include test set "unknown unknowns"

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Creating publication-quality figures with proper styling

  **Acceptance Criteria**:
  - [ ] All 4-5 figures generated at 300 DPI
  - [ ] Publication-appropriate styling (fonts, colors, layout)
  - [ ] Figures saved to `docs/paper-figures/`

  **Evidence**: `.omo/evidence/task-31-figures.txt`

  **Parallelization**: YES (with Tasks 30, 32, 33)

- [x] 32. **Create Ablation Study Tables + Analysis**

  **What to do**:
  - Compile ablation results from Tasks 22-24 into structured tables
  - For each ablation dimension, report:
    - Metric comparison (mAP50, mAP50-95)
    - Training time difference
    - Statistical significance (from Task 28)
  - Generate ablation analysis report `docs/paper-ablation-analysis.md`:
    - **Image Size**: What size gives best accuracy? Is 1280 worth the compute?
    - **Optimizer**: Does AdamW outperform SGD for this task?
    - **Augmentation**: How much does heavy augmentation help vs none?
  - Key insight for paper: which choices are most impactful?

  **Acceptance Criteria**: Same pattern as Task 30
  **Evidence**: `.omo/evidence/task-32-ablation.txt`
  **Parallelization**: YES (with Tasks 30, 31, 33)

- [x] 33. **Paper Draft + Reproducibility Instructions**

  **What to do**:
  - Create paper draft in `docs/paper/`:
    - `abstract.md`: 150-250 word abstract summarizing:
      - Problem (archaeological hole detection)
      - Method (systematic YOLO26 evaluation + comparisons)
      - Key results (best model, mAP, key finding)
      - Contribution (first systematic evaluation for archaeology)
    - `introduction.md`: Background on:
      - Archaeological site monitoring (STI context)
      - UAV-based remote sensing for archaeology
      - Object detection for cultural heritage
      - Gap: no systematic YOLO26 evaluation for excavation detection
    - `methods.md`: Technical approach:
      - Dataset description (432 images, 108 parent scenes)
      - Augmentation strategy
      - Model architectures (YOLO26 variants)
      - Training protocol (3-fold CV, HPO, hyperparameters)
      - Evaluation metrics
    - `results.md`: Key findings:
      - Model comparison table
      - Ablation results
      - Calibration analysis
      - Error analysis summary
    - `discussion.md`: Interpretation:
      - Best model recommendations
      - Failure mode analysis
      - Limitations (dataset size, single site, WDDM overhead)
      - Future work
    - `conclusion.md`: Summary and takeaway
  - Create `requirements.txt` with pinned versions for reproducibility
  - Create `README.md` with:
    - Project overview
    - Setup instructions (including GPU fix)
    - How to reproduce experiments
    - How to generate figures and tables
  - All sections should be INTERNAL DOCUMENTS (not final LaTeX) — focus on content

  **Must NOT do**:
  - Do NOT generate LaTeX/BibTeX (unless user requests)
  - Do NOT invent results or claims not supported by data

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Structured academic writing with multiple sections

  **Acceptance Criteria**:
  - [ ] All 6 paper sections drafted
  - [ ] `requirements.txt` with pinned versions
  - [ ] `README.md` with reproduction instructions

  **Evidence**: `.omo/evidence/task-33-paper-draft.txt`

  **Parallelization**: YES (with Tasks 30, 31, 32)

  **Commit**: YES
  - Message: `docs: add paper draft and reproducibility instructions`
  - Files: `docs/paper/*.md`, `requirements.txt`, `README.md`

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command, check MLflow). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.omo/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run linting on all Python scripts. Check `scripts/` for: hardcoded paths, magic numbers, missing docstrings, unused imports, overly long functions. Check configs for: missing required fields, incorrect YAML syntax. Review paper draft for coherence.
  Output: `Lint [PASS/FAIL] | Configs [N valid/N total] | Scripts [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state (`pip install -r requirements.txt`). Verify GPU works. Run 1-epoch training end-to-end. Verify MLflow logging. Run evaluation on 1 model. Verify metrics table generation. Check ALL evidence files exist.
  Output: `GPU [PASS/FAIL] | Training [PASS/FAIL] | MLflow [PASS/FAIL] | Eval [PASS/FAIL] | Evidence [N/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual implementation/diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance: no test set access before Phase 5, no Docker, no production deployment. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 0 (Tasks 1-3)**: `env: pytorch cu128 + ultralytics + model weights for Blackwell GPU`
  - Files: `requirements.txt`
  - Pre-commit: GPU tensor verification

- **Wave 1 (Tasks 4-7)**: `feat(infra): Hydra configs + MLflow tracking + GPU benchmarks`
  - Files: `configs/**`, `scripts/*.py`, `docs/gpu-benchmark.md`, `.gitignore`

- **Wave 2 (Tasks 8-10)**: `feat(data): dataset analysis + augmentation pipeline`
  - Files: `scripts/dataset_analysis.py`, `scripts/augmentation.py`, `scripts/visualize_augmentations.py`, `configs/augmentation/*.yaml`, `docs/dataset-analysis.md`

- **Wave 3 (Tasks 11-13)**: `feat(hpo): Optuna 50-trial study + best hyperparameters`
  - Files: `scripts/hpo.py`, `configs/hpo/*.yaml`, `configs/best_hparams.yaml`, `docs/hpo-analysis.md`

- **Wave 4 (Tasks 14-18)**: `feat(train): YOLO26 n/s/m/l/x 3-fold CV training`
  - Files: `scripts/train_cv.py` (updated), `experiments/yolo26*/results.json`
  - Pre-commit: verify MLflow runs exist

- **Wave 5 (Tasks 19-21)**: `feat(train): comparison detector training (v8/v11/FRCNN/DETR)`
  - Files: `scripts/train_faster_rcnn.py`, `scripts/train_detr.py`, `configs/model/faster_rcnn.yaml`, `configs/model/detr.yaml`

- **Wave 6 (Tasks 22-24)**: `feat(ablation): image size / optimizer / augmentation studies`
  - Files: `experiments/ablation*/results.json`

- **Wave 7 (Tasks 25-29)**: `feat(eval): calibration + PR + error analysis + significance + test set`
  - Files: `scripts/eval_calibration.py`, `scripts/eval_pr_curves.py`, `scripts/eval_error_analysis.py`, `scripts/eval_statistical_significance.py`, `docs/calibration/*`, `docs/pr-curves/*`, `docs/error-analysis/*`, `docs/test-set-results.md`

- **Wave 8 (Tasks 30-33)**: `docs: paper draft + metrics tables + publication figures`
  - Files: `scripts/paper_metrics.py`, `scripts/paper_figures.py`, `docs/paper-metrics.md`, `docs/paper-figures/*`, `docs/paper/*.md`, `README.md`

---

## Success Criteria

### Verification Commands
```bash
# 1. GPU verification
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('sm_120:', 'sm_120' in torch.cuda.get_arch_list()); t=torch.rand(2,2).cuda(); print(t @ t)"
# Expected: CUDA: True, sm_120: True, 2×2 matrix

# 2. Training verification
python scripts/train.py --config-name=default model=yolo26n epochs=1
# Expected: exits 0, MLflow run created

# 3. CV training verification
python scripts/train_cv.py --config-name=default model=yolo26n n_folds=3 epochs=1
# Expected: 3 folds complete, aggregated metrics

# 4. MLflow verification
mlflow runs list --experiment-id $(mlflow experiments list | tail -1 | awk '{print $1}')
# Expected: runs exist with metrics

# 5. Evaluation verification
python scripts/evaluate.py --model yolo26n --test-set
# Expected: metrics printed

# 6. Paper metrics verification
python scripts/paper_metrics.py --output docs/paper-metrics.md
# Expected: metrics table generated

# 7. Full reproducibility check
pip install -r requirements.txt && python scripts/train.py --config-name=default model=yolo26n epochs=1 seed=42
# Expected: same seed -> same metrics
```

### Final Checklist
- [ ] GPU: CUDA + sm_120 support verified
- [ ] All 40+ training runs logged to MLflow
- [ ] Cross-fold metrics (mean±std) computed for all models
- [ ] Best model identified with test set metrics
- [ ] Calibration curves + ECE/MCE for all models
- [ ] Statistical significance matrix generated
- [ ] Error analysis report completed
- [ ] Paper-ready metrics table with all 9 models
- [ ] 4+ publication-quality figures generated
- [ ] Paper draft with all 6 sections
- [ ] All ablation studies completed and documented
- [ ] `requirements.txt` with pinned versions
- [ ] Reproducibility verified (same seed -> same metrics)
- [ ] All "Must Have" items present
- [ ] All "Must NOT Have" items absent
- [ ] `.omo/evidence/` contains all scenario evidence files
