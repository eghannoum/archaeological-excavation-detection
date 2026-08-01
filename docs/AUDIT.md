# Pre-Publication Audit â€” STI Unauthorized Archaeological Excavations

**Date:** 2026-08-01
**Scope:** Full repository audit for public GitHub release
**Audit basis:** Every tracked and untracked file was reviewed (26 Python scripts, 35 YAML configs, 16 markdown docs, git history). No assumptions made â€” all findings verified against actual file contents.

---

## 0. Executive Summary

The repository is a **research-grade ML pipeline** with genuinely strong bones: a Hydra-based training system, MLflow experiment tracking, 3-fold cross-validation, HPO, ablations, a benchmark paper, and clean experiment discipline. The dataset (432 tiles) and trained weights are already properly gitignored.

It is **not yet publishable** because of five blockers:

| # | Blocker | Severity |
|---|---------|----------|
| 1 | **No README, LICENSE, or any community file** (0 of 12 expected root files exist) | Critical |
| 2 | **Absolute user paths committed to git** (`.omo/boulder.json`, `scripts/run_hpo.ps1`) | Critical |
| 3 | **`requirements.txt` is broken for most users** (CUDA-only torch pin, 2 required packages commented out, ~10 used packages unpinned) | Critical |
| 4 | **Internal tool state tracked in git** (`.omc/`, `.omo/` â€” session IDs, delegation audit logs, plans) | High |
| 5 | **Untracked junk at root** (`hpo_output.log` leaks absolute paths, scratch scripts) | High |

Scores (0â€“100%): see Â§14.

---

## 1. Repository Organization

### 1.1 Current layout (git-tracked)

```
.omc/logs/delegation-audit.jsonl   â† internal tool audit, must not publish
.omo/{boulder.json,plans,run-continuation}  â† internal tool state, absolute paths
configs/       35 YAML files + __init__.py
docs/          paper, ablations, figures, tables (16.4 MB)
scripts/       24 Python files (2,787+ lines across training variants alone)
requirements.txt
.gitignore
```

### 1.2 Untracked on disk (gitignored)

```
data/        1.0 GB  raw source (108 images, COCO json, 432 tiles, data.zip 501 MB)
dataset/     438 MB  derived YOLO dataset (432 images + 435 labels + data.yaml)
models/      319 MB  7 trained weight files
experiments/ 955 MB  fold checkpoints + results per model
runs/        1.9 GB  ultralytics training outputs
mlruns/      11.4 GB MLflow tracking db + artifacts
outputs/     1.5 MB  Hydra timestamped outputs
.venv/       5.4 GB
```

### 1.3 Findings

| # | Finding | Impact | Fix | Effort |
|---|---------|--------|-----|--------|
| R1 | **No README, LICENSE, pyproject.toml, .gitattributes, CITATION.cff, CONTRIBUTING, CHANGELOG, CODE_OF_CONDUCT, SECURITY, Dockerfile** | Critical â€” repo is anonymous and unlicensed; nobody can legally use it | Create all (see Â§8, Â§9) | M |
| R2 | **`notebooks/` is empty** | Low â€” dead directory | Remove or seed with one EDA notebook | S |
| R3 | **Root junk**: `hpo_error.log`, `hpo_output.log` (144 KB, leaks `C:\Users\<username>\...`), `_check_runs.py`, `_query_mlflow.py`, `yolo26m.pt`, `yolo26n.pt` (duplicate weights of `models/`) | Medium â€” logs leak personal paths; scratch scripts would be committed by `git add .`; duplicate weights | Delete logs + scratch scripts; delete duplicate root `.pt` | S |
| R4 | **`.omc/` + `.omo/` tracked in git** | High â€” session IDs, internal plan paths, absolute user paths | `git rm --cached` both, add to `.gitignore`, delete on disk | S |
| R5 | **`data/` raw layout is unreferenced** by any script (ETL was notebook-based; `coco_to_yolo.py` docstring examples point at a layout that doesn't exist: `data/annotations.json`) | Low â€” misleading docs; dead raw data | Fix docstring; document the real ETL | S |
| R6 | **`configs/best_hparams.yaml` is a generated artifact** living among source configs | Low â€” stale documentation (values drift from code) | Move to `docs/` or delete; derive from MLflow instead | S |
| R7 | **Duplicate YOLO label parsers (3 copies)** and **duplicate bbox denormalizers (3 copies)** across `dataset_analysis.py`, `visualize_augmentations.py`, `visualize_conversion.py` | Medium â€” drift risk | Extract to a shared `scripts/yolo_utils.py` | M |
| R8 | **`paper/` scripts** (5 publication-specific scripts) | Medium â€” pollutes `scripts/` with one-off paper tooling | Keep in `paper/` (already split); document | S |

### 1.4 Proposed target structure

```
STI-Unauthorized-Archaeological-Excavations/
â”œâ”€â”€ configs/            Hydra config system (cleaned, dead groups removed)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ sample/         8â€“12 sample images + labels for quick-start (new)
â”‚   â””â”€â”€ README.md       dataset provenance + download instructions (new)
â”œâ”€â”€ dataset/            (gitignored, reproducible via data pipeline)
â”œâ”€â”€ docs/               paper + figures + analysis reports + AUDIT.md
â”œâ”€â”€ models/             (gitignored â€” download script + HF/Drive links)
â”œâ”€â”€ paper/              5 publication scripts (already split)
â”œâ”€â”€ scripts/            CLI entrypoints (train/inference/eval/visualization)
â”œâ”€â”€ tests/              pytest suite (new)
â”œâ”€â”€ .github/
â”‚   â”œâ”€â”€ workflows/      CI + release (new)
â”‚   â””â”€â”€ ISSUE_TEMPLATE/ + PULL_REQUEST_TEMPLATE.md (new)
â”œâ”€â”€ .gitignore          (extended)
â”œâ”€â”€ .gitattributes      (new)
â”œâ”€â”€ .pre-commit-config.yaml (new)
â”œâ”€â”€ requirements.txt    (fixed: portable, complete)
â”œâ”€â”€ pyproject.toml      (new â€” ruff/black/isort/mypy config + metadata)
â”œâ”€â”€ Dockerfile          (new)
â”œâ”€â”€ README.md           (new)
â”œâ”€â”€ LICENSE             (new â€” pending user choice)
â”œâ”€â”€ CONTRIBUTING.md     (new)
â”œâ”€â”€ CHANGELOG.md        (new)
â”œâ”€â”€ CODE_OF_CONDUCT.md  (new)
â”œâ”€â”€ SECURITY.md         (new)
â””â”€â”€ CITATION.cff        (new)
```

**Rationale for keeping `scripts/` instead of a `src/` package:** 24 working scripts already import each other as `from scripts.x import y`; 6 modules depend on this contract. Renaming to `src/` would churn ~10 files with zero functional gain and risk breaking the Hydra config path resolution. The structure the user proposed (`src/`) is a generic template; this repo's working convention is preserved and documented instead. `weights/`, `inference/`, `datasets/`, `outputs/` from the template are already covered by `models/`, `scripts/inference.py`, `dataset/`, `outputs/`.

---

## 2. Code Cleanup

### 2.1 Training scripts

| File | Lines | Key findings | Impact |
|------|------:|--------------|--------|
| `train.py` | 448 | Solid Hydra+MLflow design. **Dead code:** `train_non_yolo()` raises `NotImplementedError` referencing "Task 8" (internal) while `train_detr.py`/`train_faster_rcnn.py` fully implement it; `set_seeds()` misses `random` module + `torch.use_deterministic_algorithms`; **`device` hardcoded to `0` ignoring `training.device: auto`** (L93); `print()` in summary; docstring example `ablation=imgsz320 optimizer=sgd` is broken (no `optimizer` group) | High |
| `train_cv.py` | 604 | **Subprocess return code never checked** (L464) â€” failed folds silently contribute garbage metrics; dead constants `LABELS_TRAIN/VAL`; rebuilds MLflow URI instead of importing; ignores `cv.random_seed/save_fold_models/fold_metrics`; seed hardcoded 42 | High |
| `train_detr.py` | 1354 | 4 of 5 `mlflow_utils` imports dead; **aux-loss machinery is cosmetic** (`weight_dict` built for 5 layers but criterion never receives `aux_outputs`); **`image_size=640` hardcoded inside `postprocess_detr_output`** â€” mAP breaks if `--image-size â‰  640`; ~600 lines byte-identical to `train_faster_rcnn.py`; no DataLoader `generator` â†’ shuffle not reproducible; no `n_scenes < n_folds` guard; label parsing skips malformed lines silently | High |
| `train_faster_rcnn.py` | 829 | **Latent `UnboundLocalError`**: `best_model_path` referenced outside the loop where it's conditionally assigned (L512) â€” crashes if mAP50 never improves; same 4 dead imports; same duplication; **aspect-ratio distortion** (non-square resize); no `n_scenes < n_folds` guard | High |

### 2.2 Inference & evaluation

| File | Lines | Key findings | Impact |
|------|------:|--------------|--------|
| `inference.py` | 244 | Default `device="cpu"` with stale comment ("CUDA unavailable on Blackwell" â€” false, repo trains on CUDA); **no auto device selection**; `print()` everywhere; default model path stale (`runs/train/yolo26m-hole/...` doesn't exist â€” real outputs live in `experiments/` and `runs/detect/`); **no video/webcam/batch support**; `--visualize` flag is a no-op; inline `import glob` | High |
| `evaluate.py` | 213 | `print()` everywhere; hardcoded model path (same stale path); in-function imports (`matplotlib`, `seaborn`, `shutil`); `except Exception: pass` on stats fallback; metric key extraction duplicated vs `train.py:_extract_final_metrics` | Medium |
| `eval_*.py` + `generate_*.py` (9) | ~2500 | Paper-specific analysis (calibration, error analysis, PR curves, significance, tables, figures) â€” **reusable utilities** (4 kept in scripts/) vs **paper one-offs** (5 moved to paper/); `hpo_analysis.py` **hardcodes report text** ("50 completed, 0 pruned, 19 failed", best-params YAML) â€” the report can contradict its own data | Medium |

### 2.3 HPO / augmentation / tracking

| File | Lines | Key findings | Impact |
|------|------:|--------------|--------|
| `hpo.py` | 861 | **`training.batch_size=8` hardcoded** in 3 places despite `configs/hpo/default.yaml` comment "fixed at 16"; whole-trial stdout buffered in RAM; `proc.terminate()` with no timeout; `n_jobs>1` + MLflow thread-safety unverified; `float("-inf")` as Optuna value | Medium |
| `hpo_analysis.py` | 345 | **Hardcoded narrative** (see above); no error handling anywhere; `warnings.filterwarnings('ignore')` globally; relative sqlite URI (CWD-dependent); magic baseline `0.3197` | Medium |
| `augmentation.py` | 396 | **Two sources of truth drifted**: YAML `albumentations:` list uses 1.x API (breaks on pinned albumentations 2.0.8) while Python code is 2.x; **nothing reads the YAML list**; `patch_yolodataset`/`_auto_patch`/`get_ultralytics_augmentation` dead (train.py re-implements inline); **albumentations NOT installed in .venv** despite pin â€” feature silently no-ops at train time | High |
| `mlflow_utils.py` | 412 | Creates `mlruns/` at **import time** (side effect); dead: `log_metrics`, `log_artifact`, `log_model_checkpoint`, `get_run_metrics`; `finish_mlflow` calls `mlflow.end_run()` (thread-local) while docstring claims thread-safe client; `_build_taxonomy_tags` raises unhelpful `AttributeError` if `cfg.data` missing | Medium |

### 2.4 Data pipeline & utilities

| File | Lines | Key findings | Impact |
|------|------:|--------------|--------|
| `coco_to_yolo.py` | 1187 | Best-structured of the lot. No-op loop in `validate_parents` (L423); dead params `is_dry_run`, `image_id_to_filename`; `fn.replace(".png",".txt")` misses `.jpeg`; ~30 prints; PIL handle leak (L859); docstring examples reference nonexistent layout | Medium |
| `dataset_analysis.py` | 922 | **Hardcodes 1160Ã—740 image size** ("verified against actual data") â€” silently wrong if data changes; `plot_bbox_edge_padding` **doesn't measure edge padding** (misnamed); **zero error handling**; O(nÂ²) label matching; pre-written narrative with hardcoded data claims | Medium |
| `dashboard.py` | 621 | Gradio UI; **Windows-only `os.startfile`** (L475); unused imports `subprocess`/`time`; dead `unload_model()`; silent model-load failure; `quality=92` on PNG save (meaningless); hardcoded `runs/train/yolo26m-hole` paths | Medium |
| `gpu_benchmark.py` | 471 | **Hardcoded RTX 5070 VRAM constants** (8151 MiB); writes to `.omo/evidence/task-7-*.txt` (task-scoped artifacts); unused `torch.nn.functional` import; duplicated table logic | Low |
| `visualize_augmentations.py` | 491 | **BGR colors drawn on RGB arrays** â€” "heavy" boxes render blue not red; bbox comparison assumes index order; empty grid rows on missing samples | Medium |
| `visualize_conversion.py` | 314 | 3rd copy of YOLO parsing; PIL handle leak; mixed os/pathlib styles | Low |

### 2.5 Cross-cutting

- **~600 lines byte-identical** across `train_detr.py` / `train_faster_rcnn.py` (scene-split, mAP50, precision-recall, fold aggregation, MLflow boilerplate) â†’ extraction candidate.
- **3 copies** of YOLO label parsing / bbox denormalization â†’ shared `yolo_utils.py`.
- **`print()` vs `logging` inconsistency**: only `visualize_augmentations.py` uses logging; 23 others use `print()`.
- **No type-checking or lint config anywhere** (no pyproject.toml, no ruff/black/isort config).

---

## 3. Inference Pipeline Review

Current state (`scripts/inference.py`): image-only, single-image loop, CPU-default, `print()`-based logging.

| Requirement | Status | Gap |
|---|---|---|
| Images | âœ… | â€” |
| Videos | âŒ | Source not restricted, but no explicit support/docs; `collect_images()` only handles image extensions |
| Webcam | âŒ | Not documented; would work via ultralytics source "0" but untested |
| Folders | âœ… | Recursive glob |
| Batch inference | âŒ | Processes one image at a time; no `batch` param |
| GPU | âš ï¸ | Supported via `--device` but **defaults to CPU** |
| CPU | âœ… | Default |
| Auto device | âŒ | Hardcoded default `"cpu"` |
| Error handling | âš ï¸ | Model-not-found check good; per-image corrupt check good; no try/except around `predict` |
| Logging | âŒ | `print()` only, no levels |
| CLI args | âœ… | argparse |
| Clean output | âš ï¸ | CSV/images/labels subdirs good; no timestamps â†’ overwrites |

**Fixes (high priority):** auto device selection (`cuda:0` if available else `cpu`); logging module; batch inference via ultralytics batch support; video support (`*.mp4`, `*.avi`, `*.mov`, `*.mkv`); source type detection for webcam; fix stale default model path (point at `experiments/` or make required); remove dead `--visualize` no-op.

**Production suggestions:** ONNX/TensorRT export path; async video frame processing; detection JSON output (GeoJSON for satellite use-case); `--save-crop`; streaming output for video.

---

## 4. Training Pipeline Review

| Concern | Status | Finding |
|---|---|---|
| Reproducibility | âš ï¸ | `set_seeds` misses `random`/deterministic-algorithms; **DETR/Faster-RCNN DataLoaders have no `generator=` â†’ shuffle is NOT reproducible across runs**; seed hardcoded 42 in `train_cv.py` |
| Config handling | âœ… | Hydra composition is well-designed; but **~10 of 35 YAMLs are dead** (see Â§6) |
| Checkpoint saving | âœ… | Ultralytics handles `best.pt`/`last.pt`; `_find_best_model` has 4 fallbacks |
| Resume training | âŒ | `training.resume`/`checkpoint_path` configured but **never wired** |
| Logging | âœ… | MLflow per-epoch callback + `logger`; but final summary uses `print()` |
| Validation | âœ… | `val=True`; fold-specific data YAMLs |
| Metrics | âœ… | `_extract_final_metrics` handles multiple ultralytics access patterns |
| Mixed precision | âœ… | `amp` from `precision: fp16` |
| Multi-GPU | âŒ | `device` hardcoded `0` in `_build_ultralytics_kwargs` â€” no DDP; `device: auto` config ignored |
| Dataset loading | âš ï¸ | `data.dataset_path` â†’ `./data` is **wrong** (dataset lives in `./dataset`); fold fallback logic decent |
| Gradient clip | âŒ | Config `gradient_clip` explicitly not applicable (documented in code) â€” dead config |
| Determinism | âŒ | `training.deterministic` configured, never used |
| Label smoothing / dropout | âŒ | Configured, never wired to ultralytics |

**Weaknesses to fix:** wire `device: auto`; wire `resume`; wire `deterministic`; add DataLoader `generator` to custom trainers; remove dead config keys or wire them.

---

## 5. Dataset Handling

`dataset/data.yaml` is correct and minimal:

```yaml
train: images/train   # 288 images
val: images/val       # 84 images
test: images/test     # 60 images
nc: 1
names: ['hole']
```

Findings:
- **432 images / 435 labels** â€” 3 label-only orphans (no image) in `labels/` (or vice versa). No validation script exists.
- **No dataset validation script**: nothing checks corrupt images, out-of-range bboxes, label/image parity, duplicate stems, or class id range. `dataset_analysis.py` comes closest but has zero error handling and hardcoded image size.
- **`data.yaml` comment** ("run yolo train from project root") is fragile â€” relative paths break if launched from elsewhere.
- **Parent-scene splits** are deterministic (seed 42) and properly grouped â€” good.
- `data.dataset_path`/`paths.data_dir` point at `./data` while the real dataset is `./dataset` â€” **stale config**.
- Missing: **a `scripts/validate_dataset.py`** and a `data/README.md` documenting provenance + reproducibility commands.

---

## 6. Configuration Management

The Hydra system is the strongest part of the repo â€” but ~10 of 35 YAMLs are documentation-only:

**100% dead config groups** (zero consumers):
- `configs/evaluation/` â€” no script reads `cfg.evaluation.*`
- `configs/paths/` â€” no script reads `cfg.paths.*` (every script derives paths in code); `experiment_dir` is even unresolvable (`${experiment.name}` = `???`)

**Dead keys in live groups:**
- `training`: `gradient_clip`, `label_smoothing`, `dropout`, `device`, `deterministic`, `resume`, `checkpoint_path`, `scheduler.{warmup_lr, lr_min_ratio, step_size, gamma}`
- `data`: everything except `image_size`
- `cv`: everything except `n_folds`
- `hpo`: `enabled`, `timeout`, `pruner.min_resource/max_resource`
- `experiment`: `log_frequency`, `save_frequency`, `notes`, `tags`
- `augmentation/*.albumentations` (YAML lists) â€” never parsed, duplicated in Python

**Broken by design (`@package _global_` hazard):**
- `model/yolo26{n,s,l,x}.yaml` write at config **root** â€” when loaded they're overwritten by later group slots, and the documented flow (`experiment=yolo26n`) never loads them. **The hand-tuned HPU hyperparameters (lr 0.000705 etc.) exist only in dead files â€” real runs use `training/default.yaml` values.** `model/yolo26m.yaml` is missing entirely.

**Duplication (same hyperparams in 4+ places):** lr/momentum/wd/optimizer/batch_size defined in `training/default.yaml`, `model/yolo26*.yaml`, `experiment/yolo26*.yaml`, `best_hparams.yaml` â€” and they disagree.

**No env-var support** except `GRADIO_PORT` and `AUGMENTATION_MODE`. **Hardcoded values** remain in every script (batch 8 in hpo.py, seeds, model paths).

**Recommendation:** delete the dead groups/keys; collapse `model/yolo26*.yaml` into interpolated experiment configs; make HPO search space + augmentation transforms config-driven (single source of truth); add `.env`/env-var support for paths.

---

## 7. Dependency Audit (`requirements.txt`)

**Current (broken for most users):**
```
torch==2.11.0+cu128          â† CUDA-specific build; pip install fails on CPU/macOS
torchvision==0.26.0+cu128    â† same
torchaudio==2.11.0+cu128     â† NOT USED anywhere
ultralytics==8.4.72
albumentations==2.0.8        â† NOT INSTALLED in .venv (feature silently disabled)
# mlflow==3.14.0              â† COMMENTED OUT but train.py imports it â€” install fails
# optuna==4.9.0               â† COMMENTED OUT but hpo.py imports it â€” install fails
```

**Used-but-unpinned packages:** `numpy`, `matplotlib`, `pandas`, `seaborn`, `scipy`, `gradio`, `Pillow`, `scikit-learn`, `tqdm`, `pyyaml`, `omegaconf`, `hydra-core`, `requests`.

**Fixes:**
1. Uncomment and pin `mlflow`, `optuna`.
2. Remove `torchaudio` (unused).
3. Make torch install portable: split `requirements.txt` (core) + `requirements-gpu.txt` (CUDA build) â€” or document the index URL. Pinning `+cu128` in the shared file hard-fails on CPU installs.
4. Pin all used packages at compatible versions.
5. Add `requirements-dev.txt` (pytest, ruff, black, isort, mypy, pre-commit).
6. **Security/compat:** verify `albumentations==2.0.8` vs the repo's own YAML (1.x API) and the ultralytics integration (uses `pipeline.transforms` + `A.BboxParams` â€” verified compatible with installed ultralytics 8.4.70).

---

## 8. Documentation

**Currently:** 15 markdown docs in `docs/` (paper, ablations, HPO analysis, dataset analysis, GPU benchmark, reproducibility) â€” good technical content, but **no README, no installation guide, no quickstart**. The paper (`docs/paper/paper.md`) is a genuine benchmark paper with results tables.

**To create:**
- `README.md` â€” overview, features, install, dataset prep, train, evaluate, inference, export, structure, commands, metrics, license, citation
- `CONTRIBUTING.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- `CITATION.cff`
- `docs/DATASET.md` â€” provenance + reproduction commands
- `docs/DEPLOYMENT.md` â€” ONNX/TensorRT/export notes

**Gaps:** no screenshots (add placeholder), no performance table in README (exists in paper), no badges.

---

## 9. GitHub Readiness

**Missing entirely:** CI, issue templates, PR template, release workflow, Dependabot, pre-commit hooks, `.gitattributes`, `.pre-commit-config.yaml`, branch protection docs.

**To create:**
- `.github/workflows/ci.yml` â€” pytest + ruff + mypy on ubuntu/CPU (torch CPU wheel)
- `.github/workflows/release.yml` â€” tag â†’ GitHub Release with assets (sample data, configs)
- `.github/dependabot.yml` â€” weekly updates for pip + github-actions
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.pre-commit-config.yaml` â€” ruff, black, isort, trailing-whitespace, end-of-file
- `.gitattributes` â€” `*.pt binary`, `*.png binary`, `*.ipynb` diff, LF normalization

---

## 10. Testing

**Zero tests exist.** No `tests/`, no pytest config, no coverage.

**Required suite:**
| Test | Scope |
|------|-------|
| `test_yolo_utils.py` | label parsing, bbox denormalization (the 3 duplicated copies) |
| `test_dataset.py` | data.yaml validity, label/image parity, bbox range checks |
| `test_configs.py` | every config group resolves under Hydra; no `???` leaks; no dead groups |
| `test_inference.py` | `collect_images()` for file/dir/glob; CSV writer; device resolution |
| `test_coco_to_yolo.py` | conversion round-trip, split integrity |
| `test_augmentation.py` | pipeline bbox integrity (count preservation, range checks) |
| `test_cli.py` | `--help` exit codes for all entrypoints |

Target: **â‰¥80% coverage** on `scripts/` core modules; CI gates on pass.

---

## 11. Performance Review

| Area | Finding | Impact |
|------|---------|--------|
| Inference | Per-image loop â†’ **no batch reuse**; model reloaded per call in loop context? (no â€” loaded once âœ“); `result.plot()` only when `--save-img` âœ“ | Medium |
| Preprocessing | `Image.open(...).load()` per image before predict â€” double decode (PIL + ultralytics) | Low |
| Memory | `all_detections` list unbounded (fine for CSV); whole-trial stdout buffer in hpo.py | Low |
| GPU | Training `device: 0` fixed â€” no multi-GPU; benchmark shows 8 GB VRAM headroom explored | Medium |
| Tensor copies | BGRâ†’RGB conversions done correctly (`[..., ::-1]`); no redundant `.cpu()` calls found | â€” |
| Disk I/O | `mlruns/` at 11.4 GB â€” consider artifact pruning policy | Low |
| DETR/FasterRCNN | Non-square resize (aspect distortion) in faster_rcnn; per-image inference in eval | Medium |

**Recommended:** batch inference (ultralytics batches natively), `imgsz` auto-scaling, TensorRT export for deployment, async dataloader workers already at 4.

---

## 12. Security Review

| # | Finding | Severity | Location |
|---|---------|----------|----------|
| S1 | **Absolute user path committed**: `C:\Users\<username>\Desktop\self study courses\...` | **Critical** | `.omo/boulder.json` (4Ã—, tracked), `scripts/run_hpo.ps1` L5 (tracked) |
| S2 | **Absolute user paths in untracked-but-not-ignored files** | High | `hpo_output.log` (~30 lines), `docs/error-analysis/error_summary_report.json` L3 (`weights_path`) |
| S3 | Internal tool state tracked (session IDs, delegation audit) | High | `.omc/logs/delegation-audit.jsonl`, `.omo/run-continuation/*.json`, `.omo/boulder.json` |
| S4 | **No secrets found** (API keys, tokens, private keys) | âœ… Clean | â€” |
| S5 | **No remote tracking URIs** (local sqlite only) | âœ… Clean | â€” |
| S6 | **No large files at risk** â€” all >10 MB files gitignored (verified with `git check-ignore`) | âœ… Clean | â€” |
| S7 | **Git history contains S1 + S3** â€” even after removal, history leaks paths | Medium | full history |

**Fixes:** delete/unignore log + scratch files; make `run_hpo.ps1` path-relative (`$PSScriptRoot`); scrub `weights_path` from error summary; `git rm --cached` `.omo/` `.omc/`; document `git filter-repo` history scrub as an optional step before first push (or accept leak in history since paths are non-credential).

---

## 13. Release Checklist

- [ ] Repository organization: junk removed, structure documented
- [ ] README, LICENSE, CONTRIBUTING, CHANGELOG, CODE_OF_CONDUCT, SECURITY, CITATION.cff
- [ ] `requirements.txt` portable + complete; `pyproject.toml` with tool config
- [ ] CI passing (pytest + lint); Dependabot; pre-commit
- [ ] Security: no paths/secrets in tracked files; `.gitignore` extended
- [ ] Sample dataset included (`data/sample/`); full dataset download documented
- [ ] Weights excluded from git; download script + links
- [ ] Tests â‰¥80% on core modules
- [ ] Dockerfile builds
- [ ] License chosen and applied; citation metadata matches paper
- [ ] Git tag `v0.1.0`; GitHub Release with notes
- [ ] (Optional) history scrubbed with `git filter-repo`

---

## 14. Final Deliverables & Scores

### Prioritized action plan

Status legend: **[x]** done, **[~]** partial, **[ ]** open.

| Priority | Action | Effort | Status |
|----------|--------|--------|--------|
| **P0** | Security: remove absolute paths from tracked files, untrack `.omo/`+`.omc/`, delete log/scratch junk | S | **[x]** |
| **P0** | Fix `requirements.txt` (portable torch, uncomment mlflow/optuna, pin all) | S | **[x]** |
| **P0** | Create LICENSE + README (the two minimum-viable-public items) | M | **[x]** |
| **P1** | Fix `inference.py` (auto device, logging, batch, video, model path) | M | **[x]** |
| **P1** | Fix `train.py` (`device: auto`, remove dead `train_non_yolo`, seed `random`) | S | **[x]** |
| **P1** | Add CI + pre-commit + issue/PR templates + Dependabot | M | **[x]** |
| **P1** | Fix `requirements`-adjacent breakage: `augmentation.py` albumentations drift | S | **[x]** |
| **P2** | Extract shared `yolo_utils.py` (3 duplicate parsers); fix `dataset_analysis.py` hardcoded size | M | **[x]** |
| **P2** | Add tests (dataset, config, inference, utils) | M | **[x]** |
| **P2** | Fix `hpo_analysis.py` hardcoded report; `train_cv.py` return-code check | M | **[x]** |
| **P2** | `pyproject.toml` + run black/isort/ruff across scripts | M | **[x]** |
| **P3** | Wire `resume`/`deterministic`/dead config keys or delete them | M | **[x]** (dead keys pruned) |
| **P3** | Deduplicate DETR/FasterRCNN shared code (~600 lines) | L | **[ ]** |
| **P3** | Dockerfile + ONNX/TensorRT export docs + dashboard Linux fix | M | **[~]** (Dockerfile + DEPLOYMENT.md done; dashboard `os.startfile` remains) |
| **P4** | History scrub (`git filter-repo`), git tags, first release | S | **[ ]** |

### Scores (0â€“100%)

Scores below are updated post-execution (see git log `b35ba2a..f4d5dd4`).

| Dimension | Score | Rationale |
|-----------|------:|-----------|
| **Publication readiness** | **85%** | LICENSE, README, CONTRIBUTING, CHANGELOG, CODE_OF_CONDUCT, SECURITY, .gitattributes, sample data, DEPLOYMENT.md, Dockerfile all present; paths scrubbed; CITATION.cff deliberately omitted |
| **Maintainability** | **78%** | Shared `yolo_utils.py` kills 3Ã— duplication; dead Hydra groups pruned; ruff/black clean (0 errors); modular structure intact |
| **Reproducibility** | **80%** | `random` seeded, `cv.random_seed`/`shuffle` wired, augmentation YAML aligned with 2.x API, `data/README.md` provenance, Hydra `--info` verified for all 7 experiments |
| **Code quality** | **80%** | inference.py rewritten (auto-device, video/webcam/batch, logging); train.py/train_cv.py fixed; 36 tests green; lint clean |
| **Security** | **90%** | No secrets; all absolute paths scrubbed; `.omo`/`.omc` untracked + gitignored; dependabot active; mlflow local-only |
| **Production readiness** | **75%** | 36-test suite, CI + release workflows, Dockerfile, pyproject packaging, auto-device inference, deployment guide; remaining: DETR/FasterRCNN dedupe, dashboard Linux fix, first release |

---

## 15. Additional Context Needed

- ~~**License preference**~~ **[x]** MIT (user decision).
- ~~**Author/citation details** for `CITATION.cff`~~ **[x]** Deliberately omitted (user decision — no citation file).
- ~~**Whether the 11.4 GB `mlruns/` and 1.9 GB `runs/` should be pruned**~~ **[x]** All generated outputs stay local + gitignored (train-to-reproduce; no pruning needed for the repo).
- ~~**Weights hosting preference**~~ **[x]** Skip weights entirely (user decision) — README documents train-to-reproduce.
