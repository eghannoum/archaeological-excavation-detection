# YOLO26 Dataset Preparation — COCO to YOLO Conversion

## TL;DR

> **Quick Summary**: Convert the existing COCO-annotated archaeological hole detection dataset (432 tiled images, 21,950 annotations, 108 parent scenes) into YOLO26 format with parent-level train/val/test splitting, ready for YOLO26m training.
>
> **Deliverables**:
> - `coco_to_yolo.py` — Conversion script (COCO JSON → YOLO .txt labels)
> - `visualize_conversion.py` — Visual verification tool
> - YOLO directory structure: `dataset/` with `images/{train,val,test}/` and `labels/{train,val,test}/`
> - `dataset/data.yaml` — Ultralytics config file
> - Format validation + 5 sample debug overlay images
>
> **Estimated Effort**: Short (3-4 hours) — note: determinism test (run pipeline twice) adds ~1h to base 2-3h estimate
> **Parallel Execution**: YES — Wave 1 parallel (Tasks 2 + 3), then sequential
> **Critical Path**: Task 1 → Tasks 2-3 (parallel) → Task 4 → F1-F4 (parallel) → Task 5

---

## Context

### Original Request
Prepare YOLO26 for an archaeological excavation detection dataset (satellite/aerial imagery of "holes" — unauthorized dig sites).

### Interview Summary
**Key Discussions**:
- Dataset: 432 tile-split satellite images (1160×740 PNGs) with 21,950 COCO annotations, single class "hole"
- Images sourced from 108 parent scenes (4 quadrants per scene: tl, tr, bl, br)
- Split at parent level to prevent data leakage (tiles from same parent stay together)
- YOLO26m (medium) model targeted, 80/10/10 train/val/test split

**Research Findings**:
- YOLO26 is the latest Ultralytics model (Jan 2026) — recommended for all new projects
- Features NMS-free end-to-end inference, enhanced small-object detection (ProgLoss + STAL for small objects)
- YOLO format: `class_id x_center y_center width height` (normalized 0-1), one `.txt` per image

### Metis Review
**Identified Gaps** (addressed):
- **Image dimension assumption**: Script must read dimensions dynamically per-image (not hardcoded 1160×740) — edge tiles may differ
- **Parent-level splitting**: 108 parent scenes each have exactly 4 tiles (from default `_tl,_tr,_bl,_br` suffixes) — split at parent level to prevent leakage
- **Empty annotation handling**: Images with 0 annotations get an empty `.txt` file (standard YOLO practice)
- **Reproducibility**: Fixed random seed (42) for deterministic splits
- **BBox validation**: Validate COCO bbox format (expect `[x, y, w, h]` pixel coords) before conversion
- **Dry-run mode**: `--dry-run` flag to preview split without writing files
- **Atomic writes**: Write to `.tmp` then rename to prevent partial output on interrupt

---

## Work Objectives

### Core Objective
Convert the COCO JSON dataset into YOLO26-format directory structure, ready for training with `yolo train model=yolo26m.pt data=dataset/data.yaml`.

### Concrete Deliverables
- `scripts/coco_to_yolo.py` — Converter with argparse (coco-path, image-dir, output-dir, seed, dry-run)
- `scripts/visualize_conversion.py` — Overlay YOLO boxes on sample images for visual QA
- `dataset/images/train/` (~80% of images) + `dataset/images/val/` (~10%) + `dataset/images/test/` (~10%)
- `dataset/labels/train/` (1 .txt per image, same distribution) + `dataset/labels/val/` + `dataset/labels/test/`
- `dataset/data.yaml` — `names: ['hole']`, `nc: 1`, paths to train/val/test image dirs

### Definition of Done
- [ ] All 5 acceptance criteria pass (see Verification Strategy)
- [ ] `python scripts/coco_to_yolo.py --dry-run` prints split summary (parent-group counts by split)
- [ ] `python scripts/visualize_conversion.py` produces 5 debug PNGs with correct box overlays
- [ ] `python -c "import yaml; yaml.safe_load(open('dataset/data.yaml'))"` and Ultralytics parseability pass (see AC-5)

### Must Have
- Parent-level split: all 4 tiles from same parent scene go to same split
- Deterministic output: same seed (42) produces identical results every run
- Dynamic image dimensions: read actual size via PIL, never hardcode 1160×740
- Empty .txt files for images with 0 annotations
- Atomic writes: `.tmp` → rename to prevent partial output
- `--dry-run` flag for preview without writing
- Bounding box validation: clip to [0,1], filter zero-area boxes

### Must NOT Have (Guardrails)
- Do NOT modify the original COCO JSON or original images (read-only)
- Do NOT perform data augmentation (belongs in training pipeline)
- Do NOT add model training, evaluation, or deployment scripts
- Do NOT resize or preprocess images (YOLO handles this at train time)
- Do NOT convert segmentation polygons (bbox only)
- Do NOT hardcode image dimensions or paths

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (Python 3, pip available)
- **Automated tests**: NONE (one-time data conversion pipeline — not production code)
- **QA method**: Agent-Executed Verification (5 concrete acceptance criteria)

### Acceptance Criteria (Agent-Verified)

**AC-1: Directory structure and file counts (percentage-based)**
```bash
python -c "
import json
from pathlib import Path
import sys

# Load COCO JSON (safe for large files — uses json.load, not PowerShell PSObject)
coco = json.load(open(r'data\output_annotations_notebook\combined_coco_split.json'))
total_images = len(coco['images'])
print(f'COCO JSON reports {total_images} images')

# Count images per split (empty-safe — Path.glob returns empty list, not \$null)
def count(path, pattern):
    return len(list(Path(path).glob(pattern)))

images_train = count(r'dataset\images\train', '*.png')
images_val   = count(r'dataset\images\val', '*.png')
images_test  = count(r'dataset\images\test', '*.png')

# Verify split proportions: ~80/10/10 (±2% tolerance)
train_pct = round(images_train / total_images * 100, 1)
val_pct   = round(images_val / total_images * 100, 1)
test_pct  = round(images_test / total_images * 100, 1)
print(f'Split: {train_pct}% / {val_pct}% / {test_pct}%')

# Verify image count = label count for each split
match_train = images_train == len(list(Path(r'dataset\labels\train').glob('*.txt')))
match_val   = images_val == len(list(Path(r'dataset\labels\val').glob('*.txt')))
match_test  = images_test == len(list(Path(r'dataset\labels\test').glob('*.txt')))
assert match_train and match_val and match_test, 'Image-label mismatch in one or more splits!'

# Verify total matches source
all_images = images_train + images_val + images_test
assert all_images == total_images, f'Total mismatch: {all_images} != {total_images}'
print(f'Total: {all_images} / {total_images}')

# Parent-group verification: ALL parents' tiles in same split
import re
suffix_re = re.compile(r'(_tl|_tr|_bl|_br)\.png$')
parents = {}
for img in coco['images']:
    parent = suffix_re.sub('', Path(img['file_name']).name)
    # Determine which split contains this file
    fname = Path(img['file_name']).name
    for split in ['train', 'val', 'test']:
        if Path(f'dataset/images/{split}/{fname}').exists():
            parents.setdefault(parent, set()).add(split)
            break
fail = [(p, s) for p, s in parents.items() if len(s) != 1]
if fail:
    for p, s in fail:
        print(f'FAIL: {p} spans {s}')
    sys.exit(f'{len(fail)} parents have tiles in multiple splits')
print(f'PASS: All {len(parents)} parents have tiles in a single split')
"
```

**AC-2: Label format validation**
```bash
# Inspect 5 deterministic label files
Get-ChildItem dataset\labels\train\*.txt | Sort-Object Name | Select-Object -First 5 | ForEach-Object {
  Write-Host "--- $($_.Name) ---"
  Get-Content $_.FullName
}
# Each line format: "0 <0.0-1.0> <0.0-1.0> <0.0-1.0> <0.0-1.0>"
# Verify with validation script
python scripts\coco_to_yolo.py --validate-only --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook
# Expected: "Validation PASS: 432 images, 21950 annotations, 0 errors"
```

**AC-3: Deterministic output**
```bash
# Run conversion twice to different directories
python scripts\coco_to_yolo.py --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook --output-dir dataset_run1 --seed 42
python scripts\coco_to_yolo.py --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook --output-dir dataset_run2 --seed 42

# Compare file-by-file using sorted output + Get-FileHash (filesystem-order independent)
$hash1 = Get-ChildItem dataset_run1\labels -Recurse -File | Sort-Object Name | ForEach-Object { Get-FileHash $_.FullName }
$hash2 = Get-ChildItem dataset_run2\labels -Recurse -File | Sort-Object Name | ForEach-Object { Get-FileHash $_.FullName }
$diff = Compare-Object $hash1.Hash $hash2.Hash
Write-Host "Mismatched files: $($diff.Count)"  # must be 0

# Cleanup
Remove-Item -LiteralPath "dataset_run1" -Recurse -Force
Remove-Item -LiteralPath "dataset_run2" -Recurse -Force
```

**AC-4: Visual verification**
```bash
# Generate overlay images
python scripts\visualize_conversion.py --image-dir dataset\images\train --label-dir dataset\labels\train --output-dir dataset\debug --count 5
# Verify 5 output files exist
@(Get-ChildItem dataset\debug\*.png).Count  # expect 5
```

**AC-5: data.yaml validity and label-image correspondence**
```bash
# Primary check: Verify YAML is valid and parseable
python -c "
import yaml
with open('dataset/data.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['nc'] == 1, f'nc={cfg[\"nc\"]} != 1'
assert cfg['names'] == ['hole'], f'names={cfg[\"names\"]} != [\"hole\"]'
assert 'train' in cfg, 'Missing train path'
assert 'val' in cfg, 'Missing val path'
print('PASS: data.yaml is valid')
"

# Secondary check: Every label file has a matching image
python -c "
import os
for split in ['train', 'val', 'test']:
    label_dir = f'dataset/labels/{split}'
    image_dir = f'dataset/images/{split}'
    labels = os.listdir(label_dir)
    missing = [l for l in labels if not os.path.exists(f'{image_dir}/{os.path.splitext(l)[0]}.png')]
    assert len(missing) == 0, f'{split}: {len(missing)} labels missing images: {missing[:5]}'
    print(f'{split}: {len(labels)} labels, all matched to images')
print('PASS: All labels correspond to existing images')

# Tertiary check: data.yaml paths resolve correctly
from pathlib import Path
yaml_path = Path('dataset/data.yaml')
cfg = yaml.safe_load(yaml_path.read_text())
for key in ['train', 'val']:
    target = yaml_path.parent / cfg[key]
    assert target.exists(), f'{key} path {cfg[key]} resolves to {target} — not found'
print('PASS: data.yaml paths resolve correctly')
"
```

> **Note**: The full Ultralytics `model.val()` parseability check is in Final Verification (Task F1), after the dataset is built in Task 4. The DoD validation chain is: Task 1→2→3→4→5(F1-F4).

---

## Execution Strategy

> **Pre-execution**: Run `Remove-Item -Recurse -Force .omo/evidence/ -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Path .omo/evidence/ -Force` to clear stale evidence artifacts from prior runs.

### Parallel Pipeline

Tasks 2 and 3 are independent script files — they can be written in parallel. Everything else is sequential:

```
Wave 1 (Parallel):
├── Task 2: COCO→YOLO conversion script [unspecified-high]
├── Task 3: Visualization script [quick]

Wave 2 (Sequential):
├── Task 4: Execute pipeline (needs both scripts from Wave 1)
    └── Then: Final Verification Wave (F1-F4, parallel) handles all verification
```

**Dependency Matrix**:
- Task 1: - → 2, 3
- Task 2: 1 → 4
- Task 3: 1 → 4
- Task 4: 2, 3 → F1, F2, F3, F4 (verification wave)
- F1-F4: 4 → Done (parallel final checks — fulfills verification role)

---

## TODOs

- [x] 1. Install dependencies and prepare environment

  **What to do**:
  - Install dependencies: `python -m pip install Pillow pyyaml numpy` (use `python -m pip` for safer PATH resolution; if exit code ≠ 0, retry with `--user` flag: `python -m pip install --user Pillow pyyaml numpy`; if both fail, `throw "pip install failed — check network/proxy settings"`).
  - Create `scripts/` directory at project root
  - Create `.omo/evidence/` directory for QA evidence: `New-Item -ItemType Directory -Path ".omo/evidence" -Force`
  - Verify Python 3.8+ available: `python --version`
  - Pre-check that source data exists:
    ```powershell
    if (-not (Test-Path "data\output_annotations_notebook\combined_coco_split.json")) { throw "Missing COCO JSON" }
    # NOTE: All source images are .png (confirmed from source data). Verification commands use *.png throughout.
    # If source format ever changes, update the extension in all verification commands.
    $ext = "png"
    $imgCount = @(Get-ChildItem "data\output_splits_notebook\*.$ext").Count
    Write-Host "Source images found: $imgCount (expected 432)"
    if ($imgCount -ne 432) { Write-Warning "Expected 432 images, found $imgCount" }
    ```
  - Pre-check COCO JSON extension consistency (catch format changes early): `python -c "import json; d=json.load(open('data/output_annotations_notebook/combined_coco_split.json')); exts=set(f.rsplit('.',1)[-1] for f in [i['file_name'] for i in d['images']]); print(f'Extensions in COCO JSON: {exts}'); assert exts == {'png'}, f'Expected all .png but found {exts}'"`
  - Pre-check COCO JSON validity (catch corruption early): `python -c "import json; d=json.load(open('data/output_annotations_notebook/combined_coco_split.json')); assert 'images' in d and 'annotations' in d and 'categories' in d; print(f'COCO JSON OK: {len(d[\"images\"])} images, {len(d[\"annotations\"])} annotations')"`
  - **Pre-flight source file existence check** (fail fast before conversion): `python -c "import json; from pathlib import Path; d=json.load(open(r'data/output_annotations_notebook/combined_coco_split.json')); missing=[i['file_name'] for i in d['images'] if not Path(r'data/output_splits_notebook')/i['file_name']]; assert not missing, f'{len(missing)} images missing from source: {missing[:5]}...' if len(missing)>5 else f'{len(missing)} images missing: {missing}'; print(f'All {len(d[\"images\"])} source files exist on disk')"`
  - Pre-check available disk space (need at least 4GB to accommodate: scripts + lightweight deps ~200MB, main dataset ~1GB, and two temporary full-copy directories for determinism verification ~2GB peak): `$free = (Get-PSDrive (Split-Path -Qualifier $pwd)).Free; if ($free -lt 4GB) { throw 'Insufficient disk space (<4GB free — need ~3GB peak for dataset + determinism copies)' }`. The main conversion outputs ~1GB; determinism re-run creates `dataset_qa/` which temporarily doubles storage. After F3 cleanup, only ~1GB remains.
  - Pre-check Python availability: `python --version` (must be 3.8+)
  - Create `.gitignore` with entries: `dataset/`, `.omo/evidence/` (prevents tracking generated files during Task 4)

  **Must NOT do**:
  - Do not install unnecessary packages beyond Pillow, pyyaml, and numpy (ultralytics is NOT required for dataset conversion — only for optional F1 format validation)
  - Do not modify any existing files
  - ⚠️ **Test set size note**: With 108 parents and 0.1/0.1 split using `floor()`, test/val each get ~10 parents (~40 images). This is adequate for basic model evaluation. If you need higher statistical confidence, adjust `--val-split`/`--test-split` upward (at the cost of fewer training examples). The plan's ±2% tolerance accounts for this.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple package installation and directory creation
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (start of pipeline)
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `python --version` — check current Python runtime
  - `python -m pip install Pillow pyyaml numpy` — lightweight deps for script creation and testing, with `--user` fallback

  **Acceptance Criteria**:
  - [ ] `python --version` returns 3.8+
  - [ ] `pip list | Select-String Pillow` confirms Pillow installed
  - [ ] `pip list | Select-String PyYAML` confirms PyYAML installed
  - [ ] `scripts/` directory exists
  - [ ] `.gitignore` contains `dataset/` and `.omo/evidence/` entries

  **QA Scenarios**:
  ```
  Scenario: Verify Python and packages
    Tool: Bash (interactive_bash)
    Preconditions: None
    Steps:
      1. Run: python --version
      2. Run: pip list | Select-String Pillow
      3. Run: pip list | Select-String PyYAML
      4. Run: pip list | Select-String numpy
      5. Check: scripts\ directory exists (Test-Path scripts\)
    Expected Result: Python 3.8+, Pillow/PyYAML/numpy all present, scripts/ exists
    Evidence: .omo/evidence/task-1-env-setup.txt
  ```

  **Commit**: NO (infrastructure setup)
  
- [x] 2. Create COCO-to-YOLO conversion script

  **What to do**:
  Create `scripts/coco_to_yolo.py` with these features:
  - **Argparse arguments**:
    - `--coco-path` (required): Path to COCO JSON
    - `--image-dir` (required): Path to image directory
    - `--yes` (optional): Skip interactive confirmation for `--overwrite`. Enables non-interactive/pipeline execution. Requires `--overwrite` to have any effect.
    - `--output-dir` (conditionally required — required unless `--dry-run` or `--validate-only` is set): Output directory for YOLO dataset. If directory already exists, script prints error and exits unless `--overwrite` is specified. Not required for read-only modes that don't write files. If provided alongside `--dry-run` or `--validate-only`, accept and ignore with a warning.
    - `--overwrite`: If set and `--output-dir` exists, delete it first then recreate. ⚠️ **DESTRUCTIVE**: Uses `shutil.rmtree()` then `os.makedirs()`. Safety guards: (a) reject overwriting the current directory (`.`), parent directory (`..`), or any path that is an ancestor (upward traversal) of `--image-dir` or `--coco-path` parent — implemented by checking `abspath in resolve_path.parents` for each source dir (exact `Path.parents` membership, NOT substring containment). Also reject if `abspath` equals `Path('scripts').resolve()` (prevent destroying conversion script). (b) **path-length safety**: require `--output-dir` to have at least 1 path component (enforced naturally by argparse `required=True`); `.`/`..` and source-data-parent checks above are the real protections. (c) **non-interactive safe**: require `--yes` flag to skip interactive confirmation. When `--yes` is NOT set, print full path and require user confirmation via `input("Confirm deletion of {abspath} (y/N): ")` — exit on non-`y` response. With `--yes`, skip the prompt and proceed directly. The `--yes` flag enables pipeline/automated execution. (d) **startup stale-file cleanup**: at startup, glob and remove any `*.tmp` files in the output directory to clean up after previous crashes.
    - `--seed` (default: 42): Random seed for reproducibility
    - `--dry-run`: Preview split without writing files (mutually exclusive with `--validate-only`)
    - `--validate-only`: Validate COCO JSON + bbox containment using actual image dimensions (opens images to check bbox bounds). Does NOT convert or write files. Mutually exclusive with `--dry-run`.
    - `--val-split` (default: 0.1): Validation fraction (float 0-1, must satisfy `val + test < 1.0`)
    - `--test-split` (default: 0.1): Test fraction (float 0-1, must satisfy `val + test < 1.0`)
    - `--input-bbox-format` (default: `xywh`): COCO bbox format — `xywh` for `[x, y, w, h]` or `xyxy` for `[x1, y1, x2, y2]`. Note: `xyxy` is included for future-proofing; this dataset uses standard COCO `xywh`. The `xyxy` path is specified but only `xywh` is tested by the QA scenarios.
    - `--allow-incomplete-parents`: If set, continue even when parent groups have ≠4 tiles. Incomplete parents are still included — with their available tiles. CAUTION: only use when you've verified the source has legitimate partial-parent scenes (e.g., coverage-area edges).
    - `--include-test-in-yaml`: If set, include `test: images/test` in data.yaml (for final evaluation). Default: FALSE — test split is omitted to prevent silent data leakage during hyperparameter tuning (Ultralytics `model.val()` uses the `test` split if present).
    - `--include-incomplete-log`: When `--allow-incomplete-parents` is active AND incomplete parents were found, write `{output_dir}/.incomplete_parents.txt` listing each incomplete parent, tile count, and list of tiles. Default: True. Has no effect when `--allow-incomplete-parents` is not set (script exits before writing).
    - `--quadrant-suffixes` (default: `_tl,_tr,_bl,_br`): Comma-separated list of quadrant suffixes for parent extraction
  - **Parent-level splitting logic**:
    - **Parent key extraction + grouping (single pass)**: For every filename, strip the extension via `Path(filename).stem` (or `os.path.splitext()`), then match quadrant suffixes against the stem using `stem.endswith(suffix)`. Use the *longest matching* suffix from the very end. Work on basename only (not full path). Suffixes are matched from the very end — e.g., `stem 'scene01_tl'` matches `_tl`, producing parent `scene01`. Tried in longest-first order: `_top_left` before `_tl`. Group filenames by the resulting parent key. This is a single-pass operation — there is no separate "whitelist construction" step. The parent key is whatever remains after stripping, and validation checks how many suffixes exist for each parent key. **Single-pass means**: after stripping one suffix, do NOT re-check whether the result still ends with a suffix.
    - **Double-suffix handling**: A filename like `scene_tl_bl.png` → stem is `scene_tl_bl` (strip `.png`) → longest-first suffix matching finds `_bl` at the end → single-pass produces parent key `scene_tl`. The resulting parent key `scene_tl` still ends with suffix `_tl`, but we do NOT strip again. Whitelist validation then detects that `scene_tl` has only 1 tile variant — flagged as **ambiguous** (correct behavior: the dataset doesn't have consistent quadrant naming for this file).
- **Whitelist validation**: After stripping the suffix, validate the resulting parent key against the whitelist of known parent IDs (derived from the full set of filenames). A parent is valid iff all expected quadrant suffixes for that parent exist (i.e., all `len(args.quadrant_suffixes.split(','))` suffix variants, default 4 for `_tl,_tr,_bl,_br`).
- **Ambiguous parents**: If stripping produces a parent key that doesn't have all suffix variants (exact count from `len(args.quadrant_suffixes.split(','))`), this indicates the suffix-matched filename is NOT a true tile file. These are ambiguous — raise `ValueError` listing the problematic filename and available suffixes. During `--dry-run`, log as WARNING instead of raising (allows user to review). This is separate from "incomplete parents" (which have a known parent key but fewer/more than expected tiles).
    - **Specific anti-false-positive rule**: If a filename contains any suffix string as a non-final substring (e.g., `hole_at_tl_site.png` contains `_tl` in the middle), the suffix stripping must only operate on the *terminal* suffix position. Implement by stripping the extension via `Path(filename).stem` first, then checking `stem.endswith(suffix)` for each suffix and taking the longest match — this naturally prevents mid-filename matches (since `endswith()` only matches the terminal position).
    - **Tie-breaking**: When multiple suffixes have equal length and both could match the terminal position of the stem, choose the **first** in `--quadrant-suffixes` list order. In practice, with `endswith()` this edge case is theoretically possible only for suffixes that share a common terminal substring of identical length (e.g., `_tl` and `_xl` both match `scene_01_tl`'s terminal `tl` — though unlikely). The rule exists for determinism. The default suffixes (`_tl,_tr,_bl,_br`) are all length 3 but only one matches the stem's terminal position at a time, so tie-breaking is rarely exercised.
    - Configurable quadrant suffixes via `--quadrant-suffixes` (default: `_tl,_tr,_bl,_br`)
    - **Failure mode with diagnostic error message**: If a filename doesn't match any expected suffix, raise `ValueError` with the original filename, the stripped suffix (if any), and available suffixes in the message. If incomplete parents are detected (≠ expected tile count, default 4) or ambiguous, list each parent with its tile count and the original source filenames. Example format:
      `ValueError("Parent 'area52' has 2 tiles (area52_tl.png, area52_br.png) but expected 4. Missing: area52_tr.png, area52_bl.png")`
      This enables the user to diagnose naming convention issues (e.g., a compound suffix being treated as a parent key).
    - **Incomplete parents**: By default, exit with error listing every parent with ≠ expected tile count (default 4) or ambiguous parent keys. Only continue when `--allow-incomplete-parents` is explicitly passed. When this flag is active, incomplete parents are still included in the split with their available tiles (if < expected: only existing tiles are included; if > expected: all N tiles are included — no culling). Warnings are printed but the pipeline proceeds. This prevents silent data corruption by requiring explicit opt-in.
    - Shuffle parent groups with `random.Random(seed)`
    - Compute split: use `math.floor()` for non-train splits, assign all remainder to train. `n_val = int(math.floor(num_parents * val_split))`, `n_test = int(math.floor(num_parents * test_split))`, `n_train = num_parents - n_val - n_test`. This guarantees the total equals `num_parents` and avoids rounding overshoot (unlike `round()` which can make `n_val + n_test > num_parents`). Python's `round()` uses banker's rounding (`round(10.5)=10`), so `floor()` is safer and more predictable for fractional splits.
    - **Safety check**: After computing split counts, assert `n_train >= 1`. If combined val+test split leaves fewer than 1 parent for training (including negative n_train from rounding overshoot), raise a clear `ValueError` listing the actual split values and parent counts. Also assert `n_val >= 1` and `n_test >= 1` when the respective split fraction is > 0 (prevents silently creating empty validation/test sets). Validate at argparse time that `val_split + test_split < 1.0`.
    - Split: first `n_train` for train, next `n_val` for val, remainder for test
  - **COCO parsing**:
    - Load JSON, validate categories: map by index position (not hardcoded COCO ID). Assert `len(categories) == 1`, then YOLO class_id `0` = categories[0]. This handles any COCO category_id value.
    - Build `image_id → filename` lookup
    - Build `image_id → [annotations]` lookup
    - **Per-parent tile count check**: Group filenames by parent key. For each parent, count tiles. If any parent has ≠ expected tile count (default 4), print a WARNING with details (parent name, tile count, tile list). By default the script exits with error after all warnings are printed (see `--allow-incomplete-parents` to proceed).
    - **Cross-split uniqueness assertion**: After split, verify `sum(len(p) for p in parents_per_split) == len(unique_parents)` — no parent appears in 2+ splits.
    - Validate: all image files exist on disk
    - Validate: no duplicate image IDs
  - **YOLO conversion**:
    - For each image in train/val/test:
      - Read actual pixel dimensions with PIL: `Image.open(path).size` (use raw dimensions — YOLO loads raw pixels directly, so bboxes must be normalized against raw pixel space. Do NOT apply `exif_transpose()` as that would create a dimension mismatch.)
      - Convert COCO `[x, y, w, h]` pixel → YOLO `[cx, cy, w, h]` normalized (respecting `--input-bbox-format`)
      - Write `class_id` (0 for "hole") + 4 normalized floats with 6 decimal places
    - Handle empty annotation images: write empty `.txt`
    - Use `encoding='utf-8'` on all file `open()` calls (critical on Windows where default codepage may be cp1252). For reading COCO JSON, use `encoding='utf-8-sig'` to handle Windows UTF-8 BOM (`\\xef\\xbb\\xbf`). For reading YOLO label files (which may contain user-provided non-UTF-8 data), use `encoding='utf-8', errors='replace'`.
    - Atomic write approach: write to `.tmp` file in same directory, then `os.replace()` (Python 3.3+) — unlike `os.rename()`, `os.replace()` atomically replaces the target file on Windows without raising `FileExistsError`
  - **Bbox validation**:
    - Print starting banner: `"Validating {n_images} images, {n_annotations} annotations..."` with carriage-return progress: `print(f"Validating image {i+1}/{total} ({filename})...", end='\r')`
    - **Dimension cross-check**: For each image, compare PIL dimensions against COCO `image['width']`/`image['height']`. On mismatch: `print(f"WARNING: {filename}: COCO says {coco_w}x{coco_h}, PIL reads {pil_w}x{pil_h} — using PIL as source of truth")` — do NOT abort, PIL is authority.
    - **EXIF orientation check and handling**: For each image, read EXIF orientation tag (tag 0x0112). If orientation != 1 (normal): `print(f"WARNING: {filename} has EXIF orientation {v} — bbox coordinates may have been created against software-corrected pixels, creating a mismatch with raw pixel space")`. Additionally, recommend pre-processing source images with `jhead -autorot` before conversion. The conversion script offers `--apply-exif-orientation`: if set, physically rotates the image to orientation-normal (removing EXIF rotation) AND adjusts all associated bbox coordinates (x,y,w,h) to match the rotated pixel space. When this flag is OFF (default), the warning is printed but images/bboxes are used as-is — YOLO processes raw pixels directly, and using EXIF-corrected pixels without bbox transformation would produce misaligned training data.
    - During `--validate-only`: Check bbox plausibility. For `xywh` format, validate full containment: `x >= 0`, `y >= 0`, `x + w <= image_width`, `y + h <= image_height`, and `0 < w < image_width`, `0 < h < image_height`. Also compute `avg_w / avg_image_width` across all bboxes — if > 0.5, flag as possible `xyxy` mislabeling. For `xyxy` format, validate `x1 < x2` and `y1 < y2`.
    - Flag any bbox where w > image_width (likely `xyxy` format mislabeled as `xywh`).
    - Skip (log warning) annotations where w ≤ 0 or h ≤ 0
    - Clip normalized coords to [0, 1]
    - Respect `--input-bbox-format`: if `xyxy`, convert `[x1, y1, x2, y2]` → `[x, y, w, h]` first
  - **Image copying**:
    - Copy (not move) images from `image-dir` to `output-dir/images/{split}/`
    - Preserve original filename
  - **data.yaml generation**:
    - Write `dataset/data.yaml` with **relative paths** (Ultralytics resolves paths relative to the YAML's directory, which is `dataset/`). Add header comment: `# Paths relative to this file's directory — run yolo train from project root`
    - ⚠️ **Windows forward-slash requirement**: Paths in data.yaml MUST use forward slashes (`images/train`, NOT `images\\train`). On Windows, `os.path.join()` produces backslashes — use `pathlib.Path(...).as_posix()` or explicit `.replace('\\', '/')` to convert. Ultralytics YAML parser may not resolve Windows backslashes correctly.
    - Content: `train: images/train`, `val: images/val`. Optionally includes `test: images/test` only if `--include-test-in-yaml` is passed (for post-training evaluation). By default, test split is omitted to prevent silent data leakage during hyperparameter tuning — Ultralytics `model.val()` uses the `test` split if present., `nc: 1`, `names: ['hole']`
    - ⚠️ NOT `../images/train` — that would resolve outside `dataset/` to the wrong directory
  - **Progress reporting**:
    - Print per-split summary: parent count, image count, annotation count per class
    - Print overall totals
    - During `--dry-run`: also print annotation counts per split to reveal potential class imbalance
    ```python
    # Example dry-run output:
    # DRY RUN: 108 parents → ~80/10/10 split, ~432 images
    #   train: 344 images, ~17560 annotations (class 'hole')
    #   val:   44 images, ~2195 annotations  (class 'hole')
    #   test:  44 images, ~2195 annotations  (class 'hole')
    ```
  - **Stratification**: The split is random at the parent level (no stratified sampling), but the dry-run reports annotation counts per split so imbalance is visible before execution. This is acceptable for a single-class dataset where density variation is expected.
  - **Error handling**:
    - Fail fast if COCO JSON can't be parsed
    - Warn if images missing from disk
    - Warn if bbox extends beyond image boundaries (and clip)
  - **`--dry-run` behavior**:
    - Parse COCO, group parents, compute split allocation
    - Print: "DRY RUN — would write N images to train, M to val, K to test"
    - Do NOT create directories, copy files, or write labels

  **Must NOT do**:
  - Do NOT hardcode any paths or image dimensions
  - Do NOT modify/delete original files
  - Do NOT use `shutil.move()` — use `shutil.copy2()` for images
  - Do NOT include training code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Medium-complexity Python script with multiple features (CLI parsing, file I/O, image processing, JSON parsing)
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 3 — both scripts are independent)
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - COCO JSON: `data/output_annotations_notebook/combined_coco_split.json` — annotation structure to parse
  - Image files: `data/output_splits_notebook/*.png` — 432 PNG images (1160×740, some edge tiles may differ)
  - YOLO format spec: Ultralytics expects `class_id cx cy w h` normalized [0,1], one `.txt` per image
  - PIL `Image.open().size` — for dynamic dimension reading (never hardcode)
  - `dataset/data.yaml` — output config file

  **Acceptance Criteria**:
  - [ ] `python scripts/coco_to_yolo.py --help` shows all argparse options
  - [ ] `python scripts/coco_to_yolo.py --dry-run ...` prints split summary without writing files
  - [ ] Script handles missing file gracefully (exits with error message)
  - [ ] Script validates bbox format before bulk conversion

  **QA Scenarios**:
  ```
  Scenario: CLI help works
    Tool: Bash
    Preconditions: scripts/coco_to_yolo.py exists
    Steps:
      1. Run: python scripts/coco_to_yolo.py --help
    Expected Result: Shows all arguments (coco-path, image-dir, output-dir, seed, dry-run, validate-only, val-split, test-split)
    Evidence: .omo/evidence/task-2-help.txt

  Scenario: Dry-run prints correct summary
    Tool: Bash
    Preconditions: scripts/coco_to_yolo.py exists, COCO JSON exists
    Steps:
      1. Run: python scripts/coco_to_yolo.py --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook --output-dir dataset --dry-run
    Expected Result: DRY RUN message with 108 parents split into train/val/test (~80/10/10 split). No files written.
    Evidence: .omo/evidence/task-2-dryrun.txt

  Scenario: Validate-only checks COCO structure
    Tool: Bash
    Preconditions: Same as above
    Steps:
      1. Run: python scripts/coco_to_yolo.py --validate-only --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook
    Expected Result: "Validation PASS" or specific error about structure
    Evidence: .omo/evidence/task-2-validate.txt

  Scenario: Malformed COCO JSON fails gracefully
    Tool: Bash
    Preconditions: A deliberately broken COCO JSON is needed
    Steps:
      1. Run: echo "{truncated" > ${env:TEMP}\broken_coco.json
      2. Run: python scripts\coco_to_yolo.py --validate-only --coco-path ${env:TEMP}\broken_coco.json --image-dir data\output_splits_notebook
    Expected Result: Non-zero exit code, error message mentioning "JSON" or "parse" or "decode". Script does NOT crash with unhandled exception.
    Evidence: .omo/evidence/task-2-malformed.txt

  Scenario: Double-suffix filename handled safely
    Tool: Bash
    Preconditions: scripts/coco_to_yolo.py exists
    Steps:
      1. Create a synthetic image: python -c "from PIL import Image; Image.new('RGB',(100,100)).save('${env:TEMP}\double_tl_bl.png')"
      2. Create matching COCO JSON referencing double_tl_bl.png
      3. Run validate-only: python scripts\coco_to_yolo.py --validate-only --coco-path ... --image-dir ${env:TEMP}
    Expected Result: Script reports the double-suffixed filename as ambiguous parent (only 1 tile found for stripped parent key) — exits with error unless --allow-incomplete-parents is passed. This verifies the suffix stripping only matches the terminal suffix.
    Evidence: .omo/evidence/task-2-double-suffix.txt

  Scenario: Empty annotations JSON passes validation
    Tool: Bash
    Preconditions: A valid but annotation-free COCO JSON is created alongside a matching synthetic image
    Steps:
      1. Run: python -c "from PIL import Image; Image.new('RGB',(1160,740)).save('${env:TEMP}\empty_scene.png')"
      2. Run: python -c "import json; json.dump({'images':[{'id':1,'file_name':'empty_scene.png','width':1160,'height':740}],'annotations':[],'categories':[{'id':1,'name':'hole'}]}, open('${env:TEMP}\empty_coco.json','w'))"
      3. Run: python scripts\coco_to_yolo.py --validate-only --coco-path ${env:TEMP}\empty_coco.json --image-dir ${env:TEMP}
    Expected Result: Validation PASS — empty annotations are valid (images with no holes should produce empty .txt files)
    Evidence: .omo/evidence/task-2-empty-anns.txt
  ```

  **Commit**: NO (data pipeline script)

- [x] 3. Create visualization script

  **What to do**:
  Create `scripts/visualize_conversion.py` for visual QA of YOLO-format labels:
  - **Argparse arguments**:
    - `--image-dir` (required): Path to images (e.g., `dataset/images/train`)
    - `--label-dir` (required): Path to YOLO labels (e.g., `dataset/labels/train`)
    - `--output-dir` (default: `dataset/debug`): Output directory for debug images
    - `--count` (default: 5): Number of sample images to visualize
    - `--seed` (default: 42): Random seed for reproducible image selection — ensures debug images are identical across runs. Call `random.seed(args.seed)` at startup.
  - **Logic**:
    - List all images in image-dir; filter to those WITH matching YOLO label files first
    - If `len(valid_images) < args.count`, print warning but proceed with available count
    - Randomly sample `min(args.count, len(valid_images))` from the eligible pool — ensures deterministic output count
    - For each selected image:
      - Open with PIL
      - Read corresponding `.txt` label file
      - For each line: parse class_id + normalized coords, denormalize to pixel coords
      - Draw rectangle with `PIL.ImageDraw` (red outline, width 3)
      - If the image has more than 20 annotations, cap rendering to 20 and print `WARNING: image {name} has {N} annotations — showing first 20 only (full count in label file)` to keep debug images readable
      - Save as `{original_name}_debug.png` to output-dir. Before saving, `os.makedirs(args.output_dir, exist_ok=True)` to ensure the directory exists.
  - **Error handling**: Pre-filter to images with label files; warn if fewer valid images than `--count`. If `args.count` is specified but no valid images exist, exit with clear error message.
  - **Progress**: Print filename and annotation count per image

  **Must NOT do**:
  - Do NOT modify label files or images
  - Do NOT use OpenCV (keep dependencies minimal — PIL only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small self-contained script, < 80 lines, straightforward logic
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2 — both are independent scripts)
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 4 (execution pipeline needs both scripts)
  - **Blocked By**: Task 1

  **References**:
  - PIL `ImageDraw` module — rectangle drawing
  - YOLO label format: `0 cx cy w h` normalized — for denormalization to pixel coords
  - `scripts/coco_to_yolo.py` — to understand label output format

  **Acceptance Criteria**:
  - [ ] `python scripts/visualize_conversion.py --help` shows all options
  - [ ] Script runs without error on valid YOLO-format data
  - [ ] Output debug PNGs have visible boxes

  **QA Scenarios**:
  ```
  Scenario: Visualizer produces debug images
    Tool: Bash
    Preconditions: Script file exists (verified via --help). NOTE: Full execution depends on Task 4 output — this scenario validates the script is syntactically correct and parses arguments.
    Steps:
      1. Run: python scripts\visualize_conversion.py --help
      2. Run: python -c "import ast; ast.parse(open('scripts/visualize_conversion.py').read()); print('Syntax OK')"
      3. Run synthetic functional test: Create a tiny 100x100 image and a matching label file, run the visualizer, and confirm output differs from input.
          python -c "
          from PIL import Image, ImageDraw
          import tempfile, os
          tmp = tempfile.mkdtemp()
          img = Image.new('RGB', (100,100), 'white')
          img.save(os.path.join(tmp, 'test.png'))
          with open(os.path.join(tmp, 'test.txt'), 'w') as f: f.write('0 0.5 0.5 0.2 0.2')
          out = os.path.join(tmp, 'out')
          import subprocess; subprocess.run(['python', 'scripts/visualize_conversion.py', '--image-dir', tmp, '--label-dir', tmp, '--output-dir', out, '--count', '1'], check=True)
          debug = Image.open(os.path.join(out, 'test_debug.png'))
          print(f'DEBUG: synthesized image exists — dimension {debug.size}')
          import numpy as np; assert np.array(debug).any() != np.array(img).any(), 'No overlay changes detected — bbox drawing may be broken'
          print('PASS: visualizer produces non-trivial overlays')
          "
    Expected Result: --help shows all arguments. Script parses without syntax errors. Functional test passes with non-trivial pixel diff.
    Evidence: .omo/evidence/task-3-syntax-check.txt
  ```

  **Commit**: NO (data pipeline script)

- [x] 4. Execute conversion pipeline

  **What to do**:
   Run the full conversion pipeline:
   0. **Install dependencies (minimal)**: `python -m pip install Pillow pyyaml numpy` (with `--user` fallback; see Task 1 for failure handling) if not already installed. Ultralytics/torch is NOT needed for dataset conversion — only for optional model parseability check in F1. The F1 check is skipped if ultralytics is not installed (wrap in try/except ImportError).
   1. **Optional: Verify YOLO26m model availability** (only if planning to train with this dataset): `python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')"`. If this fails, note the model name and continue — dataset conversion does not depend on it.
   2. **Preview**: Run `python scripts\coco_to_yolo.py --dry-run ...` first to verify split proportions and check for any ambiguous/incomplete parent warnings. If warnings appear, concrete escalation path:
   - **Incomplete parent warnings** (≠ expected tile count): First, re-run dry-run WITH `--allow-incomplete-parents` appended to verify warnings clear and split assignment is correct. Then write the decision to `.omo/evidence/task-4-incomplete-parents-decision.txt` (note count of incomplete parents, tiles present, and why they're acceptable). When proceeding to full conversion (Step 3), ALSO add `--allow-incomplete-parents` to the command — otherwise the full run will hard-error on the same incomplete parents. These are structural — some edge scenes genuinely have <4 tiles.
   - **Ambiguous parent warnings** (can't determine parent key): this is a hard error during non-dry-run. In dry-run mode, the agent prints the warnings for user awareness but does NOT need a decision — script exits with code 0 unless a pipe failure occurs. The agent logs the warning output to `.omo/evidence/task-4-dryrun-warnings.txt` and continues.
   3. Run `coco_to_yolo.py` with the actual dataset (full conversion)
   4. Run `visualize_conversion.py` to generate debug overlays
   5. Capture output for evidence

  **Commands**:
  ```powershell
  # Step 1: Convert
  python scripts\coco_to_yolo.py `
    --coco-path data\output_annotations_notebook\combined_coco_split.json `
    --image-dir data\output_splits_notebook `
    --output-dir dataset `
    --seed 42 `
    --val-split 0.1 `
    --test-split 0.1 `
    --overwrite `
    --yes

  # Step 2: Visualize
  python scripts\visualize_conversion.py `
    --image-dir dataset\images\train `
    --label-dir dataset\labels\train `
    --output-dir dataset\debug `
    --count 5
  ```

  **Must NOT do**:
  - Do NOT delete original source files
  - Do NOT overwrite existing meaningful files without confirmation

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Execute existing scripts, monitor output, handle errors
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: F1-F4 (verification wave)
  - **Blocked By**: Tasks 2, 3

  **References**:
  - `scripts/coco_to_yolo.py` — the conversion script
  - `scripts/visualize_conversion.py` — the visualization script
  - `data/output_annotations_notebook/combined_coco_split.json` — source annotations
  - `data/output_splits_notebook/` — source images

  **Acceptance Criteria**:
  - [ ] Script exits with code 0
  - [ ] dataset/ directory created with all subdirectories
  - [ ] Image/label file counts match expected split proportions (~80/10/10)
  - [ ] 5 debug PNGs in dataset/debug/
  - [ ] dataset/data.yaml exists with valid content
  - [ ] Dry-run annotation stats printed

  **Failure recovery**: If Task 4 fails mid-execution (crash, disk full, etc.), first inspect the error: check stderr for distinguishing between code bugs (argparse errors, KeyError, TypeError → DO NOT retry — escalate with the error message) vs environmental issues (disk full, file not found, permission denied → delete partial `dataset/` and retry). **Circuit breaker**: If 2 consecutive attempts fail (regardless of cause), do NOT retry a third time — STOP and escalate to user with error details: exit code, stderr output, and disk space status. Only re-run if the root cause was identified and fixed (e.g., freed disk space, corrected file path). The atomic write approach protects individual files, but directory-level state must be cleaned up manually.

  **QA Scenarios**:
  ```
  Scenario: Full conversion pipeline
    Tool: Bash
    Preconditions: scripts/coco_to_yolo.py exists
    Steps:
      1. Run conversion command
      2. Capture stdout and stderr
      3. Check exit code ($LASTEXITCODE)
    Expected Result: Exit code 0. Output shows parent counts (~108 parents in ~80/10/10 split)
    Evidence: .omo/evidence/task-4-conversion.txt

  Scenario: Directory structure verified
    Tool: Bash
    Preconditions: Conversion completed
    Steps:
       1. Count: @(Get-ChildItem dataset\images\train\*.png).Count
       2. Count: @(Get-ChildItem dataset\labels\train\*.txt).Count
       3. Count: @(Get-ChildItem dataset\images\val\*.png).Count
       4. Count: @(Get-ChildItem dataset\labels\val\*.txt).Count
       5. Count: @(Get-ChildItem dataset\images\test\*.png).Count
       6. Count: @(Get-ChildItem dataset\labels\test\*.txt).Count
      7. Check: Test-Path dataset\data.yaml
     Expected Result: ~80/10/10 split (label-image correspondence in each). data.yaml exists.
    Evidence: .omo/evidence/task-4-structure.txt

  Scenario: Visualizer produces debug images
    Tool: Bash
    Preconditions: Conversion completed (Step 1), visualizer script exists
    Steps:
      1. Run: python scripts\visualize_conversion.py --image-dir dataset\images\train --label-dir dataset\labels\train --output-dir dataset\debug --count 5
       2. Check: @(Get-ChildItem dataset\debug\*.png).Count
    Expected Result: 5 debug PNG files created with box overlays visible
    Evidence: .omo/evidence/task-4-visualizer-output.txt

   Scenario: Dry-run preview before conversion
    Tool: Bash
    Preconditions: scripts/coco_to_yolo.py exists
    Steps:
      1. Run: python scripts\coco_to_yolo.py --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook --output-dir dataset --dry-run
      2. Check stdout contains "DRY RUN" and parent counts
      3. Confirm no "WARNING" lines about incomplete parents (unless expected)
    Expected Result: Dry-run completes with exit code 0, shows split preview, no unexpected warnings
    Evidence: .omo/evidence/task-4-dryrun.txt

  Scenario: Dry-run with --include-test-in-yaml shows test path in output
    Tool: Bash
    Preconditions: scripts/coco_to_yolo.py exists
    Steps:
      1. Run: python scripts\coco_to_yolo.py --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook --output-dir dataset --dry-run --include-test-in-yaml
      2. Check stdout mentions "test" split path (dry-run does NOT write files — only stdout shows the proposed YAML content)
    Expected Result: Dry-run includes test split reference; without --include-test-in-yaml, no test path appears
    Evidence: .omo/evidence/task-4-include-test-yaml.txt
  ```

  **Commit**: NO (generated data — not source code)

- [x] 5. Stage and commit conversion scripts

  **What to do**:
  After all implementation tasks are complete and verified via F1-F4, stage and commit the conversion scripts:
  - **Pre-check**: Verify git is available and repo is initialized. Run `git --version` to confirm git is installed. Then `if (-not (Test-Path ".git")) { git init; Write-Host "Git repo initialized" }` — create a repo if none exists, rather than failing with a confusing error.
  - Ensure `.gitignore` already contains `dataset/` and `.omo/evidence/` (created in Task 1)
  - Run: `git add .gitignore scripts/coco_to_yolo.py scripts/visualize_conversion.py`
  - Commit: `git commit -m "feat: add COCO-to-YOLO26 conversion pipeline"`
  - Optionally commit plan separately: `git add .omo/plans/yolo26-dataset-prep.md && git commit -m "docs: add yolo26 dataset preparation plan"`
  - Verify: `git status` shows clean working tree

  **Must NOT do**:
  - Do NOT commit `dataset/`, `.omo/evidence/`, `.omo/drafts/`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on F1-F4)
  - **Blocked By**: F1, F2, F3, F4 (all must approve before committing)
  
  **Commit**: YES
  - Message: `feat: add COCO-to-YOLO26 conversion pipeline`
  - Files: `.gitignore`, `scripts/coco_to_yolo.py`, `scripts/visualize_conversion.py`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [x] F1. **Plan Compliance Audit** — `oracle` (runs in PARALLEL with F2, F3, F4)
  **Independence note**: F1 checks the PRIMARY `dataset/` output against plan requirements. F3 separately verifies determinism using `dataset_qa/`. These are independent checks — F1's verdict is based on the primary output only, and F3's separate QA does not affect F1's compliance assessment.
  Read the plan end-to-end. For each Must Have: verify implementation exists (read script files, check directory structure, run commands). For each Must NOT Have: search codebase for forbidden patterns (training code, modification of originals, hardcoded dimensions). Check evidence files exist. Compare deliverables against plan.
  **Plus**: Run lightweight YOLO format validation (no model download required — just validates the label files are parseable by Ultralytics):
  ```bash
  python -c "
  from pathlib import Path
  import sys
  errors = 0
  total_lines = 0
  for split in ['train', 'val', 'test']:
      for label_file in Path(f'dataset/labels/{split}').glob('*.txt'):
          for line in label_file.read_text().strip().split('\n'):
              if not line.strip():
                  continue
              parts = line.strip().split()
              if len(parts) != 5:
                  print(f'ERROR: {label_file.name}: expected 5 fields, got {len(parts)}')
                  errors += 1
                  continue
              cls, cx, cy, w, h = parts
              if cls != '0':
                  print(f'ERROR: {label_file.name}: class_id={cls} != 0')
                  errors += 1
              for i, val in enumerate([cx, cy, w, h]):
                  fval = float(val)
                  if not (0.0 <= fval <= 1.0):
                      print(f'ERROR: {label_file.name}: coord[{i}]={fval} out of [0,1]')
                      errors += 1
              total_lines += 1
  print(f'PASS: {total_lines} annotations validated, {errors} errors')
  assert errors == 0, f'Format validation failed with {errors} errors'
  "
  ```
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | Format Valid [PASS] | VERDICT: APPROVE/REJECT`
  Evidence: `.omo/evidence/f1-compliance.txt`

- [x] F2. **Code Quality Review** — `code-reviewer`
  Review `scripts/coco_to_yolo.py` and `scripts/visualize_conversion.py` for: unhandled exceptions, hardcoded paths/values, argparse usage, error messages, file path handling (Windows compatibility). Check for edge cases: missing COCO fields, non-standard filenames, empty annotation lists.
  Output: `Scripts [N/N clean] | Edge Cases [N covered] | VERDICT`
  Evidence: `.omo/evidence/f2-code-review.txt`

- [x] F3. **Real Manual QA** — `unspecified-high` (runs in PARALLEL with F1, F2, F4 — uses separate output dir `dataset_qa/`)

  **Full determinism + correctness check** using a secondary output dir `dataset_qa/`:
  1. Run full conversion to `dataset_qa/` (re-run of the pipeline to a separate dir — same CLI args except new `--output-dir dataset_qa`). This verifies that the **entire pipeline** is deterministic, including label file content, image copying, and `data.yaml` generation.
  2. Compare checksums between primary `dataset/` and secondary `dataset_qa/`:
     - `Get-ChildItem -Recurse dataset/labels/*.txt | ForEach-Object { Get-FileHash $_.FullName -Algorithm SHA256 | Select-Object -ExpandProperty Hash } | Sort-Object | Get-FileHash -Algorithm SHA256` — same for `dataset_qa/labels/`
     - Compare hashes: if they differ, determinism FAILS (non-deterministic loop order, random seed issue, or timing-dependent behavior)
  3. Also run dry-run twice with same seed on a **fresh copy of arguments** (no cached state) to confirm split allocation is also deterministic at the dry-run level.
  4. Run the edge-case QA scenarios from Tasks 2-4 against the primary `dataset/` output.
  5. Clean up `dataset_qa/` after verification: `Remove-Item -Recurse -Force dataset_qa/`.

  Output: `Scenarios [N/N pass] | Determinism [PASS/FAIL] | VERDICT`
  Evidence: `.omo/evidence/f3-qa-results.txt`

- [x] F4. **Scope Fidelity Check** — `unspecified-high`
  Verify: no training code added, no original files modified, no augmentation logic, no hardcoded dimensions. Confirm script is read-only on source data.
  **Plus**: Run programmatic pixel-diff check on debug images to verify boxes were actually drawn:
  ```bash
  python -c "
  import numpy as np
  from PIL import Image
  from pathlib import Path
  debug_dir = Path('dataset/debug')
  passed = 0
  for debug_png in debug_dir.glob('*_debug.png'):
      orig_name = debug_png.name.replace('_debug.png', '.png')
      orig = Image.open(f'dataset/images/train/{orig_name}')
      debug = Image.open(debug_png)
      diff = np.array(orig) != np.array(debug)
      assert diff.any(), f'NO OVERLAY: {debug_png.name} has no visible changes from source'
      passed += 1
      print(f'OK: {debug_png.name} — {diff.sum()} pixels changed')
  assert passed > 0, 'No debug images found'
  print(f'PASS: {passed} debug images verified with overlays')
  "
  ```
  Output: `Scope [PASS/FAIL] | Guardrails [PASS/FAIL] | Overlays [N verified] | VERDICT`
  Evidence: `.omo/evidence/f4-scope-check.txt`

---

## Commit Strategy

The conversion scripts are worth versioning. Create `.gitignore` entry and a single commit at the end:

- **Pre-step**: Add to `.gitignore`:
  ```
  # Generated dataset (432 images + labels — hundreds of MB, fully reproducible)
  dataset/
  # Per-run evidence artifacts
  .omo/evidence/
  ```
- **Message**: `feat: add COCO-to-YOLO26 conversion pipeline`
- **Files**: `.gitignore`, `scripts/coco_to_yolo.py`, `scripts/visualize_conversion.py`
- **Separate commit for plan**: After the feat commit, optionally commit `.omo/plans/yolo26-dataset-prep.md` separately as `docs: add yolo26 dataset preparation plan` (not included in feat commit to keep history clean)
- **Do NOT commit**: `dataset/` (generated data, can be recreated), `.omo/drafts/` (working notes), `.omo/evidence/` (per-run artifacts)

---

## Success Criteria

### Verification Commands
```powershell
# Full pipeline execution
python scripts\coco_to_yolo.py --coco-path data\output_annotations_notebook\combined_coco_split.json --image-dir data\output_splits_notebook --output-dir dataset --overwrite --yes

# File counts (proportion-based — expect ~80/10/10 split; use @() for null-safe empty results)
$total_src = @(Get-ChildItem data\output_splits_notebook\*.png).Count
$train_pct = @(Get-ChildItem dataset\images\train\*.png).Count / $total_src * 100
$val_pct   = @(Get-ChildItem dataset\images\val\*.png).Count / $total_src * 100
$test_pct  = @(Get-ChildItem dataset\images\test\*.png).Count / $total_src * 100
Write-Host "Split: $([math]::Round($train_pct,1))% / $([math]::Round($val_pct,1))% / $([math]::Round($test_pct,1))%"
# Expect each: 78-82% train, 8-12% val, 8-12% test

# Label-image match (use @() wrapper for null-safe counting)
@(Get-ChildItem dataset\labels\train\*.txt).Count -eq @(Get-ChildItem dataset\images\train\*.png).Count
@(Get-ChildItem dataset\labels\val\*.txt).Count -eq @(Get-ChildItem dataset\images\val\*.png).Count
@(Get-ChildItem dataset\labels\test\*.txt).Count -eq @(Get-ChildItem dataset\images\test\*.png).Count

# Config check
Get-Content dataset\data.yaml

# Visual check (use @() wrapper)
@(Get-ChildItem dataset\debug\*.png).Count -ge 5
```

### Final Checklist
- [ ] All "Must Have" present (parent-level split, determinism, dynamic dims, empty .txt, atomic writes, dry-run, bbox validation)
- [ ] All "Must NOT Have" absent (no training code, no original file modification, no augmentation)
- [ ] All 5 acceptance criteria pass
