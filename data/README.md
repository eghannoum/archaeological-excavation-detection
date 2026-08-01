# data/

This directory is **gitignored** — the raw dataset is **not distributed** with this repository.

## Layout (raw source data)

The full raw dataset (satellite imagery tiles + COCO annotations) lives here on a
per-developer basis. The documented layout is:

```
data/
├── input_images/                      # raw satellite image tiles (source)
├── output_annotations_notebook/
│   └── combined_coco_split.json       # COCO annotations generated in a notebook
├── output_splits_notebook/            # split artifacts generated in a notebook
└── sample/                            # committed: small demo subset (see below)
```

> Note: only `sample/` is committed. `input_images/`, `output_annotations_notebook/`,
> and `output_splits_notebook/` are developer-local inputs referenced by
> `scripts/coco_to_yolo.py` and the data-preparation notebooks.

## Reproduction pipeline

1. Place raw tiles in `data/input_images/` and the COCO annotation file at
   `data/output_annotations_notebook/combined_coco_split.json`.
2. Run `scripts/coco_to_yolo.py` to generate the YOLO-format dataset under
   `dataset/` (images + labels + `data.yaml`). See the script docstring for usage.
3. The generated `dataset/` tree is also gitignored — it is reproducible output.

## sample/ — quickstart demo subset

`sample/` contains **12 satellite image tiles** (6 train / 3 val / 3 test) with
their YOLO labels (1 class: `hole`). Use it to sanity-check the pipeline without
downloading the full dataset:

```powershell
# Dry-run the Hydra config
.venv\Scripts\python.exe scripts\train.py --info

# Inference on the sample images
.venv\Scripts\python.exe scripts\inference.py --source data\sample\images --save-img
```

The sample files are copies of real tiles from the full dataset and are covered by
the same license (MIT, see LICENSE). They are kept intentionally small so the
repository stays light.
