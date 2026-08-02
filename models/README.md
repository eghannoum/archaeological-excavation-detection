# Best-model registry

This directory holds the best-model registry used by `--model auto`, plus
optional pre-downloaded base weights. Everything here is gitignored except
this README (`models/*` with `!models/README.md`).

## Files

- `models/best.json` — registry entry written by `train.py` / `evaluate.py`:
  the model path, experiment name, metrics (`mAP50`, `mAP50-95`, `precision`,
  `recall`, `f1_score`), and an `updated_at` timestamp (UTC). It is
  overwritten only when a checkpoint with a better score arrives.
- `models/best.pt` — copy of the current best checkpoint, kept in sync with
  `best.json`.
- `models/*.pt` (optional) — pre-downloaded base weights such as
  `models/yolo26m.pt`, used by `train.py` when present. These are
  user-managed and gitignored via the `*.pt` rule.

## How it is used

- `train.py` and `evaluate.py` update the registry automatically after each
  run, keeping the highest-scoring checkpoint as the best model.
- `inference.py`, `evaluate.py`, and `dashboard.py` resolve `--model auto`
  through the registry first, then fall back to scanning `runs/` /
  `experiments/` for the highest-mAP50 checkpoint.
- An explicit `--model <path>` always overrides the registry.

Run `python scripts/train.py experiment=yolo26m` to populate this directory.
