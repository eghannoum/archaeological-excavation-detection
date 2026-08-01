# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Nothing yet.

## [0.1.0] - 2026-08-01

Pre-publication hardening: audit, license, CI, tests, inference rewrite, config cleanup.

### Added

- MIT license and community files: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY.
- `pyproject.toml` with ruff/black/isort/mypy configuration and package metadata.
- `requirements-dev.txt` (pytest, ruff, black, isort, mypy, pre-commit) and CUDA-pinned `requirements-gpu.txt`.
- Pre-commit hooks (`.pre-commit-config.yaml`) and `.gitattributes`.
- Dockerfile and `docs/DEPLOYMENT.md` (ONNX/TensorRT export and serving notes).
- Initial pytest suite (config resolution, dataset validation, inference, conversion).
- Ablation studies: image size (320/640/1280), optimizer (AdamW/SGD), augmentation (none/light/heavy).
- Faster R-CNN and DETR 3-fold CV training pipelines.

### Changed

- Rewrote `scripts/inference.py`: clarified CLI flags, image file/folder/glob sources, and CSV/label/image outputs with a corrected default model path.
- Made `requirements.txt` portable (CPU-installable torch; optional CUDA build moved to `requirements-gpu.txt`).
- Cleaned up Hydra config groups and removed dead configuration keys.
- Corrected dataset numbers in `docs/paper/reproducibility.md` (432 tiles, parent-scene splits).
- Extended `.gitignore` for generated outputs, model weights, and internal tool state.

### Fixed

- Removed absolute user paths from tracked files and untracked internal tool state.
- Fixed `albumentations` dependency drift between YAML config and installed version.
- Fixed stale model-path defaults in inference and evaluation scripts.

### Removed

- Scratch/private scripts and task-scoped log files.
- Generated `configs/best_hparams.yaml` from the source tree.

[Unreleased]: https://github.com/<owner>/STI-Unauthorized-Archaeological-Excavations/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/<owner>/STI-Unauthorized-Archaeological-Excavations/releases/tag/v0.1.0
