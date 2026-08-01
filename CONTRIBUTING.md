# Contributing

Thank you for considering a contribution to the STI Unauthorized Archaeological Excavations project. This document describes how to set up a development environment, the quality gates every change must pass, and how to open a pull request.

## Table of contents

- [Development setup](#development-setup)
- [Code quality](#code-quality)
- [Testing](#testing)
- [Commit conventions](#commit-conventions)
- [Pull request process](#pull-request-process)
- [Dataset and weights](#dataset-and-weights)

## Development setup

1. Fork the repository and clone your fork.
2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate

   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

   `requirements-dev.txt` installs pytest, ruff, black, isort, mypy, and pre-commit.

3. Install the git hooks:

   ```bash
   pre-commit install
   ```

4. Run the existing checks to confirm the environment works:

   ```bash
   pre-commit run --all-files
   pytest
   ```

## Code quality

Linting and formatting are configured in `pyproject.toml` (line length 100).

| Tool | Purpose | Command |
|------|---------|---------|
| `ruff` | Linting (E/F/W/I/UP/B/SIM rules) | `ruff check .` |
| `black` | Formatting | `black .` |
| `isort` | Import sorting (Black profile) | `isort .` |
| `mypy` | Static type checking | `mypy scripts` |

Run all of them before committing. The pre-commit hooks (see `.pre-commit-config.yaml`) enforce `ruff`, `ruff-format`, `black`, `isort`, end-of-file newlines, and trailing-whitespace removal automatically.

Keep changes focused: if you are fixing a bug, do not refactor unrelated code in the same change.

## Testing

The project uses pytest (test discovery rooted at `tests/`). Slow tests are marked `slow` and can be skipped with:

```bash
pytest                              # quick suite
pytest -m "not slow"                # explicit skip of slow tests
pytest --cov=scripts                # coverage report
```

Add tests for new functionality and for bug fixes. Target at least 80% coverage on `scripts/` core modules for new code. If you touch dataset, config, inference, or conversion code, extend the corresponding test module.

## Commit conventions

Use the semantic commit style already present in this repository:

- `feat: ...` — new feature or capability
- `fix: ...` — bug fix
- `docs: ...` — documentation only
- `refactor: ...` — code change with no behavior change
- `test: ...` — tests only
- `chore: ...` — tooling, dependencies, CI

Keep the subject under 72 characters, imperative mood, no trailing period. Split unrelated changes into separate commits (e.g. UI vs. logic vs. tests).

## Pull request process

1. Branch from `main` with a descriptive name (e.g. `fix/train-cv-return-code`).
2. Make your change, keeping the quality gates green: lint clean, tests passing, no secrets or absolute user paths introduced.
3. Write or update tests, and update documentation (`README.md`, `docs/`) if the change affects usage.
4. Push and open a pull request against `main`. Describe the motivation, the change, and how you verified it.
5. A maintainer will review. Address review comments; keep the conversation focused on the change.

Do not commit generated artifacts: `dataset/`, `runs/`, `experiments/`, `mlruns/`, `outputs/`, `data/`, model weights (`*.pt`, `*.pth`, `*.onnx`), or `configs/best_hparams.yaml` — all are gitignored.

## Dataset and weights

The full dataset and trained weights are not distributed in this repository (see the README). Reproducing the benchmark requires the raw annotated imagery and training from scratch; never commit large binaries. If your contribution needs sample data, keep it small and place it under `data/sample/`.

## Code of conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
