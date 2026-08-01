"""Hydra config composition sanity checks against the real ``configs/`` tree.

The whole module skips when hydra-core is not installed. Otherwise it
composes the top-level ``default`` config for the YOLO26 experiment variants
and asserts the composed config is fully resolved.
"""

from pathlib import Path

import pytest

pytest.importorskip("hydra")

import yaml  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS = ("yolo26m", "yolo26n")


def _compose(experiment: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="default", overrides=[f"experiment={experiment}"])


def _defaults_entries() -> list[tuple[str, str | None]]:
    """Return (group, value) pairs parsed from the raw ``defaults`` list."""
    raw = (CONFIG_DIR / "default.yaml").read_text(encoding="utf-8")
    top = yaml.safe_load(raw)
    entries: list[tuple[str, str | None]] = []
    for entry in top["defaults"]:
        if entry == "_self_":
            continue
        assert isinstance(entry, dict) and len(entry) == 1, f"unexpected defaults entry: {entry!r}"
        group, value = next(iter(entry.items()))
        entries.append((group, value))
    return entries


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_compose_experiment_has_no_missing_keys(experiment):
    cfg = _compose(experiment)
    assert "???" not in OmegaConf.to_yaml(cfg), f"unresolved keys for experiment={experiment}"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_compose_experiment_sets_expected_names(experiment):
    cfg = _compose(experiment)
    assert cfg.experiment.name == experiment
    assert cfg.model.name == experiment


def test_defaults_groups_resolve_to_existing_configs():
    for group, value in _defaults_entries():
        if value is None:
            continue  # disabled group (e.g. ablation: null)
        config_file = CONFIG_DIR / group / f"{value}.yaml"
        assert (
            config_file.is_file()
        ), f"defaults entry '{group}: {value}' has no matching config file"


def test_dead_groups_removed_from_defaults():
    groups = {group for group, _ in _defaults_entries()}
    dead = {"evaluation", "paths"}
    assert not (
        dead & groups
    ), f"dead config groups still listed in defaults: {sorted(dead & groups)}"
