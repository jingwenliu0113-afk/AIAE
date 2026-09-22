#!/usr/bin/env python3
"""Hyperparameters read from `configs/`, not compiled into the code.

A configuration file only means something if the code actually reads it, and
the usual way that fails is silent: a key is renamed, nothing raises, and the
run keeps the old compiled-in value while the file on disk says otherwise. So
an unknown key here is an error, not a no-op, and `tests/test_config.py`
proves the file is read by changing a value and watching it arrive.

Absent keys keep the dataclass default. That is deliberate: the defaults are
documented next to their validation in `src/vision/model.py`, and duplicating
all of them here would make two places to change.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from src.vision.model import Augmentation, ModelError, TrainConfig

ROOT = Path(__file__).resolve().parents[1]
VISION_CONFIG = ROOT / "configs" / "vision.yaml"


class ConfigError(ModelError):
    """The configuration file does not describe this program."""


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _section(data: dict, name: str, cls) -> dict:
    values = data.get(name) or {}
    if not isinstance(values, dict):
        raise ConfigError(f"`{name}` must be a mapping, not {type(values).__name__}")
    unknown = sorted(set(values) - _fields(cls))
    if unknown:
        raise ConfigError(
            f"`{name}` has no field(s) {', '.join(unknown)}; a key that is not "
            f"a field would be read and ignored, which is how a configuration "
            f"stops describing the run it names")
    return values


def load_train_config(path: Path | None = None) -> TrainConfig:
    """`TrainConfig` with `configs/vision.yaml` applied over the defaults."""
    path = path or VISION_CONFIG
    if not path.is_file():
        raise ConfigError(f"no configuration at {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name} must be a mapping at the top level")
    unknown = sorted(set(loaded) - {"train", "augmentation"})
    if unknown:
        raise ConfigError(f"{path.name} has unknown section(s): {', '.join(unknown)}")
    augmentation = Augmentation(**_section(loaded, "augmentation", Augmentation))
    train = _section(loaded, "train", TrainConfig)
    if "augmentation" in train:
        raise ConfigError("put augmentation in its own section, not inside `train`")
    return TrainConfig(augmentation=augmentation, **train)
