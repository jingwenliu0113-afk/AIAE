"""`configs/vision.yaml` is read, and a key that is not a field is refused.

A configuration file is easy to add and easy to leave disconnected. The two
failure modes are opposite: the file is never read, so a run silently keeps
the compiled-in value; or the file is read loosely, so a misspelled key is
ignored and the run silently keeps the compiled-in value again. Both end with
a file on disk that describes something other than what ran.

So every test here either changes a value and watches it arrive, or breaks the
file and watches it be refused.
"""

from __future__ import annotations

import pytest
import yaml

from src.config import VISION_CONFIG, ConfigError, load_train_config
from src.vision.model import Augmentation, TrainConfig


def write(tmp_path, data) -> object:
    path = tmp_path / "vision.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The shipped file
# ---------------------------------------------------------------------------

def test_the_shipped_configuration_loads():
    config = load_train_config()
    assert isinstance(config, TrainConfig)
    assert config.epochs >= 1 and config.batch_size >= 1


def test_the_shipped_configuration_states_every_hyperparameter():
    """A partially-stated config is the one that drifts from the code.

    Absent keys fall back to the dataclass default, which is fine for a
    one-off override but not for the file the repository ships: a reader has
    to be able to see the whole run in it.
    """
    loaded = yaml.safe_load(VISION_CONFIG.read_text(encoding="utf-8"))
    stated = set(loaded["train"]) | {"augmentation"}
    fields = {f for f in TrainConfig.__dataclass_fields__}
    assert fields - stated == set(), f"not stated in configs/: {sorted(fields - stated)}"
    aug = set(loaded["augmentation"])
    assert set(Augmentation.__dataclass_fields__) - aug == set()


# ---------------------------------------------------------------------------
# It is actually read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("epochs", 3), ("batch_size", 8), ("learning_rate", 0.0004),
    ("weight_decay", 0.002), ("seed", 7), ("freeze_backbone", False),
    ("trainable_stages", 2), ("label_smoothing", 0.1),
])
def test_a_changed_value_reaches_the_config(tmp_path, field, value):
    default = getattr(TrainConfig(), field)
    assert value != default, "this case would pass even if the file were ignored"
    loaded = load_train_config(write(tmp_path, {"train": {field: value}}))
    assert getattr(loaded, field) == value


def test_a_changed_augmentation_value_reaches_the_config(tmp_path):
    loaded = load_train_config(write(tmp_path, {"augmentation": {"brightness": 0.42}}))
    assert loaded.augmentation.brightness == 0.42


def test_an_absent_key_keeps_the_documented_default(tmp_path):
    loaded = load_train_config(write(tmp_path, {"train": {"epochs": 2}}))
    assert loaded.epochs == 2
    assert loaded.batch_size == TrainConfig().batch_size


# ---------------------------------------------------------------------------
# It is refused when it stops describing the program
# ---------------------------------------------------------------------------

def test_a_key_that_is_not_a_field_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no field"):
        load_train_config(write(tmp_path, {"train": {"learning_rat": 0.01}}))


def test_an_unknown_section_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="unknown section"):
        load_train_config(write(tmp_path, {"trainn": {"epochs": 2}}))


def test_a_missing_file_is_refused_rather_than_defaulted(tmp_path):
    with pytest.raises(ConfigError, match="no configuration"):
        load_train_config(tmp_path / "absent.yaml")


def test_a_value_the_dataclass_rejects_still_raises(tmp_path):
    """Validation stays where it is documented, next to the field."""
    with pytest.raises(Exception, match="learning_rate"):
        load_train_config(write(tmp_path, {"train": {"learning_rate": 5.0}}))


def test_a_scalar_where_a_mapping_belongs_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_train_config(write(tmp_path, {"train": 8}))
