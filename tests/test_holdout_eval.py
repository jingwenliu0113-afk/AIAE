"""The held-out evaluation: the criterion, and what it refuses to be.

This scores two finished arms on data neither saw. Almost nothing about it is
hard except the part that is easy to get wrong: deciding, after seeing three
numbers, which of them was the one that mattered. So the criterion is a pure
function frozen in the script above the code that produces any number, and
these tests pin it there.

Nothing here loads a model, opens the dataset or touches a device.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "21_holdout_eval.py"


def names_used(path: Path) -> set:
    """Every name and attribute the file actually references.

    Read from the syntax tree so that a docstring saying "no glob, no listing"
    is prose rather than a violation, and so that a call the text scan did not
    anticipate is still caught.
    """
    import ast

    used = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add(node.asname or node.name.split(".")[-1])
    return used


@pytest.fixture(scope="module")
def evalmod():
    spec = importlib.util.spec_from_file_location("holdout_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheCriterionIsOneNumber:

    def test_lower_mean_validation_loss_wins(self, evalmod):
        assert evalmod.winner({"H1": 0.20, "H2": 0.30}) == "H1"
        assert evalmod.winner({"H1": 0.30, "H2": 0.20}) == "H2"

    def test_exact_equality_is_a_draw(self, evalmod):
        assert evalmod.winner({"H1": 0.25, "H2": 0.25}) == "draw"

    def test_there_is_no_tolerance_band(self, evalmod):
        """A margin invented here would be a threshold chosen afterwards."""
        assert evalmod.winner({"H1": 0.25, "H2": 0.25 + 1e-12}) == "H1"
        assert evalmod.winner({"H1": 0.25 + 1e-12, "H2": 0.25}) == "H2"

    def test_a_missing_arm_is_refused_rather_than_defaulted(self, evalmod):
        for means in ({"H1": 0.2}, {"H2": 0.2}, {}, {"H1": 0.2, "H2": None},
                      {"H1": "0.2", "H2": 0.3}):
            with pytest.raises(ValueError):
                evalmod.winner(means)

    def test_the_criterion_does_not_read_anything_but_the_two_means(self,
                                                                    evalmod):
        import inspect

        source = inspect.getsource(evalmod.winner)
        for forbidden in ("seconds", "vram", "bytes", "training", "open(",
                          "Path", "json"):
            assert forbidden not in source, forbidden

    def test_speed_memory_and_size_are_declared_descriptive(self, evalmod):
        document = evalmod.criterion_document()
        assert "mean masked validation loss" in document["primary_criterion"]
        for reading in ("seconds per row", "peak VRAM", "adapter bytes",
                        "training loss"):
            assert reading in document["descriptive_only"], reading
        assert reading not in document["primary_criterion"]


class TestTheHeldOutSetIsTheFrozenOne:

    def test_it_is_the_smoke_test_s_selection(self, evalmod):
        smoke = (ROOT / "scripts" / "13_lora_smoke.py").read_text(
            encoding="utf-8")
        assert "N_VAL_PAIRS = 40" in smoke
        assert evalmod.N_VAL_PAIRS == 40
        assert evalmod.VAL_ROWS == 320
        assert evalmod.VAL_FILE == "instruct_inv_val.jsonl"

    def test_the_test_split_is_not_reachable_from_this_script(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "instruct_inv_test" not in source
        assert "_test.jsonl" not in source
        assert '"test"' not in source.replace('"test_split_read": False', "")

    def test_the_adapter_directory_is_named_not_discovered(self, evalmod,
                                                           tmp_path):
        assert evalmod.adapter_dir(tmp_path, "H1") == tmp_path / "H1" / "adapter"
        assert evalmod.adapter_dir(tmp_path, "H2") == tmp_path / "H2" / "adapter"
        with pytest.raises(ValueError):
            evalmod.adapter_dir(tmp_path, "H3")

    def test_nothing_is_globbed_sorted_or_dated(self):
        """Against the syntax tree, not the prose.

        The docstrings say the script discovers nothing, and a text scan would
        fail on the sentence that says so while passing on any spelling it did
        not anticipate. What is actually forbidden is calling those things.
        """
        called = names_used(SCRIPT)
        for forbidden in ("glob", "rglob", "iterdir", "listdir", "scandir",
                          "st_mtime", "getmtime", "walk"):
            assert forbidden not in called, forbidden

    def test_the_selection_digest_depends_on_order(self, evalmod):
        a = evalmod.selection_digest(["x", "y", "z"])
        assert a != evalmod.selection_digest(["x", "z", "y"])
        assert a == evalmod.selection_digest(["x", "y", "z"])
        assert len(a) == 64


class TestItTrainsNothing:

    def test_the_script_takes_no_gradient_and_saves_no_state(self, evalmod):
        called = names_used(SCRIPT)
        for forbidden in ("backward", "AdamW", "save_pretrained",
                          "requires_grad_", "build_model", "prepare_training",
                          "make_training_step", "zero_grad"):
            assert forbidden not in called, forbidden
        assert "no_grad" in called
        assert "eval" in called

    def test_the_command_line_cannot_change_anything_measured(self, evalmod):
        options = set()
        for action in evalmod.build_parser()._actions:
            options.update(action.option_strings)
        for forbidden in ("--lr", "--learning-rate", "--rank", "--alpha",
                          "--epochs", "--seed", "--rows", "--pairs",
                          "--max-length", "--dtype", "--split", "--test",
                          "--force", "--threshold", "--criterion-metric"):
            assert forbidden not in options, forbidden
        assert options >= {"--arms-root", "--out", "--criterion", "--device"}

    def test_it_cold_loads_through_the_one_correct_loader(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "load_finetuned" in source
        assert "verify_digest=True" in source

    def test_the_record_is_written_once(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "write_once_json" in source

    def test_it_declares_that_it_starts_nothing(self, evalmod):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "does not tune, retrain" in source

    def test_it_leaks_no_identifier(self):
        from src.training.longrun import leaked_identifiers

        assert leaked_identifiers(SCRIPT.read_text(encoding="utf-8")) == []

    def test_it_stays_out_of_the_gpu_pack(self):
        """Mac-only work does not travel to the execution node."""
        from src.training import pack

        assert pack.classify("scripts/21_holdout_eval.py")[0] == "exclude"
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        assert "scripts/21_holdout_eval.py" not in included
        assert "tests/test_holdout_eval.py" not in included
