"""Promoting the final run, or not. The rule, and what it refuses to be.

``final_H2`` and the arm it came from are both adapters on the same merged
base. One of them is the project model. The hard part is not measuring them --
it is fixing, before either number exists, what "better" means and what
happens when they tie.

A tie keeps the incumbent. That is the whole shape of the rule: replacing a
model requires beating it.

Nothing here loads a model, opens the dataset or touches a device.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "23_final_eval.py"
EARLIER = ROOT / "scripts" / "21_holdout_eval.py"


def names_used(path: Path) -> set:
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
    spec = importlib.util.spec_from_file_location("final_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestThePromotionRule:

    def test_strictly_lower_promotes_the_final_run(self, evalmod):
        assert evalmod.project_model(
            {"existing_H2": 0.30, "final_H2": 0.20}) == "final_H2"

    def test_higher_keeps_the_incumbent(self, evalmod):
        assert evalmod.project_model(
            {"existing_H2": 0.20, "final_H2": 0.30}) == "existing_H2"

    def test_a_tie_keeps_the_incumbent(self, evalmod):
        """Not a draw, not a coin toss: replacing a model requires beating it."""
        assert evalmod.project_model(
            {"existing_H2": 0.25, "final_H2": 0.25}) == "existing_H2"

    def test_there_is_no_tolerance_band_in_either_direction(self, evalmod):
        assert evalmod.project_model(
            {"existing_H2": 0.25, "final_H2": 0.25 - 1e-12}) == "final_H2"
        assert evalmod.project_model(
            {"existing_H2": 0.25, "final_H2": 0.25 + 1e-12}) == "existing_H2"

    def test_a_missing_candidate_is_refused_rather_than_defaulted(self, evalmod):
        for means in ({"existing_H2": 0.2}, {"final_H2": 0.2}, {},
                      {"existing_H2": 0.2, "final_H2": None},
                      {"existing_H2": "0.2", "final_H2": 0.3}):
            with pytest.raises(ValueError):
                evalmod.project_model(means)

    def test_the_rule_reads_nothing_but_the_two_means(self, evalmod):
        import inspect

        source = inspect.getsource(evalmod.project_model)
        for forbidden in ("seconds", "vram", "bytes", "rows", "open(",
                          "Path", "json", "steps"):
            assert forbidden not in source, forbidden

    def test_training_size_and_cost_are_declared_descriptive(self, evalmod):
        document = evalmod.criterion_document()
        assert "mean masked validation loss" in document["primary_criterion"]
        assert "keeps the incumbent" in document["primary_criterion"]
        for reading in ("training rows", "seconds per row", "peak VRAM",
                        "adapter bytes", "optimizer steps"):
            assert reading in document["descriptive_only"], reading
        assert document["incumbent"] == "existing_H2"


class TestTheHeldOutSetIsTheSameOne:

    def test_the_selection_is_imported_not_restated(self, evalmod):
        """Identical by construction, so it cannot drift."""
        earlier = evalmod.holdout()
        document = evalmod.criterion_document()
        assert document["held_out"]["file"] == earlier.VAL_FILE
        assert document["held_out"]["rows"] == earlier.VAL_ROWS == 320
        assert document["held_out"]["pairs"] == earlier.N_VAL_PAIRS == 40
        source = SCRIPT.read_text(encoding="utf-8")
        assert "21_holdout_eval.py" in source
        assert "N_VAL_PAIRS = " not in source
        assert "VAL_ROWS = " not in source

    def test_the_test_split_is_unreachable(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "instruct_inv_test" not in source
        assert "_test.jsonl" not in source

    def test_nothing_is_discovered(self):
        used = names_used(SCRIPT)
        for forbidden in ("glob", "rglob", "iterdir", "listdir", "scandir",
                          "st_mtime", "walk"):
            assert forbidden not in used, forbidden

    def test_both_adapters_are_named_by_the_caller(self, evalmod):
        options = set()
        for action in evalmod.build_parser()._actions:
            options.update(action.option_strings)
        assert "--existing-h2-adapter" in options
        assert "--final-h2-adapter" in options
        assert "--reference-record" in options


class TestThePipelineIsDemonstratedNotAsserted:

    def test_the_incumbent_is_rescored_rather_than_read(self, evalmod):
        """Its earlier number is a cross-check, never the comparison input."""
        source = SCRIPT.read_text(encoding="utf-8")
        assert "score_adapter(name" in source
        assert "reproduces its recorded mean" in source
        assert "refusing to decide" in source

    def test_the_reference_record_is_required(self, evalmod):
        parsed = evalmod.build_parser().parse_args([])
        assert parsed.reference_record is None

    def test_both_candidates_are_scored_in_one_process(self, evalmod):
        assert evalmod.CANDIDATES == ("existing_H2", "final_H2")
        source = SCRIPT.read_text(encoding="utf-8")
        assert "for name in CANDIDATES:" in source


class TestItChangesNothing:

    def test_it_trains_nothing_and_saves_no_state(self):
        used = names_used(SCRIPT)
        for forbidden in ("backward", "AdamW", "save_pretrained", "zero_grad",
                          "build_model", "prepare_training", "run_gate",
                          "run_final", "requires_grad_"):
            assert forbidden not in used, forbidden
        assert "no_grad" in used
        assert "eval" in used

    def test_it_builds_no_pack_and_publishes_nothing(self):
        used = names_used(SCRIPT)
        for forbidden in ("build", "publish", "push"):
            assert forbidden not in used, forbidden

    def test_it_cold_loads_through_the_one_correct_loader(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "load_finetuned" in source
        assert "verify_digest=True" in source

    def test_the_record_is_written_once(self):
        assert "write_once_json" in SCRIPT.read_text(encoding="utf-8")

    def test_it_leaks_no_identifier(self):
        from src.training.longrun import leaked_identifiers

        assert leaked_identifiers(SCRIPT.read_text(encoding="utf-8")) == []

    def test_it_stays_out_of_the_gpu_pack(self):
        from src.training import pack

        assert pack.classify("scripts/23_final_eval.py")[0] == "exclude"
        assert pack.classify("tests/test_final_eval.py")[0] == "exclude"
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        assert "scripts/23_final_eval.py" not in included
        assert "tests/test_final_eval.py" not in included

    def test_the_earlier_criterion_is_untouched_by_this_one(self, evalmod):
        """21 recorded the H1/H2 decision. Its rule must still say what it said.

        Read as the value the module exposes, not as text in the file: the
        constant is written across two source lines, and a check that a
        sentence appears contiguously would fail on formatting and pass on a
        reworded rule.
        """
        earlier = evalmod.holdout()
        assert earlier.ARMS == ("H1", "H2")
        assert earlier.PRIMARY_CRITERION == (
            "mean masked validation loss over the 320 frozen held-out rows; "
            "the lower value wins; exactly equal is a draw")
        assert earlier.winner({"H1": 0.25, "H2": 0.25}) == "draw"

    def test_the_two_rules_differ_where_they_should(self, evalmod):
        """A draw there, the incumbent here. Both deliberate, both frozen."""
        earlier = evalmod.holdout()
        assert earlier.winner({"H1": 0.25, "H2": 0.25}) == "draw"
        assert evalmod.project_model(
            {"existing_H2": 0.25, "final_H2": 0.25}) == "existing_H2"
