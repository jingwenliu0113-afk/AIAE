"""The demonstration: where each report says its bricks came from, and why.

No model, no GPU, no network, and no Phase 2 artifact is read -- a scanning
test asserts the last one rather than promising it. The point of the
demonstration is that a reader can run it, so the tests run it too: the CLI is
driven through ``main()`` with real arguments and its exit codes are checked.

The bulk of what is checked here is provenance, because that is what a
demonstration can get wrong without looking wrong. Three modes, mutually
exclusive; a measured token count only where something measured one; a
placement gate only where a placement gate ran; and a check nobody can decide
reported as undecided rather than failed.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constraints.placement_decode import (CONNECTIVITY_MODES,
                                              PLACEMENT_STOP_REASONS)
from src.data.bricks import PART_VOCAB, Brick, find_collisions
from src.demo import showcase
from src.demo.showcase import (MODE_DECODED, MODE_SAMPLE, MODE_SUPPLIED,
                               MODES, PLACEMENT_NOTICE, SAMPLES,
                               STANDING_NOTICE, TERM_MEASURED, TERM_OPERATOR,
                               TERM_STATED, TERM_UNAVAILABLE,
                               TERMINATION_DEPENDENT, TERMINATIONS,
                               TOKENS_DERIVED, TOKENS_MEASURED, Decoded,
                               Sample, ShowcaseError, format_report,
                               inspect_decoded, inspect_sample,
                               inspect_supplied, parse_inventory, passed,
                               plan_view, remaining, sample,
                               tokens_if_decoded, write_ldraw)
from src.constraints.inventory_decode import InventoryGate
from src.generation.brickgpt import BrickGate, Slots
from src.rendering.preview import (BASE_FONT, CJK_FONT_CANDIDATES,
                                   COLLISION_COLOUR, DEFAULT_HEADING,
                                   PART_COLOURS, PreviewError,
                                   brick_facecolours, resolve_font,
                                   safe_heading, validate_preview_path,
                                   write_preview)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "26_showcase.py"

TWO = "2x4 (0,0,0)\n2x4 (0,0,1)\n"


def stub_slots() -> Slots:
    """Distinct, checkable ids and no tokenizer. Only the gates read them."""
    return Slots(dims=list(range(1000, 1008)),
                 posns=list(range(2000, 2020)),
                 literal_x=90, literal_open=91, literal_comma=92,
                 literal_close=93, eos=99)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("m26", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decoded(**kw) -> Decoded:
    base = dict(n_tokens=21, termination="normal_eos", model="published",
                device="cpu", seed=0, temperature=0.6, max_bricks=80,
                max_tokens=800)
    return Decoded(**{**base, **kw})


GATE_COUNTERS = {"bricks_placed": 2, "eos_deferrals": 0,
                 "candidates_masked": {0: 1}, "connectivity": "off"}


# --------------------------------------------------------------------------


class TestInventoryEnteredByHand:
    def test_a_plain_inventory_parses_in_vocabulary_order(self):
        assert parse_inventory("2x2:6,2x4:10,1x2:8") == {
            "1x2": 8, "2x2": 6, "2x4": 10}

    def test_a_rotated_spelling_is_the_same_stock(self):
        assert parse_inventory("4x1:3") == {"1x4": 3}
        assert parse_inventory("4x2:1") == {"2x4": 1}

    def test_both_spellings_of_one_part_are_refused_not_summed(self):
        """Which of the two counts was meant is not the parser's guess."""
        with pytest.raises(ShowcaseError, match="same part"):
            parse_inventory("1x4:2,4x1:3")

    def test_a_part_outside_the_vocabulary_is_refused(self):
        with pytest.raises(ShowcaseError, match="not one of the eight"):
            parse_inventory("2x8:1")

    def test_every_vocabulary_part_is_accepted(self):
        for part in PART_VOCAB:
            assert parse_inventory(f"{part}:1") == {part: 1}

    def test_malformed_entries_are_refused(self):
        for bad in ("", "   ", "2x4", "2x4:", "2x4:zero", "2x4:0",
                    "2x4:-1", "banana:2", ":3"):
            with pytest.raises(ShowcaseError):
                parse_inventory(bad)

    def test_remaining_keeps_the_overdraw_visible(self):
        """Clamping at zero would hide the only thing worth seeing."""
        assert remaining({"2x4": 2}, {"2x4": 3}) == {"2x4": -1}
        assert remaining({"2x4": 2}, {}) == {"2x4": 2}


class TestThePlanView:
    def test_an_empty_structure_says_so(self):
        assert plan_view([]) == "(no bricks)"

    def test_one_layer_per_z_lowest_first(self):
        view = plan_view([Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 2)])
        assert view.index("z=0") < view.index("z=2")
        assert "z=1" not in view

    def test_a_cell_claimed_twice_is_marked(self):
        view = plan_view([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 0)])
        assert "*" in view

    def test_a_legal_structure_has_no_overlap_mark(self):
        bricks = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 4, 0)]
        assert find_collisions(bricks) == []
        grid = [ln for ln in plan_view(bricks, legend=False).splitlines()
                if ln.startswith("  x=")]
        assert grid and all("*" not in ln for ln in grid)

    def test_the_legend_names_every_brick_when_it_fits(self):
        bricks = [Brick(2, 4, 0, 0, 0), Brick(2, 2, 5, 5, 1)]
        view = plan_view(bricks)
        for b in bricks:
            assert str(b) in view

    def test_a_huge_structure_says_the_legend_was_omitted(self):
        bricks = [Brick(1, 1, x, y, 0) for x in range(9) for y in range(9)]
        assert "legend omitted" in plan_view(bricks)

    def test_it_does_not_call_itself_a_render(self):
        assert "render" not in plan_view([Brick(1, 1, 0, 0, 0)])


class TestTheCpuThreeDimensionalPreview:
    def test_it_writes_a_real_png_without_a_model(self, tmp_path):
        out = write_preview(
            tmp_path / "tower.png",
            [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)],
            title="tower")
        assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert out.stat().st_size > 1_000

    def test_it_can_write_svg(self, tmp_path):
        out = write_preview(tmp_path / "one.svg", [Brick(1, 1, 0, 0, 0)])
        assert "<svg" in out.read_text(encoding="utf-8")[:1000]

    def test_empty_out_of_bounds_and_unknown_formats_are_refused(self,
                                                                  tmp_path):
        with pytest.raises(PreviewError, match="at least one"):
            write_preview(tmp_path / "empty.png", [])
        with pytest.raises(PreviewError, match="out-of-bounds"):
            write_preview(tmp_path / "bad.png", [Brick(2, 4, 19, 0, 0)])
        with pytest.raises(PreviewError, match="does not know"):
            write_preview(tmp_path / "unknown.png", [Brick(2, 8, 0, 0, 0)])
        with pytest.raises(PreviewError, match=r"\.png or \.svg"):
            write_preview(tmp_path / "bad.jpg", [Brick(1, 1, 0, 0, 0)])

    def test_long_request_titles_are_wrapped_and_bounded(self):
        heading = safe_heading("word " * 80)["heading"]
        lines = heading.splitlines()
        assert len(lines) == 2
        assert all(len(line) <= 72 for line in lines)
        assert heading.endswith("…")

    def test_preview_path_validation_has_no_filesystem_side_effect(self,
                                                                    tmp_path):
        parent = tmp_path / "not-created"
        with pytest.raises(PreviewError, match=r"\.png or \.svg"):
            validate_preview_path(parent / "bad.jpg")
        assert not parent.exists()

    def test_the_module_denies_physics_and_stability_claims(self):
        body = (ROOT / "src/rendering/preview.py").read_text(encoding="utf-8")
        flat = " ".join(body.split()).lower()
        assert "not a photorealistic render" in flat
        assert "not a physics or stability analysis" in flat


# --------------------------------------------------------------------------
# Mode 1: the stored briefs, used whole
# --------------------------------------------------------------------------


class TestSampleModeIsUsedWhole:
    def test_every_brief_is_labelled_hand_written_and_frozen(self):
        assert "Hand-written, not generated" in Sample.__doc__
        assert "used whole" in Sample.__doc__

    def test_an_unknown_brief_is_refused(self):
        with pytest.raises(ShowcaseError, match="not a stored brief"):
            sample("no-such-brief")
        with pytest.raises(ShowcaseError, match="not a stored brief"):
            inspect_sample("no-such-brief")

    def test_the_provenance_names_the_sample_as_the_source_of_everything(self):
        report = inspect_sample("tower")
        prov = report["provenance"]
        assert prov["mode"] == MODE_SAMPLE
        assert prov["text_origin"] == "sample:tower"
        assert prov["caption_source"] == MODE_SAMPLE
        assert prov["inventory_source"] == MODE_SAMPLE
        assert prov["variant_of"] is None
        assert prov["decode"] is None

    def test_a_sample_states_its_termination_and_never_measures_one(self):
        report = inspect_sample("tower")
        assert report["provenance"]["termination"]["source"] == TERM_STATED
        assert report["provenance"]["tokens"]["source"] == TOKENS_DERIVED
        assert "not measured" in report["provenance"]["termination"]["note"]

    def test_every_brief_declares_a_termination_the_gates_can_produce(self):
        for name, s in SAMPLES.items():
            assert s.name == name and s.shows.strip()
            assert s.inventory and s.text.strip()
            assert s.termination in TERMINATIONS, name

    def test_the_tower_passes_every_check(self):
        report = inspect_sample("tower")
        assert passed(report) is True
        assert all(report["checks_determinable"].values())

    def test_the_overdrawn_brief_fails_on_stock_and_nothing_else(self):
        report = inspect_sample("overdrawn")
        failed = [k for k, v in report["checks"].items() if not v]
        assert failed == ["inventory_valid", "deterministic_core_success"]
        assert report["inventory"]["remaining"] == {"2x4": -1}
        assert report["inventory"]["count_overflow_amount"] == 1

    def test_the_colliding_brief_names_the_pair(self):
        report = inspect_sample("collision")
        assert report["checks"]["collision_free"] is False
        assert report["geometry"]["colliding_pairs"] == [[0, 1]]
        assert "*" in report["plan_view"]

    def test_the_brief_in_pieces_fails_only_connectivity(self):
        report = inspect_sample("in-pieces")
        failed = [k for k, v in report["checks"].items() if not v]
        assert failed == ["stud_only_connected", "deterministic_core_success"]
        assert report["connectivity"]["n_components"] == 2

    def test_a_sample_report_carries_no_decode_settings(self):
        assert inspect_sample("tower")["provenance"]["decode"] is None
        assert inspect_sample("tower")["request"]["placement_gate"] is False


# --------------------------------------------------------------------------
# Mode 2: supplied text, and the three claims it may not make
# --------------------------------------------------------------------------


class TestSuppliedModeRefusesWhatItCannotKnow:
    def test_a_measured_token_count_is_refused_not_relabelled(self):
        with pytest.raises(ShowcaseError, match="no measured token count"):
            inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin",
                             n_tokens=21)

    def test_the_placement_gate_cannot_be_claimed_over_supplied_text(self):
        with pytest.raises(ShowcaseError, match="not decoded here"):
            inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin",
                             placement=True)

    def test_gate_counters_are_refused_where_no_gate_ran(self):
        with pytest.raises(ShowcaseError, match="no gate ran"):
            inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin",
                             counters=GATE_COUNTERS)

    def test_an_unknown_termination_is_refused(self):
        with pytest.raises(ShowcaseError, match="is not one of"):
            inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin",
                             termination="finished")

    def test_every_gate_reason_is_accepted_as_operator_supplied(self):
        for reason in TERMINATIONS:
            report = inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin",
                                      termination=reason)
            assert report["result"]["termination"] == reason
            assert report["provenance"]["termination"]["source"] == \
                TERM_OPERATOR

    def test_the_origin_is_recorded_verbatim(self):
        for origin in ("stdin", "file:/somewhere/b.txt", "sample-variant:tower"):
            report = inspect_supplied("x", {"2x4": 2}, TWO, origin=origin)
            assert report["provenance"]["text_origin"] == origin
            assert report["provenance"]["mode"] == MODE_SUPPLIED


class TestAnUnavailableTerminationIsNotAFailure:
    def report(self, **kw):
        return inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin", **kw)

    def test_it_is_recorded_as_unavailable_rather_than_assumed(self):
        prov = self.report()["provenance"]
        assert prov["termination"]["source"] == TERM_UNAVAILABLE
        assert "will not assume" in prov["termination"]["note"]

    def test_no_eos_token_is_added_to_a_count_that_may_not_have_one(self):
        """Assuming EOS is assuming the answer, one token at a time."""
        assert tokens_if_decoded(TWO, None) == 20
        assert tokens_if_decoded(TWO, "normal_eos") == 21
        assert tokens_if_decoded(TWO, "max_bricks") == 20
        assert self.report()["result"]["n_tokens"] == 20

    def test_the_checks_that_read_it_are_undecided_not_failed(self):
        report = self.report()
        for name in TERMINATION_DEPENDENT:
            assert report["checks"][name] is None, name
            assert report["checks_determinable"][name] is False, name
        assert passed(report) is None

    def test_the_undecided_value_is_none_in_the_report_not_only_in_print(self):
        """A JSON consumer must get the same three answers a reader gets."""
        body = json.loads(json.dumps(self.report()))
        for name in TERMINATION_DEPENDENT:
            assert body["checks"][name] is None, name
        assert body["checks"]["collision_free"] is True
        assert body["checks"]["stud_only_connected"] is True

    def test_no_check_is_ever_false_merely_for_want_of_a_termination(self):
        undecided = self.report()
        decided = self.report(termination="normal_eos")
        for name, value in undecided["checks"].items():
            if value is None:
                continue
            assert value == decided["checks"][name], name

    def test_every_other_check_is_still_decided(self):
        report = self.report()
        undecided = {n for n, v in report["checks"].items() if v is None}
        assert undecided == set(TERMINATION_DEPENDENT)
        assert {n for n, ok in report["checks_determinable"].items()
                if not ok} == undecided, "the two views must agree"

    def test_a_stated_termination_puts_them_back(self):
        report = self.report(termination="normal_eos")
        assert all(report["checks_determinable"].values())
        assert all(v is not None for v in report["checks"].values())
        assert passed(report) is True

    def test_the_three_states_are_all_reachable(self):
        assert passed(inspect_sample("tower")) is True
        assert passed(inspect_sample("collision")) is False
        assert passed(self.report()) is None

    def test_the_printed_report_says_n_a_and_what_it_means(self):
        text = format_report(self.report(), show_plan=False)
        assert "n/a   termination_accepted" in text
        assert "does not have the answer, not that the answer is no" in text


class TestAVariantIsNotTheSampleItCameFrom:
    def variant(self, **kw):
        base = dict(caption="my own brief", inventory={"2x4": 1},
                    origin="sample-variant:tower", variant_of="tower")
        return inspect_supplied(text=sample("tower").text, **{**base, **kw})

    def test_it_is_reported_as_supplied_text_not_as_the_sample(self):
        report = self.variant()
        assert report["provenance"]["mode"] == MODE_SUPPLIED
        assert report["provenance"]["variant_of"] == "tower"
        assert report["provenance"]["caption_source"] == "operator"

    def test_it_records_which_of_the_fixtures_fields_were_replaced(self):
        assert self.variant()["provenance"]["changed_from_sample"] == [
            "caption", "inventory", "termination"]

    def test_replacing_nothing_is_recorded_as_replacing_nothing(self):
        brief = sample("tower")
        report = self.variant(caption=brief.caption,
                              inventory=dict(brief.inventory),
                              termination=brief.termination)
        assert report["provenance"]["changed_from_sample"] == []
        assert report["provenance"]["mode"] == MODE_SUPPLIED, \
            "still supplied: it was the operator who said so this time"

    def test_a_plain_supplied_report_has_no_variant_fields(self):
        report = inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin")
        assert report["provenance"]["variant_of"] is None
        assert report["provenance"]["changed_from_sample"] is None


# --------------------------------------------------------------------------
# Mode 3: a decode, and everything it has to bring back
# --------------------------------------------------------------------------


class TestADecodedRecordMustStandUp:
    def test_a_measured_count_is_a_positive_whole_number(self):
        for bad in (0, -1, 1.5, "21", True, None):
            with pytest.raises(ShowcaseError, match="measured count"):
                decoded(n_tokens=bad)

    def test_the_termination_must_be_one_a_gate_can_produce(self):
        with pytest.raises(ShowcaseError, match="is not one of"):
            decoded(termination="finished")
        for reason in BrickGate.STOP_REASONS + PLACEMENT_STOP_REASONS:
            assert decoded(termination=reason).termination == reason

    def test_a_gated_decode_must_record_its_connectivity_mode(self):
        with pytest.raises(ShowcaseError, match="connectivity mode"):
            decoded(placement=True, counters=GATE_COUNTERS)
        with pytest.raises(ShowcaseError, match="connectivity mode"):
            decoded(placement=True, connectivity="sometimes",
                    counters=GATE_COUNTERS)

    def test_a_gated_decode_must_carry_the_gates_own_counters(self):
        with pytest.raises(ShowcaseError, match="missing"):
            decoded(placement=True, connectivity="off")
        with pytest.raises(ShowcaseError, match="missing"):
            decoded(placement=True, connectivity="off",
                    counters={"bricks_placed": 1})

    def test_connectivity_without_the_gate_is_refused(self):
        with pytest.raises(ShowcaseError, match="was not on"):
            decoded(connectivity="off")
        with pytest.raises(ShowcaseError, match="was not on"):
            decoded(connectivity="final_eos")

    def test_counters_without_the_gate_are_refused(self):
        """The stock gate keeps none, so these describe a different run."""
        with pytest.raises(ShowcaseError, match="placement gate was not on"):
            decoded(counters=GATE_COUNTERS)
        with pytest.raises(ShowcaseError, match="placement gate was not on"):
            decoded(counters={})

    def test_an_ungated_record_keeps_both_fields_empty(self):
        rec = decoded()
        assert rec.as_dict()["connectivity"] is None
        assert rec.as_dict()["gate_counters"] is None

    def test_a_valid_gated_record_keeps_every_field(self):
        rec = decoded(placement=True, connectivity="final_eos",
                      counters=GATE_COUNTERS)
        assert rec.as_dict()["connectivity"] == "final_eos"
        assert rec.as_dict()["gate_counters"] == GATE_COUNTERS


class TestTheDecodedReportCarriesWhatItTakesToRerun:
    def report(self, **kw):
        return inspect_decoded("a stack", {"2x4": 2}, TWO,
                               decoded=decoded(**kw))

    def test_the_token_count_and_termination_are_marked_measured(self):
        prov = self.report()["provenance"]
        assert prov["tokens"]["source"] == TOKENS_MEASURED
        assert prov["termination"]["source"] == TERM_MEASURED
        assert prov["mode"] == MODE_DECODED
        assert prov["text_origin"] == "decoder"

    def test_the_settings_are_all_there(self):
        d = self.report(model="project", adapter="runs/x", device="mps",
                        seed=7, temperature=0.9, max_bricks=40,
                        max_tokens=400)["provenance"]["decode"]
        assert d["model"] == "project" and d["adapter"] == "runs/x"
        assert d["device"] == "mps" and d["seed"] == 7
        assert d["temperature"] == 0.9
        assert d["max_bricks"] == 40 and d["max_tokens"] == 400
        assert d["placement"] is False and d["connectivity"] is None

    def test_every_setting_is_printed(self):
        text = format_report(self.report(seed=7, temperature=0.9),
                             show_plan=False)
        for fragment in ("seed 7", "temperature 0.9", "max_bricks 80",
                         "max_tokens 800", "device", "weights"):
            assert fragment in text, fragment

    def test_a_gated_decode_prints_its_connectivity_and_the_notice(self):
        report = self.report(placement=True, connectivity="final_eos",
                             counters=GATE_COUNTERS)
        assert report["placement_notice"] == PLACEMENT_NOTICE
        text = format_report(report, show_plan=False)
        assert "placement True, connectivity final_eos" in text
        assert PLACEMENT_NOTICE in text

    def test_only_a_decoded_record_may_produce_a_decoded_report(self):
        with pytest.raises(ShowcaseError, match="needs a Decoded record"):
            inspect_decoded("x", {"2x4": 2}, TWO, decoded={"n_tokens": 21})

    def test_a_count_that_disagrees_with_the_bricks_says_why(self):
        """The check a derived count makes vacuous, doing its job."""
        report = self.report(n_tokens=44)
        assert report["checks"]["parse_success"] is False
        assert report["result"]["tokens_match_complete_bricks"] is False
        note = report["result"]["token_brick_note"]
        assert "stopped inside a brick" in note
        assert "44 tokens" in note
        assert note in format_report(report, show_plan=False)

    def test_a_consistent_count_leaves_no_note(self):
        report = self.report(n_tokens=21)
        assert report["result"]["token_brick_note"] is None
        assert report["checks"]["parse_success"] is True


class TestModesAreMutuallyExclusive:
    def test_there_are_exactly_three_and_every_report_names_one(self):
        assert MODES == (MODE_SAMPLE, MODE_SUPPLIED, MODE_DECODED)
        seen = {
            inspect_sample("tower")["provenance"]["mode"],
            inspect_supplied("x", {"2x4": 2}, TWO,
                             origin="stdin")["provenance"]["mode"],
            inspect_decoded("x", {"2x4": 2}, TWO,
                            decoded=decoded())["provenance"]["mode"],
        }
        assert seen == set(MODES)

    def test_a_measured_token_count_appears_in_exactly_one_mode(self):
        sources = {
            MODE_SAMPLE: inspect_sample("tower"),
            MODE_SUPPLIED: inspect_supplied("x", {"2x4": 2}, TWO,
                                            origin="stdin"),
            MODE_DECODED: inspect_decoded("x", {"2x4": 2}, TWO,
                                          decoded=decoded()),
        }
        measured = [m for m, r in sources.items()
                    if r["provenance"]["tokens"]["source"] == TOKENS_MEASURED]
        assert measured == [MODE_DECODED]

    def test_the_placement_gate_appears_in_exactly_one_mode(self):
        assert inspect_sample("tower")["request"]["placement_gate"] is False
        assert inspect_supplied("x", {"2x4": 2}, TWO,
                                origin="stdin")["request"]["placement_gate"] \
            is False
        gated = inspect_decoded("x", {"2x4": 2}, TWO,
                                decoded=decoded(placement=True,
                                                connectivity="off",
                                                counters=GATE_COUNTERS))
        assert gated["request"]["placement_gate"] is True


class TestTheReportSaysWhatItIsNot:
    def test_every_report_carries_the_standing_notice(self):
        for report in (inspect_sample("tower"),
                       inspect_supplied("x", {"2x4": 2}, TWO, origin="stdin"),
                       inspect_decoded("x", {"2x4": 2}, TWO,
                                       decoded=decoded())):
            assert report["notice"] == STANDING_NOTICE
            assert STANDING_NOTICE in format_report(report, show_plan=False)

    def test_the_placement_notice_appears_only_when_the_gate_was_on(self):
        assert inspect_sample("tower")["placement_notice"] is None
        assert PLACEMENT_NOTICE not in format_report(inspect_sample("tower"))

    def test_the_placement_notice_states_the_one_evaluation_and_its_direction(
            self):
        """gen09 measured this gate once and the direction was against it.

        The notice used to say the gate had "never been formally evaluated"
        and that "Phase 3C is not authorised". Both were true when written
        and neither has been true since round 65: gen09 ran 1,920 cells and
        scored them. A notice that still said "never evaluated" would be the
        product asserting something the project's own frozen results refute,
        so this test now pins the replacement -- the one evaluation, its
        direction, and the fact that it is still not evidence of improvement.
        """
        low = PLACEMENT_NOTICE.lower()
        assert "never been formally evaluated" not in low
        assert "not authorised" not in low
        assert "gen09" in low
        assert "-5.0pp" in low
        assert "[-9.4, -0.6]" in low
        assert "not evidence" in low

    def test_the_report_names_the_scorer_it_used(self):
        report = inspect_sample("tower")
        assert report["scored_by"] == "src.eval.scoring.score_generation"
        assert "score_generation" in format_report(report, show_plan=False)

    def test_the_prompt_is_shown_only_when_asked(self):
        report = inspect_sample("tower")
        assert "### Available Parts" not in format_report(report)
        assert "### Available Parts" in format_report(report,
                                                      show_prompt=True)

    def test_a_failing_report_prints_fail_not_a_silent_pass(self):
        text = format_report(inspect_sample("collision"))
        assert "FAIL  collision_free" in text
        assert "colliding brick pairs" in text

    def test_the_overdraw_is_flagged_in_the_inventory_panel(self):
        text = format_report(inspect_sample("overdrawn"))
        assert "OVERDRAWN" in text and "left   -1" in text

    def test_the_provenance_panel_is_printed_first(self):
        text = format_report(inspect_sample("tower"))
        assert text.index("-- provenance") < text.index("-- result")
        assert text.index("-- provenance") < text.index("-- checks")


class TestLdrawExport:
    def test_a_legal_structure_is_written(self, tmp_path):
        out = write_ldraw(inspect_sample("tower"), tmp_path / "t.ldr")
        body = out.read_text(encoding="utf-8")
        assert body.count("0 STEP") == 3
        assert body.startswith("1 115 ")

    def test_a_structure_with_no_bricks_is_refused(self, tmp_path):
        report = inspect_supplied("x", {"1x1": 1}, "not a brick line\n",
                                  origin="stdin")
        with pytest.raises(ShowcaseError, match="does not serialise"):
            write_ldraw(report, tmp_path / "t.ldr")
        assert not (tmp_path / "t.ldr").exists()


# --------------------------------------------------------------------------
# The model path, wired but never run here
# --------------------------------------------------------------------------


class TestTheModelPathStaysOutOfTheWay:
    def test_importing_the_module_pulls_in_no_model_machinery(self):
        text = (ROOT / "src" / "demo" / "showcase.py").read_text(
            encoding="utf-8")
        head = text.split("# ---", 1)[0]
        for banned in ("import torch", "import transformers", "from peft"):
            assert banned not in head, banned

    def test_an_unknown_model_name_is_refused_before_anything_loads(self):
        with pytest.raises(ShowcaseError, match="model="):
            showcase.generate("x", {"1x1": 1}, model="whatever")

    def test_an_unknown_connectivity_mode_is_refused(self):
        with pytest.raises(ShowcaseError, match="connectivity="):
            showcase.generate("x", {"1x1": 1}, placement=True,
                              connectivity="sometimes")

    def test_connectivity_without_the_gate_is_refused_before_loading(self):
        with pytest.raises(ShowcaseError, match="was not asked for"):
            showcase.generate("x", {"1x1": 1}, connectivity="final_eos")

    def test_a_missing_pointer_says_where_it_would_have_come_from(self, tmp_path):
        with pytest.raises(ShowcaseError, match="not published") as exc:
            showcase.project_adapter_dir(root=tmp_path)
        assert "24_project_model.py" in str(exc.value)

    def test_a_pointer_naming_an_absent_adapter_is_refused(self, tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "project_model.json").write_text(
            json.dumps({"adapter": {"path": "runs/nowhere"}}),
            encoding="utf-8")
        with pytest.raises(ShowcaseError, match="which is not here"):
            showcase.project_adapter_dir(root=tmp_path)

    def test_only_generate_builds_a_decoded_record(self):
        """Everything else would be claiming a decode it did not run."""
        for rel in ("src/demo/showcase.py", "scripts/26_showcase.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            uses = [ln for ln in text.splitlines()
                    if "Decoded(" in ln and "class Decoded" not in ln
                    and "isinstance" not in ln]
            assert len(uses) <= 1, f"{rel}: {uses}"


class TestTheModelPathIsWiredCorrectly:
    """The one mode that loads weights, checked without any.

    Every loader and both decode entry points are replaced, so this asserts
    the *wiring* -- which loader runs for which model, which entry point runs
    for which gate setting, and what reaches the report -- on a machine with
    no checkpoint. It does not assert that decoding works; nothing here can.
    """

    class FakeWeights:
        def to(self, device):
            return self

        def eval(self):
            return self

    def inventory_placement_gate(self):
        from src.constraints.placement_decode import InventoryPlacementGate
        from src.inventory.engine import Inventory

        return InventoryPlacementGate(
            stub_slots(), Inventory.from_parts({"2x4": 2}), enabled=True)

    def wire(self, monkeypatch, *, text=TWO, n_tokens=21,
             termination="normal_eos", gate=None, stock_gate=None):
        import src.constraints.inventory_decode as inv_mod
        import src.constraints.placement_decode as place_mod
        import src.generation.brickgpt as gpt_mod
        import src.training.lora as lora_mod
        from src.inventory.engine import Inventory
        from src.generation.brickgpt import RawGeneration

        seen = {"loaders": [], "entry": None, "kw": None, "device": None}
        raw = RawGeneration(text=text, n_tokens=n_tokens, seconds=0.5,
                            truncated=False, termination=termination)
        if stock_gate is None:
            stock_gate = InventoryGate(stub_slots(),
                                       Inventory.from_parts({"2x4": 2}))
        gate = gate if gate is not None else self.inventory_placement_gate()

        def loader(name):
            def call(*a, **kw):
                seen["loaders"].append((name, a, kw))
                return self.FakeWeights(), {}
            return call

        monkeypatch.setattr(gpt_mod, "load_tokenizer",
                            lambda *a, **kw: "TOKENIZER-STANDIN")
        monkeypatch.setattr(lora_mod, "load_merged_brickgpt", loader("merged"))
        monkeypatch.setattr(lora_mod, "load_finetuned", loader("finetuned"))
        monkeypatch.setattr(gpt_mod.BrickGPT, "from_loaded",
                            classmethod(lambda cls, model, tok, *, device:
                                        seen.__setitem__("device", device)
                                        or "GPT-STANDIN"))

        def entry(name, module, returns):
            def call(gpt, caption, *a, **kw):
                seen["entry"] = name
                seen["kw"] = {"caption": caption, "args": a, **kw}
                return raw, returns
            monkeypatch.setattr(module, f"generate_raw_with_{name}", call)

        entry("inventory", inv_mod, stock_gate)
        entry("placement", place_mod, gate)
        return seen

    def test_the_published_model_goes_through_the_merged_loader(self, monkeypatch):
        seen = self.wire(monkeypatch)
        showcase.generate("a stack", {"2x4": 2}, model="published",
                          device="cpu")
        assert [n for n, _, _ in seen["loaders"]] == ["merged"]
        assert seen["device"] == "cpu"

    def test_the_project_model_goes_through_the_finetuned_loader(
            self, monkeypatch, tmp_path):
        seen = self.wire(monkeypatch)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        monkeypatch.setattr(showcase, "project_adapter_dir",
                            lambda root=None: adapter)
        report = showcase.generate("a stack", {"2x4": 2}, model="project",
                                   device="cpu")
        assert [n for n, _, _ in seen["loaders"]] == ["finetuned"]
        _, args, kw = seen["loaders"][0]
        assert args[0] == adapter
        assert kw["verify_digest"] is True, "the manifest is checked, not trusted"
        assert report["provenance"]["decode"]["adapter"] == str(adapter)

    def test_without_the_gate_it_takes_the_stock_entry_point(self, monkeypatch):
        seen = self.wire(monkeypatch)
        report = showcase.generate("a stack", {"2x4": 2}, model="published",
                                   device="cpu")
        assert seen["entry"] == "inventory"
        assert report["provenance"]["decode"]["gate_counters"] is None
        assert report["provenance"]["decode"]["connectivity"] is None

    def test_with_the_gate_it_takes_the_opt_in_entry_point(self, monkeypatch):
        seen = self.wire(monkeypatch)
        report = showcase.generate("a stack", {"2x4": 2}, model="published",
                                   device="cpu", placement=True,
                                   connectivity="final_eos")
        assert seen["entry"] == "placement"
        assert seen["kw"]["enabled"] is True
        assert seen["kw"]["connectivity"] == "final_eos"
        assert report["provenance"]["decode"]["connectivity"] == "final_eos"

    def test_the_counters_come_from_the_gate_that_ran(self, monkeypatch):
        self.wire(monkeypatch)
        report = showcase.generate("a stack", {"2x4": 2}, model="published",
                                   device="cpu", placement=True)
        counters = report["provenance"]["decode"]["gate_counters"]
        assert counters["enabled"] is True
        assert counters["connectivity"] == "off"
        assert "candidates_masked" in counters and "eos_deferrals" in counters
        assert counters["inventory_opening"] == {"2x4": 2}

    def test_the_placement_path_demands_an_inventory_placement_gate(
            self, monkeypatch):
        """A wiring mistake must not publish a claim about a gate."""
        self.wire(monkeypatch, gate=object())
        with pytest.raises(ShowcaseError, match="InventoryPlacementGate"):
            showcase.generate("a stack", {"2x4": 2}, model="published",
                              device="cpu", placement=True)

    def test_a_bare_placement_gate_means_stock_was_never_enforced(
            self, monkeypatch):
        """It gates collision and nothing else; the stock would be free."""
        from src.constraints.placement_decode import PlacementGate

        bare = PlacementGate(stub_slots(), enabled=True)
        assert not isinstance(bare, InventoryGate), "the premise of the check"
        self.wire(monkeypatch, gate=bare)
        with pytest.raises(ShowcaseError, match="stock was never enforced"):
            showcase.generate("a stack", {"2x4": 2}, model="published",
                              device="cpu", placement=True)

    def test_the_stock_path_demands_an_inventory_gate(self, monkeypatch):
        self.wire(monkeypatch, stock_gate=object())
        with pytest.raises(ShowcaseError, match="without an InventoryGate"):
            showcase.generate("a stack", {"2x4": 2}, model="published",
                              device="cpu")

    def test_the_stock_path_refuses_a_placement_gate_it_did_not_ask_for(
            self, monkeypatch):
        """The report would say off while a placement gate was what ran."""
        gate = self.inventory_placement_gate()
        self.wire(monkeypatch, stock_gate=gate)
        with pytest.raises(ShowcaseError, match="would say the placement gate "
                                                "was off"):
            showcase.generate("a stack", {"2x4": 2}, model="published",
                              device="cpu")

    def test_the_stock_path_accepts_the_gate_it_asked_for(self, monkeypatch):
        from src.inventory.engine import Inventory

        gate = InventoryGate(stub_slots(), Inventory.from_parts({"2x4": 2}))
        self.wire(monkeypatch, stock_gate=gate)
        report = showcase.generate("a stack", {"2x4": 2}, model="published",
                                   device="cpu")
        assert report["provenance"]["decode"]["placement"] is False

    def test_the_decoded_report_carries_the_measured_token_count(self, monkeypatch):
        self.wire(monkeypatch, n_tokens=21)
        report = showcase.generate("a stack", {"2x4": 2}, model="published",
                                   device="cpu")
        assert report["provenance"]["tokens"]["source"] == TOKENS_MEASURED
        assert report["result"]["n_tokens"] == 21

    def test_the_termination_the_decoder_reported_is_the_one_scored(
            self, monkeypatch):
        self.wire(monkeypatch, termination="inventory_exhausted")
        report = showcase.generate("a stack", {"2x4": 2}, model="published",
                                   device="cpu")
        assert report["result"]["termination"] == "inventory_exhausted"
        assert report["checks"]["termination_accepted"] is True
        assert report["provenance"]["termination"]["source"] == TERM_MEASURED


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


class TestTheCommandLine:
    def test_listing_the_briefs_needs_nothing(self, cli, capsys):
        assert cli.main(["--list"]) == 0
        out = capsys.readouterr().out
        for name in SAMPLES:
            assert name in out
        assert "hand-written fixtures" in out
        assert "stated by the fixture, not measured" in out
        assert "--variant-of" in out

    def test_a_passing_brief_exits_zero(self, cli, capsys):
        assert cli.main(["--sample", "tower"]) == cli.EXIT_OK
        assert "mode        : sample" in capsys.readouterr().out

    def test_a_failing_brief_exits_one(self, cli, capsys):
        assert cli.main(["--sample", "collision"]) == cli.EXIT_CHECK_FAILED
        assert "FAIL" in capsys.readouterr().out

    def test_an_undecided_report_exits_three(self, cli, capsys, tmp_path):
        src = tmp_path / "b.txt"
        src.write_text(TWO, encoding="utf-8")
        code = cli.main(["--bricks", str(src), "--caption", "a stack",
                         "--inventory", "2x4:2"])
        assert code == cli.EXIT_UNDECIDED, "no termination, so no verdict"
        assert "n/a" in capsys.readouterr().out

    def test_a_stated_termination_gives_a_verdict(self, cli, capsys, tmp_path):
        src = tmp_path / "b.txt"
        src.write_text(TWO, encoding="utf-8")
        assert cli.main(["--bricks", str(src), "--caption", "a stack",
                         "--inventory", "2x4:2",
                         "--termination", "normal_eos"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "operator-supplied" in out

    def test_json_mode_round_trips_with_the_provenance(self, cli, capsys):
        assert cli.main(["--sample", "tower", "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["kind"] == "brickagain.showcase"
        assert body["provenance"]["mode"] == MODE_SAMPLE
        assert body["notice"] == STANDING_NOTICE

    def test_supplied_brick_text_runs_end_to_end(self, cli, capsys, tmp_path):
        src = tmp_path / "b.txt"
        src.write_text(TWO, encoding="utf-8")
        code = cli.main(["--bricks", str(src), "--caption", "a stack",
                         "--inventory", "2x4:2", "--termination", "normal_eos",
                         "--ldr", str(tmp_path / "out.ldr")])
        assert code == cli.EXIT_OK
        assert (tmp_path / "out.ldr").is_file()
        out = capsys.readouterr().out
        assert "LDraw written to" in out
        assert f"file:{src}" in out

    def test_stdin_is_recorded_as_stdin(self, cli, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(TWO))
        cli.main(["--bricks", "-", "--caption", "a stack",
                  "--inventory", "2x4:2", "--termination", "normal_eos"])
        assert "brick text  : stdin" in capsys.readouterr().out

    def test_a_variant_of_a_brief_is_labelled_as_one(self, cli, capsys):
        code = cli.main(["--variant-of", "tower", "--caption", "mine",
                         "--inventory", "2x4:1", "--json"])
        assert code == cli.EXIT_UNDECIDED
        body = json.loads(capsys.readouterr().out)
        assert body["provenance"]["mode"] == MODE_SUPPLIED
        assert body["provenance"]["variant_of"] == "tower"
        assert body["provenance"]["text_origin"] == "sample-variant:tower"

    def test_a_variant_needs_its_own_caption_and_inventory(self, cli, capsys):
        assert cli.main(["--variant-of", "tower"]) == cli.EXIT_REFUSED
        err = capsys.readouterr().err
        assert "--caption" in err and "--inventory" in err

    def test_a_bad_inventory_is_refused_with_the_help_text(self, cli, capsys):
        assert cli.main(["--bricks", "-", "--caption", "x",
                         "--inventory", "2x8:1"]) == cli.EXIT_REFUSED
        assert "not one of the eight" in capsys.readouterr().err

    def test_a_missing_file_is_refused(self, cli, capsys, tmp_path):
        assert cli.main(["--bricks", str(tmp_path / "nope.txt"),
                         "--caption", "x",
                         "--inventory", "1x1:1"]) == cli.EXIT_REFUSED
        assert "is not a file" in capsys.readouterr().err

    def test_an_empty_bricks_argument_stays_in_supplied_mode(
            self, cli, capsys, monkeypatch):
        def model_must_not_load(*args, **kwargs):
            raise AssertionError("empty --bricks was misclassified as --generate")

        monkeypatch.setattr(showcase, "generate", model_must_not_load)
        code = cli.main([
            "--bricks", "", "--caption", "x", "--inventory", "1x1:1"])
        assert code == cli.EXIT_REFUSED
        assert "is not a file" in capsys.readouterr().err

    def test_the_modes_are_mutually_exclusive(self, cli):
        for pair in (["--sample", "tower", "--generate"],
                     ["--sample", "tower", "--variant-of", "tower"],
                     ["--bricks", "-", "--generate"],
                     ["--variant-of", "tower", "--bricks", "-"]):
            with pytest.raises(SystemExit):
                cli.main(pair)

    def test_a_mode_is_required(self, cli):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_the_help_lists_the_stored_briefs(self, cli, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        assert "stored briefs" in out
        for name in SAMPLES:
            assert name in out


class TestFlagsThatDoNotApplyAreRefused:
    """Silently ignoring one produces a report that looks like it obeyed."""

    def refuse(self, cli, capsys, argv) -> str:
        assert cli.main(argv) == cli.EXIT_REFUSED, argv
        return capsys.readouterr().err

    def test_a_sample_takes_no_brief_of_its_own(self, cli, capsys):
        for flag, value in (("--caption", "x"), ("--inventory", "2x4:1"),
                            ("--termination", "normal_eos")):
            err = self.refuse(cli, capsys, ["--sample", "tower", flag, value])
            assert flag in err
            assert "used whole" in err
            assert "--variant-of tower" in err

    def test_a_sample_takes_no_decode_settings(self, cli, capsys):
        for argv in (["--seed", "3"], ["--temperature", "0.9"],
                     ["--device", "cpu"], ["--model", "published"],
                     ["--max-bricks", "10"], ["--max-tokens", "100"],
                     ["--placement"], ["--connectivity", "final_eos"]):
            err = self.refuse(cli, capsys, ["--sample", "tower", *argv])
            assert argv[0] in err
            assert "describe a decode" in err

    def test_supplied_text_takes_no_decode_settings(self, cli, capsys):
        for mode in (["--bricks", "-"], ["--variant-of", "tower"]):
            err = self.refuse(cli, capsys,
                              [*mode, "--caption", "x", "--inventory",
                               "2x4:1", "--placement"])
            assert "--placement" in err
            assert "describe a decode" in err

    def test_a_decode_states_neither_its_tokens_nor_its_termination(
            self, cli, capsys):
        err = self.refuse(cli, capsys,
                          ["--generate", "--caption", "x", "--inventory",
                           "2x4:1", "--termination", "normal_eos"])
        assert "--termination" in err
        assert "measures its own" in err

    def test_every_refusal_names_the_flag_it_refused(self, cli, capsys):
        err = self.refuse(cli, capsys,
                          ["--sample", "tower", "--seed", "1",
                           "--temperature", "0.5"])
        assert "--seed" in err and "--temperature" in err

    def test_the_output_flags_apply_everywhere(self, cli, capsys, tmp_path):
        assert cli.main(["--sample", "tower", "--no-plan", "--prompt",
                         "--ldr", str(tmp_path / "a.ldr")]) == 0
        assert cli.main(["--sample", "tower", "--json",
                         "--ldr", str(tmp_path / "b.ldr")]) == 0

    def test_list_refuses_every_kind_of_extra_flag(self, cli, capsys,
                                                    tmp_path):
        for extra in (["--seed", "7"], ["--json"],
                      ["--ldr", str(tmp_path / "a.ldr")],
                      ["--preview", str(tmp_path / "a.png")]):
            assert cli.main(["--list", *extra]) == cli.EXIT_REFUSED
            err = capsys.readouterr().err
            assert extra[0] in err and "will not be ignored" in err

    def test_list_does_not_confuse_numeric_zero_with_false(self, cli,
                                                            capsys):
        for flag in ("--seed", "--temperature", "--max-bricks",
                     "--max-tokens"):
            assert cli.main(["--list", flag, "0"]) == cli.EXIT_REFUSED
            err = capsys.readouterr().err
            assert flag in err and "will not be ignored" in err

    def test_ldraw_and_preview_must_not_share_an_output_path(
            self, cli, capsys, tmp_path):
        target = tmp_path / "same-output"
        code = cli.main(["--sample", "tower", "--ldr", str(target),
                         "--preview", str(target)])
        assert code == cli.EXIT_REFUSED
        assert not target.exists()
        err = capsys.readouterr().err
        assert "--ldr" in err and "--preview" in err
        assert "same output path" in err

    def test_an_invalid_preview_suffix_is_refused_before_ldraw_is_written(
            self, cli, capsys, tmp_path):
        ldr = tmp_path / "must-not-exist.ldr"
        preview = tmp_path / "bad.jpg"
        code = cli.main(["--sample", "tower", "--ldr", str(ldr),
                         "--preview", str(preview)])
        assert code == cli.EXIT_REFUSED
        assert not ldr.exists() and not preview.exists()
        assert ".png or .svg" in capsys.readouterr().err

    def test_a_sample_can_write_the_cpu_preview(self, cli, capsys, tmp_path):
        out = tmp_path / "tower.png"
        assert cli.main(["--sample", "tower", "--preview", str(out)]) == 0
        assert out.read_bytes().startswith(b"\x89PNG")
        assert "3-D preview written" in capsys.readouterr().out

    def test_unparsed_text_is_refused_before_previewing(self, cli, capsys,
                                                        tmp_path):
        source = tmp_path / "broken.txt"
        source.write_text("this is not a brick\n", encoding="utf-8")
        code = cli.main([
            "--bricks", str(source), "--caption", "x",
            "--inventory", "1x1:1", "--preview", str(tmp_path / "x.png")])
        assert code == cli.EXIT_REFUSED
        assert "refuses unparsed brick lines" in capsys.readouterr().err

    def test_a_flag_that_would_change_nothing_is_refused_too(self, cli,
                                                             capsys):
        """Read and ignored leaves a report looking like it obeyed."""
        for flag in ("--prompt", "--no-plan"):
            err = self.refuse(cli, capsys, ["--sample", "tower", "--json",
                                            flag])
            assert flag in err
            assert "--json prints none" in err

    def test_connectivity_without_the_gate_is_refused(self, cli, capsys):
        err = self.refuse(cli, capsys,
                          ["--generate", "--caption", "x", "--inventory",
                           "2x4:1", "--connectivity", "final_eos"])
        assert "--connectivity" in err
        assert "would change nothing" in err

    def test_connectivity_off_without_the_gate_is_refused_as_well(self, cli,
                                                                  capsys):
        """It parses and matches the default, and still changes nothing."""
        err = self.refuse(cli, capsys,
                          ["--generate", "--caption", "x", "--inventory",
                           "2x4:1", "--connectivity", "off"])
        assert "would change nothing" in err

    def test_the_allowed_sets_cover_every_optional_flag(self, cli):
        """A new flag has to be placed, not left to fall through."""
        parser = cli.build_parser()
        optional = {a.dest for a in parser._actions
                    if a.dest not in ("help", "sample", "variant_of",
                                      "bricks", "generate")}
        placed = set().union(*cli.ALLOWED.values()) | cli.OUTPUT_FLAGS
        assert optional - placed == set(), optional - placed


class TestTheCommandLineReachesGenerate:
    def test_it_passes_exactly_what_it_was_given(self, cli, monkeypatch,
                                                 capsys):
        seen = {}
        monkeypatch.setattr(
            showcase, "generate",
            lambda caption, inventory, **kw: seen.update(
                caption=caption, inventory=inventory, **kw)
            or inspect_decoded(caption, inventory, TWO,
                               decoded=decoded(
                                   placement=kw["placement"],
                                   connectivity=(kw["connectivity"]
                                                 if kw["placement"] else None),
                                   counters=(GATE_COUNTERS if kw["placement"]
                                             else None))))
        assert cli.main(["--generate", "--caption", "a chair", "--inventory",
                         "2x4:3", "--placement", "--connectivity", "final_eos",
                         "--model", "published", "--seed", "7",
                         "--temperature", "0.9", "--max-bricks", "12",
                         "--max-tokens", "120"]) == cli.EXIT_OK
        assert seen == {"caption": "a chair", "inventory": {"2x4": 3},
                        "model": "published", "device": "mps", "seed": 7,
                        "temperature": 0.9, "max_bricks": 12,
                        "max_tokens": 120, "placement": True,
                        "connectivity": "final_eos"}
        assert PLACEMENT_NOTICE in capsys.readouterr().out

    def test_the_defaults_are_applied_only_where_nothing_was_given(
            self, cli, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(
            showcase, "generate",
            lambda caption, inventory, **kw: seen.update(kw)
            or inspect_decoded(caption, inventory, TWO, decoded=decoded()))
        cli.main(["--generate", "--caption", "x", "--inventory", "2x4:1"])
        assert seen == cli.DECODE_DEFAULTS

    def test_a_decode_needs_a_caption_and_an_inventory(self, cli, capsys):
        assert cli.main(["--generate"]) == cli.EXIT_REFUSED
        err = capsys.readouterr().err
        assert "--caption" in err and "--inventory" in err


# --------------------------------------------------------------------------
# What none of it may claim
# --------------------------------------------------------------------------


#: Sentences that asserted the placement gate had no evaluation at all.
#: Assembled from fragments so this module scanning itself is not a hit, and
#: given in both languages because the interface is Chinese and the command
#: line is English -- a scan written only in English let the Chinese ones
#: survive in ``src/ui/render.py``, ``UI.md`` and ``VISION.md``.
STALE_GATE_CLAIMS: tuple[str, ...] = (
    "never been formally " "evaluated",
    "never formally " "evaluated",
    "phase 3c is not " "authorised",
    "not authorised and not " "run",
    "no metric has ever been " "computed",
    "從未" "正式評估",
    "從未經過" "正式評估",
    "從未有過" "正式指標",
    # Named, not the bare 未獲授權: that phrase is also how this file records
    # "this round was not authorised to edit that file", which is a different
    # subject and must not be swept up.
    #
    # One entry, not two. The spaced and unspaced spellings were both listed
    # at first, which was dead weight: :func:`_has` compares a
    # whitespace-stripped form as well, so this single entry already matches
    # Phase 3C未獲授權. The only thing the second entry added was the same hit
    # reported twice. :func:`test_no_two_stale_claims_normalise_to_the_same`
    # keeps that from coming back.
    "phase 3c " "未獲授權",
)

#: Every surface that says anything about the gate's evaluation status.
GATE_SURFACES: tuple[str, ...] = (
    "README.md", "SHOWCASE.md", "UI.md", "DELIVERY.md", "VISION.md",
    "src/demo/showcase.py", "src/ui/model_entry.py", "src/ui/render.py",
    "src/ui/full.py", "src/delivery/evidence.py",
    "scripts/26_showcase.py", "scripts/27_delivery.py",
    "scripts/35_full_ui.py",
)

#: A statement about the gate's evaluation has to carry all three. Each entry
#: is the set of spellings that count, because the documents use typographic
#: minus and the code uses ASCII.
GEN09_MARKERS: tuple[tuple[str, ...], ...] = (
    ("gen09",),
    ("-5.0pp", "−5.0pp"),
    ("[-9.4, -0.6]", "[−9.4, −0.6]"),
)

#: A block is judged when it makes the *affirmative* claim -- "this gate was
#: evaluated" -- in either language. Deliberately not a broad "mentions the
#: gate and the word evaluation": a paragraph saying an interface "never
#: enables the placement gate, reads no frozen evaluation case, and produces
#: no metric" is describing that interface's scope, not ruling on the gate's
#: history, and requiring gen09's numbers there would be noise. The negative
#: claim returning is caught by :data:`STALE_GATE_CLAIMS` instead.
_EVALUATED_CLAIMS: tuple[str, ...] = (
    "evaluated once", "evaluated exactly once", "was evaluated",
    "正式評估過一次", "唯一一次正式指標", "唯一的正式指標",
    "已於第六十五輪執行完成", "只被正式評估過一次",
)


def _blocks(text: str) -> list[str]:
    """Paragraphs, as a reader meets them: split on the blank line."""
    return [" ".join(b.split()) for b in text.split("\n\n") if b.strip()]


def _forms(blob: str) -> tuple[str, str]:
    """Space-joined and space-stripped, both lowered.

    Chinese prose has no spaces, so a line break inside a phrase becomes a
    space when the block is normalised and the phrase stops matching -- which
    is how ``DELIVERY.md``'s wrapped 已於第/六十五輪執行完成 first slipped
    through this check. Matching against both forms costs nothing.
    """
    low = blob.lower()
    return " ".join(low.split()), "".join(low.split())


def _has(blob: str, needle: str) -> bool:
    joined, stripped = _forms(blob)
    n = needle.lower()
    return n in joined or "".join(n.split()) in stripped


def _missing_markers(blob: str) -> list[tuple[str, ...]]:
    return [m for m in GEN09_MARKERS if not any(_has(blob, s) for s in m)]


def _stale_hits(blob: str) -> list[str]:
    """Stale claims in ``blob``, matched the way :func:`_has` matches.

    Every stale check goes through this, because the first version of this
    module wrote the whole-surface scan as a plain ``stale in flat`` while
    only the marker side used :func:`_has`. The result was a guard that
    caught 該閘門從未經過正式評估 on one line and missed the identical
    sentence wrapped across two -- a false negative in the check that was
    added to stop exactly that class of miss.
    """
    return [s for s in STALE_GATE_CLAIMS if _has(blob, s)]


class TestTheGateStatementIsCheckedPerPlace:
    """One check per notice, per paragraph and per emitted string.

    The whole-file version of this rule is not enough and the reason is on
    record: round 74 corrected ``PORTFOLIO.md`` line 111 while line 484 of the
    same file already carried the right digest, so a file-wide ``in`` passed
    over a wrong value for three rounds. The same shape applies here -- a
    document that states the gen09 result in one paragraph would let a second
    paragraph go on denying it. So each place is judged on its own text.
    """

    def test_no_surface_anywhere_still_denies_the_evaluation(self):
        """Both languages, every surface, reported per file and per phrase."""
        found: list[str] = []
        for rel in GATE_SURFACES:
            path = ROOT / rel
            assert path.is_file(), rel
            text = path.read_text(encoding="utf-8")
            for stale in _stale_hits(text):
                found.append(f"{rel}: {stale!r}")
        assert found == [], found

    @pytest.mark.parametrize("blob,why", [
        ("該閘門從未經過正式評估", "same line"),
        ("該閘門從未經過\n正式評估", "wrapped across two lines"),
        ("這個閘門從未經過\n  正式評估，開啟它不是證據",
         "wrapped and indented, as a docstring would be"),
        ("Phase 3C 未獲授權，開啟它不能作為改善的證據",
         "the named authorisation claim"),
        ("Phase 3C\n未獲授權", "the named claim, wrapped"),
        ("This has never been formally\nevaluated.", "English, wrapped"),
    ])
    def test_the_scan_catches_a_stale_claim_however_it_is_wrapped(
            self, blob, why):
        """Line breaks must not hide a stale sentence. Both languages.

        Written because the first version of this scan compared with a plain
        ``in`` against space-joined text: Chinese prose has no spaces, so a
        sentence wrapped mid-phrase became 從未經過 正式評估 and stopped
        matching. The guard passed while the claim was still on the page.
        """
        assert _stale_hits(blob), f"not caught ({why}): {blob!r}"

    def test_no_two_stale_claims_normalise_to_the_same(self):
        """The table's size is the number of rules it actually has.

        :func:`_has` matches a whitespace-stripped form, so two entries that
        differ only in spacing are one rule wearing two hats: they cannot
        catch anything the other misses, and a hit gets reported twice. The
        table briefly held ``phase 3c 未獲授權`` and ``phase 3c未獲授權`` for
        that reason.
        """
        seen: dict[str, list[str]] = {}
        for claim in STALE_GATE_CLAIMS:
            seen.setdefault("".join(claim.lower().split()), []).append(claim)
        duplicates = {k: v for k, v in seen.items() if len(v) > 1}
        assert duplicates == {}, duplicates
        assert len(seen) == len(STALE_GATE_CLAIMS)

    def test_the_scan_does_not_sweep_up_the_unrelated_authorisation_phrase(
            self):
        """本輪未獲授權更動 is about edit permission, not the gate.

        The stale entry is the named ``Phase 3C 未獲授權`` for this reason:
        a bare 未獲授權 would fail this file's own records.
        """
        assert _stale_hits("`src/generation/brickgpt.py` 本輪未獲授權更動，"
                           "因此沒有改") == []

    @pytest.mark.parametrize("where", ("src.demo.showcase",
                                       "src.ui.model_entry"))
    def test_each_placement_notice_carries_the_result_by_itself(self, where):
        """The notice a person actually reads, judged on its own words.

        Not the module it lives in: a notice that said only "this gate was
        evaluated" would pass a file-wide scan because the module docstring
        above it carries the numbers.
        """
        import importlib

        notice = importlib.import_module(where).PLACEMENT_NOTICE
        assert _stale_hits(notice) == [], (where, _stale_hits(notice))
        assert _missing_markers(notice) == [], (
            f"{where}: the notice omits {_missing_markers(notice)}")

    @pytest.mark.parametrize(
        "rel", [r for r in GATE_SURFACES if r.endswith(".md")])
    def test_each_paragraph_that_rules_on_the_gate_carries_the_numbers(
            self, rel):
        """A paragraph that says the gate was measured says what it measured.

        This is the check that cannot be satisfied from elsewhere in the file:
        the markers have to be inside the same block as the claim. A document
        that states gen09's result in one paragraph and asserts a bare "it was
        evaluated" in another fails here, which is the shape that let a wrong
        digest live in ``PORTFOLIO.md`` for three rounds.
        """
        path = ROOT / rel
        assert path.is_file(), rel
        claims, offenders = 0, []
        for block in _blocks(path.read_text(encoding="utf-8")):
            if not any(_has(block, c) for c in _EVALUATED_CLAIMS):
                continue
            claims += 1
            missing = _missing_markers(block)
            if missing:
                offenders.append((missing, block[:200]))
        assert offenders == [], offenders
        if rel.endswith(".md"):
            assert claims, f"{rel} no longer states the gate was evaluated"

    # The delivery summary's own emitted value is checked where its sealed
    # fixture lives: tests/test_delivery_evidence.py.

    @pytest.mark.parametrize("rel", ("src/ui/full.py",
                                     "scripts/27_delivery.py"))
    def test_the_machine_readable_status_field_names_the_evaluation(self, rel):
        """``method.phase_3c`` in the emitted payloads.

        A short status field, so it is not required to repeat the interval --
        the prose surfaces carry that. What it may not do is keep asserting
        the gate was never authorised and never run, which is what it said
        until round 74 and is what a consumer of the payload would read.
        """
        text = (ROOT / rel).read_text(encoding="utf-8")
        values = re.findall(r'"phase_3c":\s*"([^"]*)"', text)
        assert values, f"{rel} no longer emits a phase_3c field"
        for value in values:
            assert _stale_hits(value) == [], (rel, value)
            assert "gen09" in value.lower(), (rel, value)

    def test_the_two_page_interface_says_it_in_the_rendered_html(self):
        """What a person sees in the browser, rendered for real."""
        from src.ui.render import render_page_one

        html = render_page_one(csrf_token="t" * 24)
        low = " ".join(html.split()).lower()
        assert "placement gate" in low
        assert _stale_hits(html) == [], _stale_hits(html)
        assert _missing_markers(html) == [], _missing_markers(html)

    def test_the_command_line_help_says_it_where_the_flag_is_described(
            self, cli):
        """``--connectivity`` help text, taken from the parser itself."""
        parser = cli.build_parser()
        # The flag that turns the gate on is where the claim belongs.
        # ``--connectivity`` only selects a mode and makes no claim.
        texts = [a.help or "" for a in parser._actions
                 if (a.dest or "") == "placement"]
        assert texts, "no --placement action found"
        for text in texts:
            assert _stale_hits(text) == [], (text, _stale_hits(text))
            assert _missing_markers(text) == [], (text, _missing_markers(text))


class TestItClaimsNothingItCannotShow:
    """The naming rule for the demonstration, enforced rather than promised.

    The banned phrases are assembled from fragments so that this file scanning
    itself does not count as a violation of itself.
    """

    FILES = ("src/demo/showcase.py", "scripts/26_showcase.py", "SHOWCASE.md",
             "README.md")
    BANNED = ("improves Core " "Success", "raises Core " "Success",
              "better than " "arm", "outperform", "state of the " "art",
              "physically " "stable", "stability " "guarantee")

    def bodies(self):
        for rel in self.FILES:
            path = ROOT / rel
            assert path.is_file(), rel
            yield rel, path.read_text(encoding="utf-8")

    def test_no_delivered_file_claims_an_improvement(self):
        for rel, text in self.bodies():
            low = " ".join(text.split()).lower()
            for banned in self.BANNED:
                assert banned.lower() not in low, f"{rel} claims {banned!r}"

    def test_each_delivered_file_states_that_it_measures_nothing(self):
        for rel, text in self.bodies():
            assert "measures nothing" in " ".join(text.split()).lower(), rel

    def test_the_gate_is_stated_as_evaluated_once_and_against(self):
        """No delivered file in this class's scope still denies gen09.

        The document-wide half of the rule. The per-notice, per-paragraph and
        per-output halves are in :class:`TestTheGateStatementIsCheckedPerPlace`
        below, which is where the real work is: a whole-file scan passes as
        soon as the numbers appear *somewhere*, which is exactly how a stale
        sentence survived in one paragraph while another paragraph carried
        the correction.
        """
        for rel, text in self.bodies():
            flat = " ".join(text.split()).lower()
            if "placement" not in flat:
                continue
            assert _stale_hits(text) == [], (rel, _stale_hits(text))

    def test_no_delivered_file_calls_the_cpu_path_the_whole_pipeline(self):
        """The CPU path checks and exports a brick list; it does not make one."""
        for rel, text in self.bodies():
            flat = " ".join(text.split()).lower()
            for claim in ("whole pipeline", "entire pipeline",
                          "full pipeline", "end to end on the cpu",
                          "end-to-end on the cpu"):
                assert claim not in flat, f"{rel} claims {claim!r}"

    def test_connectivity_is_never_dressed_up_as_more(self):
        """Every use of these words is a denial, checked one use at a time.

        A file-wide "the word 'not' appears somewhere" would pass a document
        that denies it once and claims it twice, so each occurrence is read
        with the words just before it.
        """
        banned = ("sup" "port", "stabil" "ity", "phys" "ics")
        for rel, text in self.bodies():
            flat = " ".join(text.split()).lower()
            for word in banned:
                start = 0
                while True:
                    at = flat.find(word, start)
                    if at < 0:
                        break
                    start = at + len(word)
                    if flat[max(0, at - 2):at] == "un":   # unsupported
                        continue
                    if flat[max(0, at - 8):at] == 'scored["':
                        # Reading the frozen scorer's own key name is a
                        # reference, not a claim. The key belongs to
                        # src.eval.scoring and cannot be renamed from here;
                        # what it means is stated where it is printed, in the
                        # scorer's own words.
                        continue
                    before = flat[max(0, at - 60):at]
                    assert ("not" in before or "no " in before
                            or "never" in before), (
                        f"{rel} uses {word!r} without denying it: "
                        f"...{flat[max(0, at - 60):start + 20]}...")


class TestNoPhase2Data:
    """Fixtures only. The banned paths are assembled from fragments so this
    file does not trip its own scan."""

    BANNED = ("instruct" "_inv_test", "instruct" "_inv_train",
              "core_eval" "_plan", "results" ".jsonl", "scores" ".json",
              "data/" "processed", "runs/" "core_eval")

    def test_no_delivered_file_reads_a_corpus_a_plan_or_a_result(self):
        for rel in ("src/demo/showcase.py", "scripts/26_showcase.py",
                    "tests/test_showcase.py", "SHOWCASE.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for banned in self.BANNED:
                assert banned not in text, f"{rel} names {banned}"

    def test_the_only_private_path_named_is_the_model_pointer(self):
        text = (ROOT / "src" / "demo" / "showcase.py").read_text(
            encoding="utf-8")
        assert text.count('"runs/') == 1, "one path literal, and it is the pointer"
        assert 'POINTER = "runs/project_model.json"' in text


# ---------------------------------------------------------------------------
# Round 49: the preview's colours are the file's colours, and its text is text
#
# Both were reported by review as defects rather than found by these tests, so
# each one below is written to go red against the code as it stood: the drawn
# face colours are read out of the writer and out of the SVG it produced, not
# inferred from the fact that a parameter was accepted.
# ---------------------------------------------------------------------------

import re
import warnings

from src.colour.palette import BY_LDRAW


def svg_fills(path) -> set[str]:
    """Every face colour actually present in a written SVG, lower case."""
    text = Path(path).read_text(encoding="utf-8")
    return {value.lower()
            for value in re.findall(r"fill:\s*(#[0-9a-fA-F]{6})", text)}


class TestThePreviewDrawsTheAssignedColours:
    def test_with_no_assignment_it_uses_the_shape_key_and_says_so(self):
        bricks = [Brick(2, 4, 0, 0, 0), Brick(1, 1, 0, 0, 1)]
        drawn, provenance = brick_facecolours(bricks)
        assert drawn == [PART_COLOURS["2x4"], PART_COLOURS["1x1"]]
        assert provenance["colour_source"] == "part-key"

    def test_an_assignment_is_drawn_in_the_palettes_own_values(self):
        bricks = [Brick(2, 4, 0, 0, 0), Brick(1, 1, 0, 0, 1)]
        drawn, provenance = brick_facecolours(bricks, colours={0: 4, 1: 1})
        assert drawn == [BY_LDRAW[4].hex, BY_LDRAW[1].hex]
        assert provenance["colour_source"] == "assignment"
        assert drawn[0] != PART_COLOURS["2x4"] or drawn[1] != PART_COLOURS["1x1"]

    def test_the_svg_really_carries_the_assigned_colours(self, tmp_path):
        """The red light: the file on disk, not the argument list."""
        bricks = [Brick(2, 4, 0, 0, 0), Brick(1, 1, 0, 0, 1)]
        out = write_preview(tmp_path / "a.svg", bricks, colours={0: 4, 1: 1})
        fills = svg_fills(out)
        assert BY_LDRAW[4].hex.lower() in fills
        assert BY_LDRAW[1].hex.lower() in fills
        # 1x1's shape-key colour is not the assigned one, so its absence is
        # what proves the assignment was used rather than merely accepted.
        assert PART_COLOURS["1x1"].lower() not in fills

    def test_two_assignments_produce_two_different_images(self, tmp_path):
        bricks = [Brick(2, 4, 0, 0, 0)]
        red = write_preview(tmp_path / "r.svg", bricks, colours={0: 4})
        blue = write_preview(tmp_path / "b.svg", bricks, colours={0: 1})
        assert BY_LDRAW[4].hex.lower() in svg_fills(red)
        assert BY_LDRAW[1].hex.lower() in svg_fills(blue)
        assert BY_LDRAW[1].hex.lower() not in svg_fills(red)

    def test_a_partial_assignment_is_refused_not_filled_in(self, tmp_path):
        bricks = [Brick(2, 4, 0, 0, 0), Brick(1, 1, 0, 0, 1)]
        with pytest.raises(PreviewError, match="1 of 2 bricks"):
            brick_facecolours(bricks, colours={0: 4})
        with pytest.raises(PreviewError, match="partial assignment"):
            write_preview(tmp_path / "x.png", bricks, colours={0: 4})

    def test_an_assignment_naming_a_brick_that_is_not_there_is_refused(self):
        with pytest.raises(PreviewError, match="not in this structure"):
            brick_facecolours([Brick(1, 1, 0, 0, 0)], colours={0: 4, 3: 1})

    def test_a_colour_outside_the_palette_is_refused_not_substituted(self):
        with pytest.raises(PreviewError, match="not in this project's palette"):
            brick_facecolours([Brick(1, 1, 0, 0, 0)], colours={0: 999})

    def test_a_collision_still_overrides_and_the_override_is_reported(self):
        bricks = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 0)]
        assert find_collisions(bricks)
        drawn, provenance = brick_facecolours(bricks, colours={0: 4, 1: 1})
        assert drawn == [COLLISION_COLOUR, COLLISION_COLOUR]
        assert provenance["collision_overrides"] == [0, 1]
        assert "collision colour" in provenance["note"]

    def test_a_non_dict_assignment_is_refused(self):
        with pytest.raises(PreviewError, match="brick index"):
            brick_facecolours([Brick(1, 1, 0, 0, 0)], colours=[4])


class TestThePreviewDrawsCharactersOrSaysItCannot:
    def test_the_base_family_alone_cannot_draw_chinese(self):
        """The premise of the fix, asserted rather than assumed."""
        assert resolve_font("組裝步驟")["families"][0] != BASE_FONT

    def test_a_chinese_heading_resolves_to_a_family_that_covers_it(self):
        font = resolve_font("第 1 步：組裝預覽")
        assert font["undrawable"] == ()
        assert font["families"][-1] == BASE_FONT
        assert font["cjk_family"] in CJK_FONT_CANDIDATES

    def test_an_ascii_heading_does_not_reach_for_a_cjk_family(self):
        font = resolve_font("step 1/3, 2 brick(s)")
        assert font["families"] == (BASE_FONT,)
        assert font["cjk_family"] is None

    def test_a_drawable_heading_is_returned_unchanged_with_no_note(self):
        caption = safe_heading("第 1 步：組裝預覽")
        assert caption["heading"] == "第 1 步：組裝預覽"
        assert caption["note"] is None

    def test_writing_a_chinese_title_emits_no_missing_glyph_warning(self,
                                                                    tmp_path):
        """The red light for tofu: Matplotlib says so, and it must not."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_preview(tmp_path / "zh.png", [Brick(2, 4, 0, 0, 0)],
                          title="第 1 步：組裝預覽")
        missing = [str(w.message) for w in caught
                   if "missing from font" in str(w.message)]
        assert not missing, missing

    def test_with_no_usable_font_the_characters_come_out_and_it_says_how_many(
            self, monkeypatch):
        """The documented degradation, forced by pretending nothing is here."""
        import src.rendering.preview as preview

        monkeypatch.setattr(preview, "_covered",
                            lambda family, text: frozenset(
                                c for c in set(text) if c.isascii()))
        caption = preview.safe_heading("塔 tower")
        assert "塔" not in caption["heading"]
        assert caption["heading"] == "tower"
        assert caption["note"] == (
            "[1 character(s) omitted: no font on this machine can draw them]")

    def test_a_heading_with_nothing_drawable_left_falls_back_to_the_default(
            self, monkeypatch):
        import src.rendering.preview as preview

        monkeypatch.setattr(preview, "_covered",
                            lambda family, text: frozenset(
                                c for c in set(text) if c.isascii()))
        caption = preview.safe_heading("組裝")
        assert caption["heading"] == DEFAULT_HEADING
        assert "2 character(s) omitted" in caption["note"]

    def test_the_note_is_written_into_the_image(self, tmp_path, monkeypatch):
        import src.rendering.preview as preview

        monkeypatch.setattr(preview, "_covered",
                            lambda family, text: frozenset(
                                c for c in set(text) if c.isascii()))
        out = preview.write_preview(tmp_path / "n.svg",
                                    [Brick(1, 1, 0, 0, 0)], title="塔 tower")
        text = out.read_text(encoding="utf-8")
        assert "omitted" in text
        assert "no font on this machine" in text

    def test_an_absent_family_is_not_an_error(self):
        import src.rendering.preview as preview

        assert preview._font_file("No Such Family At All") is None
