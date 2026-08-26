"""The minimum two-page UI: form, run, render, serve, and the boundaries.

Every test here is offline and CPU-only.  All catalogues are synthetic train
fixtures.  No model, no weights, no network, no GPU, no frozen evaluation case
and no metric is involved, and several of these tests exist precisely to make
that non-negotiable rather than merely true today.

Nothing needs a click.  The form, the run and the rendering are plain
functions; the HTTP surface is exercised over a real socket bound to
``127.0.0.1:0`` with :mod:`http.client`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import Brick, PART_VOCAB, format_bricks, parse_bricks
from src.data.retile import RetileResult
from src.delivery import pipeline
from src.delivery.pipeline import DeliveryError
from src.demo.showcase import ShowcaseError
from src.rendering.ldr import to_ldr
from src.rendering.preview import PART_COLOURS
from src.ui import app as ui_app
from src.ui import render as ui_render
from src.ui import server as ui_server

ROOT = Path(__file__).resolve().parents[1]
UI_SOURCES = ("src/ui/__init__.py", "src/ui/app.py", "src/ui/render.py",
              "src/ui/server.py", "scripts/29_ui.py")

#: A stand-in form key for the render-only tests. The served pages use the
#: server's own; this one only has to be a non-empty string.
KEY = "test-form-key"

#: This suite is published. ``PROJECT_STATUS.md`` is not, so the one test that
#: reads it has to skip in the public tree rather than fail there.
STATUS_NODE = ("tests/test_ui.py::TestTheRecordMatchesTheWork"
               "::test_the_status_file_does_not_claim_delivery_is_complete")

_UNSET = object()

STACK = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]
SMALL = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 1)]
FLOATING = [Brick(1, 1, 0, 0, 1), Brick(1, 1, 0, 0, 2)]
DISCONNECTED = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 2, 2, 0)]


def row(sid: str, oid: str, caption: str, bricks: list[Brick], **kw) -> dict:
    return {
        "split": kw.get("split", "train"),
        "role": kw.get("role", "control"),
        "variant": kw.get("variant", "exact"),
        "object_id": oid,
        "structure_id": sid,
        "caption": caption,
        "bricks_txt": format_bricks(bricks),
    }


def catalogue_file(tmp_path, rows=None, *, name="synthetic_train.jsonl"):
    path = tmp_path / name
    body = rows or [
        row("s-car", "o-car", "a compact red car", STACK),
        row("s-tower", "o-tower", "a tiny yellow tower", SMALL),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in body),
                    encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def frozen_split_fixture(tmp_path, monkeypatch):
    """Synthetic rows get a split contract without weakening production."""
    objects = {"o-car": "train", "o-tower": "train", "o": "train",
               "o-float": "train", "val-object": "val"}
    structures = {"s-car": "o-car", "s-tower": "o-tower", "s": "o",
                  "s-float": "o-float", "s-val": "val-object"}
    manifest = tmp_path / "object_splits.json"
    manifest.write_text(json.dumps({
        "meta": {"fixture": True}, "counts": {"train": 4, "val": 1},
        "objects": objects, "structures": structures,
    }, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(pipeline, "FROZEN_SPLIT_MANIFEST_SHA256", digest)
    return manifest


def form(**over) -> dict[str, list[str]]:
    """A minimal valid page-one submission, with overrides."""
    fields = {"mode": ["compare"], "caption": ["a compact red car"],
              "qty_2x4": ["2"], "top_n": ["3"]}
    for key, value in over.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = value if isinstance(value, list) else [str(value)]
    return fields


def run(fields, catalog, **kw) -> ui_app.UiResult:
    return ui_app.run_request(ui_app.parse_form(fields), catalog=catalog, **kw)


# ---------------------------------------------------------------------------


class TestTheDeliveryContractIsBorrowedNotCopied:
    """The UI must not own a second opinion about what is deliverable."""

    def test_the_loaded_module_really_is_the_delivery_command_line(self):
        module = ui_app.load_delivery()
        assert Path(module.__file__) == ROOT / "scripts/27_delivery.py"
        for attribute in ("make_payload", "DELIVERY_CHECKS", "DEFAULT_CATALOG"):
            assert hasattr(module, attribute)

    def test_the_static_checks_are_read_not_restated(self):
        module = ui_app.load_delivery()
        assert ui_app.delivery_checks() == tuple(module.DELIVERY_CHECKS)
        assert "termination_accepted" not in ui_app.delivery_checks()
        for rel in UI_SOURCES:
            body = (ROOT / rel).read_text(encoding="utf-8")
            assert "DELIVERY_CHECKS =" not in body
            assert "static_delivery_ready =" not in body

    def test_a_command_line_missing_its_contract_is_refused(self, tmp_path):
        impostor = tmp_path / "27_delivery.py"
        impostor.write_text("DEFAULT_CATALOG = 'x'\n", encoding="utf-8")
        with pytest.raises(ui_app.UiError, match="substitute its own copy"):
            ui_app.load_delivery(impostor)

    def test_a_missing_command_line_is_a_named_refusal(self, tmp_path):
        with pytest.raises(ui_app.UiError, match="missing"):
            ui_app.load_delivery(tmp_path / "absent.py")

    def test_the_default_catalogue_is_the_command_line_s_own(self):
        assert (ui_app.default_catalog()
                == Path(ui_app.load_delivery().DEFAULT_CATALOG))


class TestPageOneAcceptsAndRefuses:
    def test_a_minimal_submission_parses(self):
        request = ui_app.parse_form(form())
        assert request.mode == "compare"
        assert request.inventory == {"2x4": 2}
        assert request.inventory_spec == "2x4:2"
        assert request.top_n == 3
        assert request.time_limit is None and request.seed is None

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ui_app.UiError, match="模式"):
            ui_app.parse_form(form(mode="generate"))

    @pytest.mark.parametrize("caption", ["", "   ", "\n\t "])
    def test_an_empty_brief_is_refused(self, caption):
        with pytest.raises(ui_app.UiError, match="文字需求"):
            ui_app.parse_form(form(caption=caption))

    def test_an_overlong_brief_is_refused_by_size(self):
        with pytest.raises(ui_app.UiError, match="超過上限"):
            ui_app.parse_form(
                form(caption="x" * (ui_app.MAX_CAPTION_CHARS + 1)))

    def test_a_field_submitted_twice_is_refused_not_silently_first_wins(self):
        with pytest.raises(ui_app.UiError, match="送出 2 次"):
            ui_app.parse_form(form(caption=["one", "two"]))

    def test_the_grid_builds_a_spec_in_vocabulary_order(self):
        request = ui_app.parse_form(
            form(qty_2x4="1", qty_1x1="2", qty_2x6="3"))
        assert request.inventory_spec == "1x1:2,2x4:1,2x6:3"
        assert list(request.inventory) == ["1x1", "2x4", "2x6"]

    def test_a_zero_or_blank_quantity_is_not_stock(self):
        request = ui_app.parse_form(form(qty_2x4="2", qty_1x1="0", qty_1x2=""))
        assert request.inventory == {"2x4": 2}

    @pytest.mark.parametrize("bad", ["-1", "two", "1.5", " "])
    def test_a_non_whole_quantity_is_refused(self, bad):
        with pytest.raises(ui_app.UiError):
            ui_app.parse_form(form(qty_2x4=bad))

    def test_rotation_is_normalised_by_the_existing_parser(self):
        request = ui_app.parse_form(
            form(qty_2x4=None, inventory_spec="4x1:3"))
        assert request.inventory == {"1x4": 3}

    def test_both_rotations_of_one_part_are_refused_not_summed(self):
        with pytest.raises(ShowcaseError, match="same part"):
            ui_app.parse_form(form(qty_2x4=None, inventory_spec="1x4:2,4x1:3"))

    def test_a_part_outside_the_eight_is_refused(self):
        with pytest.raises(ShowcaseError):
            ui_app.parse_form(form(qty_2x4=None, inventory_spec="3x3:2"))

    def test_the_two_inventory_inputs_may_not_both_be_used(self):
        with pytest.raises(ui_app.UiError, match="只用其中一種"):
            ui_app.parse_form(form(inventory_spec="1x1:1"))

    def test_no_inventory_at_all_is_refused(self):
        with pytest.raises(ui_app.UiError, match="庫存不可空白"):
            ui_app.parse_form(form(qty_2x4=None))

    @pytest.mark.parametrize("bad", ["0", "-1", str(ui_app.MAX_TOP_N + 1),
                                     "many"])
    def test_a_bad_top_n_is_refused(self, bad):
        with pytest.raises(ui_app.UiError, match="Top-N"):
            ui_app.parse_form(form(top_n=bad))

    def test_a_blank_top_n_takes_the_command_line_s_own_default(self):
        """Blank is "not given", and not given already has an answer."""
        assert ui_app.parse_form(form(top_n="")).top_n == ui_app.DEFAULT_TOP_N
        assert ui_app.DEFAULT_TOP_N == ui_app.load_delivery(
        ).build_parser().parse_args(
            ["--mode", "compare", "--caption", "x",
             "--inventory", "2x4:1"]).top_n


class TestFieldApplicability:
    """An inapplicable field is named and refused, never quietly dropped."""

    @pytest.mark.parametrize("field", ui_app.PIPELINE_ONLY_FIELDS)
    def test_a_cp_sat_control_is_refused_on_the_comparison_mode(self, field):
        with pytest.raises(ui_app.UiError) as exc:
            ui_app.parse_form(form(mode="compare", **{field: "2"}))
        assert field in str(exc.value)
        assert "不靜默忽略" in str(exc.value)

    @pytest.mark.parametrize("field", ui_app.PIPELINE_ONLY_FIELDS)
    def test_a_blank_cp_sat_control_is_fine_on_the_comparison_mode(self,
                                                                   field):
        assert ui_app.parse_form(form(mode="compare", **{field: ""})) is not None

    def test_the_delivery_command_line_refuses_the_same_pairing(self, tmp_path):
        """Even if the form let one through, the payload would not."""
        request = ui_app.parse_form(form(mode="compare"))
        smuggled = request.as_namespace(catalogue_file(tmp_path))
        smuggled.time_limit = 2.0
        with pytest.raises(DeliveryError, match="does not apply"):
            ui_app.load_delivery().make_payload(smuggled)

    def test_the_pipeline_mode_defaults_match_the_command_line(self):
        request = ui_app.parse_form(form(mode="f-pipeline"))
        assert request.time_limit == 2.0
        assert request.seed == 0
        module = ui_app.load_delivery()
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "time_limit is not None else 2.0" in source
        assert "seed is not None else 0" in source

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-2",
                                     str(ui_app.MAX_TIME_LIMIT + 1), "soon"])
    def test_a_bad_time_limit_never_reaches_cp_sat(self, bad):
        with pytest.raises(ui_app.UiError, match="time limit"):
            ui_app.parse_form(form(mode="f-pipeline", time_limit=bad))

    @pytest.mark.parametrize("bad", ["-1", "1.5", "later"])
    def test_a_bad_seed_is_refused(self, bad):
        with pytest.raises(ui_app.UiError, match="seed"):
            ui_app.parse_form(form(mode="f-pipeline", seed=bad))

    def test_the_query_object_exclusion_is_not_offered(self):
        """It guards an evaluation, and this UI has no evaluation to guard."""
        namespace = ui_app.parse_form(form()).as_namespace("x_train.jsonl")
        assert namespace.exclude_object_id is None
        for rel in UI_SOURCES:
            body = (ROOT / rel).read_text(encoding="utf-8")
            assert "exclude-object-id" not in body


class TestRunningARequest:
    def test_a_comparison_reaches_a_deliverable_result(self, tmp_path):
        result = run(form(), catalogue_file(tmp_path))
        assert result.ready is True
        assert result.payload["result"]["status"] == (
            "buildable_existing_work_found")
        assert result.report["delivery"]["static_delivery_ready"] is True
        assert result.artifacts is not None

    def test_no_buildable_candidate_is_a_result_not_an_error(self, tmp_path):
        result = run(form(qty_2x4="1"), catalogue_file(tmp_path))
        assert result.ready is False
        assert result.payload["result"]["status"] == (
            "no_buildable_existing_work_in_retrieved_set")
        assert result.artifacts is None and result.handle is None
        assert any(candidate["missing_parts"]
                   for candidate in result.payload["result"]["inventory_reranked"])

    def test_the_pipeline_re_tiles_and_independently_verifies(self, tmp_path):
        result = run(form(mode="f-pipeline", caption="a compact red car",
                          top_n="1", time_limit="2"),
                     catalogue_file(tmp_path, [row("s", "o", "red car", STACK)]))
        assert result.ready is True
        attempt = result.payload["result"]["attempts"][0]
        assert attempt["solver_status"] in ("OPTIMAL", "FEASIBLE")
        assert attempt["exact_cover_verified"] is True
        assert attempt["delivery_ready"] is True

    def test_infeasible_and_timeout_stay_distinguishable(self, tmp_path,
                                                         monkeypatch):
        catalog = catalogue_file(tmp_path, [row("s", "o", "red car", STACK)])
        infeasible = run(form(mode="f-pipeline", caption="red car",
                              qty_2x4=None, qty_1x1="1", top_n="1"), catalog)
        assert infeasible.ready is False
        assert infeasible.payload["result"]["attempts"][0][
            "solver_status"] == "INFEASIBLE"
        assert infeasible.payload["result"]["status"] == "no_valid_build"

        monkeypatch.setattr(pipeline, "retile",
                            lambda *a, **k: RetileResult(None, "UNKNOWN",
                                                         0.01, 12))
        timed_out = run(form(mode="f-pipeline", caption="red car",
                             top_n="1", time_limit="1"), catalog)
        assert timed_out.ready is False
        attempt = timed_out.payload["result"]["attempts"][0]
        assert attempt["solver_status"] == "UNKNOWN"
        assert attempt["failure"] == "solver timeout"

    def test_a_tiling_that_fails_the_checker_is_not_published_as_ready(
            self, tmp_path, monkeypatch):
        catalog = catalogue_file(tmp_path)
        monkeypatch.setattr(
            pipeline, "occupancy_of",
            lambda bricks: {cell for b in FLOATING for cell in b.cells})
        monkeypatch.setattr(
            pipeline, "retile",
            lambda *a, **k: RetileResult(FLOATING, "OPTIMAL", 0.01, 2))
        result = run(form(mode="f-pipeline", caption="a compact red car",
                          qty_2x4=None, qty_1x1="2", top_n="1"), catalog)
        assert result.ready is False
        assert result.artifacts is None
        assert result.payload["result"]["status"] == (
            "tiling_found_but_not_delivery_ready")
        assert "touch ground" in result.payload["result"]["attempts"][0]["failure"]

    def test_a_missing_catalogue_is_a_readable_refusal(self, tmp_path):
        with pytest.raises(DeliveryError, match="not a file"):
            run(form(), tmp_path / "absent_train.jsonl")

    def test_a_catalogue_that_is_not_train_only_is_refused(self, tmp_path):
        wrong_name = catalogue_file(tmp_path, name="synthetic_test.jsonl")
        with pytest.raises(DeliveryError, match="_train.jsonl"):
            run(form(), wrong_name)
        held_out = catalogue_file(
            tmp_path, [row("s-val", "val-object", "x", STACK, split="val")],
            name="held_train.jsonl")
        with pytest.raises(DeliveryError, match="train-only"):
            run(form(), held_out)

    def test_the_payload_says_no_model_and_the_ui_insists_on_it(self,
                                                                tmp_path):
        result = run(form(), catalogue_file(tmp_path))
        assert result.payload["method"]["model_loaded"] is False
        assert result.payload["method"]["phase_3c"] == (
            "not authorised and not run")

        class Liar:
            @staticmethod
            def make_payload(args):
                payload, report = ui_app.load_delivery().make_payload(args)
                payload["method"]["model_loaded"] = True
                return payload, report

        with pytest.raises(ui_app.UiError, match="refuses to display"):
            run(form(), catalogue_file(tmp_path), delivery=Liar)


class TestTheTwoOutputsAreOneBrickList:
    def test_ldraw_and_preview_come_from_the_same_text(self, tmp_path):
        result = run(form(), catalogue_file(tmp_path), store=ui_app.ResultStore())
        art = result.artifacts
        bricks = parse_bricks(result.report["result"]["text"])
        assert art.bricks_text == result.report["result"]["text"]
        assert art.ldraw == to_ldr(bricks)
        assert len(parse_bricks(art.bricks_text)) == len(bricks)

    def test_the_preview_is_a_real_png_and_reports_its_own_size(self,
                                                                tmp_path):
        art = run(form(), catalogue_file(tmp_path)).artifacts
        assert art.preview.startswith(b"\x89PNG\r\n\x1a\n")
        assert art.preview_media_type == "image/png"
        assert (art.preview_width, art.preview_height) == ui_app.png_size(
            art.preview)
        assert art.preview_width > 100 and art.preview_height > 100
        assert len(art.preview) > 5_000

    def test_a_non_png_is_refused_rather_than_measured(self):
        with pytest.raises(ui_app.UiError, match="not a PNG"):
            ui_app.png_size(b"GIF89a" + b"\x00" * 40)

    def test_nothing_is_written_into_the_project_tree(self, tmp_path,
                                                      monkeypatch):
        before = {p for p in (ROOT / "artifacts").rglob("*")}
        run(form(), catalogue_file(tmp_path), store=ui_app.ResultStore())
        assert {p for p in (ROOT / "artifacts").rglob("*")} == before

    def test_the_store_is_bounded_and_round_trips(self, tmp_path):
        store = ui_app.ResultStore(limit=2)
        catalog = catalogue_file(tmp_path)
        handles = [run(form(caption=f"a compact red car {i}"), catalog,
                      store=store).handle for i in range(3)]
        assert len(store) == 2
        assert store.get(handles[0]) is None
        assert store.get(handles[-1]) is not None
        assert len(set(handles)) == 3

    def test_a_store_with_no_room_is_refused(self):
        with pytest.raises(ValueError):
            ui_app.ResultStore(limit=0)


class TestNoModelCanBeReached:
    """The interface must not generate, and must not be one edit from it."""

    #: Text with no innocent reading. Naming a module like this is importing
    #: it, whatever the surrounding line claims.
    FORBIDDEN_IMPORTS = (
        "import torch", "from torch", "import transformers",
        "from transformers", "from peft", "import peft",
        "src.generation", "src.training", "src.eval.oracle",
        "src.model_ids", "src.constraints.placement_decode",
    )
    #: Calls, matched with their parenthesis so that prose naming a component
    #: -- "the writer aligned against the BrickGPT reference vectors" -- stays
    #: sayable while calling it does not.
    FORBIDDEN_CALLS = (
        "BrickGPT(", "load_tokenizer(", "load_finetuned(",
        "load_merged_brickgpt(", "from_pretrained(", "inspect_decoded(",
        "project_adapter_dir(", "generate(", "InventoryPlacementGate(",
    )

    @pytest.mark.parametrize("rel", UI_SOURCES)
    def test_no_ui_source_imports_model_machinery(self, rel):
        body = (ROOT / rel).read_text(encoding="utf-8")
        for forbidden in self.FORBIDDEN_IMPORTS:
            assert forbidden not in body, f"{rel} names {forbidden}"

    @pytest.mark.parametrize("rel", UI_SOURCES)
    def test_no_ui_source_calls_a_model_entry_point(self, rel):
        body = (ROOT / rel).read_text(encoding="utf-8")
        for forbidden in self.FORBIDDEN_CALLS:
            assert forbidden not in body, f"{rel} calls {forbidden}"

    def test_the_decode_flag_appears_only_where_it_is_being_refused(self):
        """The pages may say the model path is absent; they may not offer it."""
        for rel in UI_SOURCES:
            body = (ROOT / rel).read_text(encoding="utf-8")
            for line in body.splitlines():
                if "--generate" in line:
                    assert "不提供" in line or "no " in line.lower(), line

    @pytest.mark.parametrize("rel", UI_SOURCES)
    def test_no_ui_source_reaches_the_frozen_evaluation(self, rel):
        body = (ROOT / rel).read_text(encoding="utf-8")
        for forbidden in ("runs/core_eval", "core_eval_plan", "scores.json",
                          "gpu_plans", "25_core_eval", "23_final_eval"):
            assert forbidden not in body, f"{rel} names {forbidden}"

    def test_a_full_successful_run_never_calls_a_generation_entry_point(
            self, tmp_path, monkeypatch):
        """Booby-trap every door to a decode, then walk the whole happy path."""
        import src.demo.showcase as showcase
        import src.generation.brickgpt as brickgpt

        def explode(*a, **k):                     # pragma: no cover - must not run
            raise AssertionError("the UI reached a model entry point")

        monkeypatch.setattr(showcase, "generate", explode)
        monkeypatch.setattr(showcase, "inspect_decoded", explode)
        monkeypatch.setattr(showcase, "project_adapter_dir", explode)
        monkeypatch.setattr(brickgpt, "BrickGPT", explode)
        monkeypatch.setattr(brickgpt.AutoModelForCausalLM, "from_pretrained",
                            explode)
        monkeypatch.setattr(brickgpt.AutoTokenizer, "from_pretrained", explode)

        result = run(form(), catalogue_file(tmp_path),
                     store=ui_app.ResultStore())
        assert result.ready is True
        assert ui_render.render_page_two(result)

    def test_the_ui_pulls_in_nothing_the_delivery_command_line_does_not(self):
        """An honest bound: the scorer already imports torch, and the UI is
        forbidden to make that worse rather than pretending it is absent."""
        import subprocess
        probe = (
            "import sys, json; sys.path.insert(0, {root!r});\n"
            "import importlib.util;\n"
            "{load}\n"
            "print(json.dumps(sorted({{m.split('.')[0] for m in sys.modules}})))"
        )
        cli_load = (
            "spec = importlib.util.spec_from_file_location("
            "'d', {root!r} + '/scripts/27_delivery.py');\n"
            "m = importlib.util.module_from_spec(spec);"
            " spec.loader.exec_module(m)").format(root=str(ROOT))
        ui_load = "import src.ui.server"
        seen = {}
        for name, load in (("cli", cli_load), ("ui", ui_load)):
            out = subprocess.run(
                [sys.executable, "-c",
                 probe.format(root=str(ROOT), load=load)],
                capture_output=True, text=True, cwd=str(ROOT), timeout=300)
            assert out.returncode == 0, out.stderr
            seen[name] = set(json.loads(out.stdout))
        extra = seen["ui"] - seen["cli"]
        assert not (extra & {"torch", "transformers", "peft", "accelerate",
                             "datasets", "tokenizers", "safetensors"}), extra

    def test_the_documented_boundary_says_which_claim_is_being_made(self):
        body = (ROOT / "UI.md").read_text(encoding="utf-8")
        assert "torch" in body
        assert "src.eval.scoring" in body


class TestRenderingSaysWhatItKnows:
    def test_page_one_renders_the_eight_parts_in_preview_colours(self):
        html = ui_render.render_page_one(csrf_token=KEY)
        for part in PART_VOCAB:
            assert f'name="qty_{part}"' in html
            assert PART_COLOURS[part] in html
        assert 'value="compare"' in html and 'value="f-pipeline"' in html

    def test_page_one_marks_step_one_and_page_two_marks_step_two(self,
                                                                 tmp_path):
        one = ui_render.render_page_one(csrf_token=KEY)
        two = ui_render.render_page_two(run(form(), catalogue_file(tmp_path),
                                            store=ui_app.ResultStore()))
        assert one.count('<li aria-current="step">') == 1
        assert two.count('<li aria-current="step">') == 1
        current_one = re.search(r'<li aria-current="step">\s*'
                                r'<span class="n">(\d)</span>', one)
        current_two = re.search(r'<li aria-current="step">\s*'
                                r'<span class="n">(\d)</span>', two)
        assert current_one and current_one.group(1) == "1"
        assert current_two and current_two.group(1) == "2"

    def test_a_refusal_keeps_what_was_typed(self):
        fields = form(caption="my own brief", qty_2x4="4", top_n="7")
        state = ui_render.form_state(fields)
        html = ui_render.render_page_one(csrf_token=KEY, form=state,
                                          error="測試用的拒絕訊息")
        assert "my own brief" in html
        qty = re.search(r'name="qty_2x4"(.*?)>', html, re.DOTALL)
        assert qty and 'value="4"' in qty.group(1)
        top = re.search(r'name="top_n"(.*?)>', html, re.DOTALL)
        assert top and 'value="7"' in top.group(1)
        assert "測試用的拒絕訊息" in html
        assert 'role="alert"' in html

    def test_operator_text_is_escaped_not_executed(self, tmp_path):
        nasty = '<script>alert("x")</script>'
        html = ui_render.render_page_one(
            csrf_token=KEY, form=ui_render.form_state(form(caption=nasty)),
            error=nasty)
        assert nasty not in html
        assert "&lt;script&gt;" in html

        catalog = catalogue_file(
            tmp_path, [row("s", "o", '<img src=x onerror=1>', STACK)])
        page = ui_render.render_page_two(
            run(form(caption="<b>hi</b> a compact red car"), catalog,
                store=ui_app.ResultStore()))
        assert "<img src=x onerror=1>" not in page
        assert "<b>hi</b>" not in page
        assert "&lt;b&gt;hi&lt;/b&gt;" in page

    def test_a_termination_check_reads_n_a_not_fail(self, tmp_path):
        result = run(form(), catalogue_file(tmp_path),
                     store=ui_app.ResultStore())
        assert result.report["checks"]["termination_accepted"] is None
        assert result.report["checks"]["deterministic_core_success"] is None
        html = ui_render.render_page_two(result)
        rows = {r["name"]: r["label"]
                for r in ui_render._check_rows(result.report)}
        assert rows["termination_accepted"] == "n/a"
        assert rows["deterministic_core_success"] == "n/a"
        assert rows["touches_ground"] == "pass"
        assert "本介面沒有跑解碼器" in html
        assert "不寫成 false" in html

    def test_every_check_carries_a_word_not_only_a_colour(self, tmp_path):
        result = run(form(), catalogue_file(tmp_path),
                     store=ui_app.ResultStore())
        for record in ui_render._check_rows(result.report):
            assert record["label"] in ("pass", "FAIL", "n/a")
            assert record["gloss"]

    def test_the_standing_limits_appear_on_both_pages(self, tmp_path):
        pages = [ui_render.render_page_one(csrf_token=KEY),
                 ui_render.render_page_two(run(form(),
                                               catalogue_file(tmp_path),
                                               store=ui_app.ResultStore()))]
        for html in pages:
            assert ui_app.NOT_A_METRIC_ZH in html
            assert ui_app.CONNECTIVITY_LIMIT_ZH in html
            assert ui_app.NO_MODEL_ZH in html

    def test_page_two_records_the_catalogue_digest_it_actually_used(
            self, tmp_path):
        catalog = catalogue_file(tmp_path)
        result = run(form(), catalog, store=ui_app.ResultStore())
        digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
        html = ui_render.render_page_two(result)
        assert digest in html
        assert result.payload["catalog"]["sha256"] == digest
        assert pipeline.FROZEN_SPLIT_MANIFEST_SHA256 in html

    def test_no_third_party_brand_appears_anywhere_in_the_interface(self):
        for rel in UI_SOURCES + ("UI.md",):
            body = (ROOT / rel).read_text(encoding="utf-8").lower()
            for mark in ("lego", "樂高", "duplo", "minifig"):
                assert mark not in body, f"{rel} names {mark}"


class TestDownloadIsOfferedOnlyWhenEarned:
    def test_a_ready_result_offers_the_preview_and_the_file(self, tmp_path):
        result = run(form(), catalogue_file(tmp_path),
                     store=ui_app.ResultStore())
        html = ui_render.render_page_two(result)
        assert f"/artifact/{result.handle}/preview.png" in html
        assert f"/artifact/{result.handle}/model.ldr" in html
        assert 'download' in html

    def test_a_result_that_is_not_ready_offers_neither(self, tmp_path):
        result = run(form(qty_2x4="1"), catalogue_file(tmp_path),
                     store=ui_app.ResultStore())
        assert result.ready is False and result.handle is None
        html = ui_render.render_page_two(result)
        assert "/artifact/" not in html
        assert "model.ldr" not in html
        assert "preview.png" not in html
        assert "沒有可交付結果" in html

    def test_a_structure_that_will_not_serialise_never_reaches_a_download(
            self, tmp_path, monkeypatch):
        import src.ui.app as module

        def refuse(report, path):
            raise ShowcaseError("this structure does not serialise to LDraw")

        monkeypatch.setattr(module, "write_ldraw", refuse)
        with pytest.raises(ShowcaseError):
            run(form(), catalogue_file(tmp_path), store=ui_app.ResultStore())


# ---------------------------------------------------------------------------
# The HTTP surface, over a real loopback socket
# ---------------------------------------------------------------------------

class Client:
    """A tiny http.client wrapper so tests read like requests, not sockets."""

    def __init__(self, server):
        self.host, self.port = server.server_address[:2]
        self.key = server.csrf_key

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=60)

    def get(self, path, host_header=None):
        conn = self._conn()
        conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", host_header or f"{self.host}:{self.port}")
        conn.endheaders()
        response = conn.getresponse()
        body = response.read()
        headers = dict(response.getheaders())
        status = response.status
        conn.close()
        return status, headers, body.decode("utf-8", "replace"), body

    def post(self, path, fields, *, media_type=ui_server.FORM_MEDIA_TYPE,
             body=None, declared_length=None, origin=None, key=_UNSET,
             host_header=None):
        """POST a form. By default it carries the server's own form key."""
        from urllib.parse import urlencode
        sent = dict(fields)
        chosen = self.key if key is _UNSET else key
        if chosen is not None:
            sent.setdefault(ui_server.CSRF_FIELD,
                            chosen if isinstance(chosen, list) else [chosen])
        payload = (body if body is not None
                   else urlencode([(k, v) for k, values in sent.items()
                                   for v in values]).encode("utf-8"))
        conn = self._conn()
        conn.putrequest("POST", path, skip_host=True,
                        skip_accept_encoding=True)
        conn.putheader("Host", host_header or f"{self.host}:{self.port}")
        conn.putheader("Content-Type", media_type)
        conn.putheader("Content-Length",
                       str(declared_length if declared_length is not None
                           else len(payload)))
        if origin is not None:
            conn.putheader("Origin", origin)
        conn.endheaders()
        conn.send(payload)
        response = conn.getresponse()
        text = response.read().decode("utf-8", "replace")
        status = response.status
        headers = dict(response.getheaders())
        conn.close()
        return status, text, headers

    def raw(self, blob: bytes) -> bytes:
        """Write bytes at the socket and read until the server stops talking.

        The only way to see whether a refusal left a request body in the pipe
        is to look at the pipe.
        """
        sock = socket.create_connection((self.host, self.port), timeout=60)
        try:
            sock.sendall(blob)
            sock.settimeout(20)
            chunks = []
            while True:
                try:
                    piece = sock.recv(65536)
                except socket.timeout:
                    break
                if not piece:
                    break
                chunks.append(piece)
            return b"".join(chunks)
        finally:
            sock.close()


@pytest.fixture
def live(tmp_path):
    """A real server on 127.0.0.1:0, torn down after the test."""
    server = ui_server.create_server(
        host="127.0.0.1", port=0, catalog=catalogue_file(tmp_path),
        store=ui_app.ResultStore())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, Client(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


class TestTheServer:
    def test_it_binds_loopback_and_refuses_anything_else(self):
        with pytest.raises(ui_app.UiError, match="loopback only"):
            ui_server.create_server(host="0.0.0.0", port=0)
        assert ui_server.is_loopback_host("127.0.0.1")
        assert ui_server.is_loopback_host("localhost")
        assert not ui_server.is_loopback_host("0.0.0.0")
        assert not ui_server.is_loopback_host("example.invalid")

    def test_the_first_page_is_served(self, live):
        _, client = live
        status, headers, text, _ = client.get("/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert "需求與庫存" in text and 'name="caption"' in text
        assert headers["Cache-Control"] == "no-store"
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    def test_submitting_moves_to_the_second_page(self, live):
        _, client = live
        status, text, _ = client.post("/result", form())
        assert status == 200
        assert "結果與交付" in text
        assert "找到一件通過靜態交付檢查的結果" in text

    def test_a_refused_submission_returns_to_the_first_page(self, live):
        _, client = live
        status, text, _ = client.post("/result", form(mode="compare",
                                                   time_limit="2"))
        assert status == 400
        assert "需求與庫存" in text
        assert "不靜默忽略" in text
        assert "a compact red car" in text          # what was typed survives
        assert "Traceback" not in text

    def test_a_run_with_no_deliverable_result_is_still_the_second_page(self,
                                                                       live):
        _, client = live
        status, text, _ = client.post("/result", form(qty_2x4="1"))
        assert status == 200
        assert "沒有可交付結果" in text
        assert "/artifact/" not in text

    def test_the_preview_and_the_file_are_served_from_memory(self, live):
        server, client = live
        status, text, _ = client.post("/result", form())
        assert status == 200
        handle = re.search(r"/artifact/([A-Za-z0-9_-]+)/preview\.png",
                          text).group(1)

        status, headers, _, raw = client.get(f"/artifact/{handle}/preview.png")
        assert status == 200 and headers["Content-Type"] == "image/png"
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")

        status, headers, body, _ = client.get(f"/artifact/{handle}/model.ldr")
        assert status == 200
        assert headers["Content-Disposition"] == (
            'attachment; filename="brickagain.ldr"')
        assert body == server.store.get(handle).ldraw
        assert body.strip().splitlines()[0].startswith("1 ")

    def test_the_file_and_the_image_describe_the_same_structure(self, live):
        server, client = live
        _, text, _ = client.post("/result", form())
        handle = re.search(r"/artifact/([A-Za-z0-9_-]+)/preview\.png",
                          text).group(1)
        artifacts = server.store.get(handle)
        _, _, ldraw, _ = client.get(f"/artifact/{handle}/model.ldr")
        bricks = parse_bricks(artifacts.bricks_text)
        assert ldraw == to_ldr(bricks)
        assert ldraw.count(".DAT") == len(bricks)

    @pytest.mark.parametrize("path", [
        "/artifact/unknown-handle/preview.png",
        "/artifact/unknown-handle/model.ldr",
    ])
    def test_an_unknown_handle_is_a_sentence_not_a_stack_trace(self, live, path):
        _, client = live
        status, _, text, _ = client.get(path)
        assert status == 404
        assert "Traceback" not in text
        assert "只存在於本次執行的記憶體中" in text

    @pytest.mark.parametrize("path", ["/nope", "/result", "/artifact/",
                                      "/artifact/a/b/c", "/artifact/../etc"])
    def test_an_unknown_path_is_refused_readably(self, live, path):
        _, client = live
        status, _, text, _ = client.get(path)
        assert status == 404
        assert "沒有這個頁面" in text
        assert "Traceback" not in text

    def test_a_non_loopback_host_header_is_refused(self, live):
        _, client = live
        status, _, text, _ = client.get("/", host_header="brickagain.example")
        assert status == 403
        assert "只服務本機" in text

    def test_only_a_form_body_is_accepted(self, live):
        _, client = live
        status, text, _ = client.post("/result", form(),
                                   media_type="application/json")
        assert status == 415
        assert "只接受表單送出" in text

    def test_an_oversized_body_is_refused_before_it_is_read(self, live):
        _, client = live
        status, text, _ = client.post(
            "/result", {}, body=b"caption=x",
            declared_length=ui_server.MAX_BODY_BYTES + 1)
        assert status == 413
        assert "過大" in text

    def test_a_defect_is_a_sentence_and_the_trace_goes_to_stderr(
            self, live, monkeypatch, capfd):
        _, client = live

        def boom(*a, **k):
            raise RuntimeError("a deliberate defect")

        monkeypatch.setattr(ui_app, "run_request", boom)
        status, text, _ = client.post("/result", form())
        assert status == 500
        assert "未預期的錯誤" in text
        assert "Traceback" not in text
        assert "a deliberate defect" not in text
        assert "a deliberate defect" in capfd.readouterr().err

    def test_a_missing_catalogue_is_announced_before_it_is_submitted(
            self, tmp_path):
        server = ui_server.create_server(
            host="127.0.0.1", port=0, catalog=tmp_path / "absent_train.jsonl",
            store=ui_app.ResultStore())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = Client(server)
            _, _, text, _ = client.get("/")
            assert "找不到目錄檔" in text
            status, refused, _ = client.post("/result", form())
            assert status == 400
            assert "Traceback" not in refused
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)


@pytest.fixture(scope="module")
def launcher():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m29", ROOT / "scripts/29_ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheLauncher:
    def test_one_command_is_all_it_takes(self, launcher):
        parser = launcher.build_parser()
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8765
        assert args.catalog is None

    @pytest.mark.parametrize("argv,needle", [
        (["--host", "0.0.0.0"], "loopback only"),
        (["--port", "70000"], "not a TCP port"),
    ])
    def test_a_bad_bind_is_a_named_refusal(self, launcher, argv, needle,
                                           capsys):
        assert launcher.main(argv) == launcher.EXIT_REFUSED
        assert needle in capsys.readouterr().err

    def test_the_launcher_documents_what_it_will_not_do(self, launcher):
        doc = launcher.__doc__
        for promise in ("loopback only", "no model weights", "no metric"):
            assert promise in doc


class TestTheRecordMatchesTheWork:
    def test_the_handbook_names_every_boundary(self):
        body = (ROOT / "UI.md").read_text(encoding="utf-8")
        for required in (
                "scripts/29_ui.py", "127.0.0.1", "scripts/27_delivery.py",
                "不是多語 embedding", "不是物理支撐", "不是指標",
                "Phase 3", "artifacts/", "Streamlit", "Gradio", "Jinja2"):
            assert required in body

    def test_the_readme_points_at_the_interface(self):
        body = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "UI.md" in body
        assert "scripts/29_ui.py" in body
        assert "This public release has completed independent technical review" in body
        assert "pending review" not in body

    def test_the_workflow_records_the_ui_as_implemented_and_reviewed(self):
        body = (ROOT / "BRICKAGAIN_PROJECT_WORKFLOW.md").read_text(
            encoding="utf-8")
        assert "- [x] 最小兩頁式介面。" in body
        assert "本次公開版本已完成獨立技術審查" in body
        assert "使用者明確排除" in body      # the history is not rewritten
        assert "不提供 E 解碼" in body
        assert "E 或 F-pipeline" not in body

    def test_the_status_file_does_not_claim_delivery_is_complete(self):
        status = ROOT / "PROJECT_STATUS.md"
        if not status.is_file():                      # pragma: no cover
            # One prefix, so the public tree can enumerate exactly which
            # skips are allowed. The status file is a working record of the
            # private tree and is withheld on purpose; a published copy of
            # this suite must skip here rather than fail, and the release
            # gate pins this node id so the skip cannot spread.
            pytest.skip("artifact-only: PROJECT_STATUS.md is not in this tree")
        body = status.read_text(encoding="utf-8")
        assert "最小兩頁式 UI 已通過 Codex 集中複審" in body
        assert "不是新的模型、研究結果或成效證據" in body

    def test_that_skip_is_declared_and_the_file_really_is_withheld(self):
        """Red before the fix: an unguarded read of a withheld file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "m17", ROOT / "scripts/17_public_snapshot.py")
        snapshot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snapshot)

        verdict, _ = snapshot.classify("PROJECT_STATUS.md")
        assert verdict != "include", (
            "PROJECT_STATUS.md is published now; this guard and its "
            "declaration should be removed rather than left stale")
        assert snapshot.classify("UI.md")[0] == "include"
        assert snapshot.classify("tests/test_ui.py")[0] == "include"

        mine = (ROOT / "tests/test_ui.py").read_text(encoding="utf-8")
        assert "artifact-only: PROJECT_STATUS.md" in mine
        # The release gate is the one file that verifies this boundary
        # rather than being part of what it publishes, so it is withheld too.
        # Reading it is therefore conditional -- and in the public tree there
        # is no gate to check, which is the point of it not being there.
        gate_path = ROOT / "tests/test_public_snapshot.py"
        if gate_path.is_file():
            gate = gate_path.read_text(encoding="utf-8")
            assert STATUS_NODE in gate, (
                "the release gate pins allowed skips by node id; this one is "
                "not declared, so the public tree would report an undeclared "
                "skip")

    def test_in_a_tree_without_it_that_node_skips_instead_of_erroring(
            self, tmp_path):
        """The real thing: run that one node where the file is not present.

        This is what the public snapshot is. Without the guard the node does
        not skip -- it raises FileNotFoundError, and a boundary working
        exactly as intended turns the published suite red.
        """
        import shutil
        import subprocess
        tree = tmp_path / "evidence_free"
        (tree / "tests").mkdir(parents=True)
        for name in ("src", "scripts"):
            shutil.copytree(ROOT / name, tree / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "tests/test_ui.py", tree / "tests/test_ui.py")
        assert not (tree / "PROJECT_STATUS.md").exists()

        done = subprocess.run(
            [sys.executable, "-m", "pytest", f"tests/test_ui.py::{STATUS_NODE.split('::', 1)[1]}",
             "-q", "-rs", "-p", "no:cacheprovider"],
            cwd=tree, capture_output=True, text=True, timeout=600,
            env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
        combined = done.stdout + done.stderr
        assert done.returncode == 0, combined[-3000:]
        assert "1 skipped" in combined, combined[-3000:]
        assert "artifact-only: PROJECT_STATUS.md" in combined
        assert "FileNotFoundError" not in combined

    def test_no_other_withheld_file_is_read_unguarded_by_this_suite(self):
        """Every file path this suite names must be published, or guarded.

        A published test that opens a withheld file does not skip -- it
        raises FileNotFoundError, and the public tree goes red for a boundary
        working exactly as intended.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "m17b", ROOT / "scripts/17_public_snapshot.py")
        snapshot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snapshot)
        mine = (ROOT / "tests/test_ui.py").read_text(encoding="utf-8")
        named = set(re.findall(
            r'ROOT / "([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)"', mine))
        withheld = sorted(rel for rel in named
                          if snapshot.classify(rel)[0] != "include")
        assert withheld == ["PROJECT_STATUS.md",
                            "tests/test_public_snapshot.py"], withheld
        # ...and each of the two is reached behind its own guard.
        assert "artifact-only: PROJECT_STATUS.md" in mine
        assert "gate_path.is_file()" in mine

    def test_the_public_allowlist_publishes_the_handbook(self):
        script = (ROOT / "scripts/17_public_snapshot.py").read_text(
            encoding="utf-8")
        assert '"UI.md"' in script
        assert '("data/processed/**", "processed per-record dataset")' in script


class TestCrossSiteSubmissionsAreRefused:
    """Binding loopback keeps the network out. It does nothing about the
    browser: any page the operator has open can post a form here. These are
    the two checks that do, and every one of them is red without them."""

    def test_page_one_carries_a_form_key_and_each_process_has_its_own(
            self, live, tmp_path):
        server, client = live
        _, _, body, _ = client.get("/")
        assert f'name="{ui_server.CSRF_FIELD}"' in body
        assert server.csrf_key in body
        assert len(server.csrf_key) >= 20

        other = ui_server.create_server(
            host="127.0.0.1", port=0, catalog=catalogue_file(tmp_path),
            store=ui_app.ResultStore())
        try:
            assert other.csrf_key != server.csrf_key
        finally:
            other.server_close()

    def test_a_submission_with_the_key_and_no_origin_is_accepted(self, live):
        _, client = live
        status, text, _ = client.post("/result", form())
        assert status == 200 and "結果與交付" in text

    def test_a_submission_from_this_very_origin_is_accepted(self, live):
        server, client = live
        status, text, _ = client.post(
            "/result", form(), origin=f"http://127.0.0.1:{server.port}")
        assert status == 200 and "結果與交付" in text
        status, text, _ = client.post(
            "/result", form(), origin=f"http://localhost:{server.port}")
        assert status == 200

    def test_an_opaque_origin_with_the_form_key_is_accepted(self, live):
        """Real Chrome can send ``Origin: null`` from a top-level local tab.

        The opaque origin proves nothing, so it does not bypass the form-key
        check.  With the key page one supplied, however, it must not lock out
        the operator as the previous origin-first rule did.
        """
        _, client = live
        assert ui_server.origin_verdict("null", client.port) == "opaque"
        status, text, _ = client.post("/result", form(), origin="null")
        assert status == 200
        assert "結果與交付" in text

    def test_an_opaque_origin_without_the_form_key_is_refused(self, live):
        _, client = live
        status, text, _ = client.post(
            "/result", form(), origin="null", key=None)
        assert status == 403
        assert ui_server.CSRF_FIELD in text
        assert "方法與資料來源" not in text

    @pytest.mark.parametrize("origin", [
        "https://evil.example",
        "http://evil.example",
        "file://",
        "http://127.0.0.1.evil.example",
        "http://notlocalhost",
    ])
    def test_an_external_origin_is_refused(self, live, origin):
        _, client = live
        status, text, _ = client.post("/result", form(), origin=origin)
        assert status == 403, origin
        assert "這個送出來自其他來源" in text
        assert "Traceback" not in text
        assert "方法與資料來源" not in text

    def test_another_loopback_port_is_a_different_origin(self, live):
        """A neighbour on 127.0.0.1 is not this application."""
        server, client = live
        elsewhere = 1 if server.port != 1 else 2
        status, text, _ = client.post(
            "/result", form(), origin=f"http://127.0.0.1:{elsewhere}")
        assert status == 403
        assert "這個送出來自其他來源" in text

    def test_a_submission_with_no_key_is_refused(self, live):
        _, client = live
        status, text, _ = client.post("/result", form(), key=None)
        assert status == 403
        assert ui_server.CSRF_FIELD in text
        assert "Traceback" not in text
        assert "方法與資料來源" not in text

    @pytest.mark.parametrize("bad", ["", "not-the-key", "x" * 64])
    def test_a_wrong_key_is_refused(self, live, bad):
        _, client = live
        status, text, _ = client.post("/result", form(), key=bad)
        assert status == 403
        assert "Traceback" not in text
        assert "方法與資料來源" not in text

    def test_a_key_sent_twice_is_refused_rather_than_first_wins(self, live):
        server, client = live
        status, text, _ = client.post(
            "/result", form(), key=[server.csrf_key, "second"])
        assert status == 403
        assert "方法與資料來源" not in text

    def test_the_key_is_compared_without_leaking_its_length(self):
        source = (ROOT / "src/ui/server.py").read_text(encoding="utf-8")
        assert "hmac.compare_digest" in source

    def test_a_refused_submission_never_reaches_the_delivery_payload(
            self, live, monkeypatch):
        """The refusal has to happen before any work, not after it."""
        def explode(*a, **k):                     # pragma: no cover
            raise AssertionError("a cross-site submission was executed")

        monkeypatch.setattr(ui_app, "run_request", explode)
        _, client = live
        assert client.post("/result", form(), key=None)[0] == 403
        assert client.post("/result", form(),
                           origin="https://evil.example")[0] == 403

    def test_the_page_reflects_the_servers_key_and_never_a_submitted_one(
            self, live):
        server, client = live
        status, text, _ = client.post(
            "/result", form(qty_2x4="not-a-number"))
        assert status == 400
        assert "smuggled-key" not in text
        assert server.csrf_key in text            # the form still works


class TestARefusalDoesNotPoisonTheConnection:
    """A keep-alive connection answered without reading the request body
    leaves those bytes in the socket, and the next request on that connection
    starts parsing mid-body. Each of these is red without the fix."""

    def _smuggle(self, port: int, *, content_type: str,
                 declared: int | None = None) -> bytes:
        body = (b"GET /artifact/smuggled/model.ldr HTTP/1.1\r\n"
                b"Host: 127.0.0.1:%d\r\n\r\n" % port)
        length = len(body) if declared is None else declared
        return (b"POST /result HTTP/1.1\r\n"
                b"Host: 127.0.0.1:%d\r\n"
                b"Content-Type: %s\r\n"
                b"Content-Length: %d\r\n\r\n" % (port, content_type.encode(),
                                                  length)) + body

    @staticmethod
    def _responses(stream: bytes) -> int:
        return stream.count(b"HTTP/1.")

    def test_an_unread_body_is_never_parsed_as_the_next_request(self, live):
        server, client = live
        stream = client.raw(self._smuggle(server.port,
                                          content_type="application/json"))
        assert b"415" in stream.split(b"\r\n")[0]
        assert self._responses(stream) == 1, (
            "the refused body was parsed as a second request:\n"
            + stream[:400].decode("utf-8", "replace"))
        assert b"Connection: close" in stream

    def test_the_same_holds_for_a_body_too_large_to_read(self, live):
        server, client = live
        stream = client.raw(self._smuggle(
            server.port, content_type=ui_server.FORM_MEDIA_TYPE,
            declared=ui_server.MAX_BODY_BYTES + 1))
        assert b"413" in stream.split(b"\r\n")[0]
        assert self._responses(stream) == 1
        assert b"Connection: close" in stream

    def test_the_same_holds_for_a_refused_host(self, live):
        server, client = live
        body = b"caption=x"
        blob = (b"POST /result HTTP/1.1\r\n"
                b"Host: brickagain.example\r\n"
                b"Content-Type: %s\r\n"
                b"Content-Length: %d\r\n\r\n"
                % (ui_server.FORM_MEDIA_TYPE.encode(), len(body))) + body
        stream = client.raw(blob)
        assert b"403" in stream.split(b"\r\n")[0]
        assert self._responses(stream) == 1
        assert b"Connection: close" in stream

    def test_the_same_holds_for_a_cross_origin_submission(self, live):
        server, client = live
        body = b"caption=x"
        blob = (b"POST /result HTTP/1.1\r\n"
                b"Host: 127.0.0.1:%d\r\n"
                b"Origin: https://evil.example\r\n"
                b"Content-Type: %s\r\n"
                b"Content-Length: %d\r\n\r\n"
                % (server.port, ui_server.FORM_MEDIA_TYPE.encode(),
                   len(body))) + body
        stream = client.raw(blob)
        assert b"403" in stream.split(b"\r\n")[0]
        assert self._responses(stream) == 1
        assert b"Connection: close" in stream

    def test_a_body_that_was_read_leaves_the_connection_usable(self, live):
        """The fix must not close every connection: a normal exchange keeps
        keep-alive, and two requests really do share one socket."""
        _, client = live
        conn = http.client.HTTPConnection(client.host, client.port, timeout=60)
        from urllib.parse import urlencode
        payload = urlencode(
            [(k, v) for k, values in
             {**form(), ui_server.CSRF_FIELD: [client.key]}.items()
             for v in values]).encode()
        for expected in (200, 200):
            conn.putrequest("POST", "/result", skip_host=True,
                            skip_accept_encoding=True)
            conn.putheader("Host", f"{client.host}:{client.port}")
            conn.putheader("Content-Type", ui_server.FORM_MEDIA_TYPE)
            conn.putheader("Content-Length", str(len(payload)))
            conn.endheaders()
            conn.send(payload)
            response = conn.getresponse()
            text = response.read().decode()
            assert response.status == expected
            assert "結果與交付" in text
            assert response.getheader("Connection") != "close"
        conn.close()

    def test_a_chunked_body_is_refused_rather_than_half_read(self, live):
        server, client = live
        blob = (b"POST /result HTTP/1.1\r\n"
                b"Host: 127.0.0.1:%d\r\n"
                b"Content-Type: %s\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
                b"9\r\ncaption=x\r\n0\r\n\r\n"
                % (server.port, ui_server.FORM_MEDIA_TYPE.encode()))
        stream = client.raw(blob)
        assert b"411" in stream.split(b"\r\n")[0]
        assert self._responses(stream) == 1


class TestEveryNumberIsBoundedAndAscii:
    """Unicode digits, runaway lengths and impossible quantities are input,
    not incidents: each must be a named 400, never a 500."""

    UNICODE_DIGITS = ["\u0663", "\uff13", "\u0969", "\u06f3"]

    @pytest.mark.parametrize("digit", UNICODE_DIGITS)
    def test_a_unicode_digit_quantity_is_refused(self, digit):
        with pytest.raises(ui_app.UiError, match="半形"):
            ui_app.parse_form(form(qty_2x4=digit))

    @pytest.mark.parametrize("digit", UNICODE_DIGITS)
    def test_a_unicode_digit_top_n_is_refused(self, digit):
        with pytest.raises(ui_app.UiError, match="半形"):
            ui_app.parse_form(form(top_n=digit))

    @pytest.mark.parametrize("digit", UNICODE_DIGITS)
    def test_a_unicode_digit_seed_or_time_limit_is_refused(self, digit):
        with pytest.raises(ui_app.UiError, match="半形"):
            ui_app.parse_form(form(mode="f-pipeline", seed=digit))
        with pytest.raises(ui_app.UiError, match="半形"):
            ui_app.parse_form(form(mode="f-pipeline", time_limit=digit))

    @pytest.mark.parametrize("digit", UNICODE_DIGITS)
    def test_a_unicode_digit_inside_the_stock_string_is_refused(self, digit):
        with pytest.raises(ui_app.UiError, match="半形"):
            ui_app.parse_form(form(qty_2x4=None,
                                   inventory_spec=f"2x4:{digit}"))

    @pytest.mark.parametrize("field,value", [
        ("qty_2x4", "9" * 5000),
        ("top_n", "9" * 5000),
        ("seed", "9" * 5000),
    ])
    def test_a_runaway_digit_run_is_refused_by_length(self, field, value):
        mode = "f-pipeline" if field == "seed" else "compare"
        with pytest.raises(ui_app.UiError, match="位數"):
            ui_app.parse_form(form(mode=mode, **{field: value}))

    def test_a_runaway_digit_run_inside_the_stock_string_is_refused(self):
        """Red before the fix: ``int()`` raises past ~4,300 digits, and that
        ValueError is not one of this UI's refusals."""
        with pytest.raises(ui_app.UiError, match="位數"):
            ui_app.parse_form(form(qty_2x4=None,
                                   inventory_spec="2x4:" + "9" * 50))
        # A spec long enough to hit the whole-string cap is refused there
        # instead -- also a controlled 400, and named for the right reason.
        with pytest.raises(ui_app.UiError, match="庫存字串長度"):
            ui_app.parse_form(form(qty_2x4=None,
                                   inventory_spec="2x4:" + "9" * 5000))

    def test_a_runaway_time_limit_is_refused_before_float_sees_it(self):
        with pytest.raises(ui_app.UiError, match="time limit"):
            ui_app.parse_form(form(mode="f-pipeline",
                                   time_limit="9" * 5000))

    @pytest.mark.parametrize("value", ["8001", "999999", "99999999"])
    def test_a_stock_larger_than_the_world_is_refused(self, value):
        assert ui_app.MAX_PART_COUNT == 8000
        with pytest.raises(ui_app.UiError, match="必須介於"):
            ui_app.parse_form(form(qty_2x4=value))
        with pytest.raises(ui_app.UiError, match="必須介於"):
            ui_app.parse_form(form(qty_2x4=None,
                                   inventory_spec=f"2x4:{value}"))

    def test_the_largest_usable_stock_is_still_accepted(self):
        request = ui_app.parse_form(form(qty_2x4=str(ui_app.MAX_PART_COUNT)))
        assert request.inventory == {"2x4": ui_app.MAX_PART_COUNT}

    @pytest.mark.parametrize("value", ["1e3", "0x10", "+5", " 5 5", "5,5",
                                       "٣٤", "1_0"])
    def test_only_plain_half_width_digits_are_a_number_here(self, value):
        with pytest.raises(ui_app.UiError):
            ui_app.parse_form(form(top_n=value))

    @pytest.mark.parametrize("value", ["1e1", "2,5", "２.５", "nan", "inf",
                                       "-inf", "0x2", ".5", "2."])
    def test_only_plain_half_width_decimals_are_a_time_limit(self, value):
        with pytest.raises(ui_app.UiError, match="time limit"):
            ui_app.parse_form(form(mode="f-pipeline", time_limit=value))

    def test_a_refusal_quotes_a_long_value_without_reprinting_all_of_it(self):
        with pytest.raises(ui_app.UiError) as exc:
            ui_app.parse_form(form(mode="x" * 4000))
        assert len(str(exc.value)) < 300
        assert "…" in str(exc.value)

    @pytest.mark.parametrize("fields,needle", [
        (dict(qty_2x4="\u0663"), "半形"),
        (dict(qty_2x4="9" * 5000), "位數"),
        (dict(qty_2x4="99999"), "必須介於"),
        (dict(top_n="\uff13"), "半形"),
        (dict(qty_2x4=None, inventory_spec="2x4:" + "9" * 50), "位數"),
        (dict(qty_2x4=None, inventory_spec="2x4:" + "9" * 5000),
         "庫存字串長度"),
    ])
    def test_over_http_each_one_is_a_named_400_and_never_a_500(
            self, live, fields, needle):
        _, client = live
        status, text, _ = client.post("/result", form(**fields))
        assert status == 400, text[:400]
        assert needle in text
        assert "Traceback" not in text

    def test_the_pipeline_controls_are_bounded_over_http_too(self, live):
        _, client = live
        for bad in ("\u0663", "9" * 5000, "nan", "1e3"):
            status, text, _ = client.post(
                "/result", form(mode="f-pipeline", time_limit=bad, top_n="1"))
            assert status == 400, bad
            assert "Traceback" not in text
