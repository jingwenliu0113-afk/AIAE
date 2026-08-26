"""End-to-end chains, and the failure path of each one.

These tests walk the whole way across module boundaries -- photograph to
inventory, inventory to retrieval to a grounded recommendation, inventory to
CP-SAT to colours to LDraw and build steps -- because each of those handovers
is a place where two modules can each be right and the pair still be wrong.

Every chain is tested twice: once where it succeeds, and once where it fails
for a stated reason.  The failure half is the more important one. A pipeline
that produces something plausible when it should have refused is the defect
this project is most exposed to, so the refusals are named:

* nothing recognised, and low confidence, and a class outside the eight;
* stock too short to build;
* a structure that collides, is in two pieces, or floats;
* no retrieval candidate that survives the conditions;
* the solver running out of time, which is not the same as no solution;
* a model that cannot be loaded.

No weights are loaded anywhere in this file. The embedding model is replaced by
a deterministic stand-in, and the project model is exercised through its
refusals; the one real decode is a separately labelled smoke.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

from src.assembly.order import AssemblyError
from src.assembly.order import plan as build_plan
from src.assembly.order import step_descriptions, to_ldr, write_step_previews
from src.colour.assign import AssignError, assign, parse_colour_stock
from src.data.bricks import (Brick, find_collisions, format_bricks,
                             is_connected, parse_bricks, touches_ground)
from src.delivery import pipeline
from src.delivery.pipeline import load_train_catalog, run_f_pipeline
from src.eval.scoring import score_generation
from src.retrieval.explain import explain_result, format_explanation
from src.retrieval.index import Document, VectorIndex, documents_from_catalog
from src.retrieval.nlp import extract
from src.retrieval.search import search
from src.ui import full as full_module
from src.ui.corrections import adopt, colour_stock_spec, inventory_spec
from src.ui.model_entry import ModelEntryError, decode, identity
from src.vision.classes import UNKNOWN
from src.vision.detect import counts_to_inventory, detect
from src.vision.schema import METHOD_CV, from_scores


# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------

def photo(blocks, *, width=640, height=320, ground=(243, 243, 241)):
    from PIL import Image

    image = np.full((height, width, 3), ground, dtype=np.uint8)
    for left, colour in blocks:
        image[100:220, left:left + 150] = colour
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


RED = (0xC9, 0x1A, 0x09)
BLUE = (0x00, 0x55, 0xBF)
YELLOW = (0xF2, 0xCD, 0x37)

TOWER = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 1), Brick(1, 1, 0, 0, 2)]
CAR = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]


def row(sid, oid, caption, bricks):
    return {"split": "train", "role": "control", "variant": "exact",
            "object_id": oid, "structure_id": sid, "caption": caption,
            "bricks_txt": format_bricks(bricks)}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    manifest = tmp_path / "object_splits.json"
    manifest.write_text(json.dumps({
        "meta": {"fixture": True}, "counts": {"train": 2, "val": 1},
        "objects": {"o-car": "train", "o-tower": "train",
                    "o-held": "val"},
        "structures": {"s-car": "o-car", "s-tower": "o-tower",
                       "s-held": "o-held"},
    }, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(pipeline, "FROZEN_SPLIT_MANIFEST_SHA256",
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    path = tmp_path / "e2e_train.jsonl"
    path.write_text(
        json.dumps(row("s-car", "o-car", "a compact car with a flat top",
                       CAR)) + "\n"
        + json.dumps(row("s-tower", "o-tower", "a tiny narrow tower",
                         TOWER)) + "\n", encoding="utf-8")
    return load_train_catalog(path)


def stand_in_vector(text, dimension=8):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest[:dimension], dtype=np.uint8).astype(np.float32)
    raw = raw - raw.mean()
    return (raw / (float(np.linalg.norm(raw)) or 1.0)).astype(np.float32)


class StandIn:
    device = "cpu"
    dimension = 8

    def identity_digest(self):
        return "stand-in"

    def identity(self):
        return {"repo": "stand-in", "revision": "0"}


@pytest.fixture
def index(catalog, monkeypatch):
    docs = documents_from_catalog(catalog)
    vectors = np.stack([stand_in_vector(d.embed_text()) for d in docs])
    import src.retrieval.embed as embed_module

    monkeypatch.setattr(
        embed_module, "embed",
        lambda _e, texts, *, kind, batch=32: np.stack(
            [stand_in_vector(text) for text in texts]))
    return VectorIndex(
        documents=docs, vectors=vectors,
        embedding={"repo": "stand-in", "revision": "0"},
        catalog_sha256=catalog.sha256,
        split_manifest_sha256=catalog.split_manifest_sha256,
        identity_digest="stand-in", build_device="cpu")


# --------------------------------------------------------------------------
# 1. a single-brick photograph to Top-3 to a correction to an inventory
# --------------------------------------------------------------------------

class TestSingleBrickPhotoToInventory:
    def test_the_whole_chain_runs(self):
        analysis = full_module.analyse_photo(
            photo([(240, RED)]), mode=full_module.PHOTO_SINGLE)
        assert analysis.found == 1
        item = analysis.items[0]
        assert len(item.predicted_top3) == 3
        corrected = full_module.apply_corrections(
            analysis.items, {0: {"part": "2x4", "count": 1,
                                 "colour": "red"}},
            width=analysis.width, height=analysis.height)
        stock = adopt(corrected)
        assert stock.parts == {"2x4": 1}
        assert inventory_spec(stock.parts) == "2x4:1"

    def test_top3_is_offered_for_a_person_to_choose_from(self):
        analysis = full_module.analyse_photo(
            photo([(240, BLUE)]), mode=full_module.PHOTO_SINGLE)
        top3 = analysis.items[0].predicted_top3
        assert len(set(top3)) == 3

    def test_nothing_recognised_is_a_stated_outcome(self):
        """An empty frame: no boxes, no stock, and a reason on the record."""
        blank = photo([])
        analysis = full_module.analyse_photo(blank,
                                             mode=full_module.PHOTO_MULTI)
        assert analysis.found == 0
        assert adopt(analysis.items).parts == {}
        assert "empty_reason" in analysis.diagnostics

    def test_a_low_confidence_item_is_not_counted_as_stock(self):
        analysis = full_module.analyse_photo(
            photo([(240, RED)]), mode=full_module.PHOTO_SINGLE)
        low = [item for item in analysis.items
               if item.predicted_part == UNKNOWN]
        if low:
            stock = adopt(analysis.items)
            assert all(index in stock.unresolved
                       for index in [item.index for item in low])

    def test_a_class_outside_the_eight_cannot_be_adopted(self):
        analysis = full_module.analyse_photo(
            photo([(240, RED)]), mode=full_module.PHOTO_SINGLE)
        from src.ui.app import UiError

        with pytest.raises(UiError, match="2x8|not one of"):
            full_module.apply_corrections(
                analysis.items, {0: {"part": "2x8"}},
                width=analysis.width, height=analysis.height)


# --------------------------------------------------------------------------
# 2. a multi-brick photograph to boxes to per-class counts to a correction
# --------------------------------------------------------------------------

class TestMultiBrickPhotoToCounts:
    def test_three_bricks_become_three_boxes_and_a_count(self):
        analysis = full_module.analyse_photo(
            photo([(30, RED), (240, BLUE), (450, YELLOW)]),
            mode=full_module.PHOTO_MULTI)
        assert analysis.found == 3
        corrected = full_module.apply_corrections(
            analysis.items,
            {0: {"part": "2x4", "colour": "red"},
             1: {"part": "2x4", "colour": "blue"},
             2: {"part": "1x2", "colour": "yellow"}},
            width=analysis.width, height=analysis.height)
        stock = adopt(corrected)
        assert stock.parts == {"1x2": 1, "2x4": 2}
        assert stock.colour_parts == {("1x2", "yellow"): 1,
                                      ("2x4", "blue"): 1,
                                      ("2x4", "red"): 1}
        assert stock.fully_coloured

    def test_the_counts_by_class_come_from_the_detector(self):
        image = photo([(30, RED), (240, BLUE)])
        from src.vision.preprocess import decode_image

        loaded = decode_image(image)

        def as_2x4(_crop):
            scores = [0.0] * 8
            from src.vision.classes import CLASS_ORDER

            scores[CLASS_ORDER.index("2x4")] = 0.95
            return from_scores(METHOD_CV, scores)

        result = detect(loaded.rgb, classify=as_2x4)
        stock, caveats = counts_to_inventory(result)
        assert stock == {"2x4": 2}
        assert caveats["needs_review"] is False

    def test_an_unnamed_box_keeps_the_proposal_out_of_the_stock(self):
        from src.vision.preprocess import decode_image

        loaded = decode_image(photo([(30, RED), (240, BLUE)]))
        result = detect(loaded.rgb,
                        classify=lambda _c: from_scores(METHOD_CV,
                                                        [0.125] * 8))
        stock, caveats = counts_to_inventory(result)
        assert stock == {}
        assert caveats["unidentified_boxes"] == 2
        assert caveats["needs_review"] is True


# --------------------------------------------------------------------------
# 3. an image-derived inventory to Chinese retrieval to a grounded answer
# --------------------------------------------------------------------------

class TestPhotoInventoryToRetrieval:
    def _stock_from_photo(self):
        analysis = full_module.analyse_photo(
            photo([(30, RED), (240, BLUE)]), mode=full_module.PHOTO_MULTI)
        corrected = full_module.apply_corrections(
            analysis.items,
            {0: {"part": "2x4", "count": 2, "colour": "red"},
             1: {"part": "1x1", "count": 4, "colour": "blue"}},
            width=analysis.width, height=analysis.height)
        return adopt(corrected)

    def test_a_chinese_request_reaches_a_grounded_recommendation(self,
                                                                index,
                                                                catalog):
        stock = self._stock_from_photo()
        result = search(index, catalog, StandIn(), extract("我想做一座小塔"),
                        stock.parts, top_n=2)
        body = explain_result(result, stock.parts)
        assert body["status"] in ("buildable_existing_work_found",
                                  "no_buildable_existing_work_in_retrieved_set")
        assert body["candidates"]
        for candidate in body["candidates"]:
            assert candidate["verdict"] in ("buildable", "not_buildable")

    def test_a_buildable_answer_says_so_only_when_it_is(self, index, catalog):
        stock = self._stock_from_photo()
        result = search(index, catalog, StandIn(), extract("一座小塔"),
                        stock.parts, top_n=2)
        for candidate in result.ranked:
            text = " ".join(
                explain_candidate_text(candidate))
            if candidate.buildable:
                assert "可以組" in text
            else:
                assert "不能組" in text

    def test_a_short_stock_yields_no_recommendation_and_says_why(self,
                                                                 index,
                                                                 catalog):
        result = search(index, catalog, StandIn(), extract("一座小塔"),
                        {"1x1": 1}, top_n=2)
        assert result.selected is None
        body = explain_result(result, {"1x1": 1})
        assert "沒有推薦" in body["selection"]

    def test_a_condition_nothing_satisfies_gives_no_candidate(self, index,
                                                             catalog):
        result = search(index, catalog, StandIn(), extract("1 顆以內的塔"),
                        {"1x1": 8}, top_n=2)
        assert result.status == "no_semantic_candidate"
        body = explain_result(result, {"1x1": 8})
        assert "沒有任何語意候選" in body["selection"]

    def test_a_held_out_object_is_never_retrieved(self, index, catalog):
        """Train-only: the index has no held-out object to return."""
        assert all(document.object_id in {"o-car", "o-tower"}
                   for document in index.documents)
        result = search(index, catalog, StandIn(), extract("一台車"),
                        {"2x4": 4}, top_n=5,
                        exclude_object_id="o-car")
        assert all(candidate.item.object_id != "o-car"
                   for candidate in result.ranked)
        assert result.excluded_same_object == 1

    def test_the_explanation_never_prints_a_dataset_identifier(self, index,
                                                              catalog):
        result = search(index, catalog, StandIn(), extract("一台車"),
                        {"2x4": 4}, top_n=2)
        text = format_explanation(explain_result(result, {"2x4": 4}))
        for identifier in ("o-car", "o-tower", "s-car", "s-tower"):
            assert identifier not in text


def explain_candidate_text(candidate):
    from src.retrieval.explain import explain_candidate

    return explain_candidate(candidate)["sentences"]


# --------------------------------------------------------------------------
# 4. an inventory to CP-SAT to the checker to colours to LDraw and previews
# --------------------------------------------------------------------------

class TestInventoryToPipelineToColoursToOutputs:
    def test_the_whole_chain_produces_one_consistent_structure(self, catalog,
                                                              tmp_path):
        outcome = run_f_pipeline(catalog, "a tiny narrow tower",
                                {"1x1": 6}, top_n=2, time_limit=5.0, seed=0)
        assert outcome.selected is not None
        bricks = list(outcome.selected.bricks)

        # the checker
        scored = score_generation(format_bricks(bricks), inventory={"1x1": 6},
                                  n_tokens=0, termination=None)
        assert scored["checks"]["collision_free"]
        assert scored["checks"]["stud_only_connected"]
        assert scored["checks"]["touches_ground"]

        # colours
        assignment = assign(bricks, parse_colour_stock("1x1:red:6"))
        assert len(assignment.bricks) == len(bricks)
        assignment.check_within_stock()

        # build order, LDraw and previews, all from the same brick list
        plan = build_plan(bricks, max_per_step=1)
        assert plan.ready
        ldraw = to_ldr(plan, colours=assignment.colours())
        assert ldraw.count("0 STEP") == plan.n_steps
        written = write_step_previews(plan, tmp_path,
                                     colours=assignment.colours())
        assert len(written) == plan.n_steps

    def test_the_ldraw_the_steps_and_the_descriptions_are_one_structure(
            self, catalog):
        outcome = run_f_pipeline(catalog, "a tiny narrow tower",
                                {"1x1": 6}, top_n=2, time_limit=5.0, seed=0)
        bricks = list(outcome.selected.bricks)
        plan = build_plan(bricks)
        lines = step_descriptions(plan)
        assert len(lines) == plan.n_steps

        # every LDraw type-1 line is one brick of the plan, in build order
        placed = [line for line in to_ldr(plan).splitlines()
                  if line.startswith("1 ")]
        assert len(placed) == len(bricks)
        for index, brick_index in enumerate(plan.order):
            brick = plan.bricks[brick_index]
            # the LDraw y coordinate is the layer, negated and scaled
            assert placed[index].split()[3] == str(brick.z * -24)
            # and the description names the same coordinates
            assert (f"({brick.x},{brick.y},{brick.z})"
                    in lines[index]) or plan.max_per_step > 1

    def test_a_stock_too_short_for_the_shape_is_refused_by_name(self,
                                                                catalog):
        outcome = run_f_pipeline(catalog, "a tiny narrow tower",
                                {"1x1": 6}, top_n=2, time_limit=5.0, seed=0)
        bricks = list(outcome.selected.bricks)
        with pytest.raises(AssignError, match="needs"):
            assign(bricks, parse_colour_stock("1x1:red:1"))

    def test_a_colour_stock_can_never_be_overdrawn(self, catalog):
        outcome = run_f_pipeline(catalog, "a tiny narrow tower",
                                {"1x1": 6}, top_n=2, time_limit=5.0, seed=0)
        bricks = list(outcome.selected.bricks)
        stock = parse_colour_stock("1x1:red:2,1x1:blue:2,1x1:yellow:2")
        assignment = assign(bricks, stock, preferences=["blue"])
        for key, used in assignment.used.items():
            assert used <= stock[key]

    def test_a_solver_timeout_is_distinguishable_from_no_solution(self,
                                                                  catalog):
        outcome = run_f_pipeline(catalog, "a compact car with a flat top",
                                {"1x1": 1}, top_n=1, time_limit=0.001,
                                seed=0)
        statuses = {attempt.solver_status for attempt in outcome.attempts}
        failures = {attempt.failure for attempt in outcome.attempts}
        assert outcome.selected is None
        assert statuses <= {"UNKNOWN", "INFEASIBLE"}
        if "UNKNOWN" in statuses:
            assert any("timeout" in (f or "") for f in failures)
        else:
            assert any("no tiling" in (f or "") for f in failures)


# --------------------------------------------------------------------------
# 5. the structures the checker has to reject
# --------------------------------------------------------------------------

class TestTheCheckerRejectsWhatItShould:
    def test_a_collision_is_caught_and_stops_delivery(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (1,0,0)")
        assert find_collisions(bricks)
        scored = score_generation(format_bricks(bricks),
                                  inventory={"2x4": 2}, n_tokens=0,
                                  termination=None)
        assert scored["checks"]["collision_free"] is False

    def test_two_pieces_are_caught(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (10,10,0)")
        assert not is_connected(bricks)
        scored = score_generation(format_bricks(bricks),
                                  inventory={"2x4": 2}, n_tokens=0,
                                  termination=None)
        assert scored["checks"]["stud_only_connected"] is False

    def test_a_structure_off_the_ground_is_caught(self):
        bricks = parse_bricks("2x4 (0,0,3)\n2x4 (0,0,4)")
        assert not touches_ground(bricks)
        scored = score_generation(format_bricks(bricks),
                                  inventory={"2x4": 2}, n_tokens=0,
                                  termination=None)
        assert scored["checks"]["touches_ground"] is False

    def test_an_overdrawn_inventory_is_caught(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)")
        scored = score_generation(format_bricks(bricks),
                                  inventory={"2x4": 2}, n_tokens=0,
                                  termination=None)
        assert scored["checks"]["inventory_valid"] is False

    def test_a_floating_brick_cannot_be_ordered(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)\n1x1 (10,10,3)")
        with pytest.raises(AssemblyError):
            build_plan(bricks)

    def test_an_unorderable_structure_names_the_brick(self):
        bricks = parse_bricks("1x1 (0,0,0)\n1x1 (5,5,2)")
        with pytest.raises(AssemblyError, match="indices"):
            build_plan(bricks)


# --------------------------------------------------------------------------
# 6. the project model entry, through its refusals
# --------------------------------------------------------------------------

class TestTheProjectModelEntryRefusals:
    def test_a_tree_with_no_pointer_is_refused_with_a_reason(self, tmp_path):
        with pytest.raises(ModelEntryError, match="not published"):
            identity(root=tmp_path)

    def test_a_pointer_of_the_wrong_kind_is_refused(self, tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs/project_model.json").write_text(
            json.dumps({"kind": "something_else"}), encoding="utf-8")
        with pytest.raises(ModelEntryError, match="does not declare itself"):
            identity(root=tmp_path)

    def test_a_pointer_that_is_not_json_is_refused(self, tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs/project_model.json").write_text(
            "{ not json", encoding="utf-8")
        with pytest.raises(ModelEntryError, match="not valid JSON"):
            identity(root=tmp_path)

    def test_a_pointer_whose_adapter_is_absent_reports_problems(self,
                                                                tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs/project_model.json").write_text(json.dumps({
            "kind": "project_model", "model": "fake",
            "adapter": {"path": "runs/absent/adapter", "files": {}},
        }), encoding="utf-8")
        who = identity(root=tmp_path)
        assert not who.verified
        assert who.problems

    def test_decoding_with_an_unverified_pointer_loads_no_weights(self,
                                                                  tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs/project_model.json").write_text(json.dumps({
            "kind": "project_model", "model": "fake",
            "adapter": {"path": "runs/absent/adapter", "files": {}},
        }), encoding="utf-8")
        with pytest.raises(ModelEntryError, match="no weights were loaded"):
            decode("a tower", {"2x4": 2}, root=tmp_path)

    def test_connectivity_without_the_gate_is_refused(self):
        with pytest.raises(ModelEntryError, match="was not asked for"):
            decode("a tower", {"2x4": 2}, placement=False,
                   connectivity="any")

    def test_an_unknown_connectivity_mode_is_refused(self):
        with pytest.raises(ModelEntryError, match="not one of"):
            decode("a tower", {"2x4": 2}, placement=True,
                   connectivity="sideways")

    @pytest.mark.parametrize("temperature", [0, -1, 99])
    def test_a_bad_temperature_is_refused(self, temperature):
        with pytest.raises(ModelEntryError, match="temperature"):
            decode("a tower", {"2x4": 2}, temperature=temperature)

    def test_the_real_pointer_verifies_in_this_tree(self):
        """The archived model is still exactly what the record says it is.

        The pointer and the adapter it names live under ``runs/``, which is not
        published: a public checkout has neither, so this node skips there
        rather than failing. The skip is declared by node id in the release
        gate, so it cannot spread to anything behavioural.
        """
        from pathlib import Path

        if not (Path(__file__).resolve().parents[1]
                / "runs/project_model.json").is_file():   # pragma: no cover
            pytest.skip("artifact-only: runs/project_model.json is not in "
                        "this tree")
        who = identity()
        assert who.name == "final_H2"
        assert who.verified, who.problems
        assert len(who.adapter_digest) == 64


# --------------------------------------------------------------------------
# 7. the invariants the whole thing rests on
# --------------------------------------------------------------------------

class TestInvariants:
    def test_the_same_structure_and_stock_always_give_the_same_file(self,
                                                                    catalog):
        outcome = run_f_pipeline(catalog, "a tiny narrow tower",
                                {"1x1": 6}, top_n=2, time_limit=5.0, seed=0)
        bricks = list(outcome.selected.bricks)
        stock = parse_colour_stock("1x1:red:3,1x1:blue:3")
        first = to_ldr(build_plan(bricks),
                       colours=assign(bricks, stock).colours())
        again = to_ldr(build_plan(bricks),
                       colours=assign(bricks, stock).colours())
        assert first == again

    def test_a_rotated_spelling_is_one_inventory_item_all_the_way_through(
            self):
        bricks = parse_bricks("4x1 (0,0,0)\n1x4 (0,0,1)")
        scored = score_generation(format_bricks(bricks),
                                  inventory={"1x4": 2}, n_tokens=0,
                                  termination=None)
        assert scored["checks"]["inventory_valid"]
        assignment = assign(bricks, parse_colour_stock("1x4:red:2"))
        assert len(assignment.bricks) == 2

    def test_the_preview_refuses_a_structure_it_cannot_draw(self):
        from src.rendering.preview import PreviewError, write_preview

        with pytest.raises(PreviewError, match="at least one"):
            write_preview("/dev/null/x.png", [])

    def test_the_ldraw_writer_and_the_step_writer_agree_on_the_bricks(self):
        from src.rendering.ldr import to_ldr as plain

        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)")
        plan = build_plan(bricks)
        stepped = [line for line in to_ldr(plan).splitlines()
                   if line.startswith("1 ")]
        flat = [line for line in plain(bricks).splitlines()
                if line.startswith("1 ")]
        assert sorted(stepped) == sorted(flat)


# --------------------------------------------------------------------------
# Round 49: the finished result's consistency claim, checked against the files
#
# ``Finished.as_dict`` said the file, the preview and the step images came
# from "one brick list and one colour assignment" whatever the caller passed.
# With no colour stock there is no assignment, and the images were a per-shape
# legend rather than the file's colours -- so the claim was true of the
# structure and false of the colours.  These read the artefacts.
# --------------------------------------------------------------------------

class TestTheFinishedResultSaysWhichColoursItDrew:
    def structure(self, catalog):
        outcome = run_f_pipeline(catalog, "a tiny narrow tower",
                                 {"1x1": 6}, top_n=2, time_limit=5.0, seed=0)
        assert outcome.selected is not None
        return outcome.selected

    def finish_it(self, selected, **kw):
        result = full_module.FullResult(
            method="cp-sat", provenance={}, status="ok",
            text=format_bricks(selected.bricks),
            report={"delivery": {"static_delivery_ready": True}})
        assert result.ready
        return full_module.finish(result, **kw)

    def test_with_a_colour_stock_the_claim_names_the_assignment(self, catalog):
        selected = self.structure(catalog)
        finished = self.finish_it(selected, colour_stock="1x1:red:6",
                                  step_previews=False)
        body = finished.as_dict()
        assert finished.assignment is not None
        assert body["colour_source"] == "assignment"
        assert "same colours" in body["same_structure"]

    def test_without_a_colour_stock_the_claim_says_colours_differ(self,
                                                                  catalog):
        selected = self.structure(catalog)
        finished = self.finish_it(selected, step_previews=False)
        body = finished.as_dict()
        assert finished.assignment is None
        assert body["colour_source"] == "part-key"
        assert "not the same colours" in body["same_structure"]
        assert "per-shape legend" in body["same_structure"]

    def test_the_written_preview_really_carries_the_assigned_colour(
            self, catalog, monkeypatch, tmp_path):
        """The preview bytes, read back, must hold the palette's red."""
        from src.colour.palette import BY_ID
        from src.rendering.preview import PART_COLOURS

        import re

        import src.rendering.preview as preview_module

        seen = {}
        original = preview_module.write_preview

        def capture(path, bricks, **kw):
            """Write the same call to an SVG too, so the fills are readable."""
            svg = tmp_path / "capture.svg"
            original(svg, bricks, **kw)
            seen["fills"] = {
                value.lower() for value in re.findall(
                    r"fill:\s*(#[0-9a-fA-F]{6})",
                    svg.read_text(encoding="utf-8"))}
            return original(path, bricks, **kw)

        monkeypatch.setattr(preview_module, "write_preview", capture)
        selected = self.structure(catalog)
        self.finish_it(selected, colour_stock="1x1:red:6",
                       step_previews=False)
        assert BY_ID["red"].hex.lower() in seen["fills"]
        assert PART_COLOURS["1x1"].lower() not in seen["fills"]
