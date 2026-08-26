"""Minimum delivery paths: train-only comparison, F-pipeline and files.

All catalogues here are synthetic train fixtures.  No model, network, GPU,
frozen evaluation case or formal metric is involved.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shlex
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import Brick, format_bricks
from src.data.retile import RetileResult
from src.delivery import pipeline
from src.delivery.pipeline import (DeliveryError, compare_existing,
                                   lexical_similarity, load_train_catalog,
                                   retrieve, run_f_pipeline, selected_text)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/27_delivery.py"


def row(sid: str, oid: str, caption: str, bricks: list[Brick], **kw) -> dict:
    return {
        "split": kw.get("split", "train"),
        "role": kw.get("role", "control"),
        "variant": kw.get("variant", "exact"),
        "object_id": oid,
        "structure_id": sid,
        "caption": caption,
        "bricks_txt": format_bricks(bricks),
        # Deliberately wrong: the loader must derive stock from geometry.
        "used": {"1x1": 999},
    }


STACK = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]
SMALL = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 1)]
FLOATING = [Brick(1, 1, 0, 0, 1), Brick(1, 1, 0, 0, 2)]
DISCONNECTED = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 2, 2, 0)]
EXAMPLE_CAPTION = (
    "This train features a streamlined, elongated rectangular body composed "
    "of uniformly arranged bricks. The top is flat with evenly spaced small "
    "cylindrical protrusions, providing a cohesive and structured appearance."
)
EXAMPLE_BRICKS = [
    Brick(2, 6, 0, 2, 0), Brick(2, 6, 0, 8, 0),
    Brick(2, 6, 0, 14, 0), Brick(2, 6, 0, 3, 1),
    Brick(2, 4, 0, 9, 1), Brick(2, 6, 0, 13, 1),
    Brick(2, 1, 0, 19, 1),
]


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
    """Give synthetic rows a split contract without weakening production."""
    objects = {
        "o-car": "train", "o-tower": "train", "o1": "train",
        "o2": "train", "o3": "train", "o": "train",
        "only-object": "train", "o-floating": "train",
        "o-disconnected": "train", "o-example": "train",
        "val-object": "val",
    }
    structures = {
        "s-car": "o-car", "s-tower": "o-tower", "s1": "o1",
        "s2": "o2", "s3": "o3", "same": "o1", "s": "o",
        "s-only": "only-object", "s-floating": "o-floating",
        "s-disconnected": "o-disconnected", "s-example": "o-example",
        "s-val": "val-object",
    }
    manifest = tmp_path / "object_splits.json"
    manifest.write_text(json.dumps({
        "meta": {"fixture": True},
        "counts": {"train": 10, "val": 1},
        "objects": objects,
        "structures": structures,
    }, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(pipeline, "FROZEN_SPLIT_MANIFEST_SHA256", digest)
    return manifest


def documented_command(marker: str) -> list[str]:
    body = (ROOT / "DELIVERY.md").read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- {re.escape(marker)} -->\s*```bash\s*(.*?)```",
        body, flags=re.DOTALL)
    assert match, f"missing documented command marker {marker}"
    command = match.group(1).replace("\\\n", " ").strip()
    argv = shlex.split(command)
    script_at = argv.index("scripts/27_delivery.py")
    return argv[script_at + 1:]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("m27", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTheTrainOnlyBoundary:
    def test_the_filename_and_every_row_must_both_say_train(self, tmp_path):
        wrong_name = catalogue_file(tmp_path, name="catalogue.jsonl")
        with pytest.raises(DeliveryError, match="_train.jsonl"):
            load_train_catalog(wrong_name)

        wrong_row = catalogue_file(
            tmp_path, [row("s", "o", "x", SMALL, split="held-out")])
        with pytest.raises(DeliveryError, match="train-only"):
            load_train_catalog(wrong_row)

    def test_invalid_json_and_missing_fields_fail_closed(self, tmp_path):
        broken = tmp_path / "broken_train.jsonl"
        broken.write_text("{no\n", encoding="utf-8")
        with pytest.raises(DeliveryError, match="invalid JSON"):
            load_train_catalog(broken)

        missing = catalogue_file(
            tmp_path, [{"split": "train", "role": "control",
                        "variant": "exact"}])
        with pytest.raises(DeliveryError, match="object_id"):
            load_train_catalog(missing)

    @pytest.mark.parametrize("field", ["caption", "bricks_txt"])
    @pytest.mark.parametrize(
        "value", [None, 0, 1, [], {}, "", "  \t\n"])
    def test_canonical_text_fields_must_be_non_empty_strings(
            self, tmp_path, field, value):
        item = row("s", "o", "x", SMALL)
        item[field] = value
        with pytest.raises(
                DeliveryError, match=rf"field {field} must be a non-empty string"):
            load_train_catalog(catalogue_file(tmp_path, [item]))

    @pytest.mark.parametrize("value", [[], 0, "row", None])
    def test_a_non_object_json_row_is_a_controlled_refusal(self, tmp_path,
                                                            value):
        path = tmp_path / "non_object_train.jsonl"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with pytest.raises(DeliveryError, match="must be a JSON object"):
            load_train_catalog(path)

    def test_object_and_structure_ids_must_match_the_frozen_manifest(
            self, tmp_path):
        unknown = catalogue_file(
            tmp_path, [row("s", "unknown-object", "x", SMALL)])
        with pytest.raises(DeliveryError, match="object_id absent"):
            load_train_catalog(unknown)

        held_out = catalogue_file(
            tmp_path, [row("s-val", "val-object", "x", SMALL)])
        with pytest.raises(DeliveryError, match="assigned to 'val'"):
            load_train_catalog(held_out)

        wrong_owner = catalogue_file(
            tmp_path, [row("s-car", "o-tower", "x", SMALL)])
        with pytest.raises(DeliveryError, match="belongs to another"):
            load_train_catalog(wrong_owner)

    def test_the_frozen_manifest_digest_is_fail_closed(
            self, tmp_path, frozen_split_fixture):
        frozen_split_fixture.write_text("{}", encoding="utf-8")
        with pytest.raises(DeliveryError, match="manifest digest differs"):
            load_train_catalog(catalogue_file(tmp_path))

    def test_only_one_canonical_row_per_geometry_enters_the_index(self,
                                                                  tmp_path):
        rows = [
            row("s1", "o1", "one", STACK),
            row("s2", "o2", "two", SMALL, role="counterfactual"),
            row("s3", "o3", "three", SMALL, variant="loose"),
        ]
        catalog = load_train_catalog(catalogue_file(tmp_path, rows))
        assert len(catalog.items) == 1
        assert catalog.items[0].required == {"2x4": 2}
        assert "o1" not in repr(catalog.items[0])
        assert "s1" not in catalog.items[0].catalog_id

    def test_duplicate_canonical_structures_are_refused(self, tmp_path):
        rows = [row("same", "o1", "one", STACK),
                row("same", "o1", "two", SMALL)]
        with pytest.raises(DeliveryError, match="duplicate"):
            load_train_catalog(catalogue_file(tmp_path, rows))

    def test_invalid_geometry_never_becomes_a_recommendation(self, tmp_path):
        overlap = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 0)]
        with pytest.raises(DeliveryError, match="not valid geometry"):
            load_train_catalog(catalogue_file(
                tmp_path, [row("s", "o", "x", overlap)]))

    def test_a_connected_but_floating_catalogue_item_is_refused(self,
                                                                 tmp_path):
        with pytest.raises(DeliveryError, match="does not touch ground"):
            load_train_catalog(catalogue_file(
                tmp_path, [row("s-floating", "o-floating", "x",
                               FLOATING)]))

    def test_a_grounded_but_disconnected_catalogue_item_is_refused(
            self, tmp_path):
        with pytest.raises(DeliveryError, match="not one component"):
            load_train_catalog(catalogue_file(
                tmp_path, [row("s-disconnected", "o-disconnected", "x",
                               DISCONNECTED)]))

    def test_same_object_exclusion_is_enforced_before_ranking(self, tmp_path):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        got, excluded = retrieve(
            catalog, "red car", {"2x4": 2}, top_n=5,
            exclude_object_id="o-car")
        assert excluded == 1
        assert all(c.item.object_id != "o-car" for c in got)

    def test_removing_the_whole_catalogue_is_a_refusal(self, tmp_path):
        path = catalogue_file(
            tmp_path, [row("s-only", "only-object", "x", STACK)])
        with pytest.raises(DeliveryError, match="whole catalogue"):
            retrieve(load_train_catalog(path), "x", {"2x4": 2},
                     exclude_object_id="only-object")


class TestExistingWorkComparison:
    def test_lexical_similarity_is_deterministic_and_symmetric(self):
        close = lexical_similarity("a compact red car", "red compact car")
        far = lexical_similarity("a compact red car", "yellow bookshelf")
        assert close > far
        assert close == lexical_similarity("red compact car",
                                           "a compact red car")
        assert lexical_similarity("紅色小車", "一台紅色小車") > 0

    def test_it_retrieves_then_inventory_reranks_with_missing_evidence(
            self, tmp_path):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        result = compare_existing(
            catalog, "compact red car", {"1x1": 2}, top_n=2)
        assert result.retrieved[0].item.caption == "a compact red car"
        assert result.retrieved[0].missing == {"2x4": 2}
        assert result.ranked[0].item.caption == "a tiny yellow tower"
        assert result.ranked[0].buildable is True
        assert result.selected is result.ranked[0]

    def test_no_buildable_candidate_is_reported_not_hidden(self, tmp_path):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        result = compare_existing(catalog, "red car", {"1x1": 1}, top_n=2)
        assert result.selected is None
        assert all(c.missing_total > 0 for c in result.ranked)
        assert selected_text(result) is None

    @pytest.mark.parametrize("field", ["touches_ground", "connected"])
    def test_compare_requires_both_static_structure_conditions(
            self, tmp_path, field):
        catalog = load_train_catalog(catalogue_file(
            tmp_path, [row("s", "o", "x", STACK)]))
        changed = replace(catalog.items[0], **{field: False})
        result = compare_existing(
            replace(catalog, items=(changed,)), "x", {"2x4": 2}, top_n=1)
        assert result.ranked[0].missing == {}
        assert result.ranked[0].buildable is False
        assert result.selected is None

    def test_bad_top_n_caption_and_inventory_are_refused(self, tmp_path):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        for caption, inventory, top_n in (("", {"1x1": 1}, 1),
                                           ("x", {}, 1),
                                           ("x", {"1x1": 1}, 0)):
            with pytest.raises(DeliveryError):
                retrieve(catalog, caption, inventory, top_n=top_n)


class TestTheMinimumFPipeline:
    def test_the_real_solver_can_return_a_verified_connected_build(self,
                                                                   tmp_path):
        catalog = load_train_catalog(catalogue_file(
            tmp_path, [row("s", "o", "red car", STACK)]))
        result = run_f_pipeline(
            catalog, "red car", {"2x4": 2}, top_n=1, time_limit=2)
        assert result.status == "success"
        attempt = result.attempts[0]
        assert attempt.solver_status in ("OPTIMAL", "FEASIBLE")
        assert attempt.delivery_ready is True
        assert selected_text(result) == format_bricks(list(attempt.bricks))

    def test_infeasibility_is_distinct_from_a_bad_checker_result(self,
                                                                  tmp_path):
        catalog = load_train_catalog(catalogue_file(
            tmp_path, [row("s", "o", "red car", STACK)]))
        result = run_f_pipeline(
            catalog, "red car", {"1x1": 1}, top_n=1, time_limit=2)
        assert result.status == "no_valid_build"
        assert result.attempts[0].solver_returned_tiling is False
        assert result.attempts[0].solver_status == "INFEASIBLE"

    def test_timeout_is_named_as_timeout(self, tmp_path, monkeypatch):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        monkeypatch.setattr(
            pipeline, "retile",
            lambda *a, **k: RetileResult(None, "UNKNOWN", 0.01, 12))
        result = run_f_pipeline(
            catalog, "red car", {"2x4": 2}, top_n=1, time_limit=0.1)
        assert result.attempts[0].failure == "solver timeout"
        assert result.status == "no_valid_build"

    def test_a_disconnected_tiling_is_not_published_as_ready(
            self, tmp_path, monkeypatch):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        monkeypatch.setattr(
            pipeline, "occupancy_of",
            lambda bricks: {cell for b in DISCONNECTED for cell in b.cells})
        monkeypatch.setattr(
            pipeline, "retile",
            lambda *a, **k: RetileResult(DISCONNECTED, "OPTIMAL", 0.01, 2))
        result = run_f_pipeline(
            catalog, "red car", {"1x1": 2}, top_n=1, time_limit=1)
        assert result.status == "tiling_found_but_not_delivery_ready"
        assert result.attempts[0].connected is False
        assert "connectivity" in result.attempts[0].failure

    def test_a_connected_but_floating_tiling_is_not_published_as_ready(
            self, tmp_path, monkeypatch):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        monkeypatch.setattr(
            pipeline, "occupancy_of",
            lambda bricks: {cell for b in FLOATING for cell in b.cells})
        monkeypatch.setattr(
            pipeline, "retile",
            lambda *a, **k: RetileResult(FLOATING, "OPTIMAL", 0.01, 2))
        result = run_f_pipeline(
            catalog, "red car", {"1x1": 2}, top_n=1, time_limit=1)
        attempt = result.attempts[0]
        assert result.status == "tiling_found_but_not_delivery_ready"
        assert attempt.connected is True
        assert attempt.touches_ground is False
        assert "touch ground" in attempt.failure

    def test_invalid_solver_controls_are_refused(self, tmp_path):
        catalog = load_train_catalog(catalogue_file(tmp_path))
        for kw in ({"time_limit": 0}, {"time_limit": float("nan")},
                   {"time_limit": float("inf")},
                   {"time_limit": float("-inf")}, {"seed": -1},
                   {"seed": True}):
            with pytest.raises(DeliveryError):
                run_f_pipeline(catalog, "x", {"2x4": 2}, **kw)


class TestTheDeliveryCommand:
    def test_compare_goes_from_manual_stock_to_files(self, cli, tmp_path,
                                                     capsys):
        catalog = catalogue_file(tmp_path)
        ldr, png = tmp_path / "picked.ldr", tmp_path / "picked.png"
        code = cli.main([
            "--mode", "compare", "--caption", "red car",
            "--inventory", "2x4:2", "--catalog", str(catalog),
            "--top-n", "1", "--ldr", str(ldr), "--preview", str(png)])
        assert code == cli.EXIT_OK
        ldraw = ldr.read_text(encoding="utf-8")
        assert "3001.DAT" in ldraw and "0 STEP" in ldraw
        assert png.read_bytes().startswith(b"\x89PNG")
        out = capsys.readouterr().out
        assert "existing-work comparison" in out
        assert "measures nothing" in out

    def test_json_records_train_provenance_and_no_model(self, cli, tmp_path,
                                                       capsys):
        code = cli.main([
            "--mode", "f-pipeline", "--caption", "red car",
            "--inventory", "2x4:2", "--catalog",
            str(catalogue_file(tmp_path)), "--top-n", "1", "--json"])
        assert code == cli.EXIT_OK
        body = json.loads(capsys.readouterr().out)
        assert body["catalog"]["split"] == "train"
        assert body["catalog"]["split_manifest_sha256"] == \
            pipeline.FROZEN_SPLIT_MANIFEST_SHA256
        assert body["method"]["model_loaded"] is False
        assert body["method"]["retrieval"] == \
            "deterministic lexical baseline"
        assert body["showcase"]["checks"]["termination_accepted"] is None
        assert body["showcase"]["checks"]["touches_ground"] is True
        assert "touches_ground" in \
            body["showcase"]["delivery"]["checks_used"]
        assert body["showcase"]["delivery"]["static_delivery_ready"] is True

    def test_compare_refuses_cp_sat_only_flags(self, cli, tmp_path, capsys):
        for flag, value in (("--seed", "3"), ("--time-limit", "1")):
            code = cli.main([
                "--mode", "compare", "--caption", "x",
                "--inventory", "2x4:2", "--catalog",
                str(catalogue_file(tmp_path)), flag, value])
            assert code == cli.EXIT_REFUSED
            assert flag in capsys.readouterr().err

    def test_no_candidate_is_exit_one_and_writes_no_files(self, cli,
                                                           tmp_path, capsys):
        out = tmp_path / "must-not-exist.ldr"
        code = cli.main([
            "--mode", "compare", "--caption", "red car",
            "--inventory", "1x1:1", "--catalog",
            str(catalogue_file(tmp_path)), "--ldr", str(out)])
        assert code == cli.EXIT_NO_RESULT
        assert not out.exists()
        assert "No output file was written" in capsys.readouterr().out

    def test_bad_manual_inventory_is_a_readable_refusal(self, cli, tmp_path,
                                                        capsys):
        code = cli.main([
            "--mode", "compare", "--caption", "x",
            "--inventory", "2x8:1", "--catalog",
            str(catalogue_file(tmp_path))])
        assert code == cli.EXIT_REFUSED
        assert "not one of the eight" in capsys.readouterr().err

    def test_ldraw_and_preview_must_not_share_an_output_path(
            self, cli, tmp_path, capsys):
        target = tmp_path / "same-output"
        code = cli.main([
            "--mode", "compare", "--caption", "red car",
            "--inventory", "2x4:2", "--catalog",
            str(catalogue_file(tmp_path)), "--ldr", str(target),
            "--preview", str(tmp_path / "." / "same-output")])
        assert code == cli.EXIT_REFUSED
        assert not target.exists()
        err = capsys.readouterr().err
        assert "--ldr" in err and "--preview" in err
        assert "same output path" in err

    def test_an_invalid_preview_suffix_is_refused_before_ldraw_is_written(
            self, cli, tmp_path, capsys):
        ldr = tmp_path / "must-not-exist.ldr"
        preview = tmp_path / "bad.jpg"
        code = cli.main([
            "--mode", "compare", "--caption", "red car",
            "--inventory", "2x4:2", "--catalog",
            str(catalogue_file(tmp_path)), "--ldr", str(ldr),
            "--preview", str(preview)])
        assert code == cli.EXIT_REFUSED
        assert not ldr.exists() and not preview.exists()
        assert ".png or .svg" in capsys.readouterr().err

    def test_non_finite_cli_time_limit_is_a_controlled_refusal(
            self, cli, tmp_path, capsys):
        code = cli.main([
            "--mode", "f-pipeline", "--caption", "red car",
            "--inventory", "2x4:2", "--catalog",
            str(catalogue_file(tmp_path)), "--time-limit", "nan"])
        assert code == cli.EXIT_REFUSED
        assert "finite number" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "marker", ["exit-zero-compare", "exit-zero-f-pipeline"])
    def test_the_documented_delivery_examples_really_exit_zero(
            self, cli, tmp_path, capsys, marker):
        argv = documented_command(marker)
        catalog = catalogue_file(
            tmp_path, [row("s-example", "o-example", EXAMPLE_CAPTION,
                           EXAMPLE_BRICKS)])
        argv.extend(["--catalog", str(catalog)])
        for flag, suffix in (("--ldr", ".ldr"), ("--preview", ".png")):
            at = argv.index(flag) + 1
            argv[at] = str(tmp_path / f"{marker}{suffix}")
        assert cli.main(argv) == cli.EXIT_OK
        assert (tmp_path / f"{marker}.ldr").is_file()
        assert (tmp_path / f"{marker}.png").is_file()
        assert "status" in capsys.readouterr().out


class TestClaimsStayInsideTheEvidence:
    FILES = ("src/delivery/pipeline.py", "scripts/27_delivery.py")

    def test_the_baseline_is_not_called_semantic_or_a_metric(self):
        for rel in self.FILES:
            flat = " ".join((ROOT / rel).read_text(encoding="utf-8").split())
            assert "not a multilingual embedding model" in flat
            assert "Nothing it returns is a metric" in flat or \
                "not a metric" in flat

    def test_no_model_or_frozen_evaluation_module_is_imported(self):
        for rel in self.FILES:
            text = (ROOT / rel).read_text(encoding="utf-8")
            for forbidden in ("src.generation", "src.training", "torch",
                              "transformers", "src.eval.oracle"):
                assert f"import {forbidden}" not in text
                assert f"from {forbidden}" not in text

    def test_the_handoff_document_names_every_boundary(self):
        body = (ROOT / "DELIVERY.md").read_text(encoding="utf-8")
        for required in (
                "scripts/27_delivery.py", "scripts/28_delivery_evidence.py",
                "_train.jsonl", "不是多語 embedding",
                "F-pipeline 尚未正式評估", "使用者明確排除",
                "Structural Success@K", "Semantic Success@K",
                "Full Success@K", "CPU 3D", "touches_ground",
                "object_splits.json", "本次公開版本已完成獨立技術審查"):
            assert required in body

    def test_the_public_allowlist_includes_the_handoff_not_private_data(self):
        script = (ROOT / "scripts/17_public_snapshot.py").read_text(
            encoding="utf-8")
        assert '"DELIVERY.md"' in script
        assert '("data/processed/**", "processed per-record dataset")' in script

    def test_the_workflow_records_completed_product_parts_and_open_research(self):
        body = (ROOT / "BRICKAGAIN_PROJECT_WORKFLOW.md").read_text(
            encoding="utf-8")
        assert "- [x] 手動輸入庫存。" in body
        assert "- [x] 既有作品比對。" in body
        assert "- [x] 3D 圖片或預覽。" in body
        assert "- [ ] A～E 比較" in body
        assert "研究線已結束，此項不再執行" in body
        assert "使用者明確排除，不列入最低交付" in body
        assert "本次公開版本已完成獨立技術審查" in body
        assert "object_splits.json" in body
