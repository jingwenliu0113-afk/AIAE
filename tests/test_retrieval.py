"""Chinese condition extraction, the vector index, and grounded explanations.

No embedding model is loaded here. The index is built from documents and a
vector matrix directly, which exercises every refusal the loader has without
requiring half a gigabyte of weights to be present; the model itself is
exercised by ``scripts/34_rag_index.py --smoke``, which is labelled as a smoke
and is not a metric.

What is pinned:

* a condition the extractor understood but could not apply is **named**, and a
  condition it applied is applied -- neither is silently defaulted;
* the index refuses a moved catalogue, a moved model identity, a moved vector
  file and a document list that has drifted from its matrix;
* the same-object exclusion happens *inside* the search, so a caller cannot
  forget it;
* "可以組" appears only when the stock covers every part **and** the structure
  touches the ground and is one component.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.data.bricks import Brick, format_bricks
from src.delivery import pipeline
from src.delivery.pipeline import inventory_evidence, load_train_catalog
from src.retrieval.explain import (explain_candidate, explain_result,
                                   format_explanation)
from src.retrieval.index import (MANIFEST_FILE, VECTORS_FILE, Document, Hit,
                                 IndexError_, VectorIndex, check_object_rows,
                                 documents_from_catalog, load)
from src.retrieval.nlp import (MAX_REQUEST_CHARS, Conditions, NlpError,
                               extract, filter_hits, unapplied_conditions)
from src.retrieval.search import RETRIEVAL_KIND, SearchError, search


# --------------------------------------------------------------------------
# condition extraction
# --------------------------------------------------------------------------

class TestTheExampleFromThePlan:
    def test_it_extracts_all_five_conditions(self):
        got = extract("我想做一台 30 顆以內的藍色小車，顏色可以替換")
        assert got.category == "vehicle"
        assert got.max_parts == 30
        assert got.preferred_colours == ("blue",)
        assert got.allow_colour_substitution is True
        assert not got.has_unresolved


class TestConditionsAreExtractedOrNamed:
    @pytest.mark.parametrize("text,limit", [
        ("30 顆以內", 30), ("最多 40 顆", 40), ("不超過 25 塊", 25),
        ("50 顆以下", 50), ("under 20 bricks", 20), ("at most 12 pieces", 12),
    ])
    def test_a_brick_budget_is_read(self, text, limit):
        assert extract(text).max_parts == limit

    def test_a_bare_count_with_no_limiting_word_is_named_not_applied(self):
        """"50 顆" alone could be a floor, a ceiling or an exact number."""
        got = extract("我要一個 50 顆的房子")
        assert got.max_parts is None
        assert [item.field for item in got.unresolved] == ["max_parts"]
        assert "無法判斷" in got.unresolved[0].reason

    @pytest.mark.parametrize("text,colours", [
        ("藍色", ("blue",)), ("紅色的車", ("red",)),
        ("淺藍灰色的塔", ("light_bluish_grey",)),
        ("a red car", ("red",)),
        ("黑白配色", ("black", "white")),
    ])
    def test_colours_are_read_longest_word_first(self, text, colours):
        assert extract(text).preferred_colours == colours

    def test_a_colour_outside_the_palette_is_named(self):
        got = extract("青色的東西")
        assert got.preferred_colours == ()
        assert [item.field for item in got.unresolved] == ["preferred_colours"]
        assert "青色" in got.unresolved[0].text

    def test_the_word_colour_itself_is_not_a_colour(self):
        """Without this, every sentence mentioning colour reports a phantom."""
        got = extract("顏色可以替換")
        assert got.unresolved == ()
        assert got.allow_colour_substitution is True

    @pytest.mark.parametrize("text,expected", [
        ("顏色可以替換", True), ("不可替換顏色", False),
        ("顏色不限", True), ("必須是紅色", False), ("一台車", None),
    ])
    def test_substitution_is_read_or_left_unstated(self, text, expected):
        assert extract(text).allow_colour_substitution is expected

    @pytest.mark.parametrize("text,category", [
        ("小車", "vehicle"), ("房子", "building"), ("一隻狗", "animal"),
        ("椅子", "furniture"), ("機器人", "figure"), ("一棵樹", "plant"),
        ("something abstract", None),
    ])
    def test_a_category_is_read_when_it_is_there(self, text, category):
        assert extract(text).category == category

    def test_two_conflicting_method_words_are_named_not_guessed(self):
        got = extract("找現成的，或者生成一個新的")
        assert got.mode is None
        assert any(item.field == "mode" for item in got.unresolved)

    def test_one_method_word_is_read(self):
        assert extract("推薦既有作品").mode == "existing"

    def test_a_full_width_number_is_read(self):
        assert extract("３０ 顆以內的車").max_parts == 30

    def test_an_empty_request_is_refused(self):
        with pytest.raises(NlpError, match="empty"):
            extract("   ")

    def test_a_non_string_is_refused(self):
        with pytest.raises(NlpError, match="must be a string"):
            extract(None)

    def test_an_over_long_request_is_refused(self):
        with pytest.raises(NlpError, match="over the"):
            extract("車" * (MAX_REQUEST_CHARS + 1))

    def test_the_description_says_when_nothing_was_extracted(self):
        lines = extract("something abstract").describe_zh()
        assert any("沒有抽取到" in line for line in lines)

    def test_the_serialised_form_admits_it_is_rule_based(self):
        assert "not a language model" in extract("一台車").as_dict()["extractor"]


class TestWhichConditionsActOnRetrieval:
    def _hits(self, sizes):
        return [Hit(document=Document(catalog_id=f"c{i}", caption="x",
                                     n_bricks=size, required={"2x4": size},
                                     touches_ground=True, connected=True),
                    score=1.0 - i * 0.01, semantic_rank=i + 1)
                for i, size in enumerate(sizes)]

    def test_a_brick_budget_rejects_the_oversized(self):
        kept, rejected = filter_hits(self._hits([10, 50, 20]),
                                     extract("30 顆以內"))
        assert [hit.document.n_bricks for hit in kept] == [10, 20]
        assert len(rejected) == 1
        assert "超過" in rejected[0][1]

    def test_no_budget_keeps_everything(self):
        kept, rejected = filter_hits(self._hits([10, 50]), extract("一台車"))
        assert len(kept) == 2 and rejected == []

    def test_a_category_is_reported_as_not_filtering(self):
        reported = unapplied_conditions(extract("一台小車"))
        assert any(item["field"] == "category" for item in reported)

    def test_a_colour_preference_is_reported_as_not_filtering(self):
        reported = unapplied_conditions(extract("藍色的車"))
        entry = next(item for item in reported
                     if item["field"] == "preferred_colours")
        assert "配色器" in entry["reason"]


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------

def documents(count=4):
    out = []
    for index in range(count):
        out.append(Document(
            catalog_id=f"cat{index:02d}",
            caption=f"a small thing number {index}",
            n_bricks=4 + index,
            required={"2x4": 2, "1x2": 1 + index},
            touches_ground=True, connected=True,
            object_id=f"obj{index:02d}"))
    return tuple(out)


def unit_vectors(count, dimension=8, seed=3):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, dimension)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def an_index(count=4, dimension=8):
    docs = documents(count)
    return VectorIndex(
        documents=docs, vectors=unit_vectors(count, dimension),
        embedding={"repo": "fake/model", "revision": "r1"},
        catalog_sha256="c" * 64, split_manifest_sha256="s" * 64,
        identity_digest="i" * 64, build_device="cpu")


class TestTheIndexRefusesWhatItCannotTrust:
    def test_a_matrix_that_does_not_match_the_documents_is_refused(self):
        with pytest.raises(IndexError_, match="drifted apart"):
            VectorIndex(documents=documents(4), vectors=unit_vectors(3),
                        embedding={}, catalog_sha256="c",
                        split_manifest_sha256="s", identity_digest="i",
                        build_device="cpu")

    def test_vectors_that_are_not_unit_length_are_refused(self):
        with pytest.raises(IndexError_, match="not unit length"):
            VectorIndex(documents=documents(2),
                        vectors=(unit_vectors(2) * 3.0).astype(np.float32),
                        embedding={}, catalog_sha256="c",
                        split_manifest_sha256="s", identity_digest="i",
                        build_device="cpu")

    def test_a_non_float32_matrix_is_refused(self):
        with pytest.raises(IndexError_, match="float32"):
            VectorIndex(documents=documents(2),
                        vectors=unit_vectors(2).astype(np.float64),
                        embedding={}, catalog_sha256="c",
                        split_manifest_sha256="s", identity_digest="i",
                        build_device="cpu")

    def test_a_one_dimensional_matrix_is_refused(self):
        with pytest.raises(IndexError_, match="two-dimensional"):
            VectorIndex(documents=documents(1), vectors=np.zeros(4,
                                                                np.float32),
                        embedding={}, catalog_sha256="c",
                        split_manifest_sha256="s", identity_digest="i",
                        build_device="cpu")


class TestSearching:
    def test_it_returns_the_nearest_documents_in_order(self):
        index = an_index(6)
        hits = index.search(index.vectors[2], top_n=3)
        assert hits[0].document.catalog_id == "cat02"
        assert [hit.semantic_rank for hit in hits] == [1, 2, 3]
        assert hits[0].score >= hits[1].score >= hits[2].score

    def test_the_same_object_exclusion_is_applied_inside_the_search(self):
        index = an_index(6)
        hits = index.search(index.vectors[2], top_n=6,
                            exclude_object_id="obj02")
        assert all(hit.document.object_id != "obj02" for hit in hits)
        assert len(hits) == 5

    def test_excluding_everything_is_refused(self):
        docs = documents(3)
        index = VectorIndex(
            documents=tuple(Document(d.catalog_id, d.caption, d.n_bricks,
                                     d.required, True, True, "same")
                            for d in docs),
            vectors=unit_vectors(3), embedding={}, catalog_sha256="c",
            split_manifest_sha256="s", identity_digest="i", build_device="cpu")
        with pytest.raises(IndexError_, match="removed every document"):
            index.search(index.vectors[0], exclude_object_id="same")

    def test_a_query_of_the_wrong_width_is_refused(self):
        index = an_index(3, dimension=8)
        with pytest.raises(IndexError_, match="dimensions"):
            index.search(np.ones(4, dtype=np.float32))

    def test_a_zero_query_is_refused(self):
        index = an_index(3)
        with pytest.raises(IndexError_, match="query vector is zero"):
            index.search(np.zeros(8, dtype=np.float32))

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_a_bad_top_n_is_refused(self, bad):
        with pytest.raises(IndexError_, match="positive whole number"):
            an_index(3).search(an_index(3).vectors[0], top_n=bad)

    def test_the_ranking_is_reproducible(self):
        index = an_index(6)
        first = [hit.document.catalog_id
                 for hit in index.search(index.vectors[1], top_n=6)]
        again = [hit.document.catalog_id
                 for hit in index.search(index.vectors[1], top_n=6)]
        assert first == again


class TestSavingAndLoading:
    def test_it_round_trips(self, tmp_path):
        index = an_index(5)
        target, digest = index.save(tmp_path / "idx")
        loaded = load(target, expected_manifest_sha256=digest)
        assert loaded.size == index.size
        assert np.allclose(loaded.vectors, index.vectors)
        assert [d.catalog_id for d in loaded.documents] == \
            [d.catalog_id for d in index.documents]

    def test_the_object_ids_survive_for_the_exclusion(self, tmp_path):
        index = an_index(4)
        target, _digest = index.save(tmp_path / "idx")
        loaded = load(target)
        assert loaded.documents[0].object_id == "obj00"

    def test_a_wrong_manifest_digest_is_refused(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        with pytest.raises(IndexError_, match="not the expected"):
            load(target, expected_manifest_sha256="0" * 64)

    def test_a_different_model_identity_is_refused(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        with pytest.raises(IndexError_, match="not comparable"):
            load(target, expected_identity_digest="d" * 64)

    def test_a_different_catalogue_is_refused(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        with pytest.raises(IndexError_, match="catalogue"):
            load(target, expected_catalog_sha256="z" * 64)

    def test_a_tampered_vector_file_is_refused(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        payload = bytearray((target / VECTORS_FILE).read_bytes())
        payload[-1] ^= 0xFF
        (target / VECTORS_FILE).write_bytes(bytes(payload))
        with pytest.raises(IndexError_, match="hashes to"):
            load(target)

    def test_a_missing_vector_file_is_refused(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        (target / VECTORS_FILE).unlink()
        with pytest.raises(IndexError_, match="which is missing"):
            load(target)

    def test_a_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(IndexError_, match="there is no"):
            load(tmp_path / "nothing")

    def test_a_manifest_of_another_kind_is_refused(self, tmp_path):
        target = tmp_path / "idx"
        target.mkdir()
        (target / MANIFEST_FILE).write_text(json.dumps({"kind": "other"}),
                                            encoding="utf-8")
        with pytest.raises(IndexError_, match="not a retrieval index"):
            load(target)

    def test_a_row_order_that_disagrees_is_refused(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        body = json.loads((target / MANIFEST_FILE).read_text("utf-8"))
        body["rows"] = list(reversed(body["rows"]))
        (target / MANIFEST_FILE).write_text(
            json.dumps(body, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")), encoding="utf-8")
        with pytest.raises(IndexError_, match="row order"):
            load(target)

    def test_the_manifest_states_the_boundaries(self, tmp_path):
        target, _digest = an_index(3).save(tmp_path / "idx")
        body = json.loads((target / MANIFEST_FILE).read_text("utf-8"))
        assert "train split only" in body["boundary"]
        assert "never published" in body["boundary"]
        assert "measures nothing" in body["not_a_metric"]
        assert "exact cosine" in body["index_kind"]


class TestDocumentText:
    def test_the_embedded_text_carries_the_part_evidence(self):
        document = documents(1)[0]
        text = document.embed_text()
        assert document.caption in text
        assert "bricks" in text and "parts:" in text

    def test_the_object_id_is_not_in_the_public_dict(self):
        assert "object_id" not in documents(1)[0].as_dict()


# --------------------------------------------------------------------------
# search over a real catalogue, with a stand-in embedder
# --------------------------------------------------------------------------

STACK = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]
SMALL = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 1)]


def row(sid, oid, caption, bricks):
    return {"split": "train", "role": "control", "variant": "exact",
            "object_id": oid, "structure_id": sid, "caption": caption,
            "bricks_txt": format_bricks(bricks)}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    objects = {"o-car": "train", "o-tower": "train"}
    structures = {"s-car": "o-car", "s-tower": "o-tower"}
    manifest = tmp_path / "object_splits.json"
    manifest.write_text(json.dumps({
        "meta": {"fixture": True}, "counts": {"train": 2},
        "objects": objects, "structures": structures}, sort_keys=True),
        encoding="utf-8")
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(pipeline, "FROZEN_SPLIT_MANIFEST_SHA256",
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    path = tmp_path / "fixture_train.jsonl"
    path.write_text(
        json.dumps(row("s-car", "o-car", "a compact red car", STACK)) + "\n"
        + json.dumps(row("s-tower", "o-tower", "a tiny tower", SMALL)) + "\n",
        encoding="utf-8")
    return load_train_catalog(path)


class StandInEmbedder:
    """Deterministic vectors from a digest of the text.

    Not a model and not pretending to be one: it exists so the search's
    plumbing -- the identity check, the catalogue check, the re-ranking and the
    explanation -- can be tested without half a gigabyte of weights.
    """

    device = "cpu"
    dimension = 8

    def identity_digest(self) -> str:
        return "stand-in"

    def identity(self) -> dict:
        return {"repo": "stand-in", "revision": "0"}


def stand_in_vector(text: str, dimension: int = 8) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest[:dimension], dtype=np.uint8).astype(np.float32)
    raw = raw - raw.mean()
    norm = float(np.linalg.norm(raw)) or 1.0
    return (raw / norm).astype(np.float32)


@pytest.fixture
def searchable(catalog, monkeypatch):
    docs = documents_from_catalog(catalog)
    vectors = np.stack([stand_in_vector(d.embed_text()) for d in docs])
    index = VectorIndex(
        documents=docs, vectors=vectors,
        embedding={"repo": "stand-in", "revision": "0"},
        catalog_sha256=catalog.sha256,
        split_manifest_sha256=catalog.split_manifest_sha256,
        identity_digest="stand-in", build_device="cpu")

    def fake_embed(_embedder, texts, *, kind, batch=32):
        return np.stack([stand_in_vector(text) for text in texts])

    import src.retrieval.embed as embed_module

    monkeypatch.setattr(embed_module, "embed", fake_embed)
    return index, catalog, StandInEmbedder()


class TestSearchOverACatalogue:
    def test_a_buildable_work_is_selected(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台紅色小車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        assert result.status == "buildable_existing_work_found"
        assert result.selected is not None
        assert result.selected.buildable

    def test_a_short_stock_yields_no_recommendation(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 1}, top_n=2)
        assert result.status == "no_buildable_existing_work_in_retrieved_set"
        assert result.selected is None
        assert any(candidate.missing for candidate in result.ranked)

    def test_the_ranking_puts_buildable_first(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"1x1": 2}, top_n=2)
        buildable = [candidate.buildable for candidate in result.ranked]
        assert buildable == sorted(buildable, reverse=True)

    def test_a_budget_rejects_a_candidate_by_name(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder,
                        extract("1 顆以內的車"), {"2x4": 4, "1x1": 4},
                        top_n=2)
        assert result.rejected_by_conditions
        assert "超過" in result.rejected_by_conditions[0]["reason"]

    def test_a_moved_catalogue_is_refused(self, searchable):
        index, catalog, embedder = searchable
        from dataclasses import replace

        with pytest.raises(SearchError, match="not the same data"):
            search(index, replace(catalog, sha256="z" * 64), embedder,
                   extract("車"), {"2x4": 4})

    def test_a_moved_model_identity_is_refused(self, searchable):
        index, catalog, embedder = searchable

        class Other(StandInEmbedder):
            def identity_digest(self):
                return "different"

        with pytest.raises(SearchError, match="not comparable"):
            search(index, catalog, Other(), extract("車"), {"2x4": 4})

    def test_an_empty_inventory_is_refused(self, searchable):
        index, catalog, embedder = searchable
        with pytest.raises(SearchError, match="inventory is required"):
            search(index, catalog, embedder, extract("車"), {})

    def test_the_serialised_result_names_the_retrieval_kind(self, searchable):
        index, catalog, embedder = searchable
        body = search(index, catalog, embedder, extract("車"),
                      {"2x4": 4, "1x1": 4}).as_dict()
        assert body["retrieval"] == RETRIEVAL_KIND
        assert "not a promise" in body["boundary"]


# --------------------------------------------------------------------------
# grounded explanations
# --------------------------------------------------------------------------

class TestTheExplanationIsGroundedInTheNumbers:
    def test_a_buildable_candidate_says_it_can_be_built(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        body = explain_candidate(result.selected)
        assert body["verdict"] == "buildable"
        assert any("可以組" in line for line in body["sentences"])

    def test_a_short_candidate_never_says_it_can_be_built(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 1}, top_n=2)
        for candidate in result.ranked:
            body = explain_candidate(candidate)
            assert body["verdict"] == "not_buildable"
            assert all("**可以組**" not in line for line in body["sentences"])
            assert any("不能組" in line for line in body["sentences"])

    def test_the_evidence_matches_the_shared_arithmetic(self, searchable):
        index, catalog, embedder = searchable
        stock = {"2x4": 1}
        result = search(index, catalog, embedder, extract("一台車"), stock,
                        top_n=2)
        for candidate in result.ranked:
            evidence = explain_candidate(candidate)["evidence"]
            expected = inventory_evidence(candidate.item.required, stock)
            assert evidence["missing_parts"] == expected["missing"]
            assert abs(evidence["inventory_completion"]
                       - expected["completion"]) < 1e-6

    def test_the_explanation_carries_the_connectivity_caveat(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        body = explain_result(result, {"2x4": 4, "1x1": 4})
        joined = " ".join(body["notes"])
        assert "不是物理支撐" in joined
        assert "不是成效指標" in joined

    def test_it_states_that_no_model_wrote_it(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        body = explain_result(result, {"2x4": 4})
        assert "No language model" in body["generated_from"]

    def test_unresolved_conditions_are_carried_into_the_header(self,
                                                               searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("我要 50 顆的車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        body = explain_result(result, {"2x4": 4})
        assert any("沒有套用" in line for line in body["header"])

    def test_the_object_id_never_appears(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        text = format_explanation(explain_result(result, {"2x4": 4}))
        assert "o-car" not in text and "o-tower" not in text
        assert "s-car" not in text

    def test_no_candidates_is_explained_rather_than_substituted(self,
                                                                searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("1 顆以內的車"),
                        {"2x4": 4}, top_n=2)
        body = explain_result(result, {"2x4": 4})
        assert "沒有推薦" in body["selection"]

    def test_the_plain_text_form_includes_every_candidate(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 4, "1x1": 4}, top_n=2)
        text = format_explanation(explain_result(result, {"2x4": 4}))
        for candidate in result.ranked:
            assert candidate.item.catalog_id in text


# --------------------------------------------------------------------------
# Round 49: the object-id mapping is load-bearing, so it is checked
#
# ``load`` read ``manifest.get("object_rows") or {}`` and then
# ``objects.get(catalog_id, "")``.  A manifest with no mapping, a partial one,
# or one full of empty strings all loaded cleanly -- and the same-object
# exclusion then matched nothing, so the very work a caller asked to keep out
# came back ranked first with nothing saying the guard was off.
# --------------------------------------------------------------------------

def write_index(tmp_path, *, mutate=None, count=4):
    index = an_index(count)
    directory = tmp_path / "index"
    index.save(directory)
    path = directory / MANIFEST_FILE
    body = json.loads(path.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(body)
        path.write_text(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")), encoding="utf-8")
    return directory


class TestTheObjectIdMappingIsCheckedAtLoad:
    def test_a_sound_index_still_loads_and_keeps_its_object_ids(self,
                                                                 tmp_path):
        loaded = load(write_index(tmp_path))
        assert [d.object_id for d in loaded.documents] == [
            f"obj{i:02d}" for i in range(4)]

    def test_a_manifest_with_no_object_rows_is_refused(self, tmp_path):
        directory = write_index(tmp_path,
                                mutate=lambda b: b.pop("object_rows"))
        with pytest.raises(IndexError_, match="no object_rows"):
            load(directory)

    def test_a_null_mapping_is_refused_rather_than_treated_as_none_needed(
            self, tmp_path):
        directory = write_index(
            tmp_path, mutate=lambda b: b.update(object_rows=None))
        with pytest.raises(IndexError_, match="no object_rows"):
            load(directory)

    def test_a_mapping_that_is_not_a_dict_is_refused(self, tmp_path):
        directory = write_index(
            tmp_path, mutate=lambda b: b.update(object_rows=["obj00"]))
        with pytest.raises(IndexError_, match="mapping, not list"):
            load(directory)

    def test_a_partial_mapping_is_refused_by_count(self, tmp_path):
        def drop_one(body):
            body["object_rows"].pop("cat02")

        with pytest.raises(IndexError_, match="1 of 4 document row"):
            load(write_index(tmp_path, mutate=drop_one))

    def test_an_extra_entry_is_refused(self, tmp_path):
        def add_one(body):
            body["object_rows"]["cat99"] = "obj99"

        with pytest.raises(IndexError_, match="not document rows"):
            load(write_index(tmp_path, mutate=add_one))

    def test_an_empty_object_id_is_refused(self, tmp_path):
        def blank(body):
            body["object_rows"]["cat01"] = "  "

        with pytest.raises(IndexError_, match="empty or not text"):
            load(write_index(tmp_path, mutate=blank))

    def test_a_non_text_object_id_is_refused(self, tmp_path):
        def wrong_type(body):
            body["object_rows"]["cat01"] = 7

        with pytest.raises(IndexError_, match="empty or not text"):
            load(write_index(tmp_path, mutate=wrong_type))

    def test_the_checker_can_be_used_alone(self):
        assert check_object_rows({"a": "x"}, ["a"]) == {"a": "x"}
        with pytest.raises(IndexError_):
            check_object_rows({}, ["a"])

    def test_a_moved_split_manifest_is_refused(self, tmp_path):
        directory = write_index(tmp_path)
        load(directory, expected_split_manifest_sha256="s" * 64)
        with pytest.raises(IndexError_, match="not the boundary that applies"):
            load(directory, expected_split_manifest_sha256="d" * 64)


@pytest.fixture
def fake_embedding(monkeypatch):
    import src.retrieval.embed as embed_module

    monkeypatch.setattr(
        embed_module, "embed",
        lambda _e, texts, *, kind, batch=32: np.stack(
            [stand_in_vector(text) for text in texts]))
    return StandInEmbedder()


class TestTheIndexMustAgreeWithTheCatalogue:
    def build(self, catalog):
        docs = documents_from_catalog(catalog)
        vectors = np.stack([stand_in_vector(d.embed_text()) for d in docs])
        return docs, vectors.astype(np.float32)

    def an_index_over(self, catalog, docs=None, *, split=None):
        source, vectors = self.build(catalog)
        return VectorIndex(
            documents=tuple(docs if docs is not None else source),
            vectors=vectors, embedding={"repo": "stand-in", "revision": "0"},
            catalog_sha256=catalog.sha256,
            split_manifest_sha256=(split or catalog.split_manifest_sha256),
            identity_digest="stand-in", build_device="cpu")

    def test_matching_object_ids_pass(self, catalog):
        self.an_index_over(catalog).check_against_catalog(catalog)

    def test_a_wrong_object_id_is_refused_by_the_search(self, catalog,
                                                        fake_embedding):
        docs, _ = self.build(catalog)
        swapped = tuple(
            Document(catalog_id=d.catalog_id, caption=d.caption,
                     n_bricks=d.n_bricks, required=d.required,
                     touches_ground=d.touches_ground, connected=d.connected,
                     object_id="o-not-this-one")
            for d in docs)
        index = self.an_index_over(catalog, swapped)
        with pytest.raises(SearchError, match="object_id the catalogue does"):
            search(index, catalog, fake_embedding, extract("一台車"),
                   {"2x4": 4})

    def test_a_wrong_object_id_would_otherwise_return_the_excluded_work(
            self, catalog, fake_embedding):
        """Why it matters: without the check the exclusion quietly misses."""
        docs, _ = self.build(catalog)
        target = docs[0].object_id
        broken = tuple(
            Document(catalog_id=d.catalog_id, caption=d.caption,
                     n_bricks=d.n_bricks, required=d.required,
                     touches_ground=d.touches_ground, connected=d.connected,
                     object_id="")
            for d in docs)
        index = self.an_index_over(catalog, broken)
        hits = index.search(stand_in_vector("x"), top_n=4,
                            exclude_object_id=target)
        assert len(hits) == len(docs), "the exclusion silently matched nothing"
        with pytest.raises(SearchError):
            search(index, catalog, fake_embedding, extract("一台車"),
                   {"2x4": 4}, exclude_object_id=target)

    def test_a_moved_split_manifest_is_refused_by_the_search(
            self, catalog, fake_embedding):
        index = self.an_index_over(catalog, split="f" * 64)
        with pytest.raises(SearchError, match="train-only"):
            search(index, catalog, fake_embedding, extract("一台車"),
                   {"2x4": 4})


# --------------------------------------------------------------------------
# Round 49: two orderings, two names
#
# ``explain_result`` passed the re-ranked position in as ``rank`` and the
# sentence printed it as 語意排名, so a work the embedding search put third
# and the inventory sort lifted to first was reported as the most similar.
# --------------------------------------------------------------------------

class TestSemanticRankAndRerankRankAreSeparate:
    def test_semantic_rank_and_score_are_monotone_together(self):
        index = an_index(6)
        hits = index.search(index.vectors[2], top_n=6)
        assert [hit.semantic_rank for hit in hits] == [1, 2, 3, 4, 5, 6]
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == max(scores)

    def test_a_hit_serialises_its_rank_under_the_semantic_name(self):
        index = an_index(3)
        body = index.search(index.vectors[0], top_n=1)[0].as_dict()
        assert body["semantic_rank"] == 1
        assert "rank" not in body

    def test_the_reranked_list_carries_both_and_they_can_differ(self,
                                                                searchable):
        index, catalog, embedder = searchable
        # this stock makes the semantically second work the buildable one, so
        # the two orderings really are different in what follows
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 8})
        body = result.as_dict()
        assert [row["rerank_rank"] for row in body["inventory_reranked"]] == \
            list(range(1, len(body["inventory_reranked"]) + 1))
        for row in body["semantic_order"]:
            assert "rerank_rank" not in row
        semantic = {row["catalog_id"]: row["semantic_rank"]
                    for row in body["semantic_order"]}
        for row in body["inventory_reranked"]:
            assert row["semantic_rank"] == semantic[row["catalog_id"]]

    def test_the_explanation_prints_the_hits_semantic_rank_not_the_position(
            self, searchable):
        """The red light: a candidate whose two ranks differ."""
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 8})
        moved = [position for position, c in enumerate(result.ranked, 1)
                 if c.hit.semantic_rank != position]
        assert moved, "fixture no longer exercises a re-ordering"
        body = explain_result(result, {"2x4": 8})
        for position, candidate in enumerate(result.ranked, 1):
            evidence = body["candidates"][position - 1]["evidence"]
            assert evidence["semantic_rank"] == candidate.hit.semantic_rank
            assert evidence["rerank_rank"] == position
            sentence = body["candidates"][position - 1]["sentences"][1]
            assert f"語意排名第 {candidate.hit.semantic_rank}" in sentence
            assert f"重排後排第 {position}" in sentence

    def test_a_candidate_explained_alone_has_no_rerank_rank(self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 8})
        body = explain_candidate(result.ranked[0])
        assert body["evidence"]["rerank_rank"] is None
        assert body["evidence"]["semantic_rank"] == \
            result.ranked[0].hit.semantic_rank
        assert "重排後排第" not in body["sentences"][1]

    def test_the_note_tells_a_reader_which_ordering_the_list_is_in(
            self, searchable):
        index, catalog, embedder = searchable
        result = search(index, catalog, embedder, extract("一台車"),
                        {"2x4": 8})
        note = explain_result(result, {"2x4": 8})["notes"][0]
        assert "semantic_rank" in note and "rerank_rank" in note
