"""The Phase 3C result index, and every way of lying with it.

A completed run's evidence lives in a directory that anyone with write
access can edit, so the index that describes it has to be checked by
reading the bytes rather than by reading the index. What that buys is
narrow and specific: the six members have to be **the cells the frozen plan
predetermined**, each bound to the run's own execution manifest, so a
directory of plausible-looking JSONL that happens to be 320 lines long does
not pass.

Every case here works on a temporary copy. No committed byte is touched to
test what happens when a committed byte moves.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ONLY = "artifact-only:"


def _cli():
    spec = importlib.util.spec_from_file_location(
        "results_cli", ROOT / "scripts" / "61_phase3c_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


results = _cli()


def _generation() -> str:
    from src.eval import phase3c

    return phase3c.GENERATION


def _member_rel(step: int) -> str:
    from src.eval import phase3c

    return f"out_dir/{phase3c.samples_member(step)}"


@pytest.fixture()
def result(tmp_path):
    """A writable copy of the committed result, plus the archive it names."""
    generation = _generation()
    source = ROOT / results.RESULTS_ROOT / generation
    if not (source / "index.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {source.relative_to(ROOT)} is not "
                    "published")
    root = tmp_path / "tree"
    target = root / results.RESULTS_ROOT / generation
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    from src.eval import phase3c

    archive = ROOT / phase3c.ARCHIVE_ROOT / generation
    if not archive.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} {generation}'s archive is not published")
    (root / phase3c.ARCHIVE_ROOT).mkdir(parents=True, exist_ok=True)
    shutil.copytree(archive, root / phase3c.ARCHIVE_ROOT / generation)
    # The supersession record lives beside the archive, not inside it, and
    # ``archive_problems`` requires it: without it nothing says this
    # generation is the one that may run.
    supersession = ROOT / phase3c.SUPERSESSION_PATH
    if not supersession.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {generation}'s supersession record is "
                    "not published")
    shutil.copy2(supersession, root / phase3c.SUPERSESSION_PATH)
    return root, target


def _index(directory: Path) -> dict:
    return json.loads((directory / "index.json").read_text())


def _reindex(directory: Path, index: dict) -> None:
    """Rewrite the index and recompute its own digest, as a forger would."""
    index.pop("index_digest", None)
    index["index_digest"] = results.digest_obj(index)
    (directory / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _refile(directory: Path) -> dict:
    """Re-file every size and digest in the index, then re-seal it.

    The careful forger's move: change bytes, then make the whole document
    agree with the change. What has to catch that is the plan, not the
    index's own arithmetic.
    """
    index = _index(directory)
    table = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "index.json":
            blob = path.read_bytes()
            table.append({"path": str(path.relative_to(directory)),
                          "bytes": len(blob),
                          "sha256": hashlib.sha256(blob).hexdigest()})
    index["evidence_files"] = table
    for entry in index.get("members", []):
        rel = entry.get("path")
        if not rel:
            continue
        blob = (directory / rel).read_bytes()
        entry["bytes"] = len(blob)
        entry["sha256"] = hashlib.sha256(blob).hexdigest()
        entry["rows"] = blob.count(b"\n")
    index["totals"]["total_cells"] = sum(
        e.get("rows") or 0 for e in index.get("members", []))
    _reindex(directory, index)
    return index


class TestTheResultIndexHolds:
    def test_the_real_one_verifies(self, result):
        root, directory = result
        assert results.verify(directory, root=root) == []

    def test_it_verifies_against_a_relocated_tree(self, result):
        """Root-aware: not secretly reading the repository behind itself."""
        root, directory = result
        assert str(root) != str(ROOT)
        assert results.verify(directory, root=root) == []

    def test_it_binds_six_members_of_320_and_1920_cells(self, result):
        from src.eval import phase3c

        _root, directory = result
        index = _index(directory)
        assert index["totals"]["members"] == phase3c.N_STEPS
        assert index["totals"]["total_cells"] == 1920
        assert index["totals"]["total_cells_planned"] == 1920
        for position, entry in enumerate(index["members"]):
            group, arm = phase3c.step(position)
            assert entry["step_index"] == position
            assert (entry["group"], entry["arm"]) == (group, arm)
            assert entry["member"] == phase3c.samples_member(position)
            assert entry["rows"] == 320
            assert entry["exit_code"] == 0
            assert entry["command"]
            assert (directory / entry["combined_stdout_stderr"]).is_file()

    def test_it_names_every_out_dir_artefact(self, result):
        _root, directory = result
        named = {e["path"] for e in _index(directory)["evidence_files"]}
        for rel in results.OUT_DIR_ARTEFACTS:
            assert f"out_dir/{rel}" in named

    def test_it_says_the_run_is_citable(self, result):
        """The one field that separates this tree from a failed attempt."""
        _root, directory = result
        index = _index(directory)
        assert index["citable_as_a_result"] is True
        assert index["outcome"] == "complete"
        assert index["receipt_says"]["verified"] is True
        assert index["receipt_says"]["problems"] == []

    def test_the_listing_is_marked_post_hoc(self, result):
        _root, directory = result
        observation = _index(directory)["node"]["observation"]
        assert observation["kind"] == "post_hoc_read_only_observation"
        assert observation["read_only"] is True
        assert observation["nothing_was_created_moved_or_deleted"] is True

    def test_a_second_build_is_refused(self, result):
        """Write-once: the index was written when the run was fresh.

        ``main`` resolves the results tree from the repository root, so this
        checks the committed index -- which is the one a second ``--build``
        would overwrite -- and asserts its bytes did not move.
        """
        _root, _directory = result
        committed = ROOT / results.RESULTS_ROOT / _generation() / "index.json"
        before = committed.read_bytes()
        code = results.main(["--build", "--generation", _generation()])
        assert code == 2
        assert committed.read_bytes() == before


class TestTheResultIndexFailsClosed:
    def _refused(self, root, directory, needle=None):
        problems = results.verify(directory, root=root)
        assert problems, "a tampered result index verified"
        if needle:
            assert any(needle in p for p in problems), problems

    def _lines(self, directory):
        return (directory / _member_rel(0)).read_text().splitlines(True)

    def _write(self, directory, lines):
        (directory / _member_rel(0)).write_text("".join(lines))

    # -- the bytes, without re-filing --------------------------------------
    def test_altered_evidence_bytes_are_refused(self, result):
        root, directory = result
        (directory / "node" / "driver.log").write_text("rewritten\n")
        self._refused(root, directory, "digests to")

    def test_a_removed_file_is_refused(self, result):
        root, directory = result
        (directory / "mac" / "verify.txt").unlink()
        self._refused(root, directory, "is not here")

    def test_an_unnamed_extra_file_is_refused(self, result):
        root, directory = result
        (directory / "out_dir" / "planted.json").write_text("{}\n")
        self._refused(root, directory, "does not name it")

    # -- the rows, with every size and digest re-filed ---------------------
    def test_a_missing_row_is_refused(self, result):
        root, directory = result
        self._write(directory, self._lines(directory)[:-1])
        _refile(directory)
        self._refused(root, directory, "must produce are absent")

    def test_a_duplicated_row_is_refused(self, result):
        root, directory = result
        lines = self._lines(directory)
        self._write(directory, lines[:-1] + [lines[0]])
        _refile(directory)
        self._refused(root, directory, "appear more than once")

    def test_an_extra_row_is_refused(self, result):
        root, directory = result
        lines = self._lines(directory)
        row = json.loads(lines[0])
        row["seed"] = 99
        self._write(directory, lines + [json.dumps(row) + "\n"])
        _refile(directory)
        self._refused(root, directory)

    def test_a_row_moved_to_another_arm_is_refused(self, result):
        root, directory = result
        lines = self._lines(directory)
        row = json.loads(lines[0])
        row["arm"] = "B"
        lines[0] = json.dumps(row) + "\n"
        self._write(directory, lines)
        _refile(directory)
        self._refused(root, directory)

    def test_a_row_bound_to_another_manifest_is_refused(self, result):
        root, directory = result
        lines = self._lines(directory)
        row = json.loads(lines[0])
        row["manifest_digest"] = "0" * 64
        lines[0] = json.dumps(row) + "\n"
        self._write(directory, lines)
        _refile(directory)
        self._refused(root, directory)

    def test_a_corrupt_row_is_refused(self, result):
        root, directory = result
        lines = self._lines(directory)
        lines[3] = "{not json\n"
        self._write(directory, lines)
        _refile(directory)
        self._refused(root, directory)

    def test_a_member_replaced_by_another_step_is_refused(self, result):
        """320 rows of the right shape, from the wrong step."""
        root, directory = result
        other = (directory / _member_rel(1)).read_bytes()
        (directory / _member_rel(0)).write_bytes(other)
        _refile(directory)
        self._refused(root, directory)

    def test_a_declared_row_count_that_disagrees_is_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["members"][0]["rows"] = 319
        index["totals"]["total_cells"] = 1919
        _reindex(directory, index)
        self._refused(root, directory, "holds 320")

    # -- the four self-digesting documents ---------------------------------
    def test_an_altered_execution_manifest_is_refused(self, result):
        root, directory = result
        path = directory / "out_dir" / "execution_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pack_digest"] = "0" * 64
        path.write_text(json.dumps(manifest, indent=2))
        _refile(directory)
        self._refused(root, directory, "manifest_digest")

    def test_a_manifest_whose_own_digest_was_recomputed_is_still_refused(
            self, result):
        """Self-consistency is not enough: the archive is read too."""
        root, directory = result
        path = directory / "out_dir" / "execution_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pack_digest"] = "0" * 64
        manifest["manifest_digest"] = results.digest_obj(
            {k: v for k, v in manifest.items() if k != "manifest_digest"})
        path.write_text(json.dumps(manifest, indent=2))
        _refile(directory)
        self._refused(root, directory, "archived grant")

    def test_a_receipt_that_is_not_verified_is_refused(self, result):
        root, directory = result
        path = directory / "out_dir" / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["verified"] = False
        receipt["receipt_digest"] = results.digest_obj(
            {k: v for k, v in receipt.items() if k != "receipt_digest"})
        path.write_text(json.dumps(receipt, indent=2))
        _refile(directory)
        self._refused(root, directory, "not verified")

    def test_a_receipt_carrying_another_seal_digest_is_refused(self, result):
        root, directory = result
        path = directory / "out_dir" / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["carried_seal_digest"] = "1" * 64
        receipt["receipt_digest"] = results.digest_obj(
            {k: v for k, v in receipt.items() if k != "receipt_digest"})
        path.write_text(json.dumps(receipt, indent=2))
        _refile(directory)
        self._refused(root, directory)

    def test_a_score_record_from_an_unapproved_scorer_is_refused(self, result):
        root, directory = result
        path = directory / "out_dir" / "scores.json"
        scores = json.loads(path.read_text())
        scores["scorer_source_manifest_digest"] = "2" * 64
        scores["scores_digest"] = results.digest_obj(
            {k: v for k, v in scores.items() if k != "scores_digest"})
        path.write_text(json.dumps(scores, indent=2))
        _refile(directory)
        self._refused(root, directory, "the plan did not approve")

    def test_a_seal_over_the_wrong_number_of_rows_is_refused(self, result):
        root, directory = result
        path = directory / "out_dir" / "seal.json"
        seal = json.loads(path.read_text())
        seal["expected_rows"] = 1919
        seal["seal_digest"] = results.digest_obj(
            {k: v for k, v in seal.items() if k != "seal_digest"})
        path.write_text(json.dumps(seal, indent=2))
        _refile(directory)
        self._refused(root, directory, "rows, not 1920")

    # -- the index's own claims --------------------------------------------
    def test_an_index_edited_and_re_digested_is_still_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["totals"]["total_cells"] = 9999
        _reindex(directory, index)
        self._refused(root, directory, "the members hold")

    def test_an_index_that_does_not_cover_itself_is_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["totals"]["total_cells"] = 9999
        (directory / "index.json").write_text(json.dumps(index, indent=2))
        self._refused(root, directory, "index_digest is")

    def test_a_run_claiming_a_nonzero_exit_is_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["members"][3]["exit_code"] = 1
        _reindex(directory, index)
        self._refused(root, directory, "exited 1")

    def test_a_result_that_says_it_is_not_citable_is_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["citable_as_a_result"] = False
        _reindex(directory, index)
        self._refused(root, directory, "citable_as_a_result")

    def test_a_listing_that_disagrees_on_row_counts_is_refused(self, result):
        root, directory = result
        rel = _index(directory)["node"]["observation"]["listing"]
        path = directory / rel
        path.write_text(path.read_text().replace(
            "step_00_even_A POPULATED 320", "step_00_even_A POPULATED 319"))
        _refile(directory)
        self._refused(root, directory, "the listing found 319")

    def test_a_listing_not_marked_post_hoc_is_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["node"]["observation"]["kind"] = "runner_output"
        _reindex(directory, index)
        self._refused(root, directory, "post-hoc")

    def test_an_index_naming_another_grant_is_refused(self, result):
        root, directory = result
        index = _index(directory)
        index["grant_digest_it_ran_under"] = "3" * 64
        _reindex(directory, index)
        self._refused(root, directory, "another grant")

    def test_a_missing_index_is_refused(self, result):
        root, directory = result
        (directory / "index.json").unlink()
        self._refused(root, directory, "recorded nowhere")


# ---------------------------------------------------------------------------
# The end-to-end pass: re-derive the run, do not merely re-read it
# ---------------------------------------------------------------------------
#
# Everything above checks the index against the bytes and the members against
# the plan's *cell identities*. That leaves one shape of forgery standing:
# rewrite what the model said, then re-file every size, digest and count so
# the document agrees with itself again. No cell moves, so nothing above
# notices -- and the seal, the receipt, the score record and the four
# published documents were each checked for self-consistency and never
# against a re-derivation.
#
# So ``verify`` re-runs the chain through the same functions ``--verify``,
# ``--score`` and ``--report`` use: the seal is re-judged and every member
# re-measured, the receipt is re-derived from the bytes, the whole score
# record is recomputed from ``raw_text`` by the frozen scorer and compared
# field by field, and the four published documents are re-rendered and
# compared. These cases are the ones that only that pass can catch.


def _refile_index(directory: Path) -> None:
    """Re-file *everything the index says about itself*, then re-seal it.

    The file table, every member's size, digest and row count, the totals,
    and the ``artefact_digests`` block copied back out of the documents on
    disk. After this the index is completely self-consistent with the
    tampered tree, which is the point: what is left to object is the
    re-derivation and nothing else.

    Deliberately does **not** touch the seal, the receipt or the score
    record themselves. Those are the run's own evidence; re-filing them is a
    separate, stronger attack, exercised below.
    """
    index = _refile(directory)
    out = directory / "out_dir"
    for name, field in results.SELF_DIGESTS.items():
        path = out / name
        if path.is_file():
            index["artefact_digests"][
                {"execution_manifest.json": "manifest_digest",
                 "seal.json": "seal_digest",
                 "receipt.json": "receipt_digest",
                 "scores.json": "scores_digest"}[name]
            ] = json.loads(path.read_text()).get(field)
    for name, key in (("phase3c_report.md", "report_sha256"),
                      ("success_cases.json", "success_cases_sha256"),
                      ("failure_cases.json", "failure_cases_sha256"),
                      ("reproduce.md", "reproduce_sha256")):
        path = out / name
        if path.is_file():
            index["artefact_digests"][key] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    receipt = out / "receipt.json"
    if receipt.is_file():
        stored = json.loads(receipt.read_text())
        index["artefact_digests"]["carried_seal_digest"] = stored.get(
            "carried_seal_digest")
        index["receipt_says"] = {"verified": stored.get("verified"),
                                 "problems": stored.get("problems")}
    _reindex(directory, index)


class TestTheRunIsReDerivedAndNotJustReRead:
    def test_the_untampered_control_passes_the_whole_chain(self, result):
        """The control. Everything below must fail *because of the tamper*."""
        root, directory = result
        assert results.verify(directory, root=root) == []

    def test_an_edited_score_value_is_refused(self, result):
        """Recomputing ``scores_digest`` does not make a number true.

        The record is recomputed from the stored ``raw_text`` by the frozen
        scorer and compared field by field, so the edited value disagrees
        with what these samples derive.
        """
        root, directory = result
        path = directory / "out_dir" / "scores.json"
        record = json.loads(path.read_text())
        arm = record["primary_contrast"][0]
        rates = record["per_arm"][arm]["overall"]["rates"]
        before = rates["collision_free"]["value"]
        rates["collision_free"]["value"] = 0.5 if before != 0.5 else 0.25
        record["scores_digest"] = results.digest_obj(
            {k: v for k, v in record.items() if k != "scores_digest"})
        path.write_text(json.dumps(record, indent=2))
        _refile_index(directory)

        problems = results.verify(directory, root=root)
        assert problems, "an edited score value verified"
        assert any("is not what these samples derive" in p
                   for p in problems), problems

    def test_an_edited_generation_is_refused_by_the_seal(self, result):
        """``raw_text`` rewritten, then every index-level number re-filed.

        The seal came back from the node and records each member's size and
        SHA-256; it is not the index's to re-file.
        """
        root, directory = result
        member = directory / _member_rel(0)
        lines = member.read_text().splitlines(True)
        row = json.loads(lines[7])
        row["raw_text"] = "2x4 (0,0,0)\n2x4 (0,0,1)\n"
        lines[7] = json.dumps(row) + "\n"
        member.write_text("".join(lines))
        _refile_index(directory)

        problems = results.verify(directory, root=root)
        assert problems, "an edited generation verified"
        assert any("the seal says" in p for p in problems), problems

    def test_an_edited_generation_is_refused_even_with_the_seal_re_filed(
            self, result):
        """The stronger form: re-file the seal as well.

        Now the bytes agree with the seal, so the only thing left that can
        object is a re-derivation -- the receipt rebuilt from these bytes,
        and the score record recomputed from these texts.
        """
        root, directory = result
        member = directory / _member_rel(0)
        lines = member.read_text().splitlines(True)
        row = json.loads(lines[7])
        row["raw_text"] = "2x4 (0,0,0)\n2x4 (0,0,1)\n"
        lines[7] = json.dumps(row) + "\n"
        member.write_text("".join(lines))

        seal_path = directory / "out_dir" / "seal.json"
        seal = json.loads(seal_path.read_text())
        blob = member.read_bytes()
        name = f"samples/{Path(_member_rel(0)).name}"
        seal["members"][name]["size"] = len(blob)
        seal["members"][name]["sha256"] = hashlib.sha256(blob).hexdigest()
        seal["members"][name]["rows"] = blob.count(b"\n")
        seal["seal_digest"] = results.digest_obj(
            {k: v for k, v in seal.items() if k != "seal_digest"})
        seal_path.write_text(json.dumps(seal, indent=2))
        _refile_index(directory)

        problems = results.verify(directory, root=root)
        assert problems, "an edited generation with a re-filed seal verified"

    def test_an_edited_report_is_refused(self, result):
        """A published document is re-rendered from the record, not read."""
        root, directory = result
        path = directory / "out_dir" / "phase3c_report.md"
        text = path.read_text()
        assert "| `collision_free` |" in text
        path.write_text(text.replace(
            "## Main failure modes",
            "## Main failure modes\n\nPlacementGate improved Core Success.\n"))
        _refile_index(directory)

        problems = results.verify(directory, root=root)
        assert problems, "an edited report verified"
        assert any("report" in p for p in problems), problems

    def test_an_edited_case_index_is_refused(self, result):
        """The case indices are derived from the record too."""
        root, directory = result
        path = directory / "out_dir" / "success_cases.json"
        index = json.loads(path.read_text())
        index["cases"] = index["cases"][:-1]
        index["n"] = len(index["cases"])
        path.write_text(json.dumps(index, indent=2))
        _refile_index(directory)

        problems = results.verify(directory, root=root)
        assert problems, "an edited case index verified"
        assert any("case index" in p for p in problems), problems

    def test_a_deleted_published_member_is_refused(self, result):
        root, directory = result
        (directory / "out_dir" / "reproduce.md").unlink()
        _refile_index(directory)
        problems = results.verify(directory, root=root)
        assert problems, "a run missing a published member verified"

    def test_a_receipt_rebuilt_over_other_bytes_is_refused(self, result):
        """The receipt is re-derived, so its stored copy has to match."""
        root, directory = result
        path = directory / "out_dir" / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["members"] = {}
        receipt["receipt_digest"] = results.digest_obj(
            {k: v for k, v in receipt.items() if k != "receipt_digest"})
        path.write_text(json.dumps(receipt, indent=2))
        _refile_index(directory)
        problems = results.verify(directory, root=root)
        assert problems, "a rewritten receipt verified"
        assert any("receipt" in p for p in problems), problems
