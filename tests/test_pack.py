"""The private training pack: what may leave this machine, and what proves it.

The public snapshot (``scripts/17_public_snapshot.py``) answers "what may be
published to the world". This answers a different question with the same
discipline: **what may be handed to the Taichung execution node**. The node is
not the public, so the two lists differ -- but the failure mode is identical
and so is the defence. An unlisted path is excluded, never included.

Three properties are tested here rather than trusted:

* **default deny.** A file nobody thought about is not in the pack. Every
  category the brief names -- raw and processed data, per-record reports,
  session evidence, weights, checkpoints, credentials, email addresses,
  personal absolute paths and organization ids -- is checked directly, and so
  is the "new file appears" case that a denylist alone fails open on.
* **one definition, not two.** The identifier patterns and the approval table
  are module 17's. A test asserts the pack scans with *that* table, so a
  pattern renamed there cannot leave this side silently scanning nothing.
* **the manifest is the contract.** Every included file is digested, the pack
  carries one digest standing for all of them, and verification is file by
  file. A mutated byte, a missing file and an unexpected extra file are three
  distinct refusals, and each is asserted.

Nothing here loads a model, opens a socket or reads the real dataset. The
synthetic credential literals below are fixed invented strings -- no real
credential appears in this file or in anything it writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training import pack
from src.training.session import manifest_digest, sha256_file

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# A synthetic tree, so the classification tests do not depend on what happens
# to be on this disk today.
# ---------------------------------------------------------------------------

def make_tree(base: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return base


MINIMAL = {
    "requirements.txt": "torch\n",
    "src/__init__.py": "",
    "src/training/__init__.py": "",
    "src/training/gates.py": "GATE = 8\n",
    "scripts/19_gpu_gate.py": "#!/usr/bin/env python3\n",
}


class TestDefaultDeny:
    """An unlisted path is excluded. That is the whole point of the list."""

    @pytest.mark.parametrize("rel", [
        "data/raw/anything.parquet",
        "data/processed/instruct_inv_train.jsonl",
        "data/processed/counterfactual_train.jsonl",
        "data/splits/object_splits.json",
        "data/reports/05_d_arm.json",
        "data/reports/12_f_oracle.json",
        "data/reports/13_lora_smoke.json",
        "data/reports/15_mps_order/exp001/events/00-r1-launch.json",
        "data/reports/15_mps_order/exp002/report.json",
        "data/reports/16_longrun/exp001/events/00-b1-x.json",
        "artifacts/checkpoints/adapter/adapter_model.safetensors",
        "artifacts/renders/thing.png",
        "src/weights.safetensors",
        "src/opt.pt",
        "somewhere/model.bin",
        "state.ckpt",
        ".env",
        ".env.local",
        "CLAUDE.md",
        "PROJECT_STATUS.md",
        ".git/config",
        "src/__pycache__/x.pyc",
        ".DS_Store",
    ])
    def test_the_named_categories_are_excluded(self, rel):
        verdict, reason = pack.classify(rel)
        assert verdict == "exclude", f"{rel} was {verdict}: {reason}"
        assert reason, "an exclusion with no reason cannot be reviewed"

    def test_a_brand_new_unlisted_file_is_excluded_without_anyone_adding_a_rule(self):
        verdict, reason = pack.classify("some/file/nobody/thought/about.txt")
        assert verdict == "exclude"
        assert "allowlist" in reason

    def test_denial_beats_the_allowlist(self):
        """``src/**/*.py`` allows source. It must not allow a checkpoint."""
        verdict, reason = pack.classify("src/training/checkpoint.pt")
        assert verdict == "exclude"
        assert "denied by" in reason

    def test_a_report_directory_is_denied_whole_not_file_by_file(self):
        verdict, _ = pack.classify(
            "data/reports/16_longrun/exp999/some/deep/new_file.json")
        assert verdict == "exclude"

    def test_the_public_release_gate_does_not_travel(self):
        """It asserts against a tree that has the private evidence."""
        verdict, reason = pack.classify("tests/test_public_snapshot.py")
        assert verdict == "exclude"
        assert "denied by" in reason


class TestWhatTheNodeActuallyNeeds:
    """Default deny is only safe if the pack still runs."""

    @pytest.mark.parametrize("rel", [
        "src/model_ids.py",
        "src/training/gates.py",
        "src/training/gate_suite.py",
        "src/training/pack.py",
        "src/training/gpu_node.py",
        "src/training/hypotheses.py",
        "src/generation/brickgpt.py",
        "scripts/19_gpu_gate.py",
        "scripts/18_gpu_pack.py",
        "tests/test_gate_suite.py",
        "tests/test_arms.py",
        "tests/test_final_run.py",
        "src/training/arms.py",
        "src/training/final_run.py",
        "scripts/22_final_train.py",
        "requirements.txt",
        "LICENSE",
    ])
    def test_it_is_included(self, rel):
        verdict, reason = pack.classify(rel)
        assert verdict == "include", f"{rel} was {verdict}: {reason}"

    def test_src_star_star_matches_a_file_directly_under_src(self):
        """The bug module 17 had to fix: ``**/`` must match zero directories.

        ``src/model_ids.py`` is imported by every arm. A glob that quietly
        required a subdirectory would build a pack that looks clean and does
        not import.
        """
        assert pack.classify("src/model_ids.py")[0] == "include"
        assert pack.classify("src/training/lora.py")[0] == "include"

    def test_development_scripts_do_not_travel(self):
        """The node executes. It does not rebuild the dataset.

        Shipping the dataset builders would make the node able to produce its
        own data, which is the second development source the project forbids.
        """
        for rel in ("scripts/01_eda.py", "scripts/03_build_splits.py",
                    "scripts/10_build_instruction_data.py",
                    "scripts/16_longrun.py", "scripts/15_mps_order.py"):
            assert pack.classify(rel)[0] == "exclude", rel


class TestScansAreModule17s:
    """One definition of what an identifier looks like, not two."""

    def test_every_pack_scan_kind_exists_in_module_17(self):
        m17 = pack.snapshot_module()
        known = {kind for kind, _ in m17.SCANS}
        missing = sorted(set(pack.PACK_SCAN_KINDS) - known)
        assert not missing, (
            f"{missing} are scanned for by the pack but no longer exist in "
            "module 17's SCANS. Renaming a kind there must fail here rather "
            "than silently scanning for nothing.")

    def test_the_four_categories_the_brief_names_are_all_covered(self):
        for kind in ("credential", "email", "personal-path", "organization-id"):
            assert kind in pack.PACK_SCAN_KINDS

    def test_the_approval_table_is_module_17s(self):
        m17 = pack.snapshot_module()
        assert pack.approved_hits() is m17.APPROVED_HITS or \
            pack.approved_hits() == m17.APPROVED_HITS

    @pytest.mark.parametrize("kind,text", [
        ("credential", "hf_AAAABBBBAAAABBBBAAAABBBB"),
        ("email", "someone@example.invalid"),
        ("organization-id", "org-AAAABBBBAAAABBBB"),
        ("personal-path", "/Users/someone/private/thing.txt"),
    ])
    def test_an_unapproved_identifier_in_a_packed_file_refuses_the_build(
            self, tmp_path, kind, text):
        root = make_tree(tmp_path / "root", dict(
            MINIMAL, **{"src/training/leaky.py": f'X = "{text}"\n'}))
        problems = pack.pack_audit(
            [e["path"] for e in pack.manifest(root)["include"]], root=root)
        assert any("leaky.py" in p for p in problems), \
            f"a {kind} in a packed file did not refuse: {problems}"

    def test_the_audit_names_the_kind_and_the_line(self, tmp_path):
        root = make_tree(tmp_path / "root", dict(
            MINIMAL,
            **{"src/training/leaky.py": "\n\nEMAIL = 'a@example.invalid'\n"}))
        problems = pack.pack_audit(["src/training/leaky.py"], root=root)
        assert problems
        assert "email" in problems[0]
        assert "line 3" in problems[0]

    def test_a_kind_outside_the_pack_list_is_not_audited(self, tmp_path):
        """A wall-clock instant is a research identifier, not a credential.

        The pack goes to a machine the user owns. Auditing for the public
        repository's *publication* concerns here would refuse builds for
        reasons the brief does not ask for -- and, worse, would need its own
        approval table beside module 17's.

        The literal below is an invented instant. A UUID would have made the
        point equally well and is deliberately not used: the public gate
        refuses anything UUID-shaped in a published file outright, above the
        approval table, and a test fixture is not a reason to soften that.
        """
        for kind in ("dataset-uuid", "hex32", "precise-timestamp"):
            assert kind not in pack.PACK_SCAN_KINDS
        root = make_tree(tmp_path / "root", dict(
            MINIMAL,
            **{"src/training/u.py":
               'WHEN = "2001-01-01T00:00:00+00:00"\n'}))
        assert pack.pack_audit(["src/training/u.py"], root=root) == []

    def test_a_stale_approval_is_reported(self, tmp_path):
        """An approval that no longer matches anything is a lie about the file."""
        root = make_tree(tmp_path / "root", MINIMAL)
        approvals = {"src/training/gates.py": {"email|" + "0" * 16: 1}}
        problems = pack.pack_audit(["src/training/gates.py"], root=root,
                                   approved=approvals)
        assert any("stale" in p for p in problems), problems


class TestBuildRefusesRatherThanWarns:

    def test_it_copies_nothing_when_it_refuses(self, tmp_path):
        root = make_tree(tmp_path / "root", dict(
            MINIMAL,
            **{"src/training/leaky.py": 'T = "hf_AAAABBBBAAAABBBBAAAABBBB"\n'}))
        dest = tmp_path / "dest"
        with pytest.raises(pack.PackRefused):
            pack.build(dest, root=root)
        assert not dest.exists() or not any(dest.iterdir()), \
            "a refused build left files behind"

    def test_the_destination_may_not_be_inside_the_source_tree(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        with pytest.raises(pack.PackRefused) as exc:
            pack.build(root / "inner", root=root)
        assert "inside the source tree" in str(exc.value)

    def test_the_destination_may_not_already_hold_anything(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "leftover.txt").write_text("x")
        with pytest.raises(pack.PackRefused) as exc:
            pack.build(dest, root=root)
        assert "not empty" in str(exc.value)

    def test_an_allowlisted_symlink_refuses(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        # Named for what it is. A variable called "secret" followed by an
        # assignment is exactly the shape the credential-assignment rule
        # looks for, and approving a variable name as a credential teaches
        # the approval table the wrong lesson.
        outside = tmp_path / "outside_file.txt"
        outside.write_text("not for the node")
        (root / "src" / "training" / "sneaky.py").symlink_to(outside)
        with pytest.raises(pack.PackRefused) as exc:
            pack.build(tmp_path / "dest", root=root)
        assert "symbolic link" in str(exc.value)

    def test_a_symlinked_destination_refuses(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        with pytest.raises(pack.PackRefused) as exc:
            pack.build(link, root=root)
        assert "symbolic link" in str(exc.value)


class TestManifest:

    def test_a_clean_tree_builds_and_verifies(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        built = pack.build(dest, root=root)
        assert (dest / pack.MANIFEST_NAME).exists()
        assert pack.verify(dest) == []
        assert built["files"]
        for rel in built["files"]:
            assert (dest / rel).exists(), f"{rel} is in the manifest, not on disk"

    def test_every_entry_carries_a_digest_and_a_size(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        body = pack.build(dest, root=root)
        for rel, entry in body["files"].items():
            assert len(entry["sha256"]) == 64
            assert entry["sha256"] == sha256_file(root / rel)
            assert entry["bytes"] == (root / rel).stat().st_size

    def test_the_snapshot_name_is_the_path_so_the_pack_is_runnable(self, tmp_path):
        """Report 15's snapshot flattens ``a/b.py`` to ``a__b.py``.

        A pack has to be importable, so it keeps the tree -- and says so in
        the same field, which is what lets ``verify_sources`` check it without
        a second verifier being written.
        """
        root = make_tree(tmp_path / "root", MINIMAL)
        body = pack.build(tmp_path / "dest", root=root)
        for rel, entry in body["files"].items():
            assert entry["snapshot_name"] == rel

    def test_the_pack_digest_covers_every_file(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        first = pack.build(tmp_path / "d1", root=root)
        (root / "src" / "training" / "gates.py").write_text("GATE = 9\n")
        second = pack.build(tmp_path / "d2", root=root)
        assert first["files_digest"] != second["files_digest"]
        assert first["pack_digest"] != second["pack_digest"]

    def test_the_files_digest_is_the_existing_one(self, tmp_path):
        """``session.manifest_digest``, not a second digest of the same idea."""
        root = make_tree(tmp_path / "root", MINIMAL)
        body = pack.build(tmp_path / "dest", root=root)
        assert body["files_digest"] == manifest_digest(body)

    def test_the_manifest_declares_its_schema_and_kind(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        body = pack.build(tmp_path / "dest", root=root)
        assert body["schema_version"] == pack.SCHEMA_VERSION
        assert body["kind"] == pack.KIND

    def test_the_manifest_carries_no_personal_path(self, tmp_path):
        """It travels. It must not carry where it was built."""
        from src.training.longrun import leaked_identifiers

        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        text = (dest / pack.MANIFEST_NAME).read_text()
        assert leaked_identifiers(text) == []


class TestVerifyCatchesDrift:

    @pytest.fixture()
    def built(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        return dest

    def test_a_mutated_byte_is_refused(self, built):
        target = built / "src" / "training" / "gates.py"
        target.write_text("GATE = 8  # edited on the node\n")
        problems = pack.verify(built)
        assert any("gates.py" in p for p in problems), problems

    def test_a_missing_file_is_refused(self, built):
        (built / "src" / "training" / "gates.py").unlink()
        problems = pack.verify(built)
        assert any("gates.py" in p for p in problems), problems

    def test_an_unexpected_extra_file_is_refused(self, built):
        (built / "src" / "training" / "extra.py").write_text("x = 1\n")
        problems = pack.verify(built)
        assert any("extra.py" in p for p in problems), problems

    def test_an_extra_file_is_refused_even_where_nothing_imports_it(self, built):
        """"It is only a note" is how a pack stops being the audited pack."""
        (built / "NOTES.txt").write_text("ran it by hand once\n")
        assert pack.verify(built)

    def test_a_rewritten_manifest_that_matches_itself_is_still_refused(self, built):
        """Recomputing the digests after editing a file must not launder it.

        ``files_digest`` is derived from ``files``, so an attacker who edits
        both is self-consistent. ``pack_digest`` is checked against the
        *stated* one, so the rewrite has to be complete -- and the point of
        this test is that the incomplete rewrite, which is the realistic one,
        is caught.
        """
        target = built / "src" / "training" / "gates.py"
        target.write_text("GATE = 99\n")
        body = json.loads((built / pack.MANIFEST_NAME).read_text())
        body["files"]["src/training/gates.py"]["sha256"] = sha256_file(target)
        (built / pack.MANIFEST_NAME).unlink()
        (built / pack.MANIFEST_NAME).write_text(json.dumps(body, indent=2))
        problems = pack.verify(built)
        assert any("files_digest" in p or "pack_digest" in p for p in problems), \
            problems

    def test_a_missing_manifest_is_refused(self, built):
        (built / pack.MANIFEST_NAME).unlink()
        problems = pack.verify(built)
        assert problems
        assert any("manifest" in p for p in problems)

    def test_an_unreadable_manifest_is_refused_not_treated_as_empty(self, built):
        (built / pack.MANIFEST_NAME).unlink()
        (built / pack.MANIFEST_NAME).write_text("{not json")
        problems = pack.verify(built)
        assert problems

    def test_verify_returns_a_list_not_a_boolean(self, built):
        """A caller that forgets to look at *why* still gets a truthy refusal."""
        assert pack.verify(built) == []
        (built / "src" / "training" / "gates.py").unlink()
        assert isinstance(pack.verify(built), list)


class TestDataTravelsAsDigestsOnly:
    """The pack carries no dataset. It carries what the dataset must hash to."""

    def test_no_data_file_is_ever_included(self, tmp_path):
        root = make_tree(tmp_path / "root", dict(
            MINIMAL, **{"data/processed/instruct_inv_train.jsonl": '{"a":1}\n'}))
        included = [e["path"] for e in pack.manifest(root)["include"]]
        assert not [p for p in included if p.startswith("data/")]

    def test_the_required_data_is_named_and_digested(self, tmp_path):
        root = make_tree(tmp_path / "root", dict(
            MINIMAL,
            **{f"{rel}": '{"sample_id":"x"}\n' for rel in pack.REQUIRED_DATA}))
        dest = tmp_path / "dest"
        body = pack.build(dest, root=root)
        for rel in pack.REQUIRED_DATA:
            assert rel in body["data_requirements"]
            assert body["data_requirements"][rel]["sha256"] == \
                sha256_file(root / rel)

    def test_a_missing_required_data_file_records_absence_not_a_fake_digest(
            self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        body = pack.build(tmp_path / "dest", root=root)
        for rel in pack.REQUIRED_DATA:
            assert body["data_requirements"][rel]["sha256"] is None
            assert body["data_requirements"][rel]["reason"]

    def test_data_is_only_checked_when_the_caller_asks(self, tmp_path):
        """Verifying a pack in transit must not require the data beside it."""
        root = make_tree(tmp_path / "root", dict(
            MINIMAL,
            **{rel: '{"sample_id":"x"}\n' for rel in pack.REQUIRED_DATA}))
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        assert pack.verify(dest) == []
        assert pack.verify(dest, data_root=tmp_path / "nowhere") != []

    def test_drifted_data_is_refused(self, tmp_path):
        root = make_tree(tmp_path / "root", dict(
            MINIMAL,
            **{rel: '{"sample_id":"x"}\n' for rel in pack.REQUIRED_DATA}))
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        assert pack.verify(dest, data_root=root) == []
        (root / pack.REQUIRED_DATA[0]).write_text('{"sample_id":"y"}\n')
        problems = pack.verify(dest, data_root=root)
        assert any(pack.REQUIRED_DATA[0] in p for p in problems), problems

    def test_the_dataset_beside_the_pack_is_not_an_extra_file(self, tmp_path):
        """The node keeps the data where the code expects to find it.

        ``verify`` refuses unexpected files, and the dataset is the one thing
        that is legitimately present and legitimately not in the manifest. It
        is covered by the digest pin instead -- so it is checked, just not by
        being listed. If this were refused, a correctly-assembled node could
        never verify, and the fix somebody would reach for is to stop
        verifying.
        """
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        for rel in pack.REQUIRED_DATA:
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"sample_id":"x"}\n')
        assert pack.verify(dest) == []

    def test_data_recorded_as_absent_cannot_be_satisfied_later_by_anything(
            self, tmp_path):
        """A pack built without the data may not be checked against some data.

        Otherwise "the digest was never taken" and "the digest matches" become
        the same verdict, which is how an unpinned dataset gets into a run.
        """
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        other = make_tree(tmp_path / "other", {
            rel: '{"sample_id":"x"}\n' for rel in pack.REQUIRED_DATA})
        problems = pack.verify(dest, data_root=other)
        assert problems


class TestAgainstTheRealTree:
    """The rules above, applied to what is actually on this disk."""

    def test_the_real_tree_classifies_with_no_holds_and_no_surprises(self):
        m = pack.manifest(ROOT)
        assert m["include"], "the real tree produced an empty pack"
        for entry in m["include"]:
            rel = entry["path"]
            assert not rel.startswith("data/"), rel
            assert not rel.startswith("artifacts/"), rel

    def test_the_real_pack_audits_clean(self):
        """If this goes red, a new identifier appeared in packable source."""
        problems = pack.pack_audit(
            [e["path"] for e in pack.manifest(ROOT)["include"]], root=ROOT)
        assert problems == [], "\n".join(problems)

    def test_the_real_pack_contains_the_entry_point_and_its_module(self):
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        for rel in ("scripts/19_gpu_gate.py", "src/training/gates.py",
                    "src/training/gate_suite.py",
                    "src/training/gpu_node.py", "src/training/pack.py",
                    "src/training/hypotheses.py"):
            assert rel in included, rel

    def test_the_manifest_carries_every_self_contained_suite(self):
        """A decision procedure without its tests is one nothing contradicts."""
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        for rel in pack.PACKED_TEST_SUITES:
            assert rel in included, rel

    def test_the_packed_suites_are_exactly_the_four_declared(self):
        """Neither one that quietly appeared, nor one quietly dropped."""
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        packed = {rel for rel in included if rel.startswith("tests/")}
        assert packed == set(pack.PACKED_TEST_SUITES), packed
        assert len(pack.PACKED_TEST_SUITES) == 7

    def test_private_evidence_and_returns_never_enter_the_pack(self):
        """What comes back from the node must not go out again in a pack.

        ``runs/gpu_returns`` holds per-row ledgers, optimizer state, trainable
        weights and adapters from the execution node. It is the private half
        of the boundary in the strongest sense, and a pack is a thing that
        leaves this machine.
        """
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        for rel in sorted(included):
            assert not rel.startswith("runs/"), rel
            assert not rel.startswith("data/raw/"), rel
            assert not rel.startswith("data/reports/"), rel
            assert not rel.startswith("artifacts/"), rel
        for rel in ("runs/gpu_returns/pack_x/runs/gate_8/ledger.jsonl",
                    "runs/gpu_returns/pack_x/runs/gate_500/checkpoints/"
                    "000192/model_state.safetensors",
                    "runs/gate_8/plan.json",
                    "data/processed/instruct_inv_train.jsonl",
                    "data/reports/16_longrun_design.json",
                    "artifacts/checkpoints/lora_smoke/adapter_model.safetensors"):
            verdict, reason = pack.classify(rel)
            assert verdict == "exclude", f"{rel} was {verdict}: {reason}"

    def test_no_dataset_builder_travels_with_the_new_scripts(self):
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        scripts = {rel for rel in included if rel.startswith("scripts/")}
        for rel in scripts:
            assert rel in pack.PACK_ALLOW, rel

    def test_the_allowlist_and_the_declared_suites_do_not_drift(self):
        from_allowlist = {rel for rel in pack.PACK_ALLOW
                          if rel.startswith("tests/")}
        assert from_allowlist == set(pack.PACKED_TEST_SUITES)

    def test_no_packed_suite_imports_another_one(self):
        """Each travels alone, so each means the same thing on the node.

        A packed suite that imported a sibling would pass here, where the
        whole tree is present, and mean something different there.
        """
        modules = {rel.split("/")[-1][:-3] for rel in pack.PACKED_TEST_SUITES}
        for rel in pack.PACKED_TEST_SUITES:
            source = (ROOT / rel).read_text(encoding="utf-8")
            for other in modules - {rel.split("/")[-1][:-3]}:
                assert f"tests.{other}" not in source, (rel, other)
                assert f"import {other}" not in source, (rel, other)

    def test_every_packed_suite_still_imports_from_the_pack_root(self, tmp_path):
        """It travels only if it means the same thing there as here."""
        import py_compile

        dest = tmp_path / "pack"
        pack.build(dest, root=ROOT)
        for rel in pack.PACKED_TEST_SUITES:
            target = dest / rel
            assert target.is_file(), rel
            py_compile.compile(str(target), doraise=True,
                               cfile=str(target) + "c")

    def test_the_real_pack_builds_and_verifies(self, tmp_path):
        dest = tmp_path / "pack"
        body = pack.build(dest, root=ROOT)
        assert pack.verify(dest) == []
        assert body["files_digest"] == manifest_digest(body)

    def test_the_real_pack_is_importable_from_its_own_root(self, tmp_path):
        """Every module the entry point imports has to be in the pack.

        A pack that classifies beautifully and cannot import is the failure
        this catches: it is checked by compiling every packed module against
        the pack root, with no path back into the private tree.
        """
        import py_compile

        dest = tmp_path / "pack"
        pack.build(dest, root=ROOT)
        for p in sorted(dest.rglob("*.py")):
            py_compile.compile(str(p), doraise=True, cfile=str(p) + "c")

    def test_the_real_pack_verifies_itself_with_no_path_back_to_this_tree(
            self, tmp_path):
        """The failure compiling cannot see: a *runtime* reach outside the pack.

        The node has no private tree to fall back on. A module that resolves
        one of its own dependencies by absolute path into this repository
        compiles perfectly, classifies perfectly, and dies on the node the
        first time anything calls it. So this runs the pack's own verifier in
        a subprocess whose working directory and import path are the pack --
        which is exactly the node's situation and nothing like this one.
        """
        import subprocess
        import sys

        dest = tmp_path / "pack"
        pack.build(dest, root=ROOT)
        result = subprocess.run(
            [sys.executable, "scripts/18_gpu_pack.py", "--verify", "."],
            cwd=dest, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"})
        assert result.returncode == 0, (
            f"the pack could not verify itself from its own root:\n"
            f"{result.stdout}\n{result.stderr}")

    def test_the_real_pack_can_self_test_its_pipeline(self, tmp_path):
        """And the gate pipeline, likewise, from inside the pack."""

        import subprocess
        import sys

        dest = tmp_path / "pack"
        body = pack.build(dest, root=ROOT)
        result = subprocess.run(
            [sys.executable, "scripts/19_gpu_gate.py", "--self-test",
             "--run-dir", "runs/selftest",
             "--expected-pack-digest", body["pack_digest"]],
            cwd=dest, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"})
        assert result.returncode == 0, (
            f"{result.stdout}\n{result.stderr}")
        assert "verdict: passed" in result.stdout
        # And the run output must not make the pack unverifiable afterwards.
        assert pack.verify(dest) == []


class TestTheManifestCannotNameAnythingOutsideThePack:
    """A digest proves bytes. It does not prove *which file's* bytes.

    ``verify_sources`` resolves each entry as ``dest / snapshot_name``. Left
    unconstrained, that name is an instruction to look anywhere on the
    machine -- so a manifest can point at a file outside the pack, and as long
    as the bytes there match the digest beside it, every check passes. The
    pack is then "verified" while the code that will actually run came from
    somewhere nobody audited.
    """

    def build_pack(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        pack.build(dest, root=root)
        return dest

    def rewrite(self, dest, body):
        """Re-sign the manifest completely: both internal digests recomputed."""
        body["files_digest"] = manifest_digest(body)
        body["pack_digest"] = pack.pack_digest(body)
        (dest / pack.MANIFEST_NAME).unlink()
        (dest / pack.MANIFEST_NAME).write_text(json.dumps(body, indent=2))
        return body

    def read(self, dest):
        return json.loads((dest / pack.MANIFEST_NAME).read_text())

    def test_the_reproducible_escape_is_refused(self, tmp_path):
        """The exact case: escape, delete the original, plant a twin outside.

        Before this was closed, ``verify`` returned no problems at all: the
        manifest was internally consistent, the digest matched, and the bytes
        it matched were a file one directory above the pack.
        """
        dest = self.build_pack(tmp_path)
        target = "src/training/gates.py"
        body = self.read(dest)
        body["files"][target]["snapshot_name"] = "../outside.py"
        (dest / target).unlink()
        (tmp_path / "outside.py").write_text("GATE = 8\n")
        self.rewrite(dest, body)

        problems = pack.verify(dest)
        assert problems, "the pack verified against a file outside itself"
        assert any(".." in p or "outside" in p or "escape" in p
                   for p in problems), problems

    def test_an_escaping_files_key_is_refused_too(self, tmp_path):
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        entry = body["files"].pop("src/training/gates.py")
        entry["snapshot_name"] = "../outside.py"
        body["files"]["../outside.py"] = entry
        (tmp_path / "outside.py").write_text("GATE = 8\n")
        self.rewrite(dest, body)
        assert pack.verify(dest)

    @pytest.mark.parametrize("bad", [
        "/etc/passwd",
        "/absolute/path.py",
        "../outside.py",
        "src/../../outside.py",
        "./src/training/gates.py",
        "src/./training/gates.py",
        "src//training/gates.py",
        "src\\training\\gates.py",
        "C:\\Users\\someone\\thing.py",
        "src/training/",
        "",
        "   ",
        "..",
        ".",
    ])
    def test_path_problems_rejects_every_unsafe_shape(self, bad):
        assert pack.path_problems(bad), f"{bad!r} was accepted"

    @pytest.mark.parametrize("good", [
        "requirements.txt",
        "src/model_ids.py",
        "src/training/gates.py",
        "tests/test_pack.py",
        "a/b/c/d.txt",
    ])
    def test_path_problems_accepts_ordinary_relative_paths(self, good):
        assert pack.path_problems(good) == []

    @pytest.mark.parametrize("bad", [123, None, True, ["src/x.py"], {"a": 1}])
    def test_a_non_string_path_is_refused_not_coerced(self, bad):
        assert pack.path_problems(bad)

    def test_snapshot_name_must_equal_the_key_exactly(self, tmp_path):
        """A pack is importable, so its copies keep the tree.

        Any divergence between the key and the name is an entry whose digest
        describes one file and whose location describes another, which is the
        whole class of confusion this closes.
        """
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        body["files"]["src/training/gates.py"]["snapshot_name"] = \
            "src/training/other.py"
        (dest / "src" / "training" / "other.py").write_text("GATE = 8\n")
        self.rewrite(dest, body)
        problems = pack.verify(dest)
        assert problems
        assert any("snapshot_name" in p for p in problems), problems

    def test_a_missing_snapshot_name_is_refused_rather_than_defaulted(
            self, tmp_path):
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        del body["files"]["src/training/gates.py"]["snapshot_name"]
        self.rewrite(dest, body)
        assert pack.verify(dest)

    def test_a_symlinked_file_in_the_manifest_is_refused(self, tmp_path):
        dest = self.build_pack(tmp_path)
        outside = tmp_path / "twin.py"
        outside.write_text("GATE = 8\n")
        target = dest / "src" / "training" / "gates.py"
        target.unlink()
        target.symlink_to(outside)
        problems = pack.verify(dest)
        assert problems
        assert any("symbolic link" in p or "symlink" in p for p in problems), \
            problems

    def test_a_symlinked_parent_directory_is_refused(self, tmp_path):
        """The file is real. Its directory is not, and that is enough."""
        dest = self.build_pack(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "gates.py").write_text("GATE = 8\n")
        (elsewhere / "__init__.py").write_text("")
        real = dest / "src" / "training"
        for child in real.iterdir():
            child.unlink()
        real.rmdir()
        real.symlink_to(elsewhere)
        problems = pack.verify(dest)
        assert problems
        assert any("symbolic link" in p or "symlink" in p for p in problems), \
            problems

    def test_the_recorded_byte_count_is_checked(self, tmp_path):
        """A size that disagrees with the file is a manifest that is wrong.

        The digest would catch a changed file. It does not catch a manifest
        whose *own* record of the file is inconsistent, and an entry nobody
        checks is an entry anybody can write.
        """
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        body["files"]["src/training/gates.py"]["bytes"] = 999999
        self.rewrite(dest, body)
        problems = pack.verify(dest)
        assert problems
        assert any("bytes" in p for p in problems), problems

    def test_a_missing_or_nonsense_byte_count_is_refused(self, tmp_path):
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        body["files"]["src/training/gates.py"]["bytes"] = "lots"
        self.rewrite(dest, body)
        assert pack.verify(dest)

    def test_data_requirement_keys_must_be_exactly_the_declared_set(
            self, tmp_path):
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        body["data_requirements"]["data/processed/extra.jsonl"] = {
            "sha256": "a" * 64, "bytes": 1, "reason": None}
        self.rewrite(dest, body)
        problems = pack.verify(dest)
        assert problems
        assert any("data_requirements" in p for p in problems), problems

    def test_a_dropped_data_requirement_is_refused(self, tmp_path):
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        body["data_requirements"].pop(pack.REQUIRED_DATA[0])
        self.rewrite(dest, body)
        assert pack.verify(dest)

    def test_a_data_requirement_cannot_escape_either(self, tmp_path):
        dest = self.build_pack(tmp_path)
        body = self.read(dest)
        entry = body["data_requirements"].pop(pack.REQUIRED_DATA[0])
        body["data_requirements"]["../../elsewhere.jsonl"] = entry
        self.rewrite(dest, body)
        assert pack.verify(dest)

    def test_a_built_pack_always_records_the_path_as_the_snapshot_name(
            self, tmp_path):
        body = pack.build(tmp_path / "dest",
                          root=make_tree(tmp_path / "root", MINIMAL))
        assert pack.manifest_path_problems(body) == []
        for rel, entry in body["files"].items():
            assert entry["snapshot_name"] == rel
            assert pack.path_problems(rel) == []


class TestTheDigestHasToComeFromSomewhereElse:
    """Self-consistency is not trust. It is only arithmetic.

    ``files_digest`` is derived from ``files`` and ``pack_digest`` from both,
    so anyone who can write the manifest can make all three agree. What makes
    a digest mean anything is that it was carried from the Mac by a different
    route and is compared against what arrived.
    """

    def test_a_well_formed_digest_is_accepted(self):
        assert pack.expected_digest_problems("a" * 64) == []
        assert pack.expected_digest_problems("0123456789abcdef" * 4) == []

    @pytest.mark.parametrize("bad", [
        None, "", "   ", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
        "a" * 32, 12345, True, ["a" * 64], "sha256:" + "a" * 64,
        "A1B2" + "a" * 60,
    ])
    def test_anything_else_is_refused(self, bad):
        problems = pack.expected_digest_problems(bad)
        assert problems, f"{bad!r} was accepted as a digest"

    def test_a_missing_digest_is_its_own_sentence(self):
        problems = pack.expected_digest_problems(None)
        assert any("no expected" in p or "not given" in p or "missing" in p
                   for p in problems), problems

    def test_a_matching_digest_passes(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        body = pack.build(dest, root=root)
        assert pack.trusted_digest_problems(dest, body["pack_digest"]) == []

    def test_a_completely_re_signed_pack_is_still_refused(self, tmp_path):
        """The case that matters: every file, the manifest and every internal
        digest rewritten so the pack is perfectly self-consistent -- and it is
        still refused, because the value the Mac printed does not match.
        """
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        original = pack.build(dest, root=root)
        trusted = original["pack_digest"]

        # Rewrite every packed file, then re-sign completely.
        body = json.loads((dest / pack.MANIFEST_NAME).read_text())
        for rel, entry in body["files"].items():
            path = dest / rel
            path.write_text(f"# replaced\n{path.name}\n", encoding="utf-8")
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        body["files_digest"] = manifest_digest(body)
        body["pack_digest"] = pack.pack_digest(body)
        (dest / pack.MANIFEST_NAME).unlink()
        (dest / pack.MANIFEST_NAME).write_text(json.dumps(body, indent=2))

        # Internally flawless...
        assert pack.verify(dest) == [], \
            "this test is meant to start from a self-consistent pack"
        # ...and still refused.
        problems = pack.trusted_digest_problems(dest, trusted)
        assert problems
        assert any("pack_digest" in p or "expected" in p for p in problems), \
            problems

    def test_a_pack_with_no_manifest_cannot_satisfy_a_digest(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        dest = tmp_path / "dest"
        body = pack.build(dest, root=root)
        (dest / pack.MANIFEST_NAME).unlink()
        assert pack.trusted_digest_problems(dest, body["pack_digest"])

    def test_a_malformed_expected_digest_refuses_before_reading_anything(
            self, tmp_path):
        assert pack.trusted_digest_problems(tmp_path / "nowhere", "nope")


class TestTheDependencyReportIsUsableAndSafe:
    """The Mac side of the SHA-verified dependency flow.

    The node's preflight refuses when a pinned file is absent from its cache.
    For that refusal to be *actionable* there has to be a way to see, on the
    machine that has them, what those files are and what they hash to -- so
    the operator can carry the digests across and compare. That is all this
    mode does: it resolves, it reports, and it never fetches.
    """

    def run_cli(self, *args):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "scripts/18_gpu_pack.py", *args],
            cwd=ROOT, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1",
                 "TRANSFORMERS_OFFLINE": "1"})

    def test_it_runs_and_reports_rather_than_crashing(self):
        result = self.run_cli("--dependencies")
        assert result.returncode in (0, 1), (
            f"exit {result.returncode}\n{result.stdout}\n{result.stderr}")
        assert result.stdout.strip(), "it reported nothing at all"

    def test_the_report_names_no_cache_path_and_no_account(self):
        """It is carried to the other machine, so it must be safe to carry.

        A resolved dependency lives under a home directory. Reporting *where*
        it lives would put this machine's account into a document whose whole
        purpose is to travel.
        """
        from src.training.longrun import leaked_identifiers

        result = self.run_cli("--dependencies")
        leaks = leaked_identifiers(result.stdout + result.stderr)
        assert leaks == [], f"the dependency report leaked {leaks}"

    def test_it_names_the_pinned_revisions_it_resolved_against(self):
        from src.model_ids import ADAPTER, BASE_MODEL

        out = self.run_cli("--dependencies").stdout
        assert BASE_MODEL in out
        assert ADAPTER in out

    def test_it_declares_that_it_fetched_nothing(self):
        out = self.run_cli("--dependencies").stdout.lower()
        assert "network" in out or "fetch" in out or "offline" in out


EVIDENCE = {
    "schema_version": 1, "kind": "longrun_dependency_preflight",
    "network_used": False, "tensors_loaded": False, "device_initialised": False,
    "repositories": [
        {"repo_id": "Vendor/Tok", "revision": "a" * 40,
         "files": [{"name": "tokenizer.json", "bytes": 10, "sha256": "1" * 64},
                   {"name": "tokenizer_config.json", "bytes": 20,
                    "sha256": "2" * 64}]},
        {"repo_id": "Vendor/Base", "revision": "b" * 40,
         "files": [{"name": "config.json", "bytes": 30, "sha256": "3" * 64}]},
        # Same repo and revision as the first, different files. The adapter and
        # the tokenizer really do come from one repository, so the digest has
        # to keep both entries rather than collapse them by repo id.
        {"repo_id": "Vendor/Tok", "revision": "a" * 40,
         "files": [{"name": "adapter_model.safetensors", "bytes": 40,
                    "sha256": "4" * 64}]},
    ],
    "instruction_pool": {"path": "data/processed/instruct_inv_train.jsonl",
                         "sha256": "5" * 64},
}


def mutate(**changes):
    """A deep copy of EVIDENCE with one thing changed."""
    import copy

    out = copy.deepcopy(EVIDENCE)
    for path, value in changes.items():
        target = out
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else target[part]
        last = parts[-1]
        target[int(last) if last.isdigit() else last] = value
    return out


class TestTheDependencyDigestIsCanonical:
    """One value standing for *which bytes* the dependencies are.

    The node already refuses when a pinned file is absent. It could not, until
    now, tell a present file from the *right* present file: a tokenizer.json
    of the correct name resolves perfectly well whatever is in it. This digest
    is what binds the node's cache to the one the pack was built against.
    """

    def digest(self, evidence):
        from src.training.longrun import dependency_digest

        return dependency_digest(evidence)

    def test_it_is_a_lowercase_sha256(self):
        value = self.digest(EVIDENCE)
        assert len(value) == 64
        assert value == value.lower()
        assert pack.expected_digest_problems(value) == []

    def test_it_is_stable_across_calls(self):
        assert self.digest(EVIDENCE) == self.digest(EVIDENCE)

    def test_the_order_of_repositories_does_not_change_it(self):
        import copy

        shuffled = copy.deepcopy(EVIDENCE)
        shuffled["repositories"] = list(reversed(shuffled["repositories"]))
        assert self.digest(shuffled) == self.digest(EVIDENCE)

    def test_the_order_of_files_within_a_repository_does_not_change_it(self):
        import copy

        shuffled = copy.deepcopy(EVIDENCE)
        shuffled["repositories"][0]["files"] = list(
            reversed(shuffled["repositories"][0]["files"]))
        assert self.digest(shuffled) == self.digest(EVIDENCE)

    def test_unrelated_evidence_fields_do_not_change_it(self):
        """It is about the dependencies, not about the run that read them."""
        import copy

        other = copy.deepcopy(EVIDENCE)
        other["network_used"] = True
        other["schema_version"] = 99
        assert self.digest(other) == self.digest(EVIDENCE)

    @pytest.mark.parametrize("change", [
        {"repositories.0.repo_id": "Vendor/Other"},
        {"repositories.0.revision": "c" * 40},
        {"instruction_pool.path": "data/processed/other.jsonl"},
        {"instruction_pool.sha256": "9" * 64},
    ])
    def test_changing_identity_changes_it(self, change):
        assert self.digest(mutate(**change)) != self.digest(EVIDENCE)

    @pytest.mark.parametrize("field,value", [
        ("sha256", "9" * 64), ("bytes", 999), ("name", "renamed.json"),
    ])
    def test_changing_any_file_attribute_changes_it(self, field, value):
        import copy

        changed = copy.deepcopy(EVIDENCE)
        changed["repositories"][0]["files"][0][field] = value
        assert self.digest(changed) != self.digest(EVIDENCE)

    def test_the_second_entry_for_the_same_repository_is_covered(self):
        """The adapter's digest must matter as much as the tokenizer's."""
        import copy

        changed = copy.deepcopy(EVIDENCE)
        changed["repositories"][2]["files"][0]["sha256"] = "9" * 64
        assert self.digest(changed) != self.digest(EVIDENCE)

    def test_dropping_a_file_changes_it(self):
        import copy

        changed = copy.deepcopy(EVIDENCE)
        changed["repositories"][0]["files"].pop()
        assert self.digest(changed) != self.digest(EVIDENCE)

    def test_dropping_a_repository_changes_it(self):
        import copy

        changed = copy.deepcopy(EVIDENCE)
        changed["repositories"].pop()
        assert self.digest(changed) != self.digest(EVIDENCE)

    def test_empty_or_absent_evidence_still_digests_rather_than_raising(self):
        """A checker that returned nothing must not crash the comparison.

        It must produce *a* value, and one that no real cache can match --
        which is what makes the mismatch a refusal rather than an exception.
        """
        for empty in (None, {}, {"repositories": [], "instruction_pool": {}}):
            value = self.digest(empty)
            assert len(value) == 64
            assert value != self.digest(EVIDENCE)

    def test_it_lives_beside_the_preflight_it_digests(self):
        """One implementation, in the module that produces the evidence."""
        from src.training import longrun

        assert hasattr(longrun, "dependency_digest")


class TestTheDependencyDigestIsNotKeptWithTheFilesItDescribes:
    """Requirement 4: it must not travel in the pack's own manifest.

    A digest stored beside the thing it authenticates authenticates nothing.
    Whoever rewrote the pack rewrites the field too -- which is exactly the
    failure the pack digest already had to be rescued from, and repeating it
    one layer down would be the same mistake with a different name.
    """

    def test_the_pack_manifest_carries_no_dependency_digest(self, tmp_path):
        root = make_tree(tmp_path / "root", MINIMAL)
        body = pack.build(tmp_path / "dest", root=root)
        flat = json.dumps(body)
        assert "dependency_digest" not in flat
        assert "dependency" not in body

    def test_the_built_manifest_has_the_fields_it_always_had(self, tmp_path):
        """A guard on the other side: nothing was quietly added to it."""
        root = make_tree(tmp_path / "root", MINIMAL)
        body = pack.build(tmp_path / "dest", root=root)
        assert set(body) == {
            "schema_version", "kind", "created_at", "root_relative", "files",
            "files_digest", "data_requirements", "data_digest", "pack_digest"}


class TestTheMacPrintsTheDependencyDigest:
    """Requirement 2: the operator has to be able to read it off and carry it."""

    def run_cli(self, *args):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "scripts/18_gpu_pack.py", *args],
            cwd=ROOT, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1",
                 "TRANSFORMERS_OFFLINE": "1"})

    def test_it_prints_a_labelled_dependency_digest(self):
        out = self.run_cli("--dependencies").stdout
        assert "dependency_digest" in out

    def test_the_printed_value_is_a_well_formed_digest(self):
        import re

        out = self.run_cli("--dependencies").stdout
        found = re.search(r"dependency_digest\s+([0-9a-f]{64})", out)
        assert found, f"no readable digest in:\n{out}"
        assert pack.expected_digest_problems(found.group(1)) == []

    def test_the_printed_value_is_what_the_node_will_recompute(self):
        """Same evidence, same function, both sides. Not two conventions."""
        import re

        from src.training.longrun import (dependency_digest,
                                          dependency_preflight)

        out = self.run_cli("--dependencies").stdout
        found = re.search(r"dependency_digest\s+([0-9a-f]{64})", out)
        assert found
        assert found.group(1) == dependency_digest(
            dependency_preflight()["evidence"])

    def test_it_says_the_value_travels_separately(self):
        """Requirement 4, said where the operator will actually read it."""
        out = self.run_cli("--dependencies").stdout.lower()
        assert "independent channel" in out or "separate" in out
        assert "manifest" in out, \
            "it does not say the value must stay out of the pack manifest"


class TestThePackPinsOnlyDataItsOwnCodeCanRead:
    """A pin the packed code never opens is a file shipped for nothing.

    It has to be transferred, it sits on the execution node indefinitely, and
    every verify checks it -- while no module in the pack can read it. That
    over-broad contract is not academic: it is what put a validation split on
    the execution node when only the training split was ever in scope, and it
    did so while every check passed, because the manifest genuinely required
    the file it should never have required.
    """

    #: The module that *declares* the pin. Excluded from the search, because
    #: ``REQUIRED_DATA`` naming its own entries would make the check circular
    #: and self-satisfying -- which it was, on the first attempt.
    DECLARING_MODULE = "src/training/pack.py"

    def packed_sources(self) -> str:
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        return "\n".join(
            (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for rel in sorted(included)
            if rel.endswith(".py") and rel != self.DECLARING_MODULE)

    def test_every_pinned_data_file_is_named_by_packed_code(self):
        sources = self.packed_sources()
        unread = [rel for rel in pack.REQUIRED_DATA
                  if Path(rel).name not in sources]
        assert unread == [], (
            f"{unread} are pinned by REQUIRED_DATA but no module in the pack "
            "names them, so nothing shipped can read them. Either the pack "
            "needs the code that uses them, or the pin is too wide.")

    def test_the_declaring_module_is_actually_in_the_pack(self):
        """Otherwise the exclusion above silently searches the wrong set."""
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        assert self.DECLARING_MODULE in included

    def test_the_training_split_is_still_pinned(self):
        """The other direction: narrowing must not drop what is used."""
        sources = self.packed_sources()
        assert "instruct_inv_train.jsonl" in sources
        assert any("instruct_inv_train.jsonl" in rel
                   for rel in pack.REQUIRED_DATA)

    def test_the_pin_and_the_dependency_evidence_agree_on_the_pool(self):
        """``dependency_digest`` covers one instruction pool. Same one."""
        from src.training.longrun import dependency_preflight

        pool = dependency_preflight()["evidence"]["instruction_pool"]["path"]
        assert pool in pack.REQUIRED_DATA, (
            f"the dependency evidence pins {pool!r} but REQUIRED_DATA does "
            "not, so the two sides disagree about which data is the data")


# ---------------------------------------------------------------------------
# Round 49: the payload is the training code, and it is measured to be
#
# ``src/**/*.py`` shipped every module in the project, so a pack whose whole
# job is to run a fit from a read-only payload also carried the image track,
# the retrieval track and an HTTP server -- including src/vision/net.py, the
# one module here that opens an outbound connection. Nothing the node runs
# imports any of it. This computes the real import closure of the pack's own
# entry points and holds the allowlist to it in both directions.
# ---------------------------------------------------------------------------

class TestThePayloadIsTheTrainingCodeAndNothingElse:
    def test_no_denied_subtree_is_reachable_from_what_the_node_runs(self):
        """The measurement the denial rests on."""
        closure = pack.import_closure()
        assert closure, "the closure came back empty; the reader is broken"
        reached = sorted(
            name for name in closure
            if any(name.startswith(f"{subtree}/")
                   for subtree in pack.NON_PACKED_SUBTREES))
        assert reached == [], (
            "a packed entry point now imports a subtree the pack denies: "
            f"{reached}. Either the import is a mistake or the subtree has to "
            "start travelling; it cannot be both denied and needed")

    def test_every_module_the_node_runs_still_travels(self):
        closure = pack.import_closure()
        for name in sorted(closure):
            action, why = pack.classify(name)
            assert action == "include", (
                f"{name} is in the import closure of what the node runs and "
                f"the pack does not carry it ({why})")

    def test_the_denied_subtrees_classify_as_denied(self):
        for subtree in pack.NON_PACKED_SUBTREES:
            probe = f"{subtree}/anything.py"
            action, why = pack.classify(probe)
            assert action == "exclude", (probe, action, why)
            assert why != "not on the allowlist", (
                f"{probe} is excluded only because nothing matched it; it "
                "should be denied by name, with a reason")

    def test_the_outbound_http_module_does_not_travel(self):
        """Named on its own: it is the only one that opens a connection."""
        action, why = pack.classify("src/vision/net.py")
        assert action == "exclude"
        assert "outbound HTTP" in why

    def test_the_interface_server_does_not_travel(self):
        assert pack.classify("src/ui/server_full.py")[0] == "exclude"
        assert pack.classify("src/ui/app.py")[0] == "exclude"

    def test_the_preview_does_not_travel(self):
        action, why = pack.classify("src/rendering/preview.py")
        assert action == "exclude"
        assert "Matplotlib" in why

    def test_the_training_modules_do_travel(self):
        for name in ("src/training/longrun.py", "src/training/gates.py",
                     "src/eval/acceptance.py", "src/rendering/ldr.py",
                     "src/data/bricks.py"):
            assert pack.classify(name)[0] == "include", name

    def test_the_named_subtrees_and_the_denials_are_one_list(self):
        patterns = {pattern for pattern, _why in pack.PACK_DENY}
        for subtree in pack.NON_PACKED_SUBTREES:
            assert f"{subtree}/**" in patterns, subtree

    def test_the_entry_points_all_exist(self):
        for entry in pack.PACK_ENTRY_POINTS:
            assert (pack.ROOT / entry).is_file(), entry

    def test_a_subtree_that_is_neither_reached_nor_denied_is_reported(self):
        """A new src package has to be a deliberate decision, not a default."""
        closure = pack.import_closure()
        reached = {name.split("/")[1] for name in closure
                   if name.startswith("src/") and "/" in name[4:]}
        denied = {subtree.split("/")[1]
                  for subtree in pack.NON_PACKED_SUBTREES}
        present = {child.name for child in (pack.ROOT / "src").iterdir()
                   if child.is_dir() and not child.name.startswith("_")
                   and any(child.glob("*.py"))}
        undecided = sorted(present - reached - denied)
        assert undecided == [], (
            f"src/{undecided} is neither reached by anything the node runs "
            "nor named in NON_PACKED_SUBTREES. Decide which, rather than "
            "letting src/**/*.py answer it")

    def test_the_closure_reader_follows_a_chain(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "__init__.py").write_text("")
        (tmp_path / "src" / "a.py").write_text("from src.b import thing\n")
        (tmp_path / "src" / "b.py").write_text("thing = 1\n")
        (tmp_path / "entry.py").write_text("import src.a\n")
        found = pack.import_closure(root=tmp_path, entry_points=("entry.py",))
        assert "src/a.py" in found and "src/b.py" in found

    def test_the_closure_reader_ignores_a_missing_entry_point(self, tmp_path):
        assert pack.import_closure(root=tmp_path,
                                   entry_points=("nope.py",)) == set()
