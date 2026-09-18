"""Adversarial checks for the historical snapshot and post-score layer."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import shutil
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evidence_chain_v2_tests", ROOT / "scripts/71_evidence_chain.py"
)
E = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E)

ARTIFACT_ONLY = "artifact-only:"
needs_private_evidence = pytest.mark.skipif(
    not (ROOT / E.SOURCE_BINDING).is_file(),
    reason=f"{ARTIFACT_ONLY} the private evidence-v2 binding is not published",
)


def _binding() -> dict:
    return json.loads((ROOT / E.SOURCE_BINDING).read_text())


def test_execution_grant_path_uses_noncredential_internal_names():
    assert E.EXECUTION_GRANT == E.PHASE3C_DIR / "execution_authorization.json"
    assert not hasattr(E, "AUTHORIZATION")
    assert "authorization_path" in inspect.signature(E.build_phase3c_snapshot).parameters
    assert "authorization_path" in inspect.signature(E.historical_snapshot_problems).parameters


def _rehash(body: dict, field: str) -> dict:
    body[field] = E.digest_of({key: value for key, value in body.items() if key != field})
    return body


@needs_private_evidence
def test_historical_snapshot_recreates_all_sixty_frozen_members():
    binding = _binding()
    assert binding["archive"]["member_count"] == 60
    assert E.historical_snapshot_problems(
        binding, ROOT / E.SOURCE_ARCHIVE, root=ROOT
    ) == []


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
@needs_private_evidence
def test_historical_snapshot_refuses_member_tampering(tmp_path, mutation):
    source = ROOT / E.SOURCE_ARCHIVE
    target = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "w") as new:
        names = old.namelist()
        victim = names[0]
        for name in names:
            if mutation == "missing" and name == victim:
                continue
            payload = old.read(name)
            if mutation == "changed" and name == victim:
                payload += b"tampered"
            new.writestr(name, payload)
        if mutation == "extra":
            new.writestr("extra.py", b"pass\n")
    problems = E.historical_snapshot_problems(_binding(), target, root=ROOT)
    assert problems
    assert any("archive" in problem or "member" in problem for problem in problems)


@needs_private_evidence
def test_historical_snapshot_refuses_wrong_origin_or_binding_digest():
    binding = _binding()
    victim = next(iter(binding["members"]))
    binding["members"][victim]["origin"] = {
        "kind": "git_object", "commit": "0" * 40
    }
    assert "source binding digest differs" in E.historical_snapshot_problems(
        binding, ROOT / E.SOURCE_ARCHIVE, root=ROOT
    )


@needs_private_evidence
def test_live_drift_is_exactly_declared_and_independent():
    binding = _binding()
    declared = json.loads((ROOT / E.LIVE_DRIFT).read_text())
    assert E.live_drift_problems(ROOT, binding, declared) == []
    assert [row["path"] for row in declared["differing"]] == [
        "scripts/17_public_snapshot.py", "src/eval/visual_stress.py"
    ]


@needs_private_evidence
def test_live_drift_refuses_a_stale_declaration():
    binding = _binding()
    declared = json.loads((ROOT / E.LIVE_DRIFT).read_text())
    declared["differing"] = []
    declared["differing_count"] = 0
    _rehash(declared, "drift_digest")
    assert "live drift declaration does not match the current tree" in \
        E.live_drift_problems(ROOT, binding, declared)


def test_append_only_binding_accepts_a_suffix_and_rejects_edits():
    frozen = b"contract line one\ncontract line two\n"
    digest = E.sha256_bytes(frozen)
    assert E.append_only_problems(frozen, frozen + b"A11\n", digest) == []
    assert E.append_only_problems(frozen, b"edited\n" + frozen, digest)
    assert E.append_only_problems(frozen, frozen, "0" * 64)


@needs_private_evidence
def test_postscore_seal_binds_the_separate_derived_layer():
    seal = json.loads((ROOT / E.POSTSCORE_SEAL).read_text())
    assert E.postscore_seal_problems(seal, ROOT) == []
    assert seal["file_count"] == 9
    assert set(seal["expected_arms"]) == set(E.ARMS)


def _copy_postscore_tree(tmp_path: Path) -> dict:
    paths = set(E.POSTSCORE_FILES) | {
        E.PARENT_SEAL, E.FROZEN_VERIFIER, E.EVAL_PACK_MANIFEST,
        E.EXECUTION_MANIFEST,
    }
    for rel in paths:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    return json.loads((ROOT / E.POSTSCORE_SEAL).read_text())


@pytest.mark.parametrize("target", ["parent", "score", "verifier"])
@needs_private_evidence
def test_postscore_seal_refuses_tampering_even_if_resigned(tmp_path, target):
    seal = _copy_postscore_tree(tmp_path)
    rel = {
        "parent": E.PARENT_SEAL,
        "score": Path("data/reports/bricknet/eval/inv_lora_constrained.json"),
        "verifier": E.FROZEN_VERIFIER,
    }[target]
    path = tmp_path / rel
    path.write_bytes(path.read_bytes() + b"\n")

    # Model an attacker updating the row and the outer self-digest.  The
    # verifier's independent historical anchors still have to refuse it.
    for key in ("parent_output_seal", "frozen_verifier"):
        if seal[key]["path"] == str(rel):
            seal[key]["sha256"] = E.sha256_file(path)
            seal[key]["bytes"] = path.stat().st_size
    for row in seal["files"]:
        if row["path"] == str(rel):
            row["sha256"] = E.sha256_file(path)
            row["bytes"] = path.stat().st_size
    _rehash(seal, "postscore_digest")
    problems = E.postscore_seal_problems(seal, tmp_path)
    assert any("historical postscore anchor differs" in problem for problem in problems)


@needs_private_evidence
def test_postscore_seal_refuses_an_added_or_removed_file_row():
    seal = json.loads((ROOT / E.POSTSCORE_SEAL).read_text())
    removed = copy.deepcopy(seal)
    removed["files"].pop()
    _rehash(removed, "postscore_digest")
    assert "postscore file set differs" in E.postscore_seal_problems(removed, ROOT)
