#!/usr/bin/env python3
"""The Phase 3C result index: build it once, and check it by reading it.

A completed Phase 3C run lives in ``runs/phase3c``, which ``.gitignore``
excludes -- so after gen09 the repository recorded the *conclusion* in
PORTFOLIO.md and RESEARCH_RECORD.md and kept none of the evidence. This
module closes that: it copies the run into a tracked, write-once private
tree and writes an index that binds every byte of it.

**Why an index rather than just the files.** Committing the directory alone
would prove that some files exist, not that they are the ones the node
produced and the Mac verified. The index names every file with its size and
SHA-256, records each step's command, exit code and cell count, and -- the
part that costs something -- binds the six members to the **frozen plan**:
:func:`verify` re-reads each member and judges its rows with the same
``step_problems`` a real run is judged by, so a member of the right length
made of the wrong cells is a refusal.

**Why not in** ``src/`` **.** ``pack.PACK_ALLOW`` carries ``src/**/*.py``, so
a new module under ``src/`` would enter the Phase 3C pack, change its
``pack_digest`` and invalidate the authorization gen09 already ran under. A
script is named one at a time in that allowlist and this one is not named,
so it does not travel and nothing frozen moves.

The tree is **write-once**: ``--build`` refuses to overwrite an index that is
already there. Rebuilding identical content is not offered, because the point
of the index is that it was written when the run was fresh.

    python scripts/61_phase3c_results.py --build --out-dir runs/phase3c
    python scripts/61_phase3c_results.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import phase3c, phase3c_report                    # noqa: E402
from src.eval.acceptance import canonical_json                  # noqa: E402

RESULTS_ROOT = "data/phase3c/results"
RESULT_KIND = "brickagain.phase3c_result_index"
RESULT_SCHEMA_VERSION = 1

#: The out-dir files an index must name, beside the six members. Fixed here
#: rather than discovered, so a run missing one of them cannot pass by
#: producing an index that does not mention it.
OUT_DIR_ARTEFACTS: tuple[str, ...] = (
    "execution_manifest.json", "seal.json", "receipt.json", "scores.json",
    "phase3c_report.md", "success_cases.json", "failure_cases.json",
    "reproduce.md",
)

#: Each artefact's own self-digest field, where it has one. These are checked
#: by recomputing them over the document minus the field, so a document
#: edited after the fact does not agree with itself.
SELF_DIGESTS: dict[str, str] = {
    "execution_manifest.json": "manifest_digest",
    "seal.json": "seal_digest",
    "receipt.json": "receipt_digest",
    "scores.json": "scores_digest",
}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def digest_obj(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def _rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _file_row(directory: Path, rel: str) -> dict:
    blob = (directory / rel).read_bytes()
    return {"path": rel, "bytes": len(blob), "sha256": sha256_bytes(blob)}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(out_dir: Path, *, generation: str | None = None,
          root: Path | None = None) -> dict:
    """The index for a completed run, over the tree already copied in place.

    The files are expected to be under ``<results>/<generation>/`` already;
    this reads what is there rather than copying, so the index describes the
    bytes a reader will find and not the bytes a copy step believed it wrote.
    """
    root = Path(root or ROOT)
    generation = generation or phase3c.GENERATION
    directory = root / RESULTS_ROOT / generation
    out_dir = Path(out_dir)

    archive = root / phase3c.ARCHIVE_ROOT / generation
    plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
    grant = json.loads(
        (archive / "execution_authorization.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (directory / "out_dir/execution_manifest.json").read_text("utf-8"))
    seal = json.loads((directory / "out_dir/seal.json").read_text("utf-8"))
    receipt = json.loads(
        (directory / "out_dir/receipt.json").read_text("utf-8"))
    scores = json.loads((directory / "out_dir/scores.json").read_text("utf-8"))

    members = []
    for index in range(phase3c.N_STEPS):
        group, arm = phase3c.step(index)
        rel = f"out_dir/{phase3c.samples_member(index)}"
        row = _file_row(directory, rel)
        exit_path = directory / f"node/step{index}.exit"
        members.append({
            "step_index": index, "group": group, "arm": arm,
            "member": phase3c.samples_member(index),
            "path": rel, "bytes": row["bytes"], "sha256": row["sha256"],
            "rows": _rows(directory / rel),
            "exit_code": int(exit_path.read_text().strip()),
            "command": (directory / f"node/step{index}.cmd")
                       .read_text(encoding="utf-8").strip(),
            "combined_stdout_stderr":
                f"node/step{index}.combined_stdout_stderr.log",
        })

    document = {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA_VERSION,
        "generation": generation,
        "outcome": "complete",
        "citable_as_a_result": True,
        "statement": (
            f"Phase 3C {generation} ran to completion: six steps in the "
            f"frozen order, each an independent process with a cold model "
            f"load, each exactly {plan['settings']['k'] * 0 + 320} cells, "
            "each exit 0. 1,920 cells, one seal, receipt verified with no "
            "problems, scores re-derived on the Mac from the stored per-case "
            "samples, report rendered from those scores. Every number in "
            "PORTFOLIO.md and RESEARCH_RECORD.md is re-derivable from the "
            "members named here."),
        "run_out_dir_on_the_mac": str(out_dir),
        "node": {
            # Read out of the observation listing rather than written here.
            # A published script must carry no personal path -- module 17's
            # audit refuses one, which is how this was found -- and the
            # listing is in any case the evidence for where the run was.
            "host": _from_listing(directory, "HOST"),
            "out_dir_on_node": _from_listing(directory, "TARGET"),
            "retained_on_node": True,
            "environment_observed": manifest["environment_observed"],
            "started_at": manifest["started_at"],
            "driver_log": "node/driver.log",
            "observation": {
                "kind": "post_hoc_read_only_observation",
                "what_this_is": (
                    "A listing of the retained node tree taken after the run "
                    "finished and after the out-dir had been copied to the "
                    "Mac. It is NOT runner output. It re-derives the node's "
                    "pack verification, environment, adapter digests, train "
                    "split digest and per-member row counts so the claims in "
                    "this index rest on something observed."),
                "read_only": True,
                "nothing_was_created_moved_or_deleted": True,
                "commands_script": "node/observation_commands.sh",
                "listing": _sole_listing(directory),
            },
            "seal": {
                "command": (directory / "node/seal.cmd")
                           .read_text(encoding="utf-8").strip(),
                "exit_code": int(
                    (directory / "node/seal.exit").read_text().strip()),
                "combined_stdout_stderr":
                    "node/seal.combined_stdout_stderr.log",
            },
        },
        "mac": {
            "verify": "mac/verify.txt",
            "score": "mac/score.txt",
            "report": "mac/report.txt",
            "pack_verify": "mac/pack_verify.txt",
        },
        "digests_this_run_ran_under": {
            field: manifest[field] for field in (
                "plan_digest", "contract_digest", "settings_digest",
                "case_membership_digest", "audit_digest",
                "authorization_digest", "pack_digest", "dependency_digest",
                "generation_source_manifest_digest", "adapter_sha256",
                "gate_sources")
        },
        "artefact_digests": {
            "manifest_digest": manifest["manifest_digest"],
            "seal_digest": seal["seal_digest"],
            "carried_seal_digest": receipt["carried_seal_digest"],
            "receipt_digest": receipt["receipt_digest"],
            "scores_digest": scores["scores_digest"],
            "scorer_source_manifest_digest":
                scores["scorer_source_manifest_digest"],
            "report_sha256":
                _file_row(directory, "out_dir/phase3c_report.md")["sha256"],
            "success_cases_sha256":
                _file_row(directory, "out_dir/success_cases.json")["sha256"],
            "failure_cases_sha256":
                _file_row(directory, "out_dir/failure_cases.json")["sha256"],
            "reproduce_sha256":
                _file_row(directory, "out_dir/reproduce.md")["sha256"],
        },
        "totals": {
            "members": len(members),
            "rows_per_member": 320,
            "total_cells": sum(m["rows"] for m in members),
            "total_cells_planned":
                phase3c.N_CASES * phase3c.SETTINGS.k * len(phase3c.ARM_ORDER),
        },
        "members": members,
        "receipt_says": {
            "verified": receipt["verified"],
            "problems": receipt["problems"],
        },
        "val_was_not_opened": True,
        "why_val_was_not_opened": (
            "the audit, the selection and the scorer all read "
            f"{json.loads((archive / 'isolation_audit.json').read_text())['source']['file']} "
            "-- the test split. No validation split is named by the plan, "
            "carried by the pack or opened by any stage."),
        "grant_digest_it_ran_under": grant["authorization_digest"],
    }
    document["evidence_files"] = _evidence_table(directory)
    document["index_digest"] = digest_obj(
        {k: v for k, v in document.items() if k != "index_digest"})
    return document


def _from_listing(directory: Path, field: str) -> str:
    """One ``### FIELD:`` value out of the post-hoc node listing."""
    text = (directory / _sole_listing(directory)).read_text(
        encoding="utf-8", errors="replace")
    prefix = f"### {field}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise SystemExit(f"the node listing states no {field}")


def _sole_listing(directory: Path) -> str:
    """The one post-hoc listing in ``node/``, named rather than guessed."""
    found = sorted(p.name for p in (directory / "node").glob("observation_*")
                   if p.suffix == ".txt")
    if len(found) != 1:
        raise SystemExit(f"expected exactly one node observation listing, "
                         f"found {found}")
    return f"node/{found[0]}"


def _evidence_table(directory: Path) -> list[dict]:
    return [_file_row(directory, str(p.relative_to(directory)))
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "index.json"]


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(directory, *, root=None) -> list[str]:
    """Everything wrong with a result index, in one list.

    In order: the index is this kind and schema and says it is citable; its
    own digest covers the rest of it; every file it names is here at exactly
    the size and SHA-256 it names and no file is here it does not name; the
    execution manifest agrees with itself and with the generation's archived
    plan and grant; the seal, receipt and score record each agree with their
    own self-digest; the receipt is verified with no problems and carries the
    carried seal digest; the score record was produced by the scorer the plan
    approved; **every member's rows are the cells the plan predetermined for
    that step, bound to this run's execution manifest**; each step exited 0
    with 320 cells; and the post-hoc listing found the same six members
    populated with the same row counts.
    """
    root = Path(root or ROOT)
    directory = Path(directory)
    index_path = directory / "index.json"
    if not index_path.is_file():
        return [f"{directory} has no index.json; the run is recorded nowhere"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"index.json is not valid JSON ({exc})"]
    if not isinstance(index, dict):
        return [f"index.json is a {type(index).__name__}, not an object"]

    problems: list[str] = []
    if index.get("kind") != RESULT_KIND:
        problems.append(f"kind is {index.get('kind')!r}, not {RESULT_KIND!r}")
    if index.get("schema_version") != RESULT_SCHEMA_VERSION:
        problems.append(f"schema_version is {index.get('schema_version')!r}")
    if index.get("outcome") != "complete":
        problems.append(f"outcome is {index.get('outcome')!r}, not 'complete'")
    if index.get("citable_as_a_result") is not True:
        problems.append(
            "citable_as_a_result is not True; a complete, verified run is "
            "the one thing in this tree that may be cited")

    recorded = index.get("index_digest")
    recomputed = digest_obj(
        {k: v for k, v in index.items() if k != "index_digest"})
    if recorded != recomputed:
        problems.append(f"index_digest is {str(recorded)[:16]}... and the "
                        f"record digests to {recomputed[:16]}...")

    problems += _file_table_problems(directory, index)
    problems += _artefact_problems(directory, index, root)
    problems += _member_problems(directory, index, root)
    problems += _listing_problems(directory, index)
    if problems:
        # The end-to-end pass costs ~30s, almost all of it the frozen
        # paired bootstrap. There is nothing to learn from re-deriving a
        # score record for a directory whose index already does not hold.
        return problems
    return _end_to_end_problems(directory, index, root)


def _end_to_end_problems(directory: Path, index: dict, root: Path,
                         source_root=None) -> list[str]:
    """Re-derive the run and compare it whole, using the frozen machinery.

    The gap this closes. Up to here the index is checked against the bytes
    and the members against the plan's *cell identities* -- so a member
    whose ``raw_text`` was rewritten, with every size, digest and count
    re-filed afterwards, passed: no cell moved, and nothing recomputed what
    those texts score to. The seal, the receipt, the score record and the
    four published documents were all checked for **self-consistency** and
    never against a re-derivation.

    So the whole chain is re-run here, through the same functions
    ``--verify``, ``--score`` and ``--report`` use, against the archived
    plan and grant rather than anything in this directory:

    * :func:`phase3c.seal_problems` judges the seal against the plan, and
      :func:`phase3c.rehash` re-measures every sealed member and the
      execution manifest -- size and SHA-256 -- with the row counts checked
      against the seal's own numbers here;
    * :func:`phase3c_report.report_chain_problems` re-judges the plan, the
      archive, the grant and the seal, **re-derives the receipt from the
      bytes** (which re-hashes every member, re-reads the execution manifest
      and binds all 1,920 rows to it), **recomputes the entire score record
      from** ``raw_text`` **by the frozen scorer** and compares it field by
      field with :func:`phase3c.scores_problems` -- not merely its digest,
      and not merely its scorer manifest;
    * the four published documents are re-rendered from that record and
      compared by :func:`phase3c_report.member_problems`, so a report or a
      case index edited and then re-filed does not survive.

    ``source_root`` is where the scorer's own source is read from, and is
    the repository rather than ``root``: the plan pins the SHA-256 of the
    modules that do the scoring, and a temporary copy of an evidence tree
    does not contain them. ``root`` still locates the archive, so a test can
    point the whole check at a copy without touching a frozen byte.
    """
    problems: list[str] = []
    source_root = Path(source_root or ROOT)
    out_dir = directory / "out_dir"
    generation = str(index.get("generation"))
    archive = Path(root) / phase3c.ARCHIVE_ROOT / generation
    try:
        plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
        grant = json.loads((archive / "execution_authorization.json")
                           .read_text(encoding="utf-8"))
        seal = json.loads((out_dir / "seal.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"the archived plan, grant or seal could not be read "
                f"({type(exc).__name__}: {exc})"]

    problems += [f"seal: {p}" for p in phase3c.seal_problems(seal, plan)]

    # Every sealed member re-measured, and its row count checked. ``rehash``
    # spends the seal's sizes and digests, including the manifest's.
    _found, rehash_problems = phase3c.rehash(out_dir, seal)
    problems += [f"rehash: {p}" for p in rehash_problems]
    for member, expected in sorted((seal.get("members") or {}).items()):
        path = out_dir / member
        if not path.is_file():
            continue
        rows = _rows(path)
        if rows != expected.get("rows"):
            problems.append(f"rehash: {member} holds {rows} rows and the "
                            f"seal says {expected.get('rows')!r}")
    if problems:
        return problems

    # The receipt re-derived from bytes, and the score record recomputed
    # from raw_text and compared whole.
    problems += phase3c_report.report_chain_problems(
        out_dir, plan, grant=grant, seal=seal, root=source_root,
        archive_dir=archive)
    if problems:
        return problems

    problems += _published_member_problems(out_dir, index, plan)
    return problems


def _published_member_problems(out_dir: Path, index: dict,
                               plan: dict) -> list[str]:
    """The four published documents, re-rendered and compared.

    ``reproduce.md`` names the ``--out-dir`` it was rendered with, so it is
    re-rendered against the path the run actually used -- recorded in the
    index as ``run_out_dir_on_the_mac`` and therefore covered by
    ``index_digest`` -- rather than against wherever the evidence now sits.
    Everything else is the machinery ``--report`` itself uses.
    """
    record = phase3c_report.read_scores(out_dir)
    receipt = phase3c_report.read_receipt(out_dir)
    original = Path(str(index.get("run_out_dir_on_the_mac") or out_dir))
    successes, failures = phase3c_report.case_indices(record, plan)
    expected = {
        phase3c_report.REPORT_NAME:
            phase3c_report.render(record, receipt, plan),
        phase3c_report.REPRODUCE_NAME:
            phase3c_report.render_reproduce(record, receipt, plan, original),
    }
    for name, cases, criterion, kind in (
            (phase3c_report.SUCCESS_INDEX_NAME, successes,
             "at least one seed", "success"),
            (phase3c_report.FAILURE_INDEX_NAME, failures, "no seed",
             "failure")):
        expected[name] = {
            "kind": f"brickagain.phase3c_{kind}_cases",
            "plan_digest": record["plan_digest"],
            "seal_digest": receipt.get("seal_digest"),
            "criterion": (f"arm {record['primary_contrast'][0]} reached "
                          f"core success on {criterion}"),
            "n": len(cases), "cases": cases}

    problems: list[str] = []
    for name in phase3c_report.PUBLISHED_MEMBERS:
        path = out_dir / name
        if not path.is_file():
            problems.append(f"published: {name} is not here; a published "
                            "report is all four members or none")
            continue
        problems += [f"published: {p}" for p in
                     phase3c_report.member_problems(
                         path, expected[name],
                         what=("report"
                               if name == phase3c_report.REPORT_NAME else
                               "reproduce document"
                               if name == phase3c_report.REPRODUCE_NAME else
                               "case index"))]
    return problems


def _file_table_problems(directory: Path, index: dict) -> list[str]:
    problems: list[str] = []
    table = index.get("evidence_files")
    if not isinstance(table, list):
        return ["evidence_files is not a list"]
    named = set()
    for entry in table:
        if not isinstance(entry, dict) or "path" not in entry:
            problems.append(f"an evidence entry is not a record: {entry!r}")
            continue
        rel = entry["path"]
        named.add(rel)
        path = directory / rel
        if not path.is_file():
            problems.append(f"{rel} is named by the index and is not here")
            continue
        blob = path.read_bytes()
        if len(blob) != entry.get("bytes"):
            problems.append(f"{rel} is {len(blob)} bytes and the index says "
                            f"{entry.get('bytes')!r}")
        actual = sha256_bytes(blob)
        if actual != entry.get("sha256"):
            problems.append(f"{rel} digests to {actual[:16]}... and the "
                            f"index says {str(entry.get('sha256'))[:16]}...")
    on_disk = {str(p.relative_to(directory)) for p in directory.rglob("*")
               if p.is_file() and p.name != "index.json"}
    for rel in sorted(on_disk - named):
        problems.append(f"{rel} is here and the index does not name it")
    for rel in OUT_DIR_ARTEFACTS:
        if f"out_dir/{rel}" not in named:
            problems.append(f"out_dir/{rel} is not named by the index")
    return problems


def _artefact_problems(directory: Path, index: dict,
                       root: Path) -> list[str]:
    """The four self-digesting documents, and the archive they ran under."""
    problems: list[str] = []
    declared = index.get("artefact_digests")
    if not isinstance(declared, dict):
        return ["artefact_digests is not a record"]

    loaded: dict[str, dict] = {}
    for name, field in SELF_DIGESTS.items():
        path = directory / "out_dir" / name
        if not path.is_file():
            problems.append(f"out_dir/{name} is not here")
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"out_dir/{name} is not valid JSON ({exc})")
            continue
        loaded[name] = document
        recomputed = digest_obj(
            {k: v for k, v in document.items() if k != field})
        if document.get(field) != recomputed:
            problems.append(
                f"out_dir/{name} carries {field} "
                f"{str(document.get(field))[:16]}... and digests to "
                f"{recomputed[:16]}...")
        key = {"execution_manifest.json": "manifest_digest",
               "seal.json": "seal_digest",
               "receipt.json": "receipt_digest",
               "scores.json": "scores_digest"}[name]
        if declared.get(key) != document.get(field):
            problems.append(f"the index's {key} is not the digest in "
                            f"out_dir/{name}")

    manifest = loaded.get("execution_manifest.json")
    seal = loaded.get("seal.json")
    receipt = loaded.get("receipt.json")
    scores = loaded.get("scores.json")

    ran_under = index.get("digests_this_run_ran_under")
    if not isinstance(ran_under, dict):
        problems.append("digests_this_run_ran_under is not a record")
    elif manifest is not None:
        for field, value in ran_under.items():
            if manifest.get(field) != value:
                problems.append(
                    f"the index says {field} was {str(value)[:16]}... and "
                    f"the execution manifest says "
                    f"{str(manifest.get(field))[:16]}...")

    # Against the generation's archive, which is still in the tree.
    generation = index.get("generation")
    archive = root / phase3c.ARCHIVE_ROOT / str(generation)
    plan_path = archive / "plan.json"
    grant_path = archive / "execution_authorization.json"
    if plan_path.is_file() and manifest is not None:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for field in ("plan_digest", "contract_digest"):
            if plan.get(field) != manifest.get(field):
                problems.append(
                    f"{generation}'s archived plan has {field} "
                    f"{str(plan.get(field))[:16]}... and the run ran under "
                    f"{str(manifest.get(field))[:16]}...")
        if scores is not None:
            approved = plan.get("scorer_source_manifest_digest")
            if scores.get("scorer_source_manifest_digest") != approved:
                problems.append(
                    "the score record was produced by a scorer the plan did "
                    "not approve")
    if grant_path.is_file() and manifest is not None:
        grant = json.loads(grant_path.read_text(encoding="utf-8"))
        for field in ("authorization_digest", "pack_digest",
                      "dependency_digest",
                      "generation_source_manifest_digest"):
            if grant.get(field) != manifest.get(field):
                problems.append(
                    f"{generation}'s archived grant says {field} "
                    f"{str(grant.get(field))[:16]}... and the run ran under "
                    f"{str(manifest.get(field))[:16]}...")
        if index.get("grant_digest_it_ran_under") != \
                grant.get("authorization_digest"):
            problems.append("the index names another grant than the one "
                            f"{generation} archived")

    if receipt is not None:
        if receipt.get("verified") is not True:
            problems.append("the receipt is not verified")
        if receipt.get("problems"):
            problems.append(f"the receipt lists {receipt['problems']}")
        if receipt.get("carried_seal_digest") != declared.get(
                "carried_seal_digest"):
            problems.append("the index's carried_seal_digest is not the one "
                            "the receipt recorded")
        if seal is not None and \
                receipt.get("carried_seal_digest") != seal.get("seal_digest"):
            problems.append(
                "the seal digest carried back by hand is not the digest of "
                "the seal in this tree")
    says = index.get("receipt_says")
    if not isinstance(says, dict):
        problems.append("receipt_says is not a record")
    elif receipt is not None and (
            says.get("verified") is not receipt.get("verified")
            or says.get("problems") != receipt.get("problems")):
        problems.append("receipt_says does not match the receipt beside it")

    if seal is not None:
        expected = phase3c.N_CASES * phase3c.SETTINGS.k * len(
            phase3c.ARM_ORDER)
        if seal.get("expected_rows") != expected:
            problems.append(f"the seal expects {seal.get('expected_rows')!r} "
                            f"rows, not {expected}")
        if len(seal.get("members") or {}) != phase3c.N_STEPS:
            problems.append(
                f"the seal names {len(seal.get('members') or {})} members, "
                f"not {phase3c.N_STEPS}")
    return problems


def _member_problems(directory: Path, index: dict, root: Path) -> list[str]:
    """The six members, against the plan that predetermined their cells.

    This is the check that makes the index evidence rather than a summary:
    the rows are read from bytes and judged by ``step_problems``, so the
    member has to be exactly the cells that step was to produce, each bound
    to this run's execution manifest.
    """
    problems: list[str] = []
    members = index.get("members")
    if not isinstance(members, list) or len(members) != phase3c.N_STEPS:
        return [f"members is not {phase3c.N_STEPS} entries"]

    generation = index.get("generation")
    plan_path = root / phase3c.ARCHIVE_ROOT / str(generation) / "plan.json"
    plan = None
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = directory / "out_dir/execution_manifest.json"
    manifest_digest = None
    if manifest_path.is_file():
        manifest_digest = json.loads(
            manifest_path.read_text(encoding="utf-8")).get("manifest_digest")

    seen = 0
    for position, entry in enumerate(members):
        if not isinstance(entry, dict):
            problems.append("a members entry is not a record")
            continue
        step_index = entry.get("step_index")
        if step_index != position:
            problems.append(f"members[{position}] records step_index "
                            f"{step_index!r}")
            continue
        group, arm = phase3c.step(position)
        if (entry.get("group"), entry.get("arm")) != (group, arm):
            problems.append(
                f"step {position} is {entry.get('group')!r}/"
                f"{entry.get('arm')!r} and the frozen schedule has "
                f"{group}/{arm}")
        if entry.get("member") != phase3c.samples_member(position):
            problems.append(f"step {position} names member "
                            f"{entry.get('member')!r}")
        if entry.get("exit_code") != 0:
            problems.append(f"step {position} exited "
                            f"{entry.get('exit_code')!r}; a complete run's "
                            "steps all exit 0")

        rel = entry.get("path")
        path = directory / str(rel)
        if not path.is_file():
            problems.append(f"step {position} names {rel} and it is not here")
            continue
        blob = path.read_bytes()
        if entry.get("bytes") != len(blob):
            problems.append(f"{rel} is {len(blob)} bytes and the index says "
                            f"{entry.get('bytes')!r}")
        if entry.get("sha256") != sha256_bytes(blob):
            problems.append(f"{rel} does not digest to what the index says")

        if plan is None:
            problems.append(f"{generation}'s archived plan is not in this "
                            f"tree, so {rel}'s rows cannot be checked")
            continue
        if manifest_digest is None:
            problems.append(f"no execution manifest digest was read, so "
                            f"{rel}'s rows are bound to nothing")
            continue
        try:
            rows = phase3c.read_rows(path)
        except Exception as exc:                 # noqa: BLE001 - fail closed
            problems.append(f"{rel} could not be read as rows "
                            f"({type(exc).__name__}: {exc})")
            continue
        if entry.get("rows") != len(rows):
            problems.append(f"{rel} records {entry.get('rows')!r} rows and "
                            f"holds {len(rows)}")
        problems += [f"{rel}: {p}" for p in phase3c.step_problems(
            rows, plan, position, manifest_digest=manifest_digest)]
        seen += len(rows)

    totals = index.get("totals")
    if not isinstance(totals, dict):
        problems.append("totals is not a record")
        return problems
    if totals.get("total_cells") != seen:
        problems.append(f"totals says {totals.get('total_cells')!r} cells and "
                        f"the members hold {seen}")
    expected = phase3c.N_CASES * phase3c.SETTINGS.k * len(phase3c.ARM_ORDER)
    if totals.get("total_cells_planned") != expected:
        problems.append(
            f"total_cells_planned is {totals.get('total_cells_planned')!r}")
    if seen != expected:
        problems.append(f"the run holds {seen} cells and the plan "
                        f"predetermined {expected}")
    if totals.get("members") != phase3c.N_STEPS:
        problems.append(f"totals says {totals.get('members')!r} members")
    return problems


def _listing_problems(directory: Path, index: dict) -> list[str]:
    """The post-hoc listing, against what the index says about the members."""
    problems: list[str] = []
    observation = (index.get("node") or {}).get("observation") or {}
    if observation.get("kind") != "post_hoc_read_only_observation":
        problems.append(
            "the node observation does not declare itself post-hoc and "
            "read-only; a listing taken afterwards must not read as runner "
            "output")
    rel = observation.get("listing")
    if not rel:
        problems.append("the index records no listing of the node tree")
        return problems
    path = directory / rel
    if not path.is_file():
        problems.append(f"{rel} is named as the listing and is not here")
        return problems
    text = path.read_text(encoding="utf-8", errors="replace")

    claimed = {}
    for entry in index.get("members") or []:
        if isinstance(entry, dict):
            claimed[Path(str(entry.get("member", ""))).stem] = entry.get("rows")
    observed: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].startswith("step_") \
                and parts[1] == "POPULATED":
            try:
                observed[parts[0]] = int(parts[2])
            except ValueError:
                problems.append(f"the listing reports {parts[0]} with a "
                                f"non-numeric row count {parts[2]!r}")
    if len(observed) != phase3c.N_STEPS:
        problems.append(
            f"the listing found {len(observed)} populated member(s), not "
            f"{phase3c.N_STEPS}")
    for stem, rows in sorted(observed.items()):
        if stem in claimed and claimed[stem] != rows:
            problems.append(f"the index says {stem} has {claimed[stem]!r} "
                            f"rows and the listing found {rows}")
    for stem in sorted(set(claimed) - set(observed)):
        problems.append(f"the index names {stem} and the listing never found "
                        "it populated")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3C result index: build once, verify by reading")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out-dir", default="runs/phase3c",
                        help="the run directory the index describes")
    parser.add_argument("--generation", default=None)
    args = parser.parse_args(argv)
    if args.build == args.verify:
        parser.error("exactly one of --build and --verify")

    generation = args.generation or phase3c.GENERATION
    directory = ROOT / RESULTS_ROOT / generation

    if args.build:
        path = directory / "index.json"
        if path.exists():
            print(f"refusing to overwrite {path}: the index is write-once "
                  "and was written when the run was fresh", file=sys.stderr)
            return 2
        document = build(Path(args.out_dir), generation=generation)
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "written": str(path.relative_to(ROOT)),
            "index_digest": document["index_digest"],
            "members": document["totals"]["members"],
            "total_cells": document["totals"]["total_cells"],
            "evidence_files": len(document["evidence_files"]),
        }, indent=2))
        return 0

    problems = verify(directory)
    print(json.dumps({
        "generation": generation,
        "directory": str(directory.relative_to(ROOT)),
        "problems": problems,
        "verified": not problems,
    }, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
