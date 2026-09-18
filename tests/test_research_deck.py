"""The research deck's two receipts, read instead of only written.

`presentation/research/` carries a `validation.json` that pins the .pptx by
SHA-256 and byte count, and an `evidence_manifest.json` that records the digest
of all twenty files the deck was built from. Nothing read either of them: the
research report -- the document that actually goes to a reviewer -- was the
least-guarded deliverable in the tree, while `tests/test_presentation_numbers`
guards the *showcase* deck beside it. One of the twenty had already moved
(`PORTFOLIO.md`) and no test could say so.

Same shape as this project's standing rule that a receipt must be read, not
written, and the same shape as `deck_pdf` trusting a foreign manifest: a digest
nobody recomputes is a sentence, not a check.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "presentation" / "research"
DECK = RESEARCH / "BrickAgain_研究成果報告.pptx"
MANIFEST = RESEARCH / "evidence_manifest.json"
VALIDATION = RESEARCH / "validation.json"
DRIFT = RESEARCH / "source_drift.json"
ARTIFACT_ONLY = "artifact-only:"

#: Prose in this project is revised every round; reports, images and run logs
#: are outputs that are not. The split is read off the path rather than typed
#: into a list, so a source added later lands on the right side by itself.
LIVING = ".md"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slides_in(deck: Path) -> int:
    with zipfile.ZipFile(deck) as archive:
        return sum(1 for name in archive.namelist()
                   if re.fullmatch(r"ppt/slides/slide\d+\.xml", name))


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/research is not published")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validation() -> dict:
    if not VALIDATION.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/research is not published")
    return json.loads(VALIDATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def drift() -> dict:
    if not DRIFT.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/research is not published")
    return json.loads(DRIFT.read_text(encoding="utf-8"))["acknowledged"]


# --- the deck is the one the receipt describes ------------------------------

def test_the_validation_record_describes_the_deck_on_disk(validation):
    assert _digest(DECK) == validation["finalSha256"]
    assert DECK.stat().st_size == validation["byteCount"]


def test_the_slide_census_agrees_three_ways(validation, manifest):
    """The .pptx, the import record, and the per-slide manifest."""
    counted = _slides_in(DECK)
    assert counted == validation["firstPartyImport"]["slideCount"]
    assert counted == len(manifest["slides"])
    assert [entry["slide"] for entry in manifest["slides"]] == \
        list(range(1, counted + 1))


def test_a_deck_that_moved_is_refused(validation, tmp_path):
    """The check fails when the thing it describes is not the thing on disk."""
    edited = tmp_path / "edited.pptx"
    edited.write_bytes(DECK.read_bytes() + b"\x00")
    assert _digest(edited) != validation["finalSha256"]
    assert edited.stat().st_size != validation["byteCount"]


# --- every recorded source still hashes to what was recorded ----------------

def test_every_named_source_is_still_there(manifest):
    missing = [rel for rel in manifest["sources"] if not (ROOT / rel).is_file()]
    assert missing == [], missing


def test_every_frozen_artefact_matches_its_recorded_digest(manifest):
    """Reports, images and run logs. Any drift here is a refusal."""
    moved = []
    for rel, recorded in sorted(manifest["sources"].items()):
        if rel.endswith(LIVING):
            continue
        found = _digest(ROOT / rel)
        if found != recorded:
            moved.append(f"{rel}: recorded {recorded[:12]}…, found {found[:12]}…")
    assert moved == [], moved


def test_a_living_source_may_drift_only_when_it_is_acknowledged(manifest, drift):
    undeclared = []
    for rel, recorded in sorted(manifest["sources"].items()):
        if not rel.endswith(LIVING):
            continue
        found = _digest(ROOT / rel)
        if found == recorded:
            continue
        entry = drift.get(rel)
        if entry is None:
            undeclared.append(f"{rel} drifted and is not in {DRIFT.name}")
            continue
        if entry["manifest_digest"] != recorded:
            undeclared.append(
                f"{rel}: the acknowledgement records a different manifest "
                f"digest than the manifest does")
        if entry["checked_against_digest"] != found:
            undeclared.append(
                f"{rel}: the acknowledgement was written against "
                f"{entry['checked_against_digest'][:12]}…, the file is now "
                f"{found[:12]}… -- re-read the slides that cite it")
    assert undeclared == [], undeclared


def test_a_stale_acknowledgement_is_refused(manifest, drift):
    """An entry for a file that matches again has to go, or it excuses nothing."""
    stale = [rel for rel in drift
             if rel in manifest["sources"]
             and _digest(ROOT / rel) == manifest["sources"][rel]]
    assert stale == [], stale


def test_an_acknowledgement_names_a_source_the_manifest_has(manifest, drift):
    unknown = [rel for rel in drift if rel not in manifest["sources"]]
    assert unknown == [], unknown


def test_every_acknowledgement_says_which_slides_and_why(manifest, drift):
    for rel, entry in drift.items():
        slides = entry.get("cited_by_slides")
        assert slides, rel
        census = len(manifest["slides"])
        assert all(1 <= number <= census for number in slides), (rel, slides)
        assert len(entry.get("why", "")) >= 40, rel


# --- the guards fail when they should ---------------------------------------

def _checked(sources: dict[str, str], acknowledged: dict) -> list[str]:
    """The same rule as the two tests above, over injected inputs."""
    problems = []
    for rel, recorded in sorted(sources.items()):
        found = _digest(ROOT / rel)
        if found == recorded:
            continue
        if not rel.endswith(LIVING):
            problems.append(f"frozen artefact moved: {rel}")
            continue
        entry = acknowledged.get(rel)
        if entry is None or entry.get("checked_against_digest") != found:
            problems.append(f"unacknowledged drift: {rel}")
    return problems


def test_a_tampered_report_digest_is_caught(manifest, drift):
    report = next(rel for rel in manifest["sources"]
                  if rel.startswith("data/reports/"))
    before = _checked(manifest["sources"], drift)
    assert before == [], before          # the tree as it stands is clean
    injected = {**manifest["sources"], report: "0" * 64}
    assert _checked(injected, drift) == [f"frozen artefact moved: {report}"]


def test_a_drifted_markdown_with_no_entry_is_caught(manifest, drift):
    """A living source that moves without an entry is a refusal.

    PORTFOLIO.md is excluded here because it has an entry written against the
    file as it is now: blanking its *recorded* digest does not make it
    unacknowledged, which is the behaviour the entry is for.
    """
    living = [rel for rel in manifest["sources"]
              if rel.endswith(LIVING) and rel not in drift]
    assert living, "the manifest should name at least one unacknowledged .md"
    blanked = {rel: "0" * 64 for rel in living}
    assert sorted(_checked(blanked, drift)) == \
        sorted(f"unacknowledged drift: {rel}" for rel in living)


def test_an_entry_written_against_an_older_version_does_not_excuse_it(manifest):
    rel = "PORTFOLIO.md"
    if rel not in manifest["sources"]:
        pytest.skip(f"{ARTIFACT_ONLY} {rel} is not a recorded source")
    stale_entry = {rel: {"manifest_digest": manifest["sources"][rel],
                         "checked_against_digest": "0" * 64}}
    assert _checked({rel: manifest["sources"][rel]}, stale_entry) == \
        [f"unacknowledged drift: {rel}"]
    # And the real acknowledgement, which was written against the file as it
    # is, does excuse it -- so the refusal above means something.
    real = json.loads(DRIFT.read_text(encoding="utf-8"))["acknowledged"]
    assert _checked({rel: manifest["sources"][rel]}, real) == []
