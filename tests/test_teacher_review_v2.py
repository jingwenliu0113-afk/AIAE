"""M5-1: v2 is built beside v1, never over it.

v1 was handed to a teacher on 2026-09-10. Its directory still verifies against
its own SHA256SUMS.txt, and the archive is what was actually sent, so every
test here is written on the assumption that both are evidence rather than
working files.

The reproducibility proof is the positive one: two builds into two temporary
directories have to agree file for file. That replaces "rebuild v1 and compare",
which cannot be run without breaking the rule it would be testing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRESENTATION = ROOT / "presentation"
BUILDER = PRESENTATION / "tools" / "build_teacher_review.py"
ARTIFACT_ONLY = "artifact-only:"


def builder():
    spec = importlib.util.spec_from_file_location("teacher_review_builder",
                                                  BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_pdf_renders_new_date_and_markdown_cleanly(tool, tmp_path):
    # Through the `tool` fixture below, not builder() directly: presentation/
    # is not published, and loading the builder in a public checkout raised
    # FileNotFoundError instead of skipping like every other test here.
    output = tmp_path / "evidence.pdf"
    tool.markdown_pdf(
        "# Heading\n\n## Section\n\n### Detail\n\n**0.0%** result\n\n"
        "> 254 photos\n", output, "Evidence", "2026-09-24")
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() for page in PdfReader(output).pages)
    assert "2026-09-24" in text
    assert "2026-09-10" not in text
    assert "Detail" in text and "254 photos" in text
    assert "**" not in text and "###" not in text and "> 254" not in text


@pytest.fixture(scope="module")
def tool():
    if not BUILDER.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/ is not published")
    return builder()


RESEARCH_PPTX = PRESENTATION / "research" / "BrickAgain_研究成果報告.pptx"


@pytest.fixture
def slides(tool, tmp_path):
    """As many stand-in slides as the .pptx being packaged actually has.

    This was the literal 30, and the builder had the same literal, so the two
    agreed with each other and with nothing else. Reading it off the showcase
    deck's render manifest instead was worse: the manifest describes a
    *different* deck, and the count it gave -- 35 -- is how v2 came to ship
    the showcase deck under the research report's name. The document the PDF
    is supposed to be a rendering of is the only thing that can answer this.
    """
    from PIL import Image
    folder = tmp_path / "slides"
    folder.mkdir()
    for index in range(1, tool.slide_count(RESEARCH_PPTX) + 1):
        Image.new("RGB", (320, 180), (240 - index * 3, 240, 250)).save(
            folder / f"slide-{index}.png")
    return folder


@pytest.fixture
def report(tool, tmp_path):
    """A measured-shaped report covering every test the package carries."""
    carried = sorted(origin for destination, origin
                     in {**tool.COPIES, **tool.PHOTO_COPIES}.items()
                     if destination.startswith("test_samples/"))
    path = tmp_path / "test_report.json"
    path.write_text(json.dumps({
        "date": "2026-09-13",
        "revision": tool.git_revision(),
        "passed": 7,
        "groups": [{"title": "全部", "tests": carried,
                    "passed": 7, "summary": "7 passed in 0.10s"}],
    }, ensure_ascii=False), encoding="utf-8")
    return path


def build(tool, slides, tmp_path, name, monkeypatch, report=None, **over):
    """One v2 build, with VERSIONS.md and .gitattributes kept in the sandbox."""
    monkeypatch.setattr(tool, "VERSIONS", tmp_path / "VERSIONS.md")
    monkeypatch.setattr(tool, "GITATTRIBUTES", tmp_path / "gitattributes")
    (tmp_path / "gitattributes").write_text("# exact-scope rules only\n")
    argv = ["--slides-dir", str(slides), "--version", "2",
            "--test-report", str(report or _report(tool, tmp_path)),
            "--out-dir", str(tmp_path / name),
            "--archive", str(tmp_path / f"{name}.zip"),
            "--zip-date", "2026-09-13"]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    assert tool.main(argv) == 0
    return tmp_path / name, tmp_path / f"{name}.zip"


def _report(tool, tmp_path):
    """The same document the `report` fixture makes, for callers without it."""
    path = tmp_path / "test_report.json"
    if not path.is_file():
        carried = sorted(origin for destination, origin
                         in {**tool.COPIES, **tool.PHOTO_COPIES}.items()
                         if destination.startswith("test_samples/"))
        path.write_text(json.dumps({
            "date": "2026-09-13",
            "revision": tool.git_revision(),
            "passed": 7,
            "groups": [{"title": "全部", "tests": carried,
                        "passed": 7, "summary": "7 passed in 0.10s"}],
        }, ensure_ascii=False), encoding="utf-8")
    return path


# --- v1 is not touched ------------------------------------------------------

def test_version_two_does_not_write_into_the_v1_directory(tool, slides, report,
                                                          tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "VERSIONS", tmp_path / "VERSIONS.md")
    with pytest.raises(FileExistsError, match="may not write into the v1"):
        tool.main(["--slides-dir", str(slides), "--version", "2",
                   "--test-report", str(report),
                   "--out-dir", str(tool.OUT),
                   "--archive", str(tmp_path / "elsewhere.zip")])


def test_version_two_does_not_overwrite_the_v1_archive_or_its_sha256(
        tool, slides, report, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "VERSIONS", tmp_path / "VERSIONS.md")
    before = tool.sha256(tool.ARCHIVE)
    hash_file = tool.ARCHIVE.with_suffix(".zip.sha256")
    before_hash = hash_file.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="may not overwrite the v1 archive"):
        tool.main(["--slides-dir", str(slides), "--version", "2",
                   "--test-report", str(report),
                   "--out-dir", str(tmp_path / "v2"),
                   "--archive", str(tool.ARCHIVE)])
    assert tool.sha256(tool.ARCHIVE) == before
    assert hash_file.read_text(encoding="utf-8") == before_hash


def test_the_default_invocation_refuses_to_overwrite_v1(tool, slides, report,
                                                        tmp_path, monkeypatch):
    """No --out-dir still points at teacher_review, which exists."""
    monkeypatch.setattr(tool, "VERSIONS", tmp_path / "VERSIONS.md")
    before = tool.directory_digest(tool.OUT)
    with pytest.raises(FileExistsError, match="already exists"):
        tool.main(["--slides-dir", str(slides), "--test-report", str(report)])
    assert tool.directory_digest(tool.OUT) == before


def test_the_v1_directory_and_archive_digests_are_unchanged(tool, slides,
                                                            tmp_path, monkeypatch):
    # v1's own checksum file still describes the files beside it. Capture
    # both before building: main() rebinds OUT and ARCHIVE, so reading them
    # afterwards would be reading v2 and comparing it with itself.
    v1_dir, v1_archive = tool.OUT, tool.ARCHIVE
    assert tool.package_problems(v1_dir, 1) == []
    v1_digest, v1_zip = tool.directory_digest(v1_dir), tool.sha256(v1_archive)
    _, archive = build(tool, slides, tmp_path, "v2", monkeypatch)
    recorded = (tmp_path / "VERSIONS.md").read_text(encoding="utf-8")
    assert f"`{v1_digest}`" in recorded and f"`{v1_zip}`" in recorded
    # And the v2 row is about v2, not a second copy of v1's numbers.
    assert f"`{tool.sha256(archive)}`" in recorded
    assert tool.sha256(archive) != v1_zip
    # v1 itself is exactly as it was before the build ran.
    assert tool.directory_digest(v1_dir) == v1_digest
    assert tool.sha256(v1_archive) == v1_zip


# --- the version log --------------------------------------------------------

def test_versions_md_is_append_only(tool, slides, tmp_path, monkeypatch):
    build(tool, slides, tmp_path, "v2", monkeypatch)
    log = tmp_path / "VERSIONS.md"
    first = log.read_text(encoding="utf-8")
    assert "\n## v1\n" in first and "\n## v2\n" in first
    with pytest.raises(FileExistsError, match="already records v2"):
        tool.append_version(2, tmp_path / "v2", tmp_path / "v2.zip", "again")
    assert log.read_text(encoding="utf-8") == first
    # A third version appends; it does not rewrite what is above it.
    tool.append_version(3, tmp_path / "v2", tmp_path / "v2.zip", "third")
    assert log.read_text(encoding="utf-8").startswith(first)


def test_a_new_zip_without_a_gitattributes_entry_is_refused(tool, slides,
                                                            tmp_path, monkeypatch):
    _, archive = build(tool, slides, tmp_path, "v2", monkeypatch)
    body = (tmp_path / "gitattributes").read_text(encoding="utf-8")
    assert f"{tool._rel(archive)} binary" in body
    # Exact scope: the entry names one path and introduces no pattern.
    assert "*" not in body.splitlines()[-1]
    stripped = "\n".join(line for line in body.splitlines()
                         if not line.endswith(" binary"))
    assert f"{tool._rel(archive)} binary" not in stripped


# --- what the package must and must not contain -----------------------------

def test_a_stale_sha256sums_is_refused(tool, slides, tmp_path, monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    assert tool.package_problems(out, 2) == []
    target = out / "README.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nextra\n",
                      encoding="utf-8")
    problems = tool.package_problems(out, 2)
    assert any("stale" in p for p in problems), problems


def test_a_model_weight_in_the_package_is_refused(tool, slides, tmp_path,
                                                  monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    for suffix in (".pt", ".safetensors", ".bin"):
        planted = out / f"source_samples/model{suffix}"
        planted.write_bytes(b"\x00" * 16)
        problems = tool.package_problems(out, 2)
        assert any("model weight" in p for p in problems), (suffix, problems)
        planted.unlink()
    assert any("stale" in p for p in tool.package_problems(out, 2)) or \
        tool.package_problems(out, 2) == []


def test_an_absolute_home_path_in_the_package_is_refused(tool, slides, tmp_path,
                                                         monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    monkeypatch.setattr(tool, "OUT", out)
    tool.privacy_scan()                      # clean as built
    (out / "source_samples" / "leak.py").write_text(
        '# see /Users/someone/Desktop/private\n', encoding="utf-8")
    with pytest.raises(ValueError, match="privacy scan failed"):
        tool.privacy_scan()
    # Every file the photo track contributes is screened, so a sample added
    # later that carries one of these patterns fails here and not at build.
    for source in tool.PHOTO_COPIES.values():
        body = (ROOT / source).read_text(errors="replace")
        hit = [label for label, pattern in tool.PRIVATE_PATTERNS.items()
               if pattern.search(body)]
        assert hit == [], f"{source}: {hit}"


def test_a_package_without_the_synthetic_only_disclosure_is_refused(
        tool, slides, tmp_path, monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    assert tool.package_problems(out, 2) == []
    # The PDF carries the same words, and inspectable_text reads it, so the
    # sentence has to leave every file that holds it before it is gone.
    for carrier in sorted(out.rglob("*")):
        if carrier.is_file() and tool.SYNTHETIC_ONLY_DISCLOSURE in tool.inspectable_text(carrier):
            if carrier.suffix == ".md":
                carrier.write_text(carrier.read_text(encoding="utf-8").replace(
                    tool.SYNTHETIC_ONLY_DISCLOSURE, "評估以合成資料為主。"), encoding="utf-8")
            else:
                carrier.unlink()
    problems = tool.package_problems(out, 2)
    assert any("synthetic-only disclosure" in p for p in problems), problems


def test_a_package_that_summarises_the_probe_scope_instead_of_quoting_it_is_refused(
        tool, slides, tmp_path, monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    # The PDF carries the same words, and inspectable_text reads it, so the
    # sentence has to leave every file that holds it before it is gone.
    for carrier in sorted(out.rglob("*")):
        if carrier.is_file() and tool.PROBE_SCOPE_QUOTED in tool.inspectable_text(carrier):
            if carrier.suffix == ".md":
                carrier.write_text(carrier.read_text(encoding="utf-8").replace(
                    tool.PROBE_SCOPE_QUOTED, "八個零件的真實照片探針，僅供參考。"), encoding="utf-8")
            else:
                carrier.unlink()
    problems = tool.package_problems(out, 2)
    assert any("scope sentence is summarised" in p for p in problems), problems
    # That the two quoted strings are the photo track's own, and not a
    # reworded copy, is asserted from the photo track's side, in
    # tests/test_bricknet_photo_eval.py::test_the_deliverable_quotes_the_photo
    # _tracks_own_two_sentences. It cannot be asserted here: this module runs
    # in a tree that may not carry src/bricknet_ext at all, and importing it
    # would make the deliverable's tests depend on the extension -- the
    # dependency direction tests/test_bricknet_regression.py pins.


def test_an_unregistered_photo_number_in_a_deliverable_is_refused(tool, slides,
                                                                  tmp_path,
                                                                  monkeypatch):
    """The same judgment as the deck's, not a second one."""
    numbers = importlib.util.spec_from_file_location(
        "presentation_numbers", ROOT / "tests/test_presentation_numbers.py")
    module = importlib.util.module_from_spec(numbers)
    numbers.loader.exec_module(module)
    registry = json.loads(module.CLAIMS.read_text(encoding="utf-8"))
    declared = module._declared(registry) | module.STRUCTURAL
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    # v2's own section, which is what this version adds. v1's prose was never
    # under the registry and putting it there now is a different job.
    added = tool.PHOTO_EVIDENCE
    assert added in (out / "BrickAgain_技術與實作證據.md").read_text(encoding="utf-8")
    # The two sentences that must be quoted verbatim are governed by that
    # rule, not by this one: a figure inside them cannot be re-displayed or
    # rounded, which is what registration protects against. Requiring both at
    # once would mean editing a sentence the package may not edit.
    for quoted in (tool.SYNTHETIC_ONLY_DISCLOSURE, tool.PROBE_SCOPE_QUOTED):
        added = added.replace(quoted, "")
    unexplained = [shown.strip() for shown
                   in module._FIGURE.findall(module._without_quoted_block(added))
                   if shown.strip() not in declared
                   and shown.strip().replace(" ", "") not in declared]
    assert unexplained == [], unexplained
    # An unregistered figure in the same document is caught by that machinery.
    salted = module._without_quoted_block(added) + "\n\nphoto_synth_crop_top1 0.4271\n"
    assert any(shown.strip() == "0.4271"
               for shown in module._FIGURE.findall(salted))
    assert "0.4271" not in declared


def test_v2_built_twice_into_temp_directories_is_identical(tool, slides,
                                                           tmp_path, monkeypatch):
    """Reproducibility, proved forwards rather than by rebuilding v1."""
    first, first_zip = build(tool, slides, tmp_path, "one", monkeypatch)
    (tmp_path / "VERSIONS.md").unlink()
    second, second_zip = build(tool, slides, tmp_path, "two", monkeypatch)

    def files(folder):
        return {path.relative_to(folder).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(folder.rglob("*")) if path.is_file()}

    assert files(first) == files(second)
    assert (first / "SHA256SUMS.txt").read_text(encoding="utf-8") == \
        (second / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert tool.sha256(first_zip) == tool.sha256(second_zip)
    assert tool.directory_digest(first) == tool.directory_digest(second)


# --- M5-2: the PDF and the .pptx have to be the same document ---------------

def test_the_delivered_v2_pdf_is_not_the_deck_shipped_beside_it(tool):
    """Why there is a v3. This is a description of v2, not a demand on it.

    v2's `deck_pdf` took its page count from `presentation/deck/
    render_manifest.json`, which describes the showcase deck -- 11 main plus
    24 appendix. The research report has 30 slides and never grew; they are
    two different decks. So v2 shipped the showcase deck under the research
    report's file name, beside a .pptx that is the research report.
    """
    v2 = PRESENTATION / "teacher_review_v2"
    if not v2.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} v2 is not published")
    from pypdf import PdfReader
    pages = len(PdfReader(v2 / f"{tool.RESEARCH_STEM}.pdf").pages)
    slides = tool.slide_count(v2 / f"{tool.RESEARCH_STEM}.pptx")
    assert (pages, slides) == (35, 30)
    assert any("same document" in problem
               for problem in tool.package_problems(v2, 2))
    # v1 is the version that got this right, and it still does.
    assert tool.package_problems(PRESENTATION / "teacher_review", 1) == []


def test_a_render_of_another_deck_is_refused(tool, slides, report, tmp_path,
                                             monkeypatch):
    """One extra PNG is a different deck, and the build says so."""
    from PIL import Image
    Image.new("RGB", (320, 180), (9, 9, 9)).save(slides / "slide-99.png")
    monkeypatch.setattr(tool, "VERSIONS", tmp_path / "VERSIONS.md")
    monkeypatch.setattr(tool, "GITATTRIBUTES", tmp_path / "gitattributes")
    (tmp_path / "gitattributes").write_text("# exact-scope rules only\n")
    with pytest.raises(ValueError, match="a render of a different deck"):
        tool.main(["--slides-dir", str(slides), "--version", "2",
                   "--test-report", str(report),
                   "--out-dir", str(tmp_path / "v2"),
                   "--archive", str(tmp_path / "v2.zip")])


def test_an_adopted_pdf_with_the_wrong_page_count_is_refused(tool, tmp_path):
    """The carried-over route is held to the same count as a fresh render."""
    wrong = PRESENTATION / "deck" / "brickagain_showcase.pdf"
    if not wrong.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the showcase deck is not published")
    with pytest.raises(ValueError, match="not the same document"):
        tool.adopt_deck_pdf(wrong, tmp_path / "out.pdf",
                            tool.slide_count(RESEARCH_PPTX))
    assert not (tmp_path / "out.pdf").exists()
    right = PRESENTATION / "teacher_review" / f"{tool.RESEARCH_STEM}.pdf"
    tool.adopt_deck_pdf(right, tmp_path / "out.pdf",
                        tool.slide_count(RESEARCH_PPTX))
    assert (tmp_path / "out.pdf").read_bytes() == right.read_bytes()


def test_a_planted_pdf_of_the_wrong_length_is_caught_after_the_build(
        tool, slides, tmp_path, monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    assert not any("same document" in p
                   for p in tool.package_problems(out, 2))
    showcase = PRESENTATION / "deck" / "brickagain_showcase.pdf"
    if not showcase.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the showcase deck is not published")
    (out / f"{tool.RESEARCH_STEM}.pdf").write_bytes(showcase.read_bytes())
    assert any("same document" in p for p in tool.package_problems(out, 2))


# --- M5-2: the test summary is measured, not reprinted ----------------------

def test_a_report_measured_against_another_revision_is_refused(tool, tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "date": "2026-09-10", "revision": "8c71fc46aeb8", "passed": 389,
        "groups": [{"title": "A", "tests": ["tests/test_retile.py"],
                    "passed": 389, "summary": "389 passed"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="re-run the tests rather than"):
        tool.read_test_report(path, tool.git_revision())


def test_a_total_that_is_not_the_sum_of_its_groups_is_refused(tool, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "date": "2026-09-17", "revision": tool.git_revision(), "passed": 389,
        "groups": [{"title": "A", "tests": ["tests/test_retile.py"],
                    "passed": 181, "summary": "181 passed"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the sum of its groups"):
        tool.read_test_report(path, tool.git_revision())


def test_a_test_sample_no_group_ran_is_refused(tool, tmp_path):
    """v2 added six photo test files and ran none of them."""
    copies = {**tool.COPIES, **tool.PHOTO_COPIES}
    report = {"groups": [{"tests": ["tests/test_retile.py"]}]}
    problems = tool.coverage_problems(copies, report)
    assert any("test_bricknet_photo_ui.py" in p for p in problems), problems
    carried = [origin for destination, origin in copies.items()
               if destination.startswith("test_samples/")]
    assert tool.coverage_problems(
        copies, {"groups": [{"tests": carried}]}) == []


def test_the_summary_reprints_the_measured_line_and_the_measured_date(
        tool, tmp_path):
    revision = tool.git_revision()
    report = {"date": "2026-09-17", "revision": revision, "passed": 3,
              "groups": [{"title": "甲", "tests": ["tests/test_retile.py"],
                          "passed": 3, "summary": "3 passed in 1.23s"}]}
    body = tool.test_summary(report, revision)
    assert "2026-09-17" in body and "`3 passed in 1.23s`" in body
    assert "合計 **3 passed**" in body
    # None of v1's literals survive into a summary that did not measure them.
    for stale in ("2026-09-10", "389", "208", "181", "10.24s", "1.38s"):
        assert stale not in body, stale


# --- M5-2: the index describes every file that travels ----------------------

def test_every_travelling_sample_is_in_the_generated_index(tool):
    copies = {**tool.COPIES, **tool.PHOTO_COPIES}
    index = tool.source_index(copies)
    for destination, origin in copies.items():
        folder, _, name = destination.partition("/")
        if folder in ("source_samples", "test_samples") and name.endswith(".py"):
            assert f"`{name}`" in index, name
            assert f"`{origin}`" in index, origin


def test_a_source_with_no_index_line_fails_the_build(tool):
    copies = {**tool.COPIES, "source_samples/undocumented.py": "src/x.py"}
    with pytest.raises(ValueError, match="no line in SOURCE_FOCUS"):
        tool.source_index(copies)


def test_a_sample_missing_from_the_index_is_caught_after_the_build(
        tool, slides, tmp_path, monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v3", monkeypatch)
    assert tool.package_problems(out, 3) == []
    index = out / "source_samples" / "README.md"
    index.write_text(index.read_text(encoding="utf-8").replace(
        "`photo_confirm.py`", "`(removed)`"), encoding="utf-8")
    problems = tool.package_problems(out, 3)
    assert any("photo_confirm.py" in p and "not in" in p
               for p in problems), problems


def test_the_evidence_census_counts_the_files_that_travel(tool, slides,
                                                          tmp_path, monkeypatch):
    out, _ = build(tool, slides, tmp_path, "v2", monkeypatch)
    evidence = (out / "BrickAgain_技術與實作證據.md").read_text(encoding="utf-8")
    sources = len(list((out / "source_samples").glob("*.py")))
    tests = len(list((out / "test_samples").glob("*.py")))
    assert f"- {sources} 份代表性原始碼" in evidence
    assert f"- {tests} 份對應測試" in evidence
    # v1's literals were 13 and 10 while v2 carried 22 and 16.
    assert "13 份代表性原始碼" not in evidence
    assert "10 份對應測試" not in evidence
    assert "@" not in evidence.split("## 6.")[1].split("## 7.")[0]


def test_a_rehearsal_build_cannot_write_its_temp_path_into_a_tracked_file(
        tool, slides, report, tmp_path, monkeypatch):
    """A file inside the repository never records a path from outside it.

    The dry run that found this left an absolute path under the operator's
    home directory -- `~/…/dryrun/v3.zip`, spelled out in full -- in both
    presentation/VERSIONS.md and .gitattributes -- an absolute home path in a
    tracked file, which is exactly what privacy_scan keeps out of the package.
    The other tests redirect both records into tmp_path, where recording a tmp
    path is intended, so none of them could see it.
    """
    monkeypatch.setattr(tool, "GITATTRIBUTES", tmp_path / "gitattributes")
    (tmp_path / "gitattributes").write_text("# exact-scope rules only\n")
    before = tool.VERSIONS.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="record a path from outside it"):
        tool.main(["--slides-dir", str(slides), "--version", "2",
                   "--test-report", str(report),
                   "--out-dir", str(tmp_path / "rehearsal"),
                   "--archive", str(tmp_path / "rehearsal.zip"),
                   "--zip-date", "2026-09-13"])
    assert tool.VERSIONS.read_text(encoding="utf-8") == before
    assert "## v2\n" in before and str(tmp_path) not in before
    # And the same rule covers .gitattributes on its own.
    monkeypatch.setattr(tool, "GITATTRIBUTES", ROOT / ".gitattributes")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="record a path from outside it"):
        tool.note_binary(tmp_path / "rehearsal.zip")
    assert (ROOT / ".gitattributes").read_text(encoding="utf-8") == attributes
