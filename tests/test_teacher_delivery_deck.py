"""The twelve-slide deck that goes to the teacher, checked as delivered.

``presentation/tools/test_capability_deck.py`` checks the capability deck the
twelve slides are drawn from, but it sits beside its builder and runs only when
someone runs it. This file puts the delivered pair itself in the suite: both
formats, every slide string of the delivery edition, the numbers behind them,
and the full run that slide 11 cites.

The PPTX is edited by ``prepare_teacher_delivery_deck.mjs`` and, on this Mac,
by ``patch_delivery_pptx.py``, while the PDF is drawn from
``build_teacher_deck_pdf.deck("delivery")``. Two routes to one deck is exactly
where a correction lands in one file and not the other, so each slide string
is looked for in both.

Everything read here is under ``presentation/``, which is not published. In the
public tree every test in this file skips as artifact-only, and each is
declared as such in ``tests/test_public_snapshot.py``.
"""
from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "presentation" / "tools"
DECK = ROOT / "presentation" / "teacher_review_deck"
PDF = DECK / "BrickAgain_技術簡報_送審版.pdf"
PPTX = DECK / "BrickAgain_技術簡報_送審版.pptx"
ARTIFACT_ONLY = "artifact-only:"
SLIDES = 12

pytestmark = pytest.mark.skipif(
    not (PDF.is_file() and PPTX.is_file() and TOOLS.is_dir()),
    reason=f"{ARTIFACT_ONLY} the delivered teacher deck and its builders are not published")

#: Text that an earlier edition carried and this one must not.
STALE = ("11限制", "12示範", "1,251", "58.992", "Llama", "現場示範以檢索為主",
         "模型沒有把訓練資料背下來", "這是還沒解決的問題")


def squeeze(text: str) -> str:
    """Whitespace removed: the two formats break lines in different places."""
    return re.sub(r"\s+", "", text)


@pytest.fixture(scope="module")
def tools():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import build_capability_deck
    import build_teacher_deck_pdf
    import deck_content
    return build_capability_deck, build_teacher_deck_pdf, deck_content


@pytest.fixture(scope="module")
def pdf_pages() -> list[str]:
    from pypdf import PdfReader
    return [squeeze(page.extract_text() or "") for page in PdfReader(PDF).pages]


@pytest.fixture(scope="module")
def pptx_slides() -> list[str]:
    """Each slide's text, in the order presentation.xml plays them."""
    with zipfile.ZipFile(PPTX) as archive:
        order = re.findall(r'<p:sldId\b[^>]*\br:id="([^"]+)"',
                           archive.read("ppt/presentation.xml").decode("utf-8"))
        rels = archive.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
        target = {}
        for rel in re.findall(r"<Relationship\b[^>]*>", rels):
            rid = re.search(r'\bId="([^"]+)"', rel).group(1)
            path = re.search(r'\bTarget="([^"]+)"', rel).group(1)
            target[rid] = path.lstrip("/") if path.startswith("/") else f"ppt/{path}"
        return [squeeze(html.unescape(re.sub(r"<[^>]+>", "", archive.read(target[rid])
                                              .decode("utf-8"))))
                for rid in order]


def stale_citation(cited: Path, logs: Path) -> str | None:
    """A problem when a newer full-suite log than the cited one exists."""
    newest = sorted(logs.glob("full_suite_*.txt"))
    if not newest:
        return "no full-suite logs at all"
    if newest[-1].name != cited.name:
        return f"slide 11 cites {cited.name} but {newest[-1].name} is newer"
    return None


def slide_eleven_problems(text: str, tail: str) -> list[str]:
    """What slide 11 must say about the run whose pytest summary is ``tail``."""
    problems = []
    for word in ("failed", "error"):
        if word in tail:
            problems.append(f"the slide claims no failures; the log says {tail!r}")
    passed = re.search(r"(\d+) passed", tail)
    if not passed:
        return problems + [f"no pass count in {tail!r}"]
    if f"{int(passed.group(1)):,}" not in text:
        problems.append(f"the slide does not state the {passed.group(1)} passed")
    skipped = re.search(r"(\d+) skipped", tail)
    shown = str(int(skipped.group(1))) if skipped else "0"
    if f"{shown}略過" not in text:
        problems.append(f"the slide does not state the {shown} skipped")
    if "0失敗" not in text:
        problems.append("the slide does not state zero failures")
    return problems


def test_both_formats_have_twelve_slides(pdf_pages, pptx_slides):
    assert len(pdf_pages) == SLIDES
    assert len(pptx_slides) == SLIDES


def test_every_delivered_string_reaches_both_formats(tools, pdf_pages, pptx_slides):
    capability, teacher, _ = tools
    edition = teacher.deck("delivery")
    assert len(edition) == SLIDES
    missing = []
    for n, slide in enumerate(edition, 1):
        for wanted in capability.slide_strings(slide):
            # Laid-out lines break long strings; the first line's first twelve
            # characters are what survives both layouts intact.
            probe = squeeze(wanted.split("\n")[0])[:12]
            for label, pages in (("PDF", pdf_pages), ("PPTX", pptx_slides)):
                if probe not in pages[n - 1]:
                    missing.append(f"slide {n} {label}: {probe}")
    assert missing == [], missing


def test_the_numbers_the_deck_is_drawn_from_are_the_reports(tools):
    capability, _, _ = tools
    assert capability.claim_problems() == []


def test_slide_eleven_states_the_latest_full_run(tools, pdf_pages, pptx_slides):
    capability, _, content = tools
    cited = ROOT / content.FULL_SUITE
    assert cited.is_file(), f"slide 11 cites {content.FULL_SUITE}, which is not here"
    assert stale_citation(cited, cited.parent) is None
    tail = capability.pytest_summary(cited.read_text(encoding="utf-8", errors="replace"))
    assert tail, "no pytest summary line in the cited log"
    for label, pages in (("PDF", pdf_pages), ("PPTX", pptx_slides)):
        assert slide_eleven_problems(pages[10], tail) == [], label


def test_the_language_caveat_names_the_encoder_it_was_measured_on(pdf_pages, pptx_slides):
    import json
    arms = {row["arm"]: row for row in json.loads(
        (ROOT / "data/reports/bricknet/69_retrieval_arms.json").read_text(encoding="utf-8"))["arms"]}
    zh, en = arms["cap0_q4_zh_n500_dcae10fa31a5"], arms["cap0_q4_en_n500_dcae10fa31a5"]
    before = arms["cap0_q4_n2000_dcae10fa31a5"]
    assert zh["identity_digest"] == en["identity_digest"] == before["identity_digest"]
    for label, pages in (("PDF", pdf_pages), ("PPTX", pptx_slides)):
        text = pages[5]
        assert "微調前" in text, f"{label}: the split was measured before fine-tuning"
        assert f"中文{zh['recall_at_1'] * 100:.1f}%" in text, label
        assert f"英文{en['recall_at_1'] * 100:.1f}%" in text, label
        assert "上限" in text, f"{label}: the Chinese figure is a ceiling"


def test_the_delivered_deck_shows_no_date_and_no_stale_text(pdf_pages, pptx_slides):
    for label, pages in (("PDF", pdf_pages), ("PPTX", pptx_slides)):
        for n, text in enumerate(pages, 1):
            assert not re.search(r"20\d{2}[-/年.]\d{1,2}", text), f"{label} slide {n} shows a date"
            for stale in STALE:
                assert stale not in text, f"{label} slide {n} still says {stale}"


def test_the_two_guards_fail_when_they_should(tmp_path):
    """Both helpers above, against inputs where the answer is known."""
    for day in ("2026-09-18", "2026-09-30"):
        (tmp_path / f"full_suite_{day}.txt").write_text("", encoding="utf-8")
    assert stale_citation(tmp_path / "full_suite_2026-09-18.txt", tmp_path)
    assert stale_citation(tmp_path / "full_suite_2026-09-30.txt", tmp_path) is None

    tail = "7217 passed, 25 skipped, 3 warnings in 6414.13s (1:46:54)"
    assert slide_eleven_problems("7,217項通過0失敗25略過", tail) == []
    assert slide_eleven_problems("7,217項通過0失敗0略過", tail)
    assert slide_eleven_problems("7,162項測試0失敗0略過", tail)
    assert slide_eleven_problems("7,217項通過0失敗25略過",
                                 "1 failed, 7216 passed, 25 skipped in 1.00s")
