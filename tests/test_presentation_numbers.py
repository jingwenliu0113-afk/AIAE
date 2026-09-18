"""The showcase deliverables, held to the artefacts they quote.

``presentation/`` is not published, so most of this file cannot run in the
public snapshot. That is a boundary, not a licence: the parts that *can* run
anywhere -- the registry parser and the over-claim tripwire -- are exercised
against synthetic input in every tree, and the parts that need the private
evidence are named one by one in
``tests/test_public_snapshot.py::ARTIFACT_ONLY_NODES``.

Two shapes, on purpose:

* **integration** -- a fixed, small number of node ids that loop over every
  claim. Adding a claim must not add a node id, because every artifact-only
  node has to be declared by name and a declaration that grows with the
  content stops being a declaration.
* **behavioural** -- synthetic registries fed to the same parser, one defect
  each: a wrong value, a missing field, a duplicate id, a path that does not
  resolve, a display that rounds further than it declares. A parser only
  tested on a good document is a parser nobody has tested.
"""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / "presentation"
CLAIMS = PRESENTATION / "audit" / "claims.json"

ARTIFACT_ONLY = "artifact-only:"

PRECISIONS = frozenset({"int", "ratio", "pct", "pp", "float", "digest16",
                        "text", "bool"})
DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})
REQUIRED = ("claim_id", "used_in", "display", "value", "source", "precision",
            "qualifier")

MINUS = "−"          # the typographic minus the deliverables use
ELLIPSIS = "…"


# ---------------------------------------------------------------------------
# path evaluation
# ---------------------------------------------------------------------------

_SEGMENT = re.compile(r'''
    \.(?P<bare>[A-Za-z_][A-Za-z0-9_]*)      # .field
  | \["(?P<quoted>[^"]+)"\]                 # ["field-with-a-dash"]
''', re.VERBOSE)


class PathError(Exception):
    """A path that cannot be parsed, or cannot be resolved."""


def parse_path(path: str) -> list[str]:
    """`.a.b["c-d"]` -> ['a', 'b', 'c-d'].

    A hyphenated key must be written the way jq would accept it. Phase 3C's
    primary contrast is keyed `C-B`, and `.contrasts.C-B` is a subtraction in
    jq, not a lookup -- so the bracket form is required rather than preferred.
    """
    if not isinstance(path, str) or not path.startswith("."):
        raise PathError(f"a path must be a string starting with '.': {path!r}")
    at, out = 0, []
    while at < len(path):
        m = _SEGMENT.match(path, at)
        if m is None:
            raise PathError(f"cannot parse {path!r} at offset {at}")
        out.append(m.group("bare") or m.group("quoted"))
        at = m.end()
    if not out:
        raise PathError(f"empty path: {path!r}")
    return out


def resolve(doc, path: str):
    node = doc
    for key in parse_path(path):
        if not isinstance(node, dict) or key not in node:
            raise PathError(f"{path!r} does not resolve at {key!r}")
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def _select(rows, select):
    for row in rows:
        if isinstance(row, dict) and row.get(select["field"]) == select["equals"]:
            return row
    raise PathError(f"no row where {select['field']} == {select['equals']!r}")


def evaluate(claim: dict, load):
    """The value the artefact actually holds, for this claim."""
    rule = claim.get("rule")
    if rule is None:
        return resolve(load(claim["source"]), claim["path"])

    kind = rule.get("kind")
    if kind == "fraction":
        doc = load(claim["source"])
        return [resolve(doc, rule["numerator_path"]),
                resolve(doc, rule["denominator_path"])]
    if kind == "count_where":
        doc = load(claim["source"])
        rows = resolve(doc, rule["path"])
        if not isinstance(rows, list):
            raise PathError(f"{rule['path']!r} is not a list")
        n = sum(1 for r in rows if r.get(rule["field"]) == rule["equals"])
        return [n, resolve(doc, rule["denominator_path"])]
    if kind == "length":
        rows = resolve(load(claim["source"]), rule["path"])
        if not isinstance(rows, list):
            raise PathError(f"{rule['path']!r} is not a list")
        return len(rows)
    if kind == "length_sum":
        total = 0
        for source, path in rule["parts"]:
            rows = resolve(load(source), path)
            if not isinstance(rows, list):
                raise PathError(f"{path!r} in {source} is not a list")
            total += len(rows)
        return total
    if kind == "select_field":
        rows = resolve(load(claim["source"]), rule["path"])
        return _select(rows, rule["select"])[rule["field"]]
    if kind == "contains":
        rows = resolve(load(claim["source"]), rule["path"])
        row = _select(rows, rule["select"])
        blob = row.get(rule["field"])
        return json.dumps(blob, ensure_ascii=False)
    if kind == "contains_text":
        return load(claim["source"])
    raise PathError(f"unknown rule kind {kind!r}")


# ---------------------------------------------------------------------------
# display contract
# ---------------------------------------------------------------------------

_NUM = re.compile(r"[-−+]?\d[\d,]*(?:\.\d+)?")


def _decimals(text: str) -> int:
    _, _, frac = text.partition(".")
    return len(frac) if frac else 0


def _number(text: str) -> tuple[float, int]:
    m = _NUM.search(text)
    if m is None:
        raise PathError(f"no number in display {text!r}")
    raw = m.group(0).replace(",", "").replace(MINUS, "-")
    return float(raw), _decimals(raw)


def _quotes(haystack: str, needle: str) -> bool:
    """Containment, minus the way a number hides inside a longer one.

    Plain ``in`` says "9 of the plan's 56 files" is present in a document that
    only ever says **49** of them -- so a quotation that drops a digit would
    verify. A quoted figure has to start where a number starts.
    """
    at = haystack.find(needle)
    while at != -1:
        before = haystack[at - 1] if at else ""
        after_at = at + len(needle)
        after = haystack[after_at] if after_at < len(haystack) else ""
        head_ok = not (needle[:1].isdigit() and before.isdigit())
        tail_ok = not (needle[-1:].isdigit() and after.isdigit())
        if head_ok and tail_ok:
            return True
        at = haystack.find(needle, at + 1)
    return False


def display_problems(claim: dict, actual) -> list[str]:
    """Is the audience-facing string a faithful rendering of `actual`?

    Tolerance is half a unit at the precision the display itself shows, so
    0.0625 written to three decimals may be 0.062 or 0.063: which of the two
    a rounding convention picks is not evidence of anything, and a test that
    hard-codes one convention fails on a document that is not wrong.
    """
    kind, shown, declared = claim["precision"], claim["display"], claim["value"]
    cid = claim["claim_id"]
    bad: list[str] = []

    if kind == "text":
        if claim.get("rule", {}).get("kind") in {"contains", "contains_text"}:
            if not isinstance(actual, str) or not _quotes(actual, declared):
                bad.append(f"{cid}: {declared!r} is not in the artefact text")
            return bad
        if actual != declared or shown != declared:
            bad.append(f"{cid}: text {shown!r}/{declared!r} vs {actual!r}")
        return bad

    if kind == "bool":
        if actual is not declared:
            bad.append(f"{cid}: {declared!r} vs artefact {actual!r}")
        if shown != ("on" if declared else "off"):
            bad.append(f"{cid}: bool display {shown!r}")
        return bad

    if kind == "digest16":
        if not isinstance(actual, str) or not actual.startswith(declared):
            bad.append(f"{cid}: {declared!r} is not the prefix of {actual!r}")
        if len(declared) != 16:
            bad.append(f"{cid}: short codes are 16 hex characters, "
                       f"{declared!r} is {len(declared)}")
        if shown != declared + ELLIPSIS:
            bad.append(f"{cid}: digest display {shown!r}")
        return bad

    if kind == "ratio":
        if not (isinstance(declared, list) and len(declared) == 2):
            return [f"{cid}: a ratio value is [numerator, denominator]"]
        if list(actual) != list(declared):
            bad.append(f"{cid}: {declared} vs artefact {list(actual)}")
        if shown != f"{declared[0]}/{declared[1]}":
            bad.append(f"{cid}: ratio display {shown!r}")
        return bad

    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return [f"{cid}: artefact holds {actual!r}, which is not a number"]
    if abs(float(actual) - float(declared)) > 1e-9:
        bad.append(f"{cid}: declared {declared} vs artefact {actual}")

    try:
        value, digits = _number(shown)
    except PathError as exc:
        return bad + [f"{cid}: {exc}"]

    if kind == "int":
        if digits or value != float(declared):
            bad.append(f"{cid}: int display {shown!r} vs {declared}")
        return bad
    if kind in {"pct", "pp"}:
        value, digits = value / 100.0, digits + 2
    tolerance = 0.5 * 10 ** (-digits)
    if abs(value - float(declared)) > tolerance + 1e-12:
        bad.append(f"{cid}: display {shown!r} is further than half a unit "
                   f"from {declared}")
    return bad


# ---------------------------------------------------------------------------
# the registry parser, which is what the behavioural tests exercise
# ---------------------------------------------------------------------------

def registry_problems(payload, *, load) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"),
                                                       list):
        return ["the registry is not {'claims': [...]}"]
    claims = payload["claims"]
    if not claims:
        return ["the registry is empty"]

    bad: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        cid = claim.get("claim_id", "<no id>")
        missing = [f for f in REQUIRED if f not in claim]
        if missing:
            bad.append(f"{cid}: missing {missing}")
            continue
        if cid in seen:
            bad.append(f"{cid}: duplicate claim id")
            continue
        seen.add(cid)
        if claim["precision"] not in PRECISIONS:
            bad.append(f"{cid}: unknown precision {claim['precision']!r}")
            continue
        if "direction" in claim and claim["direction"] not in DIRECTIONS:
            bad.append(f"{cid}: unknown direction {claim['direction']!r}")
            continue
        if ("path" in claim) == ("rule" in claim):
            bad.append(f"{cid}: needs exactly one of path or rule")
            continue
        if not claim.get("used_in"):
            bad.append(f"{cid}: names no place it is used")
            continue
        try:
            actual = evaluate(claim, load)
        except (PathError, KeyError, TypeError, FileNotFoundError) as exc:
            bad.append(f"{cid}: {type(exc).__name__}: {exc}")
            continue
        bad.extend(display_problems(claim, actual))
    return bad


# ---------------------------------------------------------------------------
# slide geometry
#
# `presentation/tools/deckkit.py` imports these two names and runs them over
# the finished deck. They live here so the injection tests below -- a rule
# drawn through a second line, text pushed out of its panel -- run in the
# published tree as well, where `presentation/` is not present.
# ---------------------------------------------------------------------------

W_IN, H_IN = 13.333, 7.5

#: A separator belongs in the gutter. Round 71 allowed 0.014in and therefore
#: passed a rule drawn 0.017in under the second line of a two-line table cell.
RULE_CLEARANCE = 0.045
PANEL_PADDING = 0.05
PAGE_MARGIN = 0.05


def _overlap(a, b) -> tuple[float, float]:
    return (min(a[2], b[2]) - max(a[0], b[0]),
            min(a[3], b[3]) - max(a[1], b[1]))


def layout_problems(slide, texts, rules, panels, images) -> list[str]:
    """Everything the arithmetic in the builder cannot see.

    Each text entry is (label, ink, line). Two measurements, two questions:
    **ink** is where the glyphs are, and two blocks stacked against each other
    share leading, which is normal typography -- so overlap is judged on ink.
    **line** is the whole line box; a rule inside the leading is touching the
    text even when it clears the glyphs, and a panel has to hold the line box.
    """
    out: list[str] = []

    for label, box, _line in texts:
        if box[0] < PAGE_MARGIN or box[2] > W_IN - PAGE_MARGIN:
            out.append(f"slide {slide}: {label!r} runs off the page "
                       f"horizontally ({box[0]:.2f}..{box[2]:.2f}in)")
        if box[1] < 0 or box[3] > H_IN - PAGE_MARGIN:
            out.append(f"slide {slide}: {label!r} runs off the page "
                       f"vertically ({box[1]:.2f}..{box[3]:.2f}in)")

    for box in list(panels) + list(rules):
        if (box[3] > H_IN - PAGE_MARGIN or box[2] > W_IN - PAGE_MARGIN
                or box[0] < 0 or box[1] < 0):
            out.append(f"slide {slide}: a container or rule at "
                       f"({box[0]:.2f},{box[1]:.2f})-({box[2]:.2f},"
                       f"{box[3]:.2f}) runs off the page")

    for i, (a_label, a, _) in enumerate(texts):
        for b_label, b, _b in texts[i + 1:]:
            ox, oy = _overlap(a, b)
            if ox > 0.02 and oy > 0.012:
                out.append(f"slide {slide}: {a_label!r} overlaps {b_label!r} "
                           f"by {ox:.2f}x{oy:.2f}in")
        for img in images:
            ox, oy = _overlap(a, img)
            if ox > 0.02 and oy > 0.012:
                out.append(f"slide {slide}: {a_label!r} overlaps a picture "
                           f"by {ox:.2f}x{oy:.2f}in")

    for label, _ink, box in texts:
        for rule in rules:
            if min(box[2], rule[2]) - max(box[0], rule[0]) <= 0.02:
                continue
            gap = max(rule[1] - box[3], box[1] - rule[3])
            if gap < RULE_CLEARANCE:
                out.append(
                    f"slide {slide}: a rule at y={rule[1]:.2f} is {gap:.3f}in "
                    f"from {label!r} (needs {RULE_CLEARANCE}in of gutter)")

    for label, _ink, box in texts:
        for panel in panels:
            ox, oy = _overlap(box, panel)
            if ox <= 0 or oy <= 0:
                continue
            if (box[0] < panel[0] + PANEL_PADDING
                    or box[2] > panel[2] - PANEL_PADDING
                    or box[1] < panel[1] + PANEL_PADDING
                    or box[3] > panel[3] - PANEL_PADDING):
                out.append(
                    f"slide {slide}: {label!r} crosses the edge of the panel "
                    f"it sits in ({panel[0]:.2f},{panel[1]:.2f})-"
                    f"({panel[2]:.2f},{panel[3]:.2f})")
    return out


# ---------------------------------------------------------------------------
# over-claim tripwire
# ---------------------------------------------------------------------------

#: Each entry is the assertive shape of a claim the master document forbids,
#: plus the words that mark a *denial* of it. Grepping for the phrase alone
#: is how a disclaimer ends up tripping the wire written to protect it.
FORBIDDEN = (
    ("六臂 2247 到 95 是訓練效果",
     re.compile(r"2247\s*[-→–>→]+\s*95"),
     ("同時", "不能", "不得", "組合", "只能描述", "不可", "必須改成", "無法")),
    ("硬約束提升模型準確率",
     re.compile(r"(?:硬約束|約束|gate).{0,12}(?:準確率|正確率).{0,6}(?:提升|拉到|到)\s*100"),
     ("不是", "不得", "不可", "必須改成", "並非")),
    ("100% 是品質提升",
     re.compile(r"100\s*%.{0,10}(?:代表|證明|表示).{0,10}(?:品質|能力).{0,6}(?:提升|更好)"),
     ("不", "並非", "必須改成")),
    ("影像辨識準確率 58%",
     re.compile(r"(?:影像)?辨識準確率.{0,6}5[0-9](?:\.\d)?\s*%"),
     ("不是", "不得", "不可", "必須改成", "並非")),
    ("使用者看不到或無從修正",
     re.compile(r"使用者.{0,8}(?:看不到|無從修正|不會看到)"),
     ("不得", "不可", "必須改成", "並非", "錯誤說法")),
    ("connected 代表物理穩定",
     re.compile(r"connected.{0,12}(?:代表|等於|證明).{0,8}(?:物理)?(?:穩定|站得住)"),
     ("不", "並非", "必須改成")),
    ("V3 證明 OOD 偵測失效",
     re.compile(r"V3.{0,20}(?:OOD|out-of-distribution).{0,12}(?:失效|能力)"),
     ("不", "禁止", "並非", "必須改成", "尚未")),
    ("19 個世代都因缺陷作廢",
     re.compile(r"19\s*個?世代?.{0,8}(?:全部|都)(?:因|由於).{0,6}(?:缺陷|證據鏈)"),
     ("不得", "不可", "必須改成", "並非")),
    ("這是完成的樂高產品",
     re.compile(r"(?:是|為)一個?(?:已)?完成的樂高產品"),
     ("不", "並非", "還不是", "必須改成")),
)

#: Lines between these markers quote the forbidden wording in order to forbid
#: it, so the wire is lifted there and only there.
QUOTE_OPEN = "<!-- forbidden-claims-table:start -->"
QUOTE_CLOSE = "<!-- forbidden-claims-table:end -->"


#: A question is not an assertion. "connected 是否代表成品站得住？" is the
#: heading of the answer that says no, and a guard that fires on it teaches
#: the writer to stop asking.
_ASKING = ("是否", "嗎？", "？")


def overclaim_problems(text: str, *, where: str = "") -> list[str]:
    bad, quoting = [], False
    for n, line in enumerate(text.splitlines(), start=1):
        if QUOTE_OPEN in line:
            quoting = True
            continue
        if QUOTE_CLOSE in line:
            quoting = False
            continue
        if quoting:
            continue
        if any(mark in line for mark in _ASKING):
            continue
        for name, pattern, denials in FORBIDDEN:
            if pattern.search(line) and not any(d in line for d in denials):
                bad.append(f"{where}:{n}: asserts '{name}': {line.strip()}")
    return bad


# ---------------------------------------------------------------------------
# fixtures over the private tree
# ---------------------------------------------------------------------------

def _load_real(source: str):
    path = ROOT / source
    if not path.is_file():
        raise FileNotFoundError(source)
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if source.endswith(".json") else text


@pytest.fixture(scope="module")
def registry() -> dict:
    if not CLAIMS.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {CLAIMS.name} is not published")
    return json.loads(CLAIMS.read_text(encoding="utf-8"))


def _slide_text(pptx: Path) -> list[str]:
    """Every run of text in the deck, without a PowerPoint library."""
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    out = []
    with zipfile.ZipFile(pptx) as zf:
        names = sorted(n for n in zf.namelist()
                       if re.fullmatch(r"ppt/(slides|notesSlides)/[^/]+\.xml", n))
        for name in names:
            root = ET.fromstring(zf.read(name))
            # One run per line in this deck, so a newline between runs keeps
            # two neighbouring numbers from merging into a third one.
            out.append("\n".join(t.text or "" for t in root.iter(f"{ns}t")))
    return out


@pytest.fixture(scope="module")
def deliverables() -> dict[str, str]:
    if not PRESENTATION.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/ is not published")
    files = {}
    for rel in ("onepager/brickagain_onepager.md",
                "qa/qa_technical_appendix.md",
                "audit/numbers_and_sources.md",
                "audit/pre_share_checklist.md",
                "demo/demo_script.md",
                "README.md"):
        path = PRESENTATION / rel
        if not path.is_file():
            pytest.skip(f"{ARTIFACT_ONLY} presentation/{rel} is not built")
        files[rel] = path.read_text(encoding="utf-8")
    deck = PRESENTATION / "deck" / "brickagain_showcase.pptx"
    if not deck.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the deck is not built")
    for i, blob in enumerate(_slide_text(deck)):
        files[f"deck/part_{i:02d}"] = blob
    return files


# ---------------------------------------------------------------------------
# integration: three node ids, however many claims there are
# ---------------------------------------------------------------------------

def test_the_presentation_claims_match_private_evidence(registry):
    problems = registry_problems(registry, load=_load_real)
    assert problems == [], "\n".join(problems)


#: Figures that describe the deck, the world model or a version -- never a
#: result. Anything else has to be a declared claim.
STRUCTURAL = frozenset({
    "16:9", "0:30", "0:45", "1:15", "1:00", "1:30", "3:00", "15:00",
    "20×20×20", "3.2", "2026", "0.3",
    "95%",      # the interval level, which is a method constant, not a rate
})


def _without_quoted_block(text: str) -> str:
    """Drop the fenced table that quotes forbidden wording in order to forbid
    it. Scanning it would demand a source for a figure the document is
    telling the reader not to use."""
    out, quoting = [], False
    for line in text.splitlines():
        if QUOTE_OPEN in line:
            quoting = True
        elif QUOTE_CLOSE in line:
            quoting = False
        elif not quoting:
            out.append(line)
    return "\n".join(out)

_FIGURE = re.compile(
    r"(?<![\w.,/-])"
    r"(?:\d+/\d+"                    # 22/64
    r"|\d[\d,]*\.\d+\s*%"          # 58.0%
    r"|\d[\d,]*\.\d+"               # 0.4387
    r"|\d[\d,]{3,}"                  # 42,208
    r"|\d+\s*%)"                     # 100%
    # not the head of a dotted version: `1.0.2` is a pin, not a figure
    r"(?![\w,]|\.\d)")


def _declared(registry) -> set[str]:
    """Claim displays, plus the figures inside them.

    A display like `13.75%（22/160）` licenses both `13.75%` and `22/160`
    appearing on their own, because it is the same claim either way.
    """
    out: set[str] = set()
    for claim in registry["claims"]:
        shown = claim["display"]
        out.add(shown)
        out.add(shown.replace(MINUS, "-"))
        out.update(_FIGURE.findall(shown))
        out.update(_FIGURE.findall(shown.replace(MINUS, "-")))
        if claim["precision"] == "ratio":
            out.add(f"{claim['value'][0]}/{claim['value'][1]}")
    return out


def test_every_audience_facing_figure_is_a_declared_claim(registry,
                                                          deliverables):
    """A number on a slide that no claim covers has no source."""
    declared = _declared(registry) | STRUCTURAL
    unexplained = []
    for where, text in deliverables.items():
        if where == "audit/numbers_and_sources.md":
            continue        # the registry's own rendering, checked as itself
        for raw in _FIGURE.findall(_without_quoted_block(text)):
            shown = raw.strip()
            if shown in declared or shown.replace(" ", "") in declared:
                continue
            unexplained.append(f"{where}: {shown!r}")
    assert unexplained == [], (
        "every audience-facing figure must be a declared claim or a declared "
        "structural number: " + "; ".join(sorted(set(unexplained))))


def test_the_audit_table_is_the_registry_and_not_a_second_copy(registry,
                                                               deliverables):
    """The readable table has to carry every claim, with the same display."""
    table = deliverables["audit/numbers_and_sources.md"]
    missing = [c["claim_id"] for c in registry["claims"]
               if f"`{c['claim_id']}`" not in table]
    assert missing == [], f"claims absent from the audit table: {missing[:8]}"
    wrong = [c["claim_id"] for c in registry["claims"]
             if c["display"] not in table]
    assert wrong == [], f"the audit table shows a different display: {wrong[:8]}"


def test_the_pdf_was_exported_from_the_deck_on_disk():
    """A deck edited without a re-render leaves a PDF that is not it."""
    manifest = PRESENTATION / "deck" / "render_manifest.json"
    if not manifest.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the deck has not been rendered")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("pptx", "pdf"):
        path = ROOT / record[key]
        assert path.is_file(), record[key]
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record[f"{key}_sha256"], (
            f"{record[key]} changed after the render; rebuild the PDF")
    assert record["overflow_findings"] == [], record["overflow_findings"]
    assert record["main_slides"] == 11
    assert record["slides"] == record["main_slides"] + record[
        "appendix_slides"]


def test_no_deliverable_asserts_a_forbidden_claim(deliverables):
    problems = []
    for where, text in deliverables.items():
        problems.extend(overclaim_problems(text, where=where))
    assert problems == [], "\n".join(problems)


# ---------------------------------------------------------------------------
# behavioural: synthetic registries, one defect each.  These read no private
# evidence and must therefore pass in the public snapshot as well.
# ---------------------------------------------------------------------------

GOOD_DOC = {
    "rate": {"value": 0.4890625, "numerator": 313, "denominator": 640},
    "contrasts": {"C-B": {"delta": -0.05}},
    "rows": [{"termination": "eos"}, {"termination": "eos"},
             {"termination": "backtrack_exhausted"}],
    "n": 3,
    "digest": "05300f1b428570b3839220de7ed22b9832678d597d73a24f722ab4909cbe79b1",
    "voided": [{"generation": "gen08", "cells": 320, "why": "counters missing"}],
}


def fake_load(source):
    if source == "a.json":
        return GOOD_DOC
    if source == "b.json":
        return {"voided": [{"generation": "gen08"}]}
    if source == "notes.txt":
        return "the closure is 49 of the plan's 56 files\n"
    raise FileNotFoundError(source)


def claim(**over):
    base = {
        "claim_id": "demo.rate",
        "used_in": ["S6"],
        "display": "48.9%",
        "value": 0.4890625,
        "source": "a.json",
        "path": ".rate.value",
        "precision": "pct",
        "direction": "higher_is_better",
        "qualifier": "a rate over 640 draws",
    }
    base.update(over)
    return base


def problems(*claims):
    return registry_problems({"claims": list(claims)}, load=fake_load)


def test_a_faithful_registry_has_no_problems():
    assert problems(claim()) == []


def test_a_value_the_artefact_does_not_hold_is_refused():
    bad = problems(claim(value=0.49, display="49.0%"))
    assert bad and "artefact" in bad[0]


def test_a_display_that_rounds_further_than_it_shows_is_refused():
    """0.4890625 is 48.9%, and 48.0% is not a rounding of it."""
    assert problems(claim(display="48.0%")) != []


def test_either_side_of_a_tie_is_accepted_at_the_declared_precision():
    """0.0625 to three decimals is 0.062 or 0.063; neither is a defect."""
    tie = {"claim_id": "t", "used_in": ["A2"], "value": 0.0625,
           "source": "a.json", "path": ".tie", "precision": "float",
           "qualifier": "a tie"}
    doc = dict(GOOD_DOC, tie=0.0625)
    for shown in ("0.062", "0.063"):
        assert registry_problems({"claims": [dict(tie, display=shown)]},
                                 load=lambda s: doc) == [], shown
    assert registry_problems({"claims": [dict(tie, display="0.064")]},
                             load=lambda s: doc) != []


def test_a_missing_required_field_is_refused():
    thin = claim()
    del thin["qualifier"]
    bad = problems(thin)
    assert bad and "missing" in bad[0]


def test_a_duplicate_claim_id_is_refused():
    bad = problems(claim(), claim(display="48.9%"))
    assert any("duplicate" in b for b in bad)


def test_a_claim_with_neither_a_path_nor_a_rule_is_refused():
    thin = claim()
    del thin["path"]
    assert any("exactly one" in b for b in problems(thin))


def test_a_claim_with_both_a_path_and_a_rule_is_refused():
    both = claim(rule={"kind": "length", "path": ".rows"})
    assert any("exactly one" in b for b in problems(both))


def test_a_path_that_does_not_resolve_is_refused():
    assert any("does not resolve" in b
               for b in problems(claim(path=".rate.absent")))


def test_a_path_that_cannot_be_parsed_is_refused():
    assert any("cannot parse" in b or "must be a string"
               in b for b in problems(claim(path="rate.value")))


def test_a_source_that_is_not_there_is_refused():
    assert any("FileNotFoundError" in b
               for b in problems(claim(source="missing.json")))


def test_an_unknown_precision_is_refused():
    assert any("unknown precision" in b
               for b in problems(claim(precision="approximately")))


def test_an_unknown_direction_is_refused():
    assert any("unknown direction" in b
               for b in problems(claim(direction="bigger")))


def test_a_claim_that_names_no_slide_is_refused():
    assert any("names no place" in b for b in problems(claim(used_in=[])))


def test_a_hyphenated_key_needs_the_bracket_form():
    """`.contrasts.C-B` is a subtraction in jq, so it may not be written."""
    ok = claim(claim_id="c", path='.contrasts["C-B"].delta', value=-0.05,
               display=MINUS + "5.0pp", precision="pp")
    assert problems(ok) == []
    assert parse_path('.contrasts["C-B"].delta') == ["contrasts", "C-B",
                                                     "delta"]
    with pytest.raises(PathError):
        parse_path(".contrasts.C-B.delta")


def test_a_ratio_display_must_be_the_two_numbers_it_claims():
    ratio = claim(claim_id="r", value=[313, 640], display="313/640",
                  precision="ratio", path=None,
                  rule={"kind": "fraction",
                        "numerator_path": ".rate.numerator",
                        "denominator_path": ".rate.denominator"})
    del ratio["path"]
    assert problems(ratio) == []
    assert problems(dict(ratio, display="313/641")) != []
    assert problems(dict(ratio, value=[312, 640], display="312/640")) != []


def test_counting_by_a_field_is_a_count_and_not_a_guess():
    counted = claim(claim_id="k", value=[2, 3], display="2/3",
                    precision="ratio",
                    rule={"kind": "count_where", "path": ".rows",
                          "field": "termination", "equals": "eos",
                          "denominator_path": ".n"})
    del counted["path"]
    assert problems(counted) == []
    assert problems(dict(counted, value=[3, 3], display="3/3")) != []


def test_a_short_digest_must_be_sixteen_characters_of_the_real_one():
    good = claim(claim_id="d", value="05300f1b428570b3",
                 display="05300f1b428570b3" + ELLIPSIS, precision="digest16",
                 path=".digest")
    assert problems(good) == []
    seventeen = "05300f1b428570b38"
    assert any("16 hex" in b for b in problems(
        dict(good, value=seventeen, display=seventeen + ELLIPSIS)))
    assert problems(dict(good, value="0000000000000000",
                         display="0000000000000000" + ELLIPSIS)) != []


def test_a_selected_field_must_come_from_the_row_it_names():
    picked = claim(claim_id="s", value=320, display="320", precision="int",
                   rule={"kind": "select_field", "path": ".voided",
                         "select": {"field": "generation", "equals": "gen08"},
                         "field": "cells"})
    del picked["path"]
    assert problems(picked) == []
    missing_row = dict(picked, source="b.json")
    assert any("no row where" in b or "KeyError" in b
               for b in problems(dict(picked,
                                      rule={**picked["rule"],
                                            "select": {"field": "generation",
                                                       "equals": "gen99"}})))
    assert problems(missing_row) != []


def test_a_quoted_phrase_absent_from_the_artefact_is_refused():
    quoted = claim(claim_id="q", value="49 of the plan's 56 files",
                   display="49", precision="text", source="notes.txt",
                   rule={"kind": "contains_text"})
    del quoted["path"]
    del quoted["direction"]
    assert problems(quoted) == []
    assert problems(dict(quoted, value="9 of the plan's 56 files")) != []


def test_an_empty_or_malformed_registry_is_refused():
    assert registry_problems({"claims": []}, load=fake_load) != []
    assert registry_problems({}, load=fake_load) != []
    assert registry_problems([], load=fake_load) != []


# --- the over-claim tripwire, in both directions ---------------------------

def test_the_tripwire_catches_the_assertion_it_is_written_for():
    assert overclaim_problems("六臂顯示 2247 → 95，訓練確實有效。") != []
    assert overclaim_problems("硬約束把模型準確率提升到 100%。") != []
    assert overclaim_problems("影像辨識準確率 58.0%。") != []
    assert overclaim_problems("漏掉的積木使用者不會看到。") != []
    assert overclaim_problems("connected 代表結構物理穩定。") != []


def test_the_tripwire_treats_a_question_as_a_question():
    assert overclaim_problems("connected 是否代表成品站得住？") == []
    assert overclaim_problems("connected 代表結構物理穩定。") != []


def test_the_tripwire_does_not_fire_on_the_sentence_that_denies_it():
    """The failure shape this guard exists to avoid: a disclaimer that trips
    the wire written to protect it."""
    assert overclaim_problems(
        "2247 → 95 同時改了 prompt 與權重，只能描述組合差異。") == []
    assert overclaim_problems(
        "這不是硬約束把模型準確率提升到 100%，而是 gate 的性質。") == []
    assert overclaim_problems(
        "不可說「影像辨識準確率 58%」；必須改成已配對積木上的 top-1。") == []
    assert overclaim_problems(
        "不得寫使用者不會看到漏件；系統只是不為它建立候選框。") == []
    assert overclaim_problems(
        "connected 不代表物理穩定，它只是相鄰層 footprint 交集。") == []


def test_the_quoted_block_lifts_the_wire_only_between_its_markers():
    doc = (f"{QUOTE_OPEN}\n"
           "| 硬約束把模型準確率提升到 100% | ... |\n"
           f"{QUOTE_CLOSE}\n"
           "硬約束把模型準確率提升到 100%\n")
    found = overclaim_problems(doc, where="d")
    assert len(found) == 1 and found[0].startswith("d:4:")


# ---------------------------------------------------------------------------
# slide geometry, injected
#
# The round 71 check reported `overflow_findings: []` on a deck that had a
# rule through the second line of a table cell and a bullet against the bottom
# of its panel. A checker that only ever ran on the good artefact was a
# checker nobody had tested, so each shape it must catch is injected here.
# ---------------------------------------------------------------------------

def line_at(x, y, w, size=15.0, label="line"):
    """A text entry as the renderer builds it: (label, ink, line box)."""
    step = size * 1.35 / 72.0
    return (label, (x, y + step * 0.08, x + w, y + step * 0.96),
            (x, y, x + w, y + step))


def test_a_rule_through_the_second_line_of_a_cell_is_caught():
    """The exact round 71 defect: a two-line cell, and a separator placed by
    a constant row step rather than by the cell's height."""
    first = line_at(1.0, 5.00, 2.0, label="backtrack 14/64")
    second = line_at(1.0, 5.28, 2.0, label="eos 41/64；max 9/64")
    through = [(0.85, 5.30, 11.6, 5.312)]
    found = layout_problems(12, [first, second], through, [], [])
    assert any("eos 41/64" in f and "gutter" in f for f in found), found


def test_a_rule_that_grazes_the_line_box_is_still_caught():
    """0.017in of clearance passed in round 71 and should not."""
    text = line_at(1.0, 5.00, 2.0, label="cell")
    grazing = [(0.85, 5.00 + 15 * 1.35 / 72 + 0.017, 11.6, 5.6)]
    assert layout_problems(1, [text], grazing, [], []) != []


def test_a_rule_in_the_gutter_is_not_a_finding():
    text = line_at(1.0, 5.00, 2.0, label="cell")
    gutter = [(0.85, 5.00 + 15 * 1.35 / 72 + 0.12, 11.6, 5.6)]
    assert layout_problems(1, [text], gutter, [], []) == []


def test_text_that_leaves_its_panel_is_caught():
    panel = [(0.85, 5.00, 12.48, 5.60)]
    inside = line_at(1.0, 5.10, 2.0, label="inside")
    below = line_at(1.0, 5.50, 2.0, label="third bullet")
    assert layout_problems(1, [inside], [], panel, []) == []
    found = layout_problems(21, [below], [], panel, [])
    assert any("third bullet" in f and "panel" in f for f in found), found


def test_a_panel_that_leaves_the_page_is_caught():
    """A box can run off the bottom while every line inside it stays put."""
    inside = line_at(1.0, 7.00, 2.0, label="last bullet")
    panel = [(0.85, 6.90, 12.48, 7.80)]
    found = layout_problems(22, [inside], [], panel, [])
    assert any("runs off the page" in f for f in found), found
    ok = [(0.85, 6.60, 12.48, 7.30)]
    assert layout_problems(22, [line_at(1.0, 6.70, 2.0)], [], ok, []) == []


def test_text_that_leaves_the_page_is_caught():
    off_bottom = line_at(1.0, 7.40, 2.0, label="footer")
    off_right = line_at(12.0, 3.0, 2.0, label="wide")
    assert any("vertically" in f
               for f in layout_problems(1, [off_bottom], [], [], []))
    assert any("horizontally" in f
               for f in layout_problems(1, [off_right], [], [], []))


def test_two_blocks_stacked_against_each_other_are_not_a_collision():
    """Adjacent blocks share leading; that is typography, not a defect."""
    a = line_at(1.0, 2.00, 3.0, size=60, label="BrickAgain")
    b = line_at(1.0, 2.00 + 60 * 1.35 / 72, 3.0, size=30, label="subtitle")
    assert layout_problems(1, [a, b], [], [], []) == []


def test_overlapping_glyphs_are_a_collision():
    a = line_at(1.0, 2.00, 3.0, label="one")
    b = line_at(1.0, 2.10, 3.0, label="two")
    assert any("overlaps" in f for f in layout_problems(1, [a, b], [], [], []))


def test_text_on_top_of_a_picture_is_caught():
    a = line_at(1.0, 2.00, 3.0, label="caption")
    assert any("picture" in f
               for f in layout_problems(1, [a], [], [], [(0.5, 1.8, 5.0, 4.0)]))


# ---------------------------------------------------------------------------
# the README has to stay runnable
# ---------------------------------------------------------------------------

def _fences(text: str) -> list[list[str]]:
    out, current = [], None
    for line in text.splitlines():
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                out.append(current)
                current = None
            continue
        if current is not None:
            current.append(line)
    return out


def test_the_readme_survives_the_reflow_that_folds_its_prose(deliverables):
    """Round 71's reflow had no fence state and folded the file tree and the
    six rebuild commands into one line each -- six commands joined by spaces
    is not a command.

    The published order also has to be a *working* order: round 72 ran
    build_text.py before the deck was rendered, so the appendix length in the
    README came from the manifest of the round before it. Both the order and
    the number it produces are checked here rather than in a new test, so the
    node id stays one id."""
    readme = deliverables["README.md"]
    blocks = _fences(readme)
    assert blocks, "the README has no fenced blocks left"

    tree = [b for b in blocks if any("BRICKAGAIN_SHOWCASE_MASTER.md" in l
                                     for l in b)]
    assert tree, "the file tree block is gone"
    assert len(tree[0]) >= 10, (
        f"the file tree collapsed to {len(tree[0])} line(s): {tree[0][:1]}")

    commands = [l.strip() for b in blocks for l in b if l.strip()]
    rebuild = [c for c in commands if "presentation/tools/" in c]
    assert len(rebuild) == 6, f"expected six rebuild commands, got {rebuild}"
    for command in rebuild:
        parts = command.split()
        assert len(parts) == 2, (
            f"a rebuild line carries more than one command: {command!r}")
        script = ROOT / parts[1]
        assert script.is_file(), f"{parts[1]} does not exist"
        compile(script.read_text(encoding="utf-8"), str(script), "exec")

    order = [c.split()[1] for c in rebuild]
    assert order.index("presentation/tools/make_claims.py") == 0
    assert (order.index("presentation/tools/build_deck.py")
            < order.index("presentation/tools/render_deck.py")), (
        "the deck has to be built before it can be rendered")
    # Round 72 put build_text.py third, so the appendix length it published
    # came from the previous round's manifest.
    assert (order.index("presentation/tools/render_deck.py")
            < order.index("presentation/tools/build_text.py")), (
        "build_text.py reads render_manifest.json, so the published order "
        "has to run it after the step that writes that manifest")

    manifest = json.loads((PRESENTATION / "deck" / "render_manifest.json")
                          .read_text(encoding="utf-8"))
    stated = re.findall(r"附錄\s*(\d+)\s*張", readme)
    assert stated == [str(manifest["appendix_slides"])], (
        f"the README says 附錄 {stated} while the manifest says "
        f"{manifest['appendix_slides']}")


def test_every_generated_text_file_ends_in_exactly_one_newline(deliverables):
    """`git diff --check` calls a blank line at EOF a defect, and in round 71
    every generated file had one."""
    bad = []
    for rel in ("README.md", "onepager/brickagain_onepager.md",
                "qa/qa_technical_appendix.md", "demo/demo_script.md",
                "audit/pre_share_checklist.md",
                "audit/numbers_and_sources.md"):
        raw = (PRESENTATION / rel).read_bytes()
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            bad.append(f"{rel}: ends {raw[-3:]!r}")
    assert bad == [], bad


# ---------------------------------------------------------------------------
# M5-0: the two packages the teacher-review build imports, and the four it
# may not move
#
# v1 was built on 2026-09-10 and both were gone two days later, because
# nothing on disk recorded that the build needed them. Pinning them is the
# whole fix; a test that only checked they import would pass on this machine
# today and fail on the next one for the same reason as before.


PROTECTED = ("torch", "transformers", "numpy", "pillow")
BUILD_REQUIREMENTS = PRESENTATION / "requirements-presentation.txt"
FROZEN_V1_REQUIREMENTS = (PRESENTATION / "teacher_review" / "environment"
                          / "requirements.txt")


def pinned_versions(text: str) -> dict[str, str]:
    """name -> version for every ``==`` pin; refuses anything looser.

    A floating or minimum-bound line is refused rather than ignored: the
    reason these are pinned at all is that a different matplotlib moves a
    rendered line, and a ``>=`` would let that happen while still looking
    like a pin.
    """
    versions = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise AssertionError(f"unpinned requirement: {line!r}")
        name, version = line.split("==", 1)
        versions[name.strip().lower().replace("_", "-")] = version.strip()
    return versions


def test_the_pinned_requirements_list_pypdf_and_reportlab():
    if not BUILD_REQUIREMENTS.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/ is not published")
    import pypdf
    import reportlab
    pins = pinned_versions(BUILD_REQUIREMENTS.read_text())
    assert pins.get("pypdf") == pypdf.__version__
    assert pins.get("reportlab") == reportlab.Version
    # The delivered v1 stays as it shipped: its own checksum file still
    # verifies, and writing into it would make the directory disagree with
    # the archive that was handed over.
    if FROZEN_V1_REQUIREMENTS.is_file():
        frozen = pinned_versions(FROZEN_V1_REQUIREMENTS.read_text())
        assert "pypdf" not in frozen and "reportlab" not in frozen


def test_an_unpinned_version_in_requirements_is_refused():
    assert pinned_versions("pypdf==6.18.1\n# comment\n\n") == {"pypdf": "6.18.1"}
    for loose in ("pypdf", "pypdf>=6.0", "pypdf~=6.18", "reportlab>4"):
        with pytest.raises(AssertionError, match="unpinned requirement"):
            pinned_versions(f"matplotlib==3.11.1\n{loose}\n")


def test_installing_did_not_move_torch_transformers_numpy_or_pillow():
    """v1's frozen list is the baseline, which is why it is not edited."""
    if not FROZEN_V1_REQUIREMENTS.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} presentation/ is not published")
    import importlib.metadata as metadata
    frozen = pinned_versions(FROZEN_V1_REQUIREMENTS.read_text())
    for name in PROTECTED:
        assert name in frozen, name
        assert metadata.version(name) == frozen[name], (
            f"{name} moved from {frozen[name]} to {metadata.version(name)}; "
            "the frozen evidence was produced against the recorded version")
