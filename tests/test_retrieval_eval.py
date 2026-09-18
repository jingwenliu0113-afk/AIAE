"""The old core's retrieval evaluation, checked the way it checks itself.

Three things are held here, and each one has a companion that proves the
guard fails when the defect is present -- a check nobody has watched fail is
a check nobody knows is wired up.

* **the rule is a rule.** ``truncate`` and ``stated_stock`` derive the queries
  and the stock mechanically. If either ever became a hand-written list the
  measurement would stop being reproducible, so both are pinned by example.
* **the frozen test is frozen.** ``load_test`` recomputes the digest and
  refuses an edited file, so numbers cannot quietly be measured against a
  test that moved after it was written down.
* **the number is not renamed.** The report carries the sentence saying it is
  not Recall@K on user queries, and the term does not appear as a claim. This
  catalogue has one caption per object, so that number has no ground truth --
  the risk is not that it is computed wrongly, it is that what *was* computed
  gets called by the name of the thing that was not.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/72_retrieval_eval.py"
REPORTS = ROOT / "data/reports/72_retrieval"

ARTIFACT_ONLY = "artifact-only:"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("retrieval_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The rule is a rule
# ---------------------------------------------------------------------------

def test_the_query_is_the_opening_of_the_caption_cut_by_word_count(module):
    caption = ("The guitar includes a long, narrow neck with a squared "
               "headstock. The circular lower section is wide.")
    assert module.truncate(caption, 4) == "The guitar includes a"
    assert module.truncate(caption, 8) == \
        "The guitar includes a long, narrow neck with"
    # ``None`` keeps the first sentence and stops there, so a two-sentence
    # caption never leaks its second half into the query.
    assert module.truncate(caption, None) == (
        "The guitar includes a long, narrow neck with a squared headstock.")


def test_a_caption_shorter_than_the_cut_is_not_padded(module):
    """The cut is a maximum, not a length: a short caption stays itself."""
    assert module.truncate("A small car.", 8) == "A small car."


def test_the_three_stocks_are_derived_not_typed(module):
    class Item:
        def __init__(self, required):
            self.required = required

    class Catalog:
        items = [Item({"1x1": 1, "2x4": 9}), Item({"1x1": 3}),
                 Item({"1x1": 5, "2x4": 1})]

    catalog = Catalog()
    own = module.stated_stock("own", catalog.items[0], catalog)
    assert own == {"1x1": 1, "2x4": 9}

    # Median over every work, absences counted as zero -- a part most works do
    # not use should pull the median down, not be silently skipped.
    assert module.stated_stock("median", catalog.items[0], catalog) == \
        {"1x1": 3, "2x4": 1}
    assert module.stated_stock("complete", catalog.items[0], catalog) == \
        {"1x1": 5, "2x4": 9}


def test_an_unknown_stock_is_refused_rather_than_defaulted(module):
    class Catalog:
        items = []

    with pytest.raises(module.EvalRefused):
        module.stated_stock("whatever", None, Catalog())


# ---------------------------------------------------------------------------
# The frozen test is frozen
# ---------------------------------------------------------------------------

STOCKS = ("own", "median", "complete")


@pytest.mark.parametrize("stock", STOCKS)
def test_each_frozen_test_recomputes_its_own_digest(module, stock):
    path = REPORTS / f"frozen_test_{stock}.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {path.name} is not published")
    body = module.load_test(path)          # raises if it does not recompute
    assert body["stock"] == stock
    assert body["n_rows"] == len(body["rows"])


@pytest.mark.parametrize("damage,why", [
    ("edit", "a query changed after freezing"),
    ("kind", "the file is some other kind of document"),
    ("digest", "the recorded digest was overwritten"),
    ("absent", "there is no frozen test at all"),
])
def test_a_frozen_test_that_moved_is_refused(module, tmp_path, damage, why):
    """Each way the file can stop being the thing that was frozen."""
    source = REPORTS / "frozen_test_own.json"
    if not source.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {source.name} is not published")
    target = tmp_path / "frozen_test_own.json"
    if damage == "absent":
        with pytest.raises(module.EvalRefused):
            module.load_test(target)
        return

    body = json.loads(source.read_text())
    if damage == "edit":
        body["rows"][0]["queries"]["4"] = "a different query"
    elif damage == "kind":
        body["kind"] = "brickagain.something_else"
    else:
        body["test_digest"] = "0" * 64
    target.write_text(json.dumps(body))
    with pytest.raises(module.EvalRefused):
        module.load_test(target)


def test_the_unedited_copy_is_accepted_so_the_refusals_mean_something(
        module, tmp_path):
    """The other half of the parametrised test above.

    A loader that refused everything would pass all four cases and check
    nothing, so the untouched file has to be accepted here.
    """
    source = REPORTS / "frozen_test_own.json"
    if not source.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {source.name} is not published")
    target = tmp_path / "copy.json"
    target.write_bytes(source.read_bytes())
    assert module.load_test(target)["stock"] == "own"


# ---------------------------------------------------------------------------
# The number is not renamed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stock", STOCKS)
def test_the_report_says_what_it_is_not(stock):
    path = REPORTS / f"report_{stock}.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {path.name} is not published")
    report = json.loads(path.read_text())
    said = report["what_this_is_not"]
    assert "not Recall@K on user queries" in said
    assert "one caption per object" in said
    # The metric keys carry the qualifier too, so a table copied out of the
    # JSON cannot lose the caveat on the way.
    for block in report["results"].values():
        assert "truncated_recall_at_k_retrieved" in block
        assert "truncated_recall_at_k_ranked" in block
        assert not any(key == "recall_at_k" for key in block)


@pytest.mark.parametrize("stock", STOCKS)
def test_the_reported_buildable_heading_names_the_stock_it_used(stock):
    """The heading is generated from the stock, not written once.

    It was written once, and said "以該作品自身所需零件為庫存" over the
    median and complete tables as well. A heading that names the wrong
    condition is worse than no heading: it is a wrong claim in the place a
    reader trusts most.
    """
    path = REPORTS / f"report_{stock}.md"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {path.name} is not published")
    text = path.read_text()
    assert f"庫存 `{stock}`" in text
    others = [s for s in STOCKS if s != stock]
    for other in others:
        assert f"庫存 `{other}`" not in text


def test_the_semantic_recall_does_not_move_with_the_stock():
    """A stock cannot change what the embedding retrieved.

    ``retrieved`` is the embedding's own order, computed before any inventory
    is consulted, so the three reports must agree on it exactly. They are
    produced by three separate runs, so this is the check that the stock was
    applied where it belongs -- to buildability and the interface order --
    and nowhere else.
    """
    seen = {}
    for stock in STOCKS:
        path = REPORTS / f"report_{stock}.json"
        if not path.is_file():
            pytest.skip(f"{ARTIFACT_ONLY} report_{stock}.json is not published")
        report = json.loads(path.read_text())
        seen[stock] = {cut: block["truncated_recall_at_k_retrieved"]
                       for cut, block in report["results"].items()}
    first = seen[STOCKS[0]]
    for stock in STOCKS[1:]:
        assert seen[stock] == first, (
            f"{stock} reports a different semantic recall from {STOCKS[0]}; "
            "the stock has leaked into the embedding's own ranking")


def test_the_deepest_k_is_the_same_in_both_orders():
    """Stated as the identity it is, so nobody reads it as a result.

    ``ranked`` is a permutation of ``retrieved``, so recall at the deepest K
    is necessarily equal in both. If a future change made them differ, the
    two lists would no longer be the same candidates and every comparison
    between the two orders would have quietly changed meaning.
    """
    for stock in STOCKS:
        path = REPORTS / f"report_{stock}.json"
        if not path.is_file():
            pytest.skip(f"{ARTIFACT_ONLY} report_{stock}.json is not published")
        report = json.loads(path.read_text())
        deepest = str(max(int(k) for k in
                          next(iter(report["results"].values()))
                          ["truncated_recall_at_k_retrieved"]))
        for block in report["results"].values():
            assert (block["truncated_recall_at_k_retrieved"][deepest]
                    == block["truncated_recall_at_k_ranked"][deepest])
