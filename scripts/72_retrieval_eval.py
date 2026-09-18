#!/usr/bin/env python
"""Measure the old core's retrieval, and name the number it cannot produce.

    --freeze DIR    derive the queries from the catalogue and write the test
    --measure DIR   verify that test, run it, and write the report

**The number this cannot produce.** Recall@K on user queries needs a held-out
description of a work whose *other* description is what was indexed. This
catalogue does not have one: of 1,082 train objects, 1,077 carry a single
caption and 5 carry two. Querying with the indexed caption itself scores near
1.0 and measures nothing, so that number is not reported here and the milestone
does not claim it. ``scripts/69`` reports R@1 on the BrickNet track, where the
data does carry several captions per model -- those are different data, a
different index and a different metric, and the two must never be put in one
table.

**What is reported instead**, both objective and both derived mechanically:

* **truncated-description recall@K.** The query is the opening of the work's
  own caption, cut to a fixed word count. It asks whether a partial
  description still finds the work it came from. This is *not* a user query;
  it is a degraded version of the indexed text, and the report says so.
* **buildable@K.** How many of the top K can actually be built from the stated
  stock -- inventory covered, touching the ground, stud-connected. This is the
  promise the feature makes to somebody holding a box of bricks, and
  ``search()`` returns unbuildable candidates too, so it is a real fraction
  rather than a constant.

**Frozen before measured.** ``--freeze`` writes the queries, the cut lengths,
the stated stock and the K values, and hashes them; ``--measure`` refuses a
test whose digest does not recompute. The three cut lengths are all frozen and
all reported: a single length chosen after seeing the numbers would be a length
chosen *because* of the numbers.

Both ranked orders are reported, because they answer different questions.
``retrieved`` is the embedding's own order. ``ranked`` is what the interface
shows, which puts buildable candidates first -- so a work the embedding placed
eleventh can still be what the user sees first, and a recall figure over
``retrieved`` alone would not describe the product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.delivery.pipeline import DeliveryError, load_train_catalog  # noqa: E402
from src.retrieval import embed as embed_module  # noqa: E402
from src.retrieval import index as index_module  # noqa: E402
from src.retrieval.nlp import extract  # noqa: E402
from src.retrieval.search import SearchError, search  # noqa: E402

DEFAULT_CATALOG = ROOT / "data/processed/counterfactual_train.jsonl"
DEFAULT_INDEX = ROOT / "runs/retrieval/index"
DEFAULT_OUT = ROOT / "data/reports/72_retrieval"

TEST_KIND = "brickagain.retrieval_frozen_test"
REPORT_KIND = "brickagain.retrieval_report"

#: How many words of the caption's first sentence the query keeps. ``None``
#: keeps the whole first sentence. Frozen as a set, not chosen as a value.
CUTS: tuple[int | None, ...] = (4, 8, None)

#: Reported at every K, so a reader can see the shape rather than one point.
KS: tuple[int, ...] = (1, 5, 10)

#: The deepest K, which is also how many candidates each search asks for.
TOP_N = max(KS)

EXIT_OK, EXIT_REFUSED = 0, 2

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class EvalRefused(RuntimeError):
    """Raised rather than measuring against something unverified."""


def digest_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def truncate(caption: str, words: int | None) -> str:
    """The query, cut by a rule rather than by hand.

    First sentence, then the first ``words`` words of it. Splitting on the
    sentence first matters: several captions put the distinguishing noun in
    the opening clause and the geometry afterwards, so cutting on raw word
    count alone would sometimes keep a whole sentence and sometimes half of
    one, and the cut length would no longer mean the same thing across rows.
    """
    first = _SENTENCE_END.split(caption.strip(), maxsplit=1)[0].strip()
    if words is None:
        return first
    return " ".join(first.split()[:words])


#: The stated stocks, all frozen together and all reported. One stock would be
#: a stock chosen, and ``buildable@K`` reads very differently depending on it:
#:
#: * ``own`` -- exactly what this work needs. The tightest stock that still
#:   admits the work itself, so the figure is a lower bound. It also makes the
#:   target buildable by construction, which is why the buildable-first order
#:   flatters recall under this stock and the report says so rather than
#:   claiming the interface improved retrieval.
#: * ``median`` -- the element-wise median requirement across the catalogue.
#:   Mechanical, and notably *smaller* than the median work needs (35 bricks
#:   against 56), because parts are used unevenly; it is reported for what it
#:   is rather than as "a typical player".
#: * ``complete`` -- the element-wise maximum, so inventory is never the
#:   binding constraint. What is left is the structural half of buildability --
#:   touching the ground and stud connectivity -- separated from the stock.
STOCKS = ("own", "median", "complete")


def stated_stock(kind: str, item, catalog) -> dict[str, int]:
    import statistics

    if kind == "own":
        return dict(item.required)
    parts = sorted({p for i in catalog.items for p in i.required})
    counts = {p: [i.required.get(p, 0) for i in catalog.items] for p in parts}
    if kind == "median":
        return {p: int(statistics.median(v)) for p, v in counts.items()}
    if kind == "complete":
        return {p: max(v) for p, v in counts.items()}
    raise EvalRefused(f"unknown stock {kind!r}")


def build_test(catalog_path: Path, stock: str) -> dict:
    catalog = load_train_catalog(str(catalog_path))
    if stock not in STOCKS:
        raise EvalRefused(f"stock must be one of {STOCKS}, not {stock!r}")
    rows = []
    for item in catalog.items:
        queries = {str(cut): truncate(item.caption, cut) for cut in CUTS}
        rows.append({
            "catalog_id": item.catalog_id,
            "n_bricks": item.n_bricks,
            "inventory": stated_stock(stock, item, catalog),
            "queries": queries,
        })
    body = {
        "kind": TEST_KIND,
        "catalog": str(catalog_path.relative_to(ROOT)),
        "catalog_sha256": catalog.sha256,
        "split_manifest_sha256": catalog.split_manifest_sha256,
        "cuts": [("full_first_sentence" if c is None else c) for c in CUTS],
        "ks": list(KS),
        "top_n": TOP_N,
        "stock": stock,
        "not_a_user_query": (
            "every query is a truncation of the indexed caption itself. This "
            "is not Recall@K on user queries and must not be reported as one; "
            "this catalogue carries one caption per object, so that number "
            "has no ground truth here."),
        "rows": rows,
        "n_rows": len(rows),
    }
    body["test_digest"] = digest_obj({k: v for k, v in body.items()})
    return body


def load_test(path: Path) -> dict:
    if not path.is_file():
        raise EvalRefused(f"{path} does not exist; run --freeze first")
    body = json.loads(path.read_text())
    if body.get("kind") != TEST_KIND:
        raise EvalRefused(f"{path} is not a {TEST_KIND}")
    stated = body.get("test_digest")
    recomputed = digest_obj({k: v for k, v in body.items()
                             if k != "test_digest"})
    if stated != recomputed:
        raise EvalRefused(
            f"{path} does not recompute its own digest "
            f"({stated} recorded, {recomputed} here); it has been edited "
            "since it was frozen and nothing may be measured against it")
    return body


def measure(test: dict, index_dir: Path, device: str) -> dict:
    catalog = load_train_catalog(str(ROOT / test["catalog"]))
    if catalog.sha256 != test["catalog_sha256"]:
        raise EvalRefused(
            "the catalogue on disk is not the one the test was frozen "
            f"against ({test['catalog_sha256']} frozen, {catalog.sha256} "
            "here)")
    embedder = embed_module.load(device=device)
    loaded = index_module.load(
        str(index_dir),
        expected_identity_digest=embedder.identity_digest(),
        expected_catalog_sha256=catalog.sha256,
        expected_split_manifest_sha256=catalog.split_manifest_sha256)

    cuts = [str(c) for c in CUTS]
    tally = {cut: {"n": 0,
                   "hit_retrieved": {k: 0 for k in KS},
                   "hit_ranked": {k: 0 for k in KS},
                   "buildable": {k: 0 for k in KS},
                   "returned": {k: 0 for k in KS}}
             for cut in cuts}

    for row in test["rows"]:
        for cut in cuts:
            query = row["queries"][cut]
            if not query:
                continue
            conditions = extract(query)
            try:
                result = search(loaded, catalog, embedder, conditions,
                                row["inventory"], top_n=TOP_N)
            except SearchError as exc:            # a refusal is a result
                raise EvalRefused(
                    f"search refused row {row['catalog_id']} at cut {cut}: "
                    f"{exc}") from None
            bucket = tally[cut]
            bucket["n"] += 1
            by_score = [c.item.catalog_id for c in result.retrieved]
            by_rank = [c.item.catalog_id for c in result.ranked]
            for k in KS:
                if row["catalog_id"] in by_score[:k]:
                    bucket["hit_retrieved"][k] += 1
                if row["catalog_id"] in by_rank[:k]:
                    bucket["hit_ranked"][k] += 1
                top = result.ranked[:k]
                bucket["buildable"][k] += sum(1 for c in top if c.buildable)
                bucket["returned"][k] += len(top)

    def rate(hit, n):
        return round(hit / n, 4) if n else None

    results = {}
    for cut in cuts:
        b = tally[cut]
        results[cut] = {
            "n_queries": b["n"],
            "truncated_recall_at_k_retrieved": {
                str(k): rate(b["hit_retrieved"][k], b["n"]) for k in KS},
            "truncated_recall_at_k_ranked": {
                str(k): rate(b["hit_ranked"][k], b["n"]) for k in KS},
            "buildable_at_k": {
                str(k): rate(b["buildable"][k], b["returned"][k]) for k in KS},
            "candidates_returned_at_k": {
                str(k): b["returned"][k] for k in KS},
        }

    body = {
        "kind": REPORT_KIND,
        "test_digest": test["test_digest"],
        # The manifest file's own bytes, not a field inside it: this is what
        # says "the index on disk when this was measured", and an index
        # rebuilt with the same catalogue would still change these bytes.
        "index_manifest_sha256": hashlib.sha256(
            (index_dir / "index_manifest.json").read_bytes()).hexdigest(),
        "index_build_device": loaded.build_device,
        "identity_digest": embedder.identity_digest(),
        "device": device,
        "index_documents": len(loaded.documents),
        "stock": test["stock"],
        "what_this_is_not": test["not_a_user_query"],
        "results": results,
    }
    body["report_digest"] = digest_obj({k: v for k, v in body.items()})
    return body


def render(report: dict) -> str:
    lines = [
        f"# 舊核心檢索評估（8 磚軌）— 庫存 `{report['stock']}`",
        "",
        "**這不是使用者查詢上的 Recall@K。**",
        report["what_this_is_not"],
        "",
        f"- 索引文件數：{report['index_documents']}",
        f"- `test_digest`：`{report['test_digest'][:16]}…`",
        f"- `report_digest`：`{report['report_digest'][:16]}…`",
        f"- 裝置：`{report['device']}`",
        "",
        "## 截斷描述 recall@K",
        "",
        "| 保留字數 | 查詢數 | R@1（語意序） | R@5 | R@10 | R@1（介面序） | R@5 | R@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cut, block in report["results"].items():
        label = "整句" if cut == "None" else f"{cut} 字"
        s = block["truncated_recall_at_k_retrieved"]
        r = block["truncated_recall_at_k_ranked"]
        lines.append(
            f"| {label} | {block['n_queries']} | "
            + " | ".join(f"{s[str(k)]:.1%}" for k in KS) + " | "
            + " | ".join(f"{r[str(k)]:.1%}" for k in KS) + " |")
    stock_says = {"own": "以該作品自身所需零件為庫存（最緊、目標必然可建）",
                  "median": "以全目錄逐項中位數為庫存（35 磚，小於中位數作品的 56 磚）",
                  "complete": "以全目錄逐項最大值為庫存（庫存不再是限制）"}
    lines += ["", f"## buildable@K — {stock_says[report['stock']]}", "",
              "| 保留字數 | " + " | ".join(f"buildable@{k}" for k in KS) + " |",
              "|---|" + "---|" * len(KS)]
    for cut, block in report["results"].items():
        label = "整句" if cut == "None" else f"{cut} 字"
        b = block["buildable_at_k"]
        lines.append(f"| {label} | "
                     + " | ".join(f"{b[str(k)]:.1%}" for k in KS) + " |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freeze", action="store_true",
                   help="derive the queries and write the frozen test")
    p.add_argument("--measure", action="store_true",
                   help="verify the frozen test, run it, write the report")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--device", default="cpu")
    p.add_argument("--stock", default="own", choices=STOCKS,
                   help="which stated stock to freeze or measure")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze == args.measure:
        print("choose exactly one of --freeze, --measure", file=sys.stderr)
        return EXIT_REFUSED
    out = Path(args.out)
    test_path = out / f"frozen_test_{args.stock}.json"
    try:
        if args.freeze:
            if test_path.exists():
                raise EvalRefused(
                    f"{test_path} already exists. A frozen test is written "
                    "once; delete it deliberately if the catalogue changed.")
            body = build_test(Path(args.catalog), args.stock)
            out.mkdir(parents=True, exist_ok=True)
            test_path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
            print(f"frozen {body['n_rows']} rows → {test_path}")
            print(f"  test_digest {body['test_digest']}")
            return EXIT_OK

        test = load_test(test_path)
        report = measure(test, Path(args.index), args.device)
        (out / f"report_{args.stock}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        (out / f"report_{args.stock}.md").write_text(render(report))
        print(f"report_digest {report['report_digest']}")
        print(render(report))
        return EXIT_OK
    except (EvalRefused, DeliveryError, SearchError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
