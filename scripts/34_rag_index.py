#!/usr/bin/env python3
"""Build and query the multilingual retrieval index over the train catalogue.

    --fetch-model  download the pinned embedding model once; the only mode
                   that touches the network
    --check        report whether a strict-offline build would work
    --build DIR    embed the train catalogue and write the index there
    --verify DIR   re-check an index against its own digests, offline
    --query TEXT   search a built index and print the grounded explanation

The index is train-only, because the catalogue loader it goes through refuses
any row that is not ``split=train`` and checks every row against the frozen
object-level split manifest at its pinned digest.  ``--exclude-object-id``
applies the same-object exclusion inside the search for a held-out query.

Building an index measures nothing.  No retrieval quality claim follows from
running this, and none is printed: a separate frozen retrieval test would be
needed for that and has not been run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.delivery.pipeline import DeliveryError, load_train_catalog
from src.retrieval import embed as embed_module
from src.retrieval import index as index_module
from src.retrieval.explain import explain_result, format_explanation
from src.retrieval.nlp import NlpError, extract
from src.retrieval.search import SearchError, search
from src.vision.model_ids import TEXT_EMBEDDING

DEFAULT_CATALOG = ROOT / "data/processed/counterfactual_train.jsonl"
DEFAULT_INDEX = ROOT / "runs/retrieval/index"

EXIT_OK, EXIT_NO_RESULT, EXIT_REFUSED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fetch-model", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--build", metavar="DIR", nargs="?", const=str(DEFAULT_INDEX))
    p.add_argument("--verify", metavar="DIR", nargs="?",
                   const=str(DEFAULT_INDEX))
    p.add_argument("--query", metavar="TEXT")
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    p.add_argument("--inventory", help="manual stock, e.g. '2x4:10,1x2:8'")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--exclude-object-id")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    p.add_argument("--expected-manifest-sha256")
    p.add_argument("--json", action="store_true")
    return p


def fetch_model() -> dict:
    from huggingface_hub import snapshot_download

    pin = TEXT_EMBEDDING
    snapshot_download(pin.repo, revision=pin.revision,
                      allow_patterns=list(pin.files))
    return embed_module.check_cache()


def check(args) -> dict:
    cache = embed_module.check_cache()
    catalog = None
    try:
        catalog = load_train_catalog(args.catalog)
    except DeliveryError as exc:
        catalogue_state = f"unavailable: {exc}"
    else:
        catalogue_state = "loaded"
    return {
        "model": cache,
        "catalogue": catalogue_state,
        "catalogue_file": Path(args.catalog).name,
        "catalogue_sha256": catalog.sha256 if catalog else None,
        "split_manifest_sha256": (catalog.split_manifest_sha256
                                  if catalog else None),
        "canonical_train_structures": len(catalog.items) if catalog else 0,
        "boundary": ("train split only, checked row by row against the frozen "
                     "object-level split manifest"),
    }


def build_index(args) -> dict:
    catalog = load_train_catalog(args.catalog)
    embedder = embed_module.load(device=args.device)
    started = time.monotonic()
    built = index_module.build(catalog, embedder)
    target, digest = built.save(args.build)
    return {
        "index": str(Path(args.build)),
        "index_manifest_sha256": digest,
        "documents": built.size,
        "dimension": int(built.vectors.shape[1]),
        "embedding": built.embedding,
        "identity_digest": built.identity_digest,
        "catalogue_sha256": built.catalog_sha256,
        "split_manifest_sha256": built.split_manifest_sha256,
        "seconds": round(time.monotonic() - started, 2),
        "not_a_metric": ("building an index measures nothing; no retrieval "
                         "quality claim follows from this"),
    }


def verify_index(args) -> tuple[dict, list[str]]:
    problems: list[str] = []
    try:
        loaded = index_module.load(
            args.verify,
            expected_manifest_sha256=args.expected_manifest_sha256)
    except index_module.IndexError_ as exc:
        return {}, [str(exc)]
    body = {
        "index": str(Path(args.verify)),
        "documents": loaded.size,
        "dimension": int(loaded.vectors.shape[1]),
        "embedding": loaded.embedding,
        "identity_digest": loaded.identity_digest,
        "catalogue_sha256": loaded.catalog_sha256,
        "split_manifest_sha256": loaded.split_manifest_sha256,
        "build_device": loaded.build_device,
    }
    try:
        catalog = load_train_catalog(args.catalog)
    except DeliveryError as exc:
        problems.append(f"the catalogue is unavailable: {exc}")
        return body, problems
    if catalog.sha256 != loaded.catalog_sha256:
        problems.append(
            f"the index was built from catalogue {loaded.catalog_sha256} and "
            f"the catalogue now present is {catalog.sha256}")
    if catalog.split_manifest_sha256 != loaded.split_manifest_sha256:
        problems.append(
            "the index was built against frozen split manifest "
            f"{loaded.split_manifest_sha256} and the catalogue was loaded "
            f"against {catalog.split_manifest_sha256}")
    ids = {item.catalog_id for item in catalog.items}
    missing = [d.catalog_id for d in loaded.documents
               if d.catalog_id not in ids]
    if missing:
        problems.append(
            f"{len(missing)} indexed document(s) are not in the catalogue, "
            f"first {missing[:3]}")
    # The object ids too, not only the public ones: they are what the
    # same-object exclusion is applied to, so an index carrying plausible but
    # wrong ones would exclude the wrong work with every digest agreeing.
    try:
        loaded.check_against_catalog(catalog)
        body["object_ids_match_catalogue"] = True
    except index_module.IndexError_ as exc:
        body["object_ids_match_catalogue"] = False
        problems.append(str(exc))
    return body, problems


def run_query(args) -> tuple[dict, int]:
    from src.demo.showcase import parse_inventory

    if not args.inventory:
        raise SearchError("--query needs --inventory")
    inventory = parse_inventory(args.inventory)
    catalog = load_train_catalog(args.catalog)
    embedder = embed_module.load(device=args.device)
    loaded = index_module.load(
        args.index,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_identity_digest=embedder.identity_digest(),
        expected_catalog_sha256=catalog.sha256,
        expected_split_manifest_sha256=catalog.split_manifest_sha256)
    conditions = extract(args.query)
    result = search(loaded, catalog, embedder, conditions, inventory,
                    top_n=args.top_n,
                    exclude_object_id=args.exclude_object_id)
    explanation = explain_result(result, inventory)
    return {"result": result.as_dict(), "explanation": explanation}, (
        EXIT_OK if result.selected else EXIT_NO_RESULT)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [name for name in ("fetch_model", "check", "build", "verify",
                                "query") if getattr(args, name)]
    if len(chosen) != 1:
        print("choose exactly one of --fetch-model, --check, --build, "
              "--verify, --query", file=sys.stderr)
        return EXIT_REFUSED
    try:
        if args.fetch_model:
            print(json.dumps(fetch_model(), indent=2, sort_keys=True))
            return EXIT_OK
        if args.check:
            print(json.dumps(check(args), indent=2, ensure_ascii=False,
                             sort_keys=True))
            return EXIT_OK
        if args.build:
            print(json.dumps(build_index(args), indent=2, ensure_ascii=False,
                             sort_keys=True))
            return EXIT_OK
        if args.verify:
            body, problems = verify_index(args)
            print(json.dumps(body, indent=2, ensure_ascii=False,
                             sort_keys=True))
            for problem in problems:
                print(f"problem: {problem}", file=sys.stderr)
            print(f"\n{len(problems)} problem(s)")
            return EXIT_REFUSED if problems else EXIT_OK
        body, code = run_query(args)
        if args.json:
            print(json.dumps(body, indent=2, ensure_ascii=False,
                             sort_keys=True))
        else:
            print(format_explanation(body["explanation"]))
        return code
    except (SearchError, NlpError, DeliveryError, embed_module.EmbedError,
            index_module.IndexError_) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
