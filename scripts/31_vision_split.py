#!/usr/bin/env python3
"""Build and freeze the vision splits, before anything is fitted.

    --plan      print what the split would be, writing nothing
    --freeze    write the manifests; refuses to overwrite an existing one
    --verify    re-check a frozen manifest and print its digest and counts

Two manifests, because the two datasets answer different questions and their
capture groups mean different things:

``classification``
    Stratified by class, grouped by source photograph (for the real
    photographs) and by rendered instance (for the renders). Strata are the
    eight classes, so every class reaches every split.

``detection``
    Grouped by photographic session, by arrangement, or by source-photograph
    token. One stratum: the archive's boxes carry no per-brick class, so there
    is nothing to stratify over.

The split boundary is drawn between *groups*.  Thirty crops of one photograph
go to one side; a day's shooting goes to one side.  Splitting per image would
put near-duplicates on both sides of the test boundary and the number that came
back would be partly a memory score.

Test is opened once, after the model, the augmentation, the stopping rule and
the acceptance code are frozen.  The digest printed by ``--freeze`` is what
``scripts/33_vision_eval.py`` requires before it reads a test image.

This is a new split for a new task.  It has nothing to do with the 160 frozen
Phase 2 cases and no number from it may be placed beside one of theirs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import datasets
from src.vision.split import SplitError, SplitRecord, VisionSplit

RAW = ROOT / "data/raw/vision"

EXIT_OK, EXIT_PROBLEM, EXIT_REFUSED = 0, 1, 2

#: One stratum for the detection set: its ground truth has no class to balance.
DETECTION_STRATUM = "all"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=sorted(datasets.SOURCES),
                   action="append",
                   help="restrict to one dataset; repeatable, default both")
    p.add_argument("--plan", action="store_true",
                   help="print the split without writing it")
    p.add_argument("--freeze", action="store_true",
                   help="write the split manifest (never overwrites)")
    p.add_argument("--verify", action="store_true",
                   help="load a frozen manifest and re-check it")
    p.add_argument("--expected-digest", action="append", metavar="KEY=SHA256",
                   help="require this digest for a dataset's manifest")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def manifest_path(key: str) -> Path:
    return RAW / f"{key}_split.json"


def split_records(key: str, records) -> list[SplitRecord]:
    """Turn data records into split records, with the right stratum each."""
    out = []
    for record in records:
        if key == datasets.CLASSIFICATION.key:
            # Stratify by class *and* by population: a class must reach every
            # split in both the real and the synthetic population, or a test
            # set can end up with renders only for some classes and the real
            # per-class recall becomes undefined for exactly those.
            stratum = f"{record.part}/{record.population}"
        elif record.part:
            # A detection render's own filename gives its class, so those can
            # and must be stratified: the first attempt used one stratum for
            # the whole detection set and three of the eight classes reached
            # no test render at all, which would have made the per-class count
            # error undefined for them. Recorded rather than quietly fixed --
            # see VISION.md.
            stratum = f"{record.part}/{record.population}"
        else:
            # The photographs. Their boxes carry no per-brick class, so there
            # is nothing to stratify over and they share one stratum.
            stratum = DETECTION_STRATUM
        out.append(SplitRecord(item_id=record.member, group=record.group,
                               stratum=stratum, label=record.part or None))
    return out


def build(key: str) -> tuple[VisionSplit, list]:
    """The split, and the records it was built from, for the description."""
    manifest = datasets.read_manifest(RAW / f"{key}_manifest.json")
    records = datasets.records_from_manifest(manifest)
    if not records:
        raise SplitError(f"the {key} data manifest carries no records")
    digest = datasets.manifest_digest(manifest)
    note = (f"built from the {key} data manifest at sha256 {digest}. "
            "Groups are capture groups; the boundary is never drawn inside "
            "one. This is a new vision test and is unrelated to the frozen "
            "Phase 2 cases.")
    return VisionSplit.build(key, split_records(key, records), note=note), records


def describe(split: VisionSplit, records=None) -> dict:
    """The split's shape, and -- when the records are to hand -- by population.

    The real/synthetic breakdown is the number that actually matters: renders
    outnumber photographs nine to one in the classification archive, so a
    combined per-split count says almost nothing about how much real test data
    there is.
    """
    out = {
        "dataset": split.dataset,
        "digest": split.digest(),
        "items": split.counts(),
        "groups": split.group_counts(),
        "labels": split.label_counts(),
        "strata": len(set(split.strata.values())),
        "note": split.note,
    }
    if records:
        population = {name: {} for name in split.counts()}
        for record in records:
            side = split.groups.get(split.items.get(record.member, ""), None)
            if side is None:
                continue
            bucket = population[side].setdefault(record.population, {})
            key = record.part or "unlabelled"
            bucket[key] = bucket.get(key, 0) + 1
        out["by_population"] = {
            side: {name: dict(sorted(counts.items()))
                   for name, counts in sorted(buckets.items())}
            for side, buckets in population.items()}
        out["population_totals"] = {
            side: {name: sum(counts.values())
                   for name, counts in sorted(buckets.items())}
            for side, buckets in population.items()}
    return out


def _print(report: dict) -> None:
    for key, block in report.items():
        print("=" * 72)
        print(f"{key} vision split")
        print("=" * 72)
        print(f"  digest  : {block['digest']}")
        print(f"  items   : " + "  ".join(
            f"{name}={count}" for name, count in block["items"].items()))
        print(f"  groups  : " + "  ".join(
            f"{name}={count}" for name, count in block["groups"].items()))
        print(f"  strata  : {block['strata']}")
        for name, counts in block["labels"].items():
            if counts:
                print(f"    {name:11s} " + "  ".join(
                    f"{part}:{n}" for part, n in counts.items()))
        for name, buckets in (block.get("by_population") or {}).items():
            for population, counts in buckets.items():
                print(f"    {name}/{population}: "
                      + "  ".join(f"{part}:{n}"
                                  for part, n in counts.items()))
        if block.get("path"):
            print(f"  written : {block['path']}")
        print()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    keys = args.dataset or sorted(datasets.SOURCES)
    expected = {}
    for pair in args.expected_digest or []:
        name, _, value = pair.partition("=")
        if not value:
            print(f"--expected-digest wants KEY=SHA256, got {pair!r}",
                  file=sys.stderr)
            return EXIT_REFUSED
        expected[name] = value
    if not (args.plan or args.freeze or args.verify):
        print("choose --plan, --freeze or --verify", file=sys.stderr)
        return EXIT_REFUSED

    report: dict[str, dict] = {}
    problems: list[str] = []
    try:
        for key in keys:
            if args.verify:
                path = manifest_path(key)
                split = VisionSplit.load(
                    path, expected_digest=expected.get(key))
                split.check_no_leakage()
                data = datasets.read_manifest(RAW / f"{key}_manifest.json")
                block = describe(split,
                                 datasets.records_from_manifest(data))
                block["path"] = str(path.relative_to(ROOT))
                # A frozen split must still describe the data that is here.
                members = {row["member"] for row in data["records"]}
                missing = sorted(set(split.items) - members)
                extra = sorted(members - set(split.items))
                if missing:
                    problems.append(
                        f"{key}: {len(missing)} item(s) in the frozen split "
                        f"are not in the data manifest, first {missing[:3]}")
                if extra:
                    problems.append(
                        f"{key}: {len(extra)} item(s) in the data manifest are "
                        f"not in the frozen split, first {extra[:3]}. The "
                        "split was frozen against different data")
                report[key] = block
                continue

            split, records = build(key)
            split.check_no_leakage()
            block = describe(split, records)
            if args.freeze:
                path, digest = split.freeze(manifest_path(key))
                block["path"] = str(path.relative_to(ROOT))
                block["digest"] = digest
            report[key] = block
    except (SplitError, datasets.DatasetError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print(report)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} problem(s)")
        return EXIT_PROBLEM
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
