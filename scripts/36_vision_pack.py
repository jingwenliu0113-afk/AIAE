#!/usr/bin/env python3
"""Build, audit and verify the private vision pack for the CUDA node.

    --summary   counts and the largest included groups; writes nothing
    --audit     every unapproved identifier hit in the text that would travel
    --build DIR copy the included paths into an empty directory and print the
                pack digest
    --verify DIR  check a built pack file by file against a carried digest

The pack carries the source, the two frozen manifests and the eight-class
members of the public single-brick archive -- about 207 MB, because a
classifier cannot be fitted on digests.  It carries no weights of any kind, not
the detection archive, not the processed text corpus, not the frozen object
split, and no document with a personal absolute path in it.  The boundary and
its reasons are in :mod:`src.vision.pack`.

The digest printed by ``--build`` must be carried to the node by a route other
than the pack itself.  A digest that travels beside the thing it authenticates
proves that route did not contradict itself, and nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import pack

EXIT_OK, EXIT_PROBLEM, EXIT_REFUSED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="print the include/exclude manifest as JSON")
    p.add_argument("--build", metavar="DIR")
    p.add_argument("--verify", metavar="DIR")
    p.add_argument("--expected-pack-digest",
                   help="the digest the build machine printed, carried here "
                        "by a different route")
    p.add_argument("--no-working-tree-check", action="store_true",
                   help="verify the pack alone, without comparing it against "
                        "this tree; for use on the node, where there is no "
                        "source tree to compare with")
    return p


def group_of(rel: str) -> str:
    if rel.startswith("data/raw/vision/classification/photos/"):
        return "classification photographs"
    if rel.startswith("data/raw/vision/classification/renders/"):
        return "classification renders"
    if rel.startswith("src/"):
        return "source"
    if rel.startswith("tests/"):
        return "tests"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("data/raw/vision/"):
        return "frozen manifests"
    return "documents"


def summary() -> dict:
    listing = pack.manifest_paths()
    groups: dict[str, dict[str, int]] = {}
    for entry in listing["include"]:
        bucket = groups.setdefault(group_of(entry["path"]),
                                  {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += entry["bytes"]
    return {
        "include": len(listing["include"]),
        "exclude": len(listing["exclude"]),
        "total_bytes": sum(entry["bytes"] for entry in listing["include"]),
        "groups": {name: groups[name] for name in sorted(groups)},
        "required_present_problems": pack.required_present(),
        "data_agreement_problems": pack.data_agreement_problems(),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [name for name in ("summary", "audit", "list", "build", "verify")
              if getattr(args, name)]
    if len(chosen) != 1:
        print("choose exactly one of --summary, --audit, --list, --build, "
              "--verify", file=sys.stderr)
        return EXIT_REFUSED

    if args.list:
        print(json.dumps(pack.manifest_paths(), indent=2, ensure_ascii=False))
        return EXIT_OK

    if args.summary:
        report = summary()
        print(f"include : {report['include']} file(s), "
              f"{report['total_bytes'] / 1e6:.1f} MB")
        print(f"exclude : {report['exclude']} file(s)")
        for name, bucket in report["groups"].items():
            print(f"  {name:28s} {bucket['files']:6d} files  "
                  f"{bucket['bytes'] / 1e6:8.1f} MB")
        for problem in (report["required_present_problems"]
                        + report["data_agreement_problems"]):
            print(f"problem: {problem}", file=sys.stderr)
        return EXIT_PROBLEM if (report["required_present_problems"]
                                or report["data_agreement_problems"]) \
            else EXIT_OK

    if args.audit:
        listing = pack.manifest_paths()
        included = [entry["path"] for entry in listing["include"]]
        problems = pack.pack_audit(included)
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} unapproved hit(s) in "
              f"{len(pack.scanned(included))} scanned text file(s) of "
              f"{len(included)} included")
        return EXIT_PROBLEM if problems else EXIT_OK

    if args.build:
        try:
            body = pack.build(Path(args.build))
        except pack.VisionPackRefused as exc:
            print(exc, file=sys.stderr)
            return EXIT_REFUSED
        print(f"copied {body['file_count']} file(s), "
              f"{body['total_bytes'] / 1e6:.1f} MB into "
              f"{Path(args.build).expanduser().absolute()}")
        print(f"images                : {body['images']}")
        print(f"data_manifest_sha256  : {body['data_manifest_sha256']}")
        print(f"split_manifest_sha256 : {body['split_manifest_sha256']}")
        print(f"pack_digest           : {body['pack_digest']}")
        print("\nCarry pack_digest to the node by a route other than the pack.")
        return EXIT_OK

    problems = pack.verify(
        Path(args.verify), expected_digest=args.expected_pack_digest,
        check_working_tree=not args.no_working_tree_check)
    for problem in problems:
        print(problem)
    print(f"\n{len(problems)} problem(s)")
    return EXIT_PROBLEM if problems else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
