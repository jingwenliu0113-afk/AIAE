#!/usr/bin/env python3
"""Build and verify the private training pack. Mac side.

The Mac is the only development source. The Taichung machine executes a pack
this script builds, and nothing else. So this is the boundary tool for that
handover, the way ``17_public_snapshot.py`` is the boundary tool for the
public repository -- same discipline, different list, and the rules live in
``src/training/pack.py`` where both this script and the node's verifier read
them.

**There is no transfer mode, and adding one would be the whole mistake.**
This script writes a directory. Moving that directory to the other machine is
a deliberate act performed by a person who can see what is in it: it is not a
flag, it does not happen as a side effect of ``--build``, and no credential
for the other machine is read, stored or asked for here. The pack is small,
reviewable and inert until somebody carries it.

What it does:

  --summary            counts, and the first few of each verdict
  --list               the full manifest as JSON
  --audit              every identifier hit that is not individually approved
  --dependencies       resolve every pinned tokenizer, model and adapter file
                       from this machine's cache and print what each one hashes
                       to. Read-only: it fetches nothing, downloads nothing,
                       reads no credential and writes no file. This is the Mac
                       half of the dependency handover -- the node cannot
                       download them, so the digests are what the operator
                       carries across and compares against what the node's
                       --preflight reports. Prints one canonical
                       dependency_digest over the whole evidence, which is the
                       value the node requires as
                       --expected-dependency-digest.
  --build DEST         copy the included paths into DEST and write the manifest
  --verify DEST        check a built pack file by file against its manifest
  --data-root DIR      with --verify, also check the dataset against its pins
  --expected-pack-digest SHA256
                       with --verify, also check the pack against a digest
                       carried from elsewhere. Without it, --verify answers
                       only whether the pack agrees with its own manifest --
                       which everything inside the manifest was computed from,
                       so it is arithmetic rather than trust. The node always
                       requires the carried value; here it is optional because
                       this is the machine that produced it.

``--build`` refuses rather than warns: a destination that is not a dedicated
empty directory, an allowlisted symlink, an unapproved identifier, or anything
under ``data/`` or ``artifacts/`` reaching the include list all stop it before
a single byte is copied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import pack  # noqa: E402


def _summary(root: Path) -> int:
    m = pack.manifest(root)
    print(f"include : {len(m['include'])}")
    print(f"exclude : {len(m['exclude'])}")
    total = sum(e["bytes"] or 0 for e in m["include"])
    print(f"bytes   : {total:,}")
    print("\nincluded:")
    for entry in m["include"]:
        print(f"  {entry['path']}")
    print("\nrequired data (digested, never copied):")
    for rel, entry in pack.data_requirements(root).items():
        state = (entry["sha256"][:12] + "..." if entry["sha256"]
                 else f"ABSENT -- {entry['reason']}")
        print(f"  {rel}\n      {state}")
    return 0


def _dependencies() -> int:
    """What the pinned dependencies hash to on this machine.

    ``dependency_preflight`` is report 16's, not a second implementation: it
    resolves each pinned file out of the local cache and never fetches. What
    it returns carries the repo id, the revision, the file names, their sizes
    and their digests -- and deliberately not the cache path, because the path
    contains a home directory and this report is meant to travel.
    """
    from src.training.longrun import (dependency_digest,
                                       dependency_preflight)

    result = dependency_preflight()
    evidence = result["evidence"]
    print("Pinned dependencies, resolved from this machine's local cache.")
    print("No network call was made, no tensor was loaded, no credential was "
          "read, and nothing was downloaded or written.\n")
    for repo in evidence.get("repositories", []):
        print(f"{repo['repo_id']} @ {repo['revision']}")
        for f in repo["files"]:
            print(f"    {f['sha256']}  {f['bytes']:>12,}  {f['name']}")
        if not repo["files"]:
            print("    (nothing resolved)")
        print()
    pool = evidence.get("instruction_pool") or {}
    print(f"{pool.get('path')}\n    {pool.get('sha256') or '(absent)'}\n")
    for problem in result["problems"]:
        print(f"MISSING: {problem}")
    # One canonical value over everything above. The node recomputes it from
    # its own cache with the same function and refuses unless the two agree,
    # so this is what binds the node's dependencies to this machine's.
    print(f"dependency_digest  {dependency_digest(evidence)}")
    print("\nCarry that value to the node by an independent channel -- the "
          "same way as pack_digest, and deliberately not inside the pack "
          "manifest. A digest stored beside the files it authenticates is "
          "rewritten by whoever rewrites them.")
    if result["ok"]:
        print("\nevery pinned dependency resolves here.")
        return 0
    print(f"\n{len(result['problems'])} dependency problem(s). The node "
          "cannot download these: carry them across deliberately, the way the "
          "pack is carried, and compare the digests above.")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--dependencies", action="store_true")
    ap.add_argument("--build", metavar="DEST")
    ap.add_argument("--verify", metavar="DEST")
    ap.add_argument("--data-root", metavar="DIR",
                    help="with --verify, check the dataset against the "
                         "digests the pack pinned")
    ap.add_argument("--expected-pack-digest", metavar="SHA256",
                    help="with --verify, also check the pack against a "
                         "pack_digest carried from elsewhere. Without it, "
                         "--verify answers only whether the pack matches its "
                         "own manifest, which is a different question.")
    args = ap.parse_args(argv)

    root = pack.ROOT

    if args.build:
        try:
            body = pack.build(Path(args.build), root=root)
        except pack.PackRefused as exc:
            print(exc, file=sys.stderr)
            return 2
        dest = Path(args.build).expanduser().absolute()
        print(f"built {len(body['files'])} files into {dest}")
        print(f"pack_digest  {body['pack_digest']}")
        print(f"files_digest {body['files_digest']}")
        absent = [rel for rel, e in body["data_requirements"].items()
                  if e["sha256"] is None]
        if absent:
            print(f"\nWARNING: no digest was pinned for {absent}. This pack "
                  "cannot vouch for the dataset the node trains on; rebuild "
                  "it on a tree that has those files.", file=sys.stderr)
        return 0

    if args.verify:
        problems = []
        if args.expected_pack_digest is not None:
            problems += pack.trusted_digest_problems(
                Path(args.verify), args.expected_pack_digest)
        problems += pack.verify(Path(args.verify), data_root=args.data_root)
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} problem(s)")
        return 1 if problems else 0

    if args.dependencies:
        return _dependencies()

    if args.audit:
        problems = pack.pack_audit(
            [e["path"] for e in pack.manifest(root)["include"]], root=root)
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} unapproved hit(s)")
        return 1 if problems else 0

    if args.list:
        print(json.dumps(pack.manifest(root), indent=2, ensure_ascii=False))
        return 0

    return _summary(root)


if __name__ == "__main__":
    raise SystemExit(main())
