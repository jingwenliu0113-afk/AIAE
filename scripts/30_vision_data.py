#!/usr/bin/env python3
"""Audit and selectively extract the two public LEGO image archives.

    --audit       read each archive's central directory and print what the
                  eight-class filter would take. Downloads no image.
    --extract     fetch exactly those members into the private raw directory
                  and write the data manifest.
    --verify      re-check every extracted file against the manifest, offline.

Nothing else in this project reaches the network.  Both archives are about six
gigabytes and hold 447 brick classes; this project uses eight, so the archives
are read through their central directories and only the wanted members are
fetched.  The detection photographs carry no per-brick class and are 6.0 GB, so
a documented deterministic subset is taken -- and what was dropped is counted
in the manifest rather than left to look like full coverage.

The images land in ``data/raw/vision/``, which is denied by the public snapshot
allowlist and by the GPU pack.  The manifest is the only artefact that leaves
that directory.

Licence: both archives are CC BY 4.0 from Gdansk University of Technology's
Bridge of Knowledge.  Attribution is recorded in the manifest and in
``VISION.md``; the images are not redistributed by this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import datasets, source
from src.vision.classes import design_numbers

RAW = ROOT / "data/raw/vision"
MANIFEST_DIR = ROOT / "data/raw/vision"

EXIT_OK, EXIT_PROBLEM, EXIT_REFUSED = 0, 1, 2

#: Members fetched concurrently.  Small: this is a public mirror, and the
#: point of the range reader is to take less from it, not to take it faster.
FETCH_WORKERS = 8


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=sorted(datasets.SOURCES),
                   action="append",
                   help="restrict to one archive; repeatable, default both")
    p.add_argument("--audit", action="store_true",
                   help="read the central directory and report the selection")
    p.add_argument("--extract", action="store_true",
                   help="fetch the selected members and write the manifest")
    p.add_argument("--verify", action="store_true",
                   help="re-check extracted files against the manifest")
    p.add_argument("--cache-central-directory", metavar="DIR",
                   help="store the central directory so a repeated audit or "
                        "extract does not re-download 82 MB")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def _cache_path(cache: Path, key: str) -> Path:
    return Path(cache) / f"{key}_central_directory.json"


def load_archive(key: str, *, cache=None):
    """Read one archive's central directory, from the cache when present."""
    if cache:
        path = _cache_path(Path(cache), key)
        if path.is_file():
            body = json.loads(path.read_text(encoding="utf-8"))
            entries = tuple(source.ZipEntry(*row) for row in body["entries"])
            return source.RemoteZip(
                total_bytes=body["total_bytes"],
                entry_count=body["entry_count"],
                central_directory_offset=body["central_directory_offset"],
                central_directory_bytes=body["central_directory_bytes"],
                central_directory_sha256=body["central_directory_sha256"],
                entries=entries), None
    from src.vision import net

    reader = net.open_range_reader(datasets.SOURCES[key].download)
    archive = source.read_central_directory(
        reader.fetcher(), reader.total_bytes, chunk_bytes=8 << 20)
    if cache:
        path = _cache_path(Path(cache), key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "total_bytes": archive.total_bytes,
            "entry_count": archive.entry_count,
            "central_directory_offset": archive.central_directory_offset,
            "central_directory_bytes": archive.central_directory_bytes,
            "central_directory_sha256": archive.central_directory_sha256,
            "entries": [[e.name, e.method, e.crc32, e.compressed_bytes,
                         e.uncompressed_bytes, e.local_offset]
                        for e in archive.entries],
        }), encoding="utf-8")
    return archive, reader


def select(key: str, archive):
    """Apply the eight-class filter for one archive."""
    if key == datasets.CLASSIFICATION.key:
        records, summary = datasets.classification_records(archive.entries)
        return records, [], summary
    return datasets.detection_records(archive.entries)


def audit(keys, *, cache=None) -> dict:
    out = {}
    for key in keys:
        archive, _reader = load_archive(key, cache=cache)
        records, labels, summary = select(key, archive)
        by_population: dict[str, dict] = {}
        for record in records:
            bucket = by_population.setdefault(
                record.population, {"images": 0, "bytes": 0, "groups": set()})
            bucket["images"] += 1
            bucket["bytes"] += record.uncompressed_bytes
            bucket["groups"].add(record.group)
        out[key] = {
            "source": datasets.SOURCES[key].as_dict(),
            "archive_total_bytes": archive.total_bytes,
            "archive_entries": archive.entry_count,
            "central_directory_sha256": archive.central_directory_sha256,
            "selection": summary,
            "label_files": len(labels),
            "download_bytes": sum(r.uncompressed_bytes for r in records),
            "populations": {
                name: {"images": bucket["images"], "bytes": bucket["bytes"],
                       "capture_groups": len(bucket["groups"])}
                for name, bucket in sorted(by_population.items())},
        }
    return out


def _member_path(record) -> Path:
    return RAW / record.dataset / record.member


def extract(keys, *, cache=None) -> dict:
    out = {}
    for key in keys:
        archive, reader = load_archive(key, cache=cache)
        if reader is None:
            from src.vision import net

            reader = net.open_range_reader(datasets.SOURCES[key].download)
        fetch = reader.fetcher()
        records, labels, summary = select(key, archive)
        by_name = {entry.name: entry for entry in archive.entries}

        wanted = [(record.member, _member_path(record)) for record in records]
        wanted += [(name, RAW / key / name) for name in labels]
        todo = [(member, target) for member, target in wanted
                if not target.is_file()]
        print(f"[{key}] {len(wanted)} member(s) selected, "
              f"{len(todo)} still to fetch", flush=True)

        extracted: dict[str, dict] = {}
        started = time.monotonic()
        destination = {member: target for member, target in todo}
        spans = source.plan_spans([by_name[member] for member in destination])
        print(f"[{key}] {len(spans)} coalesced span(s) to read", flush=True)

        def one_span(span):
            written = []
            for entry, payload in source.read_span(
                    fetch, span, total_bytes=archive.total_bytes):
                target = destination[entry.name]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                written.append((entry.name,
                                hashlib.sha256(payload).hexdigest(),
                                len(payload)))
            return written

        done = 0
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            for index, written in enumerate(pool.map(one_span, spans), 1):
                for member, digest, size in written:
                    extracted[member] = {"sha256": digest, "bytes": size}
                done += len(written)
                if index % 5 == 0 or index == len(spans):
                    elapsed = time.monotonic() - started
                    print(f"[{key}]   span {index}/{len(spans)}  "
                          f"{done}/{len(todo)} members  {elapsed:.0f}s",
                          flush=True)

        # Files already on disk keep their digest from the disk copy, so the
        # manifest describes what is actually there rather than what a fetch
        # returned on some earlier run.
        for member, target in wanted:
            if member not in extracted:
                payload = target.read_bytes()
                extracted[member] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload)}

        label_map = {datasets.label_for(record.member): record.member
                     for record in records
                     if record.dataset == datasets.DETECTION.key}
        manifest = datasets.build_manifest(
            datasets.SOURCES[key], archive=archive, records=records,
            summary=summary, extracted=extracted,
            labels={name: label_map.get(name, "") for name in labels})
        path, digest = datasets.write_manifest(
            manifest, MANIFEST_DIR / f"{key}_manifest.json")
        out[key] = {"manifest": str(path.relative_to(ROOT)),
                    "manifest_sha256": digest,
                    "members": len(extracted),
                    "bytes": sum(v["bytes"] for v in extracted.values())}
        print(f"[{key}] manifest {path.relative_to(ROOT)} sha256={digest}",
              flush=True)
    return out


def verify(keys) -> tuple[dict, list[str]]:
    """Offline re-check: every manifest member present, right size, right hash."""
    out = {}
    problems: list[str] = []
    for key in keys:
        path = MANIFEST_DIR / f"{key}_manifest.json"
        if not path.is_file():
            problems.append(f"{key}: no manifest at {path.relative_to(ROOT)}")
            continue
        manifest = datasets.read_manifest(path)
        digest = datasets.manifest_digest(manifest)
        checked = 0
        for member, expected in sorted(manifest["extracted"].items()):
            target = RAW / key / member
            if not target.is_file():
                problems.append(f"{key}: {member} is missing")
                continue
            payload = target.read_bytes()
            if len(payload) != expected["bytes"]:
                problems.append(
                    f"{key}: {member} is {len(payload)} bytes, manifest says "
                    f"{expected['bytes']}")
                continue
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected["sha256"]:
                problems.append(f"{key}: {member} hashes to {actual}")
                continue
            checked += 1
        records = datasets.records_from_manifest(manifest)
        groups = {r.group for r in records}
        out[key] = {"manifest_sha256": digest, "members": checked,
                    "records": len(records), "capture_groups": len(groups),
                    "central_directory_sha256":
                        manifest["archive"]["central_directory_sha256"]}
    return out, problems


def _print_audit(report: dict) -> None:
    for key, block in report.items():
        source_block = block["source"]
        print("=" * 72)
        print(f"{source_block['title']}  (version {source_block['version']})")
        print("=" * 72)
        print(f"  DOI           : {source_block['doi']}")
        print(f"  licence       : {source_block['licence']}")
        print(f"  attribution   : {source_block['attribution']}")
        print(f"  archive       : {block['archive_total_bytes']} bytes, "
              f"{block['archive_entries']} entries")
        print(f"  cd sha256     : {block['central_directory_sha256']}")
        print(f"  eight classes : {' '.join(design_numbers())}")
        print(f"  would fetch   : {block['download_bytes']} bytes "
              f"({block['download_bytes'] / 1e6:.1f} MB), "
              f"{block['label_files']} label file(s)")
        for name, pop in block["populations"].items():
            print(f"    {name:10s} {pop['images']:6d} images  "
                  f"{pop['bytes'] / 1e6:8.1f} MB  "
                  f"{pop['capture_groups']:5d} capture groups")
        selection = block["selection"]
        if "archive_classes_seen" in selection:
            print(f"  classes in archive: {selection['archive_classes_seen']}")
            print(f"  skipped (other class members): "
                  f"{selection['skipped_other_class_members']}")
        for bucket in selection.get("photo_buckets", []):
            top = bucket["bricks_to"] or "+"
            print(f"  photos {bucket['bricks_from']}-{top}: "
                  f"stride {bucket['stride']}, took {bucket['taken']} of "
                  f"{bucket['available']}, dropped {bucket['dropped']}")
        print()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    keys = args.dataset or sorted(datasets.SOURCES)
    if not (args.audit or args.extract or args.verify):
        print("choose --audit, --extract or --verify", file=sys.stderr)
        return EXIT_REFUSED
    try:
        if args.audit:
            report = audit(keys, cache=args.cache_central_directory)
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False,
                                 sort_keys=True))
            else:
                _print_audit(report)
        if args.extract:
            report = extract(keys, cache=args.cache_central_directory)
            print(json.dumps(report, indent=2, ensure_ascii=False,
                             sort_keys=True))
        if args.verify:
            report, problems = verify(keys)
            print(json.dumps(report, indent=2, ensure_ascii=False,
                             sort_keys=True))
            for problem in problems:
                print(f"problem: {problem}", file=sys.stderr)
            print(f"\n{len(problems)} problem(s)")
            return EXIT_PROBLEM if problems else EXIT_OK
    except (datasets.DatasetError, source.SourceError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
