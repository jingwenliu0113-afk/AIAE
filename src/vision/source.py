"""Read selected members out of a remote ZIP without downloading the whole file.

The two public datasets this project uses are single ZIP archives of about six
gigabytes each, holding 447 brick classes.  This project needs eight.  Pulling
six gigabytes to keep a hundred megabytes would be a waste of the mirror's
bandwidth and of local disk, and the boundary rule for this work is to fetch
only what the eight classes need.  So the archives are read the way a ZIP is
designed to be read: the end-of-central-directory record, then the central
directory, then a byte range per wanted member.

Nothing here runs at import time and nothing here is reached by the delivery,
UI or scoring paths.  The network is entered only when a caller passes a real
fetcher, which is what ``scripts/30_vision_data.py`` does and what the tests
deliberately do not: they pass a fetcher backed by a local file, so the whole
reader is exercised offline.

Three refusals are the reason this module exists rather than a ``curl`` call:

* **Every member is verified.**  The inflated bytes have to match the central
  directory's CRC-32 *and* its uncompressed size.  A truncated or substituted
  member is refused, not stored.
* **A declared size is not trusted.**  A member claiming more than
  :data:`MAX_MEMBER_BYTES` is refused before a decompressor is created, and the
  decompressor is driven with a hard output cap, so a crafted entry cannot
  expand until the machine runs out of memory.
* **A member name is not a path.**  Absolute names, ``..`` segments, drive
  letters and backslashes are refused, so nothing extracted here can be
  written outside the destination a caller chose.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

#: The largest member this reader will inflate.  The datasets hold photographs
#: and renders; the biggest is a few megabytes.  Anything claiming more than
#: this is not a photograph and is refused before any memory is committed.
MAX_MEMBER_BYTES = 64 * 1024 * 1024

#: The largest central directory this reader will read.  The classification
#: archive's is about 82 MB for 620,974 entries, so the cap has room for a
#: bigger archive of the same shape without having no cap at all.
MAX_CENTRAL_DIRECTORY_BYTES = 512 * 1024 * 1024

#: Deflate and store.  A ZIP may name a dozen other methods; this reader
#: refuses them rather than silently skipping the member.
METHOD_STORE = 0
METHOD_DEFLATE = 8

_EOCD = b"PK\x05\x06"
_EOCD64 = b"PK\x06\x06"
_CENTRAL = b"PK\x01\x02"
_LOCAL = b"PK\x03\x04"

#: A ZIP comment may be 65,535 bytes, so the record can start that far back.
_EOCD_SEARCH = 65557

#: A fetcher takes an inclusive byte range and returns exactly those bytes.
Fetcher = Callable[[int, int], bytes]


class SourceError(ValueError):
    """The archive, one of its members, or a fetched range is unusable."""


@dataclass(frozen=True)
class ZipEntry:
    """One central-directory record, as the reader needs it."""

    name: str
    method: int
    crc32: int
    compressed_bytes: int
    uncompressed_bytes: int
    local_offset: int

    @property
    def is_directory(self) -> bool:
        return self.name.endswith("/")


@dataclass(frozen=True)
class RemoteZip:
    """A remote archive whose central directory has been read and pinned.

    ``central_directory_sha256`` is the identity this project records.  The
    mirror serves the archive through expiring signed URLs, so the URL is not
    a stable name for anything; the digest of the central directory is, and it
    changes if a single member is added, removed, renamed or re-compressed.
    """

    total_bytes: int
    entry_count: int
    central_directory_offset: int
    central_directory_bytes: int
    central_directory_sha256: str
    entries: tuple[ZipEntry, ...] = field(repr=False)

    def by_name(self, name: str) -> ZipEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise SourceError(f"the archive has no member named {name!r}")


def _exact(fetch: Fetcher, start: int, end: int) -> bytes:
    """Fetch an inclusive range and refuse anything but exactly those bytes."""
    if start < 0 or end < start:
        raise SourceError(f"a byte range must be forward and non-negative: "
                          f"{start}-{end}")
    want = end - start + 1
    got = fetch(start, end)
    if not isinstance(got, (bytes, bytearray)):
        raise SourceError("the fetcher returned "
                          f"{type(got).__name__}, not bytes")
    if len(got) != want:
        raise SourceError(
            f"the fetcher returned {len(got)} bytes for the {want}-byte range "
            f"{start}-{end}; a short range would be parsed as though it were "
            "the whole thing")
    return bytes(got)


def _parse_central(blob: bytes) -> tuple[ZipEntry, ...]:
    """Parse a whole central directory.  A trailing partial record refuses."""
    out: list[ZipEntry] = []
    p = 0
    n = len(blob)
    while p < n:
        if blob[p:p + 4] != _CENTRAL:
            raise SourceError(
                f"central directory record {len(out)} does not start with the "
                "expected signature; the directory is not intact")
        if p + 46 > n:
            raise SourceError("the central directory ends inside a record")
        (_sig, _ver, _needed, _flags, method, _mtime, _mdate, crc, comp, unc,
         nlen, elen, clen, _disk, _iattr, _eattr,
         lho) = struct.unpack("<IHHHHHHIIIHHHHHII", blob[p:p + 46])
        head = p + 46
        if head + nlen + elen + clen > n:
            raise SourceError("a central directory record overruns the "
                              "directory it is in")
        raw_name = blob[head:head + nlen]
        extra = blob[head + nlen:head + nlen + elen]
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            # cp437 is the pre-Unicode ZIP encoding.  Decoding it rather than
            # replacing bytes keeps the name reversible, and the path check
            # below is what actually decides whether it may be used.
            name = raw_name.decode("cp437")
        if 0xFFFFFFFF in (unc, comp, lho):
            unc, comp, lho = _zip64(extra, unc, comp, lho)
        out.append(ZipEntry(name=name, method=method, crc32=crc,
                            compressed_bytes=comp, uncompressed_bytes=unc,
                            local_offset=lho))
        p = head + nlen + elen + clen
    return tuple(out)


def _zip64(extra: bytes, unc: int, comp: int, lho: int) -> tuple[int, int, int]:
    """Read the ZIP64 extended-information field for the 0xFFFFFFFF markers."""
    q = 0
    while q + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[q:q + 4])
        body = extra[q + 4:q + 4 + size]
        if header_id == 0x0001:
            k = 0
            if unc == 0xFFFFFFFF:
                if k + 8 > len(body):
                    raise SourceError("a ZIP64 field is too short for the "
                                      "sizes it has to carry")
                unc = struct.unpack("<Q", body[k:k + 8])[0]
                k += 8
            if comp == 0xFFFFFFFF:
                if k + 8 > len(body):
                    raise SourceError("a ZIP64 field is too short for the "
                                      "sizes it has to carry")
                comp = struct.unpack("<Q", body[k:k + 8])[0]
                k += 8
            if lho == 0xFFFFFFFF:
                if k + 8 > len(body):
                    raise SourceError("a ZIP64 field is too short for the "
                                      "offset it has to carry")
                lho = struct.unpack("<Q", body[k:k + 8])[0]
            return unc, comp, lho
        q += 4 + size
    raise SourceError("a record needs ZIP64 sizes but carries no ZIP64 field")


def read_central_directory(fetch: Fetcher, total_bytes: int, *,
                           chunk_bytes: int = 1 << 20) -> RemoteZip:
    """Locate and read the central directory of a remote archive.

    ``chunk_bytes`` is how much is asked for per range.  The mirror this
    project reads drops a connection after a couple of megabytes, so the
    default is deliberately small and the caller's fetcher is expected to
    retry a range rather than the whole directory.
    """
    if not isinstance(total_bytes, int) or isinstance(total_bytes, bool) \
            or total_bytes <= 0:
        raise SourceError("the archive size must be a positive whole number")
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) \
            or chunk_bytes < 1:
        raise SourceError("chunk_bytes must be a positive whole number")

    tail_start = max(0, total_bytes - _EOCD_SEARCH)
    tail = _exact(fetch, tail_start, total_bytes - 1)
    i = tail.rfind(_EOCD)
    if i < 0 or i + 22 > len(tail):
        raise SourceError(
            "no end-of-central-directory record was found in the last "
            f"{len(tail)} bytes; this does not look like a ZIP archive")
    count, cd_size, cd_offset = struct.unpack("<HII", tail[i + 10:i + 20])
    if 0xFFFF in (count,) or 0xFFFFFFFF in (cd_size, cd_offset):
        j = tail.rfind(_EOCD64)
        if j < 0 or j + 56 > len(tail):
            raise SourceError(
                "the archive needs a ZIP64 end-of-central-directory record "
                "and does not have one")
        count, cd_size, cd_offset = struct.unpack("<QQQ", tail[j + 32:j + 56])

    if cd_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise SourceError(
            f"the central directory claims {cd_size} bytes, over the "
            f"{MAX_CENTRAL_DIRECTORY_BYTES} cap this reader will read")
    if cd_offset + cd_size > total_bytes:
        raise SourceError("the central directory does not fit inside the "
                          "archive it describes")

    blob = bytearray()
    while len(blob) < cd_size:
        start = cd_offset + len(blob)
        end = min(start + chunk_bytes, cd_offset + cd_size) - 1
        blob += _exact(fetch, start, end)
    entries = _parse_central(bytes(blob))
    if len(entries) != count:
        raise SourceError(
            f"the archive declares {count} entries and its central directory "
            f"holds {len(entries)}")
    return RemoteZip(
        total_bytes=total_bytes, entry_count=len(entries),
        central_directory_offset=cd_offset, central_directory_bytes=cd_size,
        central_directory_sha256=hashlib.sha256(bytes(blob)).hexdigest(),
        entries=entries)


def safe_member_name(name: str) -> str:
    """Refuse a member name that could escape a destination directory.

    A ZIP member name is attacker-controlled data, and ``dest / name`` with a
    name like ``../../etc/x`` writes outside ``dest``.  Refusing the name is
    the only check that holds regardless of what the caller does with it.
    """
    if not isinstance(name, str) or not name:
        raise SourceError("a member name must be a non-empty string")
    if "\x00" in name:
        raise SourceError("a member name may not contain a null byte")
    if "\\" in name:
        raise SourceError(f"member name {name!r} contains a backslash")
    if name.startswith("/"):
        raise SourceError(f"member name {name!r} is absolute")
    if len(name) >= 2 and name[1] == ":":
        raise SourceError(f"member name {name!r} carries a drive letter")
    parts = name.split("/")
    if any(part in ("..", ".") for part in parts):
        raise SourceError(f"member name {name!r} contains a relative segment")
    return name


def _inflate(payload: bytes, entry: ZipEntry) -> bytes:
    """Inflate one member under a hard output cap."""
    if entry.method == METHOD_STORE:
        return payload
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    out = decompressor.decompress(payload, entry.uncompressed_bytes + 1)
    if len(out) > entry.uncompressed_bytes:
        raise SourceError(
            f"member {entry.name!r} inflates past the {entry.uncompressed_bytes} "
            "bytes its own directory record declares")
    out += decompressor.flush()
    if len(out) > entry.uncompressed_bytes:
        raise SourceError(
            f"member {entry.name!r} inflates past the {entry.uncompressed_bytes} "
            "bytes its own directory record declares")
    return out


def fetch_member(fetch: Fetcher, entry: ZipEntry, *,
                 total_bytes: int | None = None) -> bytes:
    """Fetch, inflate and verify exactly one member.

    The local file header is read as well as the payload, because its name
    length and extra-field length are what say where the payload starts.  A
    central directory that disagrees with its own local header is refused
    rather than read at a guessed offset.
    """
    safe_member_name(entry.name)
    if entry.method not in (METHOD_STORE, METHOD_DEFLATE):
        raise SourceError(
            f"member {entry.name!r} uses compression method {entry.method}; "
            "this reader handles store and deflate only")
    if entry.uncompressed_bytes > MAX_MEMBER_BYTES:
        raise SourceError(
            f"member {entry.name!r} declares {entry.uncompressed_bytes} bytes, "
            f"over the {MAX_MEMBER_BYTES} cap this reader will inflate")
    if entry.compressed_bytes > MAX_MEMBER_BYTES:
        raise SourceError(
            f"member {entry.name!r} declares {entry.compressed_bytes} "
            f"compressed bytes, over the {MAX_MEMBER_BYTES} cap")

    header = _exact(fetch, entry.local_offset, entry.local_offset + 29)
    if header[:4] != _LOCAL:
        raise SourceError(
            f"member {entry.name!r} has no local file header at the offset "
            "its central directory record names")
    nlen, elen = struct.unpack("<HH", header[26:30])
    start = entry.local_offset + 30 + nlen + elen
    end = start + entry.compressed_bytes - 1
    if total_bytes is not None and end >= total_bytes:
        raise SourceError(
            f"member {entry.name!r} would be read past the end of the archive")
    payload = _exact(fetch, start, end) if entry.compressed_bytes else b""

    out = _inflate(payload, entry)
    if len(out) != entry.uncompressed_bytes:
        raise SourceError(
            f"member {entry.name!r} inflated to {len(out)} bytes, not the "
            f"{entry.uncompressed_bytes} its directory record declares")
    if zlib.crc32(out) & 0xFFFFFFFF != entry.crc32 & 0xFFFFFFFF:
        raise SourceError(
            f"member {entry.name!r} failed its own CRC-32; the bytes that "
            "arrived are not the bytes the archive describes")
    return out


#: How much one coalesced read may cover.  Members wanted from these archives
#: sit next to each other, so reading a span and cutting it up locally turns
#: two requests per file into a handful per megabyte.  The cap keeps a span
#: inside the retry granularity of the fetcher.
MAX_SPAN_BYTES = 8 * 1024 * 1024

#: Slack allowed for a local file header whose extra field is longer than the
#: central directory's.  A header is 30 bytes plus the name plus the extra
#: field; the extra field is normally under a hundred bytes.
_HEADER_SLACK = 1024


def _extract_from_span(span: bytes, span_start: int, entry: ZipEntry) -> bytes:
    """Pull one member out of an already-fetched span of the archive."""
    at = entry.local_offset - span_start
    if at < 0 or at + 30 > len(span):
        raise SourceError(
            f"member {entry.name!r} is not inside the fetched span")
    if span[at:at + 4] != _LOCAL:
        raise SourceError(
            f"member {entry.name!r} has no local file header inside the span")
    nlen, elen = struct.unpack("<HH", span[at + 26:at + 30])
    start = at + 30 + nlen + elen
    end = start + entry.compressed_bytes
    if end > len(span):
        raise SourceError(
            f"member {entry.name!r} runs past the end of the fetched span")
    out = _inflate(span[start:end], entry)
    if len(out) != entry.uncompressed_bytes:
        raise SourceError(
            f"member {entry.name!r} inflated to {len(out)} bytes, not the "
            f"{entry.uncompressed_bytes} its directory record declares")
    if zlib.crc32(out) & 0xFFFFFFFF != entry.crc32 & 0xFFFFFFFF:
        raise SourceError(
            f"member {entry.name!r} failed its own CRC-32; the bytes that "
            "arrived are not the bytes the archive describes")
    return out


def plan_spans(entries: Sequence[ZipEntry], *,
               max_span_bytes: int = MAX_SPAN_BYTES
               ) -> list[tuple[int, int, tuple[ZipEntry, ...]]]:
    """Group members into ``(start, end, members)`` spans to read in one go.

    Members are sorted by their position in the archive and cut into spans no
    larger than ``max_span_bytes``.  The end of a span is computed from the
    last member's own declared sizes plus header slack, so a span always
    contains every member assigned to it; :func:`fetch_members` verifies that
    and falls back to a single-member read if a header turns out to be longer
    than the slack allows.
    """
    ordered = sorted(entries, key=lambda e: (e.local_offset, e.name))
    spans: list[tuple[int, int, tuple[ZipEntry, ...]]] = []
    current: list[ZipEntry] = []
    start = 0

    def close() -> None:
        if not current:
            return
        last = current[-1]
        end = (last.local_offset + 30 + len(last.name.encode("utf-8"))
               + _HEADER_SLACK + last.compressed_bytes)
        spans.append((start, end - 1, tuple(current)))

    for entry in ordered:
        need = (entry.local_offset + 30 + len(entry.name.encode("utf-8"))
                + _HEADER_SLACK + entry.compressed_bytes)
        if not current:
            current, start = [entry], entry.local_offset
            continue
        if need - start > max_span_bytes:
            close()
            current, start = [entry], entry.local_offset
        else:
            current.append(entry)
    close()
    return spans


def read_span(fetch: Fetcher, span, *, total_bytes: int
              ) -> list[tuple[ZipEntry, bytes]]:
    """Read one planned span and return its members' verified bytes.

    Every member is still checked against its own CRC-32 and declared size --
    coalescing changes how the bytes are requested, not how they are checked.
    A member whose local header turns out longer than the planned slack, or
    whose span was clipped by the end of the archive, is re-read on its own.
    """
    start, end, members = span
    blob = _exact(fetch, start, min(end, total_bytes - 1))
    out: list[tuple[ZipEntry, bytes]] = []
    for entry in members:
        try:
            out.append((entry, _extract_from_span(blob, start, entry)))
        except SourceError:
            out.append((entry, fetch_member(fetch, entry,
                                            total_bytes=total_bytes)))
    return out


def fetch_members(fetch: Fetcher, entries: Sequence[ZipEntry], *,
                  total_bytes: int, max_span_bytes: int = MAX_SPAN_BYTES):
    """Yield ``(entry, bytes)`` for many members, reading coalesced spans."""
    for span in plan_spans(entries, max_span_bytes=max_span_bytes):
        yield from read_span(fetch, span, total_bytes=total_bytes)


def local_file_fetcher(path) -> Fetcher:
    """A :data:`Fetcher` over a file on disk.

    This is how the tests read a real ZIP through the same code path the
    network uses, with no network involved.
    """
    from pathlib import Path

    target = Path(path)

    def fetch(start: int, end: int) -> bytes:
        with target.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)

    return fetch


def select(entries: Iterable[ZipEntry], wanted: Sequence[str], *,
           prefix: str = "") -> dict[str, tuple[ZipEntry, ...]]:
    """Group members by which wanted directory component they sit under.

    ``wanted`` are directory names, matched as a whole path segment so that a
    class called ``3001`` never absorbs ``30011``.  Directory records and
    members belonging to nothing wanted are dropped, and a wanted name that
    matched nothing comes back as an empty tuple rather than being missing --
    a caller has to be able to see that a class is absent.
    """
    out: dict[str, list[ZipEntry]] = {name: [] for name in wanted}
    for entry in entries:
        if entry.is_directory:
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        segments = entry.name.split("/")
        for name in wanted:
            if name in segments[:-1]:
                out[name].append(entry)
                break
    return {name: tuple(items) for name, items in out.items()}
