"""The only place in this project that opens an outbound HTTP connection.

Kept apart from :mod:`src.vision.source` so that the ZIP reader can be tested,
audited and reasoned about without a network in the picture at all, and so the
one module that does reach the network is small enough to read in full.

The mirror these datasets live on behaves in two ways worth encoding rather
than discovering twice:

* a single connection delivers roughly twenty to thirty kilobytes a second,
  and several connections in parallel do add up, so a range download is split
  across a small pool of workers;
* a connection is dropped after a couple of megabytes, so a range is asked for
  in pieces and each piece is retried on its own rather than restarting the
  whole download.

Nothing here runs at import.  ``strict-offline`` runs and the whole test suite
never call into this module: the tests drive :mod:`src.vision.source` through
:func:`src.vision.source.local_file_fetcher` instead.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.vision.source import Fetcher, SourceError

#: Sent so the mirror's operators can see what is reading them, and so a
#: request from this project is distinguishable from a browser in their logs.
USER_AGENT = "BrickAgain/1.0 (selective public-dataset range reader)"

#: Only these two.  A dataset URL that redirects to anything else is a
#: redirect this project did not intend to follow.
ALLOWED_SCHEMES = ("https",)

#: Per-piece size.  Chosen under the observed drop threshold rather than at a
#: round number, so a piece usually completes on its first attempt.
PIECE_BYTES = 1 << 20

#: How many pieces are in flight.  Small on purpose: the point is to use the
#: bandwidth that one connection leaves unused, not to hammer a public mirror.
WORKERS = 6

RETRIES = 6
TIMEOUT_SECONDS = 180.0


class NetworkError(SourceError):
    """A range could not be fetched.  Distinct so a caller can say which."""


def _check_url(url: str) -> str:
    from urllib.parse import urlsplit

    if not isinstance(url, str) or not url.strip():
        raise NetworkError("a URL is required")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise NetworkError(
            f"scheme {parts.scheme!r} is not one of {list(ALLOWED_SCHEMES)}")
    if not parts.netloc:
        raise NetworkError(f"{url!r} has no host")
    return url


@dataclass
class RangeReader:
    """A resolved URL plus the archive size the server reported for it.

    The mirror hands out expiring signed URLs, so the resolved location is
    re-resolved rather than stored anywhere: this object lives for one run.
    """

    url: str
    total_bytes: int
    requests: int = 0
    bytes_read: int = 0

    def raw(self, start: int, end: int) -> bytes:
        """One range, retried, with the piece boundaries left to the caller."""
        want = end - start + 1
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                request = urllib.request.Request(
                    self.url,
                    headers={"User-Agent": USER_AGENT,
                             "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(
                        request, timeout=TIMEOUT_SECONDS) as response:
                    if response.status not in (200, 206):
                        raise NetworkError(
                            f"the server answered {response.status} for a "
                            "range request")
                    buffer = bytearray()
                    while len(buffer) < want:
                        chunk = response.read(min(1 << 16, want - len(buffer)))
                        if not chunk:
                            break
                        buffer += chunk
                if len(buffer) != want:
                    raise NetworkError(
                        f"the server sent {len(buffer)} of {want} bytes")
                self.requests += 1
                self.bytes_read += len(buffer)
                return bytes(buffer)
            except (urllib.error.URLError, NetworkError, OSError) as exc:
                last = exc
                time.sleep(min(2.0 * (attempt + 1), 15.0))
        raise NetworkError(
            f"gave up on bytes {start}-{end} after {RETRIES} attempts: {last}")

    def fetcher(self) -> Fetcher:
        """A :data:`~src.vision.source.Fetcher` that splits and parallelises.

        A range smaller than one piece is fetched directly; a larger one is
        cut into pieces, fetched by the pool and reassembled in order.  The
        result is exactly the requested bytes either way, so the ZIP reader
        cannot tell the difference and does not have to.
        """

        def fetch(start: int, end: int) -> bytes:
            want = end - start + 1
            if want <= PIECE_BYTES:
                return self.raw(start, end)
            bounds = []
            offset = start
            while offset <= end:
                stop = min(offset + PIECE_BYTES - 1, end)
                bounds.append((offset, stop))
                offset = stop + 1
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                pieces = list(pool.map(lambda b: self.raw(*b), bounds))
            out = b"".join(pieces)
            if len(out) != want:
                raise NetworkError(
                    f"reassembled {len(out)} bytes for a {want}-byte range")
            return out

        return fetch


def open_range_reader(url: str) -> RangeReader:
    """Resolve a URL and learn the archive size, using one one-byte range.

    The size comes from ``Content-Range`` rather than ``Content-Length``,
    because the latter on a ranged response is the length of the range.  A
    server that will not answer a range request is refused here rather than
    later: without ranges this whole approach collapses into downloading six
    gigabytes, which is the thing being avoided.
    """
    _check_url(url)
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        if response.status != 206:
            raise NetworkError(
                f"the server answered {response.status} instead of 206 for a "
                "range request; this reader needs range support")
        header = response.headers.get("Content-Range") or ""
        resolved = response.geturl()
    _check_url(resolved)
    if "/" not in header:
        raise NetworkError(
            f"the server sent no usable Content-Range: {header!r}")
    tail = header.rsplit("/", 1)[1].strip()
    if not tail.isdigit():
        raise NetworkError(f"the server did not state a total size: {header!r}")
    return RangeReader(url=resolved, total_bytes=int(tail))
