"""Where a multi-page flow keeps its state: in memory, bounded, in-process.

The full interface has four pages and a photograph that has to survive between
them, which the two-page version did not.  That is a real change and it is kept
as small as possible:

* **Nothing is written to disk.**  An uploaded photograph, its detections, the
  operator's corrections and a built assembly plan all live in this store for
  the life of the process and no longer.  Closing the interface leaves nothing
  behind, so there is nothing to clean up and nothing to leak.
* **It is bounded.**  A long session must not accumulate every photograph it
  ever looked at, so the store is a small LRU and the oldest entry is dropped.
* **Handles are opaque and unguessable.**  A handle is
  ``secrets.token_urlsafe``, matched against a strict pattern by the HTTP
  layer, and looked up in this map -- so no request can name a path on disk,
  and a handle from one process is meaningless in another.
* **A missing handle is a sentence, not a crash.**  Entries expire by
  eviction, so the pages have to cope with a handle that is no longer here and
  say why.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

#: Photographs and their corrections held at once.  Small: a person works on
#: one photograph at a time, and the previous few are only there so the back
#: button works.
PHOTO_LIMIT = 6

#: Results and assembly plans held at once.
RESULT_LIMIT = 8

#: Bytes of uploaded image held across the whole store.  A cap on the sum, not
#: only on each one, so six large photographs cannot add up to something this
#: process should not be holding.
TOTAL_IMAGE_BYTES = 48 * 1024 * 1024


class StateError(KeyError):
    """The handle is not in this process's store."""


@dataclass(frozen=True)
class PhotoSession:
    """One uploaded photograph, what was detected in it, and the edits so far."""

    handle: str
    image: bytes = field(repr=False)
    media_type: str
    filename: str
    width: int
    height: int
    mode: str
    items: tuple = field(default=(), repr=False)
    diagnostics: dict = field(default_factory=dict)
    method: str = ""
    created: float = 0.0

    @property
    def bytes(self) -> int:
        return len(self.image)


@dataclass(frozen=True)
class ResultSession:
    """One produced result: its report, artefacts and assembly plan."""

    handle: str
    kind: str
    payload: dict = field(default_factory=dict, repr=False)
    report: dict | None = field(default=None, repr=False)
    artifacts: object = field(default=None, repr=False)
    plan: object = field(default=None, repr=False)
    assignment: object = field(default=None, repr=False)
    step_previews: tuple = field(default=(), repr=False)
    extra: dict = field(default_factory=dict, repr=False)
    created: float = 0.0


class SessionStore:
    """Two bounded LRU maps, and the byte budget that spans one of them."""

    def __init__(self, *, photo_limit: int = PHOTO_LIMIT,
                 result_limit: int = RESULT_LIMIT,
                 total_image_bytes: int = TOTAL_IMAGE_BYTES) -> None:
        if photo_limit < 1 or result_limit < 1:
            raise ValueError("both stores need room for one entry")
        self.photo_limit = photo_limit
        self.result_limit = result_limit
        self.total_image_bytes = total_image_bytes
        self._photos: "OrderedDict[str, PhotoSession]" = OrderedDict()
        self._results: "OrderedDict[str, ResultSession]" = OrderedDict()

    # -- photographs ------------------------------------------------------
    def put_photo(self, *, image: bytes, media_type: str, filename: str,
                  width: int, height: int, mode: str, items=(),
                  diagnostics=None, method: str = "") -> PhotoSession:
        handle = secrets.token_urlsafe(16)
        session = PhotoSession(
            handle=handle, image=bytes(image), media_type=media_type,
            filename=filename, width=width, height=height, mode=mode,
            items=tuple(items), diagnostics=dict(diagnostics or {}),
            method=method, created=time.monotonic())
        self._photos[handle] = session
        self._photos.move_to_end(handle)
        self._evict_photos()
        return session

    def photo(self, handle: str) -> PhotoSession:
        session = self._photos.get(handle)
        if session is None:
            raise StateError(handle)
        self._photos.move_to_end(handle)
        return session

    def update_photo(self, handle: str, **changes) -> PhotoSession:
        session = self.photo(handle)
        updated = replace(session, **changes)
        self._photos[handle] = updated
        self._photos.move_to_end(handle)
        self._evict_photos()
        return updated

    def _evict_photos(self) -> None:
        while len(self._photos) > self.photo_limit:
            self._photos.popitem(last=False)
        # Then by total bytes, oldest first, so a few very large photographs
        # cannot sit here together just because the count is under the limit.
        while (sum(entry.bytes for entry in self._photos.values())
               > self.total_image_bytes and len(self._photos) > 1):
            self._photos.popitem(last=False)

    # -- results ----------------------------------------------------------
    def put_result(self, kind: str, **fields) -> ResultSession:
        handle = secrets.token_urlsafe(16)
        session = ResultSession(handle=handle, kind=kind,
                                created=time.monotonic(), **fields)
        self._results[handle] = session
        self._results.move_to_end(handle)
        while len(self._results) > self.result_limit:
            self._results.popitem(last=False)
        return session

    def result(self, handle: str) -> ResultSession:
        session = self._results.get(handle)
        if session is None:
            raise StateError(handle)
        self._results.move_to_end(handle)
        return session

    def update_result(self, handle: str, **changes) -> ResultSession:
        session = self.result(handle)
        updated = replace(session, **changes)
        self._results[handle] = updated
        self._results.move_to_end(handle)
        return updated

    # -- reporting --------------------------------------------------------
    def counts(self) -> dict:
        return {"photos": len(self._photos), "results": len(self._results),
                "image_bytes": sum(entry.bytes
                                   for entry in self._photos.values())}

    def clear(self) -> None:
        """Drop everything.  What the "start over" button does."""
        self._photos.clear()
        self._results.clear()
