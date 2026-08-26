"""``multipart/form-data`` parsing, written to refuse rather than to accept.

The existing interface takes ``application/x-www-form-urlencoded`` and nothing
else, which is why it needed no parser.  A photograph cannot travel that way,
so this module adds the one other content type -- and adds it as a parser with
hard bounds rather than by handing the body to a library that will accept
anything.

Every bound is here, and each exists because of a specific way this could go
wrong:

* **The body is bounded before it is read.**  ``Content-Length`` over
  :data:`MAX_UPLOAD_BYTES` is refused without reading a byte, so a large
  upload costs a rejection rather than memory.
* **The number of parts is bounded.**  A body of ten thousand tiny parts is a
  cheap way to make a parser spend a long time; there is a cap and it is small.
* **Only expected field names are accepted.**  An unexpected name is refused,
  not ignored, so a form that has quietly changed is visible.
* **A filename is never used as a path.**  It is recorded for display only,
  after being stripped of directories and bounded in length.  Nothing here
  writes a file at all.
* **Every image goes through one decoder.**
  :func:`src.vision.preprocess.decode_image` enforces format, byte size and
  header-declared pixel count, so a decompression bomb is refused before its
  pixels exist.

Nothing here writes to disk.  The uploaded bytes live in memory for the
duration of one request and in the bounded in-process store the result page
reads from; there is no temporary file to clean up because there is no
temporary file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: A photograph plus form fields. The public archive's own photographs are
#: about two megabytes; this leaves room for a larger phone image without
#: leaving room for an upload that is not a photograph.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

#: Parts in one body. A form with a handful of fields and one file needs very
#: few, and a body with hundreds is not that form.
MAX_PARTS = 24

#: Length of one part's headers. A header block larger than this is not a
#: browser's.
MAX_PART_HEADER_BYTES = 4096

#: Characters of a filename kept for display.
MAX_FILENAME_CHARS = 120

#: Bytes of a non-file field. Field values here are a caption and some
#: numbers; the caption's own limit is smaller and is applied by the form
#: parser.
MAX_FIELD_BYTES = 64 * 1024

#: The boundary token, per RFC 2046, is 1-70 characters from a restricted set.
MAX_BOUNDARY_CHARS = 70

MULTIPART_MEDIA_TYPE = "multipart/form-data"


class UploadError(ValueError):
    """The upload is refused.  Nothing was decoded and nothing was stored."""


@dataclass(frozen=True)
class UploadedImage:
    """One uploaded image, in memory, already decoded once and measured."""

    field_name: str
    filename: str
    media_type: str
    data: bytes = field(repr=False)
    width: int = 0
    height: int = 0
    source_format: str = ""

    @property
    def bytes(self) -> int:
        return len(self.data)

    def as_dict(self) -> dict:
        return {"field": self.field_name, "filename": self.filename,
                "bytes": self.bytes, "width": self.width,
                "height": self.height, "format": self.source_format}


@dataclass(frozen=True)
class Multipart:
    """The parsed body: text fields as a form mapping, images separately."""

    fields: dict[str, list[str]]
    images: tuple[UploadedImage, ...]

    def one_image(self, name: str) -> UploadedImage | None:
        found = [image for image in self.images if image.field_name == name]
        if len(found) > 1:
            raise UploadError(
                f"欄位 {name} 送出了 {len(found)} 個檔案；請只上傳一張照片")
        return found[0] if found else None


def boundary_of(content_type: str) -> str:
    """Extract and check the boundary token from a ``Content-Type``."""
    if not isinstance(content_type, str) or not content_type.strip():
        raise UploadError("這個請求沒有 Content-Type")
    parts = [piece.strip() for piece in content_type.split(";")]
    if not parts or parts[0].lower() != MULTIPART_MEDIA_TYPE:
        raise UploadError(
            f"Content-Type 是 {parts[0] if parts else ''!r}，"
            f"這個路徑只處理 {MULTIPART_MEDIA_TYPE}")
    for piece in parts[1:]:
        key, sep, value = piece.partition("=")
        if sep and key.strip().lower() == "boundary":
            marker = value.strip().strip('"')
            if not marker or len(marker) > MAX_BOUNDARY_CHARS:
                raise UploadError(
                    "multipart 的 boundary 缺失或過長，無法解析這個本體")
            return marker
    raise UploadError("multipart 的 Content-Type 沒有 boundary")


def _split_headers(block: bytes) -> dict[str, str]:
    if len(block) > MAX_PART_HEADER_BYTES:
        raise UploadError(
            f"某個 part 的標頭有 {len(block)} 位元組，超過上限 "
            f"{MAX_PART_HEADER_BYTES}")
    out: dict[str, str] = {}
    for line in block.split(b"\r\n"):
        if not line:
            continue
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            raise UploadError("某個 part 的標頭不是 UTF-8") from None
        key, sep, value = text.partition(":")
        if not sep:
            raise UploadError(f"無法解析 part 標頭：{text[:60]!r}")
        out[key.strip().lower()] = value.strip()
    return out


def _disposition(headers: dict[str, str]) -> tuple[str, str | None]:
    """``(field name, filename or None)`` from a part's disposition header."""
    raw = headers.get("content-disposition")
    if not raw:
        raise UploadError("某個 part 沒有 Content-Disposition")
    pieces = [piece.strip() for piece in raw.split(";")]
    if not pieces or pieces[0].lower() != "form-data":
        raise UploadError("某個 part 的 Content-Disposition 不是 form-data")
    name = None
    filename = None
    for piece in pieces[1:]:
        key, sep, value = piece.partition("=")
        if not sep:
            continue
        cleaned = value.strip().strip('"')
        if key.strip().lower() == "name":
            name = cleaned
        elif key.strip().lower() == "filename":
            filename = cleaned
    if not name:
        raise UploadError("某個 part 沒有欄位名稱")
    return name, filename


def safe_filename(raw: str | None) -> str:
    """A filename fit to *display*, never to open.

    Directory components are dropped rather than sanitised, because there is
    no case in which this project wants any of them.  Nothing in this module
    or the interface opens a path built from this value; it exists so the page
    can say which file the operator chose.
    """
    if not raw:
        return "(unnamed)"
    tail = raw.replace("\\", "/").split("/")[-1]
    tail = tail.replace("\x00", "")
    if tail in ("", ".", ".."):
        return "(unnamed)"
    if len(tail) > MAX_FILENAME_CHARS:
        tail = tail[:MAX_FILENAME_CHARS] + "…"
    return tail


def parse_multipart(body: bytes, content_type: str, *,
                    image_fields=(), text_fields=(),
                    max_bytes: int = MAX_UPLOAD_BYTES) -> Multipart:
    """Parse a body into text fields and decoded images, or refuse.

    ``image_fields`` and ``text_fields`` are the names this endpoint expects.
    A part naming anything else is refused: a form that has silently changed
    should be visible, and an endpoint that ignores what it does not recognise
    is one that accepts anything.
    """
    if not isinstance(body, (bytes, bytearray)):
        raise UploadError("multipart 的本體必須是位元組")
    if len(body) > max_bytes:
        raise UploadError(
            f"上傳內容 {len(body)} 位元組超過上限 {max_bytes}")
    boundary = boundary_of(content_type)
    marker = b"--" + boundary.encode("ascii", "strict")

    segments = bytes(body).split(marker)
    if len(segments) < 2:
        raise UploadError("這個本體裡找不到 multipart 的 boundary")
    # The first segment is the preamble and the last is the epilogue after the
    # closing "--"; neither is a part.
    parts = segments[1:-1] if segments[-1].lstrip().startswith(b"--") \
        else segments[1:]
    if len(parts) > MAX_PARTS:
        raise UploadError(
            f"這個本體有 {len(parts)} 個 part，超過上限 {MAX_PARTS}")

    fields: dict[str, list[str]] = {}
    images: list[UploadedImage] = []
    allowed_images = set(image_fields)
    allowed_text = set(text_fields)

    for segment in parts:
        chunk = segment
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.rstrip() in (b"", b"--"):
            continue
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        head, separator, payload = chunk.partition(b"\r\n\r\n")
        if not separator:
            raise UploadError("某個 part 沒有標頭與本體的分隔")
        headers = _split_headers(head)
        name, filename = _disposition(headers)

        if filename is not None or name in allowed_images:
            if name not in allowed_images:
                raise UploadError(
                    f"欄位 {name!r} 送來一個檔案，但這個表單不接受該欄位的檔案")
            images.append(_decode_part(name, filename, headers, payload))
            continue
        if name not in allowed_text:
            raise UploadError(
                f"欄位 {name!r} 不是這個表單的欄位；未預期的欄位一律拒絕，"
                "不會被忽略")
        if len(payload) > MAX_FIELD_BYTES:
            raise UploadError(
                f"欄位 {name} 有 {len(payload)} 位元組，超過上限 "
                f"{MAX_FIELD_BYTES}")
        fields.setdefault(name, []).append(
            payload.decode("utf-8", errors="replace"))
    return Multipart(fields=fields, images=tuple(images))


def _decode_part(name: str, filename: str | None, headers: dict[str, str],
                 payload: bytes) -> UploadedImage:
    """Decode one file part through the project's single image entry point."""
    from src.vision.preprocess import ImageError, decode_image

    if not payload:
        raise UploadError(f"欄位 {name} 的檔案是空的；請選擇一張照片")
    try:
        loaded = decode_image(payload, max_bytes=MAX_UPLOAD_BYTES)
    except ImageError as exc:
        # The decoder's message already names the rule that was broken --
        # format, byte size or declared pixel count -- so it is passed through
        # rather than replaced with something vaguer.
        raise UploadError(f"這張圖片被拒絕：{exc}") from None
    return UploadedImage(
        field_name=name, filename=safe_filename(filename),
        media_type=(headers.get("content-type") or "").split(";")[0].strip(),
        data=payload, width=loaded.width, height=loaded.height,
        source_format=loaded.source_format)
