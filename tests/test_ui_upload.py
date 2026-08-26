"""The multipart parser and the session store, directly.

``tests/test_ui_full.py`` drives these over a real socket, which is the right
place to check that the route behaves. This file checks the parser itself,
because a parser reached only through a route is a parser whose edge cases are
tested by whatever the route happens to send.

The parser's job is to refuse. Everything here is an attempt to get something
past it: a body that claims a boundary it does not use, a part with no
disposition, a filename that is a path, more parts than any form has, a field
nobody asked for, two files where one was expected.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from src.ui.state import SessionStore, StateError
from src.ui.upload import (MAX_FIELD_BYTES, MAX_FILENAME_CHARS, MAX_PARTS,
                           MAX_UPLOAD_BYTES, MULTIPART_MEDIA_TYPE, UploadError,
                           boundary_of, parse_multipart, safe_filename)


def png(width=64, height=48, colour=(200, 30, 30)):
    from PIL import Image

    image = np.full((height, width, 3), colour, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def body(parts, boundary="BND", closing=True):
    out = []
    for headers, payload in parts:
        out.append(f"--{boundary}\r\n{headers}\r\n\r\n".encode("utf-8"))
        out.append(payload if isinstance(payload, bytes)
                   else payload.encode("utf-8"))
        out.append(b"\r\n")
    out.append(f"--{boundary}--\r\n".encode("utf-8") if closing
               else b"")
    return b"".join(out)


def field(name, value):
    return (f'Content-Disposition: form-data; name="{name}"', value)


def file_part(name, payload, filename="a.png", media="image/png"):
    return (f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {media}', payload)


CT = f"{MULTIPART_MEDIA_TYPE}; boundary=BND"


class TestTheBoundary:
    def test_a_normal_content_type_yields_its_boundary(self):
        assert boundary_of(CT) == "BND"

    def test_a_quoted_boundary_is_unquoted(self):
        assert boundary_of(f'{MULTIPART_MEDIA_TYPE}; boundary="a-b-c"') == \
            "a-b-c"

    def test_another_media_type_is_refused(self):
        with pytest.raises(UploadError, match="只處理"):
            boundary_of("application/json")

    def test_no_boundary_is_refused(self):
        with pytest.raises(UploadError, match="沒有 boundary"):
            boundary_of(MULTIPART_MEDIA_TYPE)

    def test_an_empty_boundary_is_refused(self):
        with pytest.raises(UploadError, match="缺失或過長"):
            boundary_of(f"{MULTIPART_MEDIA_TYPE}; boundary=")

    def test_an_over_long_boundary_is_refused(self):
        with pytest.raises(UploadError, match="缺失或過長"):
            boundary_of(f"{MULTIPART_MEDIA_TYPE}; boundary={'x' * 200}")

    def test_a_missing_content_type_is_refused(self):
        with pytest.raises(UploadError, match="沒有 Content-Type"):
            boundary_of("")


class TestParsing:
    def test_a_field_and_a_file_come_back_separately(self):
        payload = png()
        parsed = parse_multipart(
            body([field("mode", "multi"), file_part("photo", payload)]), CT,
            image_fields=("photo",), text_fields=("mode",))
        assert parsed.fields == {"mode": ["multi"]}
        assert len(parsed.images) == 1
        image = parsed.images[0]
        assert image.data == payload
        assert (image.width, image.height) == (64, 48)
        assert image.source_format == "PNG"

    def test_a_jpeg_part_is_accepted(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (40, 30), (10, 20, 30)).save(buffer, format="JPEG")
        parsed = parse_multipart(
            body([file_part("photo", buffer.getvalue(), filename="a.jpg",
                            media="image/jpeg")]), CT,
            image_fields=("photo",))
        assert parsed.images[0].source_format == "JPEG"

    def test_a_body_with_no_boundary_in_it_is_refused(self):
        with pytest.raises(UploadError, match="找不到"):
            parse_multipart(b"no boundary here at all", CT,
                            image_fields=("photo",))

    def test_a_non_bytes_body_is_refused(self):
        with pytest.raises(UploadError, match="必須是位元組"):
            parse_multipart("a string", CT, image_fields=("photo",))

    def test_a_body_over_the_cap_is_refused(self):
        payload = body([file_part("photo", png())])
        with pytest.raises(UploadError, match="超過上限"):
            parse_multipart(payload, CT, image_fields=("photo",),
                            max_bytes=len(payload) - 1)

    def test_the_real_cap_leaves_room_for_a_phone_photograph(self):
        assert MAX_UPLOAD_BYTES >= 8 * 1024 * 1024

    def test_too_many_parts_are_refused(self):
        parts = [field(f"f{i}", "x") for i in range(MAX_PARTS + 5)]
        with pytest.raises(UploadError, match="超過上限"):
            parse_multipart(body(parts), CT,
                            text_fields=tuple(f"f{i}" for i in
                                              range(MAX_PARTS + 5)))

    def test_an_unexpected_text_field_is_refused_not_ignored(self):
        with pytest.raises(UploadError, match="未預期的欄位"):
            parse_multipart(body([field("surprise", "x")]), CT,
                            text_fields=("mode",))

    def test_a_file_in_an_unexpected_field_is_refused(self):
        with pytest.raises(UploadError, match="不接受該欄位的檔案"):
            parse_multipart(body([file_part("other", png())]), CT,
                            image_fields=("photo",))

    def test_a_part_with_no_disposition_is_refused(self):
        raw = (b"--BND\r\nContent-Type: text/plain\r\n\r\nvalue\r\n"
               b"--BND--\r\n")
        with pytest.raises(UploadError, match="沒有 Content-Disposition"):
            parse_multipart(raw, CT, text_fields=("mode",))

    def test_a_disposition_that_is_not_form_data_is_refused(self):
        raw = (b"--BND\r\nContent-Disposition: attachment; name=\"mode\"\r\n"
               b"\r\nvalue\r\n--BND--\r\n")
        with pytest.raises(UploadError, match="不是 form-data"):
            parse_multipart(raw, CT, text_fields=("mode",))

    def test_a_part_with_no_name_is_refused(self):
        raw = (b"--BND\r\nContent-Disposition: form-data\r\n\r\nv\r\n"
               b"--BND--\r\n")
        with pytest.raises(UploadError, match="沒有欄位名稱"):
            parse_multipart(raw, CT, text_fields=("mode",))

    def test_a_part_with_no_header_separator_is_refused(self):
        raw = b"--BND\r\nContent-Disposition: form-data; name=\"mode\"\r\n" \
              b"--BND--\r\n"
        with pytest.raises(UploadError, match="沒有標頭與本體的分隔"):
            parse_multipart(raw, CT, text_fields=("mode",))

    def test_an_over_long_part_header_is_refused(self):
        raw = (b"--BND\r\nContent-Disposition: form-data; name=\"mode\"\r\n"
               + b"X-Padding: " + b"y" * 9000 + b"\r\n\r\nv\r\n--BND--\r\n")
        with pytest.raises(UploadError, match="標頭有"):
            parse_multipart(raw, CT, text_fields=("mode",))

    def test_an_over_long_field_value_is_refused(self):
        with pytest.raises(UploadError, match="超過上限"):
            parse_multipart(
                body([field("mode", "x" * (MAX_FIELD_BYTES + 10))]), CT,
                text_fields=("mode",))

    def test_an_empty_file_is_refused(self):
        with pytest.raises(UploadError, match="檔案是空的"):
            parse_multipart(body([file_part("photo", b"")]), CT,
                            image_fields=("photo",))

    def test_a_non_image_file_is_refused_with_the_decoders_reason(self):
        with pytest.raises(UploadError, match="這張圖片被拒絕"):
            parse_multipart(body([file_part("photo", b"not a picture" * 40)]),
                            CT, image_fields=("photo",))

    def test_two_files_in_one_field_is_refused_on_access(self):
        parsed = parse_multipart(
            body([file_part("photo", png()), file_part("photo", png())]), CT,
            image_fields=("photo",))
        with pytest.raises(UploadError, match="個檔案"):
            parsed.one_image("photo")

    def test_a_missing_file_field_returns_none(self):
        parsed = parse_multipart(body([field("mode", "multi")]), CT,
                                image_fields=("photo",),
                                text_fields=("mode",))
        assert parsed.one_image("photo") is None

    def test_a_repeated_text_field_is_kept_as_a_list(self):
        """So the form layer can refuse it by name, as it does elsewhere."""
        parsed = parse_multipart(
            body([field("mode", "a"), field("mode", "b")]), CT,
            text_fields=("mode",))
        assert parsed.fields["mode"] == ["a", "b"]


class TestFilenames:
    @pytest.mark.parametrize("raw,expected", [
        ("photo.png", "photo.png"),
        ("/etc/passwd", "passwd"),
        ("../../secret.png", "secret.png"),
        ("C:\\\\Users\\\\me\\\\a.png", "a.png"),
        ("dir/sub/x.jpg", "x.jpg"),
        ("", "(unnamed)"),
        (None, "(unnamed)"),
        ("..", "(unnamed)"),
        (".", "(unnamed)"),
    ])
    def test_a_filename_is_reduced_to_something_displayable(self, raw,
                                                             expected):
        assert safe_filename(raw) == expected

    def test_a_null_byte_is_stripped(self):
        assert "\x00" not in safe_filename("a\x00b.png")

    def test_an_over_long_filename_is_truncated(self):
        out = safe_filename("x" * 400 + ".png")
        assert len(out) <= MAX_FILENAME_CHARS + 1

    def test_the_parsed_filename_is_the_reduced_one(self):
        parsed = parse_multipart(
            body([file_part("photo", png(), filename="../../evil.png")]), CT,
            image_fields=("photo",))
        assert parsed.images[0].filename == "evil.png"

    def test_the_record_says_what_it_measured(self):
        parsed = parse_multipart(body([file_part("photo", png())]), CT,
                                image_fields=("photo",))
        body_dict = parsed.images[0].as_dict()
        assert body_dict["width"] == 64 and body_dict["height"] == 48
        assert body_dict["format"] == "PNG"
        assert body_dict["bytes"] > 0


class TestTheSessionStore:
    def test_a_photograph_round_trips(self):
        store = SessionStore()
        session = store.put_photo(image=b"abc", media_type="image/png",
                                  filename="a.png", width=4, height=3,
                                  mode="single")
        again = store.photo(session.handle)
        assert again.image == b"abc" and again.width == 4

    def test_an_unknown_handle_raises(self):
        with pytest.raises(StateError):
            SessionStore().photo("nope")

    def test_updating_keeps_the_handle_and_replaces_the_field(self):
        store = SessionStore()
        session = store.put_photo(image=b"a", media_type="image/png",
                                  filename="a.png", width=1, height=1,
                                  mode="single")
        updated = store.update_photo(session.handle, items=(1, 2, 3))
        assert updated.handle == session.handle
        assert updated.items == (1, 2, 3)
        assert store.photo(session.handle).items == (1, 2, 3)

    def test_handles_are_unguessable_and_distinct(self):
        store = SessionStore()
        handles = {store.put_result("x").handle for _ in range(5)}
        assert len(handles) == 5
        assert all(len(handle) >= 16 for handle in handles)

    def test_a_result_round_trips_and_updates(self):
        store = SessionStore()
        session = store.put_result("rag", payload={"a": 1})
        assert store.result(session.handle).payload == {"a": 1}
        store.update_result(session.handle, payload={"a": 2})
        assert store.result(session.handle).payload == {"a": 2}

    def test_the_counts_report_what_is_held(self):
        store = SessionStore()
        store.put_photo(image=b"xyz", media_type="image/png",
                        filename="a.png", width=1, height=1, mode="single")
        store.put_result("x")
        counts = store.counts()
        assert counts == {"photos": 1, "results": 1, "image_bytes": 3}

    def test_a_zero_limit_is_refused(self):
        with pytest.raises(ValueError, match="room for one"):
            SessionStore(photo_limit=0)

    def test_the_most_recently_used_photograph_survives_eviction(self):
        store = SessionStore(photo_limit=2)
        first = store.put_photo(image=b"a", media_type="image/png",
                                filename="a.png", width=1, height=1,
                                mode="single")
        second = store.put_photo(image=b"b", media_type="image/png",
                                 filename="b.png", width=1, height=1,
                                 mode="single")
        store.photo(first.handle)          # touch it, so it is not the oldest
        store.put_photo(image=b"c", media_type="image/png", filename="c.png",
                        width=1, height=1, mode="single")
        assert store.photo(first.handle).filename == "a.png"
        with pytest.raises(StateError):
            store.photo(second.handle)
